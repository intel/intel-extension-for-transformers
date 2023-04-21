//  Copyright (c) 2021 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <omp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <exception>
#include "interface.hpp"
#include "gtest/gtest.h"
#include "unit_test_utils.hpp"
#include "kernels/matmul_types.hpp"
#include "kernels/matmul_ref.hpp"
#include "jit_domain/jit_matmul_vnni_8xkx48.hpp"

#define OMP_NUM_THREADS "OMP_NUM_THREADS"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;
using io = ssd::matmul_io::io;

// test trmm vnni
struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> rt_data;
  int nthr;  // 0 for not touching OMP_NUM_THREADS and using what set outside
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    n_thread_t with_n_thread(p.nthr);
    const auto& op_desc = p.op_desc;
    transpose_matmul_desc kernel_desc(op_desc);
    transpose_matmul kernel(kernel_desc);
    kernel.execute(p.rt_data);

    std::shared_ptr<const kernel_desc_t> ker_ref_desc;
    kernel_desc_t::create<matmul_ref_kd_t>(ker_ref_desc, q.op_desc);
    std::shared_ptr<const kernel_t> trmm_ref_kernel;
    kernel_t::create<matmul_ref_k_t, matmul_ref_kd_t>(trmm_ref_kernel, ker_ref_desc);
    trmm_ref_kernel->execute(q.rt_data);
  } catch (const std::exception& e) {
    return t.expect_to_fail;
  }
  if (!t.expect_to_fail) {
    auto buf1 = p.rt_data[io::DST0];
    auto size1 = p.op_desc.tensor_descs()[io::DST0].size();
    auto buf2 = q.rt_data[io::DST0];
    auto size2 = q.op_desc.tensor_descs()[io::DST0].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[io::DST0].dtype();
    if (dst_type == dt::fp32) {
      return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::s32) {
      return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::u8) {
      return compare_data<uint8_t>(buf1, size1, buf2, size2, 0);
    } else if (dst_type == dt::s8) {
      return compare_data<int8_t>(buf1, size1, buf2, size2, 0);
    }
  }
  return false;
}

class MMVNNIP2031P2013KernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  MMVNNIP2031P2013KernelTest() {}
  virtual ~MMVNNIP2031P2013KernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MMVNNIP2031P2013KernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
  for (auto op_args : {t.args.first, t.args.second})
    for (auto rt_data : op_args.rt_data) {
      char* data = reinterpret_cast<char*>(const_cast<void*>(rt_data));
      delete[] data;
    }
}

std::pair<const void*, const void*> make_data_obj(const std::vector<int64_t>& a_shape, const dt& a_dt,
                                                  bool is_clear = false, const std::vector<float>& ranges = {-10, 10}) {
  if (a_shape.size() == 0) return {nullptr, nullptr};
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), dim_t{1}, std::multiplies<dim_t>());
  int bytes_size = elem_num * type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else {
    if (a_dt == dt::fp32) {
      data_ptr = new float[elem_num];
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s32) {
      data_ptr = new int32_t[elem_num];
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::u8) {
      data_ptr = new uint8_t[elem_num];
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s8) {
      data_ptr = new int8_t[elem_num];
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, dim_t bs0, dim_t bs1, int nthr = 0,
                                         bool has_binary_add = true,
                                         std::unordered_map<std::string, std::string> attrs = {},
                                         std::vector<postop_attr> post_ops = {}) {
  // Step 1: Construct operator config
  const data_type dst_type = post_ops.size() != 0 ? post_ops.back().dt : data_type::fp32;

  const tensor_desc src0_desc{{bs1, K, bs0, M}, dt::s8, ft::ab};
  const tensor_desc src1_desc{{bs1, K, bs0, N}, dt::s8, ft::ab};
  const tensor_desc dst_desc{{bs0, bs1, M, N}, dst_type, ft::ab};
  const tensor_desc src2_desc = has_binary_add ? tensor_desc{{bs0, bs1, M, N}, dt::fp32, ft::ab} : tensor_desc();
  const std::vector<tensor_desc> ts_descs{src0_desc, src1_desc, dst_desc, src2_desc};

  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    const bool is_clear = (index == io::DST0);
    const auto ranges = std::vector<float>{-10, 10};
    const auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), is_clear, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  operator_desc op_desc(kernel_kind::transpose_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                        attrs, post_ops);

  // Step 3: op_args_t testcase pair
  op_args_t op_args = {op_desc, rt_data1, nthr};
  op_args_t op_args_copy = {op_desc, rt_data2, nthr};

  return {op_args, op_args_copy};
}

static auto case_func = []() {
  google::InitGoogleLogging("MMVNNIP2031P2013KernelTest");
  std::vector<int> nthr_cases = {1, 2, 3, 4, 0};

  std::vector<test_params_t> cases;

  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);

    cases.push_back({gen_case(96, 64, 96, 1, 1, nthr, false,
                              {
                                  {"src0_scale", "8"},
                                  {"src1_scale", "8"},
                              })});
    cases.push_back({gen_case(16, 64, 16, 1, 1, nthr, false,
                              std::unordered_map<std::string, std::string>{
                                  {"src0_scale", "8"},
                                  {"src1_scale", "8"},
                              })});
    cases.push_back({gen_case(64, 64, 64, 1, 1, nthr, false,
                              std::unordered_map<std::string, std::string>{
                                  {"src0_scale", "8"},
                                  {"src1_scale", "8"},
                              })});
    cases.push_back({gen_case(128, 64, 128, 3, 12, nthr, true,
                              std::unordered_map<std::string, std::string>{
                                  {"src0_scale", "12.3"},
                                  {"src1_scale", "11.2"},
                              })});
    cases.push_back({gen_case(128, 64, 128, 3, 12, nthr, true,
                              std::unordered_map<std::string, std::string>{
                                  {"src0_scale", "12.3"},
                                  {"src1_scale", "11.2"},
                                  {"out_scale", "0.125"},
                              },
                              std::vector<postop_attr>{
                                  postop_attr{data_type::s8, postop_type::eltwise, postop_alg::quantize, 0, 0, 0.0001},
                              })});
  }
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  auto& descs = tpi.param.args.first.op_desc.tensor_descs();
  auto attrs = tpi.param.args.first.op_desc.attrs();
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });

  const dim_t bs0 = shapes[io::DST0][0];
  const dim_t bs1 = shapes[io::DST0][1];
  const dim_t M = shapes[io::SRC0][3];  // aka src0_perm_shape[2]
  const dim_t K = shapes[io::SRC0][1];  // aka src0_perm_shape[3]
  const dim_t N = shapes[io::SRC1][3];  // aka src1_perm_shape[3]
  const bool has_binary_add = shapes[io::SRC2].size() != 0;
  std::vector<std::string> params;
  params.push_back("c" + std::to_string(static_cast<int>(tpi.param.args.first.nthr)));
  params.push_back(std::to_string(bs0));
  params.push_back(std::to_string(bs1));
  params.push_back(std::to_string(M));
  params.push_back(std::to_string(K));
  params.push_back(std::to_string(N));
  if (has_binary_add) params.push_back("badd");
  for (const postop_attr& p_attr : tpi.param.args.first.op_desc.apply_postops_list()) {
    params.push_back("post" + std::string(postop_alg_name.at(p_attr.op_alg)));
    params.push_back(std::string(data_type_name.at(p_attr.dt)));
    params.push_back(num2id(p_attr.alpha));
    params.push_back(num2id(p_attr.beta));
    params.push_back(num2id(p_attr.scale));
  }
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, MMVNNIP2031P2013KernelTest, case_func(), test_suffix);
}  // namespace jd
