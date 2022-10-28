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

#define OMP_NUM_THREADS "OMP_NUM_THREADS"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;

struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> rt_data;
  int nthr;  // 0 for not touching OMP_NUM_THREADS and using what set outside
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

void get_true_data(const operator_desc& op_desc, const std::vector<const void*>& rt_data) {
  // configure alias
  auto& descs = op_desc.tensor_descs();
  auto attrs = op_desc.attrs();
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  std::vector<jd::data_type> dtypes(descs.size());
  std::transform(descs.begin(), descs.end(), dtypes.begin(), [&](tensor_desc d) { return d.dtype(); });

  const dim_t M = shapes[ssd::SRC0][2];  // aka src0_perm_shape[2]
  const dim_t K = shapes[ssd::SRC0][3];  // aka src0_perm_shape[3]
  const dim_t N = shapes[ssd::SRC1][1];  // aka src1_perm_shape[3]
  const dim_t bs0 = shapes[ssd::SRC0][0];
  const dim_t bs1 = shapes[ssd::SRC0][1];

  // scale * src0 x src1 = dst.

  const auto& left_dt = dtypes[ssd::SRC0];
  const auto& right_dt = dtypes[ssd::SRC1];
  const auto& dst_dt = dtypes[ssd::DST0];

  /**
   *   src0: bs0 bs1 m   k ===========> bs0 bs1 m k
   *   src1: bs1 n   bs0 k ==perm2031=> bs0 bs1 k n
   *   dst:  bs1 n   bs0 m <=perm1302== bs0 bs1 m n
   */
  std::vector<dim_t> left_stride = {bs1 * M * K, M * K, K, 1};
  std::vector<dim_t> right_stride = {N * bs0 * K, bs0 * K, K, 1};
  std::vector<dim_t> dst_stride = {N * bs0 * M, bs0 * M, M, 1};

  // runtime data alias
  const auto left_data = rt_data[ssd::SRC0];
  const auto right_data = rt_data[ssd::SRC1];
  auto dst_data = const_cast<void*>(rt_data[ssd::DST0]);
  const auto scale_data = rt_data[ssd::SCALE0];

  // ptr alias
  auto left_u8 = static_cast<const uint8_t*>(left_data);
  auto right_s8 = static_cast<const int8_t*>(right_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);
  auto scale_f32 = static_cast<const float*>(scale_data);
  auto scale_value = scale_f32[0];

  // Computing the kernel
#pragma omp parallel for collapse(4)
  for (dim_t ibs0 = 0; ibs0 < bs0; ++ibs0)
    for (dim_t ibs1 = 0; ibs1 < bs1; ++ibs1)
      for (dim_t i = 0; i < M; ++i)
        for (dim_t j = 0; j < N; ++j) {
          float value = 0;
          dim_t dst_idx = ibs1 * dst_stride[0] + j * dst_stride[1] + ibs0 * dst_stride[2] + i * dst_stride[3];
#pragma omp simd
          for (dim_t k = 0; k < K; ++k) {
            dim_t l_idx = ibs0 * left_stride[0] + ibs1 * left_stride[1] + i * left_stride[2] + k * left_stride[3];
            dim_t r_idx = ibs1 * right_stride[0] + j * right_stride[1] + ibs0 * right_stride[2] + k * right_stride[3];
            auto l_value = left_dt == dt::u8 ? left_u8[l_idx] : 0.f;
            auto r_value = right_dt == dt::s8 ? right_s8[r_idx] : 0.f;
            value += l_value * r_value;
          }

          // Quantize dst data
          if (dst_dt == dt::u8) {
            jd::postop_attr quantize{
                data_type::u8,
                postop_type::eltwise,
                postop_alg::quantize,
                0,                // alpha
                0,                // beta
                1 / scale_value,  // scale
            };
            float quantized_value = apply_postop_list(value, {quantize});
            dst_u8[dst_idx] = static_cast<uint8_t>(quantized_value);
          } else {
            LOG(FATAL) << "unsupported dst type";
          }
        }
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  get_true_data(q.op_desc, q.rt_data);
  try {
    n_thread_t with_n_thread(p.nthr);
    const auto& op_desc = p.op_desc;
    transpose_matmul_desc kernel_desc(op_desc);
    transpose_matmul kernel(kernel_desc);
    kernel.execute(p.rt_data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    auto buf1 = p.rt_data[ssd::DST0];
    auto size1 = p.op_desc.tensor_descs()[ssd::DST0].size();
    auto buf2 = q.rt_data[ssd::DST0];
    auto size2 = q.op_desc.tensor_descs()[ssd::DST0].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[ssd::DST0].dtype();
    if (dst_type == dt::u8) {
      return compare_data<uint8_t>(buf1, size1, buf2, size2, 0);  // no tolerance for quantized value
    } else {
      LOG(FATAL) << "unsupported dst type";
    }
  }
  return false;
}

class MMVNNINopermP2031P1302KernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  MMVNNINopermP2031P1302KernelTest() {}
  virtual ~MMVNNINopermP2031P1302KernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MMVNNINopermP2031P1302KernelTest, ) {
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
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<dim_t>());
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
                                         std::unordered_map<std::string, std::string> attrs = {}) {
  /**
   * Step 1: Construct operator config
   *
   * Dimension details:
   *   src0: bs0 bs1 m   k ===========> bs0 bs1 m k
   *   src1: bs1 n   bs0 k ==perm2031=> bs0 bs1 k n
   *   dst:  bs1 n   bs0 m <=perm1302== bs0 bs1 m n
   */
  tensor_desc src0_desc = {{bs0, bs1, M, K}, dt::u8, ft::ab};
  tensor_desc src1_desc = {{bs1, N, bs0, K}, dt::s8, ft::ab};
  tensor_desc dst_desc = {{bs1, N, bs0, M}, dt::u8, ft::ab};
  tensor_desc src2_desc = {{}, dt::fp32, ft::ab};  // binary postop not supported
  tensor_desc scale_desc = {{1}, dt::fp32, ft::a};
  std::vector<tensor_desc> ts_descs = {src0_desc, src1_desc, dst_desc, src2_desc, scale_desc};

  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    if (index == ssd::SRC2) {
      // insert nullptr as placeholder
      rt_data1.emplace_back(nullptr);
      rt_data2.emplace_back(nullptr);
      continue;
    }
    auto& tsd = ts_descs[index];
    bool is_clear = (index == ssd::DST0);
    auto ranges = std::vector<float>{-10, 10};
    auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), is_clear, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  operator_desc op_desc(kernel_kind::transpose_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                        attrs);

  // Step 3: op_args_t testcase pair
  op_args_t op_args = {op_desc, rt_data1, nthr};
  op_args_t op_args_copy = {op_desc, rt_data2, nthr};

  return {op_args, op_args_copy};
}

static auto case_func = []() {
  google::InitGoogleLogging("MMVNNINopermP2031P1302KernelTest");
  std::vector<int> nthr_cases = {1, 2, 3, 4, 0};

  std::vector<test_params_t> cases;

  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);
    // seq = 32
    cases.push_back({gen_case(32, 32, 64, 1, 12, nthr, {})});
    cases.push_back({gen_case(32, 32, 64, 6, 12, nthr, {})});
    cases.push_back({gen_case(32, 32, 64, 8, 12, nthr, {})});

    // seq = 384
    cases.push_back({gen_case(384, 384, 64, 1, 12, nthr, {})});
    cases.push_back({gen_case(384, 384, 64, 2, 12, nthr, {})});
    cases.push_back({gen_case(384, 384, 64, 4, 12, nthr, {})});
  }
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  auto& descs = tpi.param.args.first.op_desc.tensor_descs();
  auto attrs = tpi.param.args.first.op_desc.attrs();
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });

  const dim_t bs0 = shapes[ssd::SRC0][0];
  const dim_t bs1 = shapes[ssd::SRC0][1];
  const dim_t M = shapes[ssd::SRC0][2];  // aka src0_perm_shape[2]
  const dim_t K = shapes[ssd::SRC0][3];  // aka src0_perm_shape[3]
  const dim_t N = shapes[ssd::SRC1][1];  // aka src1_perm_shape[3]
  std::vector<std::string> params;
  params.push_back("c" + std::to_string(static_cast<int>(tpi.param.args.first.nthr)));
  params.push_back(std::to_string(bs0));
  params.push_back(std::to_string(bs1));
  params.push_back(std::to_string(M));
  params.push_back(std::to_string(K));
  params.push_back(std::to_string(N));
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, MMVNNINopermP2031P1302KernelTest, case_func(), test_suffix);
}  // namespace jd
