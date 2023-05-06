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

#include "gtest/gtest.h"
#include "unit_test_utils.hpp"
#include "kernels/matmul_types.hpp"
#include "interface.hpp"
#include "kernels/matmul_ref.hpp"
#include "utils.hpp"
#include "kernels/data_pack.hpp"

#define OMP_NUM_THREADS "OMP_NUM_THREADS"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;
using io = ssd::matmul_io::io;

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

    std::shared_ptr<const kernel_desc_t> matmul_ref_desc;
    kernel_desc_t::create<matmul_ref_kd_t>(matmul_ref_desc, q.op_desc);
    std::shared_ptr<const kernel_t> matmul_ref_ref_ker;
    kernel_t::create<matmul_ref_k_t, matmul_ref_kd_t>(matmul_ref_ref_ker, matmul_ref_desc);
    matmul_ref_ref_ker->execute(q.rt_data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
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
      return compare_data<uint8_t>(buf1, size1, buf2, size2, 1);
    } else if (dst_type == dt::s8) {
      return compare_data<int8_t>(buf1, size1, buf2, size2, 1);
    } else if (dst_type == dt::bf16) {
      return compare_data<bfloat16_t>(buf1, size1, buf2, size2, 5e-2);
    }
  }
  return false;
}

class MMAVX512FP8KernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  MMAVX512FP8KernelTest() {}
  virtual ~MMAVX512FP8KernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MMAVX512FP8KernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
  for (auto op_args : {t.args.first, t.args.second}) {
    for (auto rt_data : op_args.rt_data) {
      char* data = reinterpret_cast<char*>(const_cast<void*>(rt_data));
      delete[] data;
    }
    auto attr = op_args.op_desc.attrs();
    if (attr["weight_8bit"] != "") {
      int8_t* weight = reinterpret_cast<int8_t*>(str_to_num<intptr_t>(attr["weight_8bit"]));
      delete[] weight;
    }
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
    } else if (a_dt == dt::bf16) {
      data_ptr = new bfloat16_t[elem_num];
      init_vector(static_cast<bfloat16_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::f8_e4m3 || a_dt == dt::f8_e5m2) {
      data_ptr = new float8_t[elem_num];
      float8_t* fp8 = reinterpret_cast<float8_t*>(data_ptr);
      float* fp32 = new float[elem_num];
      init_vector(fp32, elem_num, ranges[0], ranges[1]);
      for (int i = 0; i < elem_num; i++) {
        fp8[i] = cast_to<float8_t>(fp32[i], a_dt);
      }
      delete[] fp32;
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, data_type src1_dtype = dt::bf16, int nthr = 0,
                                         std::unordered_map<std::string, std::string> attrs = {},
                                         std::vector<postop_attr> postop_attr = {}) {
  // Step 1: Construct operator config
  tensor_desc src0_desc = {{M, K}, dt::bf16, ft::ab};
  tensor_desc src1_desc = {{N, K}, src1_dtype, ft::ab};
  tensor_desc dst_desc = {{M, N}, dt::bf16, ft::ab};
  tensor_desc bias_desc = {{N}, dt::bf16, ft::a};
  tensor_desc scale_desc = {{N}, dt::fp32, ft::a};
  std::vector<tensor_desc> ts_descs = {src0_desc, src1_desc, dst_desc, bias_desc, scale_desc};

  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    bool is_clear = (index == io::DST0);
    auto ranges = std::vector<float>{-10, 10};
    auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), is_clear, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  attrs["thread_nums"] = std::to_string(nthr);
  std::unordered_map<std::string, std::string> attrs1 = attrs;
  std::unordered_map<std::string, std::string> attrs2 = attrs;

  if (src1_dtype == dt::bf16) {
    attrs1["weight_bf16"] = std::to_string(reinterpret_cast<intptr_t>(rt_data1[io::SRC1]));
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(new float8_t[N * K]));
    attrs2["weight_bf16"] = std::to_string(reinterpret_cast<intptr_t>(rt_data2[io::SRC1]));
    attrs2["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(new float8_t[N * K]));
  } else if (src1_dtype == data_type::s8) {
    std::function<int8_t(int8_t)> cast_func_s8 = [](int8_t x) { return x; };
    int8_t* weight_8bit = new int8_t[N * K];
    int8_t* src1_s8 = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data1[io::SRC1]));
    pack<int8_t, int8_t>(weight_8bit, src1_s8, N, K, cast_func_s8);
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_8bit));
  } else if (src1_dtype == data_type::f8_e4m3 || src1_dtype == data_type::f8_e5m2) {
    std::function<float8_t(float8_t)> cast_func_fp8 = [](float8_t x) { return x; };
    float8_t* src1_fp8 = reinterpret_cast<float8_t*>(const_cast<void*>(rt_data1[io::SRC1]));
    float8_t* weight_8bit = new float8_t[N * K];
    pack<float8_t, float8_t>(weight_8bit, src1_fp8, N, K, cast_func_fp8);
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_8bit));
  }

  operator_desc op_desc1(kernel_kind::transpose_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                         attrs1, postop_attr);
  operator_desc op_desc2(kernel_kind::transpose_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                         attrs2, postop_attr);

  op_args_t op_args = {op_desc1, rt_data1, nthr};
  op_args_t op_args_copy = {op_desc2, rt_data2, nthr};
  return {op_args, op_args_copy};
}

static auto case_func = []() {
  google::InitGoogleLogging("MMAVX512FP8KernelTest");
  std::vector<int> nthr_cases = {1, 2, 3, 4};
  std::vector<test_params_t> cases;
  postop_attr fp32_swish_attr{data_type::fp32, postop_type::eltwise, postop_alg::swish, 2};
  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);
    cases.push_back({gen_case(4, 2, 16, dt::f8_e4m3, nthr, {{"alpha", "1.f"}, {"beta", "0.f"}})});
    cases.push_back({gen_case(4, 2, 16, dt::f8_e5m2, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back({gen_case(4, 2, 16, dt::s8, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back({gen_case(4, 4096, 4096, dt::f8_e5m2, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back({gen_case(4, 4096, 4096, dt::s8, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back(
        {gen_case(4, 4096, 4096, dt::bf16, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}, {"weight_type", "f8_e5m2"}})});
    cases.push_back({gen_case(4, 4096, 4096, dt::bf16, nthr,
                              {{"alpha", "1.f"},
                               {"beta", "1.f"},
                               {"thread_nums", std::to_string(nthr)},
                               {"weight_type", "f8_e5m2"},
                               {"postop_list", "fp32_swish"}},
                              {fp32_swish_attr})});
  }
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  auto attrs = tpi.param.args.first.op_desc.attrs();
  const auto shapes = tpi.param.args.first.op_desc.tensor_shapes();
  const auto dtypes = tpi.param.args.first.op_desc.tensor_dtypes();

  const dim_t M = shapes[io::SRC0][0];
  const dim_t K = shapes[io::SRC0][1];
  const dim_t N = shapes[io::SRC1][0];
  std::vector<std::string> params;
  params.push_back("c" + std::to_string(static_cast<int>(tpi.param.args.first.nthr)));
  params.push_back(std::to_string(M));
  params.push_back(std::to_string(K));
  params.push_back(std::to_string(N));
  params.push_back(data_type_name.at(dtypes[io::SRC1]));
  if (attrs["alpha"] != "" && str_to_num<float>(attrs["alpha"]) != 1.f)
    params.push_back(std::string("alpha") + num2id(attrs["alpha"]));
  if (attrs["weight_type"] != "") {
    params.push_back(attrs["weight_type"]);
  }
  if (attrs["postop_list"] != "") {
    params.push_back(attrs["postop_list"]);
  }
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, MMAVX512FP8KernelTest, case_func(), test_suffix);
}  // namespace jd
