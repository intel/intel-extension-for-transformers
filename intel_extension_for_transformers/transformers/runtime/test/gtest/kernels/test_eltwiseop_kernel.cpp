//  Copyright (c) 2022 Intel Corporation
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
#include <map>
#include "gtest/gtest.h"
#include "unit_test_utils.hpp"
#include "interface.hpp"

namespace test {
struct op_args_t {
  jd::operator_desc op_desc;
  std::vector<const void*> data;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

void get_true_data(const jd::operator_desc& op_desc, const std::vector<const void*>& rt_data) {
  auto src_tensor = op_desc.tensor_descs()[0];
  auto dst_tensor = op_desc.tensor_descs()[1];
  int size = src_tensor.size();
  auto src_dt = src_tensor.dtype();
  auto dst_dt = dst_tensor.dtype();

  const void* src = rt_data[0];
  void* dst = const_cast<void*>(rt_data[1]);
  float* src_fp32 = new float[size];
  if (src_dt == jd::data_type::s8) {
    cast_to_float_array<int8_t>(src, src_fp32, size);
  } else if (src_dt == jd::data_type::u8) {
    cast_to_float_array<uint8_t>(src, src_fp32, size);
  } else if (src_dt == jd::data_type::bf16) {
    cast_to_float_array<jd::bfloat16_t>(src, src_fp32, size);
  } else if (src_dt == jd::data_type::s32) {
    cast_to_float_array<int>(src, src_fp32, size);
  } else if (src_dt == jd::data_type::fp32) {
    cast_to_float_array<float>(src, src_fp32, size);
  }
  auto attr = op_desc.apply_postops_list();
  for (int i = 0; i < size; i++) {
    src_fp32[i] = apply_postop_list(src_fp32[i], attr);
  }
  if (dst_dt == jd::data_type::s8) {
    cast_from_float_array<int8_t>(src_fp32, dst, size);
  } else if (dst_dt == jd::data_type::u8) {
    cast_from_float_array<uint8_t>(src_fp32, dst, size);
  } else if (dst_dt == jd::data_type::bf16) {
    cast_from_float_array<jd::bfloat16_t>(src_fp32, dst, size);
  } else if (dst_dt == jd::data_type::s32) {
    cast_from_float_array<int>(src_fp32, dst, size);
  } else if (dst_dt == jd::data_type::fp32) {
    cast_from_float_array<float>(src_fp32, dst, size);
  }
  delete[] src_fp32;
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  try {
    const auto& op_desc = p.op_desc;
    jd::eltwiseop_desc eltwiseop_desc(op_desc);
    jd::eltwiseop eltwiseop_kern(eltwiseop_desc);
    eltwiseop_kern.execute(p.data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }

  if (!t.expect_to_fail) {
    get_true_data(q.op_desc, q.data);
    auto buf1 = p.data[1];
    auto size1 = p.op_desc.tensor_descs()[1].size();
    auto buf2 = q.data[1];
    auto size2 = q.op_desc.tensor_descs()[1].size();
    auto dst_type = p.op_desc.tensor_descs()[1].dtype();
    EXPECT_NE(buf1, buf2);
    bool ans = false;
    if (dst_type == jd::data_type::fp32) {
      ans = compare_data<float>(buf1, size1, buf2, size2, 1e-1);
    } else if (dst_type == jd::data_type::u8) {
      ans = compare_data<uint8_t>(buf1, size1, buf2, size2, 1e-2);
    } else if (dst_type == jd::data_type::s8) {
      ans = compare_data<int8_t>(buf1, size1, buf2, size2, 1e-2);
    } else if (dst_type == jd::data_type::bf16) {
      ans = compare_data<jd::bfloat16_t>(buf1, size1, buf2, size2, 1e-2);
    } else if (dst_type == jd::data_type::s32) {
      ans = compare_data<int>(buf1, size1, buf2, size2, 1e-2);
    }
    free(const_cast<void*>(p.data[0]));
    free(const_cast<void*>(p.data[1]));
    free(const_cast<void*>(q.data[0]));
    free(const_cast<void*>(q.data[1]));
    return ans;
  }
  free(const_cast<void*>(p.data[0]));
  free(const_cast<void*>(p.data[1]));
  free(const_cast<void*>(q.data[0]));
  free(const_cast<void*>(q.data[1]));
  return true;
}

class EltwiseopKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  EltwiseopKernelTest() {}
  virtual ~EltwiseopKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(EltwiseopKernelTest, TestPostfix) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

std::pair<op_args_t, op_args_t> gen_case(const std::vector<jd::tensor_desc>& ts_descs,
                                         const std::unordered_map<std::string, std::string> op_attrs,
                                         const std::vector<jd::postop_attr>& postop_attr) {
  jd::operator_desc eltwiseop_desc(jd::kernel_kind::eltwiseop, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                                   ts_descs, op_attrs, postop_attr);

  int num = get_element_num(eltwiseop_desc);
  void* src = nullptr;
  void* dst = nullptr;
  void* src_ref = nullptr;
  void* dst_ref = nullptr;
  memo_mode MALLOC = memo_mode::MALLOC;
  memo_mode MEMSET = memo_mode::MEMSET;

  auto in_dt = ts_descs[0].dtype();
  auto out_dt = ts_descs[1].dtype();

  src = sparselib_ut_memo(src, num, in_dt, MALLOC);
  dst = sparselib_ut_memo(dst, num, out_dt, MALLOC);
  dst = sparselib_ut_memo(dst, num, out_dt, MEMSET);
  src_ref = sparselib_ut_memo(src_ref, num, in_dt, MALLOC);
  dst_ref = sparselib_ut_memo(dst_ref, num, out_dt, MALLOC);
  dst_ref = sparselib_ut_memo(dst_ref, num, out_dt, MEMSET);

  const unsigned int seed = 667095;
  std::srand(seed);
  for (int i = 0; i < num; i++) {
    float rand_val = std::rand() % 256 - 128;
    assign_val(src, in_dt, rand_val, i);
    assign_val(src_ref, in_dt, rand_val, i);
  }

  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;

  rt_data1.emplace_back(reinterpret_cast<void*>(src));
  rt_data1.emplace_back(reinterpret_cast<void*>(dst));
  rt_data2.emplace_back(reinterpret_cast<void*>(src_ref));
  rt_data2.emplace_back(reinterpret_cast<void*>(dst_ref));

  op_args_t p = {eltwiseop_desc, rt_data1};
  op_args_t q = {eltwiseop_desc, rt_data2};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  jd::tensor_desc data0_desc = {{1024, 1024}, jd::data_type::fp32, jd::format_type::undef};
  jd::tensor_desc data1_desc = {{1024, 1024}, jd::data_type::bf16, jd::format_type::undef};
  jd::tensor_desc data2_desc = {{64, 1}, jd::data_type::fp32, jd::format_type::undef};
  jd::tensor_desc data3_desc = {{64, 1}, jd::data_type::bf16, jd::format_type::undef};
  jd::tensor_desc data4_desc = {{1024, 1024}, jd::data_type::u8, jd::format_type::undef};
  jd::tensor_desc data5_desc = {{1024, 1024}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc data6_desc = {{64, 1}, jd::data_type::u8, jd::format_type::undef};

  jd::postop_attr fp32_exp_attr{jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::exp};
  jd::postop_attr bf16_exp_attr{jd::data_type::bf16, jd::postop_type::eltwise, jd::postop_alg::exp};
  jd::postop_attr fp32_swish_attr{jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::swish, 2};
  jd::postop_attr bf16_swish_attr{jd::data_type::bf16, jd::postop_type::eltwise, jd::postop_alg::swish, 2};
  jd::postop_attr fp32_gelu_attr{jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::gelu};
  jd::postop_attr bf16_gelu_attr{jd::data_type::bf16, jd::postop_type::eltwise, jd::postop_alg::gelu};
  jd::postop_attr fp32_relu_attr{jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::relu, 2.0};
  jd::postop_attr quantize_s8_attr(jd::data_type::s8, jd::postop_type::eltwise, jd::postop_alg::quantize, 0, 0, 0.04);
  jd::postop_attr quantize_u8_attr(jd::data_type::u8, jd::postop_type::eltwise, jd::postop_alg::quantize, 0, 0, 0.04);
  jd::postop_attr dequantize_u8_attr(jd::data_type::u8, jd::postop_type::eltwise, jd::postop_alg::dequantize, 0, 0,
                                     0.04);
  jd::postop_attr dequantize_s8_attr(jd::data_type::s8, jd::postop_type::eltwise, jd::postop_alg::dequantize, 0, 0,
                                     0.04);
  jd::postop_attr fp32_tanh_attr{jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::tanh};
  jd::postop_attr bit8_lut_u8_attr{jd::data_type::u8, jd::postop_type::eltwise, jd::postop_alg::eltop_int_lut,
                                   8};  // u8 as input jd::data_type
  jd::postop_attr bit8_lut_s8_attr{jd::data_type::s8, jd::postop_type::eltwise, jd::postop_alg::eltop_int_lut,
                                   8};  // s8 as input jd::data_type
  jd::postop_attr bit16_lut_u8_attr{jd::data_type::u8, jd::postop_type::eltwise, jd::postop_alg::eltop_int_lut, 16,
                                    256};

  cases.push_back({gen_case({data5_desc, data0_desc}, {{"postop_list", "s8dequantize"}}, {dequantize_s8_attr}), false});

  cases.push_back({gen_case({data0_desc, data5_desc}, {{"postop_list", "s8quantize"}}, {quantize_s8_attr}), false});

  cases.push_back({gen_case({data4_desc, data1_desc}, {{"postop_list", "bit16_lut_u8+dequantize+bf16_exp"}},
                            {bit16_lut_u8_attr, dequantize_u8_attr, bf16_exp_attr}),
                   false});

  cases.push_back({gen_case({data6_desc, data3_desc}, {{"postop_list", "bit16_lut_u8+dequantize+bf16_exp"}},
                            {bit16_lut_u8_attr, dequantize_u8_attr, bf16_exp_attr}),
                   false});

  cases.push_back({gen_case({data4_desc, data4_desc}, {{"postop_list", "bit8_lut_u8+dequantize+fp32_gelu+quantize"}},
                            {bit8_lut_u8_attr, dequantize_u8_attr, fp32_gelu_attr, quantize_u8_attr}),
                   false});

  cases.push_back({gen_case({data5_desc, data5_desc}, {{"postop_list", "bit8_lut_s8+dequantize+fp32_gelu+quantize"}},
                            {bit8_lut_s8_attr, dequantize_s8_attr, fp32_gelu_attr, quantize_s8_attr}),
                   false});

  cases.push_back({gen_case({data0_desc, data0_desc}, {{"postop_list", "fp32_tanh"}}, {fp32_tanh_attr}), false});

  cases.push_back({gen_case({data4_desc, data4_desc}, {{"postop_list", "dequantize+fp32_gelu+quantize"}},
                            {dequantize_u8_attr, fp32_gelu_attr, quantize_u8_attr}),
                   false});

  cases.push_back(
      {gen_case({data0_desc, data5_desc}, {{"postop_list", "fp32_gelu+quantize"}}, {fp32_gelu_attr, quantize_s8_attr}),
       false});

  cases.push_back({gen_case({data0_desc, data0_desc}, {{"postop_list", "fp32_relu"}}, {fp32_relu_attr}), false});

  cases.push_back(
      {gen_case({data0_desc, data0_desc}, {{"postop_list", "fp32_gelu+fp32_exp"}}, {fp32_gelu_attr, fp32_exp_attr}),
       false});
  cases.push_back(
      {gen_case({data1_desc, data1_desc}, {{"postop_list", "bf16_gelu+bf16_exp"}}, {bf16_gelu_attr, bf16_exp_attr}),
       false});
  cases.push_back(
      {gen_case({data2_desc, data2_desc}, {{"postop_list", "fp32_gelu+fp32_exp"}}, {fp32_gelu_attr, fp32_exp_attr}),
       false});
  cases.push_back(
      {gen_case({data3_desc, data3_desc}, {{"postop_list", "bf16_gelu+bf16_exp"}}, {bf16_gelu_attr, bf16_exp_attr}),
       false});

  cases.push_back({gen_case({data0_desc, data0_desc}, {{"postop_list", "fp32_exp"}}, {fp32_exp_attr}), false});
  cases.push_back({gen_case({data0_desc, data0_desc}, {{"postop_list", "fp32_gelu"}}, {fp32_gelu_attr}), false});
  cases.push_back({gen_case({data1_desc, data1_desc}, {{"postop_list", "bf16_exp"}}, {bf16_exp_attr}), false});
  cases.push_back({gen_case({data1_desc, data1_desc}, {{"postop_list", "bf16_gelu"}}, {bf16_gelu_attr}), false});

  cases.push_back({gen_case({data2_desc, data2_desc}, {{"postop_list", "fp32_exp"}}, {fp32_exp_attr}), false});
  cases.push_back({gen_case({data2_desc, data2_desc}, {{"postop_list", "fp32_gelu"}}, {fp32_gelu_attr}), false});
  cases.push_back({gen_case({data3_desc, data3_desc}, {{"postop_list", "bf16_exp"}}, {bf16_exp_attr}), false});
  cases.push_back({gen_case({data3_desc, data3_desc}, {{"postop_list", "bf16_gelu"}}, {bf16_gelu_attr}), false});

  cases.push_back({gen_case({data2_desc, data2_desc}, {{"postop_list", "fp32_swish"}}, {fp32_swish_attr}), false});
  cases.push_back({gen_case({data3_desc, data3_desc}, {{"postop_list", "bf16_swish"}}, {bf16_swish_attr}), false});
  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, EltwiseopKernelTest, case_func());
}  // namespace test
