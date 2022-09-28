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

namespace jd {
int get_element_num(const operator_desc& op_desc) {
  auto ts_descs = op_desc.tensor_descs();
  int num = 1;
  for (auto&& i : ts_descs[0].shape()) num *= i;
  return num;
}

struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> data;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

void get_true_data(const operator_desc& op_desc, const std::vector<const void*>& rf_data) {
  float* src = reinterpret_cast<float*>(const_cast<void*>(rf_data[0]));
  float* dst = reinterpret_cast<float*>(const_cast<void*>(rf_data[1]));

  int num = get_element_num(op_desc);
  auto attr = op_desc.apply_postops_list();
  for (int i = 0; i < num; i++) {
    dst[i] = src[i];
    dst[i] = apply_postop_list(dst[i], attr);
  }
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  try {
    const auto& op_desc = p.op_desc;
    eltwiseop_desc eltwiseop_desc(op_desc);
    eltwiseop eltwiseop_kern(eltwiseop_desc);
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
    int num = get_element_num(q.op_desc);
    void* buf1;
    auto buf2 = q.data[1];
    auto dtype = q.op_desc.tensor_descs()[1].dtype();
    float err_rate;
    if (dtype == jd::data_type::fp32) {
      buf1 = const_cast<void*>(p.data[1]);
      err_rate = 1e-1;
    } else if (dtype == jd::data_type::bf16) {
      buf1 = reinterpret_cast<float*>(malloc(num * sizeof(float)));
      auto bf16_buf1 = const_cast<void*>(p.data[1]);
      for (int i = 0; i < num; i++) {
        *(reinterpret_cast<float*>(buf1) + i) = bf16_2_fp32(*(reinterpret_cast<bfloat16_t*>(bf16_buf1) + i));
      }
      free(bf16_buf1);
      err_rate = 5;
    } else if (dtype == jd::data_type::s8 || dtype == jd::data_type::u8) {
      err_rate = 1e-1;
      buf1 = reinterpret_cast<float*>(malloc(num * sizeof(float)));
      auto int8_buf1 = const_cast<void*>(p.data[1]);
      if (dtype == jd::data_type::u8) {
        for (int i = 0; i < num; i++)
          *(reinterpret_cast<float*>(buf1) + i) = uint8_2_int32(*(reinterpret_cast<uint8_t*>(int8_buf1) + i));
      } else {
        for (int i = 0; i < num; i++)
          *(reinterpret_cast<float*>(buf1) + i) = *(reinterpret_cast<int8_t*>(int8_buf1) + i);
      }
      free(int8_buf1);
    }
    EXPECT_NE(buf1, buf2);
    auto ans = compare_data<float>(buf1, num, buf2, num, err_rate);
    free(const_cast<void*>(p.data[0]));
    free(buf1);
    free(const_cast<void*>(q.data[0]));
    free(const_cast<void*>(q.data[1]));
    return ans;
  }
  free(const_cast<void*>(p.data[0]));
  free(const_cast<void*>(p.data[1]));
  free(const_cast<void*>(q.data[0]));
  free(const_cast<void*>(q.data[1]));
  return false;
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

std::pair<op_args_t, op_args_t> gen_case(const std::vector<tensor_desc>& ts_descs,
                                         const std::unordered_map<std::string, std::string> op_attrs,
                                         const std::vector<postop_attr>& postop_attr) {
  operator_desc eltwiseop_desc(kernel_kind::eltwiseop, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                               op_attrs, postop_attr);

  int num = get_element_num(eltwiseop_desc);
  void* src = nullptr;
  void* dst = nullptr;
  memo_mode MALLOC = memo_mode::MALLOC;
  memo_mode MEMSET = memo_mode::MEMSET;

  auto in_dt = ts_descs[0].dtype();
  auto out_dt = ts_descs[1].dtype();

  src = sparselib_ut_memo(src, num, in_dt, MALLOC);
  dst = sparselib_ut_memo(dst, num, out_dt, MALLOC);
  sparselib_ut_memo(dst, num, out_dt, MEMSET);

  float* src_ref = reinterpret_cast<float*>(malloc(num * sizeof(float)));
  float* dst_ref = reinterpret_cast<float*>(malloc(num * sizeof(float)));

  const unsigned int seed = 667095;
  memset(dst_ref, 0, num * sizeof(float));
  for (int i = 0; i < num; i++) {
    unsigned int seed_tmp = seed + i;
    float rand_val = rand_r(&seed_tmp) % 256 - 128;
    assign_val(src, in_dt, rand_val, i);
    if (in_dt == data_type::u8)
      src_ref[i] = *(reinterpret_cast<uint8_t*>(src) + i);
    else
      src_ref[i] = rand_val;
  }

  std::vector<const void*> rf_data1;
  std::vector<const void*> rf_data2;

  rf_data1.emplace_back(reinterpret_cast<void*>(src));
  rf_data1.emplace_back(reinterpret_cast<void*>(dst));
  rf_data2.emplace_back(reinterpret_cast<void*>(src_ref));
  rf_data2.emplace_back(reinterpret_cast<void*>(dst_ref));

  op_args_t p = {eltwiseop_desc, rf_data1};
  op_args_t q = {eltwiseop_desc, rf_data2};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  tensor_desc data0_desc = {{1024, 1024}, jd::data_type::fp32, jd::format_type::undef};
  tensor_desc data1_desc = {{1024, 1024}, jd::data_type::bf16, jd::format_type::undef};
  tensor_desc data2_desc = {{64, 1}, jd::data_type::fp32, jd::format_type::undef};
  tensor_desc data3_desc = {{64, 1}, jd::data_type::bf16, jd::format_type::undef};
  tensor_desc data4_desc = {{1024, 1024}, jd::data_type::u8, jd::format_type::undef};
  tensor_desc data5_desc = {{1024, 1024}, jd::data_type::s8, jd::format_type::undef};
  tensor_desc data6_desc = {{64, 1}, jd::data_type::u8, jd::format_type::undef};

  postop_attr fp32_exp_attr{data_type::fp32, postop_type::eltwise, postop_alg::exp};
  postop_attr bf16_exp_attr{data_type::bf16, postop_type::eltwise, postop_alg::exp};
  postop_attr fp32_gelu_attr{data_type::fp32, postop_type::eltwise, postop_alg::gelu};
  postop_attr bf16_gelu_attr{data_type::bf16, postop_type::eltwise, postop_alg::gelu};
  postop_attr fp32_relu_attr{data_type::fp32, postop_type::eltwise, postop_alg::relu, 2.0};
  postop_attr quantize_s8_attr(data_type::s8, postop_type::eltwise, postop_alg::quantize, -10, 0, 2);
  postop_attr quantize_u8_attr(data_type::u8, postop_type::eltwise, postop_alg::quantize, -10, 0, 2);
  postop_attr dequantize_u8_attr(data_type::u8, postop_type::eltwise, postop_alg::dequantize, -10, 0, 2);
  postop_attr dequantize_s8_attr(data_type::s8, postop_type::eltwise, postop_alg::dequantize, -10, 0, 2);
  postop_attr fp32_tanh_attr{data_type::fp32, postop_type::eltwise, postop_alg::tanh};
  // u8 as input dt
  postop_attr int8_lut_u8_attr{data_type::u8, postop_type::eltwise, postop_alg::int8_lut};
  // s8 as input dt
  postop_attr int8_lut_s8_attr{data_type::s8, postop_type::eltwise, postop_alg::int8_lut};

  cases.push_back({gen_case({data4_desc, data4_desc}, {{"postop_list", "int8_lut_u8+dequantize+fp32_gelu+quantize"}},
                            {int8_lut_u8_attr, dequantize_u8_attr, fp32_gelu_attr, quantize_u8_attr}),
                   false});

  cases.push_back({gen_case({data5_desc, data5_desc}, {{"postop_list", "int8_lut_s8+dequantize+fp32_gelu+quantize"}},
                            {int8_lut_s8_attr, dequantize_s8_attr, fp32_gelu_attr, quantize_s8_attr}),
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

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, EltwiseopKernelTest, case_func());
}  // namespace jd
