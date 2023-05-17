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
  auto src_s8 = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data[0]));
  auto src_u8 = reinterpret_cast<uint8_t*>(const_cast<void*>(rt_data[0]));
  auto dst_dt = op_desc.tensor_descs()[1].dtype();
  void* dst = const_cast<void*>(rt_data[1]);

  std::vector<jd::postop_attr> dequant_list = {op_desc.apply_postops_list().front()};
  std::vector<jd::postop_attr> quant_list;
  if (op_desc.apply_postops_list().back().op_alg == jd::postop_alg::quantize)
    quant_list.push_back(op_desc.apply_postops_list().back());
  auto src_tensor = op_desc.tensor_descs()[0];
  auto src_dt = src_tensor.dtype();
  auto tensor_shape = src_tensor.shape();
  int row = src_tensor.reduce_rows();
  int col = tensor_shape.back();
  std::vector<float> float_dst_data(row * col, 0);
  for (int i = 0; i < row; i++) {
    // step1. find max
    float max = -256;
    for (int j = 0; j < col; j++) {
      int src_idx = i * col + j;
      if (src_dt == jd::data_type::s8) {
        max = static_cast<float>(src_s8[src_idx]) > max ? static_cast<float>(src_s8[src_idx]) : max;
      } else {
        max = static_cast<float>(src_u8[src_idx]) > max ? static_cast<float>(src_u8[src_idx]) : max;
      }
    }
    // get e^M
    max = apply_postop_list(max, dequant_list);
    // step2. compute sum of exp
    float exp_sum = 0;
    for (int j = 0; j < col; j++) {
      float value = 0;
      if (src_dt == jd::data_type::s8) {
        value = apply_postop_list(static_cast<float>(src_s8[i * col + j]), dequant_list);
      } else {
        value = apply_postop_list(static_cast<float>(src_u8[i * col + j]), dequant_list);
      }
      value = get_exp(value - max);
      float_dst_data[i * col + j] = value;
      exp_sum += value;
    }

    float scale = 1 / exp_sum;
    // step3. compute softmax
    if (dst_dt == jd::data_type::bf16) {
      for (int j = 0; j < col; j++)
        reinterpret_cast<jd::bfloat16_t*>(dst)[i * col + j] = float_dst_data[i * col + j] * scale;
    } else if (dst_dt == jd::data_type::u8) {
      for (int j = 0; j < col; j++) {
        reinterpret_cast<uint8_t*>(dst)[i * col + j] =
            (uint8_t)apply_postop_list(float_dst_data[i * col + j] * scale, quant_list);
      }
    } else if (dst_dt == jd::data_type::s8) {
      for (int j = 0; j < col; j++)
        reinterpret_cast<int8_t*>(dst)[i * col + j] =
            (int8_t)apply_postop_list(float_dst_data[i * col + j] * scale, quant_list);
    } else {
      for (int j = 0; j < col; j++) reinterpret_cast<float*>(dst)[i * col + j] = float_dst_data[i * col + j] * scale;
    }
  }
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  try {
    const auto& op_desc = p.op_desc;
    jd::softmax_desc softmax_desc(op_desc);
    jd::softmax softmax_ker(softmax_desc);
    softmax_ker.execute(p.data);
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
    auto dst_dt = q.op_desc.tensor_descs()[1].dtype();
    EXPECT_NE(buf1, buf2);
    bool ans = false;
    if (dst_dt == jd::data_type::s8)
      ans = compare_data<int8_t>(buf1, size1, buf2, size2, 1e-1);
    else if (dst_dt == jd::data_type::u8)
      ans = compare_data<uint8_t>(buf1, size1, buf2, size2, 1e-1);
    else if (dst_dt == jd::data_type::bf16)
      ans = compare_data<jd::bfloat16_t>(buf1, size1, buf2, size2, 1e-1);
    else if (dst_dt == jd::data_type::fp32)
      ans = compare_data<float>(buf1, size1, buf2, size2, 1e-1);
    else
      return ans = false;
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

class SoftmaxLutKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  SoftmaxLutKernelTest() {}
  virtual ~SoftmaxLutKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SoftmaxLutKernelTest, TestPostfix) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

std::pair<op_args_t, op_args_t> gen_case(const std::vector<jd::tensor_desc>& ts_descs,
                                         const std::unordered_map<std::string, std::string> op_attrs,
                                         const std::vector<jd::postop_attr>& postop_attr) {
  jd::operator_desc softmax_desc(jd::kernel_kind::softmax, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                                 ts_descs, op_attrs, postop_attr);

  int num = get_element_num(softmax_desc);
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
    float rand_val = std::rand() % 256 - 128;  // NOLINT
    assign_val(src, in_dt, rand_val, i);
    assign_val(src_ref, in_dt, rand_val, i);
  }
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;

  rt_data1.emplace_back(reinterpret_cast<void*>(src));
  rt_data1.emplace_back(reinterpret_cast<void*>(dst));
  rt_data2.emplace_back(reinterpret_cast<void*>(src_ref));
  rt_data2.emplace_back(reinterpret_cast<void*>(dst_ref));

  op_args_t p = {softmax_desc, rt_data1};
  op_args_t q = {softmax_desc, rt_data2};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  jd::tensor_desc data0_desc = {{8, 4, 128, 128}, jd::data_type::u8, jd::format_type::undef};
  jd::tensor_desc data1_desc = {{8, 4, 128, 128}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc data2_desc = {{1024, 1024}, jd::data_type::fp32, jd::format_type::undef};
  jd::tensor_desc data3_desc = {{1024, 1024}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc data4_desc = {{4096, 384}, jd::data_type::u8, jd::format_type::undef};
  jd::tensor_desc data5_desc = {{4096, 384}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc data6_desc = {{8, 4, 128, 126}, jd::data_type::fp32, jd::format_type::undef};
  jd::tensor_desc data7_desc = {{8, 4, 128, 126}, jd::data_type::s8, jd::format_type::undef};

  jd::postop_attr dequantize_s8_attr(jd::data_type::s8, jd::postop_type::eltwise, jd::postop_alg::dequantize, 140, 0,
                                     0.643695);
  jd::postop_attr quant_u8_attr(jd::data_type::u8, jd::postop_type::eltwise, jd::postop_alg::quantize, 0, 0,
                                0.00324144);

  cases.push_back({gen_case({data1_desc, data0_desc},
                            {{"postop_list", "dequantize+scale0.653695"}, {"vec_len", "128"}, {"spec_type", "lut"}},
                            {dequantize_s8_attr, quant_u8_attr}),
                   false});

  cases.push_back({gen_case({data3_desc, data2_desc},
                            {{"postop_list", "dequantize+scale0.04+quantiuze+scale0.00324144"},
                             {"vec_len", "1024"},
                             {"spec_type", "lut"}},
                            {dequantize_s8_attr}),
                   false});
  cases.push_back({gen_case({data5_desc, data4_desc},
                            {{"postop_list", "dequantize+scale0.653695"}, {"vec_len", "128"}, {"spec_type", "lut"}},
                            {dequantize_s8_attr, quant_u8_attr}),
                   false});

  cases.push_back({gen_case({data7_desc, data6_desc},
                            {{"postop_list", "dequantize+scale0.04+quantiuze+scale0.00324144"},
                             {"vec_len", "1024"},
                             {"spec_type", "lut"}},
                            {dequantize_s8_attr}),
                   false});
  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, SoftmaxLutKernelTest, case_func());
}  // namespace test
