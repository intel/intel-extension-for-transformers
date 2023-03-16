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
#include "kernels/dynamic_quant_ref.hpp"

namespace jd {

struct op_args_t {
  operator_desc op_desc;
  std::shared_ptr<std::vector<bfloat16_t>> bf16_mat;
  std::shared_ptr<std::vector<float>> fp32_mat;
  std::shared_ptr<std::vector<float>> scale;
  std::shared_ptr<std::vector<int8_t>> dst;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  const auto& op_desc = p.op_desc;
  auto op_attr = op_desc.attrs();
  std::vector<const void*> data1, data2;
  try {
    dynamic_quant_desc dynamic_quant_desc(op_desc);
    dynamic_quant dynamic_quant_ker(dynamic_quant_desc);

    data1 = {p.dst->data(), p.scale->data()};
    data2 = {q.dst->data(), q.scale->data()};
    if (op_attr["input_dt"] == "bf16") {
      data1.insert(data1.begin(), p.bf16_mat->data());
      data2.insert(data2.begin(), q.bf16_mat->data());
    } else {
      data1.insert(data1.begin(), p.fp32_mat->data());
      data2.insert(data2.begin(), q.fp32_mat->data());
    }

    dynamic_quant_ker.execute(data1);
    std::shared_ptr<const kernel_desc_t> dynamic_quant_ref_desc;
    kernel_desc_t::create<dynamic_quant_ref_kd_t>(dynamic_quant_ref_desc, q.op_desc);
    std::shared_ptr<const kernel_t> dynamic_quant_ref_ker;
    kernel_t::create<dynamic_quant_ref_k_t, dynamic_quant_ref_kd_t>(dynamic_quant_ref_ker, dynamic_quant_ref_desc);
    dynamic_quant_ref_ker->execute(data2);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }

  if (!t.expect_to_fail) {
    auto buf1 = data1[1];
    auto size = p.dst->size();
    auto buf2 = data2[1];
    auto ans1 = compare_data<int8_t>(buf1, size, buf2, size, 1e-2);
    auto buf3 = data1[2];
    auto size2 = p.scale->size();
    auto buf4 = data2[2];
    auto ans2 = compare_data<float>(buf3, size2, buf4, size2, 5e-3);
    return ans1 && ans2;
  }
  return false;
}

class PerChannelDynamicQuantKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  PerChannelDynamicQuantKernelTest() {}
  virtual ~PerChannelDynamicQuantKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(PerChannelDynamicQuantKernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

std::pair<op_args_t, op_args_t> gen_case(const std::vector<tensor_desc>& ts_descs,
                                         std::unordered_map<std::string, std::string> op_attrs) {
  operator_desc dynamic_quant_desc(kernel_kind::dynamic_quant, kernel_prop::forward_inference, engine_kind::cpu,
                                   ts_descs, op_attrs);

  auto gen_data = [](auto type, int size, float bound1, float bound2, bool clear = false) {
    auto ptr = std::shared_ptr<std::vector<decltype(type)>>(new std::vector<decltype(type)>(size, 0));
    if (!clear) init_vector(ptr->data(), ptr->size(), bound1, bound2);
    return ptr;
  };

  auto mat_size = ts_descs[0].size();
  auto scale_size = ts_descs[1].size();
  auto fp32_mat = gen_data(static_cast<float>(1), mat_size, 100.f, 300.f);
  auto scale = gen_data(static_cast<float>(1), scale_size, 0.f, 0.f, true);
  auto correct_scale = gen_data(static_cast<float>(1), scale_size, 0.f, 0.f, true);
  std::shared_ptr<std::vector<bfloat16_t>> bf16_mat(new std::vector<bfloat16_t>(mat_size, 0));
  cast_from_float_array<bfloat16_t>(*fp32_mat, bf16_mat->data(), mat_size);
  auto dst_mat = gen_data(static_cast<int8_t>(1), mat_size, 0.f, 0.f, true);
  auto correct_dst_mat = gen_data(static_cast<int8_t>(1), mat_size, 0.f, 0.f, true);

  op_args_t p = {dynamic_quant_desc, bf16_mat, fp32_mat, scale, dst_mat};
  op_args_t q = {dynamic_quant_desc, bf16_mat, fp32_mat, correct_scale, correct_dst_mat};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;
  std::vector<std::vector<int64_t>> problem_size = {{512, 10240}, {512, 1280},  {2048, 5120},
                                                    {2048, 640},  {8192, 2560}, {8192, 320}};
  for (auto&& shape : problem_size) {
    tensor_desc mat_desc = {shape, jd::data_type::bf16, jd::format_type::undef};
    tensor_desc scale_desc = {{shape[0]}, jd::data_type::fp32, jd::format_type::undef};
    tensor_desc dst_mat_desc = {shape, jd::data_type::s8, jd::format_type::undef};
    cases.push_back({gen_case({mat_desc, scale_desc, dst_mat_desc}, {{"input_dt", "bf16"}}), false});
  }

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(SparseLib, PerChannelDynamicQuantKernelTest, case_func());
}  // namespace jd
