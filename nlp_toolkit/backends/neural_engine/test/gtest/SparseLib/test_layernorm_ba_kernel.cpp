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
#include "benchmark_utils.hpp"
#include "unit_test_utils.hpp"

namespace jd {
struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> data;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

int get_element_num(const operator_desc& op_desc) {
  auto tensor_desc = op_desc.tensor_descs();
  int row = tensor_desc[0].shape()[0];
  int col = tensor_desc[0].shape()[1];
  return row * col;
}

void get_true_data(const operator_desc& op_desc, const std::vector<const void*>& rf_data) {
  auto tensor_desc = op_desc.tensor_descs();
  int row = tensor_desc[0].shape()[0];
  int col = tensor_desc[0].shape()[1];
  float* dst = nullptr;
  float* alpha = nullptr;
  float* beta = nullptr;

  dst = (float*)rf_data[0];
  alpha = (float*)rf_data[1];
  beta = (float*)rf_data[2];

  for (int i = 0; i < col; i++) {
    // calculate mean.
    float mean = 0;
    for (int j = 0; j < row; j++) mean += dst[j * col + i];
    mean /= row;
    // calculate var
    float var = 0;
    for (int j = 0; j < row; j++) var += (dst[j * col + i] - mean) * (dst[j * col + i] - mean);
    var /= row;
    var += 1e-5;
    var = sqrt(var);
    var = 1 / var;
    // calculate layernorm.
    for (int j = 0; j < row; j++) dst[j * col + i] = (dst[j * col + i] - mean) * var;

    // affine.
    if (op_desc.attrs().count("affine") != 0)
      for (int j = 0; j < row; j++) dst[j * col + i] = dst[j * col + i] * alpha[j] + beta[j];
  }

  // apply postop.
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++) dst[i * col + j] = apply_postop_list(dst[i * col + j], op_desc.apply_postops_list());
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  try {
    const auto& op_desc = p.op_desc;
    layernorm_ba_desc layernorm_ba_desc(op_desc);
    layernorm_ba layernorm_ba_ker(layernorm_ba_desc);
    benchmarkOrExecute(&layernorm_ba_ker, p.data);
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
    float err_rate = 1e-3;
    auto buf2 = q.data[0];
    auto buf1 = p.data[0];
    EXPECT_NE(buf1, buf2);
    auto ans = compare_data<float>(buf1, num, buf2, num, err_rate);
    free(const_cast<void*>(p.data[0]));
    free(const_cast<void*>(q.data[0]));
    free(const_cast<void*>(q.data[1]));
    free(const_cast<void*>(q.data[2]));
    return ans;
  }
  return false;
}

class LayernormBaKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  LayernormBaKernelTest() {}
  virtual ~LayernormBaKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(LayernormBaKernelTest, TestPostfix) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

std::pair<op_args_t, op_args_t> gen_case(const std::vector<tensor_desc>& ts_descs,
                                         std::unordered_map<std::string, std::string> op_attrs, bool affine = false,
                                         const std::vector<postop_attr>& postop_attr = {}) {
  // malloc memory
  int row = ts_descs[0].shape()[0];
  int col = ts_descs[0].shape()[1];
  int num = row * col;
  float* src = new float[num];
  float* src_ref = new float[num];
  float* alpha = new float[row];
  float* beta = new float[row];

  // init alpha&beta
  for (int i = 0; i < row; i++) alpha[i] = 1 + rand_float_postfix();
  for (int i = 0; i < row; i++) beta[i] = 1 + rand_float_postfix();

  if (affine) {
    op_attrs["affine"] = "1";
    op_attrs["alpha"] = std::to_string(reinterpret_cast<uint64_t>(alpha));
    op_attrs["beta"] = std::to_string(reinterpret_cast<uint64_t>(beta));
  }

  if (postop_attr.size() != 0) op_attrs["postop_list"] = "fp32_gelu";

  // init matrix.
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      float tmp = 5 + rand_float_postfix();
      src[i * col + j] = tmp;
      src_ref[i * col + j] = tmp;
    }
  }

  std::vector<const void*> rf_data1;
  std::vector<const void*> rf_data2;

  rf_data1.emplace_back((void*)src);
  rf_data2.emplace_back((void*)src_ref);
  rf_data2.push_back(alpha);
  rf_data2.push_back(beta);

  operator_desc layernorm_ba_desc(kernel_kind::layernorm_ba, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                                  op_attrs, postop_attr);

  op_args_t p = {layernorm_ba_desc, rf_data1};
  op_args_t q = {layernorm_ba_desc, rf_data2};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  tensor_desc data_desc0 = {{256, 128}, jd::data_type::fp32, jd::format_type::ba};
  tensor_desc data_desc1 = {{255, 127}, jd::data_type::fp32, jd::format_type::ba};

  postop_attr fp32_gelu_attr = {data_type::fp32, postop_type::eltwise, postop_alg::gelu};

  std::string tensor_shape0 = "256x128";
  std::string tensor_shape1 = "255x127";

  cases.push_back({gen_case({data_desc0}, {{"matrix_shape", tensor_shape0}}), false});
  cases.push_back({gen_case({data_desc0}, {{"matrix_shape", tensor_shape0}}, true), false});

  cases.push_back({gen_case({data_desc0}, {{"matrix_shape", tensor_shape0}}, false, {fp32_gelu_attr}), false});
  cases.push_back({gen_case({data_desc0}, {{"matrix_shape", tensor_shape0}}, true, {fp32_gelu_attr}), false});

  cases.push_back({gen_case({data_desc1}, {{"matrix_shape", tensor_shape1}}), false});
  cases.push_back({gen_case({data_desc1}, {{"matrix_shape", tensor_shape1}}, true), false});

  cases.push_back({gen_case({data_desc1}, {{"matrix_shape", tensor_shape1}}, false, {fp32_gelu_attr}), false});
  cases.push_back({gen_case({data_desc1}, {{"matrix_shape", tensor_shape1}}, true, {fp32_gelu_attr}), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, LayernormBaKernelTest, case_func());
}  // namespace jd
