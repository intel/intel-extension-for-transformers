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

#include <map>
#include <string>

#include "../../include/common.hpp"
#include "../../include/conf.hpp"
#include "../../include/operators/rmsnorm.hpp"
#include "gtest/gtest.h"

using executor::AttrConfig;
using executor::MemoryAllocator;
using executor::OperatorConfig;
using executor::Tensor;
using executor::TensorConfig;

struct OpArgs {
  std::vector<Tensor*> input;
  std::vector<Tensor*> output;
  shared_ptr<OperatorConfig> conf;
};

struct TestParams {
  std::pair<OpArgs, OpArgs> args;
  bool expect_to_fail;
};

void RmsNormRefGT(const float* src_data, const float* gamma_data, float* dst_data, const vector<int64_t>& src_shape,
                  const float eps) {
  auto batchs = std::accumulate(src_shape.begin(), src_shape.end() - 1, 1, std::multiplies<int>());
  auto norm_dim = *(src_shape.end() - 1);
#pragma omp parallel for
  for (int i = 0; i < batchs; i++) {
    float powx_sum = 0.f;
    auto src = src_data + i * norm_dim;
    auto dst = dst_data + i * norm_dim;
    for (int j = 0; j < norm_dim; j++) powx_sum += pow(src[j], 2);
    powx_sum += eps;
    auto scale = 1.f / sqrt(powx_sum / norm_dim);
    for (int j = 0; j < norm_dim; j++) dst[j] = src[j] * gamma_data[j] * scale;
  }
}

void GetTrueData(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output,
                 const shared_ptr<OperatorConfig>& conf) {
  // config parse
  float epsilon = 1e-05;
  bool affine = false;
  auto attrs_map = conf->attributes();
  auto iter = attrs_map.find("epsilon");
  if (iter != attrs_map.end()) {
    epsilon = executor::StringToNum<float>(attrs_map["epsilon"]);
  }
  Tensor* src = input[0];
  Tensor* gamma = input[1];
  Tensor* dst = output[0];
  const vector<int64_t> src_shape = src->shape();
  dst->set_dtype(src->dtype());
  dst->set_shape(src_shape);
  assert(src->dtype() == "fp32");
  assert(src_shape.size() > 1);
  const float* src_data = static_cast<const float*>(src->data());
  const float* gamma_data = static_cast<const float*>(gamma->data());
  float* dst_data = static_cast<float*>(dst->mutable_data());
  RmsNormRefGT(src_data, gamma_data, dst_data, src_shape, epsilon);
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  executor::RmsNormOperator rms_norm(p.conf);
  rms_norm.Prepare(p.input, p.output);
  rms_norm.Reshape(p.input, p.output);
  rms_norm.Forward(p.input, p.output);

  if (!t.expect_to_fail) {
    GetTrueData(q.input, q.output, q.conf);
    // Should compare buffer with different addresses
    EXPECT_NE(p.output[0]->data(), q.output[0]->data());
    return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size(), 1e-3);
  }
  return true;
}

class RmsNormTest : public testing::TestWithParam<TestParams> {
 protected:
  RmsNormTest() {}
  ~RmsNormTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(RmsNormTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape,
                                           const std::string& epsilon) {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  const auto& gamma_shape = input_shape[1];
  shared_ptr<TensorConfig> src_config = std::make_shared<TensorConfig>("src", src_shape);
  shared_ptr<TensorConfig> gamma_config = std::make_shared<TensorConfig>("gamma", gamma_shape);
  std::vector<shared_ptr<TensorConfig>> input_config_vec = {src_config, gamma_config};
  std::vector<int64_t> dst_shape = {};
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape);
  std::vector<shared_ptr<TensorConfig>> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"epsilon", epsilon}};
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config =
      std::make_shared<OperatorConfig>("rms_norm", "fp32", input_config_vec, output_config_vec, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const shared_ptr<TensorConfig>& a_tensor_config, int life_num = 1) {
    // step1: set shape
    Tensor* a_tensor = new Tensor(*a_tensor_config);
    // step2: set tensor life
    a_tensor->add_tensor_life(life_num);
    // step3: library buffer can only be obtained afterwards
    auto tensor_data = a_tensor->mutable_data();
    executor::InitVector<float>(static_cast<float*>(tensor_data), a_tensor->size());

    Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
    a_tensor_copy->add_tensor_life(life_num);
    auto tensor_data_copy = a_tensor_copy->mutable_data();
    memcpy(tensor_data_copy, tensor_data, a_tensor_copy->size() * sizeof(float));
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };

  auto src_tensors = make_tensor_obj(src_config, 2);
  auto gamma_tensors = make_tensor_obj(gamma_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src_tensors.first, gamma_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src_tensors.second, gamma_tensors.second}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  std::string memory_strategy = getenv("DIRECT_BUFFER") == NULL ? "cycle_buffer" : "direct_buffer";
  MemoryAllocator::SetStrategy(memory_strategy);
  std::vector<TestParams> cases;
  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> gamma_shape;
  std::string epsilon;

  // case: 4*4096
  src_shape = {4, 4096};
  gamma_shape = {4096};
  epsilon = "0.00001";
  cases.push_back({GenerateFp32Case({src_shape, gamma_shape}, epsilon), false});

  // case: 32*4096
  src_shape = {32, 4096};
  gamma_shape = {4096};
  epsilon = "0.00001";
  cases.push_back({GenerateFp32Case({src_shape, gamma_shape}, epsilon), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, RmsNormTest, CasesFp32());
