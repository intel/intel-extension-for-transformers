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
#include "../../include/operators/squeeze.hpp"
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

void GetTrueData(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output,
                 const shared_ptr<OperatorConfig>& conf) {
  // just copy data
  const float* src_data = static_cast<const float*>(input[0]->data());
  float* dst_data = static_cast<float*>(output[0]->mutable_data());
  auto size = input[0]->size();
#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    dst_data[i] = src_data[i];
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::SqueezeOperator squeeze(p.conf);
    squeeze.Prepare(p.input, p.output);
    squeeze.Reshape(p.input, p.output);
    squeeze.Forward(p.input, p.output);
  } catch (const dnnl::error& e) {
    if (e.status != dnnl_status_t::dnnl_success && t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    GetTrueData(q.input, q.output, q.conf);
    // Should compare buffer with different addresses
    EXPECT_NE(p.output[0]->data(), q.output[0]->data());
    float eps = 1e-4;
    bool data_cmp = executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size(), eps);
    bool shape_cmp = executor::CompareShape(p.output[0]->shape(), q.output[0]->shape());
    return data_cmp && shape_cmp;
  }
  return false;
}

class SqueezeTest : public testing::TestWithParam<TestParams> {
 protected:
  SqueezeTest() {}
  ~SqueezeTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SqueezeTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape,
                                           std::string axes = "") {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  shared_ptr<TensorConfig> src_config = std::make_shared<TensorConfig>("src", src_shape);
  std::vector<shared_ptr<TensorConfig>> input_config = {src_config};
  std::vector<int64_t> dst_shape = input_shape[1];
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape);
  std::vector<shared_ptr<TensorConfig>> output_config = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  if (axes != "")
    attr_map = {{"axes", axes}};
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  auto op_config = std::make_shared<OperatorConfig>("squeeze", "Squeeze", input_config, output_config, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const shared_ptr<TensorConfig>& a_tensor_config) {
    // step1: set shape
    Tensor* a_tensor = new Tensor(*a_tensor_config);
    // step2: set tensor life
    a_tensor->add_tensor_life(1);
    // step3: library buffer can only be obtained afterwards
    auto tensor_data = a_tensor->mutable_data();
    executor::InitVector(static_cast<float*>(tensor_data), a_tensor->size());

    Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
    a_tensor_copy->add_tensor_life(1);
    auto tensor_data_copy = a_tensor_copy->mutable_data();
    memcpy(reinterpret_cast<void*>(tensor_data_copy), tensor_data, a_tensor_copy->size() * sizeof(float));
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };

  auto src_tensors = make_tensor_obj(src_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src_tensors.second}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  std::string memory_strategy = getenv("DIRECT_BUFFER") == NULL ? "cycle_buffer" : "direct_buffer";
  MemoryAllocator::SetStrategy(memory_strategy);
  std::vector<TestParams> cases;

  std::vector<int64_t> src_shape;
  // case: 1D
  src_shape = {1, 3, 1, 5};
  cases.push_back({GenerateFp32Case({src_shape, {3, 1, 5}}, "0")});

  cases.push_back({GenerateFp32Case({src_shape, {1, 3, 5}}, "-2")});

  cases.push_back({GenerateFp32Case({src_shape, {3, 5}})});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, SqueezeTest, CasesFp32());
