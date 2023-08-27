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
#include "../../include/operators/gather.hpp"
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

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  executor::GatherOperator gather_op(p.conf);
  gather_op.Prepare(p.input, p.output);
  gather_op.Reshape(p.input, p.output);
  gather_op.Forward(p.input, p.output);

  // Should compare buffer with different addresses
  EXPECT_NE(p.output[0]->data(), q.output[0]->data());
  return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                      q.output[0]->size());
}

class GatherOpTest : public testing::TestWithParam<TestParams> {
 protected:
  GatherOpTest() {}
  ~GatherOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(GatherOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape,
                                           std::string append_op = "") {
  // Step 1: Construct Tensor config ptr
  const auto& src0_shape = input_shape[0];
  const auto& src1_shape = input_shape[1];
  shared_ptr<TensorConfig> src0_config = std::make_shared<TensorConfig>("src0", src0_shape, "int32");
  shared_ptr<TensorConfig> src1_config = std::make_shared<TensorConfig>("src1", src1_shape);
  std::vector<shared_ptr<TensorConfig>> input_config_vec = {src0_config, src1_config};
  std::vector<int64_t> dst_shape = {src0_shape[0], src1_shape[1]};
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape);
  std::vector<shared_ptr<TensorConfig>> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map["axis"] = "0";
  attr_map["batch_dims"] = "0";
  if (append_op == "binary_add") {
    attr_map["append_op"] = "binary_add";
    shared_ptr<TensorConfig> append_config = std::make_shared<TensorConfig>("appned", input_shape[2]);
    input_config_vec.push_back(append_config);
  }
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config = std::make_shared<OperatorConfig>("gather", "fp32",
                                         input_config_vec, output_config_vec, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const shared_ptr<TensorConfig>& a_tensor_config, int life_num = 1) {
    // step1: set shape
    Tensor* a_tensor = new Tensor(*a_tensor_config);
    // step2: set tensor life
    a_tensor->add_tensor_life(life_num);
    // step3: library buffer can only be obtained afterwards
    auto tensor_data = static_cast<float*>(a_tensor->mutable_data());
    if (a_tensor->shape().size() == 2) {
      executor::InitVector<float>(static_cast<float*>(tensor_data), a_tensor->size());
    } else if (a_tensor->shape().size() == 1) {
      uint32_t seed = 123;
      for (int i = 0; i < a_tensor->size(); ++i) {
        int32_t index = (int32_t)(std::rand() % 30522);
        memcpy((tensor_data + i), &index, sizeof(int32_t));
      }
    }

    Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
    a_tensor_copy->add_tensor_life(life_num);
    auto tensor_data_copy = a_tensor_copy->mutable_data();
    memcpy(tensor_data_copy, tensor_data, a_tensor_copy->size() * sizeof(float));
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };
  std::vector<Tensor*> inputs, inputs_copy;
  for (auto tensor_config : input_config_vec) {
    auto tmp = make_tensor_obj(tensor_config);
    inputs.push_back(tmp.first);
    inputs_copy.push_back(tmp.second);
  }
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);
  auto dst_data_copy = static_cast<float*>(dst_tensor_copy->mutable_data());
  auto src0_data_copy = (const int32_t*)inputs_copy[0]->data();
  auto src1_data_copy = (const float*)inputs_copy[1]->data();
  auto src0_shape_copy = inputs_copy[0]->shape();
  auto src1_shape_copy = inputs_copy[1]->shape();
#pragma omp parallel for
  for (int i = 0; i < src0_shape_copy[0]; ++i) {
    int indices_val = src0_data_copy[i];
// copy slices
#pragma omp simd
    for (int j = 0; j < src1_shape_copy[1]; ++j) {
      dst_data_copy[i * src1_shape_copy[1] + j] = src1_data_copy[indices_val * src1_shape_copy[1] + j];
      if (append_op == "binary_add") {
        auto append_data_copy = (const float*)inputs_copy[2]->data();
        dst_data_copy[i * src1_shape_copy[1] + j] += append_data_copy[i * src1_shape_copy[1] + j];
      }
    }
  }

  OpArgs op_args = {inputs, {dst_tensor}, op_config};
  OpArgs op_args_copy = {inputs_copy, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src0_shape;
  std::vector<int64_t> src1_shape;
  std::vector<int64_t> append_shape;

  // case: simple
  src0_shape = {24576};
  src1_shape = {30522, 1024, 1, 1};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}), false});

  src0_shape = {256};
  src1_shape = {30522, 1024, 1, 1};
  append_shape = {256, 1024};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape, append_shape}, "binary_add"), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, GatherOpTest, CasesFp32());
