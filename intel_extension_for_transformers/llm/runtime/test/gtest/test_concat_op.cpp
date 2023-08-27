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
#include "../../include/operators/concat.hpp"
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
  auto attrs_map = conf->attributes();
  int axis = 0;
  auto iter = attrs_map.find("axis");
  if (iter != attrs_map.end() && iter->second != "") {
    axis = stoi(iter->second);
  }
  const int num_src = input.size();
  auto src_tensor_shape = input[0]->shape();
  vector<int64_t> dst_shape;
  for (int n = 0; n < src_tensor_shape.size(); ++n) {
    if (n != axis) {
      dst_shape.emplace_back(src_tensor_shape[n]);
    } else {
      int concat_dim_sum = 0;
      for (int i = 0; i < num_src; ++i) {
        concat_dim_sum += input[i]->shape()[axis];
      }
      dst_shape.emplace_back(concat_dim_sum);
    }
  }

  // dst shape
  output[0]->set_shape(dst_shape);
  float* dst_data = static_cast<float*>(output[0]->mutable_data());
  std::vector<const float*> src_data;
  for (const auto& pTensor : input) {
    src_data.emplace_back(static_cast<const float*>(pTensor->data()));
  }
  int size_before_concat_dim =
      std::accumulate(src_tensor_shape.begin(), src_tensor_shape.begin() + axis, 1, std::multiplies<int>());
  int size_after_concat_dim =
      std::accumulate(src_tensor_shape.begin() + axis + 1, src_tensor_shape.end(), 1, std::multiplies<int>());
  for (int i = 0; i < size_before_concat_dim; ++i) {
    float* dst_addr = dst_data + i * dst_shape[axis] * size_after_concat_dim;
    for (int n = 0; n < num_src; ++n) {
      int concat_data_size = input[n]->shape()[axis] * size_after_concat_dim;
      const float* src_addr = src_data[n] + i * concat_data_size;
      memcpy(dst_addr, src_addr, concat_data_size * sizeof(float));
      dst_addr += concat_data_size;
    }
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::ConcatOperator concat(p.conf);
    concat.Reshape(p.input, p.output);
    concat.Forward(p.input, p.output);
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return t.expect_to_fail;
  }

  if (!t.expect_to_fail) {
    GetTrueData(q.input, q.output, q.conf);
    // Should compare buffer with different addresses
    EXPECT_NE(p.output[0]->data(), q.output[0]->data());
    return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size());
  }
  return true;
}

class ConcatTest : public testing::TestWithParam<TestParams> {
 protected:
  ConcatTest() {}
  ~ConcatTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(ConcatTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape, std::string axis) {
  // Step 0: Make sure input tensors have the same shape except for size of concat_dim
  int axis_ = stoi(axis);
  for (int i = 1; i < input_shape.size(); ++i) {
    EXPECT_TRUE(input_shape[i].size() == input_shape[0].size()) << "input tensors have different ndims";
    for (int j = 0; j < input_shape[0].size(); ++j) {
      if (j == axis_) continue;
      EXPECT_TRUE(input_shape[i][j] == input_shape[0][j]) << "input tensors have different shapes at non concat_dim";
    }
  }

  // Step 1: Construct Tensor config ptr
  std::vector<shared_ptr<TensorConfig>> input_config(0);
  for (const auto& src_shape : input_shape) {
    input_config.emplace_back(std::make_shared<TensorConfig>("src", src_shape));
  }
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", std::vector<int64_t>(0));
  std::vector<shared_ptr<TensorConfig>> output_config{dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"axis", axis}};

  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config =
      std::make_shared<OperatorConfig>("concat", "fp32", input_config, output_config, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const std::vector<shared_ptr<TensorConfig>>& tensor_config) {
    std::pair<std::vector<Tensor*>, std::vector<Tensor*>> res;
    for (const auto a_tensor_config : tensor_config) {
      // step1: set shape
      Tensor* a_tensor = new Tensor(*a_tensor_config);
      // step2: set tensor life
      a_tensor->add_tensor_life(1);
      // step3: library buffer can only be obtained afterwards
      auto tensor_data = a_tensor->mutable_data();
      executor::InitVector<float>(static_cast<float*>(tensor_data), a_tensor->size());

      Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
      a_tensor_copy->add_tensor_life(1);
      auto tensor_data_copy = a_tensor_copy->mutable_data();
      memcpy(reinterpret_cast<void*>(tensor_data_copy), tensor_data, a_tensor_copy->size() * sizeof(float));

      res.first.emplace_back(a_tensor);
      res.second.emplace_back(a_tensor_copy);
    }
    return res;
  };

  auto src_tensors = make_tensor_obj(input_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {src_tensors.first, {dst_tensor}, op_config};
  OpArgs op_args_copy = {src_tensors.second, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  std::string memory_strategy = getenv("DIRECT_BUFFER") == NULL ? "cycle_buffer" : "direct_buffer";
  MemoryAllocator::SetStrategy(memory_strategy);
  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> src1_shape;

#ifndef __AVX512__
  constexpr bool expect_fail = true;  // AVX2 not impelemetned
#else
  constexpr bool expect_fail = false;  // AVX2 not impelemetned
#endif

  // case: simple for 0 axis
  src_shape = {1, 1};
  cases.push_back({GenerateFp32Case({src_shape, src_shape}, "0"), expect_fail});
  cases.push_back({GenerateFp32Case({src_shape, src_shape}, "1"), expect_fail});
  src_shape = {2, 3};
  cases.push_back({GenerateFp32Case({src_shape, src_shape}, "0"), expect_fail});
  cases.push_back({GenerateFp32Case({src_shape, src_shape}, "1"), expect_fail});
  src_shape = {100, 30};
  cases.push_back({GenerateFp32Case({src_shape, src_shape}, "0"), expect_fail});
  cases.push_back({GenerateFp32Case({src_shape, src_shape}, "1"), expect_fail});

  src_shape = {64, 1, 768};
  src1_shape = {64, 196, 768};
  cases.push_back({GenerateFp32Case({src_shape, src1_shape}, "1"), expect_fail});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, ConcatTest, CasesFp32());
