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
#include "../../include/operators/split.hpp"
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

  std::vector<int64_t> split;
  iter = attrs_map.find("split");
  if (iter != attrs_map.end()) {
    executor::StringSplit<int64_t>(&split, attrs_map["split"], ",");
  }
  auto src_tensor_shape = input[0]->shape();
  vector<int64_t> src_stride = executor::GetStrides(src_tensor_shape);
  vector<int64_t> dst_shape;
  for (int n = 0; n < src_stride.size(); ++n) {
    if (n != axis) {
      dst_shape.emplace_back(src_tensor_shape[n]);
    } else {
      dst_shape.emplace_back(split[n]);
    }
  }

  // dst shape
  const auto src_tensor_data = static_cast<const float*>(input[0]->data());
  std::vector<float*> dst_data;
  for (auto tensor : output) {
    tensor->set_shape(dst_shape);
    dst_data.push_back(static_cast<float*>(tensor->mutable_data()));
  }
  int h = dst_shape[0];
  int w = dst_shape[1];
  for (int index = 0; index < dst_data.size(); index++)
    if (axis == 0) {
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          dst_data[index][i * w + j] = src_tensor_data[(i + h * index) * src_tensor_shape[1] + j];
        }
      }
    } else if (axis == 1) {
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          dst_data[index][i * w + j] = src_tensor_data[i * src_tensor_shape[1] + j + w * index];
        }
      }
    }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  executor::SplitOperator split(p.conf);
  split.Reshape(p.input, p.output);
  split.Forward(p.input, p.output);

  if (!t.expect_to_fail) {
    GetTrueData(q.input, q.output, q.conf);
    // Should compare buffer with different addresses
    bool result = true;
    for (int i = 0; i < p.output.size(); i++) {
      EXPECT_NE(p.output[i]->data(), q.output[i]->data());
      if (executor::CompareData<float>(p.output[i]->data(), p.output[i]->size(), q.output[i]->data(),
                                       q.output[i]->size()) == false) {
        return false;
      }
    }
    return true;
  }
  return false;
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

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape, std::string axis,
                                           std::string split) {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  shared_ptr<TensorConfig> src_config = std::make_shared<TensorConfig>("src", src_shape);
  std::vector<shared_ptr<TensorConfig>> input_config = {src_config};
  std::vector<int64_t> dst_shape = {};
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape);
  std::vector<shared_ptr<TensorConfig>> output_config = {dst_config, dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"axis", axis}, {"split", split}};

  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  auto op_config = std::make_shared<OperatorConfig>("split", "fp32", input_config, output_config, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const shared_ptr<TensorConfig>& a_tensor_config) {
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
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };

  auto src_tensors = make_tensor_obj(src_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor1 = new Tensor(*dst_config);
  dst_tensor1->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);
  Tensor* dst_tensor_copy1 = new Tensor(*dst_config);
  dst_tensor_copy1->add_tensor_life(1);

  OpArgs op_args = {{src_tensors.first, src_tensors.first}, {dst_tensor, dst_tensor1}, op_config};
  OpArgs op_args_copy = {{src_tensors.second, src_tensors.second}, {dst_tensor_copy, dst_tensor_copy1}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  std::string memory_strategy = getenv("DIRECT_BUFFER") == NULL ? "cycle_buffer" : "direct_buffer";
  MemoryAllocator::SetStrategy(memory_strategy);
  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;

  // case: simple for axis=0
  cases.push_back({GenerateFp32Case({{4, 32}}, "0", "2, 2"), false});
  cases.push_back({GenerateFp32Case({{1536, 1024}}, "0", "768, 768"), false});
  cases.push_back({GenerateFp32Case({{2304, 1024}}, "0", "768, 768, 768"), false});
  // case: simple for axis=1
  cases.push_back({GenerateFp32Case({{32, 4}}, "1", "2, 2"), false});
  cases.push_back({GenerateFp32Case({{1024, 1536}}, "1", "768, 768"), false});
  cases.push_back({GenerateFp32Case({{1024, 2304}}, "1", "768, 768, 768"), false});
  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, ConcatTest, CasesFp32());
