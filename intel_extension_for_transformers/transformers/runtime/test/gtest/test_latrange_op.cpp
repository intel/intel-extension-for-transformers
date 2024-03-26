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
#include "../../include/operators/latrange.hpp"
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

void GetTrueData(const std::vector<Tensor*>& input,
                 const std::vector<Tensor*>& output,
                 const shared_ptr<OperatorConfig>& conf) {
  auto shape = input[0]->shape();

  // attrs map
  auto attrs_map = conf->attributes();
  int start = executor::StringToNum<int>(attrs_map["start"]);
  int step = executor::StringToNum<int>(attrs_map["step"]);

  // dst shape
  output[0]->set_shape(shape);
  output[0]->set_dtype("int32");

  int* dst_data = static_cast<int*>(output[0]->mutable_data());

  int bs = shape[0];
  int seq = shape.back();

  for (int i = 0; i < bs; ++i) {
    for (int j = 0, k = start; j < seq; ++j, k += step) {
      dst_data[i * seq + j] = k;
    }
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  executor::LatRangeOperator padding_seq_op(p.conf);
  padding_seq_op.Reshape(p.input, p.output);
  padding_seq_op.Forward(p.input, p.output);

  GetTrueData(q.input, q.output, q.conf);
  // Should compare buffer with different addresses
  EXPECT_NE(p.output[0]->data(), q.output[0]->data());
  return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(),
                                      q.output[0]->data(), q.output[0]->size());
}

class LatRangeOpTest : public testing::TestWithParam<TestParams> {
 protected:
  LatRangeOpTest() {}
  ~LatRangeOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(LatRangeOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateInt32Case(
    const std::vector<std::vector<int64_t> >& input_shape,
    std::string attr_start = "0", std::string attr_step = "1") {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  shared_ptr<TensorConfig> src_config = std::make_shared<TensorConfig>("input", src_shape, "int32");
  std::vector<shared_ptr<TensorConfig>> input_config_vec = {src_config};
  std::vector<int64_t> dst_shape = {};
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape);
  std::vector<shared_ptr<TensorConfig>> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map["start"] = attr_start;
  attr_map["step"] = attr_step;
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config = std::make_shared<OperatorConfig>("range", "int32", input_config_vec,
                                            output_config_vec, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const shared_ptr<TensorConfig>& a_tensor_config,
                             int life_num = 1) {
    // step1: set shape
    Tensor* a_tensor = new Tensor(*a_tensor_config);
    // step2: set tensor life
    a_tensor->add_tensor_life(life_num);
    // step3: library buffer can only be obtained afterwards
    auto tensor_data = static_cast<int*>(a_tensor->mutable_data());
    int batch_size = a_tensor->shape()[0];
    int seq_len = a_tensor->shape()[1];
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < seq_len; ++j) {
        tensor_data[i * seq_len + j] = (j < seq_len / 2) ? 1 : 0;
      }
    }

    Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
    a_tensor_copy->add_tensor_life(life_num);
    auto tensor_data_copy = a_tensor_copy->mutable_data();
    memcpy(tensor_data_copy, tensor_data,
           a_tensor_copy->size() * sizeof(float));
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

static auto CasesInt32 = []() {
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;

  // case: simple, range 2D
  src_shape = {16, 32};
  cases.push_back({GenerateInt32Case({src_shape}), false});
  src_shape = {8, 384};
  cases.push_back({GenerateInt32Case({src_shape}), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, LatRangeOpTest, CasesInt32());
