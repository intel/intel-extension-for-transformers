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
#include "../../include/operators/expand_indices.hpp"
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
  const vector<int64_t>& src_shape = input[0]->shape();
  const vector<int64_t>& tgt_shape = input[1]->shape();
  vector<int64_t> dst_shape = src_shape;

  // attrs map
  auto attrs_map = conf->attributes();
  vector<int64_t> position;
  executor::StringSplit<int64_t>(&position, attrs_map["position"], ",");

  // dst shape
  std::vector<int64_t>::iterator dst_it = dst_shape.begin();
  for (auto pos : position)
    if (pos == -1) {
      dst_shape.push_back(tgt_shape[tgt_shape.size() - 1]);
    } else {
      dst_shape.insert(dst_it + pos, tgt_shape[pos]);
      dst_it = dst_shape.begin();
    }
  output[0]->set_shape(dst_shape);
  output[0]->set_dtype("int32");

  vector<int64_t> dst_stride = executor::GetStrides(dst_shape, {});
  vector<int64_t> src_stride = executor::GetStrides(src_shape, {});

  int32_t* src_data = static_cast<int32_t*>(input[0]->mutable_data());
  int32_t* dst_data = static_cast<int32_t*>(output[0]->mutable_data());
#pragma omp parallel for
  for (int i = 0; i < output[0]->size(); i++) {
    int new_layer = 0, old_layer = 0;
    int old_idx = 0, new_idx = i;
    for (; new_layer < dst_shape.size() && old_layer < src_shape.size(); new_layer++) {
      bool is_expand = false;
      for (int i = 0; i < position.size() && !is_expand; i++) is_expand |= (position[i] == new_layer);
      if (!is_expand) {
        old_idx += new_idx / dst_stride[new_layer] * src_stride[old_layer];
        old_layer++;
      }
      new_idx %= dst_stride[new_layer];
    }
    dst_data[i] = src_data[old_idx];
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  executor::ExpandIndicesOperator op(p.conf);
  op.Reshape(p.input, p.output);
  op.Forward(p.input, p.output);

  GetTrueData(q.input, q.output, q.conf);
  // Should compare buffer with different addresses
  EXPECT_NE(p.output[0]->data(), q.output[0]->data());
  return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                      q.output[0]->size());
}

class RangeOpTest : public testing::TestWithParam<TestParams> {
 protected:
  RangeOpTest() {}
  ~RangeOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(RangeOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateInt32Case(const std::vector<std::vector<int64_t> >& input_shape,
                                            std::string attr_position = "0") {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  shared_ptr<TensorConfig> src_config = std::make_shared<TensorConfig>("input", src_shape, "int32");
  const auto& tgt_shape = input_shape[1];
  shared_ptr<TensorConfig> tgt_config = std::make_shared<TensorConfig>("input", tgt_shape, "int32");
  std::vector<shared_ptr<TensorConfig>> input_config_vec = {src_config, tgt_config};
  std::vector<int64_t> dst_shape = {};
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape);
  std::vector<shared_ptr<TensorConfig>> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map["position"] = attr_position;
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config = std::make_shared<OperatorConfig>("expand_indices", "int32",
                                         input_config_vec, output_config_vec, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const shared_ptr<TensorConfig>& a_tensor_config, int life_num = 1) {
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
    memcpy(tensor_data_copy, tensor_data, a_tensor_copy->size() * sizeof(float));
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };

  auto src_tensors = make_tensor_obj(src_config);
  auto src1_tensors = make_tensor_obj(tgt_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src_tensors.first, src1_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src_tensors.second, src1_tensors.second}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesInt32 = []() {
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> tgt_shape;

  // case: simple, expand middel dim
  src_shape = {200, 300};
  tgt_shape = {200, 1, 1, 300};
  cases.push_back({GenerateInt32Case({src_shape, tgt_shape}, "1,2"), false});

  // case: simple, expand kast dim
  src_shape = {200, 300};
  tgt_shape = {200, 300, 500};
  cases.push_back({GenerateInt32Case({src_shape, tgt_shape}, "-1"), false});

  // case: simple, expand kast dim
  src_shape = {3, 7};
  tgt_shape = {2, 3, 5, 7, 11};
  cases.push_back({GenerateInt32Case({src_shape, tgt_shape}, "0,2,-1"), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, RangeOpTest, CasesInt32());
