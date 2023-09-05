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
#include "../../include/operators/gather_elements.hpp"
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
  const vector<int64_t>& idx_shape = input[1]->shape();
  vector<int64_t> dst_shape = idx_shape;

  output[0]->set_shape(dst_shape);
  output[0]->set_dtype(input[0]->dtype());

  auto attrs_map = conf->attributes();
  int64_t axis = stoi(attrs_map["axis"]);

  vector<int64_t> dst_stride = executor::GetStrides(dst_shape, {});
  vector<int64_t> src_stride = executor::GetStrides(src_shape, {});

  char* src_data = reinterpret_cast<char*>(input[0]->mutable_data());
  int32_t* idx_data = static_cast<int32_t*>(input[1]->mutable_data());
  char* dst_data = reinterpret_cast<char*>(output[0]->mutable_data());
#pragma omp parallel for
  for (int i = 0; i < output[0]->size(); i++) {
    int target = idx_data[i];
    int outer = i / dst_stride[axis - 1];
    int inner = i % dst_stride[axis];
    int old_idx = outer * src_stride[axis - 1] + target * src_stride[axis] + inner;
    memcpy(dst_data + i * executor::type2bytes[input[0]->dtype()],
           src_data + old_idx * executor::type2bytes[input[0]->dtype()], executor::type2bytes[input[0]->dtype()]);
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  executor::GatherElementsOperator op(p.conf);
  op.Reshape(p.input, p.output);
  op.Forward(p.input, p.output);

  GetTrueData(q.input, q.output, q.conf);
  // Should compare buffer with different addresses
  EXPECT_NE(p.output[0]->data(), q.output[0]->data());
  return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                      q.output[0]->size());
}

class GatherElementsOpTest : public testing::TestWithParam<TestParams> {
 protected:
  GatherElementsOpTest() {}
  ~GatherElementsOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(GatherElementsOpTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateCase(const std::vector<std::vector<int64_t> >& input_shape,
                                       std::string attr_position = "0", std::string dtype = "fp32") {
  // Step 1: Construct Tensor config ptr
  const auto& src_shape = input_shape[0];
  shared_ptr<TensorConfig> src_config = std::make_shared<TensorConfig>("input", src_shape, dtype);
  const auto& idx_shape = input_shape[1];
  shared_ptr<TensorConfig> idx_config = std::make_shared<TensorConfig>("index", idx_shape, "int32");
  std::vector<shared_ptr<TensorConfig>> input_config_vec = {src_config, idx_config};
  std::vector<int64_t> dst_shape = {};
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape);
  std::vector<shared_ptr<TensorConfig>> output_config_vec = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map["axis"] = attr_position;
  int axis = stoi(attr_position);
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config = std::make_shared<OperatorConfig>("gather_elements",
                                         dtype, input_config_vec, output_config_vec, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const shared_ptr<TensorConfig>& a_tensor_config) {
    // step1: set shape
    Tensor* a_tensor = new Tensor(*a_tensor_config);
    // step2: set tensor life
    a_tensor->add_tensor_life(1);
    // step3: library buffer can only be obtained afterwards
    auto tensor_data = a_tensor->mutable_data();
    executor::InitVector<float>(static_cast<float*>(tensor_data),
                         a_tensor->dtype() == "fp32" ? a_tensor->size() : a_tensor->size() / 4);

    Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
    a_tensor_copy->add_tensor_life(1);
    auto tensor_data_copy = a_tensor_copy->mutable_data();
    memcpy(tensor_data_copy, tensor_data, a_tensor_copy->size() * executor::type2bytes[a_tensor->dtype()]);
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };

  auto src_tensors = make_tensor_obj(src_config);
  Tensor* idx_tensor = new Tensor(*idx_config);
  Tensor* idx_tensor_copy = new Tensor(*idx_config);
  idx_tensor->add_tensor_life(1);
  idx_tensor_copy->add_tensor_life(1);
  int32_t* idx_data = reinterpret_cast<int32_t*>(idx_tensor->mutable_data());
  int outer = 1, inner = 1;
  for (int i = 0; i < axis; i++) outer *= src_shape[i];
  for (int i = axis + 1; i < src_shape.size(); i++) inner *= src_shape[i];
  int idx = 0;
  uint32_t seed = 123;
  std::srand(seed);
  for (int i = 0; i < outer; i++) {
    for (int j = 0; j < idx_shape[axis]; j++) {
      int random_idx = std::rand() % src_shape[axis];
      for (int k = 0; k < inner; k++) idx_data[idx++] = random_idx;
    }
  }
  void* idx_data_copy = idx_tensor_copy->mutable_data();
  memcpy(idx_data_copy, idx_data, idx_tensor->size() * 4);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src_tensors.first, idx_tensor}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src_tensors.second, idx_tensor_copy}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesInt32 = []() {
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> idx_shape;

  // case: simple, expand middel dim
  src_shape = {1, 384, 384};
  idx_shape = {1, 339, 384};
  cases.push_back({GenerateCase({src_shape, idx_shape}, "1", "fp32"), false});

  src_shape = {1, 12, 384, 384};
  idx_shape = {1, 12, 384, 339};
  cases.push_back({GenerateCase({src_shape, idx_shape}, "3", "fp32"), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, GatherElementsOpTest, CasesInt32());
