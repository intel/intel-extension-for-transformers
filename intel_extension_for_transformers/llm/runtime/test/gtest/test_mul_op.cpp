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
#include "llga_kernel.hpp"
#include "llga_op_creator.hpp"
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
  // set output shape
  Tensor* input0 = input[0]->size() >= input[1]->size() ? input[0] : input[1];
  Tensor* input1 = input[1]->size() <= input[0]->size() ? input[1] : input[0];
  Tensor* out = output[0];
  const vector<int64_t>& shape0 = input0->shape();
  const vector<int64_t>& shape1 = input1->shape();
  out->set_shape(shape0);
  // get dst data
  const float* src0_data = static_cast<const float*>(input0->data());
  const float* src1_data = static_cast<const float*>(input1->data());
  float* dst_data = static_cast<float*>(out->mutable_data());
  const int64_t batch_size = input1->size() == 1 ? input0->size() : \
             accumulate(shape0.begin(), shape0.end()-shape1.size(), 1, std::multiplies<int64_t>());
  const int64_t channel = input1->size();
#pragma omp parallel for
  for (int i = 0; i < batch_size; ++i) {
#pragma omp simd
    for (int j = 0; j < channel; ++j) {
      dst_data[i*channel+j] = src0_data[i*channel+j] * src1_data[j];
    }
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::LLGAINFO llga_info;
    llga_info.InitLTFromTensorConf(p.conf, false);
    executor::LLGAOPCreator::GetInstance().CreateMultiplyOp(&llga_info, p.conf);
    executor::LLGAKernel mul(p.conf, &llga_info);
    mul.Prepare(p.input, p.output);
    mul.Reshape(p.input, p.output);
    mul.Forward(p.input, p.output);
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
    return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size(), eps);
  }
  return false;
}

class MulTest : public testing::TestWithParam<TestParams> {
 protected:
  MulTest() {}
  ~MulTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MulTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape) {
  // Step 1: Construct Tensor config ptr
  const auto& src0_shape = input_shape[0];
  shared_ptr<TensorConfig> src0_config = std::make_shared<TensorConfig>("src0", src0_shape);
  const auto& src1_shape = input_shape[1];
  shared_ptr<TensorConfig> src1_config = std::make_shared<TensorConfig>("src1", src1_shape);
  std::vector<shared_ptr<TensorConfig>> input_config = {src0_config, src1_config};
  std::vector<int64_t> dst_shape = {};
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape);
  std::vector<shared_ptr<TensorConfig>> output_config = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config = std::make_shared<OperatorConfig>("mul", "Mul",
                                         input_config, output_config, op_attr);

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

  auto src0_tensors = make_tensor_obj(src0_config);
  auto src1_tensors = make_tensor_obj(src1_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src0_tensors.first, src1_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src0_tensors.second, src1_tensors.second}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  std::string memory_strategy = getenv("DIRECT_BUFFER") == NULL ? "cycle_buffer" : "direct_buffer";
  MemoryAllocator::SetStrategy(memory_strategy);
  std::vector<TestParams> cases;

  std::vector<int64_t> src0_shape;
  std::vector<int64_t> src1_shape;
  // case: 2D x 3D
  src0_shape = {16, 1024};
  src1_shape = {2, 16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape})});

  // case: 3D x 2D
  src0_shape = {2, 16, 1024};
  src1_shape = {16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape})});

  // case: 2D x 2D
  src0_shape = {16, 1024};
  src1_shape = {16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape})});

  // case: 2D x 1D
  src0_shape = {16, 1024};
  src1_shape = {1024};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape})});

  // case: 2D x scalar
  src0_shape = {16, 1024};
  src1_shape = {1};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape})});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, MulTest, CasesFp32());
