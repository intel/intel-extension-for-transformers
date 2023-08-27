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
  Tensor* src = input[0];
  Tensor* dst = output[0];
  dst->set_shape(src->shape());

  auto attrs_map = conf->attributes();
  auto iter = attrs_map.find("output_dtype");
  string output_dtype = iter == attrs_map.end() ? "fp32": attrs_map["output_dtype"];

  if (output_dtype == "bf16") {
    const float* src_data = static_cast<const float*>(src->data());
    uint16_t* dst_data = static_cast<uint16_t*>(dst->mutable_data());
#pragma omp parallel for
    for (int i = 0; i < src->size(); ++i) {
      union {
        unsigned int u;
        float f;
      } typecast;
      typecast.f = src_data[i];
      dst_data[i] = typecast.u >> 16;
    }
  } else if (output_dtype == "fp32") {
    const uint16_t* src_data = static_cast<const uint16_t*>(src->data());
    float* dst_data = static_cast<float*>(dst->mutable_data());
#pragma omp parallel for
    for (int i = 0; i < src->size(); ++i) {
      union {
        unsigned int u;
        float f;
      } typecast;
      typecast.u = src_data[i] << 16;
      dst_data[i] = typecast.f;
    }
  } else {
    LOG(ERROR) << "Cast only supports fp32/bf16 output dtype!";
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::LLGAINFO llga_info;
    llga_info.InitLTFromTensorConf(p.conf, false);
    executor::LLGAOPCreator::GetInstance().CreateTypeCastOp(&llga_info, p.conf);
    executor::LLGAKernel cast(p.conf, &llga_info);
    cast.Prepare(p.input, p.output);
    cast.Reshape(p.input, p.output);
    cast.Forward(p.input, p.output);
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
    if (p.output[0]->dtype() == "bf16") {
      return executor::CompareData<uint16_t>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size(), 1);
    } else if (p.output[0]->dtype() == "fp32") {
      return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size(), 1e-4);
    }
  }
  return false;
}

class CastTest : public testing::TestWithParam<TestParams> {
 protected:
  CastTest() {}
  ~CastTest() override {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(CastTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

template <typename T>
std::pair<Tensor*, Tensor*> make_tensor_obj(const shared_ptr<TensorConfig>& a_tensor_config) {
  // step1: set shape
  Tensor* a_tensor = new Tensor(*a_tensor_config);
  // step2: set tensor life
  a_tensor->add_tensor_life(1);
  // step3: library buffer can only be obtained afterwards
  auto tensor_data = a_tensor->mutable_data();
  executor::InitVector<T>(static_cast<T*>(tensor_data), a_tensor->size());

  Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
  a_tensor_copy->add_tensor_life(1);
  auto tensor_data_copy = a_tensor_copy->mutable_data();
  memcpy(tensor_data_copy, tensor_data, a_tensor_copy->size() * sizeof(T));
  return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t> >& input_shape,
                                           const std::string& output_dtype = "bf16") {
  // Step 1: Construct Tensor config ptr
  const vector<int64_t> src0_shape = input_shape[0];
  string src_dtype = output_dtype == "bf16" ? "fp32" : "bf16";
  shared_ptr<TensorConfig> src0_config = std::make_shared<TensorConfig>("src0", src0_shape, src_dtype);
  std::vector<int64_t> dst_shape = {};
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape, output_dtype);
  std::vector<shared_ptr<TensorConfig>> inputs_config = {src0_config};
  std::vector<shared_ptr<TensorConfig>> output_config = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"output_dtype", output_dtype}};

  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config = std::make_shared<OperatorConfig>("cast", "Cast",
                                                   inputs_config, output_config, op_attr);

  // Step 2: Construct Tensor ptr
  auto src0_tensors = output_dtype == "bf16" ? make_tensor_obj<float>(src0_config) :\
                      make_tensor_obj<uint16_t>(src0_config);
  auto dst_tensors = output_dtype == "bf16" ? make_tensor_obj<uint16_t>(dst_config) :\
                      make_tensor_obj<float>(dst_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src0_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src0_tensors.second}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesInt8Fp32 = []() {
  MemoryAllocator::InitStrategy();
  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src0_shape;
  std::vector<int64_t> src1_shape;

  // case: 1D fp32 -> bf16
  src0_shape = {1024};
  cases.push_back({GenerateFp32Case({src0_shape}, "bf16"), false});

  // case: 1D bf16 -> fp32
  src0_shape = {1024};
  cases.push_back({GenerateFp32Case({src0_shape}, "fp32"), false});

  // case: 2D fp32 -> bf16
  src0_shape = {16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape}, "bf16"), false});

  // case: 2D bf16 -> fp32
  src0_shape = {16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape}, "fp32"), false});

  // case: 3D fp32 -> bf16
  src0_shape = {4, 16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape}, "bf16"), false});

  // case: 3D bf16 -> fp32
  src0_shape = {4, 16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape}, "fp32"), false});

  // case: 4D fp32 -> bf16
  src0_shape = {2, 4, 16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape}, "bf16"), false});

  // case: 4D bf16 -> fp32
  src0_shape = {2, 4, 16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape}, "fp32"), false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, CastTest, CasesInt8Fp32());
