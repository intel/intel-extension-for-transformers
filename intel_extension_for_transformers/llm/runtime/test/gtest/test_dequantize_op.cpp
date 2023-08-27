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
#include "../../include/operators/dequantize.hpp"
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

template<typename T>
void Dequant(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output,
             const shared_ptr<OperatorConfig>& conf) {
  auto attrs_map = conf->attributes();
  auto iter = attrs_map.find("axis");
  int64_t axis = (iter != attrs_map.end() && iter->second != "") ? std::stoi(iter->second) : 1;
  Tensor* src = input[0];
  Tensor* scales = input[1];
  Tensor* dst = output[0];
  dst->set_shape(src->shape());

  const float* scales_data = static_cast<const float*>(scales->data());
  float* dst_data = static_cast<float*>(dst->mutable_data());
  const T* src_data = static_cast<const T*>(src->data());
  if (input.size() > 2) {  // has zps
    Tensor* zps = input[2];
    const T* zps_data = static_cast<const T*>(zps->data());
    if (scales->size() == 1) {
#pragma omp parallel for
      for (int i = 0; i < src->size(); ++i) {
        dst_data[i] = (src_data[i] - zps_data[0]) * scales_data[0];
      }
    } else {
      const int64_t output_channel = accumulate(src->shape().begin(),
                               src->shape().begin()+axis, 1, std::multiplies<int64_t>());
      const int64_t axis_size = src->shape()[axis];
      const int64_t input_channel = accumulate(src->shape().begin()+axis+1,
                               src->shape().end(), 1, std::multiplies<int64_t>());
#pragma omp parallel for
      for (int i = 0; i < output_channel; ++i) {
        for (int j = 0; j < axis_size; ++j) {
          for (int k = 0; k < input_channel; ++k) {
            const int src_idx = i * axis_size * input_channel + j * input_channel + k;
            dst_data[src_idx] = (src_data[src_idx] - zps_data[j])* scales_data[j];
          }
        }
      }
    }
  } else {  // no zps
    if (scales->size() == 1) {
#pragma omp parallel for
      for (int i = 0; i < src->size(); ++i) {
        dst_data[i] = src_data[i] * scales_data[0];
      }
    } else {
      const int64_t output_channel = accumulate(src->shape().begin(),
                           src->shape().begin()+axis, 1, std::multiplies<int64_t>());
      const int64_t axis_size = src->shape()[axis];
      const int64_t input_channel = accumulate(src->shape().begin()+axis+1,
                           src->shape().end(), 1, std::multiplies<int64_t>());
#pragma omp parallel for
      for (int i = 0; i < output_channel; ++i) {
        for (int j = 0; j < axis_size; ++j) {
          for (int k = 0; k < input_channel; ++k) {
            const int src_idx = i * axis_size * input_channel + j * input_channel + k;
            dst_data[src_idx] = src_data[src_idx] * scales_data[j];
          }
        }
      }
    }
  }
}

void GetTrueData(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output,
                 const shared_ptr<OperatorConfig>& conf) {
  Tensor* src = input[0];
  if (src->dtype() == "u8") {
    Dequant<uint8_t>(input, output, conf);
  } else if (src->dtype() == "s8") {
    Dequant<int8_t>(input, output, conf);
  } else {
    LOG(ERROR) << "Dequantize only supports u8/s8 input dtype!";
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::DequantizeLinearOperator dequantize(p.conf);
    dequantize.Prepare(p.input, p.output);
    dequantize.Reshape(p.input, p.output);
    dequantize.Forward(p.input, p.output);
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
    return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size(), 1e-4);
  }
  return false;
}

bool CheckLLGAResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::LLGAINFO llga_info;
    vector<executor::Tensor*> inputs = p.input;
    llga_info.SetTensors(&inputs);
    map<string, int> tensor_map = {{"src1", 1}, {"src2", 2}};
    llga_info.SetTensorNameIndex(&tensor_map);
    llga_info.InitLTFromTensorConf(p.conf, false);
    executor::LLGAOPCreator::GetInstance().CreateDequantizeOp(&llga_info, p.conf);
    executor::LLGAKernel dequantize(p.conf, &llga_info);
    dequantize.Prepare(p.input, p.output);
    dequantize.Reshape(p.input, p.output);
    dequantize.Forward(p.input, p.output);
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
    return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size(), 1e-4);
  }
  return false;
}

class DequantizeTest : public testing::TestWithParam<TestParams> {
 protected:
  DequantizeTest() {}
  ~DequantizeTest() override {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(DequantizeTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

class DequantizeTestLLGA : public testing::TestWithParam<TestParams> {
 protected:
  DequantizeTestLLGA() {}
  ~DequantizeTestLLGA() override {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(DequantizeTestLLGA, TestPostfixLLGA) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckLLGAResult(t));
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
                                           const std::string& axis = "1", const std::string& src_dtype = "u8") {
  // Step 1: Construct Tensor config ptr
  const auto& src0_shape = input_shape[0];
  const auto& src1_shape = input_shape[1];
  shared_ptr<TensorConfig> src0_config = std::make_shared<TensorConfig>("src0", src0_shape, src_dtype);
  shared_ptr<TensorConfig> src1_config = std::make_shared<TensorConfig>("src1", src1_shape);
  std::vector<int64_t> dst_shape = {};
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape);
  std::vector<shared_ptr<TensorConfig>> inputs_config = {src0_config, src1_config};
  std::vector<shared_ptr<TensorConfig>> output_config = {dst_config};
  if (input_shape.size() > 2) {
    const auto& src2_shape = input_shape[2];
    shared_ptr<TensorConfig> src2_config = std::make_shared<TensorConfig>("src2", src2_shape, src_dtype);
    inputs_config.push_back(src2_config);
  }
  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"axis", axis}};

  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config = std::make_shared<OperatorConfig>("dequantize",
                           "DequantizeLinear", inputs_config, output_config, op_attr);

  // Step 2: Construct Tensor ptr
  auto src0_tensors = src_dtype == "u8" ? make_tensor_obj<uint8_t>(src0_config) :\
                                          make_tensor_obj<int8_t>(src0_config);
  auto src1_tensors = make_tensor_obj<float>(src1_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src0_tensors.first, src1_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src0_tensors.second, src1_tensors.second}, {dst_tensor_copy}, op_config};

  if (input_shape.size() > 2) {
    auto src2_tensors = src_dtype == "u8" ? make_tensor_obj<uint8_t>(src0_config) :\
                                            make_tensor_obj<int8_t>(src0_config);
    op_args.input.push_back(src2_tensors.first);
    op_args_copy.input.push_back(src2_tensors.second);
  }

  return {op_args, op_args_copy};
}

static auto CasesInt8Fp32LLGA = []() {
  MemoryAllocator::InitStrategy();
  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src0_shape;
  std::vector<int64_t> src1_shape;
  std::vector<int64_t> src2_shape;

  // case: 2D u8 -> fp32(per tensor)
  src0_shape = {16, 1024};
  src1_shape = {1};
  src2_shape = {1};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape, src2_shape}, "1", "u8"), false});

  // case: 2D s8 -> fp32(per tensor)
  src0_shape = {16, 1024};
  src1_shape = {1};
  src2_shape = {1};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape, src2_shape}, "1", "s8"), false});

  // Note: llga only supports one element tensor
  // case: 2D u8 -> fp32(per channel, no zp)
  src0_shape = {2, 16};
  src1_shape = {16};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "1", "u8"), false});

  // case: 2D u8 -> fp32(per channel, no zp)
  src0_shape = {2, 16};
  src1_shape = {2};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "0", "u8"), false});

  // case: 3D u8 -> fp32(per channel, no zp)
  src0_shape = {2, 3, 16};
  src1_shape = {3};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "1", "u8"), false});

  // case: 3D s8 -> fp32(per channel, no zp)
  src0_shape = {2, 3, 16};
  src1_shape = {3};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "1", "s8"), false});

  return ::testing::ValuesIn(cases);
};

static auto CasesInt8Fp32 = []() {
  MemoryAllocator::InitStrategy();
  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src0_shape;
  std::vector<int64_t> src1_shape;
  std::vector<int64_t> src2_shape;

  // case: 2D u8 -> fp32(per tensor)
  src0_shape = {16, 1024};
  src1_shape = {1};
  src2_shape = {1};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape, src2_shape}, "1", "u8"), false});

  // case: 2D s8 -> fp32(per tensor)
  src0_shape = {16, 1024};
  src1_shape = {1};
  src2_shape = {1};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape, src2_shape}, "1", "s8"), false});

  // case: 2D u8 -> fp32(per channel, no zp)
  src0_shape = {2, 16};
  src1_shape = {16};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "1", "u8"), false});

  // case: 2D u8 -> fp32(per channel, no zp)
  src0_shape = {2, 16};
  src1_shape = {2};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "0", "u8"), false});

  // case: 3D u8 -> fp32(per channel, no zp)
  src0_shape = {2, 3, 16};
  src1_shape = {3};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "1", "u8"), false});

  // case: 3D s8 -> fp32(per channel, no zp)
  src0_shape = {2, 3, 16};
  src1_shape = {3};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "1", "s8"), false});

  // case: 2D u8 -> fp32(per channel, zp)
  src0_shape = {16, 1024};
  src1_shape = {1024};
  src2_shape = {1024};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape, src2_shape}, "1", "u8"), false});

  // case: 3D s8 -> fp32(per channel, zp)
  src0_shape = {2, 3, 16};
  src1_shape = {3};
  src2_shape = {3};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape, src2_shape}, "1", "s8"), false});


  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, DequantizeTest, CasesInt8Fp32());
INSTANTIATE_TEST_SUITE_P(Prefix, DequantizeTestLLGA, CasesInt8Fp32LLGA());
