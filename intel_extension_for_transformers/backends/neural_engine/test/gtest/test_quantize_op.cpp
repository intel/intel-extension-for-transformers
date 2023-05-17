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
#include "gtest/gtest.h"
#include "../../include/common.hpp"
#include "../../include/conf.hpp"
#include "llga_kernel.hpp"
#include "llga_op_creator.hpp"
#include "../../include/operators/quantize.hpp"
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

template <typename T>
void Dynamic_Quantize_Batch(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output,
                            const shared_ptr<OperatorConfig>& conf) {
  auto attrs_map = conf->attributes();
  bool bTestCase = true;
  auto iter = attrs_map.find("batch");
  if (iter != attrs_map.end() && attrs_map["batch"] == "1") {
    bTestCase = false;
  }

  // Get input tensor descriptor
  auto src_desc = input[0];
  int quantized_dim_elt_num = src_desc->shape().back();
  int batch_num =
      std::accumulate(src_desc->shape().begin(), src_desc->shape().end() - 1, size_t(1), std::multiplies<size_t>());

  const float* data_used = static_cast<const float*>(input[0]->data());
  // Allocate memory for output tensors
  auto mat_dst = reinterpret_cast<int8_t*>(const_cast<void*>(output[0]->data()));
  auto scale_dst = reinterpret_cast<float*>(const_cast<void*>(output[2]->data()));

// Calculate dynamic quantization scale for each batch
#pragma omp parallel for
  for (int batch = 0; batch < batch_num; batch++) {
    float max = 0.f;
    for (int j = 0; j < quantized_dim_elt_num; j++)
      max = std::max(max, std::abs(data_used[batch * quantized_dim_elt_num + j]));
    scale_dst[batch] = max / 127.f;
  }

  // caculate single scale
  float max_value = 0.00001;
  for (int batch = 0; batch < batch_num; batch++) {
    for (int j = 0; j < quantized_dim_elt_num; j++)
      max_value = std::max(max_value, std::abs(data_used[batch * quantized_dim_elt_num + j]));
  }
  float scale_value = 127.f / max_value;

  if (bTestCase) {
    scale_dst[0] = scale_value;
    output[2]->set_shape({1});
  }

// Perform quantization on the input tensor
#pragma omp parallel for
  for (int batch = 0; batch < batch_num; batch++) {
    for (int j = 0; j < quantized_dim_elt_num; j++) {
      int ans = nearbyint(data_used[batch * quantized_dim_elt_num + j] /
                          (bTestCase ? (1. / scale_dst[0]) : scale_dst[batch]));
      ans = ans > 127 ? 127 : ans;
      ans = ans < -128 ? -128 : ans;
      mat_dst[batch * quantized_dim_elt_num + j] = ans;
    }
  }
}

void GetTrueData(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output,
                 const shared_ptr<OperatorConfig>& conf) {
  Dynamic_Quantize_Batch<int8_t>(input, output, conf);
}
bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::QuantizeOperator quantize(p.conf);
    quantize.Prepare(p.input, p.output);
    quantize.Reshape(p.input, p.output);
    quantize.Forward(p.input, p.output);
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
    return executor::CompareData<int8_t>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                         q.output[0]->size(), 2) &&
           executor::CompareData<float>(p.output[2]->data(), p.output[2]->size(), q.output[2]->data(),
                                        q.output[2]->size(), 1e-4);
  }
  return false;
}
class QuantizeTest : public testing::TestWithParam<TestParams> {
 protected:
  QuantizeTest() {}
  ~QuantizeTest() override {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(QuantizeTest, TestPostfix) {
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
std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape,
                                           const std::string& batch_ = "1", const std::string& dst_dtype = "u8") {
  const auto& src0_shape = input_shape[0];
  const auto& dst0_shape = input_shape[1];  // dst
  const auto& dst1_shape = input_shape[2];  // dst_min
  const auto& dst2_shape = input_shape[3];  // dst_max
  shared_ptr<TensorConfig> src0_config = std::make_shared<TensorConfig>("src0", src0_shape, "fp32");
  shared_ptr<TensorConfig> dst0_config = std::make_shared<TensorConfig>("dst0", dst0_shape, "s8");    // dst
  shared_ptr<TensorConfig> dst1_config = std::make_shared<TensorConfig>("dst1", dst1_shape, "fp32");  // dst_min
  shared_ptr<TensorConfig> dst2_config = std::make_shared<TensorConfig>("dst2", dst2_shape, "fp32");  // dst_max
  std::vector<shared_ptr<TensorConfig>> inputs_config = {src0_config};
  std::vector<shared_ptr<TensorConfig>> output_config = {dst0_config, dst1_config, dst2_config};
  std::map<std::string, std::string> attr_map;
  attr_map = {{"batch", batch_}};

  attr_map = {{"output_dtype", dst_dtype}};
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config =
      std::make_shared<OperatorConfig>("quantize", "QuantizeLinear", inputs_config, output_config, op_attr);
  // Step 2: Construct Tensor ptr
  auto src0_tensors = make_tensor_obj<float>(src0_config);

  Tensor* dst0_tensor = new Tensor(*dst0_config);
  dst0_tensor->add_tensor_life(1);
  Tensor* dst0_tensor_copy = new Tensor(*dst0_config);
  dst0_tensor_copy->add_tensor_life(1);

  Tensor* dst_tensor = new Tensor(*dst1_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst1_config);
  dst_tensor_copy->add_tensor_life(1);

  Tensor* dst2_tensor = new Tensor(*dst2_config);
  dst2_tensor->add_tensor_life(1);
  Tensor* dst2_tensor_copy = new Tensor(*dst2_config);
  dst2_tensor_copy->add_tensor_life(1);
  OpArgs op_args = {{src0_tensors.first}, {dst0_tensor, dst_tensor, dst2_tensor}, op_config};
  OpArgs op_args_copy = {{src0_tensors.second}, {dst0_tensor_copy, dst_tensor_copy, dst2_tensor_copy}, op_config};
  return {op_args, op_args_copy};
}
static auto CasesInt8Fp32 = []() {
  MemoryAllocator::InitStrategy();
  std::vector<TestParams> cases;
  // Config
  std::vector<int64_t> src0_shape;
  std::vector<int64_t> dst0_shape;  // dst
  std::vector<int64_t> dst1_shape;  // dst_min
  std::vector<int64_t> dst2_shape;  // dst_max

  src0_shape = {4, 1024};
  dst0_shape = {4, 1024};
  dst1_shape = {4};
  dst2_shape = {4};
  cases.push_back({GenerateFp32Case({src0_shape, dst0_shape, dst1_shape, dst2_shape}, "1", "s8"), false});

  src0_shape = {16, 1024};
  dst0_shape = {16, 1024};
  dst1_shape = {16};
  dst2_shape = {16};
  cases.push_back({GenerateFp32Case({src0_shape, dst0_shape, dst1_shape, dst2_shape}, "1", "s8"), false});

  src0_shape = {4, 1};
  dst0_shape = {4, 1};
  dst1_shape = {4};
  dst2_shape = {4};
  cases.push_back({GenerateFp32Case({src0_shape, dst0_shape, dst1_shape, dst2_shape}, "0", "s8"), false});

  src0_shape = {1028, 1024};
  dst0_shape = {1028, 1024};
  dst1_shape = {1028};
  dst2_shape = {1028};
  cases.push_back({GenerateFp32Case({src0_shape, dst0_shape, dst1_shape, dst2_shape}, "0", "s8"), false});

  src0_shape = {256, 1024};
  dst0_shape = {256, 1024};
  dst1_shape = {256};
  dst2_shape = {256};
  cases.push_back({GenerateFp32Case({src0_shape, dst0_shape, dst1_shape, dst2_shape}, "1", "s8"), false});

  return ::testing::ValuesIn(cases);
};
INSTANTIATE_TEST_SUITE_P(Prefix, QuantizeTest, CasesInt8Fp32());
// INSTANTIATE_TEST_SUITE_P(Prefix, QuantizeTestLLGA, CasesInt8Fp32LLGA());
