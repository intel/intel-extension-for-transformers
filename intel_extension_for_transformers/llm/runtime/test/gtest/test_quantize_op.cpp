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

void GetTrueData(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output,
                 const shared_ptr<OperatorConfig>& conf) {
  auto attrs_map = conf->attributes();
  bool per_token = false;
  auto iter = attrs_map.find("per_token");
  if (iter != attrs_map.end() && attrs_map["per_token"] == "True") {
    per_token = true;
  }
  string output_dtype;
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype = attrs_map["output_dtype"];
  }
  // Get input tensor descriptor
  auto src_desc = input[0];
  int quantized_dim_elt_num = src_desc->shape().back();
  int batch_num =
      std::accumulate(src_desc->shape().begin(), src_desc->shape().end() - 1, size_t(1), std::multiplies<size_t>());

  const float* data_used = static_cast<const float*>(input[0]->data());
  // Allocate memory for output tensors
  auto mat_dst = reinterpret_cast<int8_t*>(const_cast<void*>(output[0]->data()));
  float* scale_dst;
  float* min_dst;
  if (output.size() > 1) {
    output[1]->set_shape({batch_num});
    output[2]->set_shape({batch_num});
    min_dst = reinterpret_cast<float*>(const_cast<void*>(output[1]->data()));
    scale_dst = reinterpret_cast<float*>(const_cast<void*>(output[2]->data()));
// Calculate dynamic quantization scale for each batch
#pragma omp parallel for
    for (int batch = 0; batch < batch_num; batch++) {
      float max = 0.f;
      for (int j = 0; j < quantized_dim_elt_num; j++)
        max = std::max(max, std::abs(data_used[batch * quantized_dim_elt_num + j]));
      scale_dst[batch] = max / 127.f;
    }
  } else {
    min_dst = reinterpret_cast<float*>(input[1]->mutable_data());
    auto scale = executor::GetScales(input[1]->data(), input[2]->data(), input[1]->size(), output_dtype);
    scale_dst = new float;
    *scale_dst = 1 / scale[0];
  }
// Perform quantization on the input tensor
#pragma omp parallel for
  for (int batch = 0; batch < batch_num; batch++) {
    for (int j = 0; j < quantized_dim_elt_num; j++) {
      if (output_dtype == "s8") {
        int ans = nearbyint(data_used[batch * quantized_dim_elt_num + j] / scale_dst[per_token ? batch : 0]);
        ans = ans > 127 ? 127 : ans;
        ans = ans < -128 ? -128 : ans;
        mat_dst[batch * quantized_dim_elt_num + j] = ans;
      } else {
        int ans = nearbyint((data_used[batch * quantized_dim_elt_num + j] - min_dst[per_token ? batch : 0]) /
                            scale_dst[per_token ? batch : 0]);
        ans = ans > 255 ? 255 : ans;
        ans = ans < 0 ? 0 : ans;
        mat_dst[batch * quantized_dim_elt_num + j] = ans;
      }
    }
  }
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
    bool is_equal = executor::CompareData<int8_t>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                                  q.output[0]->size(), 2);
    if (p.output.size() > 1)
      is_equal &= executor::CompareData<float>(p.output[2]->data(), p.output[2]->size(), q.output[2]->data(),
                                               q.output[2]->size(), 1e-4);
    return is_equal;
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

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape, const bool is_dynamic,
                                           const std::string& dst_dtype = "u8") {
  const auto& src0_shape = input_shape[0];
  const auto& dst0_shape = input_shape[1];  // dst
  const auto& dst1_shape = input_shape[2];  // dst_min
  const auto& dst2_shape = input_shape[3];  // dst_max
  shared_ptr<TensorConfig> src0_config = std::make_shared<TensorConfig>("src0", src0_shape, "fp32");
  shared_ptr<TensorConfig> dst0_config = std::make_shared<TensorConfig>("dst0", dst0_shape, dst_dtype);  // dst
  shared_ptr<TensorConfig> dst1_config = std::make_shared<TensorConfig>("dst1", dst1_shape, "fp32");     // dst_min
  shared_ptr<TensorConfig> dst2_config = std::make_shared<TensorConfig>("dst2", dst2_shape, "fp32");     // dst_max
  std::vector<shared_ptr<TensorConfig>> inputs_config = {src0_config};
  std::vector<shared_ptr<TensorConfig>> output_config = {dst0_config};
  if (is_dynamic) {
    output_config.push_back(dst1_config);
    output_config.push_back(dst2_config);
  }
  std::map<std::string, std::string> attr_map;
  if (is_dynamic) attr_map["per_token"] = "True";
  attr_map["output_dtype"] = dst_dtype;
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config =
      std::make_shared<OperatorConfig>("quantize", "QuantizeLinear", inputs_config, output_config, op_attr);
  // Step 2: Construct Tensor ptr
  auto src0_tensors = make_tensor_obj<float>(src0_config);

  Tensor* dst0_tensor = new Tensor(*dst0_config);
  dst0_tensor->add_tensor_life(1);
  Tensor* dst0_tensor_copy = new Tensor(*dst0_config);
  dst0_tensor_copy->add_tensor_life(1);

  Tensor* dst1_tensor = new Tensor(*dst1_config);
  dst1_tensor->add_tensor_life(1);
  Tensor* dst1_tensor_copy = new Tensor(*dst1_config);
  dst1_tensor_copy->add_tensor_life(1);

  Tensor* dst2_tensor = new Tensor(*dst2_config);
  dst2_tensor->add_tensor_life(1);
  Tensor* dst2_tensor_copy = new Tensor(*dst2_config);
  dst2_tensor_copy->add_tensor_life(1);

  if (is_dynamic) {
    OpArgs op_args = {{src0_tensors.first}, {dst0_tensor, dst1_tensor, dst2_tensor}, op_config};
    OpArgs op_args_copy = {{src0_tensors.second}, {dst0_tensor_copy, dst1_tensor_copy, dst2_tensor_copy}, op_config};
    return {op_args, op_args_copy};
  } else {
    executor::runtime_minmax(reinterpret_cast<const float*>(src0_tensors.first->data()), src0_tensors.first->size(),
                             reinterpret_cast<float*>(dst1_tensor->mutable_data()),
                             reinterpret_cast<float*>(dst2_tensor->mutable_data()));
    executor::runtime_minmax(reinterpret_cast<const float*>(src0_tensors.first->data()), src0_tensors.first->size(),
                             reinterpret_cast<float*>(dst1_tensor_copy->mutable_data()),
                             reinterpret_cast<float*>(dst2_tensor_copy->mutable_data()));
    OpArgs op_args = {{src0_tensors.first, dst1_tensor, dst2_tensor}, {dst0_tensor}, op_config};
    OpArgs op_args_copy = {{src0_tensors.second, dst1_tensor_copy, dst2_tensor_copy}, {dst0_tensor_copy}, op_config};
    return {op_args, op_args_copy};
  }
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
  cases.push_back({GenerateFp32Case({src0_shape, dst0_shape, dst1_shape, dst2_shape}, true, "s8"), false});

  src0_shape = {16, 1024};
  dst0_shape = {16, 1024};
  dst1_shape = {16};
  dst2_shape = {16};
  cases.push_back({GenerateFp32Case({src0_shape, dst0_shape, dst1_shape, dst2_shape}, true, "s8"), false});

  src0_shape = {4, 1};
  dst0_shape = {4, 1};
  dst1_shape = {4};
  dst2_shape = {4};
  cases.push_back({GenerateFp32Case({src0_shape, dst0_shape, dst1_shape, dst2_shape}, false, "s8"), false});

  src0_shape = {1028, 1024};
  dst0_shape = {1028, 1024};
  dst1_shape = {1028};
  dst2_shape = {1028};
  cases.push_back({GenerateFp32Case({src0_shape, dst0_shape, dst1_shape, dst2_shape}, false, "u8"), false});

  src0_shape = {256, 1024};
  dst0_shape = {256, 1024};
  dst1_shape = {256};
  dst2_shape = {256};
  cases.push_back({GenerateFp32Case({src0_shape, dst0_shape, dst1_shape, dst2_shape}, true, "s8"), false});

  return ::testing::ValuesIn(cases);
};
INSTANTIATE_TEST_SUITE_P(Prefix, QuantizeTest, CasesInt8Fp32());
// INSTANTIATE_TEST_SUITE_P(Prefix, QuantizeTestLLGA, CasesInt8Fp32LLGA());
