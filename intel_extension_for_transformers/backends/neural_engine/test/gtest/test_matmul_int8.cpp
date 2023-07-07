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
#include "../../include/operators/matmul.hpp"
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
  bool is_dynamic;
};

struct TestParams {
  std::pair<OpArgs, Tensor*> args;
  bool expect_to_fail;
};

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    int scales_num = 4;
    vector<void*> tmp_data(scales_num);
    vector<Tensor> tmp_tensor(scales_num);
    if (p.is_dynamic) {
      for (int i = p.input.size() - scales_num, j = 0; i < p.input.size(); i++, j++) {
        tmp_data[j] = p.input[i]->mutable_data();
        p.input[i]->unref_data(true);
        tmp_tensor[j].add_tensor_life(1);
        tmp_tensor[j].set_data(tmp_data[j]);
      }
    }
    executor::MatmulOperator matmul(p.conf);
    matmul.Prepare(p.input, p.output);
    matmul.Reshape(p.input, p.output);
    if (p.is_dynamic) {
      for (int i = p.input.size() - scales_num, j = 0; i < p.input.size(); i++, j++) {
        tmp_tensor[j].unref_data(true);
        p.input[i]->set_data(tmp_data[j]);
      }
    }
    matmul.Forward(p.input, p.output);
  } catch (const dnnl::error& e) {
    if (e.status != dnnl_status_t::dnnl_success && t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  } catch (const std::string& e) {
    if (e == "Windows" && t.expect_to_fail)
      return true;
    else
      return false;
  }
  if (t.expect_to_fail) return true;
  bool is_equal;
  if (q->dtype() == "fp32") {
    is_equal = executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q->data(), q->size(), 0.1);
  } else if (q->dtype() == "s8") {
    is_equal = executor::CompareData<int8_t>(p.output[0]->data(), p.output[0]->size(), q->data(), q->size(), 1);
  } else if (q->dtype() == "u8") {
    is_equal = executor::CompareData<uint8_t>(p.output[0]->data(), p.output[0]->size(), q->data(), q->size(), 1);
  }
  return is_equal;
}

class MatmulInt8Test : public testing::TestWithParam<TestParams> {
 protected:
  MatmulInt8Test() {}
  ~MatmulInt8Test() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MatmulInt8Test, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

Tensor* make_int32_bias_obj(const shared_ptr<TensorConfig>& bias_tensor_config, const float* origin_data,
                            Tensor* weight_fp32, Tensor* weight_min, Tensor* weight_scale, Tensor* src_min,
                            Tensor* src_scale) {
  Tensor* bias_tensor = new Tensor(*bias_tensor_config);
  bias_tensor->add_tensor_life(1);
  int32_t* bias_data = reinterpret_cast<int32_t*>(bias_tensor->mutable_data());
  const float* weight_scales = reinterpret_cast<const float*>(weight_scale->data());
  const float* src_scales = reinterpret_cast<const float*>(src_scale->data());
  const float zp = *reinterpret_cast<const float*>(src_min->data());
  const float* weight_data = reinterpret_cast<const float*>(weight_fp32->data());
#pragma omp parallel for
  for (int y = 0; y < weight_fp32->shape()[1]; y++) {
    float compensation = 0;
    for (int x = 0; x < weight_fp32->shape()[0]; x++) compensation += weight_data[x * weight_fp32->shape()[1] + y];
    bias_data[y] = (origin_data[y] + compensation * zp) * src_scales[0] * weight_scales[y];
  }
  return bias_tensor;
}

Tensor* get_fp32_dst(const shared_ptr<TensorConfig>& dst_tensor_config, vector<Tensor*> inputs) {
  using dnnl::matmul;
  using dnnl::memory;
  Tensor* dst_tensor = new Tensor(*dst_tensor_config);
  dst_tensor->add_tensor_life(1);
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream = dnnl::stream(engine);
  Tensor* src = inputs[0];
  Tensor* weight = inputs[1];
  Tensor* post = inputs[2];
  dnnl::post_ops po;
  memory::desc binary_md;
  memory binary_m_;
  if (post != nullptr) {
    binary_md = memory::desc(post->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    binary_m_ = memory(binary_md, engine, post->mutable_data());
    po.append_binary(dnnl::algorithm::binary_add, binary_md);
  }
  auto src_md = memory::desc(src->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  auto weights_md = memory::desc(weight->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  auto dst_md = memory::desc(dst_tensor->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  auto src_mem = memory(src_md, engine, src->mutable_data());
  auto weights_mem = memory(weights_md, engine, weight->mutable_data());
  auto dst_mem = memory(dst_md, engine, dst_tensor->mutable_data());
  auto matmul_d = matmul::desc(src_md, weights_md, dst_md);
  dnnl::primitive_attr attr;
  attr.set_post_ops(po);
  auto matmul_pd = matmul::primitive_desc(matmul_d, attr, engine);
  auto matmul_prim = matmul(matmul_pd);
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, src_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  matmul_args.insert({DNNL_ARG_DST, dst_mem});
  if (post != nullptr) {
    matmul_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = binary_m_;
  }
  matmul_prim.execute(engine_stream, matmul_args);
  engine_stream.wait();
  return dst_tensor;
}

Tensor* make_fp32_tensor_obj(const shared_ptr<TensorConfig>& a_tensor_config, float bound1 = -10, float bound2 = 10) {
  // step1: set shape
  Tensor* a_tensor = new Tensor(*a_tensor_config);
  // step2: set tensor life
  a_tensor->add_tensor_life(1);
  // step3: library buffer can only be obtained afterwards
  auto tensor_data = a_tensor->mutable_data();
  executor::InitVector(static_cast<float*>(tensor_data), a_tensor->size(), bound1, bound2);
  return a_tensor;
}

vector<Tensor*> quantize2int8_tensor_obj(const vector<shared_ptr<TensorConfig>>& tensor_configs,
                                         const float* origin_fp32_data, bool per_channel = false,
                                         bool need_scale = false) {
  vector<Tensor*> tensors(3);
  for (int i = 0; i < 3; i++) {
    tensors[i] = new Tensor(*tensor_configs[i]);
    tensors[i]->add_tensor_life(1);
  }
  float* min_data = reinterpret_cast<float*>(tensors[1]->mutable_data());
  float* max_data = reinterpret_cast<float*>(tensors[2]->mutable_data());
  void* dst_data = tensors[0]->mutable_data();
  if (per_channel) {
    for (int y = 0; y < tensors[0]->shape()[1]; y++) {
      min_data[y] = origin_fp32_data[y];
      max_data[y] = origin_fp32_data[y];
      for (int x = 1; x < tensors[0]->shape()[0]; x++) {
        min_data[y] = std::min(min_data[y], origin_fp32_data[x * tensors[0]->shape()[1] + y]);
        max_data[y] = std::max(max_data[y], origin_fp32_data[x * tensors[0]->shape()[1] + y]);
      }
      vector<float> scales = executor::GetScales(min_data + y, max_data + y, 1, tensors[0]->dtype());
      for (int x = 0; x < tensors[0]->shape()[0]; x++)
        if (tensors[0]->dtype() == "u8") {
          uint8_t* dst_data_ = reinterpret_cast<uint8_t*>(dst_data) + x * tensors[0]->shape()[1] + y;
          int32_t data = nearbyint((origin_fp32_data[x * tensors[0]->shape()[1] + y] - min_data[y]) * scales[0]);
          data = data < 0 ? 0 : data;
          data = data > 255 ? 255 : data;
          *dst_data_ = static_cast<uint8_t>(data);
        } else if (tensors[0]->dtype() == "s8") {
          int8_t* dst_data_ = reinterpret_cast<int8_t*>(dst_data) + x * tensors[0]->shape()[1] + y;
          int32_t data = nearbyint(origin_fp32_data[x * tensors[0]->shape()[1] + y] * scales[0]);
          data = data < -128 ? -128 : data;
          data = data > 127 ? 127 : data;
          *dst_data_ = static_cast<int8_t>(data);
        }
      if (need_scale) max_data[y] = 1.0 / scales[0];
      // memcpy(max_data + y, scales.data(), 1 * sizeof(float));
    }
  } else {
    executor::runtime_minmax(origin_fp32_data, tensors[0]->size(), min_data, max_data);
    vector<float> scales = executor::GetScales(min_data, max_data, 1, tensors[0]->dtype());
#if __AVX512F__
    executor::Quantize_avx512(tensors[0]->size(), tensors[0]->dtype(), origin_fp32_data, min_data, scales, dst_data);
#else
    executor::Quantize(tensors[0]->size(), tensors[0]->dtype(), origin_fp32_data, min_data, scales, dst_data);
#endif
    if (need_scale) *max_data = 1.0 / scales[0];
    // memcpy(max_data, scales.data(), 1 * sizeof(float));
  }
  return tensors;
}

std::pair<OpArgs, Tensor*> GenerateInt8Case(const std::vector<std::vector<int64_t>>& input_shape, bool is_dynamic,
                                            std::string input_type = "s8", std::string output_type = "u8",
                                            std::string append_op = "") {
  // support s8s8fp32 and u8s8u8 without bias
  // Step 1: Construct Tensor config ptr
  const auto& src0_shape = input_shape[0];
  const auto& src1_shape = input_shape[1];
  std::vector<int64_t> bias_shape = {src1_shape[1]};
  std::vector<int64_t> dst_shape = {src0_shape[0], src1_shape[1]};

  auto src_fp32_config = std::make_shared<TensorConfig>("src_fp32", src0_shape, "fp32");
  auto src_u8_config = std::make_shared<TensorConfig>("src", src0_shape, input_type);
  auto src_min_config = std::make_shared<TensorConfig>("src_min", vector<int64_t>({1}), "fp32");
  auto src_scale_config = std::make_shared<TensorConfig>("src_scale", vector<int64_t>({1}), "fp32");
  Tensor* src_fp32;
  if (input_type == "u8")
    src_fp32 = make_fp32_tensor_obj(src_fp32_config, 0, 1);
  else
    src_fp32 = make_fp32_tensor_obj(src_fp32_config);
  auto src_tensors = quantize2int8_tensor_obj({src_u8_config, src_min_config, src_scale_config},
                                              reinterpret_cast<const float*>(src_fp32->data()), false, is_dynamic);
  auto weight_fp32_config = std::make_shared<TensorConfig>("weight_fp32", src1_shape, "fp32");
  auto weight_s8_config = std::make_shared<TensorConfig>("weight", src1_shape, "s8");
  auto weight_min_config = std::make_shared<TensorConfig>("weight_min", vector<int64_t>({1}), "fp32");
  auto weight_scale_config = std::make_shared<TensorConfig>("weight_scale", vector<int64_t>({1}), "fp32");
  Tensor* weight_fp32 = make_fp32_tensor_obj(weight_fp32_config);
  auto weight_tensors = quantize2int8_tensor_obj({weight_s8_config, weight_min_config, weight_scale_config},
                                                 reinterpret_cast<const float*>(weight_fp32->data()), false,
                                                 is_dynamic);  // matmul only support per_tensor
  // weight_fp32->print();
  // for (auto tensor : weight_tensors) tensor->print();
  auto post_fp32_config = std::make_shared<TensorConfig>("post", dst_shape, "fp32");
  Tensor* post_fp32 = make_fp32_tensor_obj(post_fp32_config);
  // get true fp32 result and calculate min/max
  auto dst_fp32_config = std::make_shared<TensorConfig>("dst_fp32", dst_shape, "fp32");
  auto dst_config = std::make_shared<TensorConfig>("dst", dst_shape, output_type);
  auto dst_min_config = std::make_shared<TensorConfig>("dst_min", vector<int64_t>({1}), "fp32");
  auto dst_scale_config = std::make_shared<TensorConfig>("dst_scale", vector<int64_t>({1}), "fp32");
  Tensor* dst_fp32 = get_fp32_dst(
      dst_fp32_config,
      {src_fp32, weight_fp32, ((output_type == "fp32" && append_op == "binary_add") ? post_fp32 : nullptr)});
  Tensor* dst = new Tensor(*dst_config);
  dst->add_tensor_life(1);
  Tensor* dst_min = new Tensor(*dst_min_config);
  dst_min->add_tensor_life(1);
  Tensor* dst_scale = new Tensor(*dst_scale_config);
  dst_scale->add_tensor_life(1);
  executor::runtime_minmax(reinterpret_cast<float*>(dst_fp32->mutable_data()), dst_fp32->size(),
                           reinterpret_cast<float*>(dst_min->mutable_data()),
                           reinterpret_cast<float*>(dst_scale->mutable_data()));
  vector<float> scales = executor::GetScales(dst_min->data(), dst_scale->data(), 1, output_type);
  if (is_dynamic) memcpy(dst_scale->mutable_data(), scales.data(), 1 * sizeof(float));

  vector<shared_ptr<TensorConfig>> inputs_configs = {src_u8_config,    weight_s8_config,  src_min_config,
                                                     src_scale_config, weight_min_config, weight_scale_config};
  vector<shared_ptr<TensorConfig>> output_configs = {dst_config};
  vector<Tensor*> inputs = {src_tensors[0], weight_tensors[0], src_tensors[1],
                            src_tensors[2], weight_tensors[1], weight_tensors[2]};
  vector<Tensor*> outputs = {dst};
  map<string, string> attr_map = {{"output_dtype", output_type}};
  if (output_type == "fp32" && append_op == "binary_add") {
    inputs_configs.insert(inputs_configs.begin() + 2, post_fp32_config);
    inputs.insert(inputs.begin() + 2, post_fp32);
    attr_map["append_op"] = append_op;
  }
  if (!is_dynamic) {
    inputs_configs.push_back(dst_min_config);
    inputs_configs.push_back(dst_scale_config);
    inputs.push_back(dst_min);
    inputs.push_back(dst_scale);
  } else {
    output_configs.push_back(dst_min_config);
    output_configs.push_back(dst_scale_config);
    outputs.push_back(dst_min);
    outputs.push_back(dst_scale);
  }
  // Step 1.1: Construct Operator config obj
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  auto op_config = std::make_shared<OperatorConfig>("matmul", output_type, inputs_configs, output_configs, op_attr);

  OpArgs op_args = {inputs, outputs, op_config, is_dynamic};
  if (output_type == "fp32") {
    // dst_fp32->print();
    return {op_args, dst_fp32};
  } else {
    Tensor* true_data = new Tensor(*dst_config);
    true_data->add_tensor_life(1);
#if __AVX512F__
    executor::Quantize_avx512(true_data->size(), output_type, dst_fp32->data(),
                              reinterpret_cast<const float*>(dst_min->data()), scales, true_data->mutable_data());
#else
    executor::Quantize(true_data->size(), output_type, dst_fp32->data(),
                       reinterpret_cast<const float*>(dst_min->data()), scales, true_data->mutable_data());
#endif
    // true_data->print();
    return {op_args, true_data};
  }
}

static auto CasesInt8 = []() {
#ifdef _WIN32
  constexpr auto FAIL_ON_WIN = true;
  LOG(WARNING) << "`expect_to_fail` is set to true for some test cases on Windows.";
#else
  constexpr auto FAIL_ON_WIN = false;
#endif

  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src0_shape;
  std::vector<int64_t> src1_shape;
  std::vector<int64_t> bias_shape;

  src0_shape = {4, 2};
  src1_shape = {2, 3};
  cases.push_back({GenerateInt8Case({src0_shape, src1_shape}, false, "s8", "fp32"), FAIL_ON_WIN});
  // cases.push_back({GenerateInt8Case({src0_shape, src1_shape}, false, "u8", "u8"), FAIL_ON_WIN});

  src0_shape = {5, 7};
  src1_shape = {7, 3};
  cases.push_back({GenerateInt8Case({src0_shape, src1_shape}, true, "s8", "fp32"), FAIL_ON_WIN});
  cases.push_back({GenerateInt8Case({src0_shape, src1_shape}, true, "u8", "u8"), FAIL_ON_WIN});
  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, MatmulInt8Test, CasesInt8());
