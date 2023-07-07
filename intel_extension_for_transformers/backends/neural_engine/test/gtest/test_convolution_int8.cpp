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
#include "../../include/operators/convolution.hpp"
#include "gtest/gtest.h"
using executor::AttrConfig;
using executor::MemoryAllocator;
using executor::OperatorConfig;
using executor::Tensor;
using executor::TensorConfig;

vector<int64_t> get_dst_shape(vector<int64_t> src_shape, vector<int64_t> weight_shape, vector<int64_t> pads,
                              vector<int64_t> strides, int group = 1) {
  vector<int64_t> dst_shape;
  vector<int64_t> padding_dims_l;
  vector<int64_t> padding_dims_r;
  switch (src_shape.size()) {
    case 3: {
      // src_: N * IC* IH, weight_: OC * KC * KH
      // pad: (PH_L, PH_R), stride: (SH)
      // OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
      // dst_: N * OC * OH
      const int64_t N = src_shape[0];
      const int64_t IC = src_shape[1];
      const int64_t IH = src_shape[2];
      const int64_t OC = weight_shape[0];
      const int64_t KC = weight_shape[1];
      const int64_t KH = weight_shape[2];
      const int64_t PH_L = pads[0];
      const int64_t PH_R = pads[1];
      const int64_t SH = strides[0];
      const int64_t OH = (IH - KH + PH_L + PH_R) / SH + 1;
      padding_dims_l = {PH_L};
      padding_dims_r = {PH_R};
      dst_shape = {N, OC, OH};
      break;
    }
    case 4: {
      // src_: N * IC* IH * IW, weight_: OC * KC * KH * KW
      // pad: (PH_L, PH_R, PW_L, PW_R), stride: (SH, SW)
      // OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
      // OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width
      // dst_: N * OC * OH * OW
      const int64_t N = src_shape[0];
      const int64_t IC = src_shape[1];
      const int64_t IH = src_shape[2];
      const int64_t IW = src_shape[3];
      const int64_t OC = weight_shape[0];
      const int64_t KC = weight_shape[1];
      const int64_t KH = weight_shape[2];
      const int64_t KW = weight_shape[3];
      const int64_t PH_L = pads[0];
      const int64_t PH_R = pads[1];
      const int64_t PW_L = pads[2];
      const int64_t PW_R = pads[3];
      const int64_t SH = strides[0];
      const int64_t SW = strides[1];
      const int64_t OH = (IH - KH + PH_L + PH_R) / SH + 1;
      const int64_t OW = (IW - KW + PW_L + PW_R) / SW + 1;
      padding_dims_l = {PH_L, PW_L};
      padding_dims_r = {PH_R, PW_R};
      dst_shape = {N, OC, OH, OW};
      break;
    }
    default:
      LOG(ERROR) << "Input size " << src_shape.size() << " is not supported in convolution!";
  }
  return dst_shape;
}

struct OpArgs {
  vector<Tensor*> input;
  vector<Tensor*> output;
  shared_ptr<OperatorConfig> conf;
};

struct TestParams {
  std::pair<OpArgs, vector<Tensor*>> args;
  bool expect_to_fail;
};

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  {
    executor::ConvolutionOperator convolution(p.conf);
    int scales_num = 2;
    vector<void*> tmp_data(scales_num);
    vector<Tensor> tmp_tensor(scales_num);
    for (int i = 3, j = 0; i < 5; i++, j++) {
      tmp_data[j] = p.input[i]->mutable_data();
      p.input[i]->unref_data(true);
      tmp_tensor[j].add_tensor_life(1);
      tmp_tensor[j].set_data(tmp_data[j]);
    }
    convolution.Prepare(p.input, p.output);
    convolution.Reshape(p.input, p.output);
    for (int i = 3, j = 0; i < 5; i++, j++) {
      tmp_tensor[j].unref_data(true);
      p.input[i]->set_data(tmp_data[j]);
    }
    convolution.Forward(p.input, p.output);
  }
  if (!t.expect_to_fail) {
    bool is_equal = true;
    if (q[0]->dtype() == "fp32") {
      is_equal &=
          executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q[0]->data(), q[0]->size(), 0.1);
    } else if (q[0]->dtype() == "s8") {
      is_equal &=
          executor::CompareData<int8_t>(p.output[0]->data(), p.output[0]->size(), q[0]->data(), q[0]->size(), 3);
    } else if (q[0]->dtype() == "u8") {
      is_equal &=
          executor::CompareData<uint8_t>(p.output[0]->data(), p.output[0]->size(), q[0]->data(), q[0]->size(), 3);
    }
    return is_equal;
  }
  return false;
}

class ConvolutionInt8Test : public testing::TestWithParam<TestParams> {
 protected:
  ConvolutionInt8Test() {}
  ~ConvolutionInt8Test() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(ConvolutionInt8Test, TestPostfix) {
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

Tensor* get_fp32_dst(const shared_ptr<TensorConfig>& dst_tensor_config, vector<Tensor*> inputs, vector<int64_t> strides,
                     vector<int64_t> pads) {
  // only support group==1
  using dnnl::convolution_forward;
  using dnnl::memory;
  Tensor* dst_tensor = new Tensor(*dst_tensor_config);
  dst_tensor->add_tensor_life(1);
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream = dnnl::stream(engine);
  Tensor* src = inputs[0];
  Tensor* weight = inputs[1];
  Tensor* bias = inputs[2];
  auto src_md = memory::desc(src->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::abcd);
  auto weight_md = memory::desc(weight->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::abcd);
  auto bias_md = memory::desc({bias->shape()[0]}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::a);
  auto dst_md = memory::desc(dst_tensor->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::abcd);

  auto src_mem = memory(src_md, engine, src->mutable_data());
  auto weight_mem = memory(weight_md, engine, weight->mutable_data());
  auto bias_mem = memory(bias_md, engine, bias->mutable_data());
  auto dst_mem = memory(dst_md, engine, dst_tensor->mutable_data());

  vector<int64_t> padding_dims_l(pads.begin(), pads.begin() + pads.size() / 2);
  vector<int64_t> padding_dims_r(pads.begin() + pads.size() / 2, pads.end());
  auto convolution_d =
      convolution_forward::desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto, src_md,
                                weight_md, bias_md, dst_md, strides, padding_dims_l, padding_dims_r);
  auto convolution_pd = convolution_forward::primitive_desc(convolution_d, engine);
  auto convolution_prim = convolution_forward(convolution_pd);
  std::unordered_map<int, memory> convolution_args;
  convolution_args.insert({DNNL_ARG_SRC, src_mem});
  convolution_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
  convolution_args.insert({DNNL_ARG_BIAS, bias_mem});
  convolution_args.insert({DNNL_ARG_DST, dst_mem});

  convolution_prim.execute(engine_stream, convolution_args);
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
                                         const float* origin_fp32_data, bool per_channel = false) {
  vector<Tensor*> tensors(3);
  for (int i = 0; i < 3; i++) {
    tensors[i] = new Tensor(*tensor_configs[i]);
    tensors[i]->add_tensor_life(1);
  }
  float* min_data = reinterpret_cast<float*>(tensors[1]->mutable_data());
  float* max_data = reinterpret_cast<float*>(tensors[2]->mutable_data());
  void* dst_data = tensors[0]->mutable_data();
  if (per_channel) {
    int channel_num = tensors[0]->shape()[0];
    int channel_size = tensors[0]->size() / channel_num;
    for (int y = 0; y < channel_num; y++) {
      executor::runtime_minmax(origin_fp32_data + y * channel_size, channel_size, min_data + y, max_data + y);
      vector<float> scales = executor::GetScales(min_data + y, max_data + y, 1, tensors[0]->dtype());
#if __AVX512F__
      executor::Quantize_avx512(channel_size, tensors[0]->dtype(), origin_fp32_data + y * channel_size, min_data + y,
                                scales, reinterpret_cast<char*>(dst_data) + y * channel_size);
#else
      executor::Quantize(channel_size, tensors[0]->dtype(), origin_fp32_data + y * channel_size, min_data + y, scales,
                         reinterpret_cast<char*>(dst_data) + y * channel_size);
#endif
      max_data[y] = 1.0 / scales[0];
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
    *max_data = 1.0 / scales[0];
    // memcpy(max_data, scales.data(), 1 * sizeof(float));
  }
  return tensors;
}

std::pair<OpArgs, vector<Tensor*>> GenerateInt8Case(const vector<vector<int64_t>>& input_shape, bool is_dynamic,
                                                    std::string group, std::string pads_str, std::string strides_str,
                                                    std::string input_type = "u8", std::string output_type = "fp32",
                                                    std::string append_op = "") {
  // support s8s8fp32 and u8s8u8 without bias
  // Step 1: Construct Tensor config ptr
  const auto& src0_shape = input_shape[0];
  const auto& src1_shape = input_shape[1];
  vector<int64_t> pads;
  executor::StringSplit<int64_t>(&pads, pads_str, ",");
  vector<int64_t> strides;
  executor::StringSplit<int64_t>(&strides, strides_str, ",");
  vector<int64_t> bias_shape = input_shape[2];
  vector<int64_t> dst_shape = get_dst_shape(src0_shape, src1_shape, pads, strides);

  auto src_fp32_config = std::make_shared<TensorConfig>("src_fp32", src0_shape, "fp32");
  auto src_u8_config = std::make_shared<TensorConfig>("src", src0_shape, input_type);
  auto src_min_config = std::make_shared<TensorConfig>("src_min", vector<int64_t>({1}), "fp32");
  auto src_scale_config = std::make_shared<TensorConfig>("src_scale", vector<int64_t>({1}), "fp32");
  Tensor* src_fp32;
  src_fp32 = make_fp32_tensor_obj(src_fp32_config, -5.0, 4.0);
  auto src_tensors = quantize2int8_tensor_obj({src_u8_config, src_min_config, src_scale_config},
                                              reinterpret_cast<const float*>(src_fp32->data()), false);
  // src_fp32->print();
  // for (auto tensor : src_tensors) tensor->print();
  bool per_channel = true;
  auto weight_fp32_config = std::make_shared<TensorConfig>("weight_fp32", src1_shape, "fp32");
  auto weight_s8_config = std::make_shared<TensorConfig>("weight", src1_shape, "s8");
  auto weight_min_config =
      std::make_shared<TensorConfig>("weight_min", vector<int64_t>({per_channel ? bias_shape[0] : 1}), "fp32");
  auto weight_scale_config =
      std::make_shared<TensorConfig>("weight_scale", vector<int64_t>({per_channel ? bias_shape[0] : 1}), "fp32");
  Tensor* weight_fp32 = make_fp32_tensor_obj(weight_fp32_config, -0.3, 0.4);
  auto weight_tensors = quantize2int8_tensor_obj({weight_s8_config, weight_min_config, weight_scale_config},
                                                 reinterpret_cast<const float*>(weight_fp32->data()), per_channel);
  // weight_fp32->print();
  // for (auto tensor : weight_tensors) tensor->print();
  auto bias_fp32_config = std::make_shared<TensorConfig>("bias", bias_shape, "fp32");
  Tensor* bias_fp32 = make_fp32_tensor_obj(bias_fp32_config, -0.1, 0.2);
  auto post_fp32_config = std::make_shared<TensorConfig>("post", dst_shape, "fp32");
  Tensor* post_fp32 = make_fp32_tensor_obj(post_fp32_config);
  // get true fp32 result and calculate min/max
  auto dst_fp32_config = std::make_shared<TensorConfig>("dst_fp32", dst_shape, "fp32");
  auto dst_config = std::make_shared<TensorConfig>("dst", dst_shape, output_type);
  auto dst_min_config = std::make_shared<TensorConfig>("dst_min", vector<int64_t>({1}), "fp32");
  auto dst_scale_config = std::make_shared<TensorConfig>("dst_scale", vector<int64_t>({1}), "fp32");
  Tensor* dst_fp32 = get_fp32_dst(dst_fp32_config, {src_fp32, weight_fp32, bias_fp32}, strides, pads);
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
  memcpy(dst_scale->mutable_data(), scales.data(), 1 * sizeof(float));

  vector<shared_ptr<TensorConfig>> inputs_configs = {src_u8_config,      weight_s8_config, bias_fp32_config,
                                                     src_min_config,     src_scale_config, weight_min_config,
                                                     weight_scale_config};
  vector<shared_ptr<TensorConfig>> output_configs = {dst_config};
  vector<Tensor*> inputs = {src_tensors[0], weight_tensors[0], bias_fp32,        src_tensors[1],
                            src_tensors[2], weight_tensors[1], weight_tensors[2]};
  vector<Tensor*> outputs = {dst};
  map<string, string> attr_map = {
      {"group", group}, {"pads", pads_str}, {"strides", strides_str}, {"output_dtype", output_type}};
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
    Tensor* dst_min_output = new Tensor(*dst_min_config);
    dst_min_output->add_tensor_life(1);
    Tensor* dst_scale_output = new Tensor(*dst_scale_config);
    dst_scale_output->add_tensor_life(1);
    outputs.push_back(dst_min_output);
    outputs.push_back(dst_scale_output);
  }
  // Step 1.1: Construct Operator config obj
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  auto op_config =
      std::make_shared<OperatorConfig>("convolution", output_type, inputs_configs, output_configs, op_attr);

  OpArgs op_args = {inputs, outputs, op_config};
  if (output_type == "fp32") {
    // dst_fp32->print();
    return {op_args, {dst_fp32}};
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
    return {op_args, {true_data, dst_min, dst_scale}};
  }
}

static auto CasesInt8 = []() {
  MemoryAllocator::InitStrategy();

  vector<TestParams> cases;

  // Config
  vector<int64_t> src_shape;
  vector<int64_t> weight_shape;
  vector<int64_t> bias_shape;
  vector<int64_t> post_shape;
  std::string group = "";
  std::string pads = "";
  std::string strides = "";
  std::string src_perm = "";
  std::string dst_perm = "";
  std::string append_op = "";

  src_shape = {3, 16, 13, 13};
  weight_shape = {32, 16, 3, 3};
  bias_shape = {32};
  src_perm = "0,1,2,3";
  dst_perm = "0,1,2,3";
  group = "1";
  pads = "1,1,1,1";
  strides = "4,4";
  append_op = "";

  for (string output_dtype : {"fp32", "u8"}) {
    cases.push_back(
        {GenerateInt8Case({src_shape, weight_shape, bias_shape}, true, group, pads, strides, "s8", output_dtype),
         false});
#ifndef _WIN32  // TODO(Yucheng): Check if this case work for onednn 3.x on WIN
    cases.push_back(
        {GenerateInt8Case({src_shape, weight_shape, bias_shape}, true, group, pads, strides, "u8", output_dtype),
         false});
#endif
  }

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, ConvolutionInt8Test, CasesInt8());
