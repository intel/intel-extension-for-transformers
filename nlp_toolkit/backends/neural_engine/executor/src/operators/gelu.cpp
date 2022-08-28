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

#include "gelu.hpp"

#include "common.hpp"

namespace executor {

GeluOperator::GeluOperator(const OperatorConfig& conf) : Operator(conf) {
  auto attrs_map = operator_conf_.attributes();
  auto iter = attrs_map.find("algorithm");
  if (iter != attrs_map.end()) {
    algorithm_ = iter->second;
  }

  if (attrs_map.find("in8_lut_optimize") != attrs_map.end()) {
    int8_lut_optimize = true;
  }

  if (attrs_map.find("int8_lut_acc_test") != attrs_map.end()) {
    int8_lut_acc_test = true;
  }
}

void GeluOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (int8_lut_optimize || int8_lut_acc_test) output[0]->set_dtype("u8");
}

void GeluOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (int8_lut_optimize)
    ReshapeWithSparselib(input, output);
  else if (int8_lut_acc_test)
    ReshapeWithInt8LutAccTest(input, output);
  else
    ReshapeWithOnednn(input, output);
}

void GeluOperator::ReshapeWithInt8LutAccTest(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto input_dt = input[0]->dtype();
  auto src_min = input[1];
  auto src_max = input[2];
  Tensor* dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(input[0]->shape());
}

void GeluOperator::ReshapeWithSparselib(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto input_dt = input[0]->dtype();
  auto src_min = input[1];
  auto src_max = input[2];
  Tensor* dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(input[0]->shape());
  //  get scale & zero point;
  const float* min_p = static_cast<const float*>(src_min->data());
  const float* max_p = static_cast<const float*>(src_max->data());
  float scale = (max_p[0] - min_p[0]) / 255;
  float zp;
  jd::data_type attr_dtype;
  // gen int8-lut attr
  if (input_dt == "s8") {
    attr_dtype = jd::data_type::s8;
    src_desc_ = {input[0]->shape(), jd::data_type::s8, jd::format_type::undef};
    zp = -min_p[0] / scale;
  } else if (input_dt == "u8") {
    attr_dtype = jd::data_type::u8;
    src_desc_ = {input[0]->shape(), jd::data_type::u8, jd::format_type::undef};
    zp = min_p[0] + min_p[0] * scale;
  } else {
    LOG(ERROR) << "int8-lut kernel only support s8/u8 as input.";
  }
  std::unordered_map<std::string, std::string> op_attr;
  op_attr["postop_list"] = std::to_string(scale) + "+" + std::to_string(zp) + "gelu";

  jd::postop_attr dequantize_attr{attr_dtype, jd::postop_type::eltwise, jd::postop_alg::dequantize, zp, 0, scale};
  jd::postop_attr quantize_attr{jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::quantize, zp, 0, scale};

  jd::postop_attr int8_lut_attr{attr_dtype, jd::postop_type::eltwise, jd::postop_alg::int8_lut};

  dst_desc_ = {output[0]->shape(), jd::data_type::u8, jd::format_type::undef};

  jd::postop_attr gelu_attr{jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::gelu};
  std::vector<jd::postop_attr> postop_attrs = {int8_lut_attr, dequantize_attr, gelu_attr, quantize_attr};

  jd::operator_desc op_desc(jd::kernel_kind::eltwiseop, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            {src_desc_, dst_desc_}, op_attr, postop_attrs);
  jd::eltwiseop_desc eltwiseop_desc(op_desc);
  eltwiseop_ker = jd::eltwiseop(eltwiseop_desc);
}

void GeluOperator::ReshapeWithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Prepare tensors shape and memory descriptors
  // 1.1: Prepare src tensor shape
  const memory::dims& src_shape = input[0]->shape();

  // 1.2 Set dst tensor shape
  const memory::dims& dst_shape = src_shape;
  Tensor* dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);

  // 1.3 Get tensor's strides
  memory::dims src_stride = GetStrides(src_shape);
  memory::dims dst_stride = GetStrides(dst_shape);

  // 1.4 Prepare memory descriptors
  memory::desc src_md(src_shape, memory::data_type::f32, src_stride);
  memory::desc dst_md(dst_shape, memory::data_type::f32, dst_stride);

  // 1.5 Prepare memory objects (cached)
  src_m_ = memory(src_md, eng_);
  dst_m_ = memory(dst_md, eng_);

  //// Part2: Prepare primitive
  // 2.1 Prepare op descriptors
  algorithm gelu_algorithm = algorithm::eltwise_gelu_tanh;
  if (algorithm_ == "gelu_erf") {
    gelu_algorithm = algorithm::eltwise_gelu_erf;
  } else if (algorithm_ == "gelu_tanh") {
    gelu_algorithm = algorithm::eltwise_gelu_tanh;
  } else {
    LOG(ERROR) << "Gelu algorithm is: " << algorithm_ << ", not supported. Only gelu_erf or gelu_tanh is supported.";
  }
  auto gelu_d = dnnl::eltwise_forward::desc(prop_kind::forward_inference, gelu_algorithm, src_md, 0.f, 0.f);

  // 2.2 Prepare primitive descriptors
  dnnl::eltwise_forward::primitive_desc gelu_pd(gelu_d, eng_);

  // 2.3 Prepare primitive objects (cached)
  gelu_p_ = dnnl::eltwise_forward(gelu_pd);
}

void GeluOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (int8_lut_optimize)
    ForwardWithSparselib(input, output);
  else if (int8_lut_acc_test)
    ForwardWithInt8LutAccTest(input, output);
  else
    ForwardWithOnednn(input, output);
}

void GeluOperator::ForwardWithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 1. Alias variables part
  const void* src_data = input[0]->data();
  void* dst_data = output[0]->mutable_data();

  // 2. Prepare memory objects with data_ptr
  dnnl::stream s(eng_);
  src_m_.set_data_handle(const_cast<void*>(src_data), s);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

  // 3. Insert memory args
  unordered_map<int, memory> memory_args;
  memory_args[DNNL_ARG_SRC] = src_m_;
  memory_args[DNNL_ARG_DST] = dst_m_;

  // 4. Execute the primitive
  gelu_p_.execute(s, memory_args);

  // 5. unref tensors
  this->unref_tensors(input);
}

void GeluOperator::ForwardWithSparselib(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* dst_ptr = output[0];
  dst_ptr->mutable_data();
  std::vector<const void*> runtime_data = {input[0]->data(), dst_ptr->data()};
  eltwiseop_ker.execute(runtime_data);
  // unref tensors
  this->unref_tensors(input);
}

float GeluOperator::TuneMatmulRange(float gelu_bound, float err, float step) {
  float result = 0;
  if (gelu_bound < 0) {
    result = -3.0;
  } else if (gelu_bound >= 7.2) {
    result = gelu_bound;
  } else if (gelu_bound > 0 && gelu_bound < 7.2) {
    result = 7.2;
    while (std::abs(gelu_bound - jd::get_gelu(result)) > err) result -= step;
  }
  return result;
}

void GeluOperator::ForwardWithInt8LutAccTest(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto input_dt = input[0]->dtype();
  auto output_dt = output[0]->dtype();
  const float* min_p = static_cast<const float*>(input[1]->data());
  const float* max_p = static_cast<const float*>(input[2]->data());
  float gelu_scale = (max_p[0] - min_p[0]) / 255;
  float gelu_zp = -min_p[0] / gelu_scale;
  float matmul_lb = TuneMatmulRange(min_p[0], 0.0001, 0.00001);
  float matmul_ub = TuneMatmulRange(max_p[0], 0.0001, 0.00001);
  float matmul_scale = (matmul_ub - matmul_lb) / 255;
  float matmul_zp = matmul_lb + matmul_lb * matmul_scale;

  if (input_dt != "fp32" || output_dt != "u8") {
    LOG(ERROR) << "int8-lut test acc only support fp32 as input, u8 as output.";
  }
  // get num of input element
  auto input_shape = input[0]->shape();
  int element_num = 1;
  for (auto&& i : input_shape) element_num *= i;

  float* input_ptr = static_cast<float*>(input[0]->mutable_data());
  int8_t* output_s8ptr = static_cast<int8_t*>(output[0]->mutable_data());
  uint8_t* output_u8ptr = static_cast<uint8_t*>(output[0]->mutable_data());
  // turncat->u8 quantize->u8 dequantize->gelu->u8 quantize
  for (int i = 0; i < element_num; i++) {
    if (input_ptr[i] < -3.0) input_ptr[i] = -3.0;
    output_s8ptr[i] = jd::get_quantize(input_ptr[i], matmul_zp, matmul_scale);
    input_ptr[i] = jd::get_dequantize(output_s8ptr[i], matmul_zp, matmul_scale);
    input_ptr[i] = jd::get_gelu(input_ptr[i]);
    output_u8ptr[i] = jd::get_quantize(input_ptr[i], gelu_zp, gelu_scale);
  }

  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Gelu);
}  // namespace executor
