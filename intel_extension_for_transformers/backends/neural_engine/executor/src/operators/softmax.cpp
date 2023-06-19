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

#include "softmax.hpp"

#include "common.hpp"

namespace executor {

std::unordered_map<string, jd::data_type> Operator::type2sparsemem_ = {
    {"fp32", jd::data_type::fp32}, {"s32", jd::data_type::s32}, {"fp16", jd::data_type::fp16},
    {"u8", jd::data_type::u8},     {"s8", jd::data_type::s8},   {"bf16", jd::data_type::bf16}};

static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32}, {"s32", dnnl::memory::data_type::s32},
    {"fp16", dnnl::memory::data_type::f16}, {"u8", dnnl::memory::data_type::u8},
    {"s8", dnnl::memory::data_type::s8},    {"bf16", dnnl::memory::data_type::bf16}};

#if __AVX512F__
static inline __m512 i_poly(__m512 z, __m512 src_f32, const float c[]) {
  const auto c0 = _mm512_set1_ps(c[0]);
  const auto c1 = _mm512_set1_ps(c[1]);
  const auto c2 = _mm512_set1_ps(c[2]);

  auto y = _mm512_fmadd_ps(src_f32, c0, c1);
  y = _mm512_fmadd_ps(src_f32, y, c2);
  auto exp = _mm512_scalef_ps(y, z);

  return exp;
}

static inline __m512 i_exp(__m512 x) {
  const auto _log2e = _mm512_set1_ps(1.442695f);
  const auto ln2 = _mm512_set1_ps(0.693147180f);
  const float _c[] = {0.35815147f, 0.96963238f, 1.0f};

  auto z = _mm512_ceil_ps(_mm512_mul_ps(x, _log2e));
  auto q = _mm512_sub_ps(x, _mm512_mul_ps(z, ln2));

  return i_poly(z, q, _c);
}

template <int N>
void softmax_u8(void* out, void* in, const float oscale, const int64_t ld) {
  auto pin = reinterpret_cast<float*>(in);
  auto ld_16 = (ld + 15) / 16 * 16;

  // Actually, no need for alignment
  float* dout = reinterpret_cast<float*>(aligned_alloc(64, N * ld_16 * sizeof(float)));
  __m512 vmax[N];
  for (int i = 0; i < N; ++i) {
    vmax[i] =  _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  }

  // 1. get max
  int64_t d;
  for (d = 0; d < ld / 16 * 16; d += 16) {
    for (int i = 0; i < N; ++i) {
      auto src_f32 = _mm512_loadu_ps(pin + i * ld + d);
      vmax[i] = _mm512_max_ps(src_f32, vmax[i]);
      _mm512_storeu_ps(dout + i * ld_16 + d, src_f32);
    }
  }

  if (d < ld) {
    int res = ld - d;
    __mmask16 res_mask = (1 << res) - 1;
    // Initialize the input to a small value so that the sum of boundary value (e^x) is 0.
    // It fixes the ouput fluctuation in the paddding case.
    auto min_ps = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (int i = 0; i < N; ++i) {
      auto src_f32 = _mm512_mask_loadu_ps(min_ps, res_mask, pin + i * ld + d);
      vmax[i] = _mm512_max_ps(src_f32, vmax[i]);
      _mm512_storeu_ps(dout + i * ld_16 + d, src_f32);
    }
  }

  // 2. get exp and exp sum
  __m512 vsum[N];
  for (int i = 0; i < N; ++i) {
    vmax[i] = _mm512_set1_ps(_mm512_reduce_max_ps(vmax[i]));
    vsum[i] = _mm512_setzero_ps();
  }

  float* exp_out = reinterpret_cast<float*>(aligned_alloc(64, N * ld_16 * sizeof(float)));
  for (d = 0; d < ld_16; d += 16) {
    for (int i = 0; i < N; ++i) {
      auto src_f32 = _mm512_loadu_ps(dout + i * ld_16 + d);
      auto src_sub_max_f32 = _mm512_sub_ps(src_f32, vmax[i]);

      auto exp_src_sub_max_f32 = i_exp(src_sub_max_f32);
      _mm512_storeu_ps(exp_out + i * ld_16 + d, exp_src_sub_max_f32);
      vsum[i] = _mm512_add_ps(vsum[i], exp_src_sub_max_f32);
    }
  }

  // 3. quant
  auto voscale = _mm512_set1_ps(oscale);
  for (int i = 0; i < N; ++i) {
    vsum[i] = _mm512_div_ps(voscale, _mm512_set1_ps(_mm512_reduce_add_ps(vsum[i])));
  }

  auto __0 = _mm512_set1_ps(0.);
  auto __255 = _mm512_set1_ps(255.);

  auto pout = reinterpret_cast<uint8_t*>(out);
  for (d = 0; d < ld / 16 * 16; d += 16) {
    for (int i = 0; i < N; ++i) {
      auto exp_src_sub_max_f32 = _mm512_loadu_ps(exp_out + i * ld_16 + d);
      auto softmax_f32 =
          _mm512_mul_round_ps(exp_src_sub_max_f32, vsum[i], _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT);

      // Clip [0, 255]
      auto softmax_f32_clip_255 = _mm512_min_ps(softmax_f32, __255);
      auto softmax_f32_clip_0 = _mm512_max_ps(softmax_f32_clip_255, __0);
      auto softmax_s32 = _mm512_cvtps_epi32(softmax_f32_clip_0);
      _mm512_mask_cvtusepi32_storeu_epi8(pout + i * ld + d, 0xffff, softmax_s32);
    }
  }

  if (d < ld) {
    int res = ld - d;
    __mmask16 res_mask = (1 << res) - 1;
    for (int i = 0; i < N; ++i) {
      auto exp_src_sub_max_f32 = _mm512_loadu_ps(exp_out + i * ld_16 + d);
      auto softmax_f32 =
          _mm512_mul_round_ps(exp_src_sub_max_f32, vsum[i], _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT);
      // clip
      auto softmax_f32_clip_255 = _mm512_min_ps(softmax_f32, __255);
      auto softmax_f32_clip_0 = _mm512_max_ps(softmax_f32_clip_255, __0);
      auto softmax_s32 = _mm512_cvtps_epi32(softmax_f32_clip_0);
      _mm512_mask_cvtusepi32_storeu_epi8(pout + i * ld + d, res_mask, softmax_s32);
    }
  }

  aligned_free(exp_out);
  aligned_free(dout);
}

static inline void softmax_int_kernel(uint8_t* out, float* in, float oscale, int64_t ld, int l) {
  switch (l) {
    case 1:
      return softmax_u8<1>(out, in, oscale, ld);
    case 2:
      return softmax_u8<2>(out, in, oscale, ld);
    case 3:
      return softmax_u8<3>(out, in, oscale, ld);
    case 4:
      return softmax_u8<4>(out, in, oscale, ld);
    case 5:
      return softmax_u8<5>(out, in, oscale, ld);
    case 6:
      return softmax_u8<6>(out, in, oscale, ld);
    case 7:
      return softmax_u8<7>(out, in, oscale, ld);
    case 8:
      return softmax_u8<8>(out, in, oscale, ld);
    case 9:
      return softmax_u8<9>(out, in, oscale, ld);
    case 10:
      return softmax_u8<10>(out, in, oscale, ld);
    case 11:
      return softmax_u8<11>(out, in, oscale, ld);
    case 12:
      return softmax_u8<12>(out, in, oscale, ld);
    case 13:
      return softmax_u8<13>(out, in, oscale, ld);
    case 14:
      return softmax_u8<14>(out, in, oscale, ld);
    case 15:
      return softmax_u8<15>(out, in, oscale, ld);
    case 16:
      return softmax_u8<16>(out, in, oscale, ld);
  }

  auto l1 = l / 2;
  auto l2 = l - l1;

  auto pin = reinterpret_cast<float*>(in);
  auto pout = reinterpret_cast<uint8_t*>(out);

  softmax_int_kernel(pout, pin, oscale, ld, l1);
  softmax_int_kernel(pout + l1 * ld, pin + l1 * ld, oscale, ld, l2);
}
#endif

SoftmaxOperator::SoftmaxOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("axis");
  axis_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : -1;

  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  if (attrs_map.find("lut_softmax") != attrs_map.end()) {
    lut_optimization_ = true;
  }
}

void SoftmaxOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int input_size = input.size();
  dst_ = output[0];
  if (output.size() > 1) {
    dst_min_ = output[1];
    dst_max_ = output[2];
  }
  switch (input_size) {
    case 1: {
      src_ = input[0];
      break;
    }
    case 3: {
      src_ = input[0];
      dst_min_ = input[1];
      dst_max_ = input[2];
      break;
    }
    default: {
      // LOG(ERROR) << "Input size in Softmax is: " << input_size << ", not supported!";
      src_ = input[0];
      dst_min_ = input[1];
      dst_max_ = input[2];
    }
  }
}

void SoftmaxOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
#ifndef __AVX512F__
  LOG_IF(ERROR, output_dtype_ == "u8") << "Output dtype u8 in Softmax only supports AVX512!";
#endif

  MapTensors(input, output);
  is_dynamic_ = output.size() > 1;

  dst_->set_dtype(output_dtype_);
  if (!is_dynamic_ && dst_min_ != nullptr && dst_max_ != nullptr) {
    dst_scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
  }
}

void SoftmaxOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (output_dtype_ == "fp32" || output_dtype_ == "bf16" || is_dynamic_) {
    // dynamic quantization will calculate the softmax result with fp32 and then quantization in runtime.
    Reshape_dnnl(input, output);
  } else if (lut_optimization_) {
    if (input[0]->dtype() != "u8" && input[0]->dtype() != "s8") LOG(ERROR) << "LUT softmax only support int8 input dt.";
    Reshape_Sparselib(input, output);
  } else if (output_dtype_ == "u8") {
    Reshape_u8(input, output);
  } else {
    LOG(ERROR) << "Output dtype in Softmax is: " << output_dtype_ << ", not supported!";
  }
}

void SoftmaxOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (output_dtype_ == "fp32" || output_dtype_ == "bf16" || is_dynamic_) {
    Forward_dnnl(input, output);
  } else if (lut_optimization_) {
    Forward_Sparselib(input, output);
  } else if (output_dtype_ == "u8") {
#if __AVX512F__
    Forward_u8(input, output);
#else
    /* uint8_t* dst_u8 = */ static_cast<uint8_t*>(dst_->mutable_data());
    LOG(ERROR) << "AVX2 softmax not implemented!";
    this->unref_tensors(input);
#endif
  } else {
    LOG(ERROR) << "Output dtype in Softmax is: " << output_dtype_ << ", not supported!";
  }
}

void SoftmaxOperator::Reshape_dnnl(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const memory::dims& src_shape_origin = input[0]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  vector<int64_t> dst_shape = src_shape_origin;

  // 1.3 Get tensor's adjusted strides
  vector<int64_t> src_stride = GetStrides(src_shape_origin);
  memory::dims dst_stride = src_stride;

  // 1.4 Prepare memory descriptors
  memory::desc src_md(src_shape_origin, type2mem[src_->dtype()], src_stride);
  memory::desc dst_md(dst_shape, type2mem[is_dynamic_ ? "fp32" : dst_->dtype()], dst_stride);

  // 1.5 Set dst tensor shape
  output[0]->set_shape(dst_shape);
  if (is_dynamic_) {
    dst_min_->set_shape({1});
    dst_max_->set_shape({1});
  }

  //// Part2: Derive operator's format_any memory::desc and memory.
  // 2.1 Prepare format_any memory descriptors
  // 2.2 Prepare op descriptors
  if (axis_ == -1) {
    axis_ = src_shape_origin.size() - 1;
  }
  dnnl::softmax_forward::desc softmax_d(prop_kind::forward_inference, src_md, axis_);

  // 2.3 Prepare primitive descriptors (cached)
  dnnl::softmax_forward::primitive_desc softmax_pd(softmax_d, eng_);

  // 2.4 Prepare primitive objects (cached)
  softmax_p_ = dnnl::softmax_forward(softmax_pd);

  // 2.5 Prepare memory objects (cached)
  src_m_ = memory(src_md, eng_, DNNL_MEMORY_NONE);
  dst_m_ = memory(dst_md, eng_, DNNL_MEMORY_NONE);
}

void SoftmaxOperator::Reshape_u8(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<int64_t> input_shape = src_->shape();
  dst_->set_shape(input_shape);
}

void SoftmaxOperator::Reshape_Sparselib(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<int64_t> input_shape = src_->shape();
  dst_->set_shape(input_shape);

  src_desc_ = {input[0]->shape(), type2sparsemem_[input[0]->dtype()], jd::format_type::undef};
  dst_desc_ = {output[0]->shape(), type2sparsemem_[output[0]->dtype()], jd::format_type::undef};
  vector<jd::tensor_desc> ts_descs = {src_desc_, dst_desc_};
  std::unordered_map<std::string, std::string> op_attrs;
  float input_scale, output_scale;
  const float* min_p = static_cast<const float*>(input[1]->data());
  const float* max_p = static_cast<const float*>(input[2]->data());
  input_scale = (max_p[0] - min_p[0]) / 255;

  min_p = static_cast<const float*>(input[3]->data());
  max_p = static_cast<const float*>(input[4]->data());
  output_scale = (max_p[0] - min_p[0]) / 255;

  jd::postop_attr dequantize_attr(jd::data_type::s8, jd::postop_type::eltwise, jd::postop_alg::dequantize, 0, 0,
                                  input_scale);
  jd::postop_attr quantize_attr(jd::data_type::u8, jd::postop_type::eltwise, jd::postop_alg::quantize, 0, 0,
                                output_scale);
  if (lut_optimization_) {
    op_attrs["spec_type"] = "lut";
    op_attrs["vec_len"] = std::to_string(input[0]->shape().back());
    op_attrs["quant_factor"] = "dequantize" + std::to_string(input_scale) + "quantize" + std::to_string(output_scale);
    jd::operator_desc op_desc(jd::kernel_kind::softmax, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                              ts_descs, op_attrs, {dequantize_attr, quantize_attr});
    jd::softmax_desc softmax_desc(op_desc);
    softmax_ker_ = jd::softmax(softmax_desc);
  }
}

vector<vector<string>> SoftmaxOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<vector<string>> inplace_pairs;
  // skip inplace in debug mode
  if (this->get_execution_mode() == ExecutionMode::DEBUG) {
    return inplace_pairs;
  }
  // input[0] -> output[0]
  bool forward_dnnl = (output_dtype_ == "fp32" || output_dtype_ == "bf16" || is_dynamic_) ? true : false;
  if (forward_dnnl && input.size() == 1 && input[0] != nullptr && input[0]->left_life() == 1 &&
      input[0]->size() >= output[0]->size()) {
    inplace_pairs.emplace_back(vector<string>({input[0]->name(), output[0]->name()}));
  }
  return inplace_pairs;
}

void SoftmaxOperator::Forward_dnnl(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src_data = input[0]->data();
  // when change data value please use mutable_data
  // Inplace Op
  // create a dynamic quantization output with fp32.
  Tensor fp32_res;
  Tensor* dst_ptr = output[0];
  if (is_dynamic_) {
    fp32_res = *dst_;
    fp32_res.set_dtype("fp32");
    dst_ptr = &fp32_res;
  }
  vector<Tensor*> inputs(input);
  if (input.size() == 1 && input[0] != nullptr && input[0]->left_life() == 1 && input[0]->size() >= dst_ptr->size() &&
      this->get_execution_mode() != ExecutionMode::DEBUG) {
    void* input_ptr = input[0]->mutable_data();
    input[0]->unref_data(true);
    dst_ptr->set_data(input_ptr);
    inputs = {};
  }
  auto dst_data = dst_ptr->mutable_data();

  // 1. Prepare memory objects with data_ptr
  dnnl::stream s(eng_);
  src_m_.set_data_handle(const_cast<void*>(src_data), s);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

  // 2. Reorder the data when the primitive memory and user memory are different
  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC] = src_m_;
  memory_args_[DNNL_ARG_DST] = dst_m_;

  // 4. Execute the primitive
  softmax_p_.execute(s, memory_args_);

  // 5. Reorder the data of dst memory (When it is format_any)
  // 6. unref tensors
  this->unref_tensors(inputs);

  if (output.size() > 1) {
    // quantize the fp32 result of softmax
    runtime_minmax(reinterpret_cast<float*>(fp32_res.mutable_data()), fp32_res.size(),
                   reinterpret_cast<float*>(dst_min_->mutable_data()),
                   reinterpret_cast<float*>(dst_max_->mutable_data()));
    // quantize
    if (output_dtype_ == "u8") {
      auto scales = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
      // memcpy(dst_max_->mutable_data(), scales.data(), dst_max_->size() * sizeof(float));
#if __AVX512F__
      Quantize_avx512(fp32_res.size(), dst_->dtype(), fp32_res.data(), static_cast<const float*>(dst_min_->data()),
                      scales, dst_->mutable_data());
#else
      Quantize(fp32_res.size(), dst_->dtype(), fp32_res.data(), static_cast<const float*>(dst_min_->data()), scales,
               dst_->mutable_data());
#endif
      float* dst_max_data = reinterpret_cast<float*>(dst_max_->mutable_data());
      *dst_max_data = 1.0 / scales[0];
    } else {
      LOG(ERROR) << "Output dtype in Softmax is: " << output_dtype_ << ", not supported!";
    }
    fp32_res.unref_data();
  }
}
#if __AVX512F__
void SoftmaxOperator::Forward_u8(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  float* src_f32 = static_cast<float*>(src_->mutable_data());
  uint8_t* dst_u8 = static_cast<uint8_t*>(dst_->mutable_data());

  const vector<int64_t> src_shape = src_->shape();

  if (axis_ < 0) axis_ = src_shape.size() + axis_;
  const int64_t dim0 = std::accumulate(src_shape.begin(), src_shape.begin() + axis_ - 1, 1, std::multiplies<int64_t>());
  const int64_t dim1 = *(src_shape.begin() + axis_ - 1);
  const int64_t dim2 = std::accumulate(src_shape.begin() + axis_, src_shape.end(), 1, std::multiplies<int64_t>());

#pragma omp parallel for
  for (auto d0 = 0; d0 < dim0; ++d0) {
    float* p_in = src_f32 + d0 * dim1 * dim2;
    uint8_t* p_out = dst_u8 + d0 * dim1 * dim2;
    softmax_int_kernel(p_out, p_in, dst_scales_[0], dim2, dim1);
  }

  // unref tensors
  this->unref_tensors(input);
}
#endif

void SoftmaxOperator::Forward_Sparselib(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* dst_ptr = output[0];
  if (lut_optimization_) {
    std::vector<const void*> runtime_data = {input[0]->data(), dst_ptr->data()};
    softmax_ker_.execute(runtime_data);
  }
  this->unref_tensors(input);
}

void SoftmaxOperator::RuntimeMinmax(dnnl::stream& s) {
  // use onednn reduction calculate min/max
  vector<int64_t> reduce_shape(dst_->shape().size(), 1);
  vector<int64_t> reduce_stride = GetStrides(reduce_shape);
  memory::desc dst_md(reduce_shape, memory::data_type::f32, reduce_stride);
  memory reduce_min(dst_md, eng_);
  memory reduce_max(dst_md, eng_);
  reduce_min.set_data_handle(dst_min_->mutable_data());
  reduce_max.set_data_handle(dst_max_->mutable_data());
  dnnl::reduction::desc reduce_min_d(algorithm::reduction_min, dst_m_.get_desc(), dst_md, 0.f, 0.f);
  dnnl::reduction::primitive_desc reduce_min_pd(reduce_min_d, eng_);
  dnnl::reduction(reduce_min_pd).execute(s, {{DNNL_ARG_SRC, dst_m_}, {DNNL_ARG_DST, reduce_min}});
  dnnl::reduction::desc reduce_max_d(algorithm::reduction_max, dst_m_.get_desc(), dst_md, 0.f, 0.f);
  dnnl::reduction::primitive_desc reduce_max_pd(reduce_max_d, eng_);
  dnnl::reduction(reduce_max_pd).execute(s, {{DNNL_ARG_SRC, dst_m_}, {DNNL_ARG_DST, reduce_max}});
}

void SoftmaxOperator::AdaptAttrs(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage) {
  if (stage == "in") {
    output[0]->set_tensor_format(input[0]->tensor_format());
  } else if (stage == "out") {
    return;
  } else {
    LOG(WARNING) << "Wrong stage parameter, should be in or out...";
  }
}
REGISTER_OPERATOR_CLASS(Softmax);
}  // namespace executor
