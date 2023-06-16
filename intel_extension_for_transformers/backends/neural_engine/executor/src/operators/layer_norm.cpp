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

#include "layer_norm.hpp"

#include "common.hpp"

namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32}, {"s32", dnnl::memory::data_type::s32},
    {"fp16", dnnl::memory::data_type::f16}, {"u8", dnnl::memory::data_type::u8},
    {"s8", dnnl::memory::data_type::s8},    {"bf16", dnnl::memory::data_type::bf16}};

LayerNormOperator::LayerNormOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf), weight_cached_(false) {
  auto attrs_map = operator_conf_->attributes();
  epsilon_ = StringToNum<float>(attrs_map["epsilon"]);
  auto iter = attrs_map.find("transpose_mode");
  if (iter != attrs_map.end()) {
    transpose_mode_ = true;
  }
  if (attrs_map.find("quantize_fuse") != attrs_map.end()) {
    quantize_fuse_ = true;
  }
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  iter = attrs_map.find("reshape");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&reshape_, attrs_map["reshape"], ",");
  }
  iter = attrs_map.find("reshape_dims");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&reshape_dims_, attrs_map["reshape_dims"], ",");
  }
  iter = attrs_map.find("mul");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&mul_, attrs_map["mul"], ",");
  }
}

void LayerNormOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (!transpose_mode_ || input[0]->dtype() != "fp32") {
    PreparewithOnednn(input, output);
  }
}

void LayerNormOperator::DstReshapeFusion(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (!reshape_.empty()) {
    vector<int64_t> pre_dst_shape;
    vector<int64_t> ref_shape;
    if (!reshape_dims_.empty()) {
      ref_shape = input.back()->shape();
    }
    pre_dst_shape = GetDstShape(reshape_, output[0]->size(), ref_shape, reshape_dims_);
    vector<int64_t> dst_shape(pre_dst_shape);
    if (!mul_.empty()) {
      dst_shape.clear();
      int j = 0;
      int64_t mul_size = 1;
      for (int i = 0; i < mul_.size(); ++i) mul_size *= pre_dst_shape[mul_[i]];
      for (int i = 0; i < pre_dst_shape.size(); ++i) {
        if (j < mul_.size() && i == mul_[j]) {
          if (j == 0) dst_shape.push_back(mul_size);
          j++;
        } else {
          dst_shape.push_back(pre_dst_shape[i]);
        }
      }
    }
    output[0]->set_shape(dst_shape);
  }
}

void LayerNormOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (transpose_mode_) {
    ReshapewithTransMode(input, output);
  } else {
    ReshapewithOnednn(input, output);
  }
  DstReshapeFusion(input, output);
}

void LayerNormOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (transpose_mode_) {
#ifdef __AVX512F__
    ForwardwithTransMode(input, output);
#endif
  } else {
    ForwardwithOnednn(input, output);
  }
}
#ifdef WITH_SPARSELIB
void LayerNormOperator::ReshapewithTransMode(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<int64_t> src_shape = input[0]->shape();
  src_desc_ = {src_shape, jd::data_type::fp32, jd::format_type::ba};
  jd::tensor_desc affine_desc = {{}, jd::data_type::fp32, jd::format_type::ba};
  jd::data_type dst_dt;
  if (output[0]->dtype() == "fp32") dst_dt = jd::data_type::fp32;
  if (output[0]->dtype() == "s8") dst_dt = jd::data_type::s8;
  if (output[0]->dtype() == "u8") dst_dt = jd::data_type::u8;
  dst_desc_ = {output[0]->shape(), dst_dt, jd::format_type::ba};

  vector<jd::tensor_desc> ts_descs = {src_desc_, dst_desc_, affine_desc};
  std::unordered_map<std::string, std::string> op_attrs_;
  op_attrs_["spec_type"] = "normal";
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(src_shape);

  // for kernel hasing.
  string src_shape_str;
  for (auto&& i : src_shape) {
    src_shape_str += std::to_string(i);
    src_shape_str += "x";
  }
  op_attrs_["matrix_shape"] = src_shape_str;
  vector<jd::postop_attr> postops;
  if (quantize_fuse_) {
    float zp = 0, scale = 1;
    jd::postop_attr u8_quantize = {jd::data_type::u8, jd::postop_type::eltwise, jd::postop_alg::quantize, zp, 0, scale};
    postops.push_back(u8_quantize);
    op_attrs_["postop_list"] = "s8quant+" + std::to_string(zp) + "+" + std::to_string(scale);
  }

  jd::operator_desc op_desc(jd::kernel_kind::layernorm_ba, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs_, postops);
  jd::layernorm_ba_desc layernorm_ba_desc(op_desc);
  layernorm_ba_ker = jd::layernorm_ba(layernorm_ba_desc);
}
#if __AVX512F__
void LayerNormOperator::ForwardwithTransMode(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // Inplace op.
  Tensor* dst_ptr = output[0];
  dst_ptr->mutable_data();
  std::vector<const void*> runtime_data = {input[0]->data(), dst_ptr->data(), input[1]->data(), input[2]->data()};
  layernorm_ba_ker.execute(runtime_data);
  // unref tensors
  this->unref_tensors(input);
}
#endif

#else
void LayerNormOperator::ReshapewithTransMode(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  LOG(ERROR) << "Sparse lib is not loaded!\n";
}
void LayerNormOperator::ForwardwithTransMode(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  LOG(ERROR) << "Sparse lib is not loaded!\n";
}
#endif

// The ONEDNN layer_norm primitive supports the following combinations of data types:
// src0 f32, bf16, f16, u8, s8, dst: f32, bf16, f16, u8, s8
// In-place mode requires the dst and src data types to be the same.
// Different data types will unavoidably lead to correctness issues.
// Mean, Variance and ScaleShift data types are always fp32 and independent of src or dst data types.
void LayerNormOperator::PreparewithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  LOG_IF(FATAL, (output_dtype_ == "s32" || input[0]->dtype() == "s32")) << "Unsupported dtype s32...";
  LOG_IF(FATAL, (input[1]->dtype() != "fp32" || input[2]->dtype() != "fp32")) <<
        "Onednn only support fp32 scale and shift...";
  if (output_dtype_.empty()) {
    output_dtype_ = input[0]->dtype();
  }
  output[0]->set_dtype(output_dtype_);
}

void LayerNormOperator::ReshapewithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const memory::dims& src_shape_origin = input[0]->shape();
  const memory::dims& gamma_shape_origin = input[1]->shape();
  const memory::dims& beta_shape_origin = input[2]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  memory::dims dst_shape = src_shape_origin;
  // scale & shift shape
  int64_t scale_size = gamma_shape_origin.back();
  memory::dims scale_shift_shape = {2, scale_size};

  // 1.3 Get tensor's adjusted strides
  memory::dims src_stride = GetStrides(src_shape_origin);
  memory::dims dst_stride = src_stride;

  // 1.4 Prepare memory descriptors
  memory::desc src_md(src_shape_origin, type2mem[input[0]->dtype()], src_stride);
  memory::desc scale_shift_md(scale_shift_shape, dnnl::memory::data_type::f32, memory::format_tag::nc);
  memory::desc dst_md(dst_shape, type2mem[output[0]->dtype()], dst_stride);

  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);

  //// Part2: Derive operator's format_any memory::desc and memory.
  // 2.1 Prepare format_any memory descriptors
  // 2.2 Prepare op descriptors
  dnnl::layer_normalization_forward::desc lnorm_d(prop_kind::forward_inference, src_md, epsilon_,
                                                  dnnl::normalization_flags::use_scale_shift);

  // 2.3 Prepare primitive descriptors
  dnnl::layer_normalization_forward::primitive_desc lnorm_pd(lnorm_d, eng_);

  // 2.4 Prepare primitive objects (cached)
  lnorm_p_ = dnnl::layer_normalization_forward(lnorm_pd);

  // 2.5 Prepare memory objects (cached)
  src_m_ = memory(src_md, eng_, DNNL_MEMORY_NONE);
  dst_m_ = memory(dst_md, eng_, DNNL_MEMORY_NONE);
  memory mean_m(lnorm_pd.mean_desc(), eng_);
  memory variance_m(lnorm_pd.variance_desc(), eng_);

  if (!weight_cached_) {
    scale_shift_m = memory(scale_shift_md, eng_);
    if (input[1]->is_shared() && input[2]->is_shared()) {
      int64_t scale_shift_size = scale_shift_m.get_desc().get_size();
      string scale_shift_name = input[1]->name() + input[2]->name();
      void* scale_shift_shm_ptr =
          MemoryAllocator::ManagedShm().find_or_construct<char>(scale_shift_name.c_str())[scale_shift_size](0);
      scale_shift_m.set_data_handle(scale_shift_shm_ptr);
    }
    void* scale_shift_buf = scale_shift_m.get_data_handle();
    const auto& gamma_data = input[1]->data();
    const auto& beta_data = input[2]->data();
    std::memcpy(scale_shift_buf, gamma_data, sizeof(float) * scale_size);
    std::memcpy(reinterpret_cast<float*>(scale_shift_buf) + scale_size, beta_data, sizeof(float) * scale_size);
    weight_cached_ = true;
  }

  // 2.6 Prepare memory args (cached)
  memory_args_[DNNL_ARG_MEAN] = mean_m;
  memory_args_[DNNL_ARG_VARIANCE] = variance_m;
  memory_args_[DNNL_ARG_SCALE_SHIFT] = scale_shift_m;
}

vector<vector<string>> LayerNormOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<vector<string>> inplace_pairs;
  // skip inplace in debug mode
  if (this->get_execution_mode() == ExecutionMode::DEBUG) {
    return inplace_pairs;
  }
  // input[0] -> output[0]
  if (!transpose_mode_ && input.size() == 3 && input[0] != nullptr && input[0]->left_life() == 1 &&
      input[0]->size() >= output[0]->size() && input[0]->dtype() == output[0]->dtype()) {
    inplace_pairs.emplace_back(vector<string>({input[0]->name(), output[0]->name()}));
  }
  return inplace_pairs;
}

void LayerNormOperator::ForwardwithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src_data = input[0]->data();
  // when change data value please use mutable_data
  // Inplace Op
  Tensor* dst_ptr = output[0];
  vector<Tensor*> inputs(input);
  if (input.size() == 3 && input[0] != nullptr && input[0]->left_life() == 1 &&
      input[0]->size() >= dst_ptr->size() && input[0]->dtype() == output[0]->dtype() &&
      this->get_execution_mode() != ExecutionMode::DEBUG) {
    void* input_ptr = input[0]->mutable_data();
    input[0]->unref_data(true);
    dst_ptr->set_data(input_ptr);
    inputs = {input[1], input[2]};
  }
  auto dst_data = output[0]->mutable_data();

  // 1. Prepare memory objects with data_ptr
  dnnl::stream s(eng_);
  src_m_.set_data_handle(const_cast<void*>(src_data), s);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

  // 2. Reorder the data when the primitive memory and user memory are different
  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC] = src_m_;
  memory_args_[DNNL_ARG_DST] = dst_m_;

  // 4. Execute the primitive
  lnorm_p_.execute(s, memory_args_);

  // 5. Reorder the data of dst memory (When it is format_any)
  // 6. unref tensors
  this->unref_tensors(inputs);
}

REGISTER_OPERATOR_CLASS(LayerNorm);
}  // namespace executor
