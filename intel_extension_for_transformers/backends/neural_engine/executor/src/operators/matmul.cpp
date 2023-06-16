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

#include "matmul.hpp"

#include "model.hpp"
#include "operator_registry.hpp"
namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32}, {"int32", dnnl::memory::data_type::s32},
    {"s32", dnnl::memory::data_type::s32},  {"fp16", dnnl::memory::data_type::f16},
    {"u8", dnnl::memory::data_type::u8},    {"s8", dnnl::memory::data_type::s8},
    {"bf16", dnnl::memory::data_type::bf16}};

static unordered_map<string, jd::data_type> type2sparsemem{
    {"fp32", jd::data_type::fp32}, {"s32", jd::data_type::s32}, {"fp16", jd::data_type::fp16},
    {"u8", jd::data_type::u8},     {"s8", jd::data_type::s8},   {"bf16", jd::data_type::bf16}};

MatmulOperator::MatmulOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf),
      src0_perm_({}),
      src1_perm_({}),
      dst_perm_({}),
      output_scale_(1.),
      format_any_(true),
      cache_weight_(false) {
  auto attrs_map = operator_conf_->attributes();

  auto iter = attrs_map.find("src0_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src0_perm_, attrs_map["src0_perm"], ",");
  }
  iter = attrs_map.find("src1_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src1_perm_, attrs_map["src1_perm"], ",");
  }
  iter = attrs_map.find("dst_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&dst_perm_, attrs_map["dst_perm"], ",");
  }
  iter = attrs_map.find("output_scale");
  if (iter != attrs_map.end()) {
    output_scale_ = StringToNum<float>(attrs_map["output_scale"]);
  }
  iter = attrs_map.find("format_any");
  if (iter != attrs_map.end()) {
    format_any_ = attrs_map["format_any"] == "True" || attrs_map["format_any"] == "true";
  }
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  iter = attrs_map.find("cache_weight");
  if (iter != attrs_map.end()) {
    cache_weight_ = attrs_map["cache_weight"] == "True" || attrs_map["cache_weight"] == "true";
  }
  iter = attrs_map.find("reshape");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&reshape_, attrs_map["reshape"], ",");
  }
  iter = attrs_map.find("reshape_dims");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&reshape_dims_, attrs_map["reshape_dims"], ",");
  }
  iter = attrs_map.find("append_op");
  append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
  binary_add_ = (iter != attrs_map.end() && iter->second == "binary_add") ? true : false;
  gelu_erf_ = (iter != attrs_map.end() && iter->second == "gelu_erf") ? true : false;
  gelu_tanh_ = (iter != attrs_map.end() && iter->second == "gelu_tanh") ? true : false;
  tanh_ = (iter != attrs_map.end() && iter->second == "tanh") ? true : false;
  append_eltwise_ = gelu_erf_ || gelu_tanh_ || tanh_;
  append_op_ = (iter != attrs_map.end()) ? iter->second : "";
  DLOG(INFO) << "append_op: " << append_op_;
}

MatmulOperator::~MatmulOperator() {}

void MatmulOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int input_size = input.size();
  if (!reshape_dims_.empty()) {
    input_size -= 1;
  }
  dst_ = output[0];
  if (output.size() > 1) {
    dst_min_ = output[1];
    dst_max_ = output[2];
  }
  switch (input_size) {
    case 2: {
      src0_ = input[0];
      src1_ = input[1];
      break;
    }
    case 3: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = (append_sum_ || binary_add_) ? nullptr : input[2];
      post_ = (append_sum_ || binary_add_) ? input[2] : nullptr;
      has_bias_ = !(append_sum_ || binary_add_);
      break;
    }
    case 4: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = input[2];
      post_ = (append_sum_ || binary_add_) ? input[3] : nullptr;
      has_bias_ = true;
      break;
    }
    case 6: {
      src0_ = input[0];
      src1_ = input[1];
      src0_min_ = input[2];
      src0_max_ = input[3];
      src1_min_ = input[4];
      src1_max_ = input[5];
      break;
    }
    case 7: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = (append_sum_ || binary_add_) ? nullptr : input[2];
      post_ = (append_sum_ || binary_add_) ? input[2] : nullptr;
      src0_min_ = input[3];
      src0_max_ = input[4];
      src1_min_ = input[5];
      src1_max_ = input[6];
      has_bias_ = !(append_sum_ || binary_add_);
      break;
    }
    case 8: {
      if (append_sum_ || binary_add_) {
        // dynamic quantization
        src0_ = input[0];
        src1_ = input[1];
        bias_ = input[2];
        post_ = input[3];
        src0_min_ = input[4];
        src0_max_ = input[5];
        src1_min_ = input[6];
        src1_max_ = input[7];
        has_bias_ = true;
      } else {
        // static quantization
        src0_ = input[0];
        src1_ = input[1];
        src0_min_ = input[2];
        src0_max_ = input[3];
        src1_min_ = input[4];
        src1_max_ = input[5];
        dst_min_ = input[6];
        dst_max_ = input[7];
        has_bias_ = false;
      }
      break;
    }
    case 9: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = (append_sum_ || binary_add_) ? nullptr : input[2];
      post_ = (append_sum_ || binary_add_) ? input[2] : nullptr;
      src0_min_ = input[3];
      src0_max_ = input[4];
      src1_min_ = input[5];
      src1_max_ = input[6];
      dst_min_ = input[7];
      dst_max_ = input[8];
      has_bias_ = !(append_sum_ || binary_add_);
      break;
    }
    case 10: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = input[2];
      post_ = (append_sum_ || binary_add_) ? input[3] : nullptr;
      src0_min_ = input[4];
      src0_max_ = input[5];
      src1_min_ = input[6];
      src1_max_ = input[7];
      dst_min_ = input[8];
      dst_max_ = input[9];
      has_bias_ = true;
      break;
    }
  }
}

#if (__AVX512F__ || __AVX2__) && !__AMXINT8__
void MatmulOperator::SetTransposeMode() {
  if (dst_->dtype() == "fp32" && binary_add_ && src0_->dtype() == "fp32") {
    vector<int64_t> src0_perm_transpose{2, 0, 3, 1};
    vector<int64_t> src1_perm_transpose{2, 0, 1, 3};
    transpose_mode_ = (src0_perm_ == src0_perm_transpose) && (src1_perm_ == src1_perm_transpose);
  } else if (dst_->dtype() == "u8") {
    vector<int64_t> dst_perm_transpose{1, 3, 0, 2};
    vector<int64_t> src1_perm_transpose{2, 0, 3, 1};
    transpose_mode_ = (dst_perm_ == dst_perm_transpose) && (src1_perm_ == src1_perm_transpose);
  }
}
#elif __AVX2__
void MatmulOperator::SetTransposeMode() { assert(false); }  // avx2 not implemented
#endif

void MatmulOperator::DstReshapeFusion(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (!reshape_.empty()) {
    vector<int64_t> pre_dst_shape;

    vector<int64_t> reshape(reshape_);
    if (output[0]->shape().size() == 5 && output[0]->tensor_format() == TensorFormat::BmHnHsBbS) {
      int64_t micro_bs = output[0]->shape()[0];
      reshape.insert(reshape.begin(), micro_bs);
    }

    vector<int64_t> ref_shape;
    if (!reshape_dims_.empty()) {
      ref_shape = input.back()->shape();
      vector<int64_t> dst_shape = GetDstShape(reshape, output[0]->size(), ref_shape, reshape_dims_);
      output[0]->set_shape(dst_shape);
    } else {
      vector<int64_t> dst_shape = GetDstShape(reshape, output[0]->size(), pre_dst_shape, pre_dst_shape);
      output[0]->set_shape(dst_shape);
    }
  }
}

void MatmulOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  dst_->set_dtype(output_dtype_);
  is_dynamic_ =
      output.size() > 1 || (src0_min_ != nullptr && src0_min_->raw_data() == nullptr && !src0_min_->is_shared());
  if (is_dynamic_) {
    DLOG(INFO) << this->name() << " is DYNAMIC!!!";
#ifdef _WIN32
    LOG(ERROR) << "dynamic quantization did NOT support windows now!!!";
    throw std::string("Windows");
#endif
  }
  if (!is_dynamic_ && (output_scale_ != 1.f || src0_min_ != nullptr || src1_min_ != nullptr)) {
    int ic_dim = 0;
    vector<float> rescales;
    if (src0_min_ != nullptr && src1_max_ != nullptr) {
      ic_dim = src1_min_->size() > 1 ? 0 | (1 << 1) : 0;
      vector<float> src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
      vector<float> src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
      if (dst_min_ != nullptr)
        dst_scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
      rescales = GetRescales(src0_scales, src1_scales, dst_scales_, dst_->dtype(), append_eltwise_);
    } else {
      rescales = vector<float>(1, 1.f);
    }
    if (output_scale_ != 1.f) {
      for (int i = 0; i < rescales.size(); i++) {
        rescales[i] *= output_scale_;
      }
    }
    attr_.set_output_scales(ic_dim, rescales);
    rescales_ = rescales;
  }
  SetTransposeMode();
}

void MatmulOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (transpose_mode_) {
    ReshapewithTransMode(input, output);
  } else {
    ReshapewithOnednn(input, output);
  }
}

void MatmulOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (transpose_mode_) {
#if __AVX512F__
    ForwardwithTransMode(input, output);
#endif
  } else {
    ForwardwithOnednn(input, output);
  }
}

// 1. Create primitive
void MatmulOperator::ReshapewithTransMode(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<int64_t> src0_shape_origin = src0_->shape();
  vector<int64_t> src1_shape_origin = src1_->shape();
  vector<int64_t> src0_shape = GetShapes(src0_shape_origin, src0_perm_);
  vector<int64_t> src1_shape = GetShapes(src1_shape_origin, src1_perm_);
  vector<int64_t> src0_stride = GetStrides(src0_shape_origin, src0_perm_);
  vector<int64_t> src1_stride = GetStrides(src1_shape_origin, src1_perm_);

  vector<int64_t> dst_shape_origin = src0_shape;
  dst_shape_origin.back() = src1_shape.back();
  dst_->set_shape(dst_shape_origin);
  if (!dst_perm_.empty()) {
    vector<int64_t> dst_shape_after = GetShapes(dst_shape_origin, dst_perm_);
    dst_->set_shape(dst_shape_after);
  }
  std::unordered_map<std::string, std::string> attrs{};
  attrs["alpha"] = std::to_string(output_scale_);
  attrs["beta"] = "1";
  src0_desc_ = {src0_shape_origin, type2sparsemem[src0_->dtype()], jd::format_type::ab};
  src1_desc_ = {src1_shape_origin, type2sparsemem[src1_->dtype()], jd::format_type::ab};
  dst_desc_ = {dst_->shape(), type2sparsemem[dst_->dtype()], jd::format_type::ab};
  binary_desc_ = {{}, jd::data_type::fp32, jd::format_type::ab};
  if (binary_add_) {
    binary_desc_ = {dst_->shape(), type2sparsemem[post_->dtype()], jd::format_type::ab};
  }
  scale_desc_ = {{static_cast<int64_t>(rescales_.size())}, jd::data_type::fp32, jd::format_type::a};
  zp_desc_ = {{static_cast<int64_t>(dst_scales_.size())}, jd::data_type::fp32, jd::format_type::a};
  vector<jd::tensor_desc> ts_descs;
  if (dst_->dtype() == "u8") {
    ts_descs = {src0_desc_, src1_desc_, dst_desc_, binary_desc_, scale_desc_, zp_desc_};
    ouput_zp_ = -1 * static_cast<const float*>(dst_min_->data())[0] * dst_scales_[0];
  } else {
    ts_descs = {src0_desc_, src1_desc_, dst_desc_, binary_desc_};
  }

  jd::operator_desc op_desc(jd::kernel_kind::transpose_matmul, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, attrs);
  jd::transpose_matmul_desc matmul_desc(op_desc);
  transpose_matmul_ = jd::transpose_matmul(matmul_desc);
  DstReshapeFusion(input, output);
}
#if __AVX512F__
void MatmulOperator::ForwardwithTransMode(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  void* dst_data = dst_->mutable_data();
  std::vector<const void*> runtime_data = {src0_->data(),
                                           src1_->data(),
                                           dst_data,
                                           binary_add_ ? post_->data() : nullptr,
                                           dst_->dtype() == "u8" ? &rescales_[0] : nullptr,
                                           dst_->dtype() == "u8" ? &ouput_zp_ : nullptr};

  transpose_matmul_.execute(runtime_data);
  this->unref_tensors(input);
}
#endif
// 1. Create primitive
void MatmulOperator::ReshapewithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1 Transpose tensor shape and get it
  vector<int64_t> src0_shape_origin = src0_->shape();
  vector<int64_t> src1_shape_origin = src1_->shape();

  vector<int64_t> src0_shape = GetShapes(src0_shape_origin, src0_perm_);
  vector<int64_t> src1_shape = GetShapes(src1_shape_origin, src1_perm_);
  vector<int64_t> src0_stride = GetStrides(src0_shape_origin, src0_perm_);
  vector<int64_t> src1_stride = GetStrides(src1_shape_origin, src1_perm_);

  vector<int64_t> bias_shape;
  vector<int64_t> bias_stride;
  if (has_bias_) bias_shape = bias_->shape();

  // 1.2 malloc tensor for output
  // src0_: M*K, src1_: K*N, DST: M*N
  vector<int64_t> dst_shape_origin = src0_shape;
  dst_shape_origin.back() = src1_shape.back();

  // Sub-step3: fused post transpose, notice it's different that
  // pre transpose will use the tranposed shape and stride, it's straight forward
  // post transpose will use origin shape and that means the dst buffer in matmul
  // is a buffer transposed back from dst_perm(understand tranpose to and transpose back)
  // pre_transpose: src_buffer -> pre_transpose -> target_buffer in matmul
  // post_transpose: target_buffer in matmul<- post transpose <-dst_buffer
  vector<int64_t> dst_shape = GetShapes(dst_shape_origin, dst_perm_);
  vector<int64_t> reverse_perm = ReversePerm(dst_perm_);
  vector<int64_t> dst_stride = GetStrides(dst_shape, reverse_perm);
  if (has_bias_ && bias_shape.size() != dst_shape.size()) {
    bias_shape = vector<int64_t>(dst_shape.size(), 1);
    bias_shape.back() = bias_->shape().back();
    bias_stride = GetStrides(bias_shape, reverse_perm);
  }
  // 1.4 Prepare memory descriptors
  memory::desc any_src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], memory::format_tag::any);
  memory::desc src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], src0_stride);

  memory::desc any_src1_md = memory::desc(src1_shape, type2mem[src1_->dtype()], memory::format_tag::any);
  memory::desc src1_md = memory::desc(src1_shape, type2mem[src1_->dtype()], src1_stride);

  memory::desc any_dst_md, dst_md;
  if (is_dynamic_) {
    // matmul output dtype in dynamic quantization should be fp32 and then manually quantize to u8/s8.
    any_dst_md = memory::desc(dst_shape_origin, type2mem["fp32"], memory::format_tag::any);
    dst_md = memory::desc(dst_shape_origin, type2mem["fp32"], dst_stride);
  } else {
    any_dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], memory::format_tag::any);
    dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);
  }

  memory::desc bias_md;
  memory::desc any_bias_md;
  if (has_bias_) {
    bias_md = memory::desc(bias_shape, type2mem[bias_->dtype()], bias_stride);
    any_bias_md = memory::desc(bias_shape, type2mem[bias_->dtype()], memory::format_tag::any);
  }

  // 1.5 Set dst shape and strides
  dst_->set_shape(dst_shape);
  if (output.size() > 1) {
    dst_min_->set_shape({1});
    dst_max_->set_shape({1});
  }

  if (cache_weight_) {
    src1_md = dnnl::memory::desc(src1_shape, type2mem[src1_->dtype()], memory::format_tag::any);
  }

  // 2.2 Prepare op descriptors
  dnnl::matmul::desc matmul_d =
      has_bias_ ? dnnl::matmul::desc(src0_md, src1_md, bias_md, dst_md) : dnnl::matmul::desc(src0_md, src1_md, dst_md);

  if (format_any_) {
    matmul_d = has_bias_ ? dnnl::matmul::desc(any_src0_md, any_src1_md, any_bias_md, any_dst_md)
                         : dnnl::matmul::desc(any_src0_md, any_src1_md, any_dst_md);
  }

  // 2.3 Prepare primitive descriptors (cached)
  vector<float> src0_scales;
  vector<float> src1_scales;
  vector<float> dst_scales;
  vector<float> rescales;
  dnnl::post_ops po;
  int ic_dim = 0;  // matmul only support per_tensor now
  if (is_dynamic_ && (output_scale_ != 1.f || src0_min_ != nullptr || src1_min_ != nullptr)) {
    if (src0_min_ != nullptr && src1_max_ != nullptr) {
      attr_.set_output_scales(ic_dim, {DNNL_RUNTIME_F32_VAL});
      scale_f32_mem_ = memory({{1}, memory::data_type::f32, {1}}, eng_, DNNL_MEMORY_NONE);
      zp_src0_mem_ = memory({{1}, memory::data_type::s32, {1}}, eng_, DNNL_MEMORY_NONE);
      // need zero point when src0 is u8
      if (src0_->dtype() == "u8") {
        attr_.set_zero_points(DNNL_ARG_SRC, ic_dim, {DNNL_RUNTIME_S32_VAL});
      }
    }
  }
  if (append_sum_) {
    float beta = 1.0;
    po.append_sum(beta);
  }
  if (gelu_erf_) {
    float op_scale = 1.0;
    float op_alpha = 0.0;
    float op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_gelu_erf, op_alpha, op_beta);
  }
  if (gelu_tanh_) {
    float op_scale = 1.0;
    float op_alpha = 0.0;
    float op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_gelu_tanh, op_alpha, op_beta);
  }
  if (tanh_) {
    auto op_scale = 1.0;
    auto op_alpha = 0.0;
    auto op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_tanh, op_alpha, op_beta);
  }
  if (!is_dynamic_ && dst_->dtype() == "u8" && dst_min_->data() != nullptr) {
    if (append_eltwise_) {
      float zero_point = -1 * static_cast<const float*>(dst_min_->data())[0];
      po.append_eltwise(dst_scales_[0], algorithm::eltwise_linear, 1., zero_point);
    } else {
      vector<int> dst_zero_points;
      dst_zero_points = GetZeroPoints(dst_min_->data(), dst_scales_, dst_->dtype());
      attr_.set_zero_points(DNNL_ARG_DST, ic_dim, dst_zero_points);
    }
  }
  if (binary_add_) {
    vector<int64_t> post_shape = post_->shape();
    vector<int64_t> post_stride = GetStrides(post_shape);
    memory::desc binary_md = memory::desc(post_shape, type2mem[post_->dtype()], post_stride);
    po.append_binary(algorithm::binary_add, binary_md);
    binary_m_ = memory(binary_md, eng_, DNNL_MEMORY_NONE);
  }
  attr_.set_post_ops(po);

  attr_.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  matmul_pd_ = dnnl::matmul::primitive_desc(matmul_d, attr_, eng_);

  memory::desc scratchpad_md = matmul_pd_.scratchpad_desc();

  if (scratchpad_) {
    free(scratchpad_);
    scratchpad_ = nullptr;
  }

  scratchpad_ = reinterpret_cast<void*>(
    aligned_alloc(ALIGNMENT, (scratchpad_md.get_size() / ALIGNMENT + 1) * ALIGNMENT));

  memory scratchpad_m = memory(scratchpad_md, eng_, scratchpad_);
  memory_args_[DNNL_ARG_SCRATCHPAD] = scratchpad_m;

  // 2.4 Prepare memory objects (cached)
  src0_m_ = memory(src0_md, eng_, DNNL_MEMORY_NONE);
  dst_m_ = memory(dst_md, eng_, DNNL_MEMORY_NONE);
  if (has_bias_) {
    bias_m_ = memory(bias_md, eng_, const_cast<void*>(bias_->data()));
    memory any_bias_m = bias_m_;
    if (matmul_pd_.bias_desc() != bias_m_.get_desc()) {
      any_bias_m = memory(matmul_pd_.bias_desc(), eng_);
      if (bias_->is_shared()) {
        int64_t bias_size = bias_m_.get_desc().get_size();
        void* bias_shm_ptr = MemoryAllocator::ManagedShm().find_or_construct<char>(bias_->name().c_str())[bias_size](0);
        any_bias_m.set_data_handle(bias_shm_ptr);
      }
      dnnl::reorder(bias_m_, any_bias_m).execute(eng_stream_, bias_m_, any_bias_m);
    }
    memory_args_[DNNL_ARG_BIAS] = any_bias_m;
  }
  if (cache_weight_) {
    memory::desc user_src1_md = memory::desc(src1_shape, type2mem[src1_->dtype()], memory::format_tag::ab);
    src1_m_ = memory(user_src1_md, eng_, const_cast<void*>(src1_->data()));
    any_src1_m_ = src1_m_;
    if (matmul_pd_.weights_desc() != src1_m_.get_desc()) {
      any_src1_m_ = memory(matmul_pd_.weights_desc(), eng_);
      if (src1_->is_shared()) {
        int64_t weight_size = any_src1_m_.get_desc().get_size();
        void* weight_shm_ptr =
            MemoryAllocator::ManagedShm().find_or_construct<char>(src1_->name().c_str())[weight_size](0);
        any_src1_m_.set_data_handle(weight_shm_ptr);
      }
      dnnl::reorder(src1_m_, any_src1_m_).execute(eng_stream_, src1_m_, any_src1_m_);
    }
    memory_args_[DNNL_ARG_WEIGHTS] = any_src1_m_;
  } else {
    src1_m_ = memory(src1_md, eng_, DNNL_MEMORY_NONE);
  }

  // If the matmul forward class in the cache pool, just get it from pool.
  // Otherwise, do the reshape and send the related class into the cache pool
  size_t key = MatMulPrimitiveFwdFactory::Key(src0_->dtype(), src1_->dtype(), output_dtype_, src0_->shape(),
                                              src1_->shape(), src0_perm_, src1_perm_, dst_perm_, append_op_,
                                              post_->shape(), output_scale_, &eng_);
  if (MatMulPrimitiveFwdFactory::IsInFactory(key) && !MatMulPrimitiveFwdFactory::DoNotCache()) {
    matmul_p_ = MatMulPrimitiveFwdFactory::Get(key);
  } else {
    // 2.5 Prepare primitive objects (cached)
    matmul_p_ = dnnl::matmul(matmul_pd_);
    MatMulPrimitiveFwdFactory::Set(key, matmul_p_);
  }
  DstReshapeFusion(input, output);
}

vector<vector<string>> MatmulOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<vector<string>> inplace_pairs;
  // skip inplace in debug mode
  if (this->get_execution_mode() == ExecutionMode::DEBUG) {
    return inplace_pairs;
  }
  // append_sum sum_tensor -> output[0]
  if (!transpose_mode_ && post_ != nullptr && !binary_add_ && post_->left_life() == 1) {
    inplace_pairs.emplace_back(vector<string>({post_->name(), output[0]->name()}));
  }
  return inplace_pairs;
}

// 2. inference kernel(for int8 and f32)
void MatmulOperator::ForwardwithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src0_data = src0_->data();
  const auto& src1_data = src1_->data();
  void* dst_data;
  // create a dynamic quantization output with fp32.
  Tensor matmul_fp32_res;
  if (is_dynamic_) {
    matmul_fp32_res = *dst_;
    matmul_fp32_res.set_dtype("fp32");
    dst_data = matmul_fp32_res.mutable_data();
  } else {
    dst_data = dst_->mutable_data();
  }
  // has post_op: append_sum
  if (post_ != nullptr && !binary_add_) {
    DLOG(INFO) << "matmul has post op " << post_->name();
    void* post_data_ptr = const_cast<void*>(post_->data());
    auto life_count = MemoryAllocator::get().CheckMemory(post_data_ptr);
    // MemoryAllocate::check_tensor_life
    if (life_count == 1 && this->get_execution_mode() != ExecutionMode::DEBUG) {
      post_->unref_data(true);
      if (is_dynamic_)
        matmul_fp32_res.set_data(post_data_ptr);
      else
        dst_->set_data(post_data_ptr);
      dst_data = post_data_ptr;
    } else {
      int data_size = post_->size();
      string data_type = post_->dtype();
      memcpy(dst_data, post_data_ptr, data_size * type2bytes[data_type]);
      DLOG(WARNING) << "post tensor will be used by multi node...";
    }
  }

  // 1. Prepare memory objects with data_ptr
  src0_m_.set_data_handle(const_cast<void*>(src0_data), eng_stream_);
  src1_m_.set_data_handle(const_cast<void*>(src1_data), eng_stream_);
  dst_m_.set_data_handle(dst_data, eng_stream_);

  // 2. Reorder the data when the primitive memory and user memory are different

  memory any_src0_m = src0_m_;
  memory any_src1_m = src1_m_;
  memory any_dst_m = dst_m_;

  if (matmul_pd_.src_desc() != src0_m_.get_desc()) {
    any_src0_m = memory(matmul_pd_.src_desc(), eng_);
    dnnl::reorder(src0_m_, any_src0_m).execute(eng_stream_, src0_m_, any_src0_m);
  }
  if (!cache_weight_) {
    if (matmul_pd_.weights_desc() != src1_m_.get_desc()) {
      any_src1_m = memory(matmul_pd_.weights_desc(), eng_);
      dnnl::reorder(src1_m_, any_src1_m).execute(eng_stream_, src1_m_, any_src1_m);
    }
  }

  if (matmul_pd_.dst_desc() != dst_m_.get_desc()) {
    any_dst_m = memory(matmul_pd_.dst_desc(), eng_);
  }

  // the runtime calculation of dynamic quantization

  vector<int32_t> src0_zero_points;
  vector<float> rescales;
  vector<float> dynamic_bias;
  if (is_dynamic_) {
    DynamicForward(&src0_zero_points, &rescales, &dynamic_bias);
  }

  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC_0] = any_src0_m;
  if (!cache_weight_) memory_args_[DNNL_ARG_WEIGHTS] = any_src1_m;
  memory_args_[DNNL_ARG_DST] = any_dst_m;

  // has post_op: binary_add
  if (post_ != nullptr && binary_add_) {
    void* post_ptr = post_->mutable_data();
    binary_m_.set_data_handle(post_ptr, eng_stream_);
    memory_args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = binary_m_;
  }

  // 4. Execute the primitive
  matmul_p_.execute(eng_stream_, memory_args_);

  // 5. Reorder the data of dst memory (When it is format_any)
  if (matmul_pd_.dst_desc() != dst_m_.get_desc()) {
    dnnl::reorder(any_dst_m, dst_m_).execute(eng_stream_, any_dst_m, dst_m_);
  }

  eng_stream_.wait();
  // 6. unref tensors
  this->unref_tensors(input);

  if (is_dynamic_) {
    // quantize the fp32 result of matmul
    if (output.size() > 1) {
      runtime_minmax(reinterpret_cast<float*>(matmul_fp32_res.mutable_data()), matmul_fp32_res.size(),
                     reinterpret_cast<float*>(dst_min_->mutable_data()),
                     reinterpret_cast<float*>(dst_max_->mutable_data()));
      // quantize
      if (output_dtype_ == "u8" || output_dtype_ == "s8") {
        auto scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
        float* dst_max_data = reinterpret_cast<float*>(dst_max_->mutable_data());
        *dst_max_data = 1.0 / scales_[0];
        // memcpy(dst_max_->mutable_data(), scales_.data(), dst_max_->size() * sizeof(float));
#if __AVX512F__
        Quantize_avx512(matmul_fp32_res.size(), dst_->dtype(), matmul_fp32_res.data(),
                        static_cast<const float*>(dst_min_->data()), scales_, dst_->mutable_data());
#else
        Quantize(matmul_fp32_res.size(), dst_->dtype(), matmul_fp32_res.data(),
                 static_cast<const float*>(dst_min_->data()), scales_, dst_->mutable_data());
#endif
        matmul_fp32_res.unref_data();
      } else {
        // copy fp32_res to dst if not quantize
        void* res_ptr = const_cast<void*>(matmul_fp32_res.data());
        matmul_fp32_res.unref_data(true);
        dst_->set_data(res_ptr);
      }
    } else {
      void* res_ptr = const_cast<void*>(matmul_fp32_res.data());
      matmul_fp32_res.unref_data(true);
      dst_->set_data(res_ptr);
    }
  }
}

void MatmulOperator::RuntimeMinmax() {
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
  dnnl::reduction(reduce_min_pd).execute(eng_stream_, {{DNNL_ARG_SRC, dst_m_}, {DNNL_ARG_DST, reduce_min}});
  dnnl::reduction::desc reduce_max_d(algorithm::reduction_max, dst_m_.get_desc(), dst_md, 0.f, 0.f);
  dnnl::reduction::primitive_desc reduce_max_pd(reduce_max_d, eng_);
  dnnl::reduction(reduce_max_pd).execute(eng_stream_, {{DNNL_ARG_SRC, dst_m_}, {DNNL_ARG_DST, reduce_max}});
}

void MatmulOperator::DynamicForward(vector<int32_t>* src0_zero_points_ptr, vector<float>* rescales_ptr,
                                    vector<float>* dynamic_bias_ptr) {
  auto& rescales = *rescales_ptr;
  int channel_size = src1_min_->size();  // channel_size=1 represent per_tensor
  rescales.resize(channel_size);
  const float* src0_scales = reinterpret_cast<const float*>(src0_max_->data());
  const float* src1_scales = reinterpret_cast<const float*>(src1_max_->data());
  if (channel_size == 1) {
    rescales[0] = output_scale_ * src0_scales[0] * src1_scales[0];
  } else {
#pragma omp parallel for
    for (int i = 0; i < channel_size; i++) rescales[i] = output_scale_ * src0_scales[0] * src1_scales[i];
  }
  scale_f32_mem_.set_data_handle(reinterpret_cast<void*>(rescales.data()), eng_stream_);
  memory_args_[DNNL_ARG_ATTR_OUTPUT_SCALES] = scale_f32_mem_;

  // The bias loaded from file is not scaled. So need rescaled runtime.
  if (has_bias_) {
    auto& dynamic_bias = *dynamic_bias_ptr;
    dynamic_bias.resize(bias_->size());
    void* bias_m_data = bias_m_.get_data_handle();
    if (bias_m_data != nullptr) {
      float* bias_data = reinterpret_cast<float*>(bias_m_data);
      if (channel_size == 1) {
#pragma omp parallel for
        for (int i = 0; i < bias_->size(); i++) dynamic_bias[i] = bias_data[i] / rescales[0];
      } else {
#pragma omp parallel for
        for (int i = 0; i < bias_->size(); i++) dynamic_bias[i] = bias_data[i] / rescales[i];
      }
      bias_m_.set_data_handle(reinterpret_cast<void*>(dynamic_bias.data()), eng_stream_);
    }
  }

  if (src0_->dtype() == "u8") {
    auto& src0_zero_points = *src0_zero_points_ptr;
    float tmp = 1.0 / *src0_scales;
    src0_zero_points =
        GetZeroPoints(reinterpret_cast<const float*>(src0_min_->data()), &tmp, src0_->dtype(), src0_min_->size());
    zp_src0_mem_.set_data_handle(reinterpret_cast<void*>(src0_zero_points.data()), eng_stream_);
    memory_args_[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = zp_src0_mem_;
  }
}

// handle src tensor with SparseLib 3D format
// SaprseLib Q (output 3d shape)                 SparseLib K (ouput 3d shape)
//        \                                               /
//     Reshape (output 5d shape)                     Reshape (output 5d shape)
//            \                                        /
//                             BatchMatmul (5d x 5d)
// SparseLib V (output 3d shape)       |
//               \                     |
//                             BatchMatMul (5d x 5d)
// only all Q, K, V are SparseLib 3D gemm, both BatchMatmul use 5D format
// op will fall back src shape first, then fall back src data if in mix sparsity,
// from SparseLib 3D to SparseLib 2D
void MatmulOperator::InputShapeFallBack(const vector<Tensor*>& input) {
  // Q Dense format, K SparseLib 3D format
  // or QK BatchMatmul Dense format, V SparseLib 3D format
  if ((input[0]->tensor_format() == TensorFormat::MK || input[0]->tensor_format() == TensorFormat::BHnSHs) &&
      input[0]->shape().size() == 4 && input[1]->tensor_format() == TensorFormat::MmKMb &&
      input[1]->shape().size() == 5) {
    vector<int64_t> src1_shape = input[1]->shape();
    src1_shape_bfb_ = src1_shape;
    input[1]->set_shape({src1_shape[1], src1_shape[2], src1_shape[0] * src1_shape[3], src1_shape[4]});
    // Q SparseLib 3D format, K Dense format
  } else if (input[0]->tensor_format() == TensorFormat::MmKMb && input[0]->shape().size() == 5 &&
             input[1]->tensor_format() == TensorFormat::MK && input[1]->shape().size() == 4) {
    vector<int64_t> src0_shape = input[0]->shape();
    src0_shape_bfb_ = src0_shape;
    input[0]->set_shape({src0_shape[1], src0_shape[2], src0_shape[0] * src0_shape[3], src0_shape[4]});
    // QK BatchMatmul SparseLib 3D format, V Dense format
  } else if (input[0]->tensor_format() == TensorFormat::BmBbHnSHs && input[0]->shape().size() == 5 &&
             input[1]->tensor_format() == TensorFormat::MK && input[1]->shape().size() == 4) {
    vector<int64_t> src0_shape = input[0]->shape();
    src0_shape_bfb_ = src0_shape;
    input[0]->set_shape({src0_shape[0] * src0_shape[1], src0_shape[2], src0_shape[3], src0_shape[4]});
  } else {
    return;
  }
}

void MatmulOperator::UnsqueezePerm(vector<int64_t>* perm) {
  if (!perm->empty()) {
    perm->insert(perm->begin(), -1);
    for (int i = 0; i < perm->size(); ++i) (*perm)[i]++;
  }
}

void MatmulOperator::ResetPerm(vector<int64_t>* perm, const string& perm_name) {
  if (!perm->empty()) {
    auto attrs_map = operator_conf_->attributes();
    perm->clear();
    StringSplit<int64_t>(perm, attrs_map[perm_name], ",");
  }
}

void MatmulOperator::AdaptAttrs(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage) {
  if (stage == "in") {
    // change op attr only when src0 and src1 are from SaprseLib 3D
    // 1. Q x K (both SparseLib 3D format)
    if (input[0]->tensor_format() == TensorFormat::MmKMb && input[0]->shape().size() == 5 &&
        input[1]->tensor_format() == TensorFormat::MmKMb && input[1]->shape().size() == 5) {
      DLOG(INFO) << "Operator " << name_ << " gonna modify QK Matmul related attrs for SparseLib 3D format...";
      UnsqueezePerm(&src0_perm_);
      UnsqueezePerm(&src1_perm_);
      transpose_mode_ = false;
      adapt_attrs_ = true;
      output[0]->set_tensor_format(TensorFormat::BmBbHnSHs);
      // 2. Softmax(QK) x V (both SparseLib 3D format)
    } else if (input[0]->tensor_format() == TensorFormat::BmBbHnSHs && input[0]->shape().size() == 5 &&
               input[1]->tensor_format() == TensorFormat::MmKMb && input[1]->shape().size() == 5) {
      DLOG(INFO) << "Operator " << name_ << " gonna modify QKV Matmul related attrs for SparseLib 3D format...";
      UnsqueezePerm(&src1_perm_);
      UnsqueezePerm(&dst_perm_);
      transpose_mode_ = false;
      adapt_attrs_ = true;
      output[0]->set_tensor_format(TensorFormat::BmHnHsBbS);
      // mix sparsity
    } else if (((input[0]->tensor_format() == TensorFormat::MmKMb ||
                 input[0]->tensor_format() == TensorFormat::BmBbHnSHs) &&
                input[1]->tensor_format() != TensorFormat::MmKMb) ||
               (input[1]->tensor_format() == TensorFormat::MmKMb && input[0]->tensor_format() != TensorFormat::MmKMb)) {
      DLOG(INFO) << "FALL BACK...";
      output[0]->set_tensor_format(TensorFormat::BHnSHs);
      InputShapeFallBack(input);
      // Dense
    } else {
      return;
    }
    // reshape post op binary_add so that it has same dimensions as dst
    if (adapt_attrs_ && binary_add_ && post_ != nullptr) {
      vector<int64_t> post_shape = post_->shape();
      int64_t bm = input[0]->shape()[0];
      post_shape.insert(post_shape.begin(), bm);
      post_shape[1] = model_->input_shape()[0] / bm;
      CHECK_EQ(Product(post_shape), Product(post_->shape()))
          << "Wrong post shape in operator " << name_ << " with SparseLib 3D format...";
      post_->set_shape(post_shape);
    }
  } else if (stage == "out") {
    if (adapt_attrs_) {
      ResetPerm(&src0_perm_, "src0_perm");
      ResetPerm(&src1_perm_, "src1_perm");
      ResetPerm(&dst_perm_, "dst_perm");
      if (binary_add_ && post_ != nullptr) {
        vector<int64_t> post_shape(post_->shape().size() - 1, 0);
        post_shape[0] = model_->input_shape()[0];
        for (int i = 1; i < post_shape.size(); ++i) post_shape[i] = post_->shape()[i + 1];
        post_->set_shape(post_shape);
      }
      adapt_attrs_ = false;
    }
  } else {
    LOG(WARNING) << "Wrong stage parameter, should be in or out...";
  }
}

void MatmulOperator::AdaptTensors(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage) {
  if (stage == "in") {
    // Q Dense format, K SparseLib 3D format
    // or QK BatchMatmul Dense format, V SparseLib 3D format
    if ((input[0]->tensor_format() == TensorFormat::MK || input[0]->tensor_format() == TensorFormat::BHnSHs) &&
        input[0]->shape().size() == 4 && input[1]->tensor_format() == TensorFormat::MmKMb &&
        input[1]->shape().size() == 4) {
      input[1]->reorder(src1_shape_bfb_, {0, 3, 1, 2, 4});
      vector<int64_t> src1_shape = input[1]->shape();
      input[1]->set_shape({src1_shape[0] * src1_shape[1], src1_shape[2], src1_shape[3], src1_shape[4]});
      DLOG(INFO) << "Reorder src1 tensor of operator " << name_;
      // Q SparseLib 3D format, K Dense format
    } else if (input[0]->tensor_format() == TensorFormat::MmKMb && input[0]->shape().size() == 4 &&
               input[1]->tensor_format() == TensorFormat::MK && input[1]->shape().size() == 4) {
      input[0]->reorder(src0_shape_bfb_, {0, 3, 1, 2, 4});
      vector<int64_t> src0_shape = input[0]->shape();
      input[0]->set_shape({src0_shape[0] * src0_shape[1], src0_shape[2], src0_shape[3], src0_shape[4]});
      DLOG(INFO) << "Reorder src0 tensor of operator " << name_;
      // QK BatchMatmul SparseLib 3D format, V Dense format, no need reorder
    } else {
      DLOG(INFO) << "Run with SparseLib 3D format without reorder in operator " << name_;
    }
  } else if (stage == "out") {
    return;
  } else {
    LOG(WARNING) << "Wrong stage parameter, should be in or out...";
  }
}

REGISTER_OPERATOR_CLASS(Matmul);
}  // namespace executor
