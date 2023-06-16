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

#include "convolution.hpp"

#include "operator_registry.hpp"
namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32}, {"int32", dnnl::memory::data_type::s32},
    {"s32", dnnl::memory::data_type::s32},  {"fp16", dnnl::memory::data_type::f16},
    {"u8", dnnl::memory::data_type::u8},    {"s8", dnnl::memory::data_type::s8},
    {"bf16", dnnl::memory::data_type::bf16}};

ConvolutionOperator::ConvolutionOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf),
      src_perm_({}),
      dst_perm_({}),
      output_scale_(1.),
      format_any_(true),
      gelu_split_(false),
      weight_cached_(false),
      has_bias_(false) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("src_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src_perm_, attrs_map["src_perm"], ",");
  }
  iter = attrs_map.find("dst_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&dst_perm_, attrs_map["dst_perm"], ",");
  }
  iter = attrs_map.find("group");
  if (iter != attrs_map.end()) {
    group_ = StringToNum<int64_t>(attrs_map["group"]);
  }
  iter = attrs_map.find("pads");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&pads_, attrs_map["pads"], ",");
  }
  iter = attrs_map.find("strides");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&strides_, attrs_map["strides"], ",");
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
  iter = attrs_map.find("gelu_split");
  if (iter != attrs_map.end()) {
    gelu_split_ = attrs_map["gelu_split"] == "True" || attrs_map["gelu_split"] == "true";
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
  binary_add_ = (iter != attrs_map.end() && iter->second == "binary_add") ? true : false;
  append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
  gelu_erf_ = (iter != attrs_map.end() && iter->second == "gelu_erf") ? true : false;
  gelu_tanh_ = (iter != attrs_map.end() && iter->second == "gelu_tanh") ? true : false;
  tanh_ = (iter != attrs_map.end() && iter->second == "tanh") ? true : false;
  sigmoid_ = (iter != attrs_map.end() && iter->second == "sigmoid") ? true : false;
  relu_ = (iter != attrs_map.end() && iter->second == "relu") ? true : false;
  append_eltwise_ = (gelu_erf_ && !gelu_split_) || (gelu_tanh_ && !gelu_split_) || tanh_ || sigmoid_ || relu_;
  append_op_ = (iter != attrs_map.end()) ? iter->second : "";
  DLOG(INFO) << "append_op: " << append_op_;
}

void ConvolutionOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
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
      src_ = input[0];
      weight_ = input[1];
      break;
    }
    case 3: {
      src_ = input[0];
      weight_ = input[1];
      bias_ = (append_sum_ || binary_add_) ? nullptr : input[2];
      post_ = (append_sum_ || binary_add_) ? input[2] : nullptr;
      has_bias_ = !(append_sum_ || binary_add_);
      break;
    }
    case 4: {
      src_ = input[0];
      weight_ = input[1];
      bias_ = input[2];
      post_ = (append_sum_ || binary_add_) ? input[3] : nullptr;
      has_bias_ = true;
      break;
    }
    case 6: {
      src_ = input[0];
      weight_ = input[1];
      src_min_ = input[2];
      src_max_ = input[3];
      weight_min_ = input[4];
      weight_max_ = input[5];
      break;
    }
    case 7: {
      src_ = input[0];
      weight_ = input[1];
      bias_ = (append_sum_ || binary_add_) ? nullptr : input[2];
      post_ = (append_sum_ || binary_add_) ? input[2] : nullptr;
      src_min_ = input[3];
      src_max_ = input[4];
      weight_min_ = input[5];
      weight_max_ = input[6];
      has_bias_ = !(append_sum_ || binary_add_);
      break;
    }
    case 8: {
      if (append_sum_ || binary_add_) {
        // dynamic quantization
        src_ = input[0];
        weight_ = input[1];
        bias_ = input[2];
        post_ = input[3];
        src_min_ = input[4];
        src_max_ = input[5];
        weight_min_ = input[6];
        weight_max_ = input[7];
        has_bias_ = true;
      } else {
        src_ = input[0];
        weight_ = input[1];
        src_min_ = input[2];
        src_max_ = input[3];
        weight_min_ = input[4];
        weight_max_ = input[5];
        dst_min_ = input[6];
        dst_max_ = input[7];
      }
      break;
    }
    case 9: {
      src_ = input[0];
      weight_ = input[1];
      bias_ = (append_sum_ || binary_add_) ? nullptr : input[2];
      post_ = (append_sum_ || binary_add_) ? input[2] : nullptr;
      src_min_ = input[3];
      src_max_ = input[4];
      weight_min_ = input[5];
      weight_max_ = input[6];
      dst_min_ = input[7];
      dst_max_ = input[8];
      has_bias_ = !(append_sum_ || binary_add_);
      break;
    }
    case 10: {
      src_ = input[0];
      weight_ = input[1];
      bias_ = input[2];
      post_ = (append_sum_ || binary_add_) ? input[3] : nullptr;
      src_min_ = input[4];
      src_max_ = input[5];
      weight_min_ = input[6];
      weight_max_ = input[7];
      dst_min_ = input[8];
      dst_max_ = input[9];
      has_bias_ = true;
      break;
    }
    default: {
      LOG(ERROR) << "Convolution expect at most 10 inputs but receive " << input_size;
      break;
    }
  }
}

void ConvolutionOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // only for dense gemm dispatcher now
  if (dispatch_from_ == "InnerProduct" && input[0]->dtype() != "fp32") return;
  if (dispatch_from_ == "InnerProduct" && (input[1]->location().empty() || input[1]->shape().empty())) return;
  MapTensors(input, output);
  is_dynamic_ = output.size() > 1 || (src_min_ != nullptr && src_min_->raw_data() == nullptr && !src_min_->is_shared());
  if (has_bias_) {
    DLOG(INFO) << name_ << "Convolution has bias";
  }
  if (is_dynamic_) DLOG(INFO) << name_ << "Convolution is DYNAMIC!!!";
  dst_->set_dtype(output_dtype_);

  dnnl::primitive_attr attr;

  vector<float> src_scales;
  vector<float> weight_scales;
  vector<float> dst_scales;
  vector<float> rescales;
  if (!is_dynamic_) {
    if (output_scale_ != 1.f) {
      attr.set_output_scales(0, {output_scale_});
    } else if (weight_min_ != nullptr) {
      const int ic_dim = weight_min_->size() > 1 ? 0 | (1 << 1) : 0;
      src_scales = GetScales(src_min_->data(), src_max_->data(), src_min_->size(), src_->dtype());
      weight_scales = GetScales(weight_min_->data(), weight_max_->data(), weight_min_->size(), weight_->dtype());
      dst_scales = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
      rescales = GetRescales(src_scales, weight_scales, dst_scales, dst_->dtype(), append_eltwise_);
      attr.set_output_scales(ic_dim, rescales);
    }
  }

  dnnl::post_ops po;
  if (append_sum_) {
    float beta = 1.0;
    po.append_sum(beta);
  }

  if (gelu_erf_ && !gelu_split_) {
    float op_scale = 1.0;
    float op_alpha = 0.0;
    float op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_gelu_erf, op_alpha, op_beta);
  }
  if (gelu_tanh_ && !gelu_split_) {
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
  if (sigmoid_) {
    auto op_scale = 1.0;
    auto op_alpha = 0.0;
    auto op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_logistic, op_alpha, op_beta);
  }
  if (relu_) {
    auto op_scale = 1.0;
    auto op_alpha = 0.0;
    auto op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_relu, op_alpha, op_beta);
  }
  // this is to sub zero point in fp32 to make the output u8
  if (!is_dynamic_ && append_eltwise_ && dst_->dtype() == "u8") {
    float zero_point = -1 * static_cast<const float*>(dst_min_->data())[0];
    po.append_eltwise(dst_scales[0], algorithm::eltwise_linear, 1., zero_point);
  }
  if (append_eltwise_ || append_sum_) attr.set_post_ops(po);

  // set conv attr to scratchpad_mode
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  attr_ = attr;

  // cache weight here, save weight and bias memory descriptor
  vector<int64_t> weight_shape_origin = weight_->shape();
  vector<int64_t> weight_stride_m;
  vector<int64_t> weight_shape_m;
  if (dispatch_from_ == "InnerProduct") {
    // innerproduct has not any transpose, [M, K] x [K, N], after innerproduct prepare, shape becomes [M, K] x [N, K]
    // innerproduct transpose src0, [K, M] x [K, N], after innerproduct prepare, shape becomes [K, M] x [N, K]
    // innerproduct transpose src1, [M, K] x [N, K], after innerproduct prepare, shape becomes [M, K] x [N, K]
    // innerproduct transpose src0 and src1, [K, M] x [N, K], after innerproduct prepare, shape becomes [K, M] x [N, K]
    // innerproduct just changes shape (changes src0 shape at reshape period, scr1 shape at prepare period),
    // but keep strides as origin.
    pads_ = {0, 0, 0, 0};
    strides_ = {1, 1};
    vector<int64_t> weight_perm;
    // consider if onednn transpose wight or not
    if (weight_->is_transposed()) {
      weight_shape_origin = {weight_shape_origin[1], weight_shape_origin[0], 1, 1};
      // [K, N, 1, 1] -> [N, K, 1, 1]
      weight_perm = {1, 0, 2, 3};
    } else {
      weight_shape_origin = {weight_shape_origin[0], weight_shape_origin[1], 1, 1};
      // [N, K, 1, 1] -> [N, K, 1, 1]
      weight_perm = {0, 1, 2, 3};
    }
    weight_shape_ = GetShapes(weight_shape_origin, weight_perm);
    weight_stride_m = GetStrides(weight_shape_origin, weight_perm);
    weight_shape_m = weight_shape_;
  } else {
    weight_shape_ = GetShapes(weight_shape_origin);
    weight_->set_shape(weight_shape_);
    vector<int64_t> weight_group_shape = weight_shape_origin;
    if (group_ != 1) {
      weight_group_shape.insert(weight_group_shape.begin(), group_);
      weight_group_shape[1] /= group_;
      if (weight_group_shape[1] % group_ != 0) {
        LOG(ERROR) << "Output channel(" << weight_group_shape[1] << ") is not divisible by "
                   << "group(" << group_ << ") in covolution!";
      }
    }
    vector<int64_t> weight_group_stride = GetStrides(weight_group_shape);
    weight_stride_m = weight_group_stride;
    weight_shape_m = weight_group_shape;
  }

  any_weight_md_ = memory::desc(weight_shape_m, type2mem[weight_->dtype()], memory::format_tag::any);
  weight_md_ = memory::desc(weight_shape_m, type2mem[weight_->dtype()], weight_stride_m);
  weight_m_ = memory(weight_md_, eng_, weight_->mutable_data());

  if (bias_ != nullptr) {
    const vector<int64_t> bias_shape = bias_->shape();
    const vector<int64_t> bias_stride = GetStrides(bias_shape);
    bias_md_ = memory::desc(bias_shape, type2mem[bias_->dtype()], bias_stride);
    any_bias_md_ = memory::desc(bias_shape, type2mem[bias_->dtype()], memory::format_tag::any);
    bias_m_ = memory(bias_md_, eng_, bias_->mutable_data());
  }
}

void ConvolutionOperator::DstReshapeFusion(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (!reshape_.empty()) {
    vector<int64_t> ref_shape;
    if (!reshape_dims_.empty()) {
      ref_shape = input.back()->shape();
    }
    vector<int64_t> reshape(reshape_);
    vector<int64_t> dst_shape = GetDstShape(reshape, output[0]->size(), ref_shape, reshape_dims_);
    output[0]->set_shape(dst_shape);
  }
}

// 1. Create primitive
void ConvolutionOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // Part1: Derive operator's user proper shape and strides
  // 1.1 Transpose tensor shape and get it
  vector<int64_t> src_shape_origin;
  if (dispatch_from_ == "InnerProduct") {
    CHECK_EQ(dispatch_kernel_config["InnerProduct_to_Convolution"].size(), dispatch_config_.size() - 1)
        << "InnerProduct to Convolution has wrong dispatch kernel config...";
    StringSplit<int64_t>(&src_shape_origin, dispatch_config_[1], ",");
    CHECK_EQ(Product(src_shape_origin), Product(src_->shape())) << "Wrong dispatch input shape...";
    // consider if model transpose src0 or not
    if (src_->is_transposed()) {
      // [C, N, H, W] -> [N, C, H, W]
      src_perm_ = {1, 0, 2, 3};
    } else {
      // [N, H, W, C] -> [N, C, H, W]
      src_perm_ = {0, 3, 1, 2};
    }
    // N, C, H, W ->N, H, W, C
    dst_perm_ = {0, 2, 3, 1};
  } else {
    src_shape_origin = src_->shape();
  }
  vector<int64_t> src_shape = GetShapes(src_shape_origin, src_perm_);
  vector<int64_t> src_stride = GetStrides(src_shape_origin, src_perm_);
  if (dispatch_from_.empty()) src_->set_shape(src_shape);
  if (is_dynamic_ && (output_scale_ != 1.f || src_min_ != nullptr || weight_min_ != nullptr)) {
    if (src_min_ != nullptr && weight_max_ != nullptr) {
      int mask = weight_min_->size() > 1 ? 2 : 0;
      attr_.set_output_scales(mask, {DNNL_RUNTIME_F32_VAL});
      vector<int64_t> scale_shape;
      // scale_shape.push_back(1);
      scale_shape.push_back(weight_min_->size());
      memory::desc scale_md_ = memory::desc(scale_shape, memory::data_type::f32, GetStrides(scale_shape));
      scale_f32_mem_ = memory(scale_md_, eng_, DNNL_MEMORY_NONE);
      // if (!(dst_->dtype() == "u8" || dst_->dtype() == "s8")) {
      //   dnnl::post_ops po;
      //   po.append_binary(algorithm::binary_mul, scale_md_);
      //   attr_.set_post_ops(po);
      // }
      // need zero point when src0 is u8
      if (src_->dtype() == "u8") {
        mask = src_min_->size() > 1 ? 2 : 0;
        zp_src0_mem_ = memory(
            {{static_cast<dnnl_dim_t>(src_min_->size())}, memory::data_type::s32, GetStrides(scale_shape)}, eng_,
            DNNL_MEMORY_NONE);
        attr_.set_zero_points(DNNL_ARG_SRC, mask, {DNNL_RUNTIME_S32_VAL});
      }
    }
  }
  // 1.2 malloc tensor for output
  vector<int64_t> dst_shape_origin;
  vector<int64_t> padding_dims_l;
  vector<int64_t> padding_dims_r;
  vector<int64_t> dst_shape_after_dispatch;
  switch (src_shape_origin.size()) {
    case 3: {
      // src_: N * IC* IH, weight_: OC * KC * KH
      // pad: (PH_L, PH_R), stride: (SH)
      // OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
      // dst_: N * OC * OH
      const int64_t N = src_shape[0];
      const int64_t IC = src_shape[1];
      const int64_t IH = src_shape[2];
      const int64_t OC = weight_shape_[0];
      const int64_t KC = weight_shape_[1];
      const int64_t KH = weight_shape_[2];
      const int64_t PH_L = pads_[0];
      const int64_t PH_R = pads_[1];
      const int64_t SH = strides_[0];
      if (KC * group_ != IC) {
        LOG(ERROR) << "Multiplying kernel channel(" << KC << " and group(" << group_
                   << ") does not equal input channel(" << IC << ") in covolution!";
      }
      const int64_t OH = (IH - KH + PH_L + PH_R) / SH + 1;
      padding_dims_l = {PH_L};
      padding_dims_r = {PH_R};
      dst_shape_origin = {N, OC, OH};
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
      const int64_t OC = weight_shape_[0];
      const int64_t KC = weight_shape_[1];
      const int64_t KH = weight_shape_[2];
      const int64_t KW = weight_shape_[3];
      const int64_t PH_L = pads_[0];
      const int64_t PH_R = pads_[1];
      const int64_t PW_L = pads_[2];
      const int64_t PW_R = pads_[3];
      const int64_t SH = strides_[0];
      const int64_t SW = strides_[1];
      const int64_t OH = (IH - KH + PH_L + PH_R) / SH + 1;
      const int64_t OW = (IW - KW + PW_L + PW_R) / SW + 1;
      if (KC * group_ != IC) {
        LOG(ERROR) << "Multiplying kernel channel(" << KC << " and group(" << group_
                   << ") does not equal input channel(" << IC << ") in covolution!";
      }
      padding_dims_l = {PH_L, PW_L};
      padding_dims_r = {PH_R, PW_R};
      dst_shape_origin = {N, OC, OH, OW};
      // if conv as a dispatch kernel, may need reshape after execute
      if (dispatch_from_ == "InnerProduct") dst_shape_after_dispatch = {N * OH * OW, OC};
      break;
    }
    default:
      LOG(ERROR) << "Input size " << src_shape_origin.size() << " is not supported in convolution!";
  }

  // Sub-step3: fused post transpose, notice it's different that
  // pre transpose will use the tranposed shape and stride, it's straight forward
  // post transpose will use origin shape and that means the dst buffer in matmul
  // is a buffer transposed back from dst_perm(understand tranpose to and transpose back)
  // pre_transpose: src_buffer -> pre_transpose -> target_buffer in matmul
  // post_transpose: target_buffer in matmul<- post transpose <-dst_buffer
  vector<int64_t> dst_shape = GetShapes(dst_shape_origin, dst_perm_);
  vector<int64_t> reverse_perm = ReversePerm(dst_perm_);
  vector<int64_t> dst_stride = GetStrides(dst_shape, reverse_perm);

  // 1.4 Prepare memory descriptors
  memory::desc any_src_md = memory::desc(src_shape, type2mem[src_->dtype()], memory::format_tag::any);
  memory::desc src_md = memory::desc(src_shape, type2mem[src_->dtype()], src_stride);

  std::string dynamic_output_dtype = output_dtype_ == "bf16" ? "bf16" : "fp32";
  memory::desc any_dst_md = memory::desc(dst_shape_origin, type2mem[is_dynamic_ ? dynamic_output_dtype : output_dtype_],
                                         memory::format_tag::any);
  memory::desc dst_md =
      memory::desc(dst_shape_origin, type2mem[is_dynamic_ ? dynamic_output_dtype : output_dtype_], dst_stride);

  // 1.5 Set dst shape and strides
  dst_->set_shape(dst_shape);
  if (output.size() > 1) {
    dst_min_->set_shape({1});
    dst_max_->set_shape({1});
  }

  if (dispatch_from_ == "InnerProduct") dst_->set_shape(dst_shape_after_dispatch);

  // 2.2 Prepare op descriptors
  dnnl::convolution_forward::desc convolution_d =
      has_bias_
          ? dnnl::convolution_forward::desc(prop_kind::forward_inference, algorithm::convolution_auto, src_md,
                                            weight_md_, bias_md_, dst_md, strides_, padding_dims_l, padding_dims_r)
          : dnnl::convolution_forward::desc(prop_kind::forward_inference, algorithm::convolution_auto, src_md,
                                            weight_md_, dst_md, strides_, padding_dims_l, padding_dims_r);
  if (format_any_) {
    convolution_d = has_bias_ ? dnnl::convolution_forward::desc(
                                    prop_kind::forward_inference, algorithm::convolution_auto, any_src_md,
                                    any_weight_md_, any_bias_md_, any_dst_md, strides_, padding_dims_l, padding_dims_r)
                              : dnnl::convolution_forward::desc(prop_kind::forward_inference,
                                                                algorithm::convolution_auto, any_src_md, any_weight_md_,
                                                                any_dst_md, strides_, padding_dims_l, padding_dims_r);
  }

  if (gelu_erf_ && gelu_split_) {
    memory::desc gelu_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);
    auto gelu_d =
        dnnl::eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_gelu_erf, gelu_md, 0.f, 0.f);
    gelu_pd_ = dnnl::eltwise_forward::primitive_desc(gelu_d, gelu_eng_);
    gelu_p_ = dnnl::eltwise_forward(gelu_pd_);
    gelu_m_ = memory(gelu_md, gelu_eng_, DNNL_MEMORY_NONE);
  }
  if (gelu_tanh_ && gelu_split_) {
    memory::desc gelu_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);
    auto gelu_d =
        dnnl::eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_gelu_tanh, gelu_md, 0.f, 0.f);
    gelu_pd_ = dnnl::eltwise_forward::primitive_desc(gelu_d, gelu_eng_);
    gelu_p_ = dnnl::eltwise_forward(gelu_pd_);
    gelu_m_ = memory(gelu_md, gelu_eng_, DNNL_MEMORY_NONE);
  }
  if (binary_add_) {
    // The binary primitive requires all source and destination tensors to have the same number of dimensions.
    dnnl::primitive_attr attr;
    dnnl::post_ops po;
    vector<int64_t> post_shape = post_->shape();
    vector<int64_t> post_stride = GetStrides(post_shape);
    if (dispatch_from_ == "InnerProduct") {
      // [M, N] -> [N, H, W, C] -> [N, C, H, W]
      vector<int64_t> post_perm = {0, 3, 1, 2};
      post_shape = GetShapes(dst_shape_origin, post_perm);
      post_stride = GetStrides(dst_shape_origin, post_perm);
    }
    memory::desc binary_md = memory::desc(post_shape, type2mem[post_->dtype()], post_stride);
    po.append_binary(algorithm::binary_add, binary_md);
    attr.set_post_ops(po);
    binary_m_ = memory(binary_md, eng_, DNNL_MEMORY_NONE);
    attr_ = attr;
  }

  convolution_pd_ = dnnl::convolution_forward::primitive_desc(convolution_d, attr_, eng_);
  memory::desc scratchpad_md = convolution_pd_.scratchpad_desc();
  if (scratchpad_) {
    aligned_free(scratchpad_);
    scratchpad_ = nullptr;
  }

  scratchpad_ =
      reinterpret_cast<void*>(aligned_alloc(ALIGNMENT, (scratchpad_md.get_size() / ALIGNMENT + 1) * ALIGNMENT));
  memory scratchpad_m = memory(scratchpad_md, eng_, scratchpad_);
  memory_args_[DNNL_ARG_SCRATCHPAD] = scratchpad_m;

  // 2.4 Prepare memory objects (cached)
  src_m_ = memory(src_md, eng_, DNNL_MEMORY_NONE);
  dst_m_ = memory(dst_md, eng_, DNNL_MEMORY_NONE);
  if (!weight_cached_) {
    memory any_weight_m = weight_m_;
    if (convolution_pd_.weights_desc() != weight_m_.get_desc()) {
      any_weight_m = memory(convolution_pd_.weights_desc(), eng_);
      dnnl::reorder(weight_m_, any_weight_m).execute(eng_stream_, weight_m_, any_weight_m);
    }
    memory_args_[DNNL_ARG_WEIGHTS] = any_weight_m;
    if (!is_dynamic_ && has_bias_) {
      memory any_bias_m = bias_m_;
      if (convolution_pd_.bias_desc() != bias_m_.get_desc()) {
        any_bias_m = memory(convolution_pd_.bias_desc(), eng_);
        dnnl::reorder(bias_m_, any_bias_m).execute(eng_stream_, bias_m_, any_bias_m);
      }
      memory_args_[DNNL_ARG_BIAS] = any_bias_m;
    }
    weight_cached_ = (dispatch_from_ == "InnerProduct") ? false : true;
  }

  // If the convolution forward class in the cache pool, just get it from the pool.
  // Otherwise, do the reshape and send the related class into the cache pool
  size_t key = ConvolutionPrimitiveFwdFactory::Key(src_->dtype(), weight_->dtype(), output_dtype_, src_shape,
                                                   weight_->shape(), dst_perm_, append_op_, post_->shape(),
                                                   output_scale_, group_, pads_, strides_, &eng_);
  if (ConvolutionPrimitiveFwdFactory::IsInFactory(key) && !ConvolutionPrimitiveFwdFactory::DoNotCache()) {
    convolution_p_ = ConvolutionPrimitiveFwdFactory::Get(key);
  } else {
    // 2.5 Prepare primitive objects (cached)
    convolution_p_ = dnnl::convolution_forward(convolution_pd_);
    ConvolutionPrimitiveFwdFactory::Set(key, convolution_p_);
  }
  DstReshapeFusion(input, output);
}

vector<vector<string>> ConvolutionOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<vector<string>> inplace_pairs;
  // skip inplace in debug mode
  if (this->get_execution_mode() == ExecutionMode::DEBUG) {
    return inplace_pairs;
  }
  // append_sum sum_tensor -> output[0]
  if (post_ != nullptr && !binary_add_ && post_->left_life() == 1) {
    inplace_pairs.emplace_back(vector<string>({post_->name(), output[0]->name()}));
  }
  return inplace_pairs;
}

// 2. inference kernel(for int8 and f32)
void ConvolutionOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  Tensor convolution_dynamic_res;
  void* dst_data;
  if (is_dynamic_) {
    convolution_dynamic_res = *dst_;
    convolution_dynamic_res.set_dtype(output_dtype_ == "bf16" ? "bf16" : "fp32");
    dst_data = convolution_dynamic_res.mutable_data();
  } else {
    dst_data = dst_->mutable_data();
  }
  // has post_op: append_sum
  if (post_ != nullptr && !binary_add_) {
    DLOG(INFO) << "Convolution has post op " << post_->name();
    void* post_data_ptr = const_cast<void*>(post_->data());
    auto life_count = MemoryAllocator::get().CheckMemory(post_data_ptr);
    // MemoryAllocate::check_tensor_life
    if (life_count == 1 && this->get_execution_mode() != ExecutionMode::DEBUG) {
      post_->unref_data(true);
      if (is_dynamic_)
        convolution_dynamic_res.set_data(post_data_ptr);
      else
        dst_->set_data(post_data_ptr);
    } else {
      void* dst_data_ptr = dst_->mutable_data();
      int data_size = post_->size();
      string data_type = post_->dtype();
      memcpy(dst_data_ptr, post_data_ptr, data_size * type2bytes[data_type]);
      LOG(WARNING) << "post tensor will be used by multi node...";
    }
  }
  const auto& src_data = src_->data();

  // 1. Prepare memory objects with data_ptr
  src_m_.set_data_handle(const_cast<void*>(src_data), eng_stream_);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), eng_stream_);
  memory any_src_m = src_m_;
  memory any_bias_m = bias_m_;
  memory any_dst_m = dst_m_;
  // 2. Reorder the data when the primitive memory and user memory are different
  if (convolution_pd_.src_desc() != src_m_.get_desc()) {
    any_src_m = memory(convolution_pd_.src_desc(), eng_);
    dnnl::reorder(src_m_, any_src_m).execute(eng_stream_, src_m_, any_src_m);
  }
  if (convolution_pd_.dst_desc() != dst_m_.get_desc()) {
    any_dst_m = memory(convolution_pd_.dst_desc(), eng_);
  }
  // the runtime calculation of dynamic quantization
  vector<int32_t> src0_zero_points;
  vector<float> rescales;
  vector<float> dynamic_bias;
  if (is_dynamic_) DynamicForward(&src0_zero_points, &rescales, &dynamic_bias, &any_bias_m);

  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC_0] = any_src_m;
  memory_args_[DNNL_ARG_DST] = any_dst_m;
  // has post_op: binary_add
  if (post_ != nullptr && binary_add_) {
    void* post_ptr = post_->mutable_data();
    binary_m_.set_data_handle(post_ptr, eng_stream_);
    memory_args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0 + is_dynamic_) | DNNL_ARG_SRC_1] = binary_m_;
  }
  // 4. Execute the primitive
  convolution_p_.execute(eng_stream_, memory_args_);
  // 5. Reorder the data of dst memory (When it is format_any)
  if (convolution_pd_.dst_desc() != dst_m_.get_desc()) {
    dnnl::reorder(any_dst_m, dst_m_).execute(eng_stream_, any_dst_m, dst_m_);
  }
  // gelu seperate
  if ((gelu_split_ && gelu_tanh_) || (gelu_split_ && gelu_erf_)) {
    dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), gelu_eng_stream_);
    gelu_m_.set_data_handle(reinterpret_cast<void*>(dst_data), gelu_eng_stream_);
    gelu_memory_args_[DNNL_ARG_SRC] = dst_m_;
    gelu_memory_args_[DNNL_ARG_DST] = gelu_m_;
    gelu_p_.execute(gelu_eng_stream_, gelu_memory_args_);
  }
  eng_stream_.wait();
  this->unref_tensors(input);
  if (is_dynamic_) {
    if (output.size() > 1) {
      runtime_minmax(reinterpret_cast<float*>(convolution_dynamic_res.mutable_data()), convolution_dynamic_res.size(),
                     reinterpret_cast<float*>(dst_min_->mutable_data()),
                     reinterpret_cast<float*>(dst_max_->mutable_data()));
      // quantize
      if (output_dtype_ == "u8" || output_dtype_ == "s8") {
        auto scales = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
#if __AVX512F__
        Quantize_avx512(convolution_dynamic_res.size(), dst_->dtype(), convolution_dynamic_res.data(),
                        static_cast<const float*>(dst_min_->data()), scales, dst_->mutable_data());
#else
        Quantize(convolution_dynamic_res.size(), dst_->dtype(), convolution_dynamic_res.data(),
                 static_cast<const float*>(dst_min_->data()), scales, dst_->mutable_data());
#endif
        convolution_dynamic_res.unref_data();
        // float* dst_min_data = reinterpret_cast<float*>(dst_min_->mutable_data());
        float* dst_max_data = reinterpret_cast<float*>(dst_max_->mutable_data());
        *dst_max_data = 1.0 / scales[0];
        // memcpy(dst_max_->mutable_data(), scales.data(), dst_max_->size() * sizeof(float));
      } else {
        // copy fp32_res to dst if not quantize
        void* res_ptr = const_cast<void*>(convolution_dynamic_res.data());
        convolution_dynamic_res.unref_data(true);
        dst_->set_data(res_ptr);
      }
    } else {
      void* res_ptr = const_cast<void*>(convolution_dynamic_res.data());
      convolution_dynamic_res.unref_data(true);
      dst_->set_data(res_ptr);
    }
  }
}

void ConvolutionOperator::DynamicForward(vector<int32_t>* src0_zero_points_ptr, vector<float>* rescales_ptr,
                                         vector<float>* dynamic_bias_ptr, memory* any_bias_m_ptr) {
  auto& rescales = *rescales_ptr;
  int channel_size = weight_min_->size();  // channel_size=1 represent per_tensor
  rescales.resize(channel_size);
  const float* src0_scales = reinterpret_cast<const float*>(src_max_->data());
  const float* src1_scales = reinterpret_cast<const float*>(weight_max_->data());
  // std::cout << name_ << "\tsrc0:\t" << src0_scales[0] << std::endl;
  // std::cout << name_ << "\tsrc1:\t" << src1_scales[0] << std::endl;
  if (channel_size == 1) {
    rescales[0] = output_scale_ * src1_scales[0] * src0_scales[0];
  } else {
#pragma omp parallel for
    for (int i = 0; i < channel_size; i++) rescales[i] = output_scale_ * src1_scales[i] * src0_scales[0];
  }
  scale_f32_mem_.set_data_handle(reinterpret_cast<void*>(rescales.data()), eng_stream_);
  memory_args_[DNNL_ARG_ATTR_OUTPUT_SCALES] = scale_f32_mem_;
  // The bias loaded from file is not scaled. So need rescaled runtime.
  if (has_bias_) {
    auto& dynamic_bias = *dynamic_bias_ptr;
    auto& any_bias_m = *any_bias_m_ptr;
    dynamic_bias.resize(bias_->size());
    float* bias_data = reinterpret_cast<float*>(bias_->mutable_data());
    if (channel_size == 1) {
#pragma omp parallel for
      for (int i = 0; i < bias_->size(); i++) dynamic_bias[i] = bias_data[i] / rescales[0];
    } else {
#pragma omp parallel for
      for (int i = 0; i < bias_->size(); i++) dynamic_bias[i] = bias_data[i] / rescales[i];
    }
    bias_m_.set_data_handle(reinterpret_cast<void*>(dynamic_bias.data()), eng_stream_);
    if (convolution_pd_.bias_desc() != bias_m_.get_desc()) {
      any_bias_m = memory(convolution_pd_.bias_desc(), eng_);
      dnnl::reorder(bias_m_, any_bias_m).execute(eng_stream_, bias_m_, any_bias_m);
    }
    memory_args_[DNNL_ARG_BIAS] = any_bias_m;
  }

  if (src_->dtype() == "u8") {
    auto& src0_zero_points = *src0_zero_points_ptr;
    float tmp = 1.0f / *src0_scales;
    src0_zero_points =
        GetZeroPoints(reinterpret_cast<const float*>(src_min_->data()), &tmp, src_->dtype(), src_min_->size());
    zp_src0_mem_.set_data_handle(reinterpret_cast<void*>(src0_zero_points.data()), eng_stream_);
    memory_args_[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = zp_src0_mem_;
  }
}

REGISTER_OPERATOR_CLASS(Convolution);
// InnerProduct dispathcer class can have convolution kernel implementation
REGISTER_KERNEL_CLASS(InnerProduct, Convolution);
}  // namespace executor
