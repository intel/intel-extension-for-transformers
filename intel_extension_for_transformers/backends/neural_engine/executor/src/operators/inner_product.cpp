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

#include "inner_product.hpp"
#include "data_type/data_types.hpp"
#include "kernels/matmul_types.hpp"
#include "kernels/data_pack.hpp"
#include "kernels/sparse_data.hpp"
#include "engine_factory.hpp"
#include "model.hpp"
#include "kernels/exposed_enum.hpp"

namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32}, {"s32", dnnl::memory::data_type::s32},
    {"fp16", dnnl::memory::data_type::f16}, {"u8", dnnl::memory::data_type::u8},
    {"s8", dnnl::memory::data_type::s8},    {"bf16", dnnl::memory::data_type::bf16}};

InnerProductOperator::InnerProductOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf),
      src0_perm_({}),
      src1_perm_({}),
      dst_perm_({}),
      output_scale_(1.),
      format_any_(true),
      gelu_split_(false),
      weight_cached_(false),
      has_bias_(false) {
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
  iter = attrs_map.find("per_token");
  if (iter != attrs_map.end()) {
    per_token_ = (attrs_map["per_token"] == "True" || attrs_map["per_token"] == "true");
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
  iter = attrs_map.find("squeeze_dims");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&squeeze_dims_, attrs_map["squeeze_dims"], ",");
  }
  iter = attrs_map.find("append_op");
  binary_add_ = (iter != attrs_map.end() && iter->second == "binary_add") ? true : false;
  append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
  gelu_erf_ = (iter != attrs_map.end() && iter->second == "gelu_erf") ? true : false;
  gelu_tanh_ = (iter != attrs_map.end() && iter->second == "gelu_tanh") ? true : false;
  tanh_ = (iter != attrs_map.end() && iter->second == "tanh") ? true : false;
  sigmoid_ = (iter != attrs_map.end() && iter->second == "sigmoid") ? true : false;
  swish_ = (iter != attrs_map.end() && iter->second == "swish") ? true : false;
  relu_ = (iter != attrs_map.end() && iter->second == "relu") ? true : false;
  append_eltwise_ = (gelu_erf_ && !gelu_split_) || (gelu_tanh_ && !gelu_split_) || tanh_ || sigmoid_ || relu_ || swish_;
  append_op_ = (iter != attrs_map.end()) ? iter->second : "";
  DLOG(INFO) << "append_op: " << append_op_;
  auto wcptr = weight_compression_context::get_instance();
  weight_comp_.type_ = wcptr->global_type_;
}

void InnerProductOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
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
      has_bias_ = false;
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
      has_bias_ = false;
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

void InnerProductOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // set output dtype and primitive attr(without post_ops) in Prepare
  MapTensors(input, output);
  DLOG(INFO) << "inner product has bias add " << has_bias_;
  dst_->set_dtype(output_dtype_);
  if (src0_->dtype() == "fp32" && src1_->dtype() == "fp32") {
    kernel_type_ = Dense;
    weight_zero_ratio_ = GetSparseRatio<float>(static_cast<const float*>(src1_->data()), src1_->shape(), blocksize_);
    if (weight_zero_ratio_ >= sparse_threshold_) kernel_type_ = Dense;
    DLOG(INFO) << "weight zero ratio: " << weight_zero_ratio_;
  } else if (src1_->dtype() == "s8") {
    kernel_type_ = Dense;
    blocksize_ = {4, 16};
    weight_zero_ratio_ = GetSparseRatio<int8_t>(static_cast<const int8_t*>(src1_->data()), src1_->shape(), blocksize_);
    if (weight_zero_ratio_ >= sparse_threshold_) kernel_type_ = Dense;
    DLOG(INFO) << "weight zero ratio: " << weight_zero_ratio_;
  } else if (src0_->dtype() == "s8" && src1_->dtype() == "u8") {
    blocksize_ = {4, 1};
    weight_zero_ratio_ = GetSparseRatio<int8_t>(static_cast<const int8_t*>(src0_->data()), src0_->shape(), blocksize_);
    if (weight_zero_ratio_ >= sparse_threshold_) kernel_type_ = SparseLib;
    DLOG(INFO) << "weight zero ratio: " << weight_zero_ratio_;
  } else if (src1_->dtype() == "bf16") {
    kernel_type_ = Dense;
    auto shape = src1_->shape();
    if (weight_comp_.enabled() && shape.size() == 2 && dst_->dtype() == "bf16") {
      int _N = shape[0], _K = shape[1];
      auto src_bf16 = reinterpret_cast<const jd::bfloat16_t*>(src1_->data());
      if (_N % 32 == 0 && _K % 32 == 0) {  // this limitation will be removed by kernel updates
        jd::tensor_desc src0_desc = {{weight_comp_.PreferedM, shape[1]}, jd::data_type::bf16, jd::format_type::ab};
        jd::tensor_desc src1_desc;
        jd::tensor_desc dst_desc = {{weight_comp_.PreferedM, shape[0]}, jd::data_type::bf16, jd::format_type::ab};
        jd::tensor_desc bias_desc = {{}, jd::data_type::bf16, jd::format_type::a};
        jd::tensor_desc scales_desc = {{}, jd::data_type::fp32, jd::format_type::a};

        if (weight_comp_.type_ == jd::data_type::f8_e4m3) {
          src1_desc = {{shape[0], shape[1]}, jd::data_type::f8_e4m3, jd::format_type::ab};
          weight_comp_.mem_.resize(_N * _K * 1);
          vector<jd::float8_e4m3_t> tmpfp8(_N * _K);
          float8_auto_scale<jd::float8_e4m3_t>::auto_scale_T_bf16(src_bf16, tmpfp8.data(), _N, _K,
                                                                  &weight_comp_.scale_);

          std::function<jd::float8_e4m3_t(jd::float8_e4m3_t)> cast_func_fp8 = [](jd::float8_e4m3_t x) { return x; };
          jd::pack<jd::float8_e4m3_t, jd::float8_e4m3_t>(reinterpret_cast<jd::float8_e4m3_t*>(weight_comp_.mem_.data()),
                                                         tmpfp8.data(), _N, _K, cast_func_fp8);
          weight_comp_.valid = true;
        } else if (weight_comp_.type_ == jd::data_type::s8) {
          src1_desc = {{_N, _K}, jd::data_type::s8, jd::format_type::ab};
          scales_desc = {{_N}, jd::data_type::fp32, jd::format_type::a};
          weight_comp_.mem_.resize(_N * _K * 1);
          weight_comp_.scales_.resize(_N);
          weight_comp_.scale_ = 1.f;
          vector<int8_t> tmp(_N * _K);
          int8_quantize::quantize_T_bf16_percn(src_bf16, tmp.data(), _N, _K, weight_comp_.scales_.data());
          std::function<int8_t(int8_t)> cast_func_int8 = [](int8_t x) { return x; };
          jd::pack<int8_t, int8_t>(reinterpret_cast<int8_t*>(weight_comp_.mem_.data()), tmp.data(), _N, _K,
                                   cast_func_int8);
          weight_comp_.valid = true;
        } else if (weight_comp_.type_ == jd::data_type::f8_e5m2) {
          src1_desc = {{shape[0], shape[1]}, jd::data_type::f8_e5m2, jd::format_type::ab};
          weight_comp_.mem_.resize(_N * _K * 1);
          vector<jd::float8_e5m2_t> tmpfp8(_N * _K);
          float8_auto_scale<jd::float8_e5m2_t>::auto_scale_T_bf16(src_bf16, tmpfp8.data(), _N, _K,
                                                                  &weight_comp_.scale_);
          std::function<jd::float8_e5m2_t(jd::float8_e5m2_t)> cast_func_fp8 = [](jd::float8_e5m2_t x) { return x; };
          jd::pack<jd::float8_e5m2_t, jd::float8_e5m2_t>(reinterpret_cast<jd::float8_e5m2_t*>(weight_comp_.mem_.data()),
                                                         tmpfp8.data(), _N, _K, cast_func_fp8);
          weight_comp_.valid = true;
        }
        if (weight_comp_.valid) {
          std::unordered_map<std::string, std::string> attrs = {{"alpha", std::to_string(weight_comp_.scale_)},
                                                                {"beta", std::to_string(has_bias_ ? 1.f : 0.f)}};
          attrs["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_comp_.mem_.data()));
          if (has_bias_) {
            bias_desc = {{shape[0]}, jd::data_type::bf16, jd::format_type::ab};
          }
          std::vector<jd::tensor_desc> ts_descs = {src0_desc, src1_desc, dst_desc, bias_desc, scales_desc};

          vector<jd::postop_attr> postop_chain;
          if (append_sum_) {
            op_attrs_["postop_list"] = "append_sum";
            jd::tensor_desc append_sum_desc = {
                {weight_comp_.PreferedM, shape[0]}, jd::data_type::bf16, jd::format_type::ab};
            jd::tensor_desc zp0_desc = {{1}, jd::data_type::fp32, jd::format_type::a};
            ts_descs = {src0_desc, src1_desc, dst_desc, bias_desc, scales_desc, zp0_desc, append_sum_desc};
          }
          if (gelu_tanh_) {
            jd::postop_attr gelu_attr(jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::gelu);
            postop_chain.push_back(gelu_attr);
          }
          if (swish_) {
            op_attrs_["postop_list"] = "swish";
            // default swish alpha is 1
            jd::postop_attr gelu_attr(jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::swish, 1);
            postop_chain.push_back(gelu_attr);
          }
          jd::operator_desc op_desc(jd::kernel_kind::transpose_matmul, jd::kernel_prop::forward_inference,
                                    jd::engine_kind::cpu, ts_descs, attrs, postop_chain);

          static jd::engine_factory factory;
          cpu_engine_ = factory.create(jd::engine_kind::cpu, jd::runtime_kind::undef);
          jd::stream_t* stream = nullptr;
          cpu_engine_->create_stream(reinterpret_cast<jd::stream_t**>(&stream));
          cpu_engine_->create_kernel(op_desc, matmul_kernel_, stream);
        }
      }
    }
  }
  int64_t M;
  if (kernel_type_ == SparseLib)
    M = src0_->shape()[!src0_perm_.empty() && src0_perm_ == vector<int64_t>{0, 1} ? 1 : 0];
  else
    M = src1_->shape()[!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1} ? 1 : 0];
  if (M % blocksize_[0] != 0) {
    if (kernel_type_ == Sparse)
      kernel_type_ = Dense;
    else if (kernel_type_ == SparseLib)
      kernel_type_ = Unsupported;
  }

  DLOG(INFO) << "Innerproduct " << name_ << " execute kenel: " << kernel_type_;
  if (kernel_type_ == Unsupported)
    LOG(ERROR) << "Innerproduct not support: " << src0_->dtype() << " X " << src1_->dtype() << " = " << dst_->dtype();

#ifndef __AVX512F__
  if (kernel_type_ != Dense) {
    if (!src1_) {
      kernel_type_ = Dense;
      LOG(ERROR) << "Sparse fp32 kernel in InnerProduct only supports AVX512!";
    } else {
#ifndef __AVX512VNNI__
      kernel_type_ = Dense;
      LOG(ERROR) << "Sparse int8 kernel in InnerProduct only supports AVX512VNNI!";
#endif
    }
  }
#endif
  is_dynamic_ =
      (output.size() > 1) || (src0_min_ != nullptr && src0_min_->raw_data() == nullptr && !src0_min_->is_shared());
  if (is_dynamic_) {
    DLOG(INFO) << this->name() << " is DYNAMIC!!!";
    if (kernel_type_ != Dense) DLOG(ERROR) << "Transpose IP not support dynamic quantization yet!\n";
    kernel_type_ = Runtime;
#ifdef _WIN32
    DLOG(ERROR) << "dynamic quantization did NOT support windows now!!!";
    throw std::string("Windows");
#endif
  }
  switch (kernel_type_) {
    case Dense:
      PrepareDense(input, output);
      break;
    case Sparse:
      PrepareSparse(input, output);
      break;
    case SparseLib:
#ifdef WITH_SPARSELIB
      PrepareSparseLib(input, output);
#else
      LOG(ERROR) << "Sparse lib is not loaded!\n";
#endif
      break;
    case Runtime:
      DynamicPrepare(input, output);
      break;
    default:
      LOG(FATAL) << "Unkown kernrl_type";
  }
  if (kernel_type_ != Dense) monopolize_dispatcher_ = true;
}

void InnerProductOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  switch (kernel_type_) {
    case Dense:
      if (this->do_shape_infer()) break;
      ReshapeDense(input, output);
      break;
    case Sparse:
      ReshapeSparse(input, output);
      break;
    case SparseLib:
#ifdef WITH_SPARSELIB
      if (this->do_shape_infer()) break;
      ReshapeSparseLib(input, output);
#else
      LOG(ERROR) << "Sparse lib is not loaded!\n";
#endif
      break;
    case Runtime:
      DynamicReshape(input, output);
      break;
  }
}

void InnerProductOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  switch (kernel_type_) {
    case Dense:
      ForwardDense(input, output);
      break;
    case Sparse:
#if __AVX512F__
      ForwardSparse(input, output);
#endif
      break;
    case SparseLib:
#ifdef WITH_SPARSELIB
#if __AVX512F__
      ForwardSparseLib(input, output);
#endif
#else
      LOG(ERROR) << "Sparse lib is not loaded!\n";
#endif
      break;
    case Runtime:
      DynamicForward(input, output);
      break;
  }
  this->unref_tensors(input);
}

void InnerProductOperator::ShapeInfer(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (kernel_type_ == Unsupported || kernel_type_ == Sparse) {
    return;
  } else if (kernel_type_ == Dense) {
    ShapeInferDense(input, output);
  } else {
#ifdef WITH_SPARSELIB
    ShapeInferSparseLib(input, output);
#endif
  }
}

void InnerProductOperator::DstReshapeFusion(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (!reshape_.empty()) {
    vector<int64_t> ref_shape;
    if (!reshape_dims_.empty()) {
      ref_shape = input.back()->shape();
    }
    vector<int64_t> reshape(reshape_);
    if (output[0]->shape().size() == 3 && output[0]->tensor_format() == TensorFormat::MmKMb) {
      int64_t micro_bs = output[0]->shape()[0];
      int64_t model_input_bs = model_->input_shape()[0];
      reshape.insert(reshape.begin(), micro_bs);
      // replace batch size of ref_shape
      ref_shape[0] = model_input_bs / micro_bs;
    }
    vector<int64_t> dst_shape = GetDstShape(reshape, output[0]->size(), ref_shape, reshape_dims_);
    output[0]->set_shape(dst_shape);
  }
  if (!squeeze_dims_.empty()) {
    vector<int64_t> dst_shape = output[0]->shape();
    vector<int64_t> squeeze_shape;
    int j = 0;
    int64_t axis = 0;
    for (int i = 0; i < dst_shape.size(); ++i) {
      if (j < squeeze_dims_.size()) {
        axis = squeeze_dims_[j] < 0 ? squeeze_dims_[j] + dst_shape.size() : squeeze_dims_[j];
      }
      if (axis != i) {
        squeeze_shape.push_back(dst_shape[i]);
      } else {
        LOG_IF(FATAL, dst_shape[i] != 1) << "Only support to squeeze axis which has size 1!";
        j++;
      }
    }
    output[0]->set_shape(squeeze_shape);
  }
}

void reorder_dynamic_weight(int8_t* src1, int8_t* dst, int k, int n, int pad_n, bool need_transpose) {
  int8_t* src;
  if (need_transpose) {
    src = new int8_t[k * n];
#pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++)
      for (int j = 0; j < n; j++) {
        src[i * n + j] = src1[j * k + i];
      }
  } else {
    src = src1;
  }
  int8_t* tmp = new int8_t[k * pad_n];
#pragma omp parallel for collapse(2)
  for (int k_loop = 0; k_loop < k / 4; k_loop++) {
    for (int n_loop = 0; n_loop < n; n_loop++) {
      tmp[k_loop * pad_n * 4 + n_loop * 4] = src[k_loop * n * 4 + n_loop];
      tmp[k_loop * pad_n * 4 + n_loop * 4 + 1] = src[k_loop * n * 4 + n + n_loop];
      tmp[k_loop * pad_n * 4 + n_loop * 4 + 2] = src[k_loop * n * 4 + 2 * n + n_loop];
      tmp[k_loop * pad_n * 4 + n_loop * 4 + 3] = src[k_loop * n * 4 + 3 * n + n_loop];
    }
  }
  int tile_k = 64;
  while (k % tile_k != 0) tile_k -= 4;
  auto contain_block_row = k / tile_k;
  auto contain_block_col = pad_n / 16;
  auto block_size = tile_k * 16;
#pragma omp parallel for collapse(4)
  for (int k_loop = 0; k_loop < contain_block_row; k_loop++)
    for (int n_loop = 0; n_loop < contain_block_col; n_loop++)
      for (int i = 0; i < tile_k / 4; i++)
        for (int j = 0; j < 64; j++)
          dst[n_loop * contain_block_row * block_size + k_loop * 64 + i * contain_block_row * 64 + j] =
              tmp[k_loop * contain_block_col * block_size + n_loop * 64 + i * pad_n * 4 + j];
  delete[] tmp;
  if (need_transpose) delete[] src;
}

void InnerProductOperator::DynamicReshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (src0_max_->size() > 1) {
    if (!per_token_ && output_dtype_ != "fp32")
      DLOG(ERROR) << "Quantized inputs with pert_batch should have fp32 output or per_token quantization output";
    vector<int64_t> src0_shape_origin = src0_->shape();
    vector<int64_t> src0_shape = GetShapes(src0_shape_origin, src0_perm_);
    src0_->set_shape(src0_shape);
    vector<int64_t> src1_shape = src1_->shape();
    vector<int64_t> dst_shape_origin = {src0_shape[0], src1_shape[0]};
    vector<int64_t> dst_shape = GetShapes(dst_shape_origin, dst_perm_);
    dst_->set_shape(dst_shape);
    if (output.size() > 1) {
      dst_min_->set_shape({dst_->shape()[0]});
      dst_max_->set_shape({dst_->shape()[0]});
    }
    vector<jd::tensor_desc> tensor_desc(jd::exposed_enum::dynamic_quant_matmul::SIZE, jd::tensor_desc());
    rt_data_.resize(jd::exposed_enum::dynamic_quant_matmul::SIZE);
    tensor_desc[jd::exposed_enum::dynamic_quant_matmul::ACTIVATION] =
        jd::tensor_desc(src0_shape, jd::data_type::s8, jd::format_type::ab);
    tensor_desc[jd::exposed_enum::dynamic_quant_matmul::SCALE_A] =
        jd::tensor_desc(src0_max_->shape(), jd::data_type::fp32, jd::format_type::undef);
    tensor_desc[jd::exposed_enum::dynamic_quant_matmul::WEIGHT] =
        jd::tensor_desc(GetShapes(src1_shape, {1, 0}), jd::data_type::s8, jd::format_type::undef);
    tensor_desc[jd::exposed_enum::dynamic_quant_matmul::SCALE_W] =
        jd::tensor_desc(src1_max_->shape(), jd::data_type::fp32, jd::format_type::undef);
    tensor_desc[jd::exposed_enum::dynamic_quant_matmul::DST] =
        jd::tensor_desc(dst_shape, type2sparsemem_[output_dtype_], jd::format_type::ab);
    if (output_dtype_ == "s8")
      tensor_desc[jd::exposed_enum::dynamic_quant_matmul::SCALE_DST] =
          jd::tensor_desc({dst_shape[0]}, jd::data_type::fp32, jd::format_type::undef);
    if (has_bias_)
      tensor_desc[jd::exposed_enum::dynamic_quant_matmul::BIAS] =
          jd::tensor_desc(bias_->shape(), jd::data_type::fp32, jd::format_type::undef);
    op_attrs_["large_wei_threshold"] = "0.8";
    std::vector<jd::postop_attr> postop_attrs;
    if (gelu_tanh_) {
      op_attrs_["postop_list"] = "gelu";
      postop_attrs.push_back({jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::gelu});
    }
    if (swish_) {
      op_attrs_["postop_list"] = "swish";
      postop_attrs.push_back({jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::swish, 1.f});
    }
    if (append_sum_ && output_dtype_ != "s8") op_attrs_["append_sum"] = "true";
    jd::operator_desc op_desc(jd::kernel_kind::dynamic_quant_matmul, jd::kernel_prop::forward_inference,
                              jd::engine_kind::cpu, tensor_desc, op_attrs_, postop_attrs);
    jd::dynamic_quant_matmul_desc dynamic_quant_matmul_desc(op_desc);
    dynamic_quant_matmul_ker_ = jd::dynamic_quant_matmul(dynamic_quant_matmul_desc);
    if (scratchpad_) MemoryAllocator::get().UnrefMemory(scratchpad_);
    scratchpad_ = MemoryAllocator::get().GetMemory(dynamic_quant_matmul_ker_.get_workspace_size(), 1);
    if (!weight_cached_) {
      int k = src1_shape[1], n = src1_shape[0];
      int pad_n = ((n + 15) / 16) * 16;
      if (transposed_weight_) MemoryAllocator::get().UnrefMemory(transposed_weight_);
      transposed_weight_ = MemoryAllocator::get().GetMemory(k * pad_n, 1);
      reorder_dynamic_weight(reinterpret_cast<int8_t*>(src1_->mutable_data()),
                             reinterpret_cast<int8_t*>(transposed_weight_), k, n, pad_n,
                             src1_perm_.empty() || src1_perm_ == vector<int64_t>{0, 1});
      weight_cached_ = true;
    }
    DstReshapeFusion(input, output);
  } else {
    ReshapeDense(input, output);
    vector<int64_t> dst_shape = dst_->shape();
    if (per_token_) {
      dst_min_->set_shape({dst_->shape()[0]});
      dst_max_->set_shape({dst_->shape()[0]});
      jd::tensor_desc src_desc = {dst_shape, jd::data_type::fp32, jd::format_type::undef};
      jd::tensor_desc dst_desc = {dst_shape, jd::data_type::s8, jd::format_type::undef};
      jd::tensor_desc scale_desc = {{dst_shape[0]}, jd::data_type::fp32, jd::format_type::undef};
      jd::operator_desc op_desc(jd::kernel_kind::dynamic_quant, jd::kernel_prop::forward_inference,
                                jd::engine_kind::cpu, {src_desc, dst_desc, scale_desc}, {{"input_dt", "fp32"}});
      jd::dynamic_quant_desc dynamic_quant_desc(op_desc);
      dynamic_quant_ker_ = jd::dynamic_quant(dynamic_quant_desc);
    } else {
      if (output.size() > 1) {
        dst_min_->set_shape({1});
        dst_max_->set_shape({1});
      }
    }
  }
}
void InnerProductOperator::DynamicForward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (src0_max_->size() > 1) {
    void* dst_data = dst_->mutable_data();
    if (post_ != nullptr && !binary_add_) {
      DLOG(INFO) << "inner product has post op " << post_->name();
      void* post_data_ptr = const_cast<void*>(post_->data());
      auto life_count = MemoryAllocator::get().CheckMemory(post_data_ptr);
      // MemoryAllocate::check_tensor_life
      if (life_count == 1) {
        post_->unref_data(true);
        dst_->set_data(post_data_ptr);
        dst_data = post_data_ptr;
      } else {
        int data_size = post_->size();
        string data_type = post_->dtype();
        memcpy(dst_data, post_data_ptr, data_size * type2bytes[data_type]);
        DLOG(WARNING) << "post tensor will be used by multi node...";
      }
    }
    rt_data_[jd::exposed_enum::dynamic_quant_matmul::ACTIVATION] = src0_->data();
    rt_data_[jd::exposed_enum::dynamic_quant_matmul::SCALE_A] = src0_max_->data();
    rt_data_[jd::exposed_enum::dynamic_quant_matmul::WEIGHT] = transposed_weight_;
    rt_data_[jd::exposed_enum::dynamic_quant_matmul::SCALE_W] = src1_max_->data();
    rt_data_[jd::exposed_enum::dynamic_quant_matmul::DST] = dst_data;
    rt_data_[jd::exposed_enum::dynamic_quant_matmul::WORKSPACE] = scratchpad_;
    if (output_dtype_ == "s8") rt_data_[jd::exposed_enum::dynamic_quant_matmul::SCALE_DST] = dst_max_->data();
    if (has_bias_) {
      rt_data_[jd::exposed_enum::dynamic_quant_matmul::BIAS] = bias_->data();
    }
    dynamic_quant_matmul_ker_.execute(rt_data_);
  } else {
    inner_product_dynamic_res_ = new Tensor(*dst_);
    inner_product_dynamic_res_->set_dtype(output_dtype_ == "bf16" ? "bf16" : "fp32");
    ForwardDense(input, output);
    //  quantize the fp32 result of inner_porduct
    if (output.size() > 1) {
      if (per_token_) {
        dynamic_quant_ker_.execute({
            inner_product_dynamic_res_->data(),
            dst_->data(),
            dst_max_->data(),
        });
        inner_product_dynamic_res_->unref_data();
      } else {
        runtime_minmax(reinterpret_cast<float*>(inner_product_dynamic_res_->mutable_data()),
                       inner_product_dynamic_res_->size(), reinterpret_cast<float*>(dst_min_->mutable_data()),
                       reinterpret_cast<float*>(dst_max_->mutable_data()));
        // quantize
        if (output_dtype_ == "u8" || output_dtype_ == "s8") {
          auto scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
          float* dst_max_data = reinterpret_cast<float*>(dst_max_->mutable_data());
          *dst_max_data = 1.0 / scales_[0];
          // memcpy(dst_max_->mutable_data(), scales_.data(), dst_max_->size() * sizeof(float));
#if __AVX512F__
          Quantize_avx512(inner_product_dynamic_res_->size(), dst_->dtype(), inner_product_dynamic_res_->data(),
                          static_cast<const float*>(dst_min_->data()), scales_, dst_->mutable_data());
#else
          Quantize(inner_product_dynamic_res_->size(), dst_->dtype(), inner_product_dynamic_res_->data(),
                   static_cast<const float*>(dst_min_->data()), scales_, dst_->mutable_data());
#endif
          inner_product_dynamic_res_->unref_data();
        } else {
          // copy fp32_res to dst if not quantize
          void* res_ptr = const_cast<void*>(inner_product_dynamic_res_->data());
          inner_product_dynamic_res_->unref_data(true);
          dst_->set_data(res_ptr);
        }
      }
    } else {
      void* res_ptr = const_cast<void*>(inner_product_dynamic_res_->data());
      inner_product_dynamic_res_->unref_data(true);
      dst_->set_data(res_ptr);
    }
  }
}
void InnerProductOperator::DynamicPrepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  PrepareDense(input, output);
  if (per_token_ && output_dtype_ != "s8") DLOG(ERROR) << "Dynamic Innerproduct with per_token only support output s8";
}

void InnerProductOperator::PrepareSparse(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t> weight_shape = src1_->shape();
  vector<int64_t> weight_shape_perm = weight_shape;
  if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
    weight_shape_perm = {weight_shape[1], weight_shape[0]};
  }
  src1_->set_shape(weight_shape_perm);
  if (!src1_min_) {  // fp32 kernel prepare
    const float* weight_data = static_cast<const float*>(src1_->data());
    float* weight_data_perm = static_cast<float*>(malloc(src1_->size() * sizeof(float)));
    memcpy(weight_data_perm, weight_data, src1_->size() * sizeof(float));
    if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
      TransposeMatrix<float>(weight_data, weight_shape, weight_data_perm);
    }
    sparse_weight_ = create_bsc_matrix<float>(weight_data_perm, weight_shape_perm, blocksize_);
    free(weight_data_perm);
  } else {  // int8 kernel prepare
    const int8_t* weight_data = static_cast<const int8_t*>(src1_->data());
    int8_t* weight_data_perm = static_cast<int8_t*>(malloc(src1_->size() * sizeof(int8_t)));
    memcpy(weight_data_perm, weight_data, src1_->size() * sizeof(int8_t));
    if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
      TransposeMatrix<int8_t>(weight_data, weight_shape, weight_data_perm);
    }
    sparse_weight_int8_ = create_bsc_matrix<int8_t>(weight_data_perm, weight_shape_perm, blocksize_);
    reorder_bsc_int8_4x16(sparse_weight_int8_);
    free(weight_data_perm);
    vector<float> src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
    vector<float> src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
    vector<float> dst_scales = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
    vector<float> rescales;
    for (int i = 0; i < src1_scales.size(); i++) {
      rescales.emplace_back(dst_scales[0] / (src0_scales[0] * src1_scales[i]));
    }
    rescales_ = rescales;
  }
}

void InnerProductOperator::ReshapeSparse(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<int64_t> src0_shape_origin = src0_->shape();
  vector<int64_t> src0_shape = src0_shape_origin;
  if (!src0_perm_.empty() && src0_perm_ == vector<int64_t>{0, 1}) {
    src0_shape = {src0_shape[1], src0_shape[0]};
  }
  src0_->set_shape(src0_shape);

  vector<int64_t> dst_shape = {src0_shape[0], src1_->shape()[1]};
  dst_->set_shape(dst_shape);
}

#if __AVX512F__
void InnerProductOperator::ForwardSparse(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int64_t M = src0_->shape()[0];
  int64_t N = src1_->shape()[1];
  int64_t K = src0_->shape()[1];
  // fp32 kernel
  if (!src1_min_) {
    const int64_t* rowidxs = sparse_weight_->rowidxs;
    const int64_t* colptr = sparse_weight_->colptr;
    const int64_t ncolptr = sparse_weight_->ncolptr;
    const float* A = static_cast<const float*>(src0_->data());
    const float* B = static_cast<const float*>(sparse_weight_->data);
    float* C = static_cast<float*>(dst_->mutable_data());
    if (has_bias_) {
      const float* bias = static_cast<const float*>(bias_->data());
      if (append_op_ == "") {
        sparse_gemm_bsc_bias_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, C, M_NBLK_);
      } else if (append_op_ == "relu") {
        sparse_gemm_bsc_bias_relu_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, C, M_NBLK_);
      } else if (append_op_ == "sum") {
        const float* post = static_cast<const float*>(post_->data());
        sparse_gemm_bsc_bias_sum_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, post, C, M_NBLK_);
      } else if (append_op_ == "tanh") {
        sparse_gemm_bsc_bias_tanh_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, C, M_NBLK_);
      } else if (append_op_ == "gelu_tanh") {
        sparse_gemm_bsc_bias_gelu_tanh_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, C, M_NBLK_);
      } else if (append_op_ == "sigmoid") {
        sparse_gemm_bsc_bias_sigmod_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, C, M_NBLK_);
      } else {
        DLOG(INFO) << "inner product has no such sparse kernel, output tensor is" << output[0]->name();
      }
    } else {
      if (append_op_ == "") {
        sparse_gemm_bsc_f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, C, M_NBLK_);
      }
    }
  } else {  // int8 kernel
#if __AVX512VNNI__
    const int64_t* rowidxs = sparse_weight_int8_->rowidxs;
    const int64_t* colptr = sparse_weight_int8_->colptr;
    const int64_t ncolptr = sparse_weight_int8_->ncolptr;
    const uint8_t* A = static_cast<const uint8_t*>(src0_->data());
    const int8_t* B = static_cast<const int8_t*>(sparse_weight_int8_->data);
    if (src1_->size() > 1) {  // per channel kernel
      if (output[0]->dtype() == "u8") {
        uint8_t* C = static_cast<uint8_t*>(dst_->mutable_data());
        if (has_bias_) {
          const int32_t* bias = static_cast<const int32_t*>(bias_->data());
          if (append_op_ == "relu") {
            sparse_gemm_bsc_4x16_u8s8u8_pc_relu(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_, C,
                                                M_NBLK_);
          }
        }
      } else if (output[0]->dtype() == "s8") {
        int8_t* C = static_cast<int8_t*>(dst_->mutable_data());
        if (has_bias_) {
          const int32_t* bias = static_cast<const int32_t*>(bias_->data());
          if (append_op_ == "") {
            sparse_gemm_bsc_4x16_u8s8s8_pc(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_, C,
                                           M_NBLK_);
          }
        }
      }
    } else {  // per tensor kernel
      if (output[0]->dtype() == "fp32") {
        float* C = static_cast<float*>(dst_->mutable_data());
        if (has_bias_) {
          const int32_t* bias = static_cast<const int32_t*>(bias_->data());
          if (append_op_ == "") {
            sparse_gemm_bsc_4x16_u8s8f32(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_[0], C,
                                         M_NBLK_);
          } else if (append_op_ == "relu") {
            sparse_gemm_bsc_4x16_u8s8f32_relu(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_[0],
                                              C, M_NBLK_);
          }
        }
      } else if (output[0]->dtype() == "u8") {
        uint8_t* C = static_cast<uint8_t*>(dst_->mutable_data());
        if (has_bias_) {
          const int32_t* bias = static_cast<const int32_t*>(bias_->data());
          if (append_op_ == "relu") {
            sparse_gemm_bsc_4x16_u8s8u8_relu(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_[0], C,
                                             M_NBLK_);
          }
        }
      } else if (output[0]->dtype() == "s8") {
        int8_t* C = static_cast<int8_t*>(dst_->mutable_data());
        if (has_bias_) {
          const int32_t* bias = static_cast<const int32_t*>(bias_->data());
          if (append_op_ == "") {
            sparse_gemm_bsc_4x16_u8s8s8(M, N, K, A, B, rowidxs, colptr, ncolptr, blocksize_, bias, rescales_[0], C,
                                        M_NBLK_);
          }
        }
      }
    }
#endif
  }
}
#endif

#ifdef WITH_SPARSELIB
void InnerProductOperator::PrepareSparseLib(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // Step 1: Construct operator config
  op_attrs_ = {{"mkn_blocks", "1,1,1"}, {"tile_shape", "4,4"}, {"append_sum", append_sum_ ? "true" : ""}};
  // Step 2: sparse data encoding
  auto sparse_ptr = new jd::bsr_data_t<int8_t>(jd::spns::reorder_to_bsr_group<int8_t, 4>(
      src0_->shape()[0], src0_->shape()[1], 4, 1, static_cast<const int8_t*>(src0_->data())));
  op_attrs_["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));

  // Step 3: prepare desc and calculate scale for static quantization

  if (!is_dynamic_ && (output_scale_ != 1.f || src0_min_ != nullptr || src1_min_ != nullptr)) {
    int ic_dim = 0;
    if (src0_min_ != nullptr && src1_max_ != nullptr) {
      ic_dim = src1_min_->size() > 1 ? 0 | (1 << 1) : 0;
      vector<float> src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
      vector<float> src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
      if (dst_min_ != nullptr) {
        dst_scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
      }
      rescales_ = GetRescales(src1_scales, src0_scales, dst_scales_, dst_->dtype(), append_eltwise_);
    } else {
      rescales_ = vector<float>(1, 1.f);
    }
    if (output_scale_ != 1.f) {
      for (int i = 0; i < rescales_.size(); i++) {
        rescales_[i] *= output_scale_;
      }
    }
    scales_desc_ = {{(int64_t)rescales_.size(), 1}, jd::data_type::fp32, jd::format_type::ab};
  }
  // cache weight here, save weight and bias memory descriptor
  src0_shape_origin_ = src0_->shape();
  vector<int64_t> src0_shape = GetShapes(src0_shape_origin_, src0_perm_);
  vector<int64_t> src0_stride = GetStrides(src0_shape_origin_, src0_perm_);
  src0_->set_shape(src0_shape);
  src0_desc_ = {src0_->shape(), jd::data_type::s8, jd::format_type::bsr};
  if (has_bias_) bias_desc_ = {bias_->shape(), jd::data_type::s32, jd::format_type::ab};
  // set sparse weight, src activation, dst activation and append_sum post
  // activation tensor format
  src0_->set_tensor_format(TensorFormat::NK);
  src1_->set_tensor_format(TensorFormat::KM);
  dst_->set_tensor_format(TensorFormat::KM);
  if (post_ != nullptr && !binary_add_) post_->set_tensor_format(TensorFormat::KM);
}

void InnerProductOperator::ShapeInferSparseLib(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // shape infer for dispatcher
  // save the SparseLib primitive class creation time
  if (dispatch_from_ == "InnerProduct" && !dispatch_config_.empty() && dispatch_config_[0] == "SparseLib") {
    CHECK_EQ(dispatch_kernel_config["InnerProduct_to_SparseLib"].size(), dispatch_config_.size() - 1)
        << "InnerProduct to SparseLib has wrong dispatch kernel config...";
    DLOG(INFO) << "Operator " << name_ << " dispatch configs are " << dispatch_config_[0] << "," << dispatch_config_[1]
               << "," << dispatch_config_[2] << "," << dispatch_config_[3];
    // pass 3D shape and tensor_format
    vector<int64_t> src1_3d_shape;
    StringSplit<int64_t>(&src1_3d_shape, dispatch_config_[1], ",");
    dst_->set_tensor_format(TensorFormat::MmKMb);

    dst_->set_shape({src1_3d_shape[0], src0_->shape()[0], src1_3d_shape[2]});
  } else {
    vector<int64_t> src1_shape = src1_->shape();
    if (src1_shape.size() == 2) {
      if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
        src1_shape = {src1_shape[1], src1_shape[0]};
      }
      dst_->set_shape({src0_->shape()[0], src1_shape[1]});
      src0_->set_tensor_format(TensorFormat::NK);
      src1_->set_tensor_format(TensorFormat::KM);
      dst_->set_tensor_format(TensorFormat::KM);
      if (post_ != nullptr && !binary_add_) post_->set_tensor_format(TensorFormat::KM);
    } else {
      dst_->set_shape({src1_shape[0], src0_->shape()[0], src1_shape[2]});
      dst_->set_tensor_format(TensorFormat::MmKMb);
    }
  }
  DstReshapeFusion(input, output);
}

void InnerProductOperator::ReshapeSparseLib(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<int64_t> src1_shape_origin = src1_->shape();
  vector<int64_t> src1_shape = src1_shape_origin;

  // set dispatch config from tuning
  if (dispatch_from_ == "InnerProduct" && !dispatch_config_.empty() && dispatch_config_[0] == "SparseLib") {
    // e.g. dispatch_config_ = {"SparseLib", "1,256,128", "1,1,1", "4,4", "1"};
    CHECK_EQ(dispatch_kernel_config["InnerProduct_to_SparseLib"].size(), dispatch_config_.size() - 1)
        << "InnerProduct to SparseLib has wrong dispatch kernel config...";
    // 3D
    vector<int64_t> src1_3d_shape;
    StringSplit<int64_t>(&src1_3d_shape, dispatch_config_[1], ",");
    op_attrs_["micro_oc"] = dispatch_config_[2];
    op_attrs_["sub_func"] = dispatch_config_[3];

    vector<int64_t> dst_3d_shape = {src1_3d_shape[0], src0_->shape()[0], src1_3d_shape[2]};
    vector<int64_t> dst_2d_shape = {src0_->shape()[0], src1_3d_shape[0] * src1_3d_shape[2]};
    // pass SparseLib gemm 3D shape in INFERENCE mode
    // the ops after it will do attrs and tensor adaptation
    if (this->get_execution_mode() == ExecutionMode::INFERENCE) {
      src1_->set_shape(src1_3d_shape);
      dst_->set_shape(dst_3d_shape);
    } else {
      src1_->set_shape(src1_shape);
      dst_->set_shape(dst_2d_shape);
    }
    src1_desc_ = {src1_3d_shape, jd::data_type::u8, jd::format_type::ab};
    dst_desc_ = {dst_3d_shape, type2sparsemem_[dst_->dtype()], jd::format_type::ab};
  } else {
    vector<int64_t> dst_shape;
    // 2D
    if (input[1]->shape().size() == 2) {
      if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
        src1_shape = {src1_shape[1], src1_shape[0]};
      }
      dst_shape = {src0_->shape()[0], src1_shape[1]};
      // 3D
    } else if (input[1]->shape().size() == 3) {
      dst_shape = {src1_shape[0], src0_->shape()[0], src1_shape[2]};
    } else {
      LOG(FATAL) << "Wrong input shape for InnerProduct SparseLib, must be 2D "
                    "or 3D...";
    }
    src1_->set_shape(src1_shape);
    src1_desc_ = {src1_shape, jd::data_type::u8, jd::format_type::ab};
    dst_->set_shape(dst_shape);
    dst_desc_ = {dst_shape, type2sparsemem_[dst_->dtype()], jd::format_type::ab};
  }

  vector<jd::postop_attr> postop_chain;

  if (gelu_tanh_) {
    jd::postop_attr gelu_attr(jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::gelu);
    postop_chain.push_back(gelu_attr);
    op_attrs_["postop_list"] = "fp32gelu";
  }

  if (gelu_tanh_ && (output_dtype_ == "u8" || output_dtype_ == "s8")) {
    float zp, scale;
    const float* min_p = static_cast<const float*>(dst_min_->data());
    const float* max_p = static_cast<const float*>(dst_max_->data());
    scale = (max_p[0] - min_p[0]) / 255;
    zp = -min_p[0] / scale;
    assert(dst_->dtype() == "s8" || dst_->dtype() == "u8");
    jd::postop_attr quantize_attr(type2sparsemem_[dst_->dtype()], jd::postop_type::eltwise, jd::postop_alg::quantize,
                                  zp, 0, scale);
    op_attrs_["postop_list"] +=
        "+" + dst_->dtype() + "quantize" + "scale" + std::to_string(scale) + "zp" + std::to_string(zp);
    postop_chain.push_back(quantize_attr);
  }

  vector<jd::tensor_desc> ts_descs = {src0_desc_, src1_desc_, bias_desc_, dst_desc_, scales_desc_};
  jd::operator_desc op_desc(jd::kernel_kind::sparse_matmul, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs_, postop_chain);
  jd::sparse_matmul_desc spmm_desc(op_desc);
  spmm_kern_ = jd::sparse_matmul(spmm_desc);
  DstReshapeFusion(input, output);
}

#if __AVX512F__
void InnerProductOperator::ForwardSparseLib(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  void* dst_data = dst_->mutable_data();
  // has op: append_sum
  if (post_ != nullptr && !binary_add_) {
    DLOG(INFO) << "inner product has post op " << post_->name();
    // The sum primitive requires all source and destination tensors to have the
    // same shape. Implicit broadcasting is not supported.
    void* post_data_ptr = const_cast<void*>(post_->data());
    auto life_count = MemoryAllocator::get().CheckMemory(post_data_ptr);
    // MemoryAllocate::check_tensor_life
    if (life_count == 1 && this->get_execution_mode() != ExecutionMode::DEBUG) {
      post_->unref_data(true);
      dst_->set_data(post_data_ptr);
      dst_data = post_data_ptr;
    } else {
      int data_size = post_->size();
      string data_type = post_->dtype();
      memcpy(dst_data, post_data_ptr, data_size * type2bytes[data_type]);
      LOG(WARNING) << "post tensor will be used by multi node...";
    }
  }
  std::vector<const void*> runtime_data = {src0_->data(), src1_->data(), has_bias_ ? bias_->data() : nullptr, dst_data,
                                           rescales_.data()};
  spmm_kern_.execute(runtime_data);
}
#endif

#endif

void InnerProductOperator::AdaptTensors(const vector<Tensor*>& input, const vector<Tensor*>& output,
                                        const string& stage) {
  if (stage == "in") {
    // reorder 2d to 3d in SparseLib
    // 2D dense: [256, 768] x [768 328] -> [256, 328]
    // 2D sparselib: [328, 768] x [768, 256] -> [328, 256]
    // dispatch 3D sparselib: [328, 768] x [2, 768, 128] -> [2, 328, 128]
    // reorder src and post activation
    if (kernel_type_ == SparseLib) {
      // reshape to 3d then reorder
      vector<int64_t> src1_3d_shape;
      if (!dispatch_config_.empty() && dispatch_config_[0] == "SparseLib") {
        StringSplit<int64_t>(&src1_3d_shape, dispatch_config_[1], ",");
      }
      if (src1_3d_shape.empty()) {
        src1_3d_shape = input[1]->shape();
      }
      if (src1_3d_shape.size() == 3) {
        if (input[1]->tensor_format() == TensorFormat::KM) {
          vector<int64_t> src1_3d_shape_origin = {src1_3d_shape[1], src1_3d_shape[0], src1_3d_shape[2]};
          input[1]->reorder(src1_3d_shape_origin);
          input[1]->set_tensor_format(TensorFormat::MmKMb);
          DLOG(INFO) << "Reorder src1 tensor from KM to MmKMb of operator " << name_;
        }
        if (post_ != nullptr && !binary_add_ && post_->tensor_format() == TensorFormat::KM) {
          vector<int64_t> post_3d_shape_origin = {src0_->shape()[0], src1_3d_shape[0], src1_3d_shape[2]};
          post_->reorder(post_3d_shape_origin);
          post_->set_tensor_format(TensorFormat::MmKMb);
          DLOG(INFO) << "Reorder post tensor from KM to MmKMb of operator " << name_;
        }
        // set dst activation tensor format
        output[0]->set_tensor_format(TensorFormat::MmKMb);
        output[0]->set_shape({src1_3d_shape[0], src0_->shape()[0], src1_3d_shape[2]});
        DstReshapeFusion(input, output);
      }
    } else if (kernel_type_ == Dense) {
      // SparseLib 3D gemm - Dense gemm (no Reorder between)
      // BatchMatmul (receive SparseLib 3d format) - Reshape - Dense gemm
      // (reshape does not change format)
      if (input[0]->tensor_format() == TensorFormat::MmKMb || input[0]->tensor_format() == TensorFormat::BmHnHsBbS) {
        if (input[0]->tensor_format() == TensorFormat::MmKMb) input[0]->reorder(input[0]->shape(), {0, 2, 1});
        input[0]->set_tensor_format(TensorFormat::MK);
        output[0]->set_tensor_format(TensorFormat::MK);
        input[0]->set_shape({input[0]->shape()[0] * input[0]->shape()[1], input[0]->shape()[2]});
      } else {
        return;
      }
    } else {
      return;
    }
  } else if (stage == "out") {
    if (kernel_type_ == SparseLib) {
      // INFERENCE mode will try to reduce order operations times as few as
      // possible
      if (this->get_execution_mode() != ExecutionMode::INFERENCE) {
        // reorder dst, src and post activation back (optional)
        // DEBUG and TUNING mode require all the tensors' format to keep same
        // before and after the op
        if (!dispatch_config_.empty() && dispatch_config_[0] == "SparseLib") {
          vector<int64_t> src1_3d_shape;
          StringSplit<int64_t>(&src1_3d_shape, dispatch_config_[1], ",");
          vector<int64_t> dst_3d_shape_origin = {src1_3d_shape[0], src0_->shape()[0], src1_3d_shape[2]};
          output[0]->reorder(dst_3d_shape_origin);
          output[0]->set_tensor_format(TensorFormat::KM);
          output[0]->set_shape({output[0]->shape()[0], output[0]->shape()[1] * output[0]->shape()[2]});
          DstReshapeFusion(input, output);
          DLOG(INFO) << "Reorder dst tensor from MmKMb to KM of operator " << name_;
          // reorder src and post back
          input[1]->set_tensor_format(TensorFormat::KM);
          if (input[1]->left_life() > 0) input[1]->reorder(src1_3d_shape);
          input[1]->set_shape({src1_3d_shape[1], src1_3d_shape[0] * src1_3d_shape[2]});
          DLOG(INFO) << "Reorder src1 tensor from MmKMb to KM of operator " << name_;
          if (post_ != nullptr && !binary_add_) {
            post_->set_tensor_format(TensorFormat::KM);
            if (post_->left_life() > 0) post_->reorder(dst_3d_shape_origin);
            post_->set_shape(output[0]->shape());
            DLOG(INFO) << "Reorder post tensor from MmKMb to KM of operator " << name_;
          }
        }
      }
    }
  } else {
    LOG(WARNING) << "Wrong stage parameter, should be in or out...";
  }
}

void InnerProductOperator::ResetOpStatus(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (kernel_type_ == Dense) {
    src0_->set_tensor_format(TensorFormat::MK);
    src1_->set_tensor_format(TensorFormat::KN);
    dst_->set_tensor_format(TensorFormat::MK);
    if (post_ != nullptr && !binary_add_) post_->set_tensor_format(TensorFormat::MK);
  } else if (kernel_type_ == SparseLib) {
    src0_->set_tensor_format(TensorFormat::NK);
    if (src1_->tensor_format() == TensorFormat::MmKMb) src1_->set_tensor_format(TensorFormat::KM);
    if (dst_->tensor_format() == TensorFormat::MmKMb) {
      dst_->set_tensor_format(TensorFormat::KM);
    }
    if (post_ != nullptr && !binary_add_ && post_->tensor_format() == TensorFormat::MmKMb) {
      post_->set_tensor_format(TensorFormat::KM);
    }
  } else {
    return;
  }
}

void InnerProductOperator::PrepareDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (!is_dynamic_ && (output_scale_ != 1.f || src0_min_ != nullptr || src1_min_ != nullptr)) {
    int ic_dim = 0;
    if (src0_min_ != nullptr && src1_max_ != nullptr) {
      ic_dim = src1_min_->size() > 1 ? 0 | (1 << 1) : 0;
      vector<float> src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
      vector<float> src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
      if (dst_min_ != nullptr) {
        dst_scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
      }
      rescales_ = GetRescales(src0_scales, src1_scales, dst_scales_, dst_->dtype(), append_eltwise_);
    } else {
      rescales_ = vector<float>(1, 1.f);
    }
    if (output_scale_ != 1.f) {
      for (int i = 0; i < rescales_.size(); i++) {
        rescales_[i] *= output_scale_;
      }
    }
    attr_.set_output_scales(ic_dim, rescales_);
  }
  // cache weight here, save weight and bias memory descriptor
  src1_shape_origin_ = src1_->shape();
  vector<int64_t> src1_shape = GetShapes(src1_shape_origin_, src1_perm_);
  vector<int64_t> src1_stride = GetStrides(src1_shape_origin_, src1_perm_);
  src1_->set_shape(src1_shape);
  any_src1_md_ = memory::desc(src1_shape, type2mem[src1_->dtype()], memory::format_tag::any);
  src1_md_ = memory::desc(src1_shape, type2mem[src1_->dtype()], src1_stride);
  src1_m_ = memory(src1_md_, eng_, src1_->mutable_data());
  // for weight cache
  any_src1_m_last_ = memory(src1_md_, eng_, src1_->mutable_data());
  if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{1, 0}) src1_->set_transpose();
  if (!src0_perm_.empty() && src0_perm_ == vector<int64_t>{1, 0}) src0_->set_transpose();

  if (has_bias_ || (is_dynamic_ && src0_->dtype() == "u8")) {
    vector<int64_t> bias_shape = {src1_shape[0]};
    vector<int64_t> bias_stride = GetStrides(bias_shape);
    if (has_bias_) {
      bias_md_ = memory::desc(bias_shape, type2mem[bias_->dtype()], bias_stride);
      any_bias_md_ = memory::desc(bias_shape, type2mem[bias_->dtype()], memory::format_tag::any);
      bias_m_ = memory(bias_md_, eng_, bias_->mutable_data());
      any_bias_m_last_ = memory(bias_md_, eng_, bias_->mutable_data());
    } else {
      bias_md_ = memory::desc(bias_shape, dnnl::memory::data_type::f32, bias_stride);
      any_bias_md_ = memory::desc(bias_shape, dnnl::memory::data_type::f32, memory::format_tag::any);
      bias_m_ = memory(bias_md_, eng_);
    }
  }
  // set src activation, dense weight, dst activation and append_sum post
  // activationtensor format
  src0_->set_tensor_format(TensorFormat::MK);
  src1_->set_tensor_format(TensorFormat::KN);
  dst_->set_tensor_format(TensorFormat::MK);
  if (post_ != nullptr && !binary_add_) post_->set_tensor_format(TensorFormat::MK);
}

void InnerProductOperator::CalculateCompensation(const vector<int64_t>& src1_shape, const vector<int64_t>& src1_stride,
                                                 const vector<int64_t>& zero_point_stride) {
  int src1_length = src1_shape.size();
  int64_t zp_size = src1_->size() / src1_shape[src1_length - 1];
  // calculate the compensation=inner_product(ones_like(src0),src1)
  int8_t* weight_data = reinterpret_cast<int8_t*>(src1_->mutable_data());
  compensation_.resize(zp_size);
  int block_stride = (src1_length > 2) ? src1_stride[src1_length - 3] : 0;
  int offset_stride = src1_stride[src1_length - 2];
  int step_stride = src1_stride[src1_length - 1];
#pragma omp parallel for
  for (int i = 0; i < zp_size; i++) {
    compensation_[i] = 0;
    int outer = (i / zero_point_stride[src1_length - 2]) * block_stride +
                (i % zero_point_stride[src1_length - 2]) * offset_stride;
    for (int j = 0; j < src1_shape[src1_length - 1]; j++) compensation_[i] += weight_data[outer + j * step_stride];
  }
}

void InnerProductOperator::ShapeInferDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // shape infer for dispatcher
  // save the OneDNN primitive forward class creation time
  vector<int64_t> src0_shape_origin = src0_->shape();
  vector<int64_t> src0_shape = GetShapes(src0_shape_origin, src0_perm_);
  vector<int64_t> dst_shape_origin = {src0_shape[0], src1_->shape()[0]};
  vector<int64_t> dst_shape = GetShapes(dst_shape_origin, dst_perm_);
  dst_->set_shape(dst_shape);
  if (output.size() > 1) {
    dst_min_->set_shape({1});
    dst_max_->set_shape({1});
  }
  DstReshapeFusion(input, output);
}

// 1. Create primitive
void InnerProductOperator::ReshapeDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  dnnl::post_ops po;
  vector<int64_t> src1_shape = src1_->shape();
  vector<int64_t> src1_stride = GetStrides(src1_shape_origin_, src1_perm_);

  if (is_dynamic_ && (output_scale_ != 1.f || src0_min_ != nullptr || src1_min_ != nullptr)) {
    if (src0_min_ != nullptr && src1_max_ != nullptr) {
      int ic_dim = src1_min_->size() > 1 ? 2 : 0;
      attr_.set_output_scales(ic_dim, {DNNL_RUNTIME_F32_VAL});
      vector<int64_t> scale_shape;
      scale_shape.push_back(src1_min_->size());
      scale_f32_mem_ = memory({scale_shape, memory::data_type::f32, GetStrides(scale_shape)}, eng_, DNNL_MEMORY_NONE);
      // need zero point when src0 is u8
      if (src0_->dtype() == "u8") {
        vector<int64_t> zero_point_shape(src1_shape);
        zero_point_shape[src1_shape.size() - 1] = src1_shape[src1_shape.size() - 2];
        zero_point_shape[src1_shape.size() - 2] = 1;
        vector<int64_t> zero_point_stride = GetStrides(zero_point_shape, {});
        CalculateCompensation(src1_shape, src1_stride, zero_point_stride);
      }
    }
  }

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
  if (swish_) {
    auto op_scale = 1.0;
    auto op_alpha = 1.0;
    auto op_beta = 0.0;
    po.append_eltwise(op_scale, algorithm::eltwise_swish, op_alpha, op_beta);
  }
  // this is to sub zero point in fp32 to make the output u8
  if (!is_dynamic_ && append_eltwise_ && dst_->dtype() == "u8") {
    float zero_point = -1 * static_cast<const float*>(dst_min_->data())[0];
    po.append_eltwise(dst_scales_[0], algorithm::eltwise_linear, 1., zero_point);
  }

  // Part1: Derive operator's user proper shape and strides
  // 1.1 Transpose tensor shape and get it
  // for decoder-only transformers dnnl amx bf16 brgemm weight reorder process
#if __AMX_BF16__
  if (src1_->dtype() == "bf16" && model_ != nullptr && model_->input_shape().size() > 1) {
    if ((seq_len_ != 0 && seq_len_ > 128 && model_->input_shape()[1] == 1) ||
        (seq_len_ != 0 && seq_len_ == 1 && model_->input_shape()[1] > 128)) {
      weight_cached_ = false;
    }
    seq_len_ = model_->input_shape()[1];
  }
#endif
  vector<int64_t> src0_shape_origin = src0_->shape();
  vector<int64_t> src0_shape = GetShapes(src0_shape_origin, src0_perm_);
  vector<int64_t> src0_stride = GetStrides(src0_shape_origin, src0_perm_);
  src0_->set_shape(src0_shape);

  // 1.2 malloc tensor for output
  // src0_: M*K, src1_: K*N, DST: M*N
  vector<int64_t> dst_shape_origin = {src0_shape[0], src1_->shape()[0]};

  // Sub-step3: fused post transpose, notice it's different that
  // pre transpose will use the tranposed shape and stride, it's straight
  // forward post transpose will use origin shape and that means the dst buffer
  // in matmul is a buffer transposed back from dst_perm(understand tranpose to
  // and transpose back) pre_transpose: src0_buffer -> pre_transpose ->
  // target_buffer in matmul post_transpose: target_buffer in matmul<- post
  // transpose <-dst_buffer
  vector<int64_t> dst_shape = GetShapes(dst_shape_origin, dst_perm_);
  vector<int64_t> reverse_perm = ReversePerm(dst_perm_);
  vector<int64_t> dst_stride = GetStrides(dst_shape, reverse_perm);

  // 1.4 Prepare memory descriptors
  memory::desc any_src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], memory::format_tag::any);
  memory::desc src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], src0_stride);

  memory::desc any_dst_md, dst_md;
  if (is_dynamic_) {
    // inner_product output dtype in dynamic quantization should be fp32 and
    // then manually quantize to u8/s8.
    any_dst_md = memory::desc(dst_shape_origin, type2mem["fp32"], memory::format_tag::any);
    dst_md = memory::desc(dst_shape_origin, type2mem["fp32"], dst_stride);
  } else {
    any_dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], memory::format_tag::any);
    dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);
  }
  // 1.5 Set dst shape and strides
  dst_->set_shape(dst_shape);

  // 2.2 Prepare op descriptors
  dnnl::inner_product_forward::desc inner_product_d =
      has_bias_ || (is_dynamic_ && src0_->dtype() == "u8")
          ? dnnl::inner_product_forward::desc(prop_kind::forward_inference, src0_md, src1_md_, bias_md_, dst_md)
          : dnnl::inner_product_forward::desc(prop_kind::forward_inference, src0_md, src1_md_, dst_md);

  if (format_any_) {
    if (!weight_reorded_) {
      inner_product_d =
          has_bias_ || (is_dynamic_ && src0_->dtype() == "u8")
              ? dnnl::inner_product_forward::desc(prop_kind::forward_inference, any_src0_md, any_src1_md_, any_bias_md_,
                                                  any_dst_md)
              : dnnl::inner_product_forward::desc(prop_kind::forward_inference, any_src0_md, any_src1_md_, any_dst_md);
    } else {
      inner_product_d =
          has_bias_ || (is_dynamic_ && src0_->dtype() == "u8")
              ? dnnl::inner_product_forward::desc(prop_kind::forward_inference, any_src0_md,
                                                  any_src1_m_last_.get_desc(), any_bias_m_last_.get_desc(), any_dst_md)
              : dnnl::inner_product_forward::desc(prop_kind::forward_inference, any_src0_md,
                                                  any_src1_m_last_.get_desc(), any_dst_md);
    }
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
    vector<int64_t> post_shape = post_->shape();
    vector<int64_t> post_stride = GetStrides(post_shape);
    memory::desc binary_md = memory::desc(post_shape, type2mem[post_->dtype()], post_stride);
    po.append_binary(algorithm::binary_add, binary_md);
    binary_m_ = memory(binary_md, eng_, DNNL_MEMORY_NONE);
  }
  if (append_eltwise_ || append_sum_ || binary_add_ || is_dynamic_) attr_.set_post_ops(po);

  attr_.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  inner_product_pd_ = dnnl::inner_product_forward::primitive_desc(inner_product_d, attr_, eng_);

  memory::desc scratchpad_md = inner_product_pd_.scratchpad_desc();

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
  if (!weight_cached_) {
    memory any_src1_m = any_src1_m_last_;
    if (inner_product_pd_.weights_desc() != any_src1_m_last_.get_desc()) {
      void* cached_w_ptr;
      any_src1_m = memory(inner_product_pd_.weights_desc(), eng_, DNNL_MEMORY_NONE);
      if (src1_->is_shared()) {
        int64_t weight_size = any_src1_m.get_desc().get_size();
        void* weight_shm_ptr =
            MemoryAllocator::ManagedShm().find_or_construct<char>(src1_->name().c_str())[weight_size](0);
        any_src1_m.set_data_handle(weight_shm_ptr);
        cached_w_ptr = weight_shm_ptr;
      } else {
        cached_w_ptr = reinterpret_cast<void*>(
          aligned_alloc(ALIGNMENT, (any_src1_m.get_desc().get_size() / ALIGNMENT + 1) * ALIGNMENT));
        any_src1_m.set_data_handle(cached_w_ptr);
      }
      dnnl::reorder(any_src1_m_last_, any_src1_m).execute(eng_stream_, any_src1_m_last_, any_src1_m);
      if (src1_->is_shared() && this->get_execution_mode() == ExecutionMode::INFERENCE &&
          src1_->life() <= 1) {
        MemoryAllocator::ManagedShm().destroy_ptr(src1_->mutable_data());
        src1_->set_shm_handle(MemoryAllocator::ManagedShm().get_handle_from_address(cached_w_ptr));
      } else {
        if (this->get_execution_mode() == ExecutionMode::INFERENCE && src1_->life() <= 1) {
          aligned_free(src1_->mutable_data());
          src1_->set_data(cached_w_ptr);
          weight_reorded_ = true;
        }
        any_src1_m_last_ = memory(inner_product_pd_.weights_desc(), eng_, cached_w_ptr);
      }
    }
    memory_args_[DNNL_ARG_WEIGHTS] = any_src1_m;
    if (!is_dynamic_ && has_bias_) {
      memory any_bias_m = any_bias_m_last_;
      if (inner_product_pd_.bias_desc() != any_bias_m_last_.get_desc()) {
        void* cached_b_ptr;
        any_bias_m = memory(inner_product_pd_.bias_desc(), eng_, DNNL_MEMORY_NONE);
        if (bias_->is_shared()) {
          int64_t bias_size = bias_m_.get_desc().get_size();
          void* bias_shm_ptr =
              MemoryAllocator::ManagedShm().find_or_construct<char>(bias_->name().c_str())[bias_size](0);
          any_bias_m.set_data_handle(bias_shm_ptr);
          cached_b_ptr = bias_shm_ptr;
         } else {
          cached_b_ptr = reinterpret_cast<void*>(
            aligned_alloc(ALIGNMENT, (bias_m_.get_desc().get_size() / ALIGNMENT + 1) * ALIGNMENT));
          any_bias_m.set_data_handle(cached_b_ptr);
        }
        dnnl::reorder(any_bias_m_last_, any_bias_m).execute(eng_stream_, any_bias_m_last_, any_bias_m);
        if (bias_->is_shared() && this->get_execution_mode() == ExecutionMode::INFERENCE &&
            bias_->life() <= 1) {
          MemoryAllocator::ManagedShm().destroy_ptr(bias_->mutable_data());
          bias_->set_shm_handle(MemoryAllocator::ManagedShm().get_handle_from_address(cached_b_ptr));
        } else {
          if (this->get_execution_mode() == ExecutionMode::INFERENCE && bias_->life() <= 1) {
              aligned_free(bias_->mutable_data());
            bias_->set_data(cached_b_ptr);
          }
          any_bias_m_last_ = memory(inner_product_pd_.bias_desc(), eng_, cached_b_ptr);
        }
      }
      memory_args_[DNNL_ARG_BIAS] = any_bias_m;
    }
    weight_cached_ = true;
  }

  // If the inner product forward class in the cache pool, just get it from the
  // pool. Otherwise, do the reshape and send the related class into the cache
  // pool
  size_t key =
      InnerProductPrimitiveFwdFactory::Key(src0_->dtype(), src1_->dtype(), output_dtype_, src0_->shape(),
                                           src1_->shape(), dst_perm_, append_op_, post_->shape(), output_scale_, &eng_);
  if (InnerProductPrimitiveFwdFactory::IsInFactory(key) && !InnerProductPrimitiveFwdFactory::DoNotCache()) {
    inner_product_p_ = InnerProductPrimitiveFwdFactory::Get(key);
  } else {
    // 2.5 Prepare primitive objects (cached)
    inner_product_p_ = dnnl::inner_product_forward(inner_product_pd_);
    InnerProductPrimitiveFwdFactory::Set(key, inner_product_p_);
  }
  DstReshapeFusion(input, output);
}

vector<vector<string>> InnerProductOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
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

//  2. inference kernel(for int8 and f32)
void InnerProductOperator::ForwardDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  // create a dynamic quantization output with fp32.
  void* dst_data;
  if (is_dynamic_) {
    dst_data = inner_product_dynamic_res_->mutable_data();
  } else {
    dst_data = dst_->mutable_data();
  }
  // has post_op: append_sum
  if (post_ != nullptr && !binary_add_) {
    DLOG(INFO) << "inner product has post op " << post_->name();
    void* post_data_ptr = const_cast<void*>(post_->data());
    auto life_count = MemoryAllocator::get().CheckMemory(post_data_ptr);
    // MemoryAllocate::check_tensor_life
    if (life_count == 1 && this->get_execution_mode() != ExecutionMode::DEBUG) {
      post_->unref_data(true);
      if (is_dynamic_)
        inner_product_dynamic_res_->set_data(post_data_ptr);
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
  const auto& src0_data = src0_->data();

  auto& src0_shape_ = input[0]->shape();
  auto& dst_shape_ = output[0]->shape();
  if (weight_comp_.valid && src0_shape_[0] == weight_comp_.PreferedM) {
    auto& src1_shape = src1_->shape();
    std::vector<const void*> rt{src0_->data(), NULL, dst_->data(), has_bias_ ? bias_->data() : NULL,
                                weight_comp_.scales_.data()};
    if (append_sum_) {
      // inplace use the append sum as dst data
      rt = {src0_->data(), NULL,        dst_->data(), has_bias_ ? bias_->data() : NULL, weight_comp_.scales_.data(),
            NULL,          dst_->data()};
    }

    jd::exec_context_t context(nullptr);
    for (size_t index = 0; index < rt.size(); index++) {
      jd::memory_storage_t* mem;
      cpu_engine_->create_memory_storage(&mem);
      mem->set_handle(const_cast<void*>(rt[index]));
      if (index == jd::ssd::matmul_io::SRC0) {
        context.set_dynamic_shape({input[0]->shape()[0]});
      }
      if (index == jd::ssd::matmul_io::DST0) {
        context.add_output(mem);
      } else {
        context.add_input(mem);
      }
    }
    matmul_kernel_->execute(context);
    return;
  }

  // 1. Prepare memory objects with data_ptr
  src0_m_.set_data_handle(const_cast<void*>(src0_data), eng_stream_);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), eng_stream_);

  memory any_src0_m = src0_m_;
  memory any_dst_m = dst_m_;
  memory any_bias_m;
  if (is_dynamic_) {
    any_bias_m = bias_m_;
  }
  // 2. Reorder the data when the primitive memory and user memory are different
  if (inner_product_pd_.src_desc() != src0_m_.get_desc()) {
    any_src0_m = memory(inner_product_pd_.src_desc(), eng_);
    dnnl::reorder(src0_m_, any_src0_m).execute(eng_stream_, src0_m_, any_src0_m);
  }
  if (inner_product_pd_.dst_desc() != dst_m_.get_desc()) {
    any_dst_m = memory(inner_product_pd_.dst_desc(), eng_);
  }
  // the runtime calculation of dynamic quantization
  vector<float> dynamic_bias;
  if (is_dynamic_) RuntimeMemoryArgs(&dynamic_bias, &any_bias_m);
  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC_0] = any_src0_m;
  memory_args_[DNNL_ARG_DST] = any_dst_m;
  // has post_op: binary_add
  if (post_ != nullptr && binary_add_) {
    void* post_ptr = post_->mutable_data();
    binary_m_.set_data_handle(post_ptr, eng_stream_);
    memory_args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = binary_m_;
  }

  // 4. Execute the primitive
  inner_product_p_.execute(eng_stream_, memory_args_);
  // 5. Reorder the data of dst memory (When it is format_any)
  if (inner_product_pd_.dst_desc() != dst_m_.get_desc()) {
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
}

void InnerProductOperator::RuntimeMinmax() {
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

void InnerProductOperator::RuntimeMemoryArgs(vector<float>* dynamic_bias_ptr, memory* any_bias_m_ptr) {
  auto& dynamic_bias = *dynamic_bias_ptr;
  auto& any_bias_m = *any_bias_m_ptr;
  int channel_size = src1_min_->size();  // channel_size=1 represent per_tensor
  rescales_.resize(channel_size);
  const float* src0_scales = reinterpret_cast<const float*>(src0_max_->data());
  const float* src1_scales = reinterpret_cast<const float*>(src1_max_->data());
  if (channel_size == 1) {
    rescales_[0] = output_scale_ * src0_scales[0] * src1_scales[0];
  } else {
#pragma omp parallel for
    for (int i = 0; i < channel_size; i++) rescales_[i] = output_scale_ * src0_scales[0] * src1_scales[i];
  }
  scale_f32_mem_.set_data_handle(reinterpret_cast<void*>(rescales_.data()), eng_stream_);
  memory_args_[DNNL_ARG_ATTR_OUTPUT_SCALES] = scale_f32_mem_;
  // The bias loaded from file is not scaled. So need rescaled runtime.
  // the compensation is src0_scale*sr0_min*ones_like(src0)*src1. compensation
  // will be add as bias
  if (has_bias_ || src0_->dtype() == "u8") {
    dynamic_bias.resize(src1_->shape()[0]);
    float* bias_data = has_bias_ ? reinterpret_cast<float*>(bias_->mutable_data()) : nullptr;
    float com_scale = (*(reinterpret_cast<float*>(src0_min_->mutable_data()))) / output_scale_ / src0_scales[0];
#pragma omp parallel for
    for (int i = 0; i < src1_->shape()[0]; i++) {
      dynamic_bias[i] = 0;
      if (has_bias_) dynamic_bias[i] += bias_data[i] / rescales_[channel_size == 1 ? 0 : i];
      if (src0_->dtype() == "u8") dynamic_bias[i] += compensation_[i] * com_scale;
    }
    bias_m_.set_data_handle(reinterpret_cast<void*>(dynamic_bias.data()), eng_stream_);
    if (inner_product_pd_.bias_desc() != bias_m_.get_desc()) {
      any_bias_m = memory(inner_product_pd_.bias_desc(), eng_);
      dnnl::reorder(bias_m_, any_bias_m).execute(eng_stream_, bias_m_, any_bias_m);
    }
    memory_args_[DNNL_ARG_BIAS] = any_bias_m;
  }
}
REGISTER_OPERATOR_CLASS(InnerProduct);
}  // namespace executor
