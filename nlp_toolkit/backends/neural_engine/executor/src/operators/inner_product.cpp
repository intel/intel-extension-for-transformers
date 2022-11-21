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
    format_any_ = attrs_map["format_any"] == "true";
  }
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  iter = attrs_map.find("gelu_split");
  if (iter != attrs_map.end()) {
    gelu_split_ = attrs_map["gelu_split"] == "true";
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
  LOG(INFO) << "append_op: " << append_op_;
}

InnerProductOperator::~InnerProductOperator() {}

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
  LOG(INFO) << "inner product has bias add " << has_bias_;
  dst_->set_dtype(output_dtype_);
  if (src0_->dtype() == "fp32" && src1_->dtype() == "fp32") {
    kernel_type_ = Dense;
    weight_zero_ratio_ = GetSparseRatio<float>(static_cast<const float*>(src1_->data()), src1_->shape(), blocksize_);
    if (weight_zero_ratio_ >= sparse_threshold_) kernel_type_ = Dense;
    LOG(INFO) << "weight zero ratio: " << weight_zero_ratio_;
  } else if (src0_->dtype() == "u8" && src1_->dtype() == "s8") {
    kernel_type_ = Dense;
    blocksize_ = {4, 16};
    weight_zero_ratio_ = GetSparseRatio<int8_t>(static_cast<const int8_t*>(src1_->data()), src1_->shape(), blocksize_);
    if (weight_zero_ratio_ >= sparse_threshold_) kernel_type_ = Dense;
    LOG(INFO) << "weight zero ratio: " << weight_zero_ratio_;
  } else if (src0_->dtype() == "s8" && src1_->dtype() == "u8") {
    blocksize_ = {4, 1};
    weight_zero_ratio_ = GetSparseRatio<int8_t>(static_cast<const int8_t*>(src0_->data()), src0_->shape(), blocksize_);
    if (weight_zero_ratio_ >= sparse_threshold_) kernel_type_ = SparseLib;
    LOG(INFO) << "weight zero ratio: " << weight_zero_ratio_;
  } else if (src1_->dtype() == "bf16") {
    kernel_type_ = Dense;
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

  LOG(INFO) << "Innerproduct " << name_ << " execute kenel: " << kernel_type_;
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
      output.size() > 1 || (src0_min_ != nullptr && src0_min_->raw_data() == nullptr && !src0_min_->is_shared());
  if (is_dynamic_) LOG(INFO) << this->name() << " is DYNAMIC!!!";
  switch (kernel_type_) {
    case Dense:
      PrepareDense(input, output);
      break;
    case Sparse:
      if (is_dynamic_) LOG(ERROR) << "Sparse kernel not support dynamic quantization yet!\n";
      PrepareSparse(input, output);
      break;
    case SparseLib:
      if (is_dynamic_) LOG(ERROR) << "Sparse kernel not support dynamic quantization yet!\n";
#ifdef WITH_SPARSELIB
      PrepareSparseLib(input, output);
#else
      LOG(ERROR) << "Sparse lib is not loaded!\n";
#endif
      break;
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
        LOG(INFO) << "inner product has no such sparse kernel, output tensor is" << output[0]->name();
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
      if (dst_min_ != nullptr)
        dst_scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
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
}

void InnerProductOperator::ShapeInferSparseLib(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // shape infer for dispatcher
  // save the SparseLib primitive class creation time
  vector<int64_t> src1_shape_origin = src1_->shape();
  vector<int64_t> src1_shape = src1_shape_origin;
  if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
    src1_shape = {src1_shape[1], src1_shape[0]};
  }
  vector<int64_t> dst_shape = {src0_->shape()[0], src1_shape[1]};
  dst_->set_shape(dst_shape);
  if (!reshape_.empty()) {
    vector<int64_t> ref_shape;
    if (!reshape_dims_.empty()) {
      ref_shape = input.back()->shape();
    }
    vector<int64_t> dst_shape = GetDstShape(reshape_, output[0]->size(), ref_shape, reshape_dims_);
    output[0]->set_shape(dst_shape);
  }
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

    src1_->set_shape(src1_shape);
    src1_desc_ = {src1_3d_shape, jd::data_type::u8, jd::format_type::ab};

    vector<int64_t> dst_3d_shape = {src1_3d_shape[0], src0_->shape()[0], src1_3d_shape[2]};
    vector<int64_t> dst_2d_shape = {src0_->shape()[0], src1_3d_shape[0] * src1_3d_shape[2]};
    dst_->set_shape(dst_2d_shape);
    dst_desc_ = {dst_3d_shape, type2sparsemem_[dst_->dtype()], jd::format_type::ab};
  } else {
    // 2D
    if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{0, 1}) {
      src1_shape = {src1_shape[1], src1_shape[0]};
    }
    src1_->set_shape(src1_shape);
    src1_desc_ = {src1_shape, jd::data_type::u8, jd::format_type::ab};

    vector<int64_t> dst_shape = {src0_->shape()[0], src1_shape[1]};
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

  if (!reshape_.empty()) {
    vector<int64_t> ref_shape;
    if (!reshape_dims_.empty()) {
      ref_shape = input.back()->shape();
    }
    vector<int64_t> dst_shape = GetDstShape(reshape_, output[0]->size(), ref_shape, reshape_dims_);
    output[0]->set_shape(dst_shape);
  }
}

#if __AVX512F__
void InnerProductOperator::ForwardSparseLib(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  void* dst_data = dst_->mutable_data();
  // has op: append_sum
  if (post_ != nullptr && !binary_add_) {
    LOG(INFO) << "inner product has post op " << post_->name();
    // The sum primitive requires all source and destination tensors to have the same shape.
    // Implicit broadcasting is not supported.
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
  // no use of sparselib
  if (dispatch_config_.empty()) return;
  if (stage == "in") {
    // reorder 2d to 3d in SparseLib
    // 2D dense: [256, 768] x [768 328] -> [256, 328]
    // 2D sparselib: [328, 768] x [768, 256] -> [328, 256]
    // dispatch 3D sparselib: [328, 768] x [2, 768, 128] -> [2, 328, 128]
    // reorder src and post activation
    if (dispatch_config_[0] == "SparseLib") {
      // reshape to 3d then reorder
      vector<int64_t> src1_3d_shape;
      StringSplit<int64_t>(&src1_3d_shape, dispatch_config_[1], ",");
      vector<int64_t> src1_3d_shape_origin = {src1_3d_shape[1], src1_3d_shape[0], src1_3d_shape[2]};
      input[1]->reorder(src1_3d_shape_origin);
      if (post_ != nullptr && !binary_add_) {
        vector<int64_t> post_3d_shape_origin = {src0_->shape()[0], src1_3d_shape[0], src1_3d_shape[2]};
        post_->reorder(post_3d_shape_origin);
      }
    }
  } else if (stage == "out") {
    if (dispatch_config_[0] == "SparseLib") {
      // reorder dst activation (optional)
      vector<int64_t> src1_3d_shape;
      StringSplit<int64_t>(&src1_3d_shape, dispatch_config_[1], ",");
      vector<int64_t> dst_3d_shape_origin = {src1_3d_shape[0], src0_->shape()[0], src1_3d_shape[2]};
      output[0]->reorder(dst_3d_shape_origin);
      // reorder src and post back
      if (input[1]->left_life() > 0) input[1]->reorder(src1_3d_shape);
      if (post_ != nullptr && !binary_add_ && post_->left_life() > 0) post_->reorder(dst_3d_shape_origin);
    }
  } else {
    LOG(WARNING) << "Wrong stage parameter, should be in or out...";
  }
}

void InnerProductOperator::PrepareDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (!is_dynamic_ && (output_scale_ != 1.f || src0_min_ != nullptr || src1_min_ != nullptr)) {
    int ic_dim = 0;
    if (src0_min_ != nullptr && src1_max_ != nullptr) {
      ic_dim = src1_min_->size() > 1 ? 0 | (1 << 1) : 0;
      vector<float> src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
      vector<float> src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
      if (dst_min_ != nullptr)
        dst_scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
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
  if (!src1_perm_.empty() && src1_perm_ == vector<int64_t>{1, 0}) src1_->set_transpose();
  if (!src0_perm_.empty() && src0_perm_ == vector<int64_t>{1, 0}) src0_->set_transpose();

  if (has_bias_) {
    vector<int64_t> bias_shape = {src1_shape[0]};
    vector<int64_t> bias_stride = GetStrides(bias_shape);
    bias_md_ = memory::desc(bias_shape, type2mem[bias_->dtype()], bias_stride);
    any_bias_md_ = memory::desc(bias_shape, type2mem[bias_->dtype()], memory::format_tag::any);
    bias_m_ = memory(bias_md_, eng_, bias_->mutable_data());
  }
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
  if (!reshape_.empty()) {
    vector<int64_t> ref_shape;
    if (!reshape_dims_.empty()) {
      ref_shape = input.back()->shape();
    }
    vector<int64_t> dst_shape = GetDstShape(reshape_, output[0]->size(), ref_shape, reshape_dims_);
    output[0]->set_shape(dst_shape);
  }
}

// 1. Create primitive
void InnerProductOperator::ReshapeDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  dnnl::post_ops po;
  vector<int64_t> src1_shape = src1_->shape();
  vector<int64_t> src1_stride = GetStrides(src1_shape_origin_, src1_perm_);

  if (is_dynamic_ && (output_scale_ != 1.f || src0_min_ != nullptr || src1_min_ != nullptr)) {
    if (src0_min_ != nullptr && src1_max_ != nullptr) {
      int ic_dim = src1_min_->size() > 1 ? 2 : 0;
      // attr_.set_output_scales(ic_dim, {DNNL_RUNTIME_F32_VAL});
      vector<int64_t> scale_shape(src1_shape.size(), 1);
      scale_shape[src1_shape.size() - 1] = src1_min_->size();
      scale_md_ = memory::desc(scale_shape, memory::data_type::f32, GetStrides(scale_shape));
      po.append_binary(algorithm::binary_mul, scale_md_);
      // need zero point when src0 is u8
      if (src0_->dtype() == "u8") {
        vector<int64_t> zero_point_shape(src1_shape);
        zero_point_shape[src1_shape.size() - 1] = src1_shape[src1_shape.size() - 2];
        zero_point_shape[src1_shape.size() - 2] = 1;
        vector<int64_t> zero_point_stride = GetStrides(zero_point_shape, {});
        CalculateCompensation(src1_shape, src1_stride, zero_point_stride);
        // use binary_add to calculate zero_point
        compensation_md_ = memory::desc(zero_point_shape, memory::data_type::f32, zero_point_stride);
        po.append_binary(algorithm::binary_add, compensation_md_);
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
  // this is to sub zero point in fp32 to make the output u8/s8
  if (!is_dynamic_ && append_eltwise_ && (dst_->dtype() == "u8" || dst_->dtype() == "s8")) {
    float zero_point = -1 * static_cast<const float*>(dst_min_->data())[0];
    if (dst_->dtype() == "s8") {
      zero_point = 0;
    }
    po.append_eltwise(dst_scales_[0], algorithm::eltwise_linear, 1., zero_point);
  }

  // Part1: Derive operator's user proper shape and strides
  // 1.1 Transpose tensor shape and get it
  vector<int64_t> src0_shape_origin = src0_->shape();
  vector<int64_t> src0_shape = GetShapes(src0_shape_origin, src0_perm_);
  vector<int64_t> src0_stride = GetStrides(src0_shape_origin, src0_perm_);
  src0_->set_shape(src0_shape);

  // 1.2 malloc tensor for output
  // src0_: M*K, src1_: K*N, DST: M*N
  vector<int64_t> dst_shape_origin = {src0_shape[0], src1_->shape()[0]};

  // Sub-step3: fused post transpose, notice it's different that
  // pre transpose will use the tranposed shape and stride, it's straight forward
  // post transpose will use origin shape and that means the dst buffer in matmul
  // is a buffer transposed back from dst_perm(understand tranpose to and transpose back)
  // pre_transpose: src0_buffer -> pre_transpose -> target_buffer in matmul
  // post_transpose: target_buffer in matmul<- post transpose <-dst_buffer
  vector<int64_t> dst_shape = GetShapes(dst_shape_origin, dst_perm_);
  vector<int64_t> reverse_perm = ReversePerm(dst_perm_);
  vector<int64_t> dst_stride = GetStrides(dst_shape, reverse_perm);

  // 1.4 Prepare memory descriptors
  memory::desc any_src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], memory::format_tag::any);
  memory::desc src0_md = memory::desc(src0_shape, type2mem[src0_->dtype()], src0_stride);

  memory::desc any_dst_md, dst_md;
  if (is_dynamic_) {
    // inner_product output dtype in dynamic quantization should be fp32 and then manually quantize to u8/s8.
    any_dst_md = memory::desc(dst_shape_origin, type2mem["fp32"], memory::format_tag::any);
    dst_md = memory::desc(dst_shape_origin, type2mem["fp32"], dst_stride);
  } else {
    any_dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], memory::format_tag::any);
    dst_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);
  }
  // 1.5 Set dst shape and strides
  dst_->set_shape(dst_shape);
  if (output.size() > 1) {
    dst_min_->set_shape({1});
    dst_max_->set_shape({1});
  }

  // 2.2 Prepare op descriptors
  dnnl::inner_product_forward::desc inner_product_d =
      has_bias_ ? dnnl::inner_product_forward::desc(prop_kind::forward_inference, src0_md, src1_md_, bias_md_, dst_md)
                : dnnl::inner_product_forward::desc(prop_kind::forward_inference, src0_md, src1_md_, dst_md);

  if (format_any_) {
    inner_product_d = has_bias_ ? dnnl::inner_product_forward::desc(prop_kind::forward_inference, any_src0_md,
                                                                    any_src1_md_, any_bias_md_, any_dst_md)
                                : dnnl::inner_product_forward::desc(prop_kind::forward_inference, any_src0_md,
                                                                    any_src1_md_, any_dst_md);
  }

  if (gelu_erf_ && gelu_split_) {
    memory::desc gelu_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);
    auto gelu_d =
        dnnl::eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_gelu_erf, gelu_md, 0.f, 0.f);
    gelu_pd_ = dnnl::eltwise_forward::primitive_desc(gelu_d, gelu_eng_);
    gelu_p_ = dnnl::eltwise_forward(gelu_pd_);
    gelu_m_ = memory(gelu_md, gelu_eng_);
  }
  if (gelu_tanh_ && gelu_split_) {
    memory::desc gelu_md = memory::desc(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);
    auto gelu_d =
        dnnl::eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_gelu_tanh, gelu_md, 0.f, 0.f);
    gelu_pd_ = dnnl::eltwise_forward::primitive_desc(gelu_d, gelu_eng_);
    gelu_p_ = dnnl::eltwise_forward(gelu_pd_);
    gelu_m_ = memory(gelu_md, gelu_eng_);
  }
  if (binary_add_) {
    vector<int64_t> post_shape = post_->shape();
    vector<int64_t> post_stride = GetStrides(post_shape);
    memory::desc binary_md = memory::desc(post_shape, type2mem[post_->dtype()], post_stride);
    po.append_binary(algorithm::binary_add, binary_md);
    binary_m_ = memory(binary_md, eng_);
  }
  if (append_eltwise_ || append_sum_ || binary_add_ || is_dynamic_) attr_.set_post_ops(po);
  inner_product_pd_ = dnnl::inner_product_forward::primitive_desc(inner_product_d, attr_, eng_);

  // 2.4 Prepare memory objects (cached)
  src0_m_ = memory(src0_md, eng_);
  dst_m_ = memory(dst_md, eng_);
  if (!weight_cached_) {
    memory any_src1_m = src1_m_;
    if (inner_product_pd_.weights_desc() != src1_m_.get_desc()) {
      any_src1_m = memory(inner_product_pd_.weights_desc(), eng_);
      if (src1_->is_shared()) {
        int64_t weight_size = any_src1_m.get_desc().get_size();
        void* weight_shm_ptr =
            MemoryAllocator::ManagedShm().find_or_construct<char>(src1_->name().c_str())[weight_size](0);
        any_src1_m.set_data_handle(weight_shm_ptr);
      }
      dnnl::reorder(src1_m_, any_src1_m).execute(eng_stream_, src1_m_, any_src1_m);
    }
    memory_args_[DNNL_ARG_WEIGHTS] = any_src1_m;
    if (!is_dynamic_ && has_bias_) {
      memory any_bias_m = bias_m_;
      if (inner_product_pd_.bias_desc() != bias_m_.get_desc()) {
        any_bias_m = memory(inner_product_pd_.bias_desc(), eng_);
        if (bias_->is_shared()) {
          int64_t bias_size = bias_m_.get_desc().get_size();
          void* bias_shm_ptr =
              MemoryAllocator::ManagedShm().find_or_construct<char>(bias_->name().c_str())[bias_size](0);
          any_bias_m.set_data_handle(bias_shm_ptr);
        }
        dnnl::reorder(bias_m_, any_bias_m).execute(eng_stream_, bias_m_, any_bias_m);
      }
      memory_args_[DNNL_ARG_BIAS] = any_bias_m;
    }
    weight_cached_ = true;
  }

  // If the inner product forward class in the cache pool, just get it from the pool.
  // Otherwise, do the reshape and send the related class into the cache pool
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

  if (!reshape_.empty()) {
    vector<int64_t> ref_shape;
    if (!reshape_dims_.empty()) {
      ref_shape = input.back()->shape();
    }
    vector<int64_t> dst_shape = GetDstShape(reshape_, output[0]->size(), ref_shape, reshape_dims_);
    output[0]->set_shape(dst_shape);
  }
}

// 2. inference kernel(for int8 and f32)
void InnerProductOperator::ForwardDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // create a dynamic quantization output with fp32.
  Tensor inner_product_fp32_res;
  void* dst_data;
  if (is_dynamic_) {
    inner_product_fp32_res = *dst_;
    inner_product_fp32_res.set_dtype("fp32");
    dst_data = inner_product_fp32_res.mutable_data();
  } else {
    dst_data = dst_->mutable_data();
  }
  // has post_op: append_sum
  if (post_ != nullptr && !binary_add_) {
    LOG(INFO) << "inner product has post op " << post_->name();
    void* post_data_ptr = const_cast<void*>(post_->data());
    auto life_count = MemoryAllocator::get().CheckMemory(post_data_ptr);
    // MemoryAllocate::check_tensor_life
    if (life_count == 1) {
      post_->unref_data(true);
      if (is_dynamic_)
        inner_product_fp32_res.set_data(post_data_ptr);
      else
        dst_->set_data(post_data_ptr);
      dst_data = post_data_ptr;
    } else {
      int data_size = post_->size();
      string data_type = post_->dtype();
      memcpy(dst_data, post_data_ptr, data_size * type2bytes[data_type]);
      LOG(WARNING) << "post tensor will be used by multi node...";
    }
  }
  // 0. Alias variables part
  const auto& src0_data = src0_->data();
  // when change data value please use mutable_data

  // 1. Prepare memory objects with data_ptr
  src0_m_.set_data_handle(const_cast<void*>(src0_data), eng_stream_);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), eng_stream_);

  memory any_src0_m = src0_m_;
  memory any_dst_m = dst_m_;
  memory any_bias_m = bias_m_;

  // 2. Reorder the data when the primitive memory and user memory are different
  if (inner_product_pd_.src_desc() != src0_m_.get_desc()) {
    any_src0_m = memory(inner_product_pd_.src_desc(), eng_);
    dnnl::reorder(src0_m_, any_src0_m).execute(eng_stream_, src0_m_, any_src0_m);
  }

  if (inner_product_pd_.dst_desc() != dst_m_.get_desc()) {
    any_dst_m = memory(inner_product_pd_.dst_desc(), eng_);
  }

  // the runtime calculation of dynamic quantization
  vector<float> src0_compensation;
  vector<float> dynamic_bias;
  if (is_dynamic_) DynamicForward(&src0_compensation, &dynamic_bias, &any_bias_m);

  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC_0] = any_src0_m;
  memory_args_[DNNL_ARG_DST] = any_dst_m;
  // has post_op: binary_add
  if (post_ != nullptr && binary_add_) {
    void* post_ptr = post_->mutable_data();
    binary_m_.set_data_handle(post_ptr, eng_stream_);
    // dynamic quantization inserts additional post_ops
    int op_idx = 0;
    if (is_dynamic_) {
      op_idx++;
      if (src0_->dtype() == "u8") op_idx++;
    }
    memory_args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(op_idx) | DNNL_ARG_SRC_1] = binary_m_;
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
  if (is_dynamic_) {
    // quantize the fp32 result of inner_porduct
    if (output.size() > 1) {
      runtime_minmax(reinterpret_cast<float*>(inner_product_fp32_res.mutable_data()), inner_product_fp32_res.size(),
                     reinterpret_cast<float*>(dst_min_->mutable_data()),
                     reinterpret_cast<float*>(dst_max_->mutable_data()));
      // quantize
      if (output_dtype_ == "u8" || output_dtype_ == "s8") {
        auto scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
#if __AVX512F__
        Quantize_avx512(inner_product_fp32_res.size(), dst_->dtype(), inner_product_fp32_res.data(),
                        static_cast<const float*>(dst_min_->data()), scales_, dst_->mutable_data());
#else
        Quantize(inner_product_fp32_res.size(), dst_->dtype(), inner_product_fp32_res.data(),
                 static_cast<const float*>(dst_min_->data()), scales_, dst_->mutable_data());
#endif
        inner_product_fp32_res.unref_data();
      } else {
        // copy fp32_res to dst if not quantize
        void* res_ptr = const_cast<void*>(inner_product_fp32_res.data());
        inner_product_fp32_res.unref_data(true);
        dst_->set_data(res_ptr);
      }
    } else {
      void* res_ptr = const_cast<void*>(inner_product_fp32_res.data());
      inner_product_fp32_res.unref_data(true);
      dst_->set_data(res_ptr);
    }
  }
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

void InnerProductOperator::DynamicForward(vector<float>* src0_compensation_ptr, vector<float>* dynamic_bias_ptr,
                                          memory* any_bias_m_ptr) {
  auto& src0_compensation = *src0_compensation_ptr;
  auto& dynamic_bias = *dynamic_bias_ptr;
  auto& any_bias_m = *any_bias_m_ptr;
  int channel_size = src1_min_->size();  // channel_size=1 represent per_tensor
  memory scale_f32_mem(scale_md_, eng_);
  memory compensation_mem(compensation_md_, eng_);
  src0_compensation.resize(compensation_.size());
  rescales_.resize(channel_size);
  vector<float> src0_scales;
  vector<float> src1_scales;
  src0_scales = GetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), src0_->dtype());
  src1_scales = GetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), src1_->dtype());
  if (channel_size == 1) {
    rescales_[0] = output_scale_ / src0_scales[0] / src1_scales[0];
  } else {
#pragma omp parallel for
    for (int i = 0; i < channel_size; i++) rescales_[i] = output_scale_ / src0_scales[0] / src1_scales[i];
  }
  scale_f32_mem.set_data_handle(reinterpret_cast<void*>(rescales_.data()), eng_stream_);
  // memory_args_[DNNL_ARG_ATTR_OUTPUT_SCALES] = scale_f32_mem;
  memory_args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = scale_f32_mem;
  // The bias loaded from file is not scaled. So need rescaled runtime.
  if (has_bias_) {
    dynamic_bias.resize(bias_->size());
    float* bias_data = reinterpret_cast<float*>(bias_->mutable_data());
    if (channel_size == 1) {
#pragma omp parallel for
      for (int i = 0; i < bias_->size(); i++) dynamic_bias[i] = bias_data[i] / rescales_[0];
    } else {
#pragma omp parallel for
      for (int i = 0; i < bias_->size(); i++) dynamic_bias[i] = bias_data[i] / rescales_[i];
    }
    bias_m_.set_data_handle(reinterpret_cast<void*>(dynamic_bias.data()), eng_stream_);
    if (inner_product_pd_.bias_desc() != bias_m_.get_desc()) {
      any_bias_m = memory(inner_product_pd_.bias_desc(), eng_);
      dnnl::reorder(bias_m_, any_bias_m).execute(eng_stream_, bias_m_, any_bias_m);
    }
    memory_args_[DNNL_ARG_BIAS] = any_bias_m;
  }
  if (src0_->dtype() == "u8") {
    // the compensation is src0_scale*sr0_min*ones_like(src0)*src1
    float src0_min = *(reinterpret_cast<float*>(src0_min_->mutable_data()));
    if (channel_size == 1) {
#pragma omp parallel for
      for (int i = 0; i < compensation_.size(); i++)
        src0_compensation[i] = compensation_[i] * src0_min / src1_scales[0];
    } else {
#pragma omp parallel for
      for (int i = 0; i < compensation_.size(); i++)
        src0_compensation[i] = compensation_[i] * src0_min / src1_scales[i];
    }
    compensation_mem.set_data_handle(reinterpret_cast<void*>(src0_compensation.data()), eng_stream_);
    memory_args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1] = compensation_mem;
  }
}
REGISTER_OPERATOR_CLASS(InnerProduct);
}  // namespace executor
