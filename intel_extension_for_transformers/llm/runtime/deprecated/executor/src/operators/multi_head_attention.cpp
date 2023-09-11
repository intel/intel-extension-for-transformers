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

#include "multi_head_attention.hpp"
#include "kernels/exposed_enum.hpp"

#include "operator_registry.hpp"
using dt = jd::data_type;
using ft = jd::format_type;
using io = jd::exposed_enum::mha_dense::io;

namespace executor {
static unordered_map<string, dt> type2sparsemem{{"fp32", dt::fp32}, {"s32", dt::s32}, {"fp16", dt::fp16},
                                                {"u8", dt::u8},     {"s8", dt::s8},   {"bf16", dt::bf16}};

MultiHeadAttentionOperator::MultiHeadAttentionOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf),
      Q_perm_({}),
      K_perm_({}),
      V_perm_({}),
      dst_perm_({}),
      output_scale_(1.),
      rt_data_(io::SIZE, nullptr) {
  auto attrs_map = operator_conf_->attributes();

  auto iter = attrs_map.find("Q_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&Q_perm_, attrs_map["Q_perm"], ",");
  }
  iter = attrs_map.find("K_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&K_perm_, attrs_map["K_perm"], ",");
  }
  iter = attrs_map.find("V_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&V_perm_, attrs_map["V_perm"], ",");
  }
  iter = attrs_map.find("dst_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&dst_perm_, attrs_map["dst_perm"], ",");
  }
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) DLOG(WARNING) << "MHA not support output_dtype attr, its output_type is fixed by input";
  iter = attrs_map.find("output_scale");
  if (iter != attrs_map.end()) {
    output_scale_ = StringToNum<float>(attrs_map["output_scale"]);
  }
  iter = attrs_map.find("reshape");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&dst_reshape_, attrs_map["reshape"], ",");
  }

  iter = attrs_map.find("stable_softmax");
  if (iter != attrs_map.end()) {
    stable_softmax_ = true;
  }

  if (dst_reshape_.size() > 0 && dst_reshape_[0] != -1) {
    is_sparse_ = true;
    int max_threads = std::min(32, omp_get_max_threads());
    trans_mha_tmpbuf = reinterpret_cast<uint8_t*>(aligned_alloc(64, max_threads * Size2M));
  }
}

MultiHeadAttentionOperator::~MultiHeadAttentionOperator() {
  if (is_sparse_) aligned_free(trans_mha_tmpbuf);
  if (workspace_) aligned_free(workspace_);
}

void MultiHeadAttentionOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int input_size = input.size();
  dst_ = output[0];
  if (output.size() > 1) {
    dst_min_ = output[1];
    dst_max_ = output[2];
  }
  switch (input_size) {
    case 3: {
      Q_ = input[0];
      K_ = input[1];
      V_ = input[2];
      break;
    }
    case 4: {
      Q_ = input[0];
      K_ = input[1];
      V_ = input[2];
      att_mask_ = input[3];
      break;
    }
    case 5: {
      Q_ = input[0];
      K_ = input[1];
      V_ = input[2];
      att_mask_ = input[3];
      binary_add_mask_ = input[4];
      break;
    }
    case 9: {
      Q_ = input[0];
      K_ = input[1];
      V_ = input[2];
      Q_min_ = input[3];
      Q_max_ = input[4];
      K_min_ = input[5];
      K_max_ = input[6];
      V_min_ = input[7];
      V_max_ = input[8];
      break;
    }
    case 10: {
      Q_ = input[0];
      K_ = input[1];
      V_ = input[2];
      binary_add_mask_ = input[3];
      Q_min_ = input[4];
      Q_max_ = input[5];
      K_min_ = input[6];
      K_max_ = input[7];
      V_min_ = input[8];
      V_max_ = input[9];
      break;
    }
    case 12: {
      QKV_ = input[0];
      att_mask_ = input[1];
      Q_min_ = input[2];
      Q_max_ = input[3];
      K_min_ = input[4];
      K_max_ = input[5];
      V_min_ = input[6];
      V_max_ = input[7];
      QK_min_ = input[8];
      QK_max_ = input[9];
      dst_min_ = input[10];
      dst_max_ = input[11];
      break;
    }
    case 13: {
      QKV_ = input[0];
      att_mask_ = input[1];
      binary_add_mask_ = input[2];
      Q_min_ = input[3];
      Q_max_ = input[4];
      K_min_ = input[5];
      K_max_ = input[6];
      V_min_ = input[7];
      V_max_ = input[8];
      QK_min_ = input[9];
      QK_max_ = input[10];
      dst_min_ = input[11];
      dst_max_ = input[12];
      break;
    }
    case 14: {
      Q_ = input[0];
      K_ = input[1];
      V_ = input[2];
      att_mask_ = input[3];
      Q_min_ = input[4];
      Q_max_ = input[5];
      K_min_ = input[6];
      K_max_ = input[7];
      V_min_ = input[8];
      V_max_ = input[9];
      QK_min_ = input[10];
      QK_max_ = input[11];
      dst_min_ = input[12];
      dst_max_ = input[13];
      break;
    }
    case 15: {
      Q_ = input[0];
      K_ = input[1];
      V_ = input[2];
      att_mask_ = input[3];
      binary_add_mask_ = input[4];
      Q_min_ = input[5];
      Q_max_ = input[6];
      K_min_ = input[7];
      K_max_ = input[8];
      V_min_ = input[9];
      V_max_ = input[10];
      QK_min_ = input[11];
      QK_max_ = input[12];
      dst_min_ = input[13];
      dst_max_ = input[14];
      break;
    }
  }
}

void MultiHeadAttentionOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  LOG_IF(FATAL, binary_add_mask_ != nullptr && is_sparse_)
      << "one more mask (binary_add_mask) is not supported for sparse MHA kernel!";
  string dtype;
  if (Q_ != nullptr)
    dtype = Q_->dtype();
  else
    dtype = QKV_->dtype();
  LOG_IF(FATAL, dtype != "s8" && dtype != "bf16") << "only support int8/bf16, but get " << dtype;
  is_dynamic_ = (Q_max_ && Q_max_->raw_data() == nullptr) || (K_max_ && K_max_->raw_data() == nullptr) ||
                (V_max_ && V_max_->raw_data() == nullptr);
  if (dtype == "bf16") {
    dst_->set_dtype("bf16");
    if (is_dynamic_) LOG(ERROR) << "bf16 not support dynamic";
  } else if (dtype == "s8") {
    if (is_dynamic_) {
      dst_min_->set_dtype("fp32");
      dst_max_->set_dtype("fp32");
      dst_->set_dtype("s8");
      DLOG(INFO) << name_ << "is dynamic!!!";
    } else {
      dst_->set_dtype("u8");
      Q_scales_ = GetScales(Q_min_->data(), Q_max_->data(), Q_min_->size(), dtype);
      K_scales_ = GetScales(K_min_->data(), K_max_->data(), K_min_->size(), dtype);
      V_scales_ = GetScales(V_min_->data(), V_max_->data(), V_min_->size(), dtype);
      softmax_scales_ = GetScales(QK_min_->data(), QK_max_->data(), QK_min_->size(), "u8");  // after_softmax
      dst_scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
      if (is_sparse_)
        QKV_zeropoint_ = GetZeroPoints(dst_min_->data(), dst_scales_, dst_->dtype())[0];
      else
        QKV_zeropoint_ = (dst_->dtype() == "fp32") ? 0 : GetZeroPoints(dst_min_->data(), dst_scales_, dst_->dtype())[0];

      for (int i = 0; i < Q_scales_.size(); i++) Q_scales_[i] = 1 / Q_scales_[i];
      for (int i = 0; i < K_scales_.size(); i++) K_scales_[i] = 1 / K_scales_[i];
      for (int i = 0; i < V_scales_.size(); i++) V_scales_[i] = 1 / V_scales_[i];
      for (int i = 0; i < dst_scales_.size(); i++) dst_scales_[i] = 1 / dst_scales_[i];
    }
  }
}

void MultiHeadAttentionOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (is_sparse_)
    ReshapeSparse(input, output);
  else
    ReshapeDense(input, output);
}

void MultiHeadAttentionOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (is_sparse_)
    ForwardSparse(input, output);
  else
    ForwardDense(input, output);
}

void MultiHeadAttentionOperator::ReshapeSparse(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  std::unordered_map<std::string, std::string> op_attrs;
  src_shape_ = Q_->shape();
  if (Q_->tensor_format() != TensorFormat::MmKMb) {
    auto& Q_shape = Q_->shape();
    bs_ = Q_shape[2];
    seq_len_q_ = Q_shape[3];
    head_num_ = Q_shape[0];
    head_size_qk_ = Q_shape[1];
  } else {
    auto& Q_shape = Q_->shape();
    bs_ = Q_shape[0];
    seq_len_q_ = Q_shape[4];
    head_num_ = Q_shape[1];
    head_size_qk_ = Q_shape[2];
  }

  hidden_size_ = head_num_ * head_size_qk_;
  op_attrs["seq_pad"] = std::to_string(seq_len_q_);
  op_attrs["batch"] = std::to_string(bs_);
  op_attrs["head_num"] = std::to_string(head_num_);
  op_attrs["k"] = std::to_string(head_size_qk_);
  op_attrs["seq_len"] = std::to_string(seq_len_q_);
  op_attrs["scaleQ"] = std::to_string(Q_scales_[0]);
  op_attrs["scaleK"] = std::to_string(K_scales_[0]);
  op_attrs["scaleV"] = std::to_string(V_scales_[0]);
  op_attrs["scaleRet"] = std::to_string(dst_scales_[0]);
  op_attrs["zeropointRet"] = std::to_string(QKV_zeropoint_);

  jd::tensor_desc K_desc = {{bs_, head_num_, head_size_qk_, seq_len_q_}, dt::s8, ft::undef};
  jd::tensor_desc Q_desc = {{bs_, head_num_, head_size_qk_, seq_len_q_}, dt::s8, ft::undef};
  jd::tensor_desc mask_desc = {{bs_, seq_len_q_}, dt::fp32, ft::undef};
  jd::tensor_desc V_desc = {{bs_, head_num_, head_size_qk_, seq_len_q_}, dt::s8, ft::undef};
  jd::tensor_desc ret_desc = {{bs_, head_num_, head_size_qk_, seq_len_q_}, dt::u8, ft::undef};

  std::vector<jd::tensor_desc> ts_descs = {K_desc, Q_desc, mask_desc, V_desc, ret_desc};

  jd::operator_desc trans_attention_desc(jd::kernel_kind::transpose_mha, jd::kernel_prop::forward_inference,
                                         jd::engine_kind::cpu, ts_descs, op_attrs);
  jd::transpose_mha_desc transpose_mha_desc(trans_attention_desc);
  mha_transpose_ = jd::transpose_mha(transpose_mha_desc);
  dst_->set_shape({bs_, seq_len_q_, head_num_, head_size_qk_});
  if (!dst_reshape_.empty()) {
    vector<int64_t> dst_shape = GetDstShape(dst_reshape_, dst_->size(), {}, {});

    if (Q_->tensor_format() == TensorFormat::MmKMb) {
      dst_->set_shape({bs_, hidden_size_, seq_len_q_});
    } else {
      dst_->set_shape(dst_shape);
    }
  }
}

void MultiHeadAttentionOperator::ReshapeDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  std::unordered_map<std::string, std::string> attr_map;
  std::vector<int64_t> attn_shape;
  // Get shape info
  if (Q_ != nullptr) {
    vector<int64_t> Q_shape = GetShapes(Q_->shape(), Q_perm_);
    vector<int64_t> K_shape = GetShapes(K_->shape(), K_perm_);
    vector<int64_t> V_shape = GetShapes(V_->shape(), V_perm_);
    bs_ = Q_shape[0];
    head_num_ = Q_shape[1];
    seq_len_q_ = Q_shape[2];
    LOG_IF(ERROR, K_shape[3] != V_shape[2])
        << "seq_len of K should be equal with V, but get" << K_shape[3] << "VS" << V_shape[2];
    seq_len_kv_ = K_shape[3];
    LOG_IF(ERROR, Q_shape[3] != K_shape[2])
        << "head_size of Q should be equal with K, but get" << Q_shape[3] << "VS" << K_shape[2];
    head_size_qk_ = Q_shape[3];
    head_size_v_ = V_shape[3];
    attr_map["merged_QKV"] = "False";
    attn_shape = {bs_, seq_len_q_, head_num_, head_size_v_};
  } else {
    auto& QKV_shape = QKV_->shape();
    bs_ = QKV_shape[0];
    seq_len_q_ = QKV_shape[1];
    seq_len_kv_ = QKV_shape[1];
    head_num_ = QKV_shape[3];
    head_size_qk_ = QKV_shape[4];
    head_size_v_ = QKV_shape[4];
    attr_map["merged_QKV"] = "True";
    hidden_size_ = head_num_ * head_size_qk_;
    attn_shape = {bs_, seq_len_q_, head_num_, head_size_v_};
  }
  dst_->set_shape(attn_shape);
  if (output.size() > 1) {
    dst_min_->set_shape({bs_ * seq_len_q_});
    dst_max_->set_shape({bs_ * seq_len_q_});
  }
  // set kernel attr
  attr_map["approx_exp"] = "True";
  if (stable_softmax_ == false) {
    attr_map["stable_softmax"] = Q_->dtype() == "s8" ? "True" : "False";
  } else {
    attr_map["stable_softmax"] = "True";
  }

  std::vector<jd::tensor_desc> ts_descs(io::SIZE + 1, jd::tensor_desc());
  if (is_dynamic_) {
    ts_descs[io::SRC_Q] = {{bs_, seq_len_q_, head_num_, head_size_qk_}, dt::s8, ft::abcd};
    ts_descs[io::SRC_K] = {{bs_, seq_len_kv_, head_num_, head_size_qk_}, dt::s8, ft::abcd};
    ts_descs[io::SRC_V] = {{bs_, seq_len_kv_, head_num_, head_size_v_}, dt::s8, ft::abcd};
    ts_descs[io::DST] = {{bs_, seq_len_q_, head_num_, head_size_v_}, dt::s8, ft::abcd};
    if (att_mask_) {
      LOG(ERROR) << "Dynamic MHA not support mask now. Please use binary add mask instead";
      ts_descs[io::MASK] = {{bs_}, dt::s32, ft::a};
    }
    if (binary_add_mask_) ts_descs[io::BINARY_ADD] = {{bs_, 1, 1, seq_len_kv_}, dt::fp32, ft::ab};
    ts_descs[io::Q_SCALE] = {{bs_, seq_len_q_}, dt::fp32, ft::ab};
    ts_descs[io::K_SCALE] = {{bs_, seq_len_kv_}, dt::fp32, ft::ab};
    ts_descs[io::V_SCALE] = {{bs_, seq_len_kv_}, dt::fp32, ft::ab};
    ts_descs[io::ATT_SCALE] = {{1}, dt::fp32, ft::a};
    ts_descs[io::DST_SCALE] = {{bs_, seq_len_q_}, dt::fp32, ft::ab};
    rt_data_[io::ATT_SCALE] = &output_scale_;
    attr_map["stable_softmax"] = "False";
  } else {
    // scale and zero point
    const jd::tensor_desc desc_f32_scalar{{1}, dt::fp32, ft::a};
    ts_descs[io::ATT_SCALE] = desc_f32_scalar;
    rt_data_[io::ATT_SCALE] = &output_scale_;
    if (Q_->dtype() == "s8") {
      attr_map["softmax_rescale"] = std::to_string(softmax_scales_[0]);
      ts_descs[io::Q_SCALE] = desc_f32_scalar;
      ts_descs[io::K_SCALE] = desc_f32_scalar;
      ts_descs[io::V_SCALE] = desc_f32_scalar;
      ts_descs[io::SRC_DST_SCALE] = desc_f32_scalar;
      ts_descs[io::SRC_DST_ZP] = {{1}, dt::s32, ft::a};

      rt_data_[io::Q_SCALE] = &(Q_scales_[0]);
      rt_data_[io::K_SCALE] = &(K_scales_[0]);
      rt_data_[io::V_SCALE] = &(V_scales_[0]);
      rt_data_[io::SRC_DST_SCALE] = &(dst_scales_[0]);
      rt_data_[io::SRC_DST_ZP] = &QKV_zeropoint_;
    }
    ft qkv_ft = ft::abcd;
    auto qkv_dtype = (dst_->dtype() == "bf16") ? dt::bf16 : dt::s8;
    if (Q_ != nullptr) {
      ts_descs[io::SRC_Q] = {{bs_, seq_len_q_, head_num_, head_size_qk_}, qkv_dtype, qkv_ft};
      ts_descs[io::SRC_K] = {{bs_, seq_len_kv_, head_num_, head_size_qk_}, qkv_dtype, qkv_ft};
      ts_descs[io::SRC_V] = {{bs_, seq_len_kv_, head_num_, head_size_v_}, qkv_dtype, qkv_ft};
    } else {
      ts_descs[io::SRC_Q] = {attn_shape, qkv_dtype, qkv_ft};
      ts_descs[io::SRC_K] = {attn_shape, qkv_dtype, qkv_ft};
      ts_descs[io::SRC_V] = {attn_shape, qkv_dtype, qkv_ft};
    }
    if (att_mask_ != nullptr) {
      ts_descs[io::MASK] = {{bs_}, dt::s32, ft::a};
    }
    ts_descs[io::DST] = {attn_shape, (dst_->dtype() == "bf16") ? dt::bf16 : dt::u8, qkv_ft};

    if (binary_add_mask_ != nullptr) {
      const auto& badd_mask_size = binary_add_mask_->shape().size();
      LOG_IF(FATAL, badd_mask_size > dst_->shape().size()) << "Unsupported binary add mask dimension";
      ts_descs[io::BINARY_ADD] = {binary_add_mask_->shape(), dt::fp32, jd::plain_format(badd_mask_size)};
    }
  }
  jd::operator_desc op_desc(jd::kernel_kind::mha_dense, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, attr_map);
  jd::mha_dense_desc mha_dense_d(op_desc);
  mha_dense_ = jd::mha_dense(mha_dense_d);
  if (workspace_) aligned_free(workspace_);
  workspace_ =
      reinterpret_cast<void*>(aligned_alloc(ALIGNMENT, (mha_dense_.get_workspace_size() / ALIGNMENT + 1) * ALIGNMENT));
  rt_data_[io::WORKSPACE] = workspace_;
  if (!dst_reshape_.empty()) {
    vector<int64_t> dst_shape = GetDstShape(dst_reshape_, dst_->size(), {}, {});
    dst_->set_shape(dst_shape);
  }
}

void MultiHeadAttentionOperator::ForwardSparse(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int8_t *Q_data = nullptr, *K_data = nullptr, *V_data = nullptr;
  if (Q_->tensor_format() != TensorFormat::MmKMb) {
    vector<int64_t> src_prem = {2, 0, 1, 3};
    Q_->set_shape(src_shape_);
    K_->set_shape(src_shape_);
    V_->set_shape(src_shape_);
    Q_->reorder(Q_->shape(), src_prem);
    K_->reorder(K_->shape(), src_prem);
    V_->reorder(V_->shape(), src_prem);
  }
  if (Q_ != nullptr) {
    Q_data = reinterpret_cast<int8_t*>(Q_->mutable_data());
    K_data = reinterpret_cast<int8_t*>(K_->mutable_data());
    V_data = reinterpret_cast<int8_t*>(V_->mutable_data());
  } else {
    int8_t* QKV_data = reinterpret_cast<int8_t*>(QKV_->mutable_data());
    Q_data = QKV_data;
    K_data = QKV_data + hidden_size_;
    V_data = QKV_data + 2 * hidden_size_;
  }
  float* att_mask_data = reinterpret_cast<float*>(att_mask_->mutable_data());
  uint8_t* dst_data = reinterpret_cast<uint8_t*>(dst_->mutable_data());
  rt_data_ = {K_data,          Q_data,           att_mask_data,     V_data,
              dst_data,        trans_mha_tmpbuf, &seq_len_q_,       &bs_,
              &head_num_,      &head_size_qk_,   &seq_len_q_,       &(Q_scales_[0]),
              &(K_scales_[0]), &(V_scales_[0]),  &(dst_scales_[0]), &QKV_zeropoint_};
  mha_transpose_.execute(rt_data_);
  if (Q_->tensor_format() != TensorFormat::MmKMb) {
    vector<int64_t> dst_shape = dst_->shape();
    output[0]->reorder(Q_->shape(), {1, 2, 0, 3});
    dst_->set_shape(dst_shape);
  }
  this->unref_tensors(input);
}

void MultiHeadAttentionOperator::ForwardDense(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int8_t *Q_data = nullptr, *K_data = nullptr, *V_data = nullptr;
  if (Q_ != nullptr) {
    Q_data = reinterpret_cast<int8_t*>(Q_->mutable_data());
    K_data = reinterpret_cast<int8_t*>(K_->mutable_data());
    V_data = reinterpret_cast<int8_t*>(V_->mutable_data());
    // for decoder_only transformers, q_seq_len != k_seq_len
    // bs x seq_len x head_num x head_size (Q, K, V)
    if (att_mask_ != nullptr) {
      bool decoder = (Q_->shape()[1] != K_->shape()[1]);
      for (int i = 0; i < att_mask_->shape()[0]; ++i) {
        if (*(reinterpret_cast<int32_t*>(att_mask_->mutable_data()) + i) != 1) {
          decoder = false;
          break;
        }
      }
      if (decoder) {
        int32_t k_seq_len = K_->shape()[1];
#pragma omp parallel for
        for (int i = 0; i < att_mask_->shape()[0]; ++i) {
          *(reinterpret_cast<int32_t*>(att_mask_->mutable_data()) + i) = k_seq_len;
        }
      }
    }
  } else {
    int8_t* QKV_data = reinterpret_cast<int8_t*>(QKV_->mutable_data());
    Q_data = QKV_data;
    K_data = QKV_data + hidden_size_;
    V_data = QKV_data + 2 * hidden_size_;
  }

  int8_t* dst_data = reinterpret_cast<int8_t*>(dst_->mutable_data());
  rt_data_[io::SRC_Q] = Q_data;
  rt_data_[io::SRC_K] = K_data;
  rt_data_[io::SRC_V] = V_data;
  if (att_mask_ != nullptr) rt_data_[io::MASK] = reinterpret_cast<int32_t*>(att_mask_->mutable_data());
  rt_data_[io::DST] = dst_data;
  if (binary_add_mask_ != nullptr) {
    float* binary_add_mask_data = reinterpret_cast<float*>(binary_add_mask_->mutable_data());
    rt_data_[io::BINARY_ADD] = binary_add_mask_data;
  }
  if (is_dynamic_) {
    rt_data_[io::Q_SCALE] = Q_max_->mutable_data();
    rt_data_[io::K_SCALE] = K_max_->mutable_data();
    rt_data_[io::V_SCALE] = V_max_->mutable_data();
    if (output.size() > 1) rt_data_[io::DST_SCALE] = dst_max_->mutable_data();
  }
  mha_dense_.execute(rt_data_);
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(MultiHeadAttention);
}  // namespace executor
