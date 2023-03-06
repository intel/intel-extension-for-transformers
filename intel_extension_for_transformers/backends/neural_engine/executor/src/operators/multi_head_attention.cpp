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

#include "operator_registry.hpp"
namespace executor {
static unordered_map<string, jd::data_type> type2sparsemem{
    {"fp32", jd::data_type::fp32}, {"s32", jd::data_type::s32}, {"fp16", jd::data_type::fp16},
    {"u8", jd::data_type::u8},     {"s8", jd::data_type::s8},   {"bf16", jd::data_type::bf16}};

MultiHeadAttenionOperator::MultiHeadAttenionOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf), Q_perm_({}), K_perm_({}), V_perm_({}), dst_perm_({}), output_scale_(1.) {
  auto attrs_map = operator_conf_->attributes();

  int max_threads = std::min(32, omp_get_max_threads());
  trans_mha_tmpbuf = reinterpret_cast<uint8_t*>(aligned_alloc(64, max_threads * Size2M));

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
  iter = attrs_map.find("output_scale");
  if (iter != attrs_map.end()) {
    output_scale_ = StringToNum<float>(attrs_map["output_scale"]);
  }
  iter = attrs_map.find("reshape");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&dst_reshape_, attrs_map["reshape"], ",");
  }
}

MultiHeadAttenionOperator::~MultiHeadAttenionOperator() { aligned_free(trans_mha_tmpbuf); }

void MultiHeadAttenionOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  dst_ = output[0];
  if (input.size() == 14) {
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
  } else {
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
  }
}

void MultiHeadAttenionOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  dst_->set_dtype("u8");
  string dtype;
  if (Q_ != nullptr)
    dtype = Q_->dtype();
  else
    dtype = QKV_->dtype();
  LOG_IF(FATAL, dtype != "s8") << "only support int8, but get " << dtype;
  Q_scales = GetScales(Q_min_->data(), Q_max_->data(), Q_min_->size(), dtype);
  K_scales = GetScales(K_min_->data(), K_max_->data(), K_min_->size(), dtype);
  V_scales = GetScales(V_min_->data(), V_max_->data(), V_min_->size(), dtype);
  QK_scales = GetScales(QK_min_->data(), QK_max_->data(), QK_min_->size(), "u8");  // after_softmax
  dst_scales = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
  QK_rescales = GetRescales(Q_scales, K_scales, {}, "fp32");
  QK_rescale_ = QK_rescales[0] * output_scale_;
  softmax_rescale_ = QK_scales[0];
  QKV_zeropoint_ = GetZeroPoints(dst_min_->data(), dst_scales, dst_->dtype())[0];
  QKV_rescale_ = GetRescales(QK_scales, V_scales, dst_scales, dst_->dtype())[0];
}

void MultiHeadAttenionOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  std::unordered_map<std::string, std::string> op_attrs;
  src_shape_ = Q_->shape();
  if (Q_->tensor_format() != TensorFormat::MmKMb) {
    auto& Q_shape = Q_->shape();
    bs_ = Q_shape[2];
    seq_len_ = Q_shape[3];
    head_num_ = Q_shape[0];
    head_size_ = Q_shape[1];
  } else {
    auto& Q_shape = Q_->shape();
    bs_ = Q_shape[0];
    seq_len_ = Q_shape[4];
    head_num_ = Q_shape[1];
    head_size_ = Q_shape[2];
  }

  hidden_size_ = head_num_ * head_size_;
  op_attrs["seq_pad"] = std::to_string(seq_len_);
  op_attrs["batch"] = std::to_string(bs_);
  op_attrs["head_num"] = std::to_string(head_num_);
  op_attrs["k"] = std::to_string(head_size_);
  op_attrs["seq_len"] = std::to_string(seq_len_);
  op_attrs["scaleQ"] = std::to_string(1 / Q_scales[0] * 0.125);
  op_attrs["scaleK"] = std::to_string(1 / K_scales[0]);
  op_attrs["scaleV"] = std::to_string(1 / V_scales[0]);
  op_attrs["scaleRet"] = std::to_string(1 / dst_scales[0]);
  op_attrs["zeropointRet"] = std::to_string(QKV_zeropoint_);

  scaleQ = 1 / Q_scales[0] * 0.125;
  scaleK = 1 / K_scales[0];
  scaleV = 1 / V_scales[0];
  scaleRet = 1 / dst_scales[0];
  zeropointRet = QKV_zeropoint_;

  jd::data_type dt = jd::data_type::s8;
  jd::format_type ft = jd::format_type::undef;

  jd::tensor_desc K_desc = {{bs_, head_num_, head_size_, seq_len_}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc Q_desc = {{bs_, head_num_, head_size_, seq_len_}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc mask_desc = {{bs_, seq_len_}, jd::data_type::fp32, jd::format_type::undef};
  jd::tensor_desc V_desc = {{bs_, head_num_, head_size_, seq_len_}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc ret_desc = {{bs_, head_num_, head_size_, seq_len_}, jd::data_type::u8, jd::format_type::undef};

  std::vector<jd::tensor_desc> ts_descs = {K_desc, Q_desc, mask_desc, V_desc, ret_desc};

  jd::operator_desc trans_attention_desc(jd::kernel_kind::transpose_mha, jd::kernel_prop::forward_inference,
                                         jd::engine_kind::cpu, ts_descs, op_attrs);
  jd::transpose_mha_desc transpose_mha_desc(trans_attention_desc);
  mha_transpose_ = jd::transpose_mha(transpose_mha_desc);
  dst_->set_shape({bs_, seq_len_, head_num_, head_size_});
  if (!dst_reshape_.empty()) {
    vector<int64_t> dst_shape = GetDstShape(dst_reshape_, dst_->size(), {}, {});

    if (Q_->tensor_format() == TensorFormat::MmKMb) {
      dst_->set_shape({bs_, hidden_size_, seq_len_});
    } else {
      dst_->set_shape(dst_shape);
    }
  }
}

template <typename _T>
static void matrix_transpose(_T* mat, size_t rows, size_t cols, _T* tmat) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      tmat[j * rows + i] = mat[i * cols + j];
    }
  }
}

template <typename T1, typename T2>
static void ref_mm_row_NN_f32(T1* matA, T2* matB, float* matC, float* matD, int m, int n, int k, float alpha,
                              float beta) {
  int NBlock = 128;
  if (matD != NULL) matD[0] = matD[0];
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i += NBlock) {
    for (int j = 0; j < m; j++) {
      int remainn = i + NBlock <= n ? NBlock : n - i;
      for (int ii = 0; ii < remainn; ii++) {
        auto tmp = 0.f;
        for (int ik = 0; ik < k; ik++) {
          float v1 = matA[ik + j * k];
          float v2 = matB[ik * n + i + ii];
          tmp += v1 * v2 * alpha;
        }
        tmp += beta;
        matC[(i + ii) + j * n] = tmp;
      }
    }
  }
}

void MultiHeadAttenionOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int8_t *Q_data = nullptr, *K_data = nullptr, *V_data = nullptr;
  if (Q_->tensor_format() != TensorFormat::MmKMb) {
    vector<int64_t> src_prem = {2, 0, 1, 3};
    Q_->set_shape(src_shape_);
    K_->set_shape(src_shape_);
    V_->set_shape(src_shape_);
    Q_->reorder(Q_->shape(), src_prem);
    K_->reorder(K_->shape(), src_prem);
    V_->reorder(V_->shape(), src_prem);
    std::cout << "+++++++++++++++++++++++++++not 3D+++++++" << std::endl;
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
  rt_data_ = {K_data,     Q_data,      att_mask_data, V_data,  dst_data, trans_mha_tmpbuf, &seq_len_, &bs_,
              &head_num_, &head_size_, &seq_len_,     &scaleQ, &scaleK,  &scaleV,          &scaleRet, &zeropointRet};
  mha_transpose_.execute(rt_data_);
  if (Q_->tensor_format() != TensorFormat::MmKMb) {
    vector<int64_t> dst_shape = dst_->shape();
    output[0]->reorder(Q_->shape(), {1, 2, 0, 3});
    dst_->set_shape(dst_shape);
  }
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(MultiHeadAttenion);
}  // namespace executor
