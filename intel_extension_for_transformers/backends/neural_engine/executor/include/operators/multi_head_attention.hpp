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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_MULTI_HEAD_ATTENTION_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_MULTI_HEAD_ATTENTION_HPP_

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "../common.hpp"
#include "../operator.hpp"
#ifdef WITH_SPARSELIB
#include "kernels/include/interface.hpp"
#endif

namespace executor {

// \brief MULTI_HEAD_ATTENTION operators
class MultiHeadAttentionOperator : public Operator {
 public:
  explicit MultiHeadAttentionOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~MultiHeadAttentionOperator();

 public:
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  void ReshapeDense(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void ReshapeSparse(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void ForwardDense(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void ForwardSparse(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);
  // Converting string variables from operators attrs to boolean, or int/float
 protected:
  Tensor *Q_ = nullptr, *K_ = nullptr, *V_ = nullptr, *QKV_ = nullptr;
  Tensor* att_mask_ = nullptr;
  Tensor* binary_add_mask_ = nullptr;
  // all scale is per_tensor now
  Tensor *Q_min_ = nullptr, *Q_max_ = nullptr, *K_min_ = nullptr, *K_max_ = nullptr, *V_min_ = nullptr,
         *V_max_ = nullptr, *QK_min_ = nullptr, *QK_max_ = nullptr, *dst_min_ = nullptr, *dst_max_ = nullptr;
  Tensor* dst_ = nullptr;

  uint8_t* trans_mha_tmpbuf = nullptr;
  const int Size2M = 1 << 21;
  int inf_count = 0;

  vector<int64_t> Q_perm_, K_perm_, V_perm_, dst_perm_;
  float output_scale_ = 1.f;
  vector<int64_t> dst_reshape_;
  vector<int64_t> src_shape_;
  vector<float> Q_scales_, K_scales_, V_scales_, softmax_scales_, dst_scales_;
  int bs_ = 0, seq_len_q_ = 0, seq_len_kv_ = 0, head_num_ = 0, head_size_qk_ = 0, head_size_v_ = 0, hidden_size_ = 0,
      QKV_zeropoint_ = 0;

  bool stable_softmax_ = false;
  bool is_sparse_ = false;
  bool is_dynamic_ = false;
  jd::mha_dense mha_dense_;
  jd::transpose_mha mha_transpose_;
  std::vector<const void*> rt_data_;
  void* workspace_ = nullptr;
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_MULTI_HEAD_ATTENTION_HPP_
