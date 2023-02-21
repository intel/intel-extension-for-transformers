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
class MultiHeadAttenionOperator : public Operator {
 public:
  explicit MultiHeadAttenionOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~MultiHeadAttenionOperator();

 public:
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

  void mha(const vector<Tensor*>& input, const vector<Tensor*>& output);
  // void ref_transmha(int8_t* matAs8, int8_t* matBs8, int8_t* matDs8, float* matC, uint8_t* matEu8);

 private:
  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);
  // Converting string variables from operators attrs to boolean, or int/float
 protected:
  Tensor *Q_ = nullptr, *K_ = nullptr, *V_ = nullptr, *QKV_ = nullptr;
  Tensor* att_mask_;
  Tensor *Q_min_, *Q_max_, *K_min_, *K_max_, *V_min_, *V_max_;  // all scale is per_tensor now
  Tensor *QK_min_, *QK_max_, *dst_min_, *dst_max_;              // all scale is per_tensor now
  Tensor* dst_;
  uint8_t* trans_mha_tmpbuf;
  const int Size2M = 1 << 21;
  int inf_count = 0;

  vector<int64_t> Q_perm_, K_perm_, V_perm_, dst_perm_;
  float output_scale_ = 1.f;
  vector<int64_t> dst_reshape_;
  vector<int64_t> src_shape_;
  float QK_rescale_, softmax_rescale_, QKV_rescale_, QKV_zeropoint_;

  vector<float> Q_scales, K_scales, V_scales, QK_scales, dst_scales, QK_rescales;

  float scaleQ, scaleK, scaleV, scaleRet;
  int bs_, seq_len_, head_num_, head_size_, hidden_size_, zeropointRet;
  //   jd::mha_dense mha_dense_;
  jd::transpose_mha mha_transpose_;
  std::vector<const void*> rt_data_;
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_MULTI_HEAD_ATTENTION_HPP_
