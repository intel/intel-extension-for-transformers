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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_MATMUL_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_MATMUL_HPP_
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "../common.hpp"
#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "kernels/include/interface.hpp"

namespace executor {

using dnnl::algorithm;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;

// \brief Matmul or Batchmatmul operators
class MatmulOperator : public Operator {
 public:
  explicit MatmulOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~MatmulOperator();

 public:
  // void ParseOperatorConfig();
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void AdaptAttrs(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage) override;
  void AdaptTensors(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage) override;

  void ReshapewithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void ForwardwithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output);

  void ReshapewithTransMode(const vector<Tensor*>& input, const vector<Tensor*>& output);
#if __AVX512F__
  void ForwardwithTransMode(const vector<Tensor*>& input, const vector<Tensor*>& output);
#endif

  vector<vector<string>> InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void DynamicForward(vector<int32_t>* src0_zero_points_ptr, vector<float>* rescales_ptr,
                      vector<float>* dynamic_bias_ptr);
  void RuntimeMinmax();
  void SetTransposeMode();
  void DstReshapeFusion(const vector<Tensor*>& input, const vector<Tensor*>& output);
  inline void InputShapeFallBack(const vector<Tensor*>& input);
  inline void UnsqueezePerm(vector<int64_t>* perm);
  inline void ResetPerm(vector<int64_t>* perm, const string& perm_name);
  // Converting string variables from operators attrs to boolean, or int/float
 protected:
  // Matrix can optionally be adjointed (to adjoint a matrix means to transpose and conjugate it).
  bool has_bias_ = false;
  bool format_any_;
  bool append_sum_;
  bool gelu_erf_;
  bool gelu_tanh_;
  bool tanh_;
  bool append_eltwise_;
  bool cache_weight_;
  bool binary_add_;
  bool is_dynamic_ = false;
  bool transpose_mode_ = false;
  float output_scale_ = 1.f;
  float ouput_zp_ = 0.f;
  void* scratchpad_ = nullptr;
  string output_dtype_ = "fp32";
  vector<float> dst_scales_;
  vector<int64_t> src0_perm_;
  vector<int64_t> src1_perm_;
  vector<int64_t> dst_perm_;
  vector<int64_t> reshape_;
  vector<int64_t> reshape_dims_;
  vector<float> rescales_;
  dnnl::primitive_attr attr_;
  memory scale_f32_mem_;
  memory zp_src0_mem_;

  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::stream eng_stream_ = dnnl::stream(eng_);
  dnnl::matmul::primitive_desc matmul_pd_;
  dnnl::matmul matmul_p_;
  unordered_map<int, memory> memory_args_;

  memory src0_m_;
  memory src1_m_;
  memory bias_m_;
  memory dst_m_;
  memory binary_m_;

  memory any_src0_m_;
  memory any_src1_m_;
  memory any_dst_m_;
  dnnl::reorder reorder_prim_src_;
  dnnl::reorder reorder_prim_weight_;
  dnnl::reorder reorder_prim_dst_;
  bool src_reorder_ = false;
  bool weight_reorder_ = false;
  bool dst_reorder_ = false;

  Tensor* src0_ = nullptr;
  Tensor* src1_ = nullptr;
  Tensor* bias_ = nullptr;
  Tensor* post_ = nullptr;
  Tensor* dst_ = nullptr;

  Tensor* src0_min_ = nullptr;
  Tensor* src0_max_ = nullptr;

  Tensor* src1_min_ = nullptr;
  Tensor* src1_max_ = nullptr;

  Tensor* dst_min_ = nullptr;
  Tensor* dst_max_ = nullptr;
  string append_op_;

  // src tensor shape before fall back
  vector<int64_t> src0_shape_bfb_;
  vector<int64_t> src1_shape_bfb_;

  jd::tensor_desc src0_desc_;
  jd::tensor_desc src1_desc_;
  jd::tensor_desc binary_desc_;
  jd::tensor_desc dst_desc_;
  jd::tensor_desc scale_desc_;
  jd::tensor_desc zp_desc_;
  jd::transpose_matmul transpose_matmul_;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_MATMUL_HPP_
