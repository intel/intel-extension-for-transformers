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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_QUANTIZE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_QUANTIZE_HPP_
#include <string>
#include <vector>
#include <memory>

#include "../common.hpp"
#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "kernels/exposed_enum.hpp"

namespace executor {
using dnnl::algorithm;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;

/**
 * @brief A Quantize operator.
 *
 */

class QuantizeOperator : public Operator {
  using io = jd::exposed_enum::dynamic_quant::io;

 public:
  explicit QuantizeOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~QuantizeOperator();

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::stream eng_stream_ = dnnl::stream(eng_);
  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void RuntimeMinmax();
  jd::tensor_desc mat_desc_;
  jd::tensor_desc scale_desc_;
  jd::tensor_desc dst_mat_desc_;

 protected:
  string output_dtype_ = "fp32";
  vector<float> scales_;
#ifdef WITH_SPARSELIB
  jd::dynamic_quant dynamic_quant_;
#endif
  Tensor* src_ = nullptr;
  Tensor* src_min_ = nullptr;
  Tensor* src_max_ = nullptr;
  Tensor* dst_ = nullptr;
  Tensor* dst_min_ = nullptr;
  Tensor* dst_max_ = nullptr;
  bool is_dynamic_ = false;
  bool per_token_ = false;  // false represent per_tensor
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_QUANTIZE_HPP_
