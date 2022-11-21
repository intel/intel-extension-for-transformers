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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_GELU_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_GELU_HPP_
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#ifdef WITH_SPARSELIB
#include "SparseLib/include/interface.hpp"
#endif
namespace executor {
using dnnl::algorithm;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;

/**
 * @brief A Gelu operator.
 *
 */

class GeluOperator : public Operator {
 public:
  explicit GeluOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~GeluOperator() {}

  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

  void ReshapeWithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void ForwardWithOnednn(const vector<Tensor*>& input, const vector<Tensor*>& output);

#ifdef WITH_SPARSELIB
  void ReshapeWithInt8LutAccTest(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void ForwardWithInt8LutAccTest(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void ReshapeWithSparselib(const vector<Tensor*>& input, const vector<Tensor*>& output);
  void ForwardWithSparselib(const vector<Tensor*>& input, const vector<Tensor*>& output);
  float TuneMatmulRange(float gelu_bound, float err, float step);
#endif

 private:
  string algorithm_;
  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::eltwise_forward gelu_p_;
  memory src_m_;
  memory dst_m_;

  bool int8_lut_optimize = false;
  bool int8_lut_acc_test = false;
#ifdef WITH_SPARSELIB
  jd::tensor_desc src_desc_;
  jd::tensor_desc dst_desc_;
  jd::eltwiseop eltwiseop_ker;
#endif
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_GELU_HPP_
