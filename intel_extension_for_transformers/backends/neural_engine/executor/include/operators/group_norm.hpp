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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_GROUP_NORM_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_GROUP_NORM_HPP_
#include <assert.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#ifdef WITH_SPARSELIB
#include "kernels/include/interface.hpp"
#endif
namespace executor {

/**
 * @brief A Group Normalization operator.
 *
 */

void GroupNormRef(const float* src_data, const float* gamma_data, const float* beta_data, float* dst_data,
                  const vector<int64_t>& src_shape, const float eps, const int64_t group, const int64_t channels,
                  const bool affine);

class GroupNormOperator : public Operator {
 public:
  explicit GroupNormOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~GroupNormOperator() {
#ifdef WITH_SPARSELIB
    if (work_space != nullptr) aligned_free(work_space);
#endif
  }

  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  float epsilon_ = 1e-05;
  int64_t group_ = 1;
  int64_t channels_ = -1;
  bool affine_ = false;
  bool swish_fusion_ = false;
  float swish_beta_ = 1.f;
#ifdef WITH_SPARSELIB
  jd::tensor_desc src_desc_;
  jd::tensor_desc dst_desc_;
  jd::tensor_desc gamma_desc_;
  jd::tensor_desc beta_desc_;
  jd::groupnorm groupnorm_ker;
  void* work_space = nullptr;
#endif
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_GROUP_NORM_HPP_
