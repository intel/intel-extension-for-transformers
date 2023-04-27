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
#include <immintrin.h>

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
  typedef void (*sum_callback)(int64_t, int, char*, __m512*, __m512*);
  typedef void (*norm_callback)(int, int, int, const float*, const float*, char*, char*, __m512*, __m512*);

  enum fwd_mode { parallelG, parallelC };

 public:
  explicit GroupNormOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~GroupNormOperator() {
#ifdef WITH_SPARSELIB
    if (work_space != nullptr) free(work_space);
#endif
  }

  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void GroupNormParallelG(const void* src_data, const float* gamma_data, const float* beta_data, void* dst_data,
                          const vector<int64_t>& src_shape);
  void NormGroup(char* src_data, const float* gamma_data, const float* beta_data, char* dst_data, int map_size);

 private:
  float epsilon_ = 1e-05;
  int64_t group_ = 1;
  int64_t channels_ = -1;
  int64_t channels_per_group_ = -1;
  bool affine_ = false;
  int dt_bytewidth_ = 2;  // default bfloat16
  sum_callback sum_func = nullptr;
  norm_callback norm_func = nullptr;
  fwd_mode mode = parallelG;
#ifdef WITH_SPARSELIB
  jd::tensor_desc src_desc_;
  jd::tensor_desc dst_desc_;
  jd::tensor_desc gamma_desc_;
  jd::tensor_desc beta_desc_;
  jd::groupnorm groupnorm_ker;
  void* work_space;
#endif
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_GROUP_NORM_HPP_
