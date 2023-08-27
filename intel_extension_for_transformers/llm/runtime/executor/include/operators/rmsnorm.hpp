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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_RMS_NORM_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_RMS_NORM_HPP_
#include <immintrin.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "../common.hpp"
#include "../operator.hpp"

namespace executor {

/**
 * @brief A RMS Normalization operator.
 *
 */

class RmsNormOperator : public Operator {
#if __AVX512F__
  using parallelBNormCallback = void (*)(char*, const float*, char*, int, __m512*);
#else
  using parallelBNormCallback = void (*)(char*, const float*, char*, int, __m256*);
#endif

 public:
  explicit RmsNormOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~RmsNormOperator() {}

  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  template <int dt_bytewidth>
  void RmsNormParallelB(const void* src_data, const float* gamma_data, void* dst_data);

 private:
  float epsilon_ = 1e-05;
  int dt_bytewidth_ = 4;  // fp32 inference by default
  int64_t norm_dim_ = -1;
  int batchs_ = -1;
  parallelBNormCallback parallelB_norm_callback_ = nullptr;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_RMS_NORM_HPP_
