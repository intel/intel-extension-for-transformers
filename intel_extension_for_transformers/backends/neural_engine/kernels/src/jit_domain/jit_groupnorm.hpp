//  Copyright (c) 2022 Intel Corporation
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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_GROUPNORM_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_GROUPNORM_HPP_

#include <memory>
#include <string>
#include <functional>
#include "jit_generator.hpp"
#include "utils.hpp"

namespace jd {

struct groupnorm_param_t {
  data_type dt;
  int64_t HW;
  int channels;
  int groups;
};

struct channelwise_sum_data_t {
  void* src;
  float* sum_x_ptr;
  float* sum_powx_ptr;
};

struct channelwise_norm_data_t {
  void* src;
  void* dst;
  float* group_sum_x_ptr;
  float* group_sum_powx_ptr;
  float* gamma;
  float* beta;
};

class jit_channelwise_sum_t : public jit_generator {
 public:
  explicit jit_channelwise_sum_t(const groupnorm_param_t& param) : jit_generator(), param_(param) {}
  virtual ~jit_channelwise_sum_t() {}

 private:
  groupnorm_param_t param_;
  // Opmask sum_write_mask = Opmask(2);
  Opmask sum_write_mask;
  int unroll;

 private:
  void generate() override;
};

class jit_channelwise_norm_t : public jit_generator {
 public:
  explicit jit_channelwise_norm_t(const groupnorm_param_t& param) : jit_generator(), param_(param) {}
  virtual ~jit_channelwise_norm_t() {}

 private:
  groupnorm_param_t param_;

 private:
  void generate() override;
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_GROUPNORM_HPP_
