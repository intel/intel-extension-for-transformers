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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_DYNAMIC_QUANT_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_DYNAMIC_QUANT_HPP_

#include <memory>
#include <string>
#include "jit_generator.hpp"
#include "src/utils.hpp"

namespace jd {

struct dynamic_quant_param_t {
  data_type input_dt;
  data_type output_dt;
  size_t quantized_dim_elt_num;
  size_t ld_src;  // src leading dimension in terms of #elements
  size_t ld_dst;  // dst leading dimension in terms of #elements
};

struct dynamic_quant_data_t {
  void* src;
  void* mat_dst;
  void* scale;  // use as rcp input if scale_input_ is true
  // int process_channel;
};

class jit_dynamic_quant_t : public jit_generator {
 public:
  explicit jit_dynamic_quant_t(const dynamic_quant_param_t& param, int process_channel, bool scale_input = false)
      : jit_generator(), param_(param), process_channel_(process_channel), scale_input_(scale_input) {}
  virtual ~jit_dynamic_quant_t() {}

 private:
  dynamic_quant_param_t param_;
  int process_channel_;
  bool scale_input_;  // whether use scales input or dynamically calculate scales

 private:
  void generate() override;
};
}  // namespace jd

#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_DYNAMIC_QUANT_HPP_
