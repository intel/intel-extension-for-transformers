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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_AMX_S8S8_DYNAMIC_QUANT_MATMUL_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_AMX_S8S8_DYNAMIC_QUANT_MATMUL_HPP_

#include <memory>
#include <string>
#include "jit_generator.hpp"
#include "utils.hpp"
#include "kernels/dynamic_quant_matmul_types.hpp"
namespace jd {
class jit_amx_s8s8_dynamic_quant_matmul_t : public jit_generator {
 public:
  explicit jit_amx_s8s8_dynamic_quant_matmul_t(const ssd::dynamic_quant_matmul_param_t& param)
      : jit_generator(), param_(param) {}
  virtual ~jit_amx_s8s8_dynamic_quant_matmul_t() {}

 private:
  ssd::dynamic_quant_matmul_param_t param_;

 private:
  void generate() override;
  Opmask matC_n_mask = Opmask(2);
  Opmask scaleC_mask = Opmask(3);
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_AMX_S8S8_DYNAMIC_QUANT_MATMUL_HPP_
