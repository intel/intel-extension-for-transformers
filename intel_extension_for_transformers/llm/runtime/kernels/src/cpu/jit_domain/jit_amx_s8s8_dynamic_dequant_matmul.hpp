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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_AMX_S8S8_DYNAMIC_DEQUANT_MATMUL_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_AMX_S8S8_DYNAMIC_DEQUANT_MATMUL_HPP_

#include <memory>
#include <string>

#include "jit_generator.hpp"
#include "kernels/dynamic_quant_matmul_types.hpp"
#include "src/utils.hpp"
#include "jit_eltwise_injector.hpp"
namespace jd {
class jit_amx_s8s8_dynamic_dequant_matmul_t : public jit_generator {
 public:
  explicit jit_amx_s8s8_dynamic_dequant_matmul_t(const ssd::dynamic_quant_matmul_param_t& param, size_t dst_n_dim,
                                                 bool calc_abs_max)
      : jit_generator(), param_(param), dst_n_dim_(dst_n_dim), calc_abs_max_(calc_abs_max) {
    eltwise_injector_.eltwise_injector_init(this, param_.postop_attrs);
  }
  virtual ~jit_amx_s8s8_dynamic_dequant_matmul_t() {}

 private:
  ssd::dynamic_quant_matmul_param_t param_;
  jit_eltwise_injector eltwise_injector_;

 private:
  void generate() override;
  size_t dst_n_dim_;
  bool calc_abs_max_;
  const int align_build_M = 16;
  const int align_build_N = 16;
  const int reorder_block_col_eltnum = 64;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_AMX_S8S8_DYNAMIC_DEQUANT_MATMUL_HPP_
