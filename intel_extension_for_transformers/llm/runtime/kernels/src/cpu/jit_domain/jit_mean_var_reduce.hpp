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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MEAN_VAR_REDUCE_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MEAN_VAR_REDUCE_HPP_

#include "src/utils.hpp"
#include "jit_generator.hpp"
#include "kernels/mean_var_reduce_types.hpp"

namespace jd {

class jit_mean_var_reduce_t : public jit_generator {
 public:
  explicit jit_mean_var_reduce_t(const ssd::mean_var_reduce_param_t& param) : jit_generator(), param_(param) {
    tail_BM_ = (param_.M % param_.BM == 0) ? 0 : param_.M % param_.BM;
    reciprocal_M_ = 1.f / param_.M;
  }
  virtual ~jit_mean_var_reduce_t() {}

 private:
  void generate() override;
  void load_params();
  void calc_mean_var();

 private:
  ssd::mean_var_reduce_param_t param_;
  int tail_BM_;
  float reciprocal_M_;

  const Xbyak::Reg64& reg_param = rdi;

  const Xbyak::Reg64& reg_mean_in = rsi;
  const Xbyak::Reg64& reg_var_in = rdx;
  const Xbyak::Reg64& reg_mean_out = rcx;
  const Xbyak::Reg64& reg_var_out = r8;

  const Xbyak::Reg64& num_tmp = r9;

  const Xbyak::Zmm& avg_a = zmm0;
  const Xbyak::Zmm& avg_b = zmm1;
  const Xbyak::Zmm& M_a = zmm2;
  const Xbyak::Zmm& M_b = zmm3;
  const Xbyak::Zmm& n_a = zmm4;
  const Xbyak::Zmm& n_a_float = zmm5;
  const Xbyak::Zmm& n_b = zmm6;
  const Xbyak::Zmm& delta = zmm7;
  const Xbyak::Zmm& pow2_delta = zmm8;
  const Xbyak::Zmm& scale = zmm9;
  const Xbyak::Zmm& scalee = zmm10;
  const Xbyak::Zmm& reciprocal_M = zmm11;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SOFTMAX_HPP_
