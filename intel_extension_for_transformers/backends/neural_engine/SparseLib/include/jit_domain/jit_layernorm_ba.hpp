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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_LAYERNORM_BA_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_LAYERNORM_BA_HPP_

#include <utility>
#include <vector>
#include <map>
#include <set>
#include "jit_generator.hpp"
#include "utils.hpp"
#include "kernels/layernorm_ba_types.hpp"
#include "jit_domain/jit_eltwise_injector.hpp"

#define LNBA_GET_OFF(field) offsetof(ssd::layernorm_ba_data_t, field)

namespace jd {
class jit_layernorm_ba_t : public jit_generator {
  using Zmm = Xbyak::Zmm;
  using Reg64 = Xbyak::Reg64;
  using Opmask = Xbyak::Opmask;

 public:
  explicit jit_layernorm_ba_t(const ssd::layernorm_ba_param_t& param) : jit_generator(), param_(param) {
    eltwise_injector.init_tb_allocate_set(param_.postop_attrs);
    // xmm max num=16
    unroll_degree = 16;
    while (param_.row_num % unroll_degree != 0) unroll_degree -= 1;
    assign_regs();
    eltwise_injector.eltwise_injector_init(this, param_.postop_attrs);
    gen_load_offset();
  }
  virtual ~jit_layernorm_ba_t() {}

 private:
  void generate() override;
  void reset_unroll_reg_idxs(int degree);
  std::pair<int, int> get_unroll_add_idx(int begin);
  bool check_unroll_add_done();
  void binary_add(int degree, Zmm dst);
  void assign_regs();
  void escape_regs(int degree);
  void gen_load_offset();

  void load_params() {
    mov(src_addr, ptr[reg_param + LNBA_GET_OFF(src)]);
    mov(dst_addr, ptr[reg_param + LNBA_GET_OFF(dst)]);
    mov(reg_alpha, ptr[reg_param + LNBA_GET_OFF(alpha)]);
    mov(reg_beta, ptr[reg_param + LNBA_GET_OFF(beta)]);
    mov(one_div_n, ptr[reg_param + LNBA_GET_OFF(one_div_n)]);
  }

 private:
  ssd::layernorm_ba_param_t param_;
  jit_eltwise_injector eltwise_injector;
  int unroll_degree;
  const int zmm_byte_size = 64;
  std::vector<int> unroll_reg_idxs;
  std::map<int, int> src_load_offset;
  std::map<int, int> dst_load_offset;
  std::map<reg_type, std::set<int>> reg_map;

  Zmm zmm_one;
  Zmm zmm_mean;
  Zmm zmm_mean_pow;
  Zmm zmm_powx_mean;
  Zmm zmm_var;
  Zmm zmm_alpha;
  Zmm zmm_beta;
  Zmm zmm_eps;
  Reg64 reg_param;
  Reg64 src_addr;
  Reg64 dst_addr;
  Reg64 one_div_n;
  Reg64 reg_col;
  Reg64 reg_row;
  Reg64 reg_src_offset;
  Reg64 reg_dst_offset;
  Reg64 reg_alpha;
  Reg64 reg_beta;
  Reg64 reg_affine_offset;

  Xbyak::Label col_loop_start;
  Xbyak::Label mean_loop_start;
  Xbyak::Label var_loop_start;
  Xbyak::Label norm_loop_start;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_LAYERNORM_BA_HPP_
