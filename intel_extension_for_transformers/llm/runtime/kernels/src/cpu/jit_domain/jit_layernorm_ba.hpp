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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_LAYERNORM_BA_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_LAYERNORM_BA_HPP_

#include <utility>
#include <vector>
#include <map>
#include <set>
#include "jit_generator.hpp"
#include "src/utils.hpp"
#include "kernels/layernorm_ba_types.hpp"
#include "jit_eltwise_injector.hpp"
#include "jit_binary_injector.hpp"

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
    binary_injector.binary_injector_init(this);
  }
  virtual ~jit_layernorm_ba_t() {}

 private:
  void generate() override;
  void normal_gen();
  void normal_reset_unroll_reg_idxs(int degree);
  std::pair<int, int> normal_get_unroll_add_idx(int begin);
  bool normal_check_unroll_add_done();
  void normal_binary_add(int degree, Zmm dst);
  void direct_handel_var(int degree, Reg64 reg_var);
  void direct_handel_norm(int degree, Reg64 src_addr, Reg64 dst_addr, Reg64 reg_mean, bool tail = false);
  void assign_regs();
  void escape_regs(int degree);
  void normal_gen_load_offset();

  void normal_load_params() {
    mov(src_addr, ptr[reg_param + LNBA_GET_OFF(src)]);
    mov(dst_addr, ptr[reg_param + LNBA_GET_OFF(dst)]);
    mov(reg_alpha, ptr[reg_param + LNBA_GET_OFF(alpha)]);
    mov(reg_beta, ptr[reg_param + LNBA_GET_OFF(beta)]);
  }

  void direct_gen();
  void direct_load_params() {
    mov(src_addr, ptr[reg_param + LNBA_GET_OFF(src)]);
    mov(dst_addr, ptr[reg_param + LNBA_GET_OFF(dst)]);
    mov(reg_alpha, ptr[reg_param + LNBA_GET_OFF(alpha)]);
    mov(reg_beta, ptr[reg_param + LNBA_GET_OFF(beta)]);
    if (param_.split_output) mov(dst2_addr, ptr[reg_param + LNBA_GET_OFF(dst2)]);
    // reg alisa
    Reg64 reg_mean = reg_src_offset;
    Reg64 reg_var = reg_dst_offset;
    mov(reg_mean, ptr[reg_param + LNBA_GET_OFF(mean)]);
    mov(reg_var, ptr[reg_param + LNBA_GET_OFF(var)]);
  }

 private:
  ssd::layernorm_ba_param_t param_;
  jit_eltwise_injector eltwise_injector;
  jit_binary_injector binary_injector;
  int unroll_degree;
  const int zmm_byte_size = 64;
  const int xmm_byte_size = 16;
  std::vector<bool> unroll_reg_idxs;
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
  Reg64 reg_param;
  Reg64 src_addr;
  Reg64 dst_addr;
  Reg64 dst2_addr;
  Reg64 dst2_addr_cp;
  Reg64 reg_col;
  Reg64 reg_row;
  Reg64 reg_src_offset;
  Reg64 reg_dst_offset;
  Reg64 reg_alpha;
  Reg64 reg_beta;
  Reg64 reg_affine_offset;
  Reg64 reg_batch;
  Opmask tail_mask;

  Xbyak::Label col_loop_start;
  Xbyak::Label mean_loop_start;
  Xbyak::Label var_loop_start;
  Xbyak::Label norm_loop_start;
  Xbyak::Label batch_loop_start;
  Xbyak::Label tail_loop_start;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_LAYERNORM_BA_HPP_
