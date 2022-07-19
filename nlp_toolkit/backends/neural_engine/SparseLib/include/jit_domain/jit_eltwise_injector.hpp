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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_ELTWISE_INJECTOR_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_ELTWISE_INJECTOR_HPP_

#include "jit_generator.hpp"
#include "utils.hpp"
#include "param_types.hpp"
#include <map>
#include <set>

namespace jd {
class jit_eltwise_injector {
  using Zmm = Xbyak::Zmm;
  using Ymm = Xbyak::Ymm;
  using Xmm = Xbyak::Xmm;

 public:
  explicit jit_eltwise_injector(){};
  virtual ~jit_eltwise_injector() {}

  void eltwise_injector_init(jit_generator* ptr, const std::vector<postop_attr>& postop_attrs);
  void vector_compute(const Xbyak::Zmm& zmm_src, const std::vector<postop_attr>& postop_attrs);
  void escape_regs(reg_type type, int reg_idx);
  void escape_erase(reg_type type, int reg_idx = -1);
  void prepare_table();

 private:
  void assign_regs();
  void exp_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void tanh_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void gelu_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void relu_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void quantize_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void dequantize_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void register_table_entries(const std::vector<postop_attr>& postop_attrs);
  void assert_check(const std::vector<postop_attr>& postop_attrs);
  void load_table_addr() { h->mov(p_table, l_table); };

 private:
  postop_attr cur_postop_attr_;
  int cur_iter_idx_;  // for alpha,beta,scale lookup.
  jit_generator* h;
  std::unordered_map<reg_type, std::set<int>> used_regs;

  /*labels*/
  Xbyak::Label l_table;

  /*register for fwd*/
  Xbyak::Reg64 p_table;
  Xbyak::Reg64 reg64_tmp;

  Zmm zmm_mask, zmm_aux0, zmm_aux1, zmm_aux2, zmm_aux3, zmm_aux4, zmm_tmp;
  Ymm ymm_tmp;
  Xmm xmm_tmp;
  Xbyak::Opmask k_mask;
  static constexpr int n_mantissa_bits = 23;
  static constexpr int max_mask_idx = 7;
  static constexpr int max_zmm_idx = 31;
  static constexpr int max_reg64_idx = 15;

  enum {
    _cmp_eq_oq = 0u,
    _cmp_lt_os = 1u,
    _cmp_le_os = 2u,
    _cmp_neq_uq = 4u,
    _cmp_nlt_us = 5u,
    _cmp_nle_us = 6u,

    _op_floor = 1u,
    _op_mxcsr = 4u,
  };

  enum key_t {
    scale = 0,                            // scale argument
    alpha,                                // alpha argument
    beta,                                 // beta argument
    zero,                                 // 0.f
    half,                                 // 0.5f
    one,                                  // 1.f  or  mask for exponent bits
    two,                                  // 2.f
    three,                                // 3.f
    six,                                  // 6.f
    minus_one,                            // -1.f  or  changes sign to opposite
    minus_two,                            // -2.f
    minus_three,                          // -3.f
    ln2f,                                 // 0.69314718f
    positive_mask,                        // changes sign to positive
    sign_mask,                            // gets sign value
    exponent_bias,                        // (127 = 2^7 - 1), gets exponent bits
    exp_log2ef,                           // 1.44269502f - formula-based for approx
    exp_ln_flt_max_f,                     // logf(FLT_MAX) - max normal value
    exp_ln_flt_min_f,                     // logf(FLT_MIN) - min normal value
    exp_pol,                              // see correspondent table for float values
    gelu_tanh_fitting_const,              // 0.044715f
    gelu_tanh_fitting_const_times_three,  // 0.134145f
    gelu_tanh_sqrt_two_over_pi,           // sqrtf(2.f/pi) = 0.797884f
    gelu_tanh_flt_max_x,
    gelu_tanh_flt_min_x,
    tanh_idx_bias,
    tanh_idx_mask,
    tanh_linear_ubound,
    tanh_saturation_lbound,
    tanh_pol_table,
    exchange_zmm_low256_high256,
    undef_key,
  };

  size_t table_off(key_t key, size_t key_off_val_shift = 0);
  Xbyak::Address table_val(key_t key, size_t key_off_val_shift = 0);
  using table_entry_val_t = uint32_t;
  using table_entry_offset_t = size_t;  // offsets are in bytes wrt p_table
  using table_entry_bcast_t = bool;

  struct table_entry_t {
    table_entry_val_t val;
    table_entry_bcast_t bcast;
  };
  struct mapped_table_entry_t {
    table_entry_offset_t off;
    table_entry_val_t val;
    table_entry_bcast_t bcast;
  };
  using table_t = std::multimap<key_t, table_entry_t>;
  using mapped_table_t = std::multimap<key_t, mapped_table_entry_t>;
  mapped_table_t entry_map;
};
}  // namespace jd
#endif