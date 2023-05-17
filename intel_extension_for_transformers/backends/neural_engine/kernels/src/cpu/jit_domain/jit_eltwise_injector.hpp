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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_ELTWISE_INJECTOR_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_ELTWISE_INJECTOR_HPP_

#include <glog/logging.h>
#include <utility>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include "jit_generator.hpp"
#include "src/utils.hpp"
#include "param_types.hpp"
#include "regs_pool.hpp"

namespace jd {
class jit_eltwise_injector {
 public:
  jit_eltwise_injector() {}
  virtual ~jit_eltwise_injector() {}

  void eltwise_injector_init(jit_generator* ptr, const std::vector<postop_attr>& postop_attrs);
  void vector_compute(const Xbyak::Zmm& zmm_src, const std::vector<postop_attr>& postop_attrs,
                      std::vector<int> postop_idxs = {});
  void escape_regs(reg_type type, int reg_idx);
  template <typename reg_t>
  inline typename std::enable_if<std::is_base_of<Xbyak::Xmm, reg_t>::value, void>::type escape_regs(const reg_t& reg) {
    escape_regs(reg_type::zmm, reg.getIdx());
  }
  template <typename reg_t>
  inline typename std::enable_if<std::is_base_of<Xbyak::Reg32e, reg_t>::value, void>::type escape_regs(
      const reg_t& reg) {
    escape_regs(reg_type::reg64, reg.getIdx());
  }
  template <typename reg_t>
  inline typename std::enable_if<std::is_base_of<Xbyak::Opmask, reg_t>::value, void>::type escape_regs(
      const reg_t& reg) {
    escape_regs(reg_type::mask, reg.getIdx());
  }

  void escape_erase(reg_type type, int reg_idx = -1);
  void escape_rp_all_type(regs_pool* rp);
  void init_tb_allocate_set(const std::vector<postop_attr>& postop_attrs);
  int max_zmm_allocate_num() { return zmm_tb_allocate.size(); }
  int max_mask_allocate_num() { return mask_tb_allocate.size(); }
  int max_reg64_allocate_num() { return reg64_tb_allocate.size(); }
  void prepare_table();

 private:
  void assign_regs();
  void exp_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void low_precision_exp_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void swish_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void tanh_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void gelu_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void relu_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void quantize_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void dequantize_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void linear_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void bit8_lut_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void bit16_lut_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void register_table_entries(const std::vector<postop_attr>& postop_attrs);
  void assert_check(const std::vector<postop_attr>& postop_attrs);
  template <typename REG_TYPE>
  void escape_from_rp(reg_type type, regs_pool* rp);

  uint32_t get_bit8_lut_term(int integer, const std::vector<postop_attr>& postop_attrs, data_type output_dt);
  uint32_t get_bit16_lut_term(int integer, const std::vector<postop_attr>& postop_attrs, data_type output_dt);
  std::string get_attr_idx_key(const postop_attr& attr);  // for get the key of alpha_idx,beta_idx,scale_idx map.

  void load_table_addr() { h->mov(p_table, l_table); }

 private:
  postop_attr cur_postop_attr_;
  jit_generator* h = nullptr;
  std::unordered_map<reg_type, std::set<int>> used_regs;
  std::unordered_map<std::string, int> alpha_idx_map;
  std::unordered_map<std::string, int> beta_idx_map;
  std::unordered_map<std::string, int> scale_idx_map;
  std::set<Xbyak::Reg*> reg64_tb_allocate;
  std::set<Xbyak::Reg*> mask_tb_allocate;
  std::set<Xbyak::Reg*> zmm_tb_allocate;

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
    low_precision_exp_const_v0,
    low_precision_exp_const_v1,
    low_precision_exp_const_v2,
    exchange_zmm_low256_high256,
    bit8_lut_term,
    bit8_64,
    bit8_255,
    bit16_lut_term,
    bit16_32,
    bit16_255,
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
  mapped_table_t entry_map = {};
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_ELTWISE_INJECTOR_HPP_
