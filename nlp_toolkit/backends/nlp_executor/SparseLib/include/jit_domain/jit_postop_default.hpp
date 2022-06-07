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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_POSTOP_DEFAULT_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_POSTOP_DEFAULT_HPP_

#include "jit_generator.hpp"
#include "utils.hpp"
#include "kernels/postop_types.hpp"
#include <map>

#define GET_OFF(field) offsetof(ssd::postop_data_t, field)

namespace jd {
class jit_postop_default_t : public jit_generator {
  using Zmm = Xbyak::Zmm;
  using Ymm = Xbyak::Ymm;
  using Xmm = Xbyak::Xmm;

 public:
  explicit jit_postop_default_t(const ssd::postop_param_t& param) : jit_generator(), param_(param) {
    register_table_entries();
    assign_regs();
  }
  virtual ~jit_postop_default_t() {}

 private:
  void generate() override;
  void assign_regs();
  void vector_compute(const Xbyak::Zmm& zmm_src);
  void exp_compute_vector_fwd(const Xbyak::Zmm& zmm_src);
  void load_bf16_cvt_to_f32(Xbyak::Zmm reg_src, Xbyak::Reg64 src_addr, bool is_tail = false, size_t offset = 0);
  void cvt_f32_to_bf16_store(Xbyak::Zmm reg_src, Xbyak::Reg64 addr_dst, bool is_tail = false, size_t offset = 0);
  void init_vcvtneps2bf16();

  bool is_bf16() {
    if (param_.dt == ssd::data_type::bf16) return true;
    return false;
  };

  size_t vlen() {
    switch (param_.dt) {
      case jd::ssd::data_type::fp32:
        return 64u;
      case jd::ssd::data_type::bf16:
        return 32u;
    }
    return 0;
  };

  size_t dtype_size() {
    switch (param_.dt) {
      case jd::ssd::data_type::fp32:
        return 4u;
      case jd::ssd::data_type::bf16:
        return 2u;
    }
    return 0;
  };

 public:
  const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;
  const int EVEX_max_8b_offt = 0x200;

 private:
  ssd::postop_param_t param_;

  /*labels*/
  Xbyak::Label l_table;
  Xbyak::Label vectorized_loop_start;
  Xbyak::Label vectorized_loop_end;
  Xbyak::Label reminder_loop_start;
  Xbyak::Label reminder_loop_end;

  /* registers for fwd*/
  Xbyak::Reg64 reg_param = rdi;
  Xbyak::Reg64 p_table = Xbyak::util::rax;
  Zmm reg_src;
  Xbyak::Reg64 addr_src = r15;
  Xbyak::Reg64 addr_dst = r14;
  Xbyak::Reg64 remain_element_num = rsi;

  /* register for bf16 tasks*/
  Xbyak::Opmask remain_task_mask;
  Zmm one_, even_, selector_, tr0_;
  Xbyak::Reg64 scratch_;

  Zmm zmm_mask, zmm_aux0, zmm_aux1, zmm_aux2, zmm_aux3, zmm_aux4, zmm_tmp;
  Ymm ymm_tmp;
  Xmm xmm_tmp;
  Xbyak::Opmask k_mask;
  static constexpr int n_mantissa_bits = 23;

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
    scale = 0,         // scale argument
    alpha,             // alpha argument
    beta,              // beta argument
    zero,              // 0.f
    half,              // 0.5f
    one,               // 1.f  or  mask for exponent bits
    two,               // 2.f
    three,             // 3.f
    six,               // 6.f
    minus_one,         // -1.f  or  changes sign to opposite
    minus_two,         // -2.f
    minus_three,       // -3.f
    ln2f,              // 0.69314718f
    positive_mask,     // changes sign to positive
    sign_mask,         // gets sign value
    exponent_bias,     // (127 = 2^7 - 1), gets exponent bits
    exp_log2ef,        // 1.44269502f - formula-based for approx
    exp_ln_flt_max_f,  // logf(FLT_MAX) - max normal value
    exp_ln_flt_min_f,  // logf(FLT_MIN) - min normal value
    exp_pol,           // see correspondent table for float values
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
  void load_table_addr() { mov(p_table, l_table); };
  void register_table_entries();
  void prepare_table();
  void prepare_bf16_mask();

  void load_params() {
    mov(addr_dst, ptr[reg_param + GET_OFF(dst)]);
    mov(addr_src, ptr[reg_param + GET_OFF(src)]);
    mov(remain_element_num, ptr[reg_param + GET_OFF(element_num)]);
  }

  mapped_table_t entry_map;
};
}  // namespace jd
#endif