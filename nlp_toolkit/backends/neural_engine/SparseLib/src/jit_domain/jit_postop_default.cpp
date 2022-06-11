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

#include "jit_domain/jit_postop_default.hpp"
#include "kernels/postop_types.hpp"
namespace jd {

void jit_postop_default_t::generate() {
  this->preamble();
  load_params();
  load_table_addr();
  if (is_bf16()) {
    prepare_bf16_mask();
    init_vcvtneps2bf16();
  }
  const auto shift = vlen();
  cmp(remain_element_num, 16);
  jl(reminder_loop_start, T_NEAR);
  L(vectorized_loop_start);
  if (is_bf16()) {
    load_bf16_cvt_to_f32(reg_src, addr_src);
    vector_compute(reg_src);
    cvt_f32_to_bf16_store(reg_src, addr_dst);
  } else {
    vmovups(reg_src, ptr[addr_src]);
    vector_compute(reg_src);
    vmovups(ptr[addr_dst], reg_src);
  }
  add(addr_src, shift);
  add(addr_dst, shift);
  sub(remain_element_num, 16);
  cmp(remain_element_num, 16);
  jge(vectorized_loop_start, T_NEAR);

  L(vectorized_loop_end);
  L(reminder_loop_start);
  cmp(remain_element_num, 0);
  jle(reminder_loop_end, T_NEAR);
  if (is_bf16()) {
    load_bf16_cvt_to_f32(reg_src, addr_src, true);
    vector_compute(reg_src);
    cvt_f32_to_bf16_store(reg_src, addr_dst, true);
  } else {
    vmovss(Xmm(reg_src.getIdx()), ptr[addr_src]);
    vector_compute(reg_src);
    vmovss(ptr[addr_dst], Xmm(reg_src.getIdx()));
  }

  add(addr_src, dtype_size());
  add(addr_dst, dtype_size());
  dec(remain_element_num);
  jmp(reminder_loop_start, T_NEAR);

  L(reminder_loop_end);
  this->postamble();

  prepare_table();
}

size_t jit_postop_default_t::table_off(key_t key, size_t key_off_val_shift) {
  const auto it = entry_map.find(key);
  assert(it != entry_map.end());
  const auto& te = (*it).second;
  const auto scale = te.bcast ? vlen() : sizeof(table_entry_val_t);
  return te.off + key_off_val_shift * scale;
}

Xbyak::Address jit_postop_default_t::table_val(key_t key, size_t key_off_val_shift) {
  auto off = table_off(key, key_off_val_shift);
  return ptr[p_table + off];
}

void jit_postop_default_t::vector_compute(const Zmm& zmm_src) {
  switch (param_.scheme) {
    case ssd::post_op_scheme::exp:
      exp_compute_vector_fwd(zmm_src);
      break;

    default:
      break;
  }
}

void jit_postop_default_t::exp_compute_vector_fwd(const Zmm& zmm_src) {
  /* exp code */
  vcmpps(k_mask, reg_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);
  vminps(reg_src, reg_src, table_val(exp_ln_flt_max_f));
  vmaxps(reg_src, reg_src, table_val(exp_ln_flt_min_f));
  vmovups(zmm_aux1, reg_src);

  vmulps(reg_src, reg_src, table_val(exp_log2ef));
  vaddps(reg_src, reg_src, table_val(half));
  vrndscaleps(zmm_aux2, reg_src, _op_floor & 0x3);

  // keep reg_src = fx for further computations
  vmovups(reg_src, zmm_aux2);

  // x = x - fx * ln2
  vfnmadd231ps(zmm_aux1, zmm_aux2, table_val(ln2f));

  // We do not count 2^n here, because n can reach 128 and 2^128 is not
  // representable by fp32, so to get around this problem, instead of computing
  // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
  // and 2 are numbers representable in fp32.

  // compute 2^(n-1)
  vsubps(reg_src, reg_src, table_val(one));

  vcvtps2dq(zmm_aux2, reg_src);

  vpaddd(zmm_aux2, zmm_aux2, table_val(exponent_bias));

  vpslld(zmm_aux2, zmm_aux2, n_mantissa_bits);

  // use reg_src as tmp zmm_zero when applying mask
  vxorps(reg_src, reg_src, reg_src);

  // set zeroes at those points which were < log(FLT_MIN)
  vblendmps(zmm_aux2 | k_mask, zmm_aux2, reg_src);

  // compute polynomial

  vmovups(reg_src, table_val(exp_pol, 4));

  vfmadd213ps(reg_src, zmm_aux1, table_val(exp_pol, 3));
  vfmadd213ps(reg_src, zmm_aux1, table_val(exp_pol, 2));
  vfmadd213ps(reg_src, zmm_aux1, table_val(exp_pol, 1));

  vfmadd213ps(reg_src, zmm_aux1, table_val(one));
  vfmadd213ps(reg_src, zmm_aux1, table_val(exp_pol, 0));

  // y = y * 2^n

  vmulps(reg_src, reg_src, zmm_aux2);

  vmulps(reg_src, reg_src, table_val(two));
}

void jit_postop_default_t::assign_regs() {
  k_mask = Xbyak::Opmask(1);
  zmm_aux1 = Zmm(1);
  zmm_aux2 = Zmm(2);
  zmm_aux3 = Zmm(3);
  zmm_aux4 = Zmm(4);
  reg_src = Zmm(6);
}

void jit_postop_default_t::prepare_bf16_mask() {
  remain_task_mask = Xbyak::Opmask(6);
  scratch_ = Xbyak::Reg64(r10);
  one_ = Zmm(26);
  even_ = Zmm(27);
  selector_ = Zmm(28);
  tr0_ = Zmm(29);

  sub(rsp, 8);
  mov(ptr[rsp], scratch_);
  mov(scratch_.cvt32(), 0x1);
  kmovd(remain_task_mask, scratch_.cvt32());
  mov(scratch_, ptr[rsp]);
  add(rsp, 8);
}

void jit_postop_default_t::init_vcvtneps2bf16() {
  xor_(scratch_, scratch_);
  mov(scratch_.cvt32(), 0x1);
  vpbroadcastd(one_, scratch_.cvt32());

  xor_(scratch_, scratch_);
  mov(scratch_.cvt32(), 0x7fff);
  vpbroadcastd(even_, scratch_.cvt32());

  xor_(scratch_, scratch_);
  mov(scratch_.cvt32(), 0x110022);
  vpbroadcastd(selector_, scratch_.cvt32());
}

void jit_postop_default_t::load_bf16_cvt_to_f32(Xbyak::Zmm reg_src, Xbyak::Reg64 addr_src, bool is_tail,
                                                size_t offset) {
  reg_src = is_tail ? reg_src | remain_task_mask | Xbyak::util::T_z : reg_src;
  vpmovzxwd(reg_src, ptr[addr_src + offset]);
  vpslld(reg_src, reg_src, 16);
}

void jit_postop_default_t::cvt_f32_to_bf16_store(Xbyak::Zmm reg_src, Xbyak::Reg64 addr_dst, bool is_tail,
                                                 size_t offset) {
  Ymm ymm_bf16 = Ymm(reg_src.getIdx());
  vpsrld(tr0_, reg_src, 16);
  vpandd(tr0_, tr0_, one_);
  vpaddd(tr0_, even_, tr0_);
  vpaddd(tr0_, reg_src, tr0_);
  vfixupimmps(tr0_, reg_src, selector_, 0);
  vpsrad(tr0_, tr0_, 16);
  vpmovdw(ymm_bf16, tr0_);

  if (!is_tail) {
    vmovdqu16(ptr[addr_dst + offset], ymm_bf16);
  } else {
    vmovdqu16(ptr[addr_dst + offset] | remain_task_mask, ymm_bf16);
  }
}

void jit_postop_default_t::prepare_table() {
  align(64);
  L(l_table);

  assert(sizeof(table_entry_val_t) == 4);

  for (auto it = entry_map.begin(); it != entry_map.end(); it++) {
    const auto& te = (*it).second;
    const auto len = te.bcast ? 64u : sizeof(table_entry_val_t);
    for (size_t d = 0; d < len; d += sizeof(table_entry_val_t)) dd(te.val);
  }
}

void jit_postop_default_t::register_table_entries() {
  static const table_t common_values{{zero, {0x00000000, true}},      {half, {0x3f000000, true}},
                                     {one, {0x3f800000, true}},       {two, {0x40000000, true}},
                                     {minus_one, {0xbf800000, true}}, {minus_two, {0xc0000000, true}},
                                     {ln2f, {0x3f317218, true}},      {positive_mask, {0x7fffffff, true}},
                                     {sign_mask, {0x80000000, true}}, {exponent_bias, {0x0000007f, true}}};

  static const table_t exp_consts{
      {exp_log2ef, {0x3fb8aa3b, true}}, {exp_ln_flt_max_f, {0x42b17218, true}}, {exp_ln_flt_min_f, {0xc2aeac50, true}}};

  static const table_t exp_polynomial{
      // p0 = 1.0f
      {exp_pol, {0x3f7ffffb, true}},  // p1 = 0.999999701f
      {exp_pol, {0x3efffee3, true}},  // p2 = 0.499991506f
      {exp_pol, {0x3e2aad40, true}},  // p3 = 0.166676521f
      {exp_pol, {0x3d2b9d0d, true}},  // p4 = 0.0418978221f
      {exp_pol, {0x3c07cfce, true}}   // p5 = 0.00828929059f
  };

  auto push_arg_entry_of = [&](const key_t key, const table_entry_val_t val, const bool broadcast) {
    mapped_table_entry_t te{0, val, broadcast};
    entry_map.insert(std::make_pair(key, te));
  };

  auto push_entries_of = [&](const table_t& t) {
    for (auto it = t.begin(); it != t.end(); it++) {
      auto key = it->first;
      auto te = it->second;
      push_arg_entry_of(key, te.val, te.bcast);
    }
  };

  struct need_t {
    need_t(ssd::post_op_scheme op_type) {
      switch (op_type) {
        case ssd::post_op_scheme::exp:
          exp_ = true;
          break;
        default:
          break;
      }
    }

    bool exp_ = false;

    bool exp() const { return exp_; }
  };

  need_t need(param_.scheme);

  push_entries_of(common_values);
  if (need.exp()) {
    push_entries_of(exp_consts);
    push_entries_of(exp_polynomial);
  }

  size_t off = 0;
  for (auto it = entry_map.begin(); it != entry_map.end(); it++) {
    auto& te = (*it).second;
    te.off = off;
    off += te.bcast ? 64u : sizeof(table_entry_val_t);
  }
}
}  // namespace jd