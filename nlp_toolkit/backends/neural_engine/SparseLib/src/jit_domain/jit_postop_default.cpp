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
    case ssd::post_op_scheme::gelu:
      gelu_compute_vector_fwd(zmm_src);
      break;
    default:
      break;
  }
}

void jit_postop_default_t::tanh_compute_vector_fwd(const Zmm& zmm_src) {
  // register mapping
  Zmm zmm_dst = zmm_aux1, zmm_src_shift = zmm_aux1, zmm_coeff = zmm_aux1, zmm_pol = zmm_aux2, zmm_indices = zmm_aux3,
      zmm_src_original = zmm_aux4, zmm_sign = zmm_aux4;

  const int tanh_n_polynomials = 32;

  // We split the positive domain in 33 intervals:
  // a) [0; linear_ubound]: in this interval tanh(x) = x
  // b) [linear_ubound; 0x1.8p-12]: This interval spans part of a
  //    half binade
  // c) [0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]:
  //    one interval for each half binade, there are 29 of those
  // d) [0x1.0p3; saturation_ubound]:
  //    This interval spans part of a half binade
  // e) [0x1.205966p3; saturation_ubound]: in this interval, tanh(x) = 1
  // For b-d, we need 31 polynomials and will do a table lookup for those.
  // To simplify the logic, we will also put a) in the table.
  auto coeffs_off = [&](int coeff_off, int off = 0) {
    return table_off(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
  };
  auto coeffs_address = [&](int coeff_off, int off = 0) {
    return table_val(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
  };
  auto gather_coefficient = [&](Zmm vmm_coeff, int coeff_idx, Zmm vmm_pol_idx) {
    Zmm zmm_coeff(vmm_coeff.getIdx());
    Zmm zmm_pol_idx(vmm_pol_idx.getIdx());
    vmovups(zmm_coeff, coeffs_address(coeff_idx, 0));
    vpermt2ps(zmm_coeff, zmm_pol_idx, coeffs_address(coeff_idx, 16));
  };

  // because tanh(x) = -tanh(-x), we extract sign to make x postive
  // and reapply sign at the end
  vmovups(zmm_src_original, zmm_src);
  vmovups(zmm_src_original, zmm_src);
  vandps(zmm_src, zmm_src, table_val(positive_mask));

  // We compute the indices for the table lookup
  vmovups(zmm_indices, zmm_src);
  vpsubd(zmm_indices, zmm_indices, table_val(tanh_idx_bias));
  vandps(zmm_indices, zmm_indices, table_val(tanh_idx_mask));
  vpsrld(zmm_indices, zmm_indices, 22);

  // we do the argument reduction
  vmovups(zmm_src_shift, zmm_src);
  vandps(zmm_src_shift, zmm_src_shift, table_val(tanh_idx_mask));
  vsubps(zmm_src, zmm_src, zmm_src_shift);

  // we gather and evaluate the polynonials
  gather_coefficient(zmm_pol, 6, zmm_indices);
  for (int deg = 5; deg >= 0; --deg) {
    gather_coefficient(zmm_coeff, deg, zmm_indices);
    vfmadd213ps(zmm_pol, zmm_src, zmm_coeff);
  }

  // we restore src with cleared sign, and keep sign
  vmovups(zmm_src, zmm_src_original);
  vandps(zmm_sign, zmm_sign, table_val(sign_mask));
  vandps(zmm_src, zmm_src, table_val(positive_mask));

  // Now we blend the results
  // [saturation_ubound; +inf[ : we return +/- 1
  vmovups(zmm_dst, table_val(one));
  // [linear_ubound; saturation_lbound] : we return +/- P(x)
  vmovups(zmm_mask, table_val(tanh_saturation_lbound));
  vcmpps(k_mask, zmm_mask, zmm_src, _cmp_nle_us);
  vblendmps(zmm_dst | k_mask, zmm_dst, zmm_pol);
  // [0; linear_ubound]  : we return x
  vmovups(zmm_mask, table_val(tanh_linear_ubound));
  vcmpps(k_mask, zmm_mask, zmm_src, _cmp_nle_us);
  vblendmps(zmm_dst | k_mask, zmm_dst, zmm_src);

  // We reapply the sign and return
  vxorps(zmm_dst, zmm_dst, zmm_sign);
  vmovups(zmm_src, zmm_dst);
}

void jit_postop_default_t::gelu_compute_vector_fwd(const Zmm& zmm_src) {
  vmovups(zmm_aux0, zmm_src);

  // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
  vmulps(zmm_src, zmm_src, zmm_src);
  vmovups(zmm_aux1, table_val(gelu_tanh_fitting_const));
  vfmadd213ps(zmm_src, zmm_aux1, table_val(one));
  vmulps(zmm_src, zmm_src, zmm_aux0);
  vmulps(zmm_src, zmm_src, table_val(gelu_tanh_sqrt_two_over_pi));

  // save x on stack as tanh uses vmm_aux0
  sub(rsp, 64);
  vmovups(ptr[rsp], zmm_aux0);

  // compute tanh(G(x))
  tanh_compute_vector_fwd(zmm_src);

  vmovups(zmm_aux0, ptr[rsp]);
  add(rsp, 64);

  // compute 0.5 * x * (1 + tanh(G(x)))
  vaddps(zmm_src, zmm_src, table_val(one));
  vmulps(zmm_src, zmm_src, table_val(half));
  vmulps(zmm_src, zmm_src, zmm_aux0);
}

void jit_postop_default_t::exp_compute_vector_fwd(const Zmm& zmm_src) {
  /* exp code */
  vcmpps(k_mask, zmm_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);
  vminps(zmm_src, zmm_src, table_val(exp_ln_flt_max_f));
  vmaxps(zmm_src, zmm_src, table_val(exp_ln_flt_min_f));
  vmovups(zmm_aux1, zmm_src);
  vmulps(zmm_src, zmm_src, table_val(exp_log2ef));
  vaddps(zmm_src, zmm_src, table_val(half));
  vrndscaleps(zmm_aux2, zmm_src, _op_floor & 0x3);

  // keep zmm_src = fx for further computations
  vmovups(zmm_src, zmm_aux2);

  // x = x - fx * ln2
  vfnmadd231ps(zmm_aux1, zmm_aux2, table_val(ln2f));

  // We do not count 2^n here, because n can reach 128 and 2^128 is not
  // representable by fp32, so to get around this problem, instead of computing
  // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
  // and 2 are numbers representable in fp32.

  // compute 2^(n-1)
  vsubps(zmm_src, zmm_src, table_val(one));
  vcvtps2dq(zmm_aux2, zmm_src);
  vpaddd(zmm_aux2, zmm_aux2, table_val(exponent_bias));
  vpslld(zmm_aux2, zmm_aux2, n_mantissa_bits);

  // use zmm_src as tmp zmm_zero when applying mask
  vxorps(zmm_src, zmm_src, zmm_src);

  // set zeroes at those points which were < log(FLT_MIN)
  vblendmps(zmm_aux2 | k_mask, zmm_aux2, zmm_src);

  // compute polynomial
  vmovups(zmm_src, table_val(exp_pol, 4));
  vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 3));
  vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 2));
  vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 1));
  vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 0));
  vfmadd213ps(zmm_src, zmm_aux1, table_val(one));

  // y = y * 2^n

  vmulps(zmm_src, zmm_src, zmm_aux2);
  vmulps(zmm_src, zmm_src, table_val(two));
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

  vcvtneps2bf16(ymm_bf16, reg_src);
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

  static const table_t gelu_tanh_const{{gelu_tanh_fitting_const, {0x3d372713, true}},
                                       {gelu_tanh_fitting_const_times_three, {0x3e095d4f, true}},
                                       {gelu_tanh_sqrt_two_over_pi, {0x3f4c422a, true}},
                                       {gelu_tanh_flt_max_x, {0x4154C480, true}},
                                       {gelu_tanh_flt_min_x, {0xC154C480, true}}};

  // tanh(x) constants for four interval approximation
  static const table_t tanh_consts{{tanh_idx_bias, {0x39800000, true}},
                                   {tanh_idx_mask, {0xffc00000, true}},
                                   {tanh_linear_ubound, {0x39ddb3d7, true}},
                                   {tanh_saturation_lbound, {0x41102cb3, true}}};

  // tanh(x) polynomial approximation
  // For each coefficient, there is 32 entries
  static const table_t tanh_polynomial_table{
      // coefficients of degree 0
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0x39bfffff, false}},
      {tanh_pol_table, {0x39ffffff, false}},
      {tanh_pol_table, {0x3a3ffffe, false}},
      {tanh_pol_table, {0x3a7ffffb, false}},
      {tanh_pol_table, {0x3abffff7, false}},
      {tanh_pol_table, {0x3affffeb, false}},
      {tanh_pol_table, {0x3b3fffdc, false}},
      {tanh_pol_table, {0x3b7fffab, false}},
      {tanh_pol_table, {0x3bbfff70, false}},
      {tanh_pol_table, {0x3bfffeab, false}},
      {tanh_pol_table, {0x3c3ffdc0, false}},
      {tanh_pol_table, {0x3c7ffaab, false}},
      {tanh_pol_table, {0x3cbff701, false}},
      {tanh_pol_table, {0x3cffeaad, false}},
      {tanh_pol_table, {0x3d3fdc08, false}},
      {tanh_pol_table, {0x3d7faacd, false}},
      {tanh_pol_table, {0x3dbf7081, false}},
      {tanh_pol_table, {0x3dfeacc9, false}},
      {tanh_pol_table, {0x3e3dc7fd, false}},
      {tanh_pol_table, {0x3e7acbf5, false}},
      {tanh_pol_table, {0x3eb77a9f, false}},
      {tanh_pol_table, {0x3eec9a9f, false}},
      {tanh_pol_table, {0x3f22991f, false}},
      {tanh_pol_table, {0x3f42f7d6, false}},
      {tanh_pol_table, {0x3f67b7cc, false}},
      {tanh_pol_table, {0x3f76ca83, false}},
      {tanh_pol_table, {0x3f7ebbe9, false}},
      {tanh_pol_table, {0x3f7fd40c, false}},
      {tanh_pol_table, {0x3f7fff32, false}},
      {tanh_pol_table, {0x3f7ffffc, false}},
      {tanh_pol_table, {0x3f800000, false}},
      // coefficients of degree 1
      {tanh_pol_table, {0x3f800000, false}},
      {tanh_pol_table, {0x3f800018, false}},
      {tanh_pol_table, {0x3f7fffe8, false}},
      {tanh_pol_table, {0x3f7fffda, false}},
      {tanh_pol_table, {0x3f7fffdc, false}},
      {tanh_pol_table, {0x3f7fffdc, false}},
      {tanh_pol_table, {0x3f7fffac, false}},
      {tanh_pol_table, {0x3f7fff70, false}},
      {tanh_pol_table, {0x3f7ffeec, false}},
      {tanh_pol_table, {0x3f7ffdc0, false}},
      {tanh_pol_table, {0x3f7ffbed, false}},
      {tanh_pol_table, {0x3f7ff704, false}},
      {tanh_pol_table, {0x3f7feff5, false}},
      {tanh_pol_table, {0x3f7fdbca, false}},
      {tanh_pol_table, {0x3f7fbfff, false}},
      {tanh_pol_table, {0x3f7f7041, false}},
      {tanh_pol_table, {0x3f7f009b, false}},
      {tanh_pol_table, {0x3f7dc36c, false}},
      {tanh_pol_table, {0x3f7c0aa8, false}},
      {tanh_pol_table, {0x3f7734b8, false}},
      {tanh_pol_table, {0x3f70a4de, false}},
      {tanh_pol_table, {0x3f5f1fd8, false}},
      {tanh_pol_table, {0x3f495493, false}},
      {tanh_pol_table, {0x3f18b9ec, false}},
      {tanh_pol_table, {0x3ed706cb, false}},
      {tanh_pol_table, {0x3e390b06, false}},
      {tanh_pol_table, {0x3d90b11f, false}},
      {tanh_pol_table, {0x3c21a053, false}},
      {tanh_pol_table, {0x3aaf7fdb, false}},
      {tanh_pol_table, {0x37ccc1a3, false}},
      {tanh_pol_table, {0x355c6733, false}},
      {tanh_pol_table, {0x00000000, false}},
      // coefficients of degree 2
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0xbe4e0ff1, false}},
      {tanh_pol_table, {0x3d25b1b1, false}},
      {tanh_pol_table, {0x3d6b6dab, false}},
      {tanh_pol_table, {0x3c9fb1d5, false}},
      {tanh_pol_table, {0xbabff06f, false}},
      {tanh_pol_table, {0x3c07b3f6, false}},
      {tanh_pol_table, {0xbb3fc1bc, false}},
      {tanh_pol_table, {0x3a9f5921, false}},
      {tanh_pol_table, {0xbbbf06f2, false}},
      {tanh_pol_table, {0xbbb0f402, false}},
      {tanh_pol_table, {0xbc47db9e, false}},
      {tanh_pol_table, {0xbc73d5e7, false}},
      {tanh_pol_table, {0xbca25bda, false}},
      {tanh_pol_table, {0xbcfca780, false}},
      {tanh_pol_table, {0xbd40e07c, false}},
      {tanh_pol_table, {0xbd7dab03, false}},
      {tanh_pol_table, {0xbdbe4a0f, false}},
      {tanh_pol_table, {0xbdfb14a5, false}},
      {tanh_pol_table, {0xbe36cc8d, false}},
      {tanh_pol_table, {0xbe6bd102, false}},
      {tanh_pol_table, {0xbe9fe7c5, false}},
      {tanh_pol_table, {0xbeba0f10, false}},
      {tanh_pol_table, {0xbec206a8, false}},
      {tanh_pol_table, {0xbea3c388, false}},
      {tanh_pol_table, {0xbe277d62, false}},
      {tanh_pol_table, {0xbd8b7960, false}},
      {tanh_pol_table, {0xbc209f49, false}},
      {tanh_pol_table, {0xbaad44ca, false}},
      {tanh_pol_table, {0xb7c6eeac, false}},
      {tanh_pol_table, {0xb663aa41, false}},
      {tanh_pol_table, {0x00000000, false}},
      // coefficients of degree 3
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0x45b3ae96, false}},
      {tanh_pol_table, {0xc414eb20, false}},
      {tanh_pol_table, {0xc450e02e, false}},
      {tanh_pol_table, {0xc3152b4e, false}},
      {tanh_pol_table, {0xbead2f56, false}},
      {tanh_pol_table, {0xc2162e02, false}},
      {tanh_pol_table, {0xbeb4bd5a, false}},
      {tanh_pol_table, {0xc11a59a4, false}},
      {tanh_pol_table, {0xbed2f507, false}},
      {tanh_pol_table, {0xc020d32c, false}},
      {tanh_pol_table, {0x3dd0f506, false}},
      {tanh_pol_table, {0xbf2a75e2, false}},
      {tanh_pol_table, {0xbff950e3, false}},
      {tanh_pol_table, {0xbed47334, false}},
      {tanh_pol_table, {0xbe809b8c, false}},
      {tanh_pol_table, {0xbeb64532, false}},
      {tanh_pol_table, {0xbe961a5b, false}},
      {tanh_pol_table, {0xbe9b63ac, false}},
      {tanh_pol_table, {0xbea0d4b2, false}},
      {tanh_pol_table, {0xbe828a77, false}},
      {tanh_pol_table, {0xbe378612, false}},
      {tanh_pol_table, {0xbdc20908, false}},
      {tanh_pol_table, {0x3d2d3957, false}},
      {tanh_pol_table, {0x3dd46e89, false}},
      {tanh_pol_table, {0x3db3f629, false}},
      {tanh_pol_table, {0x3d2c5e7b, false}},
      {tanh_pol_table, {0x3bd20403, false}},
      {tanh_pol_table, {0x3a59dfae, false}},
      {tanh_pol_table, {0x3770af45, false}},
      {tanh_pol_table, {0x372cc014, false}},
      {tanh_pol_table, {0x00000000, false}},
      // coefficients of degree 4
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0xcc981a1b, false}},
      {tanh_pol_table, {0x4a7edd3d, false}},
      {tanh_pol_table, {0x4ab1007c, false}},
      {tanh_pol_table, {0x48fedd9c, false}},
      {tanh_pol_table, {0x41a557b5, false}},
      {tanh_pol_table, {0x477ee32a, false}},
      {tanh_pol_table, {0x422557f5, false}},
      {tanh_pol_table, {0x45ff3ce4, false}},
      {tanh_pol_table, {0x42a55641, false}},
      {tanh_pol_table, {0x446e0867, false}},
      {tanh_pol_table, {0xc33dc19a, false}},
      {tanh_pol_table, {0x42915214, false}},
      {tanh_pol_table, {0x43af4fad, false}},
      {tanh_pol_table, {0x4110fe88, false}},
      {tanh_pol_table, {0xc1099b75, false}},
      {tanh_pol_table, {0x3fc8a8dc, false}},
      {tanh_pol_table, {0xbfbeaef5, false}},
      {tanh_pol_table, {0xbe365aad, false}},
      {tanh_pol_table, {0x3f4d9652, false}},
      {tanh_pol_table, {0x3ddfa08f, false}},
      {tanh_pol_table, {0x3e34e9b8, false}},
      {tanh_pol_table, {0x3e2d07a6, false}},
      {tanh_pol_table, {0x3dc63567, false}},
      {tanh_pol_table, {0x3cdaeb78, false}},
      {tanh_pol_table, {0xbcd17537, false}},
      {tanh_pol_table, {0xbc92829c, false}},
      {tanh_pol_table, {0xbb43ab99, false}},
      {tanh_pol_table, {0xb9b471dd, false}},
      {tanh_pol_table, {0xb6baad5a, false}},
      {tanh_pol_table, {0xb78bafc7, false}},
      {tanh_pol_table, {0x00000000, false}},
      // coefficients of degree 5
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0x52f688d5, false}},
      {tanh_pol_table, {0xd0505c72, false}},
      {tanh_pol_table, {0xd08f98e3, false}},
      {tanh_pol_table, {0xce505cc9, false}},
      {tanh_pol_table, {0xc7162b8a, false}},
      {tanh_pol_table, {0xcc5061d6, false}},
      {tanh_pol_table, {0xc7162bdf, false}},
      {tanh_pol_table, {0xca50b37f, false}},
      {tanh_pol_table, {0xc7162a3a, false}},
      {tanh_pol_table, {0xc8422086, false}},
      {tanh_pol_table, {0x471a714e, false}},
      {tanh_pol_table, {0xc5ece1f1, false}},
      {tanh_pol_table, {0xc70e3d90, false}},
      {tanh_pol_table, {0xc3eba94a, false}},
      {tanh_pol_table, {0x43e0c424, false}},
      {tanh_pol_table, {0xc21f4552, false}},
      {tanh_pol_table, {0x42217cc8, false}},
      {tanh_pol_table, {0x405e7dc4, false}},
      {tanh_pol_table, {0xc10dd401, false}},
      {tanh_pol_table, {0x3e96b602, false}},
      {tanh_pol_table, {0xbd1a6d2f, false}},
      {tanh_pol_table, {0xbd393883, false}},
      {tanh_pol_table, {0xbd674682, false}},
      {tanh_pol_table, {0xbd310016, false}},
      {tanh_pol_table, {0xb961e269, false}},
      {tanh_pol_table, {0x3ba32495, false}},
      {tanh_pol_table, {0x3a7680d5, false}},
      {tanh_pol_table, {0x38b3173c, false}},
      {tanh_pol_table, {0x35a9deea, false}},
      {tanh_pol_table, {0x375c3f2a, false}},
      {tanh_pol_table, {0x00000000, false}},
      // coefficients of degree 6
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0xd8995ed1, false}},
      {tanh_pol_table, {0x558285ea, false}},
      {tanh_pol_table, {0x55b2cd69, false}},
      {tanh_pol_table, {0x53028625, false}},
      {tanh_pol_table, {0x4bc9991f, false}},
      {tanh_pol_table, {0x5082898a, false}},
      {tanh_pol_table, {0x4b4999b3, false}},
      {tanh_pol_table, {0x4e02c07c, false}},
      {tanh_pol_table, {0x4ac99764, false}},
      {tanh_pol_table, {0x4b72c822, false}},
      {tanh_pol_table, {0xca40c0e1, false}},
      {tanh_pol_table, {0x489413e4, false}},
      {tanh_pol_table, {0x49b12224, false}},
      {tanh_pol_table, {0x46134c4e, false}},
      {tanh_pol_table, {0xc60c2d57, false}},
      {tanh_pol_table, {0x43c83910, false}},
      {tanh_pol_table, {0xc3c872d1, false}},
      {tanh_pol_table, {0xc186bc9e, false}},
      {tanh_pol_table, {0x42325bc3, false}},
      {tanh_pol_table, {0xbf2ffa4a, false}},
      {tanh_pol_table, {0x3d9a203c, false}},
      {tanh_pol_table, {0xbc545a43, false}},
      {tanh_pol_table, {0xbae08fee, false}},
      {tanh_pol_table, {0x3c80225d, false}},
      {tanh_pol_table, {0x3b1fd1df, false}},
      {tanh_pol_table, {0xba36b9d1, false}},
      {tanh_pol_table, {0xb91de544, false}},
      {tanh_pol_table, {0xb71f100f, false}},
      {tanh_pol_table, {0xb408e2ed, false}},
      {tanh_pol_table, {0xb685fec8, false}},
      {tanh_pol_table, {0x00000000, false}},
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
        case ssd::post_op_scheme::gelu:
          gelu_ = true;
          tanh_ = true;
          exp_ = true;
        default:
          break;
      }
    }

    bool exp_ = false;
    bool tanh_ = false;
    bool gelu_ = false;
    bool exp() const { return exp_; }
    bool tanh() const { return tanh_; }
    bool gelu() const { return gelu_; }
  };

  need_t need(param_.scheme);

  push_entries_of(common_values);

  if (need.exp()) {
    push_entries_of(exp_consts);
    push_entries_of(exp_polynomial);
  }

  if (need.tanh()) {
    push_entries_of(tanh_consts);
    push_entries_of(tanh_polynomial_table);
  }
  if (need.gelu()) push_entries_of(gelu_tanh_const);

  size_t off = 0;
  for (auto it = entry_map.begin(); it != entry_map.end(); it++) {
    auto& te = (*it).second;
    te.off = off;
    off += te.bcast ? 64u : sizeof(table_entry_val_t);
  }
}
}  // namespace jd