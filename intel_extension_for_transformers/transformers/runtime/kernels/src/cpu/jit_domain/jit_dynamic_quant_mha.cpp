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

#include "jit_dynamic_quant_mha.hpp"

#include <memory>
#include <functional>
#include <cmath>

namespace jd {

/* jit_mmsoftmax_batch_amx_s8_ab_BA16b4a_u8_16x */
#define PARAM_OFF(field) offsetof(rt_data_t, field)
void jit_mmsoftmax_batch_amx_s8_ab_BA16b4a_u8_16x::mm_exp_sum(  //
    regs_pool* const rp, const Reg64& reg_tmpf32, const std::array<Zmm, 16UL> zmm_expsum) {
  const std::array<Tmm, 2> tmm_dst{tmm0, tmm1};
  const auto reg_nbsize = rp->reg<Reg64>();  // number of blocks of size 16
  const auto reg_nbiter = rp->reg<Reg64>();
  const auto reg_64ULL = rp->reg<Reg64>();
  mov(reg_64ULL, 64ULL);

  vxorps(zmm_expsum[0], zmm_expsum[0], zmm_expsum[0]);
  for (int i = 1; i < 16; ++i) vmovdqa32(zmm_expsum[i], zmm_expsum[0]);

  const auto runtime_kloop = K < 0 || K > 64;
  const std::array<Tmm, 1> tmm_src0{tmm4};
  if (!runtime_kloop) {
    const auto reg_src0 = rp->reg<Reg64>();
    const auto reg_ld_src0 = rp->reg<Reg64>();
    mov(reg_src0, qword[rp->p[0] + PARAM_OFF(src0)]);
    mov(reg_ld_src0.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_src0)]);
    tileloaddt1(tmm_src0[0], ptr[reg_src0 + reg_ld_src0]);
  }

  const auto mm_exp_sum_16xkx16tw = [&](int tw) {
    {  // k loop part
      const std::array<Tmm, 2> tmm_src1{tmm6, tmm7};
      const auto reg_ld_src0 = rp->reg<Reg64>();
      const auto reg_ld_src1 = rp->reg<Reg64>();
      const auto reg_src0 = rp->reg<Reg64>();
      const auto reg_src1 = rp->reg<Reg64>();
      const auto reg_kiter = rp->reg<Reg64>();
      if (runtime_kloop) mov(reg_ld_src0.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_src0)]);
      mov(reg_ld_src1.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_src1)]);
      if (runtime_kloop) mov(reg_src0, qword[rp->p[0] + PARAM_OFF(src0)]);
      mov(reg_src1, reg_ld_src1);
      imul(reg_src1, reg_nbiter);
      add(reg_src1, qword[rp->p[0] + PARAM_OFF(src1)]);  // reg_src1 = src1 + reg_ld_src1 * reg_nbiter
      if (runtime_kloop) xor_(reg_kiter.cvt32(), reg_kiter.cvt32());
      // clear tmm
      for (int j = 0; j < tw; ++j) tilezero(tmm_dst[j]);
      Xbyak::Label l_kloop;
      L(l_kloop);
      {  // k loop body
        for (int j = 0; j < tw; ++j) {
          // load src0
          if (runtime_kloop && j == 0) tileloadd(tmm_src0[0], ptr[reg_src0 + reg_ld_src0]);

          // load src1
          if (j == 0) {
            tileloadd(tmm_src1[j], ptr[reg_src1 + reg_64ULL]);
          } else {
            const auto reg_curr_src1 = rp->reg<Reg64>();
            lea(reg_curr_src1, ptr[reg_src1 + j * reg_ld_src1]);
            tileloadd(tmm_src1[j], ptr[reg_curr_src1 + reg_64ULL]);
          }

          // dp
          tdpbssd(tmm_dst[j], tmm_src0[0], tmm_src1[j]);
        }
      }
      if (runtime_kloop) {
        lea(reg_src0, ptr[reg_src0 + 64]);
        lea(reg_src1, ptr[reg_src1 + BYTES_TMM]);
        lea(reg_kiter, ptr[reg_kiter + 64]);
        cmp(reg_kiter.cvt32(), dword[rp->p[0] + PARAM_OFF(K)]);
        jb(l_kloop);
      }
    }

    {  // tile epilogue: exp & sum & store
      const auto reg_src0scale = rp->reg<Reg64>();
      const auto reg_src1scale = rp->reg<Reg64>();
      const auto reg_bias = rp->reg<Reg64>();
      const auto vreg_src0scale = rp->reg<Zmm>();
      const auto vreg_src1scale = rp->regs<Zmm, 2>();
      const auto vreg_bias = rp->regs<Zmm, 2>();
      const auto vreg_c = rp->regs<Zmm, 3>();
      const auto zmm_x = rp->reg<Zmm>();
      const auto zmm_scale = rp->reg<Zmm>();
      vpbroadcastd(zmm_scale, dword[rp->p[0] + PARAM_OFF(scale)]);
      mov(reg_src0scale, qword[rp->p[0] + PARAM_OFF(scale_src0)]);
      imul(reg_src1scale, reg_nbiter, BYTES_ZMM);
      if (has_bias) mov(reg_bias, reg_src1scale);
      add(reg_src1scale, qword[rp->p[0] + PARAM_OFF(scale_src1)]);
      if (has_bias) add(reg_bias, qword[rp->p[0] + PARAM_OFF(src_bias)]);
      vpbroadcastd(vreg_c[0], dword[rip + l_poly_c[0]]);
      vpbroadcastd(vreg_c[1], dword[rip + l_poly_c[1]]);
      vpbroadcastd(vreg_c[2], dword[rip + l_poly_c[2]]);
      for (int ii = 0; ii < 16; ++ii) {
        for (int j = 0; j < tw; ++j) {
          if (ii == 0) tilestored(ptr[reg_tmpf32 + reg_64ULL + j * BYTES_TMM], tmm_dst[j]);
          if (j == 0) vmulps(vreg_src0scale, zmm_scale, zword_b[reg_src0scale + ii * sizeof(float)]);
          if (ii == 0) vmovups(vreg_src1scale[j], zword[reg_src1scale + j * BYTES_ZMM]);
          if (ii == 0 && has_bias) vmovups(vreg_bias[j], zword[reg_bias + j * BYTES_ZMM]);

          const auto tmp_dst_addr = ptr[reg_tmpf32 + j * BYTES_TMM + ii * BYTES_ZMM];
          vcvtdq2ps(zmm_x, tmp_dst_addr);
          vmulps(zmm_x, zmm_x, vreg_src0scale);
          (has_bias)  //
              ? vfmadd213ps(zmm_x, vreg_src1scale[j], vreg_bias[j])
              : vmulps(zmm_x, zmm_x, vreg_src1scale[j]);
          exp_approx_f32(zmm_x, zmm_x, zword_b[rip + l_log2ef], zword_b[rip + l_ln2], vreg_c, rp->regs<Zmm, 2>());
          vmovaps(tmp_dst_addr, zmm_x);
          vaddps(zmm_expsum[ii], zmm_expsum[ii], zmm_x);
        }
      }
    }
  };

  // n loop
  mov(reg_nbsize.cvt32(), dword[rp->p[0] + PARAM_OFF(N)]);
  add(reg_nbsize, 15);
  shr(reg_nbsize, 4);  // nbsize = ceil_div(N, 16)
  xor_(reg_nbiter, reg_nbiter);
  Xbyak::Label l_nloop_mm;
  L(l_nloop_mm);
  mm_exp_sum_16xkx16tw(TW_);  // n loop body
  lea(reg_tmpf32, ptr[reg_tmpf32 + TW_ * BYTES_TMM]);
  lea(reg_nbiter, ptr[reg_nbiter + TW_]);
  {  // scope tmp
    const auto tmp = rp->reg<Reg64>();
    lea(tmp, ptr[reg_nbiter + 1]);  // tmp = n + 1
    cmp(tmp, reg_nbsize);           // jmp if n + 1 < N
  }
  jb(l_nloop_mm);

  // N Tail
  static_assert(TW_ == 2, "Only implement tail for TW_ == 2");
  Xbyak::Label l_end;
  cmp(reg_nbiter, reg_nbsize);
  je(l_end, T_NEAR);
  mm_exp_sum_16xkx16tw(1);  // tail processing

  L(l_end);
}

void jit_mmsoftmax_batch_amx_s8_ab_BA16b4a_u8_16x::quant255_store(regs_pool* const rp, const Xbyak::Reg64& reg_tmpf32,
                                                                  const std::array<Xbyak::Zmm, 16UL>& zmm_expscale) {
  const auto zmm_tmp = rp->reg<Zmm>();
  const auto reg_dst = rp->reg<Reg64>();
  const auto reg_nbsize = rp->reg<Reg64>();
  const auto reg_nbiter = rp->reg<Reg64>();
  const auto reg_tmp_ld_dst = rp->reg<Reg64>();
  mov(reg_dst, ptr[rp->p[0] + PARAM_OFF(dst)]);
  mov(reg_nbsize.cvt32(), dword[rp->p[0] + PARAM_OFF(N)]);
  add(reg_nbsize, 15);
  shr(reg_nbsize, 4);  // nbsize = ceil_div(N, 16)
  xor_(reg_nbiter.cvt32(), reg_nbiter.cvt32());
  mov(reg_tmp_ld_dst.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_dst)]);
  Xbyak::Label l_nloop_mm;
  L(l_nloop_mm);
  {  // n loop body
    const auto reg_dst_off = rp->reg<Reg64>();
    xor_(reg_dst_off.cvt32(), reg_dst_off.cvt32());
    for (int ii = 0; ii < 16; ++ii) {
      const auto& zmm_x = zmm_tmp;
      vmulps(zmm_x, zmm_expscale[ii], zword[reg_tmpf32 + ii * BYTES_ZMM]);  // quant
      vcvtps2udq(zmm_x, zmm_x);
      vpmovusdb(ptr[reg_dst + reg_dst_off], zmm_x);
      if (ii != 15) lea(reg_dst_off, ptr[reg_dst_off + reg_tmp_ld_dst]);
    }
  }
  lea(reg_tmpf32, ptr[reg_tmpf32 + BYTES_TMM]);
  lea(reg_dst, ptr[reg_dst + VEC * sizeof(uint8_t)]);
  lea(reg_nbiter, ptr[reg_nbiter + 1]);
  cmp(reg_nbiter, reg_nbsize);  // jmp if n + 1 < N
  jb(l_nloop_mm);
}

void jit_mmsoftmax_batch_amx_s8_ab_BA16b4a_u8_16x::generate() {
  bool need_cfg_amx = pre_amx_cfg_ != nullptr && *pre_amx_cfg_ != required_amx_cfg_;

  std::shared_ptr<void> use_loacl_label = {(inLocalLabel(), nullptr), [&](...) { outLocalLabel(); }};
  {
    constexpr auto tmp_mem_size = BYTES_ZMM;
    const auto stack_size = (need_cfg_amx ? sizeof(tileconfig_t) : 0) + tmp_mem_size;
    regs_pool rp(this, 1, {12, 32, 0}, stack_size, regs_pool::DefaultFlags, 64);
    std::shared_ptr<void> local_cfg;
    if (need_cfg_amx) {  // create a local amx config environment
      sttilecfg(ptr[rsp]);
      ldtilecfg(ptr[rip + L_amx_cfg]);
      lea(rsp, ptr[rsp + sizeof(tileconfig_t)]);
      local_cfg = {nullptr, [&](...) { (lea(rsp, ptr[rsp - sizeof(tileconfig_t)]), ldtilecfg(ptr[rsp])); }};
    }

    const auto reg_stacksize = rp.reg<Reg64>();
    const auto reg_bleft = rp.reg<Reg64>();

    // stacksize = (N + 16) * M * sizeof(float) + 64  // 16 as enough padding; 64 for tmp memory
    mov(reg_stacksize.cvt32(), dword[rp.p[0] + PARAM_OFF(N)]);
    imul(reg_stacksize, reg_stacksize, M * sizeof(float));                 // (N) * M * sizeof(float)
    lea(reg_stacksize, ptr[reg_stacksize + 16 * M * sizeof(float) + 64]);  // (+16) * M * sizeof(float)
    std::shared_ptr<void> use_dyn_stack = {(sub(rsp, reg_stacksize), nullptr), [&](...) { add(rsp, reg_stacksize); }};
    const auto m512_tmp = rsp - 64;

    mov(reg_bleft.cvt32(), dword[rp.p[0] + PARAM_OFF(batch_size)]);
    Xbyak::Label l_bloop;
    L(l_bloop);
    {
      const auto reg_tmp = rp.reg<Reg64>();
      mov(reg_tmp, rsp);
      const auto zmm_expsum = rp.regs<Zmm, 16>();
      mm_exp_sum(&rp, reg_tmp, zmm_expsum);
      transpose_16x16_ps(zmm_expsum, rp.regs<Zmm, 16>());  // 4 inst per zmm
      reduce_vmms(zmm_expsum, &CodeGenerator::vaddps);

      const auto zmm_tmp = rp.reg<Zmm>();
      vpbroadcastd(zmm_tmp, dword[rip + l_255]);
      vdivps(zmm_tmp, zmm_tmp, zmm_expsum[0]);  // 255 / sum
      vmovdqa32(zword[m512_tmp], zmm_tmp);
      const auto& zmm_expscale = zmm_expsum;
      for (int ii = 0; ii < 16; ++ii) vpbroadcastd(zmm_expscale[ii], dword[m512_tmp + ii * sizeof(float)]);

      mov(reg_tmp, rsp);
      quant255_store(&rp, reg_tmp, zmm_expscale);
    }
    const auto r64_tmp = rp.reg<Reg64>();
    mov(r64_tmp, qword[rp.p[0] + PARAM_OFF(batchstep_src0)]);
    add(r64_tmp, qword[rp.p[0] + PARAM_OFF(src0)]);
    mov(qword[rp.p[0] + PARAM_OFF(src0)], r64_tmp);
    mov(r64_tmp, qword[rp.p[0] + PARAM_OFF(batchstep_src0scale)]);
    add(r64_tmp, qword[rp.p[0] + PARAM_OFF(scale_src0)]);
    mov(qword[rp.p[0] + PARAM_OFF(scale_src0)], r64_tmp);
    mov(r64_tmp, qword[rp.p[0] + PARAM_OFF(batchstep_src1)]);
    add(r64_tmp, qword[rp.p[0] + PARAM_OFF(src1)]);
    mov(qword[rp.p[0] + PARAM_OFF(src1)], r64_tmp);
    mov(r64_tmp, qword[rp.p[0] + PARAM_OFF(batchstep_src1scale)]);
    add(r64_tmp, qword[rp.p[0] + PARAM_OFF(scale_src1)]);
    mov(qword[rp.p[0] + PARAM_OFF(scale_src1)], r64_tmp);
    mov(r64_tmp, qword[rp.p[0] + PARAM_OFF(batchstep_dst)]);
    add(r64_tmp, qword[rp.p[0] + PARAM_OFF(dst)]);
    mov(qword[rp.p[0] + PARAM_OFF(dst)], r64_tmp);
    sub(reg_bleft, 1);
    jg(l_bloop);
  }  // end of call stack

  // .data
  if (need_cfg_amx) {
    configure_tiles(required_amx_cfg_, &reqired_tile_cfg_);
    align(sizeof(tileconfig_t));
    L(L_amx_cfg);
    db(reinterpret_cast<const uint8_t*>(&reqired_tile_cfg_), sizeof(tileconfig_t));
  }

  align(sizeof(int32_t));
  L(l_log2ef);
  db(bit_cast<uint32_t>(std::log2f(std::exp(1.f))), sizeof(float));
  L(l_ln2);
  db(bit_cast<uint32_t>(std::log(2.f)), sizeof(float));
  L(l_halff);
  db(bit_cast<uint32_t>(.5f), sizeof(float));
  L(l_255);
  db(bit_cast<uint32_t>(255.f), sizeof(float));

  L(l_poly_c[0]);
  db(bit_cast<uint32_t>(exp_approx_f32_coeff[0]), sizeof(float));
  L(l_poly_c[1]);
  db(bit_cast<uint32_t>(exp_approx_f32_coeff[1]), sizeof(float));
  L(l_poly_c[2]);
  db(bit_cast<uint32_t>(exp_approx_f32_coeff[2]), sizeof(float));
}
#undef PARAM_OFF

/* jit_mmexp_amx_s8_ab_BA16b4a_u8_16x */
#define PARAM_OFF(field) offsetof(rt_data_t, field)
void jit_mmexp_amx_s8_ab_BA16b4a_u8_16x::mm_exp_sum(  //
    regs_pool* const rp, const std::array<Zmm, 16UL>& zmm_expsum, const Xbyak::RegExp& addr_expmax) {
  const std::array<Tmm, 2> tmm_dst{tmm0, tmm1};
  const auto reg_nbsize = rp->reg<Reg64>();  // number of blocks of size 16
  const auto reg_nbiter = rp->reg<Reg64>();
  const auto reg_64ULL = rp->reg<Reg64>();
  mov(reg_64ULL, 64ULL);
  const auto reg_dst = rp->reg<Reg64>();
  mov(reg_dst, qword[rp->p[0] + PARAM_OFF(dst)]);

  vxorps(zmm_expsum[0], zmm_expsum[0], zmm_expsum[0]);
  for (int i = 1; i < 16; ++i) vmovdqa32(zmm_expsum[i], zmm_expsum[0]);

  const auto runtime_kloop = K < 0 || K > 64;
  const std::array<Tmm, 1> tmm_src0{tmm4};
  if (!runtime_kloop) {
    const auto reg_src0 = rp->reg<Reg64>();
    const auto reg_ld_src0 = rp->reg<Reg64>();
    mov(reg_src0, qword[rp->p[0] + PARAM_OFF(src0)]);
    mov(reg_ld_src0.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_src0)]);
    tileloaddt1(tmm_src0[0], ptr[reg_src0 + reg_ld_src0]);
  }

  const auto mm_exp_sum_16xkx16tw = [&](int tw) {
    {  // k loop part
      const std::array<Tmm, 2> tmm_src1{tmm6, tmm7};
      const auto reg_ld_src0 = rp->reg<Reg64>();
      const auto reg_ld_src1 = rp->reg<Reg64>();
      const auto reg_src0 = rp->reg<Reg64>();
      const auto reg_src1 = rp->reg<Reg64>();
      const auto reg_kiter = rp->reg<Reg64>();
      if (runtime_kloop) mov(reg_ld_src0.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_src0)]);
      mov(reg_ld_src1.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_src1)]);
      if (runtime_kloop) mov(reg_src0, qword[rp->p[0] + PARAM_OFF(src0)]);
      mov(reg_src1, reg_ld_src1);
      imul(reg_src1, reg_nbiter);
      add(reg_src1, qword[rp->p[0] + PARAM_OFF(src1)]);  // reg_src1 = src1 + reg_ld_src1 * reg_nbiter
      if (runtime_kloop) xor_(reg_kiter.cvt32(), reg_kiter.cvt32());
      // clear tmm
      for (int j = 0; j < tw; ++j) tilezero(tmm_dst[j]);
      Xbyak::Label l_kloop;
      L(l_kloop);
      {  // k loop body
        for (int j = 0; j < tw; ++j) {
          // load src0
          if (runtime_kloop && j == 0) tileloadd(tmm_src0[0], ptr[reg_src0 + reg_ld_src0]);

          // load src1
          if (j == 0) {
            tileloadd(tmm_src1[j], ptr[reg_src1 + reg_64ULL]);
          } else {
            const auto reg_curr_src1 = rp->reg<Reg64>();
            lea(reg_curr_src1, ptr[reg_src1 + j * reg_ld_src1]);
            tileloadd(tmm_src1[j], ptr[reg_curr_src1 + reg_64ULL]);
          }

          // dp
          tdpbssd(tmm_dst[j], tmm_src0[0], tmm_src1[j]);
        }
      }
      if (runtime_kloop) {
        lea(reg_src0, ptr[reg_src0 + 64]);
        lea(reg_src1, ptr[reg_src1 + BYTES_TMM]);
        lea(reg_kiter, ptr[reg_kiter + 64]);
        cmp(reg_kiter.cvt32(), dword[rp->p[0] + PARAM_OFF(K)]);
        jb(l_kloop);
      }
    }

    {  // tile epilogue: exp & sum & store
      const auto reg_src0scale = rp->reg<Reg64>();
      const auto reg_src1scale = rp->reg<Reg64>();
      const auto reg_bias = rp->reg<Reg64>();
      const auto vreg_src0scale = rp->reg<Zmm>();
      const auto vreg_src1scale = rp->regs<Zmm, 2>();
      const auto vreg_bias = rp->regs<Zmm, 2>();
      const auto vreg_c = rp->regs<Zmm, 3>();
      const auto zmm_x = rp->reg<Zmm>();
      const auto zmm_max = rp->reg<Zmm>();
      const auto zmm_scale = rp->reg<Zmm>();
      vpbroadcastd(zmm_scale, dword[rp->p[0] + PARAM_OFF(scale)]);
      mov(reg_src0scale, qword[rp->p[0] + PARAM_OFF(scale_src0)]);
      imul(reg_src1scale, reg_nbiter, BYTES_ZMM);
      if (has_bias) mov(reg_bias, reg_src1scale);
      add(reg_src1scale, qword[rp->p[0] + PARAM_OFF(scale_src1)]);
      if (has_bias) add(reg_bias, qword[rp->p[0] + PARAM_OFF(src_bias)]);
      vpbroadcastd(vreg_c[0], dword[rip + l_poly_c[0]]);
      vpbroadcastd(vreg_c[1], dword[rip + l_poly_c[1]]);
      vpbroadcastd(vreg_c[2], dword[rip + l_poly_c[2]]);
      for (int ii = 0; ii < 16; ++ii) {
        for (int j = 0; j < tw; ++j) {
          if (ii == 0) tilestored(ptr[reg_dst + reg_64ULL + j * BYTES_TMM], tmm_dst[j]);
          if (j == 0) vmulps(vreg_src0scale, zmm_scale, zword_b[reg_src0scale + ii * sizeof(float)]);
          if (ii == 0) vmovups(vreg_src1scale[j], zword[reg_src1scale + j * BYTES_ZMM]);
          if (ii == 0 && has_bias) vmovups(vreg_bias[j], zword[reg_bias + j * BYTES_ZMM]);

          const auto curr_dst_addr = ptr[reg_dst + j * BYTES_TMM + ii * BYTES_ZMM];
          vcvtdq2ps(zmm_x, curr_dst_addr);
          vmulps(zmm_x, zmm_x, vreg_src0scale);
          (has_bias)  //
              ? vfmadd213ps(zmm_x, vreg_src1scale[j], vreg_bias[j])
              : vmulps(zmm_x, zmm_x, vreg_src1scale[j]);
          exp_approx_f32(zmm_x, zmm_x, zword_b[rip + l_log2ef], zword_b[rip + l_ln2], vreg_c, rp->regs<Zmm, 2>());
          vmovaps(curr_dst_addr, zmm_x);
          vmaxps(zmm_max, zmm_x, zword[addr_expmax + ii * BYTES_ZMM]);
          vmovaps(zword[addr_expmax + ii * BYTES_ZMM], zmm_max);
          vaddps(zmm_expsum[ii], zmm_expsum[ii], zmm_x);
        }
      }
    }
  };

  // n loop
  mov(reg_nbsize.cvt32(), dword[rp->p[0] + PARAM_OFF(N)]);
  add(reg_nbsize, 15);
  shr(reg_nbsize, 4);  // nbsize = ceil_div(N, 16)
  xor_(reg_nbiter, reg_nbiter);
  Xbyak::Label l_nloop_mm;
  L(l_nloop_mm);
  mm_exp_sum_16xkx16tw(TW_);  // n loop body
  lea(reg_dst, ptr[reg_dst + TW_ * BYTES_TMM]);
  lea(reg_nbiter, ptr[reg_nbiter + TW_]);
  {  // scope tmp
    const auto tmp = rp->reg<Reg64>();
    lea(tmp, ptr[reg_nbiter + 1]);  // tmp = n + 1
    cmp(tmp, reg_nbsize);           // jmp if n + 1 < N
  }
  jb(l_nloop_mm);

  // N Tail
  static_assert(TW_ == 2, "Only implement tail for TW_ == 2");
  Xbyak::Label l_end;
  cmp(reg_nbiter, reg_nbsize);
  je(l_end, T_NEAR);
  mm_exp_sum_16xkx16tw(1);  // tail processing

  L(l_end);
}

void jit_mmexp_amx_s8_ab_BA16b4a_u8_16x::generate() {
  bool need_cfg_amx = pre_amx_cfg_ != nullptr && *pre_amx_cfg_ != required_amx_cfg_;

  std::shared_ptr<void> use_loacl_label = {(inLocalLabel(), nullptr), [&](...) { outLocalLabel(); }};
  {
    constexpr auto tmp_mem_size = 16 * BYTES_ZMM;  // for addr_max
    const auto stack_size = (need_cfg_amx ? sizeof(tileconfig_t) : 0) + tmp_mem_size;
    regs_pool rp(this, 1, {10, 32, 0}, stack_size, regs_pool::DefaultFlags, 64);
    std::shared_ptr<void> local_cfg;
    if (need_cfg_amx) {  // create a local amx config environment
      sttilecfg(ptr[rsp]);
      ldtilecfg(ptr[rip + L_amx_cfg]);
      lea(rsp, ptr[rsp + sizeof(tileconfig_t)]);
      local_cfg = {nullptr, [&](...) { (lea(rsp, ptr[rsp - sizeof(tileconfig_t)]), ldtilecfg(ptr[rsp])); }};
    }

    const Xbyak::RegExp addr_max = rsp;
    {  // init max exp; 0 should be small enough as the range of exp greater than 0
      const auto vreg_zero = rp.reg<Zmm>();
      vxorps(vreg_zero, vreg_zero, vreg_zero);
      for (int i = 0; i < 16; ++i) vmovaps(zword[addr_max + i * BYTES_ZMM], vreg_zero);
    }

    {
      const auto zmm_expsum = rp.regs<Zmm, 16>();
      mm_exp_sum(&rp, zmm_expsum, addr_max);
      transpose_16x16_ps(zmm_expsum, rp.regs<Zmm, 16>());  // 4 inst per zmm
      reduce_vmms(zmm_expsum, &CodeGenerator::vaddps);

      const auto reg_tmp = rp.reg<Reg64>();
      mov(reg_tmp, qword[rp.p[0] + PARAM_OFF(dst_sum)]);
      vmovdqa32(zword[reg_tmp], zmm_expsum[0]);
    }
    {
      const auto zmm_expmax = rp.regs<Zmm, 16>();
      for (int i = 0; i < 16; ++i) vmovaps(zmm_expmax[i], zword[addr_max + i * BYTES_ZMM]);
      transpose_16x16_ps(zmm_expmax, rp.regs<Zmm, 16>());
      reduce_vmms(zmm_expmax, &CodeGenerator::vmaxps);
      const auto reg_tmp = rp.reg<Reg64>();
      mov(reg_tmp, qword[rp.p[0] + PARAM_OFF(dst_max)]);
      vmovdqa32(zword[reg_tmp], zmm_expmax[0]);
    }
  }  // end of call stack

  // .data
  if (need_cfg_amx) {
    configure_tiles(required_amx_cfg_, &reqired_tile_cfg_);
    align(sizeof(tileconfig_t));
    L(L_amx_cfg);
    db(reinterpret_cast<const uint8_t*>(&reqired_tile_cfg_), sizeof(tileconfig_t));
  }

  align(sizeof(int32_t));
  L(l_log2ef);
  db(bit_cast<uint32_t>(std::log2f(std::exp(1.f))), sizeof(float));
  L(l_ln2);
  db(bit_cast<uint32_t>(std::log(2.f)), sizeof(float));
  L(l_halff);
  db(bit_cast<uint32_t>(.5f), sizeof(float));
  L(l_255);
  db(bit_cast<uint32_t>(255.f), sizeof(float));

  L(l_poly_c[0]);
  db(bit_cast<uint32_t>(exp_approx_f32_coeff[0]), sizeof(float));
  L(l_poly_c[1]);
  db(bit_cast<uint32_t>(exp_approx_f32_coeff[1]), sizeof(float));
  L(l_poly_c[2]);
  db(bit_cast<uint32_t>(exp_approx_f32_coeff[2]), sizeof(float));
}
#undef PARAM_OFF

/* jit_mm_batch_amx_u8s8_ab_AB16a4b_dynamic_quant_16x */
#define PARAM_OFF(field) offsetof(rt_data_t, field)
void jit_mm_batch_amx_u8s8_ab_AB16a4b_dynamic_quant_16x::mm_absmax(regs_pool* const rp, const Xbyak::Reg64& reg_tmpf32,
                                                                   std::array<Xbyak::Zmm, 16UL> zmm_absmax) {
  const auto reg_nbiter = rp->reg<Reg64>();  // iterations of blocks of size 16
  xor_(reg_nbiter.cvt32(), reg_nbiter.cvt32());

  const auto mm_absmax_16xkxtw = [&](int tw) {
    const std::array<Tmm, TW_> tmm_dst{tmm0, tmm1, tmm2};
    const auto reg_64ULL = rp->reg<Reg64>();
    mov(reg_64ULL, 64ULL);
    {  // k loop part
      const std::array<Tmm, 1> tmm_src0{tmm4};
      const std::array<Tmm, 3> tmm_src1{tmm5, tmm6, tmm7};
      const auto reg_ld_src0 = rp->reg<Reg64>();
      const auto reg_ld_src1 = rp->reg<Reg64>();
      const auto reg_src0 = rp->reg<Reg64>();
      const auto reg_src1 = rp->reg<Reg64>();
      const auto reg_kiter = rp->reg<Reg64>();
      mov(reg_ld_src0.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_src0)]);
      mov(reg_ld_src1.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_src1)]);
      mov(reg_src0, ptr[rp->p[0] + PARAM_OFF(src0)]);
      mov(reg_src1, reg_ld_src1);
      imul(reg_src1, reg_nbiter);
      add(reg_src1, qword[rp->p[0] + PARAM_OFF(src1)]);  // reg_src1 = src1 + reg_ld_src1 * reg_nbiter
      xor_(reg_kiter.cvt32(), reg_kiter.cvt32());
      // clear tmm
      for (int j = 0; j < tw; ++j) tilezero(tmm_dst[j]);
      Xbyak::Label l_kloop;
      L(l_kloop);
      {  // k loop body
        for (int j = 0; j < tw; ++j) {
          // load src0
          if (j == 0) tileloadd(tmm_src0[0], ptr[reg_src0 + reg_ld_src0]);

          // load src1
          if (j == 0) {
            tileloadd(tmm_src1[j], ptr[reg_src1 + reg_64ULL]);
          } else {
            const auto reg_curr_src1 = rp->reg<Reg64>();
            lea(reg_curr_src1, ptr[reg_src1 + j * reg_ld_src1]);
            tileloadd(tmm_src1[j], ptr[reg_curr_src1 + reg_64ULL]);
          }

          // dp
          tdpbusd(tmm_dst[j], tmm_src0[0], tmm_src1[j]);
        }
      }
      lea(reg_src0, ptr[reg_src0 + 64]);
      lea(reg_src1, ptr[reg_src1 + BYTES_TMM]);
      lea(reg_kiter, ptr[reg_kiter + 64]);
      cmp(reg_kiter.cvt32(), dword[rp->p[0] + PARAM_OFF(K)]);
      jb(l_kloop);
    }
    {  // tile epilogue: scale & absmax & store
      const auto reg_src1scale = rp->reg<Reg64>();
      const auto vreg_scale = rp->regs<Zmm, 3>();
      const auto zmm_x = rp->reg<Zmm>();
      imul(reg_src1scale, reg_nbiter, BYTES_ZMM);
      add(reg_src1scale, qword[rp->p[0] + PARAM_OFF(scale_src1)]);
      for (int ii = 0; ii < 16; ++ii) {
        for (int j = 0; j < tw; ++j) {
          if (ii == 0) {
            tilestored(ptr[reg_tmpf32 + reg_64ULL + j * BYTES_TMM], tmm_dst[j]);
            vmovups(vreg_scale[j], zword[reg_src1scale + j * BYTES_ZMM]);
            vmulps(vreg_scale[j], vreg_scale[j], zword_b[rip + l_rcp255]);
          }
          const auto tmp_dst_addr = ptr[reg_tmpf32 + j * BYTES_TMM + ii * BYTES_ZMM];
          vcvtdq2ps(zmm_x, tmp_dst_addr);
          vmulps(zmm_x, zmm_x, vreg_scale[j]);
          vmovaps(tmp_dst_addr, zmm_x);
          vrangeps(zmm_absmax[ii], zmm_absmax[ii], zmm_x, 0b1011);  // 1011 for absmax
        }
      }
    }
  };

  Xbyak::Label l_nloop_mm, l_nloop_mm_end;
  cmp(dword[rp->p[0] + PARAM_OFF(N)], (TW_ - 1) * 16);
  jle(l_nloop_mm_end, T_NEAR);
  L(l_nloop_mm);
  mm_absmax_16xkxtw(TW_);  // n loop body
  lea(reg_tmpf32, ptr[reg_tmpf32 + TW_ * BYTES_TMM]);
  lea(reg_nbiter, ptr[reg_nbiter + TW_]);
  {  // scope tmp to check break condition
    const auto tmp = rp->reg<Xbyak::Reg32>();
    imul(tmp, reg_nbiter, -16);
    add(tmp, dword[rp->p[0] + PARAM_OFF(N)]);
    cmp(tmp, (TW_ - 1) * 16);  // jmp if N - nb * 16 > (TW_ - 1) * 16
  }
  jg(l_nloop_mm);
  L(l_nloop_mm_end);

  // N Tail
  static_assert(TW_ == 3, "Only implement tail for TW_ == 3");
  Xbyak::Label l_end2;
  {  // is N_tile processing required?
    const auto tmp = rp->reg<Xbyak::Reg32>();
    imul(tmp, reg_nbiter.cvt32(), -16);
    add(tmp, dword[rp->p[0] + PARAM_OFF(N)]);
    cmp(tmp, 16);  // jmp if N - nb * 16 <= 16
    jle(l_end2, T_NEAR);
  }
  mm_absmax_16xkxtw(2);  // n loop body
  L(l_end2);
  Xbyak::Label l_end1;
  {  // is N_tile processing required?
    const auto tmp = rp->reg<Xbyak::Reg32>();
    imul(tmp, reg_nbiter.cvt32(), -16);
    add(tmp, dword[rp->p[0] + PARAM_OFF(N)]);  // jmp if N - nb * 16 <= 0
    jle(l_end1, T_NEAR);
  }
  mm_absmax_16xkxtw(1);  // n loop body
  L(l_end1);
}

void jit_mm_batch_amx_u8s8_ab_AB16a4b_dynamic_quant_16x::quant_store(  //
    regs_pool* const rp, const Xbyak::Reg64& reg_tmpf32, const std::array<Xbyak::Zmm, 16UL>& zmm_rcpscale) {
  Xbyak::Label l_nloop_mm, l_ntail, l_end;
  const auto reg_dst = rp->reg<Reg64>();
  const auto reg_nsize = rp->reg<Reg64>();
  const auto reg_niter = rp->reg<Reg64>();
  const auto reg_ld_dst = rp->reg<Reg64>();
  mov(reg_dst, ptr[rp->p[0] + PARAM_OFF(dst)]);
  mov(reg_nsize.cvt32(), dword[rp->p[0] + PARAM_OFF(N)]);
  and_(reg_nsize, -16);  // N / 16 * 16
  jz(l_ntail, T_NEAR);
  xor_(reg_niter.cvt32(), reg_niter.cvt32());
  mov(reg_ld_dst.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_dst)]);
  L(l_nloop_mm);
  {  // n loop body
    const auto reg_dst_off = rp->reg<Reg64>();
    xor_(reg_dst_off.cvt32(), reg_dst_off.cvt32());
    for (int ii = 0; ii < 16; ++ii) {
      const auto zmm_x = rp->reg<Zmm>();
      vmulps(zmm_x, zmm_rcpscale[ii], zword[reg_tmpf32 + ii * BYTES_ZMM]);  // quant
      vcvtps2dq(zmm_x, zmm_x);
      vpmovsdb(ptr[reg_dst + reg_dst_off], zmm_x);
      if (ii != 15)  // omit update the last reg_dst_off to save one instruction
        lea(reg_dst_off, ptr[reg_dst_off + reg_ld_dst]);
    }
  }
  lea(reg_tmpf32, ptr[reg_tmpf32 + BYTES_TMM]);
  lea(reg_dst, ptr[reg_dst + VEC * sizeof(uint8_t)]);
  lea(reg_niter, ptr[reg_niter + 16]);
  cmp(reg_niter, reg_nsize);
  jb(l_nloop_mm);

  // tail processing
  L(l_ntail);
  const auto reg_ntail = rp->reg<Reg64>();
  mov(reg_ntail.cvt32(), dword[rp->p[0] + PARAM_OFF(N)]);
  and_(reg_ntail, 16 - 1);  // reg_ntail = N % 16
  jz(l_end, T_NEAR);
  {
    const auto reg_tmp = rp->reg<Reg64>();
    const auto mask_tail16 = rp->reg<Opmask>();
    mov(reg_tmp, 1U);                   // (1 << (reg_ntail % 16)) - 1
    shlx(reg_tmp, reg_tmp, reg_ntail);  // (1 << (reg_ntail % 16)) - 1
    sub(reg_tmp, 1);                    // (1 << (reg_ntail % 16)) - 1
    kmovd(mask_tail16, reg_tmp.cvt32());
    {
      const auto reg_dst_off = rp->reg<Reg64>();
      xor_(reg_dst_off.cvt32(), reg_dst_off.cvt32());
      for (int ii = 0; ii < 16; ++ii) {
        const auto zmm_x = rp->reg<Zmm>();
        vmulps(zmm_x, zmm_rcpscale[ii], zword[reg_tmpf32 + ii * BYTES_ZMM]);  // quant
        vcvtps2dq(zmm_x, zmm_x);
        vpmovsdb(ptr[reg_dst + reg_dst_off] | mask_tail16, zmm_x);
        if (ii != 15)  // omit update the last reg_dst_off to save one instruction
          lea(reg_dst_off, ptr[reg_dst_off + reg_ld_dst]);
      }
    }
  }
  L(l_end);
}

void jit_mm_batch_amx_u8s8_ab_AB16a4b_dynamic_quant_16x::generate() {
  bool need_cfg_amx = pre_amx_cfg_ != nullptr && *pre_amx_cfg_ != required_amx_cfg_;
  std::shared_ptr<void> use_loacl_label = {(inLocalLabel(), nullptr), [&](...) { outLocalLabel(); }};
  {
    constexpr auto tmp_mem_size = BYTES_ZMM;
    const auto stack_size = (need_cfg_amx ? sizeof(tileconfig_t) : 0) + tmp_mem_size;
    regs_pool rp(this, 1, {12, 32, 1}, stack_size, regs_pool::DefaultFlags, 64);
    std::shared_ptr<void> local_cfg;
    if (need_cfg_amx) {  // create a local amx config environment
      sttilecfg(ptr[rsp]);
      ldtilecfg(ptr[rip + L_amx_cfg]);
      lea(rsp, ptr[rsp + sizeof(tileconfig_t)]);
      local_cfg = {nullptr, [&](...) {
                     lea(rsp, ptr[rsp - sizeof(tileconfig_t)]);
                     ldtilecfg(ptr[rsp]);
                   }};
    }

    const auto reg_headstacksize = rp.reg<Reg64>();
    const auto reg_stacksize = rp.reg<Reg64>();

    // stacksize = (N + 16) * M * headsize * sizeof(float) + 64  // 16 as enough padding; 64 for tmp memory
    mov(reg_headstacksize.cvt32(), dword[rp.p[0] + PARAM_OFF(N)]);
    imul(reg_headstacksize.cvt32(), reg_headstacksize.cvt32(), M * sizeof(float));  // (N) * M * sizeof(float)
    lea(reg_headstacksize.cvt32(),
        ptr[reg_headstacksize.cvt32() + 16 * M * sizeof(float) + 64]);  // (+16) * M * sizeof(float)
    mov(reg_stacksize.cvt32(), reg_headstacksize.cvt32());
    imul(reg_stacksize.cvt32(), dword[rp.p[0] + PARAM_OFF(batch_size)]);  // (N + 16) * M * sizeof(float) * headsize
    std::shared_ptr<void> use_dyn_stack = {(sub(rsp, reg_stacksize), nullptr), [&](...) { add(rsp, reg_stacksize); }};
    const auto m512_tmp = rsp - 64;

    const auto zmm_absmax = rp.regs<Zmm, 16>();
    vxorps(zmm_absmax[0], zmm_absmax[0], zmm_absmax[0]);
    for (int i = 1; i < 16; ++i) vmovdqa32(zmm_absmax[i], zmm_absmax[0]);

    // iter batch for matmul
    {
      const auto reg_biter = rp.reg<Reg64>();
      xor_(reg_biter.cvt32(), reg_biter.cvt32());
      Xbyak::Label l_bloop_mm;
      L(l_bloop_mm);
      {  // call mm_absmax
        const auto preg64_tmp = rp.reg<Reg64>();
        mov(preg64_tmp, reg_headstacksize);
        imul(preg64_tmp, reg_biter);
        lea(preg64_tmp, ptr[rsp + preg64_tmp]);
        mm_absmax(&rp, preg64_tmp, zmm_absmax);
      }
      const auto r64_tmp = rp.reg<Reg64>();
      mov(r64_tmp, qword[rp.p[0] + PARAM_OFF(batchstep_src0)]);
      add(r64_tmp, qword[rp.p[0] + PARAM_OFF(src0)]);
      mov(qword[rp.p[0] + PARAM_OFF(src0)], r64_tmp);  // src0 += batchstep_src0
      mov(r64_tmp, qword[rp.p[0] + PARAM_OFF(batchstep_src1)]);
      add(r64_tmp, qword[rp.p[0] + PARAM_OFF(src1)]);
      mov(qword[rp.p[0] + PARAM_OFF(src1)], r64_tmp);  // src1 += batchstep_src1
      mov(r64_tmp, qword[rp.p[0] + PARAM_OFF(batchstep_src1scale)]);
      add(r64_tmp, qword[rp.p[0] + PARAM_OFF(scale_src1)]);
      mov(qword[rp.p[0] + PARAM_OFF(scale_src1)], r64_tmp);  // scale_src1+=step
      lea(reg_biter, ptr[reg_biter + 1]);
      cmp(reg_biter.cvt32(), dword[rp.p[0] + PARAM_OFF(batch_size)]);
      jb(l_bloop_mm);
    }

    // processing intermediate absmax
    const auto& zmm_rcpscale = zmm_absmax;
    {
      transpose_16x16_ps(zmm_absmax, rp.regs<Zmm, 16>());  // 4 inst per zmm
      reduce_vmms(zmm_absmax, [this](auto& _0, auto& _1, auto& _2) { vrangeps(_0, _1, _2, 0b1011); });
      const auto zmm_127 = rp.reg<Zmm>();
      const auto zmm_scale = rp.reg<Zmm>();
      const auto zmm_tmp = rp.reg<Zmm>();
      vpbroadcastd(zmm_127, dword[rip + l_127f]);
      vdivps(zmm_scale, zmm_absmax[0], zmm_127);
      const auto reg_dst_scale = rp.reg<Reg64>();
      mov(reg_dst_scale, qword[rp.p[0] + PARAM_OFF(dst_scale)]);
      vmovups(zword[reg_dst_scale], zmm_scale);
      vaddps(zmm_absmax[0], zmm_absmax[0], zword_b[rip + l_float_epsilon]);  // avoid zero division
      vdivps(zmm_tmp, zmm_127, zmm_absmax[0]);                               // 127 / max
      vmovdqa32(zword[m512_tmp], zmm_tmp);                                   // rcpscale
      for (int ii = 0; ii < 16; ++ii) vpbroadcastd(zmm_rcpscale[ii], dword[m512_tmp + ii * sizeof(float)]);
    }

    // iter batch for quant & store
    {
      const auto reg_bleft = rp.reg<Reg64>();
      const auto reg_tmpf32off = rp.reg<Reg64>();
      mov(reg_bleft.cvt32(), dword[rp.p[0] + PARAM_OFF(batch_size)]);
      xor_(reg_tmpf32off.cvt32(), reg_tmpf32off.cvt32());
      Xbyak::Label l_bloop_q10n;
      L(l_bloop_q10n);
      {
        const auto preg64_tmp = rp.reg<Reg64>();
        lea(preg64_tmp, ptr[rsp + reg_tmpf32off]);
        quant_store(&rp, preg64_tmp, zmm_rcpscale);
      }
      const auto r64_tmp = rp.reg<Reg64>();
      mov(r64_tmp, qword[rp.p[0] + PARAM_OFF(batchstep_dst)]);
      add(r64_tmp, qword[rp.p[0] + PARAM_OFF(dst)]);
      mov(qword[rp.p[0] + PARAM_OFF(dst)], r64_tmp);  // dst += batchstep_dst
      lea(reg_tmpf32off, ptr[reg_tmpf32off + reg_headstacksize]);
      sub(reg_bleft, 1);
      jg(l_bloop_q10n);
    }
  }  // end of call stack

  // .data
  if (need_cfg_amx) {
    configure_tiles(required_amx_cfg_, &reqired_tile_cfg_);
    align(sizeof(tileconfig_t));
    L(L_amx_cfg);
    db(reinterpret_cast<const uint8_t*>(&reqired_tile_cfg_), sizeof(tileconfig_t));
  }

  align(sizeof(int32_t));
  L(l_127f);
  db(bit_cast<uint32_t>(127.f), sizeof(float));
  L(l_rcp255);
  db(bit_cast<uint32_t>(1.f / 255.f), sizeof(float));
  L(l_float_epsilon);
  db(bit_cast<uint32_t>(1e-9f), sizeof(float));
}

#undef PARAM_OFF
/* jit_scale_mm_amx_u8s8_ab_BA16b_16x */
#define PARAM_OFF(field) offsetof(rt_data_t, field)
void jit_scale_mm_amx_u8s8_ab_BA16b_16x::mm_absmax(regs_pool* const rp, std::array<Xbyak::Zmm, 16UL> zmm_absmax) {
  const auto reg_nbiter = rp->reg<Reg64>();  // iterations of blocks of size 16
  xor_(reg_nbiter.cvt32(), reg_nbiter.cvt32());
  const auto reg_dst = rp->reg<Reg64>();
  mov(reg_dst, qword[rp->p[0] + PARAM_OFF(dst)]);
  const auto reg_lb_dst = rp->reg<Reg64>();
  imul(reg_lb_dst.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_dst)], sizeof(float));

  const auto mm_absmax_16xkxtw = [&](int tw) {
    const auto reg_prescale0 = rp->reg<Reg64>();
    mov(reg_prescale0, qword[rp->p[0] + PARAM_OFF(prescale_src0)]);

    const std::array<Tmm, 3> tmm_dst{tmm0, tmm1, tmm2};
    {  // k loop part
      const auto reg_64ULL = rp->reg<Reg64>();
      mov(reg_64ULL, 64ULL);
      const std::array<Tmm, 1> tmm_src0{tmm4};
      const std::array<Tmm, 3> tmm_src1{tmm5, tmm6, tmm7};
      const auto reg_ld_src1 = rp->reg<Reg64>();
      const auto reg_src0 = rp->reg<Reg64>();
      const auto reg_src1 = rp->reg<Reg64>();
      const auto reg_kiter = rp->reg<Reg64>();
      mov(reg_ld_src1.cvt32(), dword[rp->p[0] + PARAM_OFF(ld_src1)]);
      mov(reg_src0, ptr[rp->p[0] + PARAM_OFF(src0)]);
      mov(reg_src1, reg_ld_src1);
      imul(reg_src1, reg_nbiter);
      add(reg_src1, qword[rp->p[0] + PARAM_OFF(src1)]);  // reg_src1 = src1 + reg_ld_src1 * reg_nbiter
      xor_(reg_kiter.cvt32(), reg_kiter.cvt32());
      // clear tmm
      for (int j = 0; j < tw; ++j) tilezero(tmm_dst[j]);
      Xbyak::Label l_kloop;
      L(l_kloop);
      {  // k loop body
        for (int j = 0; j < tw; ++j) {
          // load src0
          if (j == 0) {
            for (size_t tj = 0; tj < sizeof(float); ++tj) {
              for (int ii = 0; ii < 16; ++ii) {
                const auto xs = rp->reg<Zmm>();
                vmovaps(xs, zword[reg_src0 + ii * BYTES_ZMM + tj * BYTES_TMM]);
                vmulps(xs, xs, zword_b[reg_prescale0 + ii * sizeof(float)]);
                vcvtps2udq(xs | T_rn_sae, xs);
                vpmovusdb(xword[rsp + ii * BYTES_ZMM + tj * BYTES_XMM], xs);
              }
            }
            tileloadd(tmm_src0[0], ptr[rsp + reg_64ULL]);
          }

          // load src1
          if (j == 0) {
            tileloadd(tmm_src1[j], ptr[reg_src1 + reg_64ULL]);
          } else {
            const auto reg_curr_src1 = rp->reg<Reg64>();
            lea(reg_curr_src1, ptr[reg_src1 + j * reg_ld_src1]);
            tileloadd(tmm_src1[j], ptr[reg_curr_src1 + reg_64ULL]);
          }

          // dp
          tdpbusd(tmm_dst[j], tmm_src0[0], tmm_src1[j]);
        }
      }
      lea(reg_src0, ptr[reg_src0 + BYTES_TMM * sizeof(float)]);  // 4 float tile => 1 u8 tile
      lea(reg_src1, ptr[reg_src1 + BYTES_TMM]);
      lea(reg_kiter, ptr[reg_kiter + 64]);
      cmp(reg_kiter.cvt32(), dword[rp->p[0] + PARAM_OFF(K)]);
      jb(l_kloop);
    }
    {  // tile epilogue: scale & absmax & store
      const auto reg_src1scale = rp->reg<Reg64>();
      const auto vreg_scale = rp->regs<Zmm, 3>();
      const auto zmm_x = rp->reg<Zmm>();
      const auto reg_src0scale = rp->reg<Reg64>();
      mov(reg_src0scale, qword[rp->p[0] + PARAM_OFF(scale_src0)]);

      imul(reg_src1scale, reg_nbiter, BYTES_ZMM);
      add(reg_src1scale, qword[rp->p[0] + PARAM_OFF(scale_src1)]);
      for (int ii = 0; ii < 16; ++ii) {
        const auto reg_curr_dst = rp->reg<Reg64>();
        (ii == 0) ? mov(reg_curr_dst, reg_dst) : lea(reg_curr_dst, ptr[reg_curr_dst + reg_lb_dst]);
        for (int j = 0; j < tw; ++j) {
          if (ii == 0) {
            tilestored(ptr[reg_dst + reg_lb_dst + j * BYTES_ZMM], tmm_dst[j]);
            vmovups(vreg_scale[j], zword[reg_src1scale + j * BYTES_ZMM]);
          }

          const auto tmp_dst_addr = ptr[reg_curr_dst + j * BYTES_ZMM];
          vcvtdq2ps(zmm_x, tmp_dst_addr);
          vmulps(zmm_x, zmm_x, vreg_scale[j]);
          vmulps(zmm_x, zmm_x, zword_b[reg_src0scale + ii * sizeof(float)]);
          vmovaps(tmp_dst_addr, zmm_x);
          vrangeps(zmm_absmax[ii], zmm_absmax[ii], zmm_x, 0b1011);  // 1011 for absmax
        }
      }
    }
  };

  Xbyak::Label l_nloop_mm, l_nloop_mm_end;
  cmp(dword[rp->p[0] + PARAM_OFF(N)], (TW_ - 1) * 16);
  jle(l_nloop_mm_end, T_NEAR);
  L(l_nloop_mm);
  mm_absmax_16xkxtw(TW_);  // n loop body
  lea(reg_dst, ptr[reg_dst + TW_ * BYTES_ZMM]);
  lea(reg_nbiter, ptr[reg_nbiter + TW_]);
  {  // scope tmp to check break condition
    const auto tmp = rp->reg<Xbyak::Reg32>();
    imul(tmp, reg_nbiter, -16);
    add(tmp, dword[rp->p[0] + PARAM_OFF(N)]);
    cmp(tmp, (TW_ - 1) * 16);  // jmp if N - nb * 16 > (TW_ - 1) * 16
  }
  jg(l_nloop_mm);
  L(l_nloop_mm_end);

  // N Tail
  static_assert(TW_ == 3, "Only implement tail for TW_ == 3");
  Xbyak::Label l_end2;
  {  // is N_tile processing required?
    const auto tmp = rp->reg<Xbyak::Reg32>();
    imul(tmp, reg_nbiter.cvt32(), -16);
    add(tmp, dword[rp->p[0] + PARAM_OFF(N)]);
    cmp(tmp, 16);  // jmp if N - nb * 16 <= 16
    jle(l_end2, T_NEAR);
  }
  mm_absmax_16xkxtw(2);  // n loop body
  L(l_end2);
  Xbyak::Label l_end1;
  {  // is N_tile processing required?
    const auto tmp = rp->reg<Xbyak::Reg32>();
    imul(tmp, reg_nbiter.cvt32(), -16);
    add(tmp, dword[rp->p[0] + PARAM_OFF(N)]);  // jmp if N - nb * 16 <= 0
    jle(l_end1, T_NEAR);
  }
  mm_absmax_16xkxtw(1);  // n loop body
  L(l_end1);
}

void jit_scale_mm_amx_u8s8_ab_BA16b_16x::generate() {
  bool need_cfg_amx = pre_amx_cfg_ != nullptr && *pre_amx_cfg_ != required_amx_cfg_;
  std::shared_ptr<void> use_loacl_label = {(inLocalLabel(), nullptr), [&](...) { outLocalLabel(); }};
  {
    constexpr auto tmp_mem_size = BYTES_TMM * TH_;
    regs_pool rp(this, 1, {10, 32, 0}, (need_cfg_amx ? sizeof(tileconfig_t) : 0) + tmp_mem_size, 0, 64);
    std::shared_ptr<void> local_cfg;
    if (need_cfg_amx) {  // create a local amx config environment
      sttilecfg(ptr[rsp]);
      ldtilecfg(ptr[rip + L_amx_cfg]);
      lea(rsp, ptr[rsp + sizeof(tileconfig_t)]);
      local_cfg = {nullptr, [&](...) {
                     lea(rsp, ptr[rsp - sizeof(tileconfig_t)]);
                     ldtilecfg(ptr[rsp]);
                   }};
    }

    const auto zmm_absmax = rp.regs<Zmm, 16>();
    vxorps(zmm_absmax[0], zmm_absmax[0], zmm_absmax[0]);
    for (int i = 1; i < 16; ++i) vmovdqa32(zmm_absmax[i], zmm_absmax[0]);

    {  // call mm_absmax
      mm_absmax(&rp, zmm_absmax);
    }

    // processing absmax
    {
      transpose_16x16_ps(zmm_absmax, rp.regs<Zmm, 16>());  // 4 inst per zmm
      reduce_vmms(zmm_absmax, [this](auto& _0, auto& _1, auto& _2) { vrangeps(_0, _1, _2, 0b1011); });
      const auto& vreg_absmax = zmm_absmax[0];
      const auto reg_absmax_dst = rp.reg<Reg64>();
      mov(reg_absmax_dst, qword[rp.p[0] + PARAM_OFF(absmax_dst)]);
      vrangeps(vreg_absmax, vreg_absmax, zword[reg_absmax_dst], 0b1011);
      vmovaps(zword[reg_absmax_dst], vreg_absmax);
    }
  }  // end of call stack

  // .data
  if (need_cfg_amx) {
    configure_tiles(required_amx_cfg_, &reqired_tile_cfg_);
    align(sizeof(tileconfig_t));
    L(L_amx_cfg);
    db(reinterpret_cast<const uint8_t*>(&reqired_tile_cfg_), sizeof(tileconfig_t));
  }

  align(sizeof(int32_t));
  L(l_127f);
  db(bit_cast<uint32_t>(127.f), sizeof(float));
  L(l_float_epsilon);
  db(bit_cast<uint32_t>(1e-9f), sizeof(float));
}

#undef PARAM_OFF

}  // namespace jd
