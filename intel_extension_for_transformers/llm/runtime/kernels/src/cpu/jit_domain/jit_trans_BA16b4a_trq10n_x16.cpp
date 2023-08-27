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
#include "jit_trans_BA16b4a_trq10n_x16.hpp"

#include <memory>

#define PARAM_OFF(field) offsetof(jit_trans_BA16b4a_trq10n_x16::rt_data_t, field)

namespace jd {

void jit_trans_BA16b4a_trq10n_x16::generate() {
  std::shared_ptr<void> use_loacl_label = {(inLocalLabel(), nullptr), [&](...) { outLocalLabel(); }};
  regs_pool rp(this, 1, {11, 9, 3}, 0, regs_pool::DisableEpilog);

  const auto reg_stacksize = rp.reg<Xbyak::Reg64>();
  {
    const auto reg_src = rp.reg<Xbyak::Reg64>();
    const auto reg_dst = rp.reg<Xbyak::Reg64>();
    const auto reg_ld_src = rp.reg<Xbyak::Reg64>();
    const auto reg_3ld_src = rp.reg<Xbyak::Reg64>();
    const auto reg_scale = rp.reg<Xbyak::Reg64>();
    const auto reg_n = rp.reg<Xbyak::Reg64>();
    const auto reg_msize = rp.reg<Xbyak::Reg64>();
    const auto reg_miter = rp.reg<Xbyak::Reg64>();
    const auto reg_f32src = rp.reg<Xbyak::Reg64>();
    const auto reg_tmp = rp.reg<Xbyak::Reg64>();

    // stacksize = (m + 4) * 16 * sizeof(float) + 64 // 64 for alignment
    mov(reg_stacksize.cvt32(), dword[rp.p[0] + PARAM_OFF(M)]);
    imul(reg_stacksize, reg_stacksize, 16 * sizeof(float));
    lea(reg_stacksize, ptr[reg_stacksize + 4 * 16 * sizeof(float) + 64]);
    sub(rsp, reg_stacksize);

    // align stack tmp mem
    mov(reg_f32src, rsp);
    and_(reg_f32src, -64);
    add(reg_f32src, 64);

    // prepare control registers
    const auto vreg_vpshufb_bw_ctl = rp.reg<Xbyak::Zmm>();
    vmovdqa32(vreg_vpshufb_bw_ctl, zword[rip + l_vpshufb_bw_ctl]);
    const auto vreg_vpshufb_wd_ctl = rp.reg<Xbyak::Zmm>();
    vmovdqa32(vreg_vpshufb_wd_ctl, zword[rip + l_vpshufb_wd_ctl]);
    const auto vpshufb_bw_k = rp.reg<Opmask>();
    mov(reg_tmp, vpshufb_mask_bw);
    kmovq(vpshufb_bw_k, reg_tmp);
    mov(reg_tmp, vpshufb_mask_wd);
    const auto vpshufb_wd_k = rp.reg<Opmask>();
    kmovq(vpshufb_wd_k, reg_tmp);
    const auto vreg_zero = rp.reg<Xbyak::Zmm>();
    vxorps(vreg_zero, vreg_zero, vreg_zero);

    mov(reg_src, ptr[rp.p[0] + PARAM_OFF(src)]);
    mov(reg_dst, reg_f32src);
    mov(reg_ld_src.cvt32(), dword[rp.p[0] + PARAM_OFF(ld_src)]);
    imul(reg_3ld_src, reg_ld_src, 3);
    mov(reg_scale, ptr[rp.p[0] + PARAM_OFF(src_scale)]);

    mov(reg_n.cvt32(), dword[rp.p[0] + PARAM_OFF(N)]);
    mov(reg_tmp, l_nsize_tbl);
    jmp(ptr[reg_tmp + reg_n * sizeof(void*)], T_NEAR);  // switch(reg_n) ...
    for (int j = 1; j <= 16; ++j) {                     // case(1) case(2) ... case(16)
      // param: reg_src reg_dst reg_scale
      const auto dequant_4x16_max = [&](int m_tile, const Zmm& vreg_absmax, const Opmask& mask_n16) {
        SPARSE_LOG_IF(FATAL, m_tile < 0 || m_tile > 4) << "M tile should between 1 to 4";
        const auto mat = rp.regs<Xbyak::Zmm, 4>();
        for (int i = 0; i < m_tile; ++i) {
          const auto addr_src = i != 3 ? ptr[reg_src + i * reg_ld_src] : ptr[reg_src + reg_3ld_src];
          const auto move_dst = j == 16 ? mat[i] : mat[i] | mask_n16 | T_z;
          vpmovsxbd(move_dst, addr_src);                           // move in
          vcvtdq2ps(mat[i], mat[i]);                               // int32 => float
          vmulps(mat[i], zword_b[reg_scale + i * sizeof(float)]);  // dequant
          vmovaps(ptr[reg_dst + i * BYTES_ZMM], mat[i]);           // reg_dst should now pointing at part of reg_f32src
          vrangeps(vreg_absmax, vreg_absmax, mat[i], 0b1011);      // 0b1011 for abs(absmax)
        }
        for (int i = m_tile; i < 4; ++i) vmovaps(ptr[reg_dst + i * BYTES_ZMM], vreg_zero);  // tail
      };
      const auto quant_4x16_interleave = [&](const Zmm& vreg_dstrcpscale) {  // param: reg_src reg_dst
        const auto mat = rp.regs<Xbyak::Zmm, 4>();
        for (int i = 0; i < 4; ++i) {
          vmulps(mat[i], vreg_dstrcpscale, zword[reg_src + i * BYTES_ZMM]);  // mat[i] should now bwteen -127 and 127
        }                                                                    // lo ......... hi
        vcvtps2dq(mat[0], mat[0]);                                           // a0--- ... a15---
        vcvtps2dq(mat[1], mat[1]);                                           // b0--- ... b15---
        vcvtps2dq(mat[2], mat[2]);                                           // c0--- ... c15---
        vcvtps2dq(mat[3], mat[3]);                                           // d0--- ... d15---
        vpshufb(mat[0] | vpshufb_bw_k, mat[1], vreg_vpshufb_bw_ctl);         // a0b0--   ... a15b15--
        vpshufb(mat[2] | vpshufb_bw_k, mat[3], vreg_vpshufb_bw_ctl);         // c0d0--   ... c15d15--
        vpshufb(mat[0] | vpshufb_wd_k, mat[2], vreg_vpshufb_wd_ctl);         // a0b0c0d0 ... a15b15c15d15
        vmovups(zword[reg_dst], mat[0]);                                     // move out
      };

      L(l_nsize_case[j]);  // case(j) starts
      const auto mask_n16 = rp.reg<Opmask>();
      if (j != 16) {
        mov(reg_tmp, (1ULL << j) - 1);
        kmovq(mask_n16, reg_tmp);
      }

      Xbyak::Label l_mtail, l_dequant_epilogue;
      const auto vreg_absmax = rp.reg<Xbyak::Zmm>();
      vmovaps(vreg_absmax, vreg_zero);
      mov(reg_msize.cvt32(), dword[rp.p[0] + PARAM_OFF(M)]);
      and_(reg_msize, -4);  // reg_msize = M / 4 * 4
      jz(l_mtail, T_NEAR);  // using the ZF flag set by `AND r/m64, imm32`
      // m loop de-q10n
      {
        xor_(reg_miter, reg_miter);
        Xbyak::Label l_mloop_dequant;
        L(l_mloop_dequant);
        dequant_4x16_max(4, vreg_absmax, mask_n16);
        lea(reg_src, ptr[reg_src + 4 * reg_ld_src]);
        lea(reg_dst, ptr[reg_dst + 4 * BYTES_ZMM]);
        lea(reg_scale, ptr[reg_scale + 4 * sizeof(float)]);
        lea(reg_miter, ptr[reg_miter + 4]);
        cmp(reg_miter, reg_msize);
        jb(l_mloop_dequant);
      }

      L(l_mtail);
      Xbyak::Label l_mtail_case[4], l_mtail_tbl;
      mov(reg_msize.cvt32(), dword[rp.p[0] + PARAM_OFF(M)]);
      and_(reg_msize, 4U - 1);  // M % 4
      jz(l_dequant_epilogue, T_NEAR);
      mov(reg_tmp, l_mtail_tbl);
      jmp(ptr[reg_tmp + reg_msize * sizeof(void*)], T_NEAR);  // switch(reg_msize) ...
      for (int i = 1; i < 4; ++i) {                           // case(1) case(2) case(3)
        L(l_mtail_case[i]);
        dequant_4x16_max(i, vreg_absmax, mask_n16);
        // no need to update reg_src/reg_dst/reg_scale
        jmp(l_dequant_epilogue, T_NEAR);  // break from the current case
      }
      align(sizeof(void*));
      L(l_mtail_tbl);
      db(reinterpret_cast<uint64_t>(nullptr), sizeof(void*));  // case 0 should not occour
      for (int i = 1; i < 4; ++i) putL(l_mtail_case[i]);

      L(l_dequant_epilogue);
      const auto vreg_tmp = rp.reg<Xbyak::Zmm>();
      vdivps(vreg_tmp | T_z | (j == 16 ? k0 : mask_n16), vreg_absmax, zword_b[rip + L_127f]);  // scale
      mov(reg_scale, ptr[rp.p[0] + PARAM_OFF(dst_scale)]);
      vmovdqu32(zword[reg_scale], vreg_tmp);
      vpbroadcastd(vreg_tmp, dword[rip + L_127f]);

      const auto& vreg_dstrcpscale = vreg_absmax;  // note that absmax and dstscale are the same vreg
      vdivps(vreg_dstrcpscale | T_z | (j == 16 ? k0 : mask_n16), vreg_tmp, vreg_absmax);  // rcp scale; s8 = f32 / scale

      // quantize & interleave
      mov(reg_src, reg_f32src);
      mov(reg_dst, ptr[rp.p[0] + PARAM_OFF(dst)]);
      mov(reg_msize.cvt32(), dword[rp.p[0] + PARAM_OFF(M)]);
      xor_(reg_miter, reg_miter);
      {
        Xbyak::Label l_mloop_quant;
        L(l_mloop_quant);
        quant_4x16_interleave(vreg_dstrcpscale);
        lea(reg_src, ptr[reg_src + 4 * BYTES_ZMM]);
        lea(reg_dst, ptr[reg_dst + BYTES_ZMM]);
        lea(reg_miter, ptr[reg_miter + 4]);
        cmp(reg_miter, reg_msize);  // implicitly pad to 4
        jb(l_mloop_quant);
      }
      add(rsp, reg_stacksize);
      rp.close();  // return from the outer switch case of N from 1 to 16
    }
  }
  align(sizeof(void*));
  L(l_nsize_tbl);
  db(reinterpret_cast<uint64_t>(nullptr), sizeof(void*));  // case 0 should never occour
  for (int i = 1; i <= 16; ++i) putL(l_nsize_case[i]);

  align(64);
  const uint8_t vpshufb_control_bw[16] = {
      // used with mask bcast_1to16(0b0010)
      0, 0,  0, 0,  // 1st dword in every 128 bits
      0, 4,  0, 0,  // 2nd dword in every 128 bits
      0, 8,  0, 0,  // 3rd dword in every 128 bits
      0, 12, 0, 0,  // 4th dword in every 128 bits
  };
  const uint8_t vpshufb_control_wd[16] = {
      // used with mask bcast_1to16(0b1100)
      0, 0, 0,  1,   // 1st dword in every 128 bits
      0, 0, 4,  5,   // 2nd dword in every 128 bits
      0, 0, 8,  9,   // 3rd dword in every 128 bits
      0, 0, 12, 13,  // 4th dword in every 128 bits
  };
  L(l_vpshufb_bw_ctl);
  for (int i = 0; i < 4; ++i) db(vpshufb_control_bw, 16);
  L(l_vpshufb_wd_ctl);
  for (int i = 0; i < 4; ++i) db(vpshufb_control_wd, 16);
  L(L_int32_max);
  db(INT32_MAX, sizeof(int32_t));
  L(L_127f);
  db(bit_cast<int32_t>(127.f), sizeof(float));
}

}  // namespace jd
