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
#include "jit_trans_AB16a4b_16x.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#define PARAM_OFF(field) offsetof(jit_trans_AB16a4b_16x::rt_data_t, field)

namespace jd {

void jit_trans_AB16a4b_16x::transpose_16x16_ps(  //
    regs_pool* const rp, const Reg64& src, const Reg64& dst, const Reg64& ld_src, const Reg64& reg_tmp, const int M,
    const Xbyak::Opmask& mask) {
  const auto mat = rp->regs<Xbyak::Zmm, 16>();
  const auto tmp = rp->regs<Xbyak::Zmm, 16>();
  if (cvt_s8u8) vmovdqa32(tmp[0], zword[rip + l_128x64]);

  // move in
  for (int i = 0; i < 16; ++i) {
    if (i < M) {
      imul(reg_tmp, ld_src, i);
      if (mask != k0)
        !cvt_s8u8 ? vmovdqu8(mat[i] | mask | T_z, zword[src + reg_tmp])
                  : vpaddb(mat[i] | mask | T_z, tmp[0], zword[src + reg_tmp]);
      else
        !cvt_s8u8 ? vmovdqu8(mat[i], zword[src + reg_tmp]) : vpaddb(mat[i], tmp[0], zword[src + reg_tmp]);
    } else {
      vpxord(Xbyak::Xmm(mat[i]), Xbyak::Xmm(mat[i]), Xbyak::Xmm(mat[i]));
    }
  }

  // transpose 16 zmms
  if (M > 0) jit_generator::transpose_16x16_ps(mat, tmp, 16);

  // move out
  for (int i = 0; i < (mask == k0 ? 16 : pad_n / 4); ++i) vmovdqa32(ptr[dst + i * tile_m * BYTES_ZMM], mat[i]);
}

void jit_trans_AB16a4b_16x::generate() {
  regs_pool rp(this, 1, {7 + (tile_m > 1 ? 2 : 0), 32, 1}, 0, regs_pool::DisableEpilog);
  const auto reg_src = rp.reg<Reg64>();
  const auto reg_dst = rp.reg<Reg64>();
  const auto reg_ld_src = rp.reg<Reg64>();
  const auto reg_m = rp.reg<Reg64>();
  const auto reg_nsize = rp.reg<Reg64>();
  const auto reg_niter = rp.reg<Reg64>();
  const auto reg_tmp = rp.reg<Reg64>();
  const auto mask_n = rp.reg<Opmask>();

  mov(reg_src, ptr[rp.p[0] + PARAM_OFF(src)]);
  mov(reg_dst, ptr[rp.p[0] + PARAM_OFF(dst)]);
  mov(reg_ld_src.cvt32(), dword[rp.p[0] + PARAM_OFF(ld_src)]);
  mov(reg_m.cvt32(), dword[rp.p[0] + PARAM_OFF(M)]);
  mov(reg_nsize.cvt32(), dword[rp.p[0] + PARAM_OFF(N)]);

  mov(reg_tmp, l_msize_tbl);  // switch(reg_m) ...
  jmp(ptr[reg_tmp + reg_m * sizeof(void*)], T_NEAR);
  std::vector<Xbyak::Label> l_msize_case(tile_m * 16 + 1);
  for (int i = 0; i <= 16 * tile_m; ++i) {  // case(0) case(1) ... case(16)
    L(l_msize_case[i]);

    Xbyak::Label l_tail, l_end;
    and_(reg_nsize, -64);  // reg_nsize = N / 64 * 64
    jz(l_tail, T_NEAR);    // using the ZF flag set by `AND r/m64, imm32`
    xor_(reg_niter, reg_niter);

    {  // n loop
      Xbyak::Label l_nloop;
      L(l_nloop);

      for (int ii = 0; ii < tile_m * 16; ii += 16) {
        auto reg_src_tmp = (ii != 0) ? rp.reg<Reg64>() : reg_src;
        auto reg_dst_tmp = (ii != 0) ? rp.reg<Reg64>() : reg_dst;
        if (ii != 0) {
          imul(reg_src_tmp, reg_ld_src, ii);
          lea(reg_src_tmp, ptr[reg_src_tmp + reg_src]);
          lea(reg_dst_tmp, ptr[reg_dst + ii / 16 * BYTES_ZMM]);
        }
        transpose_16x16_ps(&rp, reg_src_tmp, reg_dst_tmp, reg_ld_src, reg_tmp, std::max(0, std::min(16, i - ii)), k0);
      }

      lea(reg_src, ptr[reg_src + 64]);
      lea(reg_dst, ptr[reg_dst + tile_m * BYTES_ZMM * 16]);
      lea(reg_niter, ptr[reg_niter + 64]);
      cmp(reg_niter, reg_nsize);
      jb(l_nloop);
    }

    L(l_tail);
    mov(reg_nsize.cvt32(), dword[rp.p[0] + PARAM_OFF(N)]);
    and_(reg_nsize, 64U - 1);  // N % 64
    jz(l_end, T_NEAR);         // using the ZF flag set by `AND r/m64, imm32`

    mov(reg_tmp, 1ULL);                 // (1 << (N % 16)) - 1
    shlx(reg_tmp, reg_tmp, reg_nsize);  // (1 << (N % 16)) - 1
    sub(reg_tmp, 1);                    // (1 << (N % 16)) - 1
    kmovq(mask_n, reg_tmp);

    for (int ii = 0; ii < tile_m * 16; ii += 16) {
      auto reg_src_tmp = (ii != 0) ? rp.reg<Reg64>() : reg_src;
      auto reg_dst_tmp = (ii != 0) ? rp.reg<Reg64>() : reg_dst;
      if (ii != 0) {
        imul(reg_src_tmp, reg_ld_src, ii);
        lea(reg_src_tmp, ptr[reg_src_tmp + reg_src]);
        lea(reg_dst_tmp, ptr[reg_dst + ii / 16 * BYTES_ZMM]);
      }
      transpose_16x16_ps(&rp, reg_src_tmp, reg_dst_tmp, reg_ld_src, reg_tmp, std::max(0, std::min(16, i - ii)), mask_n);
    }
    L(l_end);
    rp.close();
  }

  align(sizeof(void*));
  L(l_msize_tbl);
  for (int i = 0; i <= 16 * tile_m; ++i) putL(l_msize_case[i]);
  align(BYTES_ZMM);
  L(l_128x64);
  for (uint64_t i = 0; i < BYTES_ZMM; ++i) db(128);
}

}  // namespace jd
