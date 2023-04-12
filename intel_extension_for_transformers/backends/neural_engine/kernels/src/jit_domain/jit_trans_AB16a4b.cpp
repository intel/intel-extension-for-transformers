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
#include "jit_domain/jit_trans_AB16a4b.hpp"

#include <memory>

#define GET_OFF(field) offsetof(jit_trans_AB16a4b::rt_data_t, field)

namespace jd {

void jit_trans_AB16a4b::transpose_16x16_ps(regs_pool* const rp, const Xbyak::Reg64& src, const Xbyak::Reg64& dst,
                                           const Xbyak::Opmask& mask, bool is_tail) {
  const auto mat = rp->regs<Zmm, 16>();
  const auto tmp = rp->regs<Zmm, 16>();
  const auto tail_len = M % 16;

  // move in
  for (int i = 0; i < 16; ++i) {
    if (!is_tail || i < tail_len) {
      if (mask != k0)
        vmovdqu8(mat[i] | mask | T_z, ptr[src + i * ld_src]);
      else
        vmovdqu8(mat[i], ptr[src + i * ld_src]);
    } else {
      vpxord(Xbyak::Xmm(mat[i]), Xbyak::Xmm(mat[i]), Xbyak::Xmm(mat[i]));
    }
  }

  for (int i = 0; i < 8; ++i) {
    vpunpckldq(tmp[2 * i + 0], mat[2 * i], mat[2 * i + 1]);
    vpunpckhdq(tmp[2 * i + 1], mat[2 * i], mat[2 * i + 1]);
  }

  for (int i = 0; i < 4; ++i) {
    vpunpcklqdq(mat[4 * i + 0], tmp[4 * i + 0], tmp[4 * i + 2]);
    vpunpckhqdq(mat[4 * i + 1], tmp[4 * i + 0], tmp[4 * i + 2]);
    vpunpcklqdq(mat[4 * i + 2], tmp[4 * i + 1], tmp[4 * i + 3]);
    vpunpckhqdq(mat[4 * i + 3], tmp[4 * i + 1], tmp[4 * i + 3]);
  }

  for (int i = 0; i < 2; ++i) {
    vshufi32x4(tmp[8 * i + 0], mat[8 * i + 0], mat[8 * i + 4], 0x88);
    vshufi32x4(tmp[8 * i + 1], mat[8 * i + 1], mat[8 * i + 5], 0x88);
    vshufi32x4(tmp[8 * i + 2], mat[8 * i + 2], mat[8 * i + 6], 0x88);
    vshufi32x4(tmp[8 * i + 3], mat[8 * i + 3], mat[8 * i + 7], 0x88);
    vshufi32x4(tmp[8 * i + 4], mat[8 * i + 0], mat[8 * i + 4], 0xdd);
    vshufi32x4(tmp[8 * i + 5], mat[8 * i + 1], mat[8 * i + 5], 0xdd);
    vshufi32x4(tmp[8 * i + 6], mat[8 * i + 2], mat[8 * i + 6], 0xdd);
    vshufi32x4(tmp[8 * i + 7], mat[8 * i + 3], mat[8 * i + 7], 0xdd);
  }

  // last step and move out
  for (int i = 0; i < pad_n / 4; ++i) {
    if (i < N / 4)
      vshufi32x4(mat[i], tmp[i % 8], tmp[8 + i % 8], i < 8 ? 0x88 : 0xdd);
    else
      vpxord(Xbyak::Xmm(mat[i]), Xbyak::Xmm(mat[i]), Xbyak::Xmm(mat[i]));
    // move out
    vmovdqa32(ptr[dst + i * BYTES_ZMM], mat[i]);
  }
}

void jit_trans_AB16a4b::generate() {
  std::shared_ptr<void> use_loacl_label = {(inLocalLabel(), nullptr), [&](...) { outLocalLabel(); }};
  const auto m_tiles = M / 16;
  regs_pool rp(this, 1, {m_tiles <= 1 ? 3 : 5, 32, 1});

  const auto reg_src = rp.reg<Reg64>();
  const auto reg_dst = rp.reg<Reg64>();
  const auto reg_tmp = rp.reg<Reg64>();
  const auto mask_n = rp.reg<Opmask>();

  mov(reg_src, ptr[rp.p[0] + GET_OFF(src)]);
  mov(reg_dst, ptr[rp.p[0] + GET_OFF(dst)]);

  if (N != 64) {
    mov(reg_tmp, (1ULL << N) - 1ULL);
    kmovq(mask_n, reg_tmp);
  }

  if (m_tiles == 1) {
    transpose_16x16_ps(&rp, reg_src, reg_dst, N == 64 ? k0 : mask_n);
    lea(reg_src, ptr[reg_src + ld_src * 16]);
    lea(reg_dst, ptr[reg_dst + BYTES_ZMM * (pad_n / 4)]);
  } else if (m_tiles > 1) {
    Xbyak::Label m_loop;
    const auto reg_msize = rp.reg<Reg64>();
    const auto reg_iterm = rp.reg<Reg64>();
    mov(reg_msize, m_tiles);
    xor_(reg_iterm, reg_iterm);
    L(m_loop);
    transpose_16x16_ps(&rp, reg_src, reg_dst, N == 64 ? k0 : mask_n);
    lea(reg_src, ptr[reg_src + ld_src * 16]);
    lea(reg_dst, ptr[reg_dst + BYTES_ZMM * (pad_n / 4)]);
    lea(reg_iterm, ptr[reg_iterm + 1]);
    cmp(reg_iterm, reg_msize);
    jb(m_loop);
  }

  const auto m_tail = M % 16;
  if (m_tail > 0) transpose_16x16_ps(&rp, reg_src, reg_dst, N == 64 ? k0 : mask_n, true);
}

}  // namespace jd
