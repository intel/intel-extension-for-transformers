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
#include "jit_trans_BA16b4a.hpp"

#include <memory>

#define GET_OFF(field) offsetof(jit_trans_BA16b4a::rt_data_t, field)

namespace jd {

void jit_trans_BA16b4a::transpose_4x64(regs_pool* const rp, const Xbyak::Reg64& src, const Xbyak::Reg64& dst,
                                       const Xbyak::Reg64& ld_dst, const Xbyak::Reg64& ld_dst3x, bool is_tail) {
  const auto mat = rp->regs<Zmm, 4>();
  const auto tmp = rp->regs<Zmm, 4>();

  if (M == 0) {  // M == 0 is a special case where 4x64 memory are zeroed
    vxorps(Xmm(mat[0].getIdx()), Xmm(mat[0].getIdx()), Xmm(mat[0].getIdx()));
    if (N > 0 * VEC) vmovdqu8(zword[dst + 0], mat[0]);
    if (N > 1 * VEC) vmovdqu8(zword[dst + ld_dst], mat[0]);
    if (N > 2 * VEC) vmovdqu8(zword[dst + 2 * ld_dst], mat[0]);
    if (N > 3 * VEC) vmovdqu8(zword[dst + ld_dst3x], mat[0]);
    return;
  }

  for (int i = 0; i < 4; ++i) {
    // a00a01a02a03...a60a61a62a63
    // b00b01b02b03...b60b61b62b63
    // c00c01c02c03...c60c61c62c63
    // d00d01d02d03...d60d61d62d63
    if (!is_tail || M % 4 > i)
      vmovdqu8(mat[i], zword[src + i * ld_src]);
    else
      vxorps(Xmm(mat[i].getIdx()), Xmm(mat[i].getIdx()), Xmm(mat[i].getIdx()));
  }
  vpunpcklbw(tmp[0], mat[0], mat[1]);           // ab00..ab07 ab16..ab23 ab32..ab39 ab48..ab55
  vpunpckhbw(tmp[1], mat[0], mat[1]);           // ab08..ab15 ab24..ab31 ab40..ab47 ab56..ab63
  vpunpcklbw(tmp[2], mat[2], mat[3]);           // cd00..cd07 cd16..cd23 cd32..cd39 cd48..cd55
  vpunpckhbw(tmp[3], mat[2], mat[3]);           // cd08..cd15 cd24..cd31 cd40..cd47 cd56..cd63
                                                //
  vpunpcklwd(mat[0], tmp[0], tmp[2]);           // abcd00..abcd03 abcd16..abcd19 abcd32..abcd35 abcd48..abcd51
  vpunpckhwd(mat[1], tmp[0], tmp[2]);           // abcd04..abcd07 abcd20..abcd23 abcd36..abcd39 abcd52..abcd55
  vpunpcklwd(mat[2], tmp[1], tmp[3]);           // abcd08..abcd11 abcd24..abcd27 abcd40..abcd43 abcd56..abcd59
  vpunpckhwd(mat[3], tmp[1], tmp[3]);           // abcd12..abcd15 abcd28..abcd31 abcd44..abcd47 abcd60..abcd63
                                                // ; 0x88 == 0q2020 ; 0xdd == 0q3131
  vshufi32x4(tmp[0], mat[0], mat[1], 0x88);     // abcd00..abcd03 abcd16..abcd19 abcd32..abcd35 abcd48..abcd51
  vshufi32x4(tmp[1], mat[2], mat[3], 0x88);     // abcd04..abcd07 abcd20..abcd23 abcd36..abcd39 abcd52..abcd55
  vshufi32x4(tmp[2], mat[0], mat[1], 0xdd);     // abcd08..abcd11 abcd24..abcd27 abcd40..abcd43 abcd56..abcd59
  vshufi32x4(tmp[3], mat[2], mat[3], 0xdd);     // abcd12..abcd15 abcd28..abcd31 abcd44..abcd47 abcd60..abcd63
                                                //
  vshufi32x4(mat[0], tmp[0], tmp[1], 0x88);     // abcd00..abcd03 abcd04..abcd07 abcd08..abcd11 abcd12..abcd15
  vshufi32x4(mat[1], tmp[0], tmp[1], 0xdd);     // abcd32..abcd35 abcd36..abcd39 abcd40..abcd43 abcd44..abcd47
  vshufi32x4(mat[2], tmp[2], tmp[3], 0x88);     // abcd16..abcd19 abcd20..abcd23 abcd24..abcd27 abcd28..abcd31
  vshufi32x4(mat[3], tmp[2], tmp[3], 0xdd);     // abcd48..abcd51 abcd52..abcd55 abcd56..abcd59 abcd60..abcd63
                                                //
  if (N > 0 * VEC)                              //
    vmovdqu8(zword[dst + 0], mat[0]);           // abcd00..abcd03 abcd04..abcd07 abcd08..abcd11 abcd12..abcd15
  if (N > 1 * VEC)                              //
    vmovdqu8(zword[dst + ld_dst], mat[2]);      // abcd16..abcd19 abcd20..abcd23 abcd24..abcd27 abcd28..abcd31
  if (N > 2 * VEC)                              //
    vmovdqu8(zword[dst + 2 * ld_dst], mat[1]);  // abcd32..abcd35 abcd36..abcd39 abcd40..abcd43 abcd44..abcd47
  if (N > 3 * VEC)                              //
    vmovdqu8(zword[dst + ld_dst3x], mat[3]);    // abcd48..abcd51 abcd52..abcd55 abcd56..abcd59 abcd60..abcd63
}

void jit_trans_BA16b4a::generate() {
  std::shared_ptr<void> use_loacl_label = {(inLocalLabel(), nullptr), [&](...) { outLocalLabel(); }};
  const auto m_tiles = M / 4;
  regs_pool rp(this, 1, {m_tiles <= 1 ? 4 : 6, 8, 0});
  const auto reg_src = rp.reg<Reg64>();
  const auto reg_dst = rp.reg<Reg64>();
  const auto reg_ld_dst = rp.reg<Reg64>();
  const auto reg_3ld_dst = rp.reg<Reg64>();

  mov(reg_src, ptr[rp.p[0] + GET_OFF(src)]);
  mov(reg_dst, ptr[rp.p[0] + GET_OFF(dst)]);
  mov(reg_ld_dst.cvt32(), dword[rp.p[0] + GET_OFF(ld_dst)]);
  imul(reg_3ld_dst, reg_ld_dst, 3);

  if (m_tiles == 1) {
    transpose_4x64(&rp, reg_src, reg_dst, reg_ld_dst, reg_3ld_dst);
    lea(reg_src, ptr[reg_src + ld_src * 4]);
    lea(reg_dst, ptr[reg_dst + VEC * 4]);
  } else if (m_tiles > 1) {
    Xbyak::Label m_loop;
    const auto reg_msize = rp.reg<Reg64>();
    const auto reg_iterm = rp.reg<Reg64>();
    mov(reg_msize, M / 4);
    xor_(reg_iterm, reg_iterm);
    L(m_loop);
    transpose_4x64(&rp, reg_src, reg_dst, reg_ld_dst, reg_3ld_dst);
    lea(reg_src, ptr[reg_src + ld_src * 4]);
    lea(reg_dst, ptr[reg_dst + VEC * 4]);
    lea(reg_iterm, ptr[reg_iterm + 1]);
    cmp(reg_iterm, reg_msize);
    jb(m_loop);
  }

  const auto m_tail = M % 4;
  if (m_tail > 0 || M == 0)  // M == 0 is a special case where 4x64 memory are zeroed
    transpose_4x64(&rp, reg_src, reg_dst, reg_ld_dst, reg_3ld_dst, true);
}

}  // namespace jd
