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
#include "jit_seq_cpy_2x8x8.hpp"

#define GET_OFF(field) offsetof(jit_seq_cpy_2x8x8::rt_data_t, field)

namespace jd {

void jit_seq_cpy_2x8x8::generate() {
  preamble();
  inLocalLabel();  // use local label for multiple instance

  auto temp_r32 = Xbyak::Reg32(reg_tmp.getIdx());
  mov(temp_r32, 0xf0);
  kmovb(reg_k, temp_r32);
  vpmovzxbd(vpermt2d_arg_idx, ptr[rip + l_vpermt2d_control]);
  vbroadcasti32x4(vpshufb_arg_b, ptr[rip + l_vpshufb_control]);

  mov(reg_src, ptr[parambase + GET_OFF(src)]);
  mov(reg_dst, ptr[parambase + GET_OFF(dst)]);
  mov(reg_ld_dst.cvt32(), dword[parambase + GET_OFF(ld_dst)]);
  mov(reg_ld_src.cvt32(), dword[parambase + GET_OFF(ld_src)]);
  imul(reg_3ld_src, reg_ld_src, 3);

  const auto& v_res0 = zmm0;
  const auto& v_res1 = zmm1;
  if (val_offset) {
    const Xbyak::Reg8 reg_tmp8 = reg_tmp.cvt8();
    mov(reg_tmp8, val_offset);
    vpbroadcastb(vreg_val_offset, reg_tmp8);
  }

  // if (N / 16 * 16 != 0)
  mov(reg_nsize.cvt32(), dword[parambase + GET_OFF(N)]);
  and_(reg_nsize.cvt32(), -16);
  cmp(reg_nsize, 0);
  je(l_n_loop_end, T_NEAR);
  {
    xor_(reg_itern, reg_itern);
    L(l_n_loop);
    // first 16x4
    vmovdqu8(Xbyak::Xmm(v_res0.getIdx()), ptr[reg_src]);
    vbroadcasti32x4(Xbyak::Ymm(v_res0.getIdx()) | reg_k, ptr[reg_src + reg_ld_src]);
    vmovdqu8(Xbyak::Xmm(vreg_t1.getIdx()), ptr[reg_src + reg_ld_src * 2]);
    vbroadcasti32x4(Xbyak::Ymm(vreg_t1.getIdx()) | reg_k, ptr[reg_src + reg_3ld_src]);
    vpermt2d(v_res0, vpermt2d_arg_idx, vreg_t1);
    vpshufb(v_res0, v_res0, vpshufb_arg_b);
    if (val_offset) vpaddb(v_res0, v_res0, vreg_val_offset);

    // second 16x4
    lea(reg_tmp, ptr[reg_src + reg_ld_src * 4]);
    vmovdqu8(Xbyak::Xmm(v_res1.getIdx()), ptr[reg_tmp]);
    vbroadcasti32x4(Xbyak::Ymm(v_res1.getIdx()) | reg_k, ptr[reg_tmp + reg_ld_src]);
    vmovdqu8(Xbyak::Xmm(vreg_t1.getIdx()), ptr[reg_tmp + reg_ld_src * 2]);
    vbroadcasti32x4(Xbyak::Ymm(vreg_t1.getIdx()) | reg_k, ptr[reg_tmp + reg_3ld_src]);
    vpermt2d(v_res1, vpermt2d_arg_idx, vreg_t1);
    vpshufb(v_res1, v_res1, vpshufb_arg_b);
    if (val_offset) vpaddb(v_res1, v_res1, vreg_val_offset);

    // bland and store result
    vshuff32x4(vreg_t1, v_res0, v_res1, (4 << 4) | 4);
    vmovdqa32(zword[reg_dst], vreg_t1);  // store with non-temp mem hint
    vshuff32x4(v_res0, v_res0, v_res1, (14 << 4) | 14);
    vmovdqa32(zword[reg_dst + reg_ld_dst], v_res0);  // store with non-temp mem hint

    lea(reg_src, ptr[reg_src + 16]);
    lea(reg_dst, ptr[reg_dst + reg_ld_dst * 2]);
    lea(reg_itern, ptr[reg_itern + 16]);
    cmp(reg_itern, reg_nsize);  // k iteration variable
    jb(l_n_loop);
  }
  L(l_n_loop_end);

  // if (N % 16 != 0)
  mov(reg_nsize.cvt32(), dword[parambase + GET_OFF(N)]);
  and_(reg_nsize.cvt32(), 16U - 1);
  cmp(reg_nsize, 0);
  je(l_ntail_end, T_NEAR);
  {
    mov(temp_r32, 1U);                            // (1 << (N % 16)) - 1
    shlx(temp_r32, temp_r32, reg_nsize.cvt32());  // (1 << (N % 16)) - 1
    sub(temp_r32, 1);                             // (1 << (N % 16)) - 1
    kmovw(reg_k_tail, temp_r32);
    if (val_offset) vmovdqa32(vreg_val_offset | reg_k_tail | T_z, vreg_val_offset);

    vmovdqu8(Xbyak::Xmm(v_res0.getIdx()) | reg_k_tail | T_z, ptr[reg_src]);
    vmovdqu8(Xbyak::Xmm(vreg_t2.getIdx()) | reg_k_tail | T_z, ptr[reg_src + reg_ld_src]);
    vshuff32x4(Xbyak::Ymm(v_res0.getIdx()), Xbyak::Ymm(v_res0.getIdx()), Xbyak::Ymm(vreg_t2.getIdx()), 0b00);
    vmovdqu8(Xbyak::Xmm(vreg_t1.getIdx()) | reg_k_tail | T_z, ptr[reg_src + reg_ld_src * 2]);
    vmovdqu8(Xbyak::Xmm(vreg_t2.getIdx()) | reg_k_tail | T_z, ptr[reg_src + reg_3ld_src]);
    vshuff32x4(Xbyak::Ymm(vreg_t1.getIdx()), Xbyak::Ymm(vreg_t1.getIdx()), Xbyak::Ymm(vreg_t2.getIdx()), 0b00);
    vpermt2d(v_res0, vpermt2d_arg_idx, vreg_t1);
    vpshufb(v_res0, v_res0, vpshufb_arg_b);
    if (val_offset) vpaddb(v_res0, v_res0, vreg_val_offset);

    lea(reg_tmp, ptr[reg_src + reg_ld_src * 4]);
    vmovdqu8(Xbyak::Xmm(v_res1.getIdx()) | reg_k_tail | T_z, ptr[reg_tmp]);
    vmovdqu8(Xbyak::Xmm(vreg_t2.getIdx()) | reg_k_tail | T_z, ptr[reg_tmp + reg_ld_src * 1]);
    vshuff32x4(Xbyak::Ymm(v_res1.getIdx()), Xbyak::Ymm(v_res1.getIdx()), Xbyak::Ymm(vreg_t2.getIdx()), 0b00);
    vmovdqu8(Xbyak::Xmm(vreg_t1.getIdx()) | reg_k_tail | T_z, ptr[reg_tmp + reg_ld_src * 2]);
    vmovdqu8(Xbyak::Xmm(vreg_t2.getIdx()) | reg_k_tail | T_z, ptr[reg_tmp + reg_3ld_src]);
    vshuff32x4(Xbyak::Ymm(vreg_t1.getIdx()), Xbyak::Ymm(vreg_t1.getIdx()), Xbyak::Ymm(vreg_t2.getIdx()), 0b00);
    vpermt2d(v_res1, vpermt2d_arg_idx, vreg_t1);
    vpshufb(v_res1, v_res1, vpshufb_arg_b);
    if (val_offset) vpaddb(v_res1, v_res1, vreg_val_offset);

    // bland and store result
    vshuff32x4(vreg_t1, v_res0, v_res1, (4 << 4) | 4);
    vmovdqu8(zword[reg_dst], vreg_t1);

    // if (N % 16 > 8)
    cmp(reg_nsize, 8);
    jle(l_ntail_end);
    {
      vshuff32x4(v_res0, v_res0, v_res1, (14 << 4) | 14);
      vmovdqu8(zword[reg_dst + reg_ld_dst], v_res0);
    }
  }
  L(l_ntail_end);

  postamble();  // end of function

  align(16);
  const uint8_t vpermt2d_control[16] = {0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23};
  const uint8_t vpshufb_control[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
  L(l_vpermt2d_control);
  db(vpermt2d_control, 16);
  L(l_vpshufb_control);
  db(vpshufb_control, 16);
  outLocalLabel();  // end of local label
}

}  // namespace jd
