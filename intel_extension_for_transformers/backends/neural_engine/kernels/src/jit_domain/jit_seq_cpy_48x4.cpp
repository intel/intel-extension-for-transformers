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
#include "jit_domain/jit_seq_cpy_48x4.hpp"

#define GET_OFF(field) offsetof(jit_seq_cpy_48x4::rt_data_t, field)

namespace jd {

void jit_seq_cpy_48x4::trans_4x16(const Zmm& zreg_res, const Zmm& zreg_sum, const Xbyak::Operand& op0,
                                  const Xbyak::Operand& op1, const Xbyak::Operand& op2, const Xbyak::Operand& op3,
                                  bool is_tail_x) {
  if (!is_tail_x) {
    vmovdqu8(Xbyak::Xmm(zreg_res.getIdx()), op0);
    vbroadcasti32x4(Xbyak::Ymm(zreg_res.getIdx()) | reg_k, op1);
    vmovdqu8(Xbyak::Xmm(vreg_t1.getIdx()), op2);
    vbroadcasti32x4(Xbyak::Ymm(vreg_t1.getIdx()) | reg_k, op3);
    vpermt2d(zreg_res, vpermt2d_arg_idx, vreg_t1);
    vpshufb(zreg_res, zreg_res, vpshufb_arg_b);
  } else {
    vmovdqu8(Xbyak::Xmm(zreg_res.getIdx()) | reg_k_tail | T_z, op0);
    vmovdqu8(Xbyak::Xmm(vreg_t2.getIdx()) | reg_k_tail | T_z, op1);
    vshuff32x4(Xbyak::Ymm(zreg_res.getIdx()), Xbyak::Ymm(zreg_res.getIdx()), Xbyak::Ymm(vreg_t2.getIdx()), 0b00);
    vmovdqu8(Xbyak::Xmm(vreg_t1.getIdx()) | reg_k_tail | T_z, op2);
    vmovdqu8(Xbyak::Xmm(vreg_t2.getIdx()) | reg_k_tail | T_z, op3);
    vshuff32x4(Xbyak::Ymm(vreg_t1.getIdx()), Xbyak::Ymm(vreg_t1.getIdx()), Xbyak::Ymm(vreg_t2.getIdx()), 0b00);
    vpermt2d(zreg_res, vpermt2d_arg_idx, vreg_t1);
    vpshufb(zreg_res, zreg_res, vpshufb_arg_b);
  }
  if (sum_m) {
    Xbyak::Label L_sum_prepare_zero;
    Xbyak::Label L_sum_prepare_end;

    cmp(reg_sum_append, 0);  // if append
    je(L_sum_prepare_zero);
    if (!is_tail_x) {
      vmovdqa32(zreg_sum, zword[reg_sum]);
    } else {
      vmovdqa32(zreg_sum | reg_k_tail | T_z, zword[reg_sum]);
    }
    jmp(L_sum_prepare_end);
    L(L_sum_prepare_zero);  // if overwrite
    vxorps(Xmm(zreg_sum), Xmm(zreg_sum), Xmm(zreg_sum));
    L(L_sum_prepare_end);

    if (is_unsigned) {
      vpdpbusds(zreg_sum, zreg_res, vreg_oneb);
    } else {
      vpdpbusds(zreg_sum, vreg_oneb, zreg_res);
    }
    vmovdqa32(zword[reg_sum], zreg_sum);
    lea(reg_sum, ptr[reg_sum + BYTES_ZMM]);
  }
}

void jit_seq_cpy_48x4::generate() {
  preamble();
  inLocalLabel();  // use local label for multiple instance

  auto temp_r32 = Xbyak::Reg32(reg_tmp.getIdx());
  mov(temp_r32, 0xf0);
  kmovb(reg_k, temp_r32);
  vpmovzxbd(vpermt2d_arg_idx, ptr[rip + l_vpermt2d_control]);
  vbroadcasti32x4(vpshufb_arg_b, ptr[rip + l_vpshufb_control]);

  if (sum_m) {
    mov(reg_sum, ptr[parambase + GET_OFF(dst_sum)]);
    const Xbyak::Reg8 reg_tmp8 = reg_tmp.cvt8();
    mov(reg_tmp8, 1);  // "1" in either s8 or u8
    vpbroadcastb(vreg_oneb, reg_tmp8);
  }

  mov(reg_src, ptr[parambase + GET_OFF(src)]);
  mov(reg_dst, ptr[parambase + GET_OFF(dst)]);
  if (sum_m) mov(reg_sum_append, byte[parambase + GET_OFF(sum_append)]);

  const int N48 = N / 48 * 48;
  const int N_tail = N - N48;

  const Xbyak::Zmm& v_res0 = zmm0;
  const Xbyak::Zmm& v_res1 = zmm1;
  const Xbyak::Zmm& v_res2 = zmm2;
  const Xbyak::Zmm& v_sum0 = zmm3;
  const Xbyak::Zmm& v_sum1 = zmm4;
  const Xbyak::Zmm& v_sum2 = zmm5;
  if (N48 != 0) {
    mov(reg_nsize.cvt32(), N48);
    xor_(reg_itern, reg_itern);
    L(l_n_loop);
    // first 16x4
    RegExp s_base = reg_src;
    trans_4x16(v_res0, v_sum0, ptr[s_base], ptr[s_base + ld_src], ptr[s_base + ld_src * 2], ptr[s_base + ld_src * 3]);
    vmovdqa32(zword[reg_dst + BYTES_ZMM * 0], v_res0);  // store with non-temp mem hint

    // second 16x4
    s_base = s_base + 16;
    trans_4x16(v_res1, v_sum1, ptr[s_base], ptr[s_base + ld_src], ptr[s_base + ld_src * 2], ptr[s_base + ld_src * 3]);
    vmovdqa32(zword[reg_dst + BYTES_ZMM * 1], v_res1);

    // third 16x4
    s_base = s_base + 16;
    trans_4x16(v_res2, v_sum2, ptr[s_base], ptr[s_base + ld_src], ptr[s_base + ld_src * 2], ptr[s_base + ld_src * 3]);
    vmovdqa32(zword[reg_dst + BYTES_ZMM * 2], v_res2);

    lea(reg_src, ptr[reg_src + 48]);
    lea(reg_dst, ptr[reg_dst + stride_dst]);
    lea(reg_itern, ptr[reg_itern + 48]);
    cmp(reg_itern, reg_nsize);  // k iteration variable
    jb(l_n_loop);
  }

  if (N_tail != 0) {
    if (N_tail % 16 != 0) {
      uint32_t k_tail = (1 << (N_tail % 16)) - 1;
      mov(temp_r32, k_tail);
      kmovw(reg_k_tail, temp_r32);
    }
    RegExp s_base = reg_src;
    if (N_tail > 0) {
      trans_4x16(v_res0, v_sum0, ptr[s_base], ptr[s_base + ld_src], ptr[s_base + ld_src * 2], ptr[s_base + ld_src * 3],
                 N_tail < 16);
      vmovdqa32(zword[reg_dst + BYTES_ZMM * 0], v_res0);
    }
    if (N_tail > 16) {
      s_base = s_base + 16;
      trans_4x16(v_res1, v_sum1, ptr[s_base], ptr[s_base + ld_src], ptr[s_base + ld_src * 2], ptr[s_base + ld_src * 3],
                 N_tail < 32);
      vmovdqa32(zword[reg_dst + BYTES_ZMM * 1], v_res1);
    }
    if (N_tail > 32) {
      s_base = s_base + 16;
      trans_4x16(v_res2, v_sum2, ptr[s_base], ptr[s_base + ld_src], ptr[s_base + ld_src * 2], ptr[s_base + ld_src * 3],
                 N_tail < 48);
      vmovdqa32(zword[reg_dst + BYTES_ZMM * 2], v_res2);
    }
  }

  postamble();  // end of function

  const uint8_t vpermt2d_control[16] = {0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23};
  const uint8_t vpshufb_control[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
  L(l_vpermt2d_control);
  db(vpermt2d_control, 16);
  L(l_vpshufb_control);
  db(vpshufb_control, 16);
  outLocalLabel();  // end of local label
}

}  // namespace jd
