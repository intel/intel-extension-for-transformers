//  Copyright (c) 2021 Intel Corporation
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

#include "jit_domain/jit_transpose_attention.hpp"

namespace jd {
void configure_tiles(tileconfig_t* tc, int TILE_K) {
  // Filling tile configure structure. Could be done offline.
  tc->palette_id = 1;
  // Configure C tiles
  for (int t = 0; t < 4; ++t) {
    tc->rows[t] = TILE_M;
    tc->colb[t] = TILE_N * sizeof(int32_t);
  }
  // Configure A tiles
  for (int t = 4; t < 6; ++t) {
    tc->rows[t] = TILE_M;
    tc->colb[t] = TILE_K * sizeof(uint8_t);
  }
  // Configure B tile. B effectively has 64 rows and 16 columns.
  for (int t = 6; t < 8; ++t) {
    tc->rows[t] = TILE_K / KPACK;
    tc->colb[t] = TILE_N * KPACK * sizeof(int8_t);
  }
}

void MHA_stage1_kernel::packedf32_bf16(int idx0, int idx1) {
  auto f32_2_bf16 = [&](int idx) {
    if (isa_available(avx512_core_bf16)) {
      vcvtneps2bf16(Ymm(idx), Zmm(idx));
    } else {
      vpsrld(Zmm(idx), Zmm(idx), 16);
      vpmovdw(Ymm(idx), Zmm(idx));
    }
  };

  f32_2_bf16(idx0);
  f32_2_bf16(idx1);
  vshufi32x4(Zmm(idx0), Zmm(idx0), Zmm(idx1), (4 << 4) | 4);
}

void MHA_stage1_kernel::exp_f32_lowprecision(Zmm log2e, Zmm ln2, Zmm c[3], Zmm x, Zmm y, Zmm tmp[]) {
  vmulps(tmp[0], x, log2e);
  vrndscaleps(tmp[0], tmp[0], 0x2);
  auto z = tmp[0];
  vmulps(tmp[1], tmp[0], ln2);
  vsubps(tmp[1], x, tmp[1]);
  auto q = tmp[1];
  vmovaps(y, c[1]);
  vfmadd231ps(y, q, c[0]);
  vfmadd213ps(y, q, c[2]);
  vscalefps(y, y, z);
}

void MHA_stage2_kernel::loadbf16_norm_rows(int idx, RegExp load_addr, Zmm scale) {
  vpmovzxwd(Zmm(idx), yword[load_addr]);
  vpslld(Zmm(idx), Zmm(idx), 16);
  normalize(idx, scale);
}

void MHA_stage2_kernel::normalize(int idx, Zmm scale) {
  vmulps(Zmm(idx), Zmm(idx), scale);
  vcvtps2dq(Zmm(idx), Zmm(idx));
  vpmovdb(Xmm(idx), Zmm(idx));
}

void TransposeCopy8x8_1B_kernel::generate() {
  inLocalLabel();  // use local label for multiple instance

  StackFrame st(this, 1, 6, 16 * 10);
  const Xbyak::Reg64& parambase = st.p[0];
  const Xbyak::Reg64& reg_srcptr = st.t[0];
  const Xbyak::Reg64& reg_dstptr = st.t[1];
  const Xbyak::Reg64& reg_srcstep = st.t[2];
  const Xbyak::Reg64& reg_dststep = st.t[5];
  const Xbyak::Reg64& reg_tmp1 = st.t[3];
  const Xbyak::Reg64& reg_iterk = st.t[4];
  const Xbyak::Reg64& reg_ret = rax;
  for (int i = 0; i < 10; i++) {
    movaps(xword[rsp + i * 16], Xmm(6 + i));
  }
  mov(reg_srcptr, ptr[parambase + offsetof(ssd::transpose_copy_params, srcptr)]);
  mov(reg_dstptr, ptr[parambase + offsetof(ssd::transpose_copy_params, dstptr)]);
  xor_(reg_srcstep, reg_srcstep);
  mov(reg_srcstep.cvt32(), ptr[parambase + offsetof(ssd::transpose_copy_params, srcstride)]);
  xor_(reg_dststep, reg_dststep);
  mov(reg_dststep.cvt32(), ptr[parambase + offsetof(ssd::transpose_copy_params, dststride)]);

  xor_(reg_ret, reg_ret);
  std::vector<Ymm> inputs(8), tmp(8);
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs[i] = Ymm(i);
  }
  for (size_t i = 0; i < tmp.size(); i++) {
    tmp[i] = Ymm(8 + i);
  }
  xor_(reg_iterk, reg_iterk);
  L(".loop");
  mov(reg_tmp1, reg_srcptr);
  for (int ii = 0; ii < 8; ii++) {
    vpmovzxbd(Ymm(ii), ptr[reg_tmp1]);

    add(reg_tmp1, reg_srcstep);
  }
  auto outputs = transpose8x4B(inputs.data(), tmp.data());
  for (int ii = 0; ii < 8; ii++) {
    vpmovdb(ptr[reg_dstptr], outputs[ii]);
    add(reg_dstptr, reg_dststep);
  }
  add(reg_srcptr, 8);
  add(reg_iterk, 8);
  cmp(reg_iterk.cvt32(), ptr[parambase + offsetof(ssd::transpose_copy_params, k)]);
  jb(".loop");

  mov(reg_ret, 0);
  for (int i = 0; i < 10; i++) {
    movaps(Xmm(i + 6), xword[rsp + i * 16]);
  }
  outLocalLabel();  // end of local label
}

std::vector<Ymm> TransposeCopy8x8_1B_kernel::transpose8x4B(Ymm* rows, Ymm* tmp) {
  vunpcklps(tmp[0], rows[0], rows[1]);
  vunpcklps(tmp[1], rows[2], rows[3]);
  vunpckhps(tmp[2], rows[0], rows[1]);
  vunpcklps(tmp[3], rows[4], rows[5]);
  vunpcklps(rows[0], rows[6], rows[7]);

  vshufps(tmp[4], tmp[0], tmp[1], 0x4e);
  vblendps(rows[1], tmp[0], tmp[4], 0xcc);
  vshufps(tmp[0], tmp[3], rows[0], 0x4e);

  vunpckhps(tmp[5], rows[2], rows[3]);
  vblendps(rows[2], tmp[3], tmp[0], 0xCC);
  vblendps(rows[3], tmp[4], tmp[1], 0xCC);
  vperm2f128(tmp[4], rows[1], rows[2], 0x20);
  // tmp[4] ready

  vunpckhps(tmp[7], rows[4], rows[5]);
  vblendps(rows[4], tmp[0], rows[0], 0xcc);
  vunpckhps(tmp[6], rows[6], rows[7]);
  vperm2f128(rows[7], rows[3], rows[4], 0x20);
  // rows[7] ready

  vshufps(rows[5], tmp[2], tmp[5], 0x4e);
  vblendps(rows[6], rows[5], tmp[5], 0xcc);
  vshufps(tmp[5], tmp[7], tmp[6], 0x4e);
  vblendps(tmp[2], tmp[2], rows[5], 0xcc);
  vblendps(tmp[7], tmp[7], tmp[5], 0xcc);
  vperm2f128(tmp[0], tmp[2], tmp[7], 0x020);
  // tmp[0] ready

  vblendps(tmp[6], tmp[5], tmp[6], 0xcc);
  vperm2f128(tmp[5], rows[6], tmp[6], 0x20);
  // tmp[5] ready

  vperm2f128(tmp[1], rows[1], rows[2], 0x31);
  // tmp[1] ready
  vperm2f128(tmp[3], rows[3], rows[4], 0x31);
  // tmp[3] ready

  vperm2f128(tmp[7], tmp[2], tmp[7], 0x31);
  // tmp[7] ready
  vperm2f128(tmp[6], rows[6], tmp[6], 0x31);
  // tmp[6] ready
  return std::vector<Ymm>{
      tmp[4], rows[7], tmp[0], tmp[5], tmp[1], tmp[3], tmp[7], tmp[6],
  };
}

void SeqCopy_1B_avx512_Nx4_Temp::generate() {
  inLocalLabel();  // use local label for multiple instance
  typedef Xbyak::util::StackFrame StackFrame;
  StackFrame st(this, 1, 9, 16 * 10);
  const Xbyak::Reg64& parambase = st.p[0];
  const Xbyak::Reg64& reg_srcptr = st.t[0];
  const Xbyak::Reg64& reg_src1ptr = st.t[6];
  const Xbyak::Reg64& reg_srcstride = st.t[7];
  const Xbyak::Reg64& reg_dstptr = st.t[1];
  const Xbyak::Reg64& reg_ksize = st.t[2];
  const Xbyak::Reg64& reg_dststride = st.t[3];
  const Xbyak::Reg64& reg_iterk = st.t[4];
  const Xbyak::Reg64& reg_tmp = st.t[5];
  const Xbyak::Reg64& reg_ret = rax;
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(xword[rsp + i * 16], Xmm(6 + i));
  }
#endif

  mov(reg_srcptr, ptr[parambase + offsetof(ssd::seq_vnni_copy_params, srcptr)]);
  mov(reg_dstptr, ptr[parambase + offsetof(ssd::seq_vnni_copy_params, dstptr)]);
  mov(reg_ksize.cvt32(), ptr[parambase + offsetof(ssd::seq_vnni_copy_params, k)]);
  mov(reg_ksize, reg_ksize.cvt32());

  xor_(reg_iterk, reg_iterk);
  mov(reg_dststride.cvt32(), dword[parambase + offsetof(ssd::seq_vnni_copy_params, dststride)]);
  mov(reg_srcstride.cvt32(), dword[parambase + offsetof(ssd::seq_vnni_copy_params, srcstride)]);
  mov(reg_dststride, reg_dststride.cvt32());
  mov(reg_srcstride, reg_srcstride.cvt32());

  mov(reg_src1ptr, reg_srcptr);
  imul(reg_tmp, reg_srcstride, 3);
  add(reg_src1ptr, reg_tmp);
  int ZMM_Count = NTile / 16;
  assert(NTile % 16 == 0);

  L(".kloop");
  lea(reg_tmp, ptr[reg_iterk + NTile]);
  cmp(reg_tmp, reg_ksize);
  jbe(".main");
  mov(reg_tmp, reg_ksize);
  sub(reg_tmp, 64);
  sub(reg_iterk, reg_tmp);
  sub(reg_srcptr, reg_iterk);
  sub(reg_src1ptr, reg_iterk);
  mov(reg_iterk, reg_ksize);

  L(".main");
  int idx = 0;
  for (int i = 0; i < ZMM_Count; i++) {
    idx = (i % 2) * 6;
    vmovups(Xmm(idx + 0), yword[reg_srcptr + i * 16]);
    vmovups(Xmm(idx + 1), yword[reg_srcptr + reg_srcstride * 1 + i * 16]);
    vmovups(Xmm(idx + 2), yword[reg_srcptr + reg_srcstride * 2 + i * 16]);
    vmovups(Xmm(idx + 3), yword[reg_src1ptr + i * 16]);
    vnni_interleave_load_6regs(idx);
    vmovups(zword[reg_dstptr + 64 * i], Zmm(idx));
  }

  add(reg_dstptr, reg_dststride);

  add(reg_srcptr, NTile);
  add(reg_src1ptr, NTile);
  add(reg_iterk, NTile);

  cmp(reg_iterk, reg_ksize);  // k iteration variable
  jb(".kloop");

  L(".ret");
  mov(reg_ret, 0);
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(Xmm(i + 6), xword[rsp + i * 16]);
  }
#endif
  outLocalLabel();  // end of local label
}

void SeqCopy_1B_avx512_Nx4_Temp::vnni_interleave_load_6regs(int startIdx) {
  vpunpcklbw(Xmm(startIdx + 4), Xmm(startIdx + 0), Xmm(startIdx + 1));
  vpunpckhbw(Xmm(startIdx + 5), Xmm(startIdx + 0), Xmm(startIdx + 1));
  vpunpcklbw(Xmm(startIdx + 0), Xmm(startIdx + 2), Xmm(startIdx + 3));
  vpunpckhbw(Xmm(startIdx + 1), Xmm(startIdx + 2), Xmm(startIdx + 3));

  vpunpcklwd(Xmm(startIdx + 2), Xmm(startIdx + 4), Xmm(startIdx + 0));
  vpunpckhwd(Xmm(startIdx + 3), Xmm(startIdx + 4), Xmm(startIdx + 0));
  vperm2f128(Ymm(startIdx + 2), Ymm(startIdx + 2), Ymm(startIdx + 3), 32);

  vpunpcklwd(Xmm(startIdx + 0), Xmm(startIdx + 5), Xmm(startIdx + 1));
  vpunpckhwd(Xmm(startIdx + 4), Xmm(startIdx + 5), Xmm(startIdx + 1));
  vperm2f128(Ymm(startIdx + 0), Ymm(startIdx + 0), Ymm(startIdx + 4), 32);

  vshuff32x4(Zmm(startIdx + 0), Zmm(startIdx + 2), Zmm(startIdx + 0), (4 << 4) | 4);
}

void SeqCopy_1B_avx512_Nx2_Temp::generate() {
  inLocalLabel();  // use local label for multiple instance

  StackFrame st(this, 1, 9, 16 * 10);
  const Xbyak::Reg64& parambase = st.p[0];
  const Xbyak::Reg64& reg_srcptr = st.t[0];
  const Xbyak::Reg64& reg_srcstride = st.t[7];
  const Xbyak::Reg64& reg_dstptr = st.t[1];
  const Xbyak::Reg64& reg_ksize = st.t[2];
  const Xbyak::Reg64& reg_dststride = st.t[3];
  const Xbyak::Reg64& reg_iterk = st.t[4];
  const Xbyak::Reg64& reg_ret = rax;
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(xword[rsp + i * 16], Xmm(6 + i));
  }
#endif

  mov(reg_srcptr, ptr[parambase + offsetof(ssd::seq_vnni_copy_params, srcptr)]);
  mov(reg_dstptr, ptr[parambase + offsetof(ssd::seq_vnni_copy_params, dstptr)]);
  mov(reg_ksize.cvt32(), ptr[parambase + offsetof(ssd::seq_vnni_copy_params, k)]);
  mov(reg_ksize, reg_ksize.cvt32());

  xor_(reg_iterk, reg_iterk);
  mov(reg_dststride.cvt32(), dword[parambase + offsetof(ssd::seq_vnni_copy_params, dststride)]);
  mov(reg_srcstride.cvt32(), dword[parambase + offsetof(ssd::seq_vnni_copy_params, srcstride)]);
  mov(reg_dststride, reg_dststride.cvt32());
  mov(reg_srcstride, reg_srcstride.cvt32());

  int ZMM_Count = NTile / 32;
  assert(NTile % 32 == 0);

  L(".kloop");
  int idx = 0;
  for (int i = 0; i < ZMM_Count; i++) {
    idx = (i % 2) * 3;
    vmovups(Ymm(idx + 0), yword[reg_srcptr + i * 32]);
    vmovups(Ymm(idx + 1), yword[reg_srcptr + reg_srcstride * 1 + i * 32]);
    vnni_word_interleave_load_3regs(idx);
    vmovups(zword[reg_dstptr + 64 * i], Zmm(idx));
  }
  add(reg_dstptr, reg_dststride);
  add(reg_srcptr, NTile);
  add(reg_iterk, NTile);
  cmp(reg_iterk, reg_ksize);  // k iteration variable
  jb(".kloop");

  mov(reg_ret, 0);
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(Xmm(i + 6), xword[rsp + i * 16]);
  }
#endif
  outLocalLabel();  // end of local label
}

void SeqCopy_1B_avx512_Nx2_Temp::vnni_word_interleave_load_3regs(int startIdx) {
  vpunpcklbw(Ymm(startIdx + 2), Ymm(startIdx + 0), Ymm(startIdx + 1));
  vpunpckhbw(Ymm(startIdx + 0), Ymm(startIdx + 0), Ymm(startIdx + 1));
  vshuff32x4(Zmm(startIdx + 0), Zmm(startIdx + 2), Zmm(startIdx + 0), (4 << 4) | 4);
  vshuff32x4(Zmm(startIdx + 0), Zmm(startIdx + 0), Zmm(startIdx + 0), 216);
}

void MHA_s8s8s8_row_amx_32x32_batchk_binary_exp::generate() {
  configure_tiles(const_cast<tileconfig_t*>(&tc), KTile);
  inLocalLabel();  // use local label for multiple instance
  int XmmReserve = 16 * (10 + 2);
  int TmmReserve = MTile * NTile * 4 * BatchK;
  int TmpValueReserve = 64;
  int TmpSpace = XmmReserve + TmmReserve + TmpValueReserve;
  int TTmmStart = XmmReserve;
  Xbyak::Label tmpfvariable;
  {
    StackFrame st(this, 1, 11, TmpSpace);
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(xword[rsp + i * 16], Xmm(6 + i));
    }
#endif
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_matAptr = st.t[0];
    const Xbyak::Reg64& reg_matBptr = st.t[1];
    const Xbyak::Reg64& reg_astep = st.t[2];
    const Xbyak::Reg64& reg_cstep = st.t[9];
    const Xbyak::Reg64& reg_matCptr = st.t[3];
    const Xbyak::Reg64& reg_matDptr = st.t[4];
    const Xbyak::Reg64& reg_tmp = st.t[5];
    const Xbyak::Reg64& reg_sumptr = st.t[6];
    const Xbyak::Reg64& reg_batch = st.t[7];
    const Xbyak::Reg64& reg_TmpPtr = st.t[8];
    const Xbyak::Reg64& reg_iterm = st.t[10];
    const Xbyak::Reg64& reg_ret = rax;

    mov(reg_tmp, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, cfg)]);
    ldtilecfg(ptr[reg_tmp]);

    int ZIDX_ScaleAB = 31;
    vbroadcastss(Zmm(ZIDX_ScaleAB), ptr[parambase + offsetof(ssd::transpose_attention_step1_params, scaleAB)]);

    int ZIDX_LOG2E = 30;
    vbroadcastss(Zmm(ZIDX_LOG2E), ptr[rip + tmpfvariable]);
    int ZIDX_LN2 = 29;
    vbroadcastss(Zmm(ZIDX_LN2), ptr[rip + tmpfvariable + 4]);
    int ZIDX_C0 = 28;
    vbroadcastss(Zmm(ZIDX_C0), ptr[rip + tmpfvariable + 8]);
    int ZIDX_C1 = 27;
    vbroadcastss(Zmm(ZIDX_C1), ptr[rip + tmpfvariable + 12]);
    int ZIDX_C2 = 26;
    vbroadcastss(Zmm(ZIDX_C2), ptr[rip + tmpfvariable + 16]);
    int ZIDX_TMP = 24;
    Zmm c[] = {
        Zmm(ZIDX_C0),
        Zmm(ZIDX_C1),
        Zmm(ZIDX_C2),
    };
    Zmm tmp[] = {Zmm(ZIDX_TMP), Zmm(ZIDX_TMP + 1)};
    int ZIDX_FF = 23;
    vbroadcastss(Zmm(ZIDX_FF), ptr[rip + tmpfvariable + 20]);

    mov(reg_matBptr, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, matB)]);
    mov(reg_sumptr, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, expsum)]);
    int ZIDX_ExpSum = 8;
    xor_(reg_astep, reg_astep);
    mov(reg_astep.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, astep)]);

    xor_(reg_cstep, reg_cstep);
    mov(reg_cstep.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, cstep)]);

    xor_(reg_batch, reg_batch);
    int CMTile = 2;
    int const ZMM_PerROW = NTile / 16;

    mov(reg_matCptr, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, matC)]);

    L(".batchkloop");
    mov(reg_matAptr, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, matA)]);
    imul(reg_tmp, reg_batch, KTile);
    add(reg_matAptr, reg_tmp);
    mov(reg_matDptr, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, matD)]);
    for (int i = 0; i < ZMM_PerROW; i++) {
      vxorps(Zmm(ZIDX_ExpSum + i), Zmm(ZIDX_ExpSum + i));
    }
    xor_(reg_iterm, reg_iterm);

    mov(reg_tmp, NTile * 4);  // b,c stride
    tileloaddt1(Tmm(6), ptr[reg_matBptr + reg_tmp + 0]);
    tileloaddt1(Tmm(7), ptr[reg_matBptr + reg_tmp + 64]);
    add(reg_matBptr, NTile * KTile);

    L(".mloop");
    tileloadd(Tmm(4), ptr[reg_matAptr + reg_astep]);
    imul(reg_tmp, reg_astep, 16);
    lea(reg_TmpPtr, ptr[reg_matAptr + reg_tmp]);
    for (int i = 0; i < 4; i++) {
      tilezero(Tmm(i));
    }
    tdpbssd(Tmm(0), Tmm(4), Tmm(6));
    tileloadd(Tmm(5), ptr[reg_TmpPtr + reg_astep]);
    tdpbssd(Tmm(1), Tmm(4), Tmm(7));
    tdpbssd(Tmm(2), Tmm(5), Tmm(6));
    tdpbssd(Tmm(3), Tmm(5), Tmm(7));
    mov(reg_tmp, NTile * 4);  // b,c stride
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        tilestored(ptr[rsp + reg_tmp + TTmmStart + i * 16 * 64 * 2 + j * 64], Tmm(i * 2 + j));
      }
    }

    lea(reg_TmpPtr, ptr[rsp + TTmmStart]);
    for (int tt = 0; tt < MTile; tt += CMTile) {
      for (int i = 0; i < CMTile; i++) {
        for (int j = 0; j < ZMM_PerROW; j++) {
          vcvtdq2ps(Zmm(i * ZMM_PerROW + j), ptr[reg_TmpPtr + j * 64]);
          vmulps(Zmm(i * ZMM_PerROW + j), Zmm(ZIDX_ScaleAB));
          vaddps(Zmm(i * ZMM_PerROW + j), zword_b[reg_matDptr + (tt + i) * sizeof(float)]);
          exp_f32_lowprecision(Zmm(ZIDX_LOG2E), Zmm(ZIDX_LN2), c, Zmm(i * ZMM_PerROW + j), Zmm(i * ZMM_PerROW + j),
                               tmp);
          vaddps(Zmm(ZIDX_ExpSum + j), Zmm(ZIDX_ExpSum + j), Zmm(i * ZMM_PerROW + j));
        }
        for (int j = 0; j < ZMM_PerROW; j += 2) {
          packedf32_bf16(i * ZMM_PerROW + j, i * ZMM_PerROW + j + 1);
          vmovups(ptr[reg_matCptr + j * 32], Zmm(i * ZMM_PerROW + j));
        }
        add(reg_TmpPtr, NTile * 4);
        add(reg_matCptr, reg_cstep);
      }
    }
    add(reg_iterm, MTile);
    imul(reg_tmp, reg_astep, MTile);
    lea(reg_matAptr, ptr[reg_matAptr + reg_tmp]);
    add(reg_matDptr, MTile * sizeof(float));
    cmp(reg_iterm.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, m)]);
    jb(".mloop");
    for (int i = 0; i < ZMM_PerROW; i++) {
      vrcp14ps(Zmm(ZIDX_ExpSum + i), Zmm(ZIDX_ExpSum + i));
      vmulps(Zmm(ZIDX_ExpSum + i), Zmm(ZIDX_FF));
      vmovups(ptr[reg_sumptr + i * 64], Zmm(ZIDX_ExpSum + i));
    }
    mov(reg_tmp.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, sumstep)]);
    add(reg_sumptr, reg_tmp.cvt32());
    add(reg_batch, 1);
    cmp(reg_batch.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, batchk)]);
    jb(".batchkloop");

    mov(reg_ret, 0);
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(Xmm(i + 6), xword[rsp + i * 16]);
    }
#endif
  }
  outLocalLabel();  // end of local label
  L(tmpfvariable);
  const float tmparr[] = {1.442695f, 0.693147180f, 0.35815147f, 0.96963238f, 1.0f, 255.f};
  db((uint8_t*)tmparr, sizeof(tmparr));
}

void MHA_s8s8s8_row_vnni_8x32_batchk_binary_exp::generate() {
  int const ZMM_PerROW = NTile / 16;

  inLocalLabel();  // use local label for multiple instance
  int XmmReserve = 16 * (10 + 2);
  int TmmReserve = 0;
  int TmpValueReserve = 64;
  int TmpSpace = XmmReserve + TmmReserve + TmpValueReserve;
  Xbyak::Label tmpfvariable;
  {
    StackFrame st(this, 1, 12, TmpSpace);
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(xword[rsp + i * 16], Xmm(6 + i));
    }
#endif
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_matAptr = st.t[0];
    const Xbyak::Reg64& reg_matBptr = st.t[1];
    const Xbyak::Reg64& reg_astep = st.t[2];
    const Xbyak::Reg64& reg_cstep = st.t[9];
    const Xbyak::Reg64& reg_matCptr = st.t[3];
    const Xbyak::Reg64& reg_matDptr = st.t[4];
    const Xbyak::Reg64& reg_tmp = st.t[5];
    const Xbyak::Reg64& reg_sumptr = st.t[6];
    const Xbyak::Reg64& reg_batch = st.t[7];
    const Xbyak::Reg64& reg_TmpPtr = st.t[8];
    const Xbyak::Reg64& reg_iterm = st.t[10];
    const Xbyak::Reg64& reg_TmpPtr1 = st.t[11];
    const Xbyak::Reg64& reg_ret = rax;

    int ZIDX_ScaleAB = 31;
    vbroadcastss(Zmm(ZIDX_ScaleAB), ptr[parambase + offsetof(ssd::transpose_attention_step1_params, scaleAB)]);

    int ZIDX_LOG2E = 30;
    vbroadcastss(Zmm(ZIDX_LOG2E), ptr[rip + tmpfvariable]);
    int ZIDX_LN2 = 29;
    vbroadcastss(Zmm(ZIDX_LN2), ptr[rip + tmpfvariable + 4]);
    int ZIDX_C0 = 28;
    vbroadcastss(Zmm(ZIDX_C0), ptr[rip + tmpfvariable + 8]);
    int ZIDX_C1 = 27;
    vbroadcastss(Zmm(ZIDX_C1), ptr[rip + tmpfvariable + 12]);
    int ZIDX_C2 = 26;
    vbroadcastss(Zmm(ZIDX_C2), ptr[rip + tmpfvariable + 16]);
    int ZIDX_TMP = 24;
    Zmm c[] = {
        Zmm(ZIDX_C0),
        Zmm(ZIDX_C1),
        Zmm(ZIDX_C2),
    };
    Zmm tmp[] = {Zmm(ZIDX_TMP), Zmm(ZIDX_TMP + 1)};
    int ZIDX_FF = 23;
    vbroadcastss(Zmm(ZIDX_FF), ptr[rip + tmpfvariable + 20]);

    // C reg 16 A reg 1 B reg 2: 19
    // expsum reg 2
    int ZIDX_ExpSum = 20;
    int ZIDX_CReg = 0;
    int CReg_Count = MTile * ZMM_PerROW;
    int ZIDX_AReg = 16;
    int ZIDX_BReg = 18;

    mov(reg_matBptr, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, matB)]);
    mov(reg_sumptr, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, expsum)]);
    xor_(reg_astep, reg_astep);
    mov(reg_astep.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, astep)]);

    xor_(reg_cstep, reg_cstep);
    mov(reg_cstep.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, cstep)]);

    xor_(reg_batch, reg_batch);

    mov(reg_matCptr, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, matC)]);

    L(".batchkloop");
    mov(reg_matAptr, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, matA)]);
    mov(reg_tmp, reg_batch);
    imul(reg_tmp.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, k)]);
    add(reg_matAptr, reg_tmp);
    mov(reg_matDptr, ptr[parambase + offsetof(ssd::transpose_attention_step1_params, matD)]);
    for (int i = 0; i < ZMM_PerROW; i++) {
      vxorps(Zmm(ZIDX_ExpSum + i), Zmm(ZIDX_ExpSum + i));
    }
    xor_(reg_iterm, reg_iterm);

    L(".mloop");
    lea(reg_TmpPtr1, ptr[reg_matBptr]);
    for (int i = 0; i < CReg_Count; i++) {
      vxorps(Zmm(ZIDX_CReg + i), Zmm(ZIDX_CReg + i));
    }
    xor_(reg_tmp, reg_tmp);
    L(".kloop");
    for (int i = 0; i < KTile; i += 2) {
      vpmovsxbw(Zmm(ZIDX_BReg + 0), ptr[reg_TmpPtr1 + 0]);  // s8 => s16, src 32*2 s8
      vpmovsxbw(Zmm(ZIDX_BReg + 1), ptr[reg_TmpPtr1 + 32]);
      lea(reg_TmpPtr, ptr[reg_matAptr + reg_tmp + i]);
      for (int j = 0; j < MTile; j++) {
        vpbroadcastw(Ymm(ZIDX_AReg), ptr[reg_TmpPtr]);
        vpmovsxbw(Zmm(ZIDX_AReg), Ymm(ZIDX_AReg));
        vpdpwssds(Zmm(ZIDX_CReg + j * 2 + 0), Zmm(ZIDX_AReg), Zmm(ZIDX_BReg + 0));
        vpdpwssds(Zmm(ZIDX_CReg + j * 2 + 1), Zmm(ZIDX_AReg), Zmm(ZIDX_BReg + 1));
        add(reg_TmpPtr, reg_astep);
      }
      add(reg_TmpPtr1, 32 * 2);
    }

    add(reg_tmp, KTile);
    cmp(reg_tmp.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, k)]);
    jb(".kloop");
    for (int i = 0; i < MTile; i++) {
      for (int j = 0; j < ZMM_PerROW; j++) {
        vcvtdq2ps(Zmm(ZIDX_CReg + i * ZMM_PerROW + j), Zmm(ZIDX_CReg + i * ZMM_PerROW + j));
        vmulps(Zmm(ZIDX_CReg + i * ZMM_PerROW + j), Zmm(ZIDX_ScaleAB));
        vaddps(Zmm(ZIDX_CReg + i * ZMM_PerROW + j), zword_b[reg_matDptr + i * sizeof(float)]);
        exp_f32_lowprecision(Zmm(ZIDX_LOG2E), Zmm(ZIDX_LN2), c, Zmm(ZIDX_CReg + i * ZMM_PerROW + j),
                             Zmm(ZIDX_CReg + i * ZMM_PerROW + j), tmp);
        vaddps(Zmm(ZIDX_ExpSum + j), Zmm(ZIDX_ExpSum + j), Zmm(ZIDX_CReg + i * ZMM_PerROW + j));
      }
      for (int j = 0; j < ZMM_PerROW; j += 2) {
        packedf32_bf16(ZIDX_CReg + i * ZMM_PerROW + j, ZIDX_CReg + i * ZMM_PerROW + j + 1);
        vmovups(ptr[reg_matCptr + j * 32], Zmm(ZIDX_CReg + i * ZMM_PerROW + j));
      }
      add(reg_matCptr, reg_cstep);
    }
    add(reg_iterm, MTile);
    imul(reg_tmp, reg_astep, MTile);
    lea(reg_matAptr, ptr[reg_matAptr + reg_tmp]);
    add(reg_matDptr, MTile * sizeof(float));
    cmp(reg_iterm.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, m)]);
    jb(".mloop");
    mov(reg_matBptr, reg_TmpPtr1);
    for (int i = 0; i < ZMM_PerROW; i++) {
      vrcp14ps(Zmm(ZIDX_ExpSum + i), Zmm(ZIDX_ExpSum + i));
      vmulps(Zmm(ZIDX_ExpSum + i), Zmm(ZIDX_FF));
      vmovups(ptr[reg_sumptr + i * 64], Zmm(ZIDX_ExpSum + i));
    }
    mov(reg_tmp.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, sumstep)]);
    add(reg_sumptr, reg_tmp.cvt32());
    add(reg_batch, 1);
    cmp(reg_batch.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step1_params, batchk)]);
    jb(".batchkloop");

    mov(reg_ret, 0);
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(Xmm(i + 6), xword[rsp + i * 16]);
    }
#endif
  }
  outLocalLabel();  // end of local label
  L(tmpfvariable);
  const float tmparr[] = {1.442695f, 0.693147180f, 0.35815147f, 0.96963238f, 1.0f, 255.f};
  db((uint8_t*)tmparr, sizeof(tmparr));
}

void MHA_norm_quantize_reorder_prescale_packed::generate() {
  inLocalLabel();  // use local label for multiple instance
  int SF_TmpSize = 64;
  StackFrame st(this, 1, 10, 16 * 10 + SF_TmpSize);
  const Xbyak::Reg64& parambase = st.p[0];
  const Xbyak::Reg64& reg_srcptr = st.t[0];
  const Xbyak::Reg64& reg_src1ptr = st.t[6];
  const Xbyak::Reg64& reg_srcstride = st.t[7];
  const Xbyak::Reg64& reg_dstptr = st.t[1];
  const Xbyak::Reg64& reg_sumptr = st.t[9];
  const Xbyak::Reg64& reg_ksize = st.t[2];
  const Xbyak::Reg64& reg_dststride = st.t[3];
  const Xbyak::Reg64& reg_iterk = st.t[4];
  const Xbyak::Reg64& reg_tmp = st.t[5];
  const Xbyak::Reg64& reg_tmp1 = st.t[8];
  const Xbyak::Reg64& reg_ret = rax;
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(xword[rsp + i * 16], Xmm(6 + i));
  }
#endif

  mov(reg_srcptr, ptr[parambase + offsetof(ssd::transpose_attention_step2_params, srcptr)]);
  mov(reg_dstptr, ptr[parambase + offsetof(ssd::transpose_attention_step2_params, dstptr)]);
  mov(reg_sumptr, ptr[parambase + offsetof(ssd::transpose_attention_step2_params, sumptr)]);
  mov(reg_ksize.cvt32(), ptr[parambase + offsetof(ssd::transpose_attention_step2_params, k)]);
  mov(reg_ksize, reg_ksize.cvt32());

  xor_(reg_iterk, reg_iterk);
  mov(reg_dststride.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step2_params, dststride)]);
  mov(reg_srcstride.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step2_params, srcstride)]);
  mov(reg_dststride, reg_dststride.cvt32());
  mov(reg_srcstride, reg_srcstride.cvt32());

  int ZMM_Count = NTile / 16;
  assert(NTile % 16 == 0);
  int ZMM_Unloop = NPacked / 4;

  L(".kloop");
  int ZIDX_Scale = 31;
  int constexpr SRC_ELESIZE = sizeof(uint16_t);
  int constexpr SUM_ELESIZE = sizeof(float);
  int constexpr SRC_STEP = 16 * SRC_ELESIZE;
  int idx = 0;

  for (int iloop = 0; iloop < ZMM_Unloop; iloop++) {
    if (iloop == 0) {
      mov(reg_tmp, 0);
    } else {
      imul(reg_tmp, reg_srcstride, 4 * iloop);
    }
    lea(reg_tmp1, ptr[reg_srcptr + reg_tmp]);
    imul(reg_tmp, reg_srcstride, 3);
    lea(reg_src1ptr, ptr[reg_tmp1 + reg_tmp]);
    for (int i = 0; i < ZMM_Count; i++) {
      idx = (i % 2) * 6;
      vmovups(Zmm(ZIDX_Scale), ptr[reg_sumptr + 64 * i]);
      loadbf16_norm_rows(idx, reg_tmp1 + SRC_STEP * i, Zmm(ZIDX_Scale));
      loadbf16_norm_rows(idx + 1, reg_tmp1 + reg_srcstride + SRC_STEP * i, Zmm(ZIDX_Scale));
      loadbf16_norm_rows(idx + 2, reg_tmp1 + 2 * reg_srcstride + SRC_STEP * i, Zmm(ZIDX_Scale));
      loadbf16_norm_rows(idx + 3, reg_src1ptr + SRC_STEP * i, Zmm(ZIDX_Scale));
      vnni_interleave_load_6regs(idx);
      vmovups(zword[reg_dstptr + 64 * i + iloop * NTile * 4], Zmm(idx));
    }
  }
  add(reg_dstptr, reg_dststride);
  add(reg_sumptr, NTile * SUM_ELESIZE);
  add(reg_srcptr, NTile * SRC_ELESIZE);

  add(reg_iterk, NTile);
  cmp(reg_iterk, reg_ksize);  // k iteration variable
  jb(".kloop");

  L(".ret");
  mov(reg_ret, 0);
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(Xmm(i + 6), xword[rsp + i * 16]);
  }
#endif
  outLocalLabel();  // end of local label
}

void MHA_norm_quantize_reorder_vnniw_prescale_packed::generate() {
  inLocalLabel();  // use local label for multiple instance
  int SF_TmpSize = 64;
  StackFrame st(this, 1, 9, 16 * 10 + SF_TmpSize);
  const Xbyak::Reg64& parambase = st.p[0];
  const Xbyak::Reg64& reg_srcptr = st.t[0];
  const Xbyak::Reg64& reg_srcstride = st.t[7];
  const Xbyak::Reg64& reg_dstptr = st.t[1];
  const Xbyak::Reg64& reg_sumptr = st.t[6];
  const Xbyak::Reg64& reg_ksize = st.t[2];
  const Xbyak::Reg64& reg_dststride = st.t[3];
  const Xbyak::Reg64& reg_iterk = st.t[4];
  const Xbyak::Reg64& reg_tmp = st.t[5];
  const Xbyak::Reg64& reg_tmp1 = st.t[8];
  const Xbyak::Reg64& reg_ret = rax;
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(xword[rsp + i * 16], Xmm(6 + i));
  }
#endif

  mov(reg_srcptr, ptr[parambase + offsetof(ssd::transpose_attention_step2_params, srcptr)]);
  mov(reg_dstptr, ptr[parambase + offsetof(ssd::transpose_attention_step2_params, dstptr)]);
  mov(reg_sumptr, ptr[parambase + offsetof(ssd::transpose_attention_step2_params, sumptr)]);
  mov(reg_ksize.cvt32(), ptr[parambase + offsetof(ssd::transpose_attention_step2_params, k)]);
  mov(reg_ksize, reg_ksize.cvt32());

  xor_(reg_iterk, reg_iterk);
  mov(reg_dststride.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step2_params, dststride)]);
  mov(reg_srcstride.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step2_params, srcstride)]);
  mov(reg_dststride, reg_dststride.cvt32());
  mov(reg_srcstride, reg_srcstride.cvt32());

  int ZMM_Count = NTile / 16;
  assert(NTile % 16 == 0);
  int constexpr RowPerLoop = 2;
  int ZMM_Unloop = NPacked / RowPerLoop;

  L(".kloop");
  int ZIDX_Scale = 31;
  int constexpr SRC_ELESIZE = sizeof(uint16_t);
  int constexpr SUM_ELESIZE = sizeof(float);
  int constexpr SRC_STEP = 16 * SRC_ELESIZE;
  int idx = 0;

  for (int iloop = 0; iloop < ZMM_Unloop; iloop++) {
    if (iloop == 0) {
      mov(reg_tmp, 0);
    } else {
      imul(reg_tmp, reg_srcstride, RowPerLoop * iloop);
    }
    lea(reg_tmp1, ptr[reg_srcptr + reg_tmp]);
    for (int i = 0; i < ZMM_Count; i++) {
      idx = (i % 4) * 3;
      vmovups(Zmm(ZIDX_Scale), ptr[reg_sumptr + 64 * i]);
      loadbf16_norm_rows(idx, reg_tmp1 + SRC_STEP * i, Zmm(ZIDX_Scale));
      loadbf16_norm_rows(idx + 1, reg_tmp1 + reg_srcstride + SRC_STEP * i, Zmm(ZIDX_Scale));
      vnni_word_interleave_load_3regs(idx);
      vmovups(zword[reg_dstptr + 32 * i + iloop * NTile * RowPerLoop], Ymm(idx));
    }
  }
  add(reg_dstptr, reg_dststride);
  add(reg_sumptr, NTile * SUM_ELESIZE);
  add(reg_srcptr, NTile * SRC_ELESIZE);

  add(reg_iterk, NTile);
  cmp(reg_iterk, reg_ksize);  // k iteration variable
  jb(".kloop");

  L(".ret");
  mov(reg_ret, 0);
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(Xmm(i + 6), xword[rsp + i * 16]);
  }
#endif
  outLocalLabel();  // end of local label
}

void MHA_norm_quantize_reorder_prescale_packed::vnni_interleave_load_6regs(int startIdx) {
  vpunpcklbw(Xmm(startIdx + 4), Xmm(startIdx + 0), Xmm(startIdx + 1));
  vpunpckhbw(Xmm(startIdx + 5), Xmm(startIdx + 0), Xmm(startIdx + 1));
  vpunpcklbw(Xmm(startIdx + 0), Xmm(startIdx + 2), Xmm(startIdx + 3));
  vpunpckhbw(Xmm(startIdx + 1), Xmm(startIdx + 2), Xmm(startIdx + 3));

  vpunpcklwd(Xmm(startIdx + 2), Xmm(startIdx + 4), Xmm(startIdx + 0));
  vpunpckhwd(Xmm(startIdx + 3), Xmm(startIdx + 4), Xmm(startIdx + 0));
  vperm2f128(Ymm(startIdx + 2), Ymm(startIdx + 2), Ymm(startIdx + 3), 32);

  vpunpcklwd(Xmm(startIdx + 0), Xmm(startIdx + 5), Xmm(startIdx + 1));
  vpunpckhwd(Xmm(startIdx + 4), Xmm(startIdx + 5), Xmm(startIdx + 1));
  vperm2f128(Ymm(startIdx + 0), Ymm(startIdx + 0), Ymm(startIdx + 4), 32);

  vshuff32x4(Zmm(startIdx + 0), Zmm(startIdx + 2), Zmm(startIdx + 0), (4 << 4) | 4);
}

void MHA_norm_quantize_reorder_vnniw_prescale_packed::vnni_word_interleave_load_3regs(int startIdx) {
  vpunpcklbw(Xmm(startIdx + 2), Xmm(startIdx + 0), Xmm(startIdx + 1));
  vpunpckhbw(Xmm(startIdx + 0), Xmm(startIdx + 0), Xmm(startIdx + 1));
  vshuff32x4(Ymm(startIdx + 0), Ymm(startIdx + 2), Ymm(startIdx + 0), (4 << 4) | 4);
}

void MHA_Matmul_s8u8u8_amx_32x32::generate() {
  configure_tiles(const_cast<tileconfig_t*>(&tc), KTile);
  inLocalLabel();  // use local label for multiple instance
  int XmmReserve = 16 * (10 + 2);
  int TmmReserve = MTile * NTile * 4;
  int TmpValueReserve = 64;
  int TmpSpace = XmmReserve + TmmReserve + TmpValueReserve;
  int TTmmStart = XmmReserve;
  int TValueStart = TTmmStart + TmmReserve;
  StackFrame st(this, 1, 7, TmpSpace);
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(xword[rsp + i * 16], Xmm(6 + i));
  }
#endif
  const Xbyak::Reg64& parambase = st.p[0];
  const Xbyak::Reg64& reg_matAptr = st.t[0];
  const Xbyak::Reg64& reg_matBptr = st.t[1];
  const Xbyak::Reg64& reg_matCptr = st.t[1];
  const Xbyak::Reg64& reg_astep = st.t[2];
  const Xbyak::Reg64& reg_cstep = st.t[2];
  const Xbyak::Reg64& reg_ksize = st.t[3];
  const Xbyak::Reg64& reg_iterk = st.t[4];
  const Xbyak::Reg64& reg_tmp = st.t[5];
  const Xbyak::Reg64& reg_matA1ptr = st.t[6];
  const Xbyak::Reg64& reg_ret = rax;

  mov(reg_tmp, ptr[parambase + offsetof(ssd::transpose_attention_step3_params, cfg)]);
  ldtilecfg(ptr[reg_tmp]);

  mov(reg_tmp.cvt32(), 0xff);
  mov(ptr[rsp + TValueStart], reg_tmp.cvt32());

  xor_(reg_ksize, reg_ksize);
  mov(reg_ksize.cvt32(), ptr[parambase + offsetof(ssd::transpose_attention_step3_params, k)]);
  xor_(reg_astep, reg_astep);
  mov(reg_astep.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step3_params, astep)]);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      tilezero(Tmm(i * 2 + j));
    }
  }
  mov(reg_matAptr, ptr[parambase + offsetof(ssd::transpose_attention_step3_params, matA)]);
  mov(reg_tmp, reg_astep);
  shl(reg_tmp, 4);
  lea(reg_matA1ptr, ptr[reg_matAptr + reg_tmp]);
  mov(reg_matBptr, ptr[parambase + offsetof(ssd::transpose_attention_step3_params, matB)]);
  xor_(reg_iterk, reg_iterk);

  tileloadd(Tmm(4), ptr[reg_matAptr + reg_astep]);
  add(reg_matAptr, KTile);
  mov(reg_tmp, NTile * 4);  // b,c stride
  tileloadd(Tmm(6), ptr[reg_matBptr + reg_tmp + 0]);
  L(".kloop");
  tileloadd(Tmm(7), ptr[reg_matBptr + reg_tmp + 64]);
  add(reg_matBptr, NTile * KTile);
  tdpbsud(Tmm(0), Tmm(4), Tmm(6));
  tileloadd(Tmm(5), ptr[reg_matA1ptr + reg_astep]);
  add(reg_matA1ptr, KTile);
  tdpbsud(Tmm(1), Tmm(4), Tmm(7));
  tileloadd(Tmm(4), ptr[reg_matAptr + reg_astep]);
  add(reg_matAptr, KTile);
  tdpbsud(Tmm(2), Tmm(5), Tmm(6));
  tileloadd(Tmm(6), ptr[reg_matBptr + reg_tmp + 0]);
  tdpbsud(Tmm(3), Tmm(5), Tmm(7));
  add(reg_iterk, KTile);
  cmp(reg_iterk, reg_ksize);  // k iteration variable
  jb(".kloop");
  int ZIDX_ScaleABC = 31;
  int ZIDX_ScaleC = 30;
  int ZIDX_Zero = 29;
  int ZIDX_FF = 28;
  int ZIDX_ZeropointC = 27;
  vbroadcastss(Zmm(ZIDX_ScaleABC), ptr[parambase + offsetof(ssd::transpose_attention_step3_params, scaleAB)]);
  vbroadcastss(Zmm(ZIDX_ScaleC), ptr[parambase + offsetof(ssd::transpose_attention_step3_params, scaleC)]);
  vrcp14ps(Zmm(ZIDX_ScaleC), Zmm(ZIDX_ScaleC));
  vmulps(Zmm(ZIDX_ScaleABC), Zmm(ZIDX_ScaleC));
  vxorps(Zmm(ZIDX_Zero), Zmm(ZIDX_Zero));
  vpbroadcastd(Zmm(ZIDX_FF), ptr[rsp + TValueStart]);
  vpbroadcastd(Zmm(ZIDX_ZeropointC), ptr[parambase + offsetof(ssd::transpose_attention_step3_params, zeropointC)]);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      tilestored(ptr[rsp + reg_tmp + TTmmStart + i * 16 * 64 * 2 + j * 64], Tmm(i * 2 + j));
    }
  }

  mov(reg_matCptr, ptr[parambase + offsetof(ssd::transpose_attention_step3_params, matC)]);
  xor_(reg_cstep, reg_cstep);
  mov(reg_cstep.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step3_params, cstep)]);

  xor_(reg_iterk, reg_iterk);
  lea(reg_matAptr, ptr[rsp + TTmmStart]);
  L(".mloop");
  int ZMM_COUNT = NTile / 16;
  for (int j = 0; j < ZMM_COUNT; j++) {
    vcvtdq2ps(Zmm(j), ptr[reg_matAptr + j * 64]);
    vmulps(Zmm(j), Zmm(ZIDX_ScaleABC));
    vcvtps2dq(Zmm(j), Zmm(j));
    vpaddd(Zmm(j), Zmm(j), Zmm(ZIDX_ZeropointC));
    vpmaxsd(Zmm(j), Zmm(j), Zmm(ZIDX_Zero));
    vpminsd(Zmm(j), Zmm(j), Zmm(ZIDX_FF));
    vpmovdw(Ymm(j), Zmm(j));
  }
  add(reg_matAptr, ZMM_COUNT * 64);
  for (int j = 0; j < ZMM_COUNT; j += 2) {
    vshufi32x4(Zmm(j * 2), Zmm(j * 2), Zmm(j * 2 + 1), (4 << 4) | 4);
    vpmovwb(ptr[reg_matCptr + 16 * j], Zmm(j * 2));
  }
  add(reg_matCptr, reg_cstep);
  add(reg_iterk, 1);
  cmp(reg_iterk, MTile);  // k iteration variable
  jb(".mloop");

  mov(reg_ret, 0);
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(Xmm(i + 6), xword[rsp + i * 16]);
  }
#endif
  outLocalLabel();  // end of local label
}

void MHA_Matmul_s8u8u8_vnni_word_8x32::generate() {
  inLocalLabel();  // use local label for multiple instance
  int XmmReserve = 16 * (10 + 2);
  int TmmReserve = 0;
  int TmpValueReserve = 64;
  int TmpSpace = XmmReserve + TmmReserve + TmpValueReserve;
  int TTmmStart = XmmReserve;
  int TValueStart = TTmmStart + TmmReserve;
  StackFrame st(this, 1, 7, TmpSpace);
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(xword[rsp + i * 16], Xmm(6 + i));
  }
#endif
  const Xbyak::Reg64& parambase = st.p[0];
  const Xbyak::Reg64& reg_matAptr = st.t[0];
  const Xbyak::Reg64& reg_matBptr = st.t[1];
  const Xbyak::Reg64& reg_matCptr = st.t[1];
  const Xbyak::Reg64& reg_astep = st.t[2];
  const Xbyak::Reg64& reg_cstep = st.t[2];
  const Xbyak::Reg64& reg_ksize = st.t[3];
  const Xbyak::Reg64& reg_iterk = st.t[4];
  const Xbyak::Reg64& reg_tmp = st.t[5];
  const Xbyak::Reg64& reg_TmpPtr = st.t[6];
  const Xbyak::Reg64& reg_ret = rax;
  int ZIDX_ScaleABC = 31;
  int ZIDX_ScaleC = 30;
  int ZIDX_Zero = 29;
  int ZIDX_FF = 28;
  int ZIDX_ZeropointC = 27;
  int ZIDX_CReg = 0;
  int ZIDX_AReg = 16;
  int ZIDX_BReg = 18;
  mov(reg_tmp.cvt32(), 0xff);
  mov(ptr[rsp + TValueStart], reg_tmp.cvt32());

  xor_(reg_ksize, reg_ksize);
  mov(reg_ksize.cvt32(), ptr[parambase + offsetof(ssd::transpose_attention_step3_params, k)]);
  xor_(reg_astep, reg_astep);
  mov(reg_astep.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step3_params, astep)]);

  mov(reg_matAptr, ptr[parambase + offsetof(ssd::transpose_attention_step3_params, matA)]);
  mov(reg_matBptr, ptr[parambase + offsetof(ssd::transpose_attention_step3_params, matB)]);
  xor_(reg_iterk, reg_iterk);
  for (int i = 0; i < 16; i++) {
    vxorps(Zmm(ZIDX_CReg + i), Zmm(ZIDX_CReg + i));
  }

  L(".kloop");
  for (int i = 0; i < KTile; i += 2) {
    vpmovzxbw(Zmm(ZIDX_BReg + 0), ptr[reg_matBptr + 0]);  // u8 => s16, src 32*2 u8
    vpmovzxbw(Zmm(ZIDX_BReg + 1), ptr[reg_matBptr + 32]);
    lea(reg_TmpPtr, ptr[reg_matAptr + reg_iterk + i]);
    for (int j = 0; j < MTile; j++) {
      vpbroadcastw(Ymm(ZIDX_AReg), ptr[reg_TmpPtr]);
      vpmovsxbw(Zmm(ZIDX_AReg), Ymm(ZIDX_AReg));
      vpdpwssds(Zmm(ZIDX_CReg + j * 2 + 0), Zmm(ZIDX_AReg), Zmm(ZIDX_BReg + 0));
      vpdpwssds(Zmm(ZIDX_CReg + j * 2 + 1), Zmm(ZIDX_AReg), Zmm(ZIDX_BReg + 1));
      add(reg_TmpPtr, reg_astep);
    }
    add(reg_matBptr, 32 * 2);
  }
  add(reg_iterk, KTile);
  cmp(reg_iterk, reg_ksize);  // k iteration variable
  jb(".kloop");

  vbroadcastss(Zmm(ZIDX_ScaleABC), ptr[parambase + offsetof(ssd::transpose_attention_step3_params, scaleAB)]);
  vbroadcastss(Zmm(ZIDX_ScaleC), ptr[parambase + offsetof(ssd::transpose_attention_step3_params, scaleC)]);
  vrcp14ps(Zmm(ZIDX_ScaleC), Zmm(ZIDX_ScaleC));
  vmulps(Zmm(ZIDX_ScaleABC), Zmm(ZIDX_ScaleC));
  vxorps(Zmm(ZIDX_Zero), Zmm(ZIDX_Zero));
  vpbroadcastd(Zmm(ZIDX_FF), ptr[rsp + TValueStart]);
  vpbroadcastd(Zmm(ZIDX_ZeropointC), ptr[parambase + offsetof(ssd::transpose_attention_step3_params, zeropointC)]);

  mov(reg_matCptr, ptr[parambase + offsetof(ssd::transpose_attention_step3_params, matC)]);
  xor_(reg_cstep, reg_cstep);
  mov(reg_cstep.cvt32(), dword[parambase + offsetof(ssd::transpose_attention_step3_params, cstep)]);

  int ZMM_COUNT = NTile / 16;
  for (int i = 0; i < MTile; i++) {
    for (int j = 0; j < ZMM_COUNT; j++) {
      vcvtdq2ps(Zmm(ZIDX_CReg + i * ZMM_COUNT + j), Zmm(ZIDX_CReg + i * ZMM_COUNT + j));
      vmulps(Zmm(ZIDX_CReg + i * ZMM_COUNT + j), Zmm(ZIDX_ScaleABC));
      vcvtps2dq(Zmm(ZIDX_CReg + i * ZMM_COUNT + j), Zmm(ZIDX_CReg + i * ZMM_COUNT + j));
      vpaddd(Zmm(ZIDX_CReg + i * ZMM_COUNT + j), Zmm(ZIDX_CReg + i * ZMM_COUNT + j), Zmm(ZIDX_ZeropointC));
      vpmaxsd(Zmm(ZIDX_CReg + i * ZMM_COUNT + j), Zmm(ZIDX_CReg + i * ZMM_COUNT + j), Zmm(ZIDX_Zero));
      vpminsd(Zmm(ZIDX_CReg + i * ZMM_COUNT + j), Zmm(ZIDX_CReg + i * ZMM_COUNT + j), Zmm(ZIDX_FF));
      vpmovdb(ptr[reg_matCptr + 16 * j], Zmm(ZIDX_CReg + i * ZMM_COUNT + j));
    }
    add(reg_matCptr, reg_cstep);
  }

  mov(reg_ret, 0);
#ifdef _WIN32
  for (int i = 0; i < 10; i++) {
    movaps(Xmm(i + 6), xword[rsp + i * 16]);
  }
#endif
  outLocalLabel();  // end of local label
}

}  // namespace jd