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
#include "jit_softmax_Ab16a.hpp"

#define GET_OFF(field) offsetof(jit_softmax_Ab16a::rt_data_t, field)
#define SHUFFLE(fp3, fp2, fp1, fp0) (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0))
namespace jd {

void jit_softmax_Ab16a::scaleph2i16(const Zmm& zmm1, const Zmm& zmm_scale, const Zmm& zmm_max, const Zmm& zmm_min) {
  vmulph(zmm1, zmm1, zmm_scale);
  vrndscaleph(zmm1, zmm1, 0x00);
  vminph(zmm1, zmm1, zmm_max);
  vmaxph(zmm1, zmm1, zmm_min);
  vcvtph2w(zmm1, zmm1);
}

void jit_softmax_Ab16a::cvtepi16_epi8_shuffle_storeu(RegExp dst, const Zmm& zmm0, const Zmm& zmm1, const Zmm& zmm2,
                                                     const Zmm& zmm3, const Zmm& tmp0, const Zmm& tmp1, const Zmm& tmp2,
                                                     const Zmm& tmp3) {
  vshufi32x4(tmp0, zmm0, zmm1, SHUFFLE(1, 0, 1, 0));
  vshufi32x4(tmp1, zmm0, zmm1, SHUFFLE(3, 2, 3, 2));
  vshufi32x4(tmp2, zmm2, zmm3, SHUFFLE(1, 0, 1, 0));
  vshufi32x4(tmp3, zmm2, zmm3, SHUFFLE(3, 2, 3, 2));
  vpmovwb(ptr[dst], tmp0);
  vpmovwb(ptr[dst + 32], tmp2);
  vpmovwb(ptr[dst + 64], tmp1);
  vpmovwb(ptr[dst + 96], tmp3);
}

void jit_softmax_Ab16a::generate() {
  Xbyak::Label l_exp_approx_coeff;
  Xbyak::Label l_log2e;
  Xbyak::Label l_ln2;

  this->preamble();
  mov(src_addr, ptr[reg_param + GET_OFF(src)]);         // r8
  mov(dst_addr, ptr[reg_param + GET_OFF(dst)]);         // r9
  mov(att_tile, dword[reg_param + GET_OFF(att_tile)]);  // r10d
  if (param_.att_tail) {
    mov(r11d, (1 << param_.att_tail) - 1);
    kmovw(mask, r11d);
  }
  // alloc memory temporarily
  mov(rbp, rsp);  // save rsp
  mov(r15d, att_tile);
  add(r15d, 2);
  sal(r15d, 9);
  sub(rsp, r15d);
  and_(rsp, 0xffffffc0);

  // load data from src and calculation max
  mov(r15d, bit_cast<uint32_t>(-INFINITY));
  vpbroadcastd(zmm0, r15d);
  for (int i = 1; i < 16; ++i) vmovaps(Zmm(i), zmm0);  // zmm0~15 is vmax
  mov(ebx, 0);
  mov(r13, src_addr);
  // convert int32 to fp32 and scale by QK_rescale_
  mov(r15d, dword[reg_param + GET_OFF(QK_rescale)]);
  vpbroadcastd(zmm17, r15d);  // zmm17 is vscale
  mov(r15d, bit_cast<uint32_t>(-10000.f));
  vpbroadcastd(zmm18, r15d);

  if (param_.has_badd) mov(r11, qword[reg_param + GET_OFF(src_badd)]);
  if (param_.has_badd) mov(r12d, dword[reg_param + GET_OFF(ld_badd)]);
  if (param_.has_badd) lea(r12d, ptr[r12d * sizeof(float)]);
  cmp(att_tile, 0);
  jle("load_max_in_softmax_end", T_NEAR);
  L("load_max_in_softmax");
  for (int i = 0; i < 16; ++i) {
    const auto& vreg_x = zmm16;
    vcvtdq2ps(vreg_x, zword[r13 + i * 16 * 4]);
    if (!param_.has_badd) {
      vmulps(vreg_x, vreg_x, zmm17);
    } else {
      if (i == 0)
        mov(r14, r11);
      else
        lea(r14, ptr[r14 + r12]);
      vmaxps(zmm19, zmm18, zword[r14]);
      vfmadd213ps(vreg_x, zmm17, zmm19);
    }
    vmovaps(zword[r13 + i * 16 * 4], vreg_x);
    vmaxps(Zmm(i), Zmm(i), vreg_x);
  }
  add(r13, 16 * 16 * 4);
  lea(r11, ptr[r11 + BYTES_ZMM]);
  add(ebx, 1);
  cmp(ebx, att_tile);
  jne("load_max_in_softmax");
  L("load_max_in_softmax_end");
  if (param_.att_tail) {
    for (int i = 0; i < 16; ++i) {
      const auto& vreg_x = zmm16;
      vmovaps(vreg_x, zmm18);
      vcvtdq2ps(vreg_x | mask, zword[r13 + i * 16 * 4]);
      if (!param_.has_badd) {
        vmulps(vreg_x, vreg_x, zmm17);
      } else {
        if (i == 0)
          mov(r14, r11);
        else
          lea(r14, ptr[r14 + r12]);
        vmaxps(zmm19, zmm18, zword[r14]);
        vfmadd213ps(vreg_x | mask, zmm17, zmm19);
      }
      vmovaps(zword[r13 + i * 16 * 4], vreg_x);
      vmaxps(Zmm(i), Zmm(i), vreg_x);
    }
  }
  vpxorq(zmm16, zmm16, zmm16);
  for (int i = 0; i < 16; ++i) {
    reduce_dwords(Zmm(i), zmm30, &CodeGenerator::vmaxps);
    vsubps(Zmm(i), zmm16, Zmm(i));  // negate
  }
  // calculation exp and sum
  for (int i = 16; i < 24; ++i) vpxorq(Zmm(i), Zmm(i), Zmm(i));  // zmm16~23 is vsum
  vpbroadcastw(zmm26, word[rip + l_log2e]);
  vpbroadcastw(zmm27, word[rip + l_ln2]);
  vpbroadcastw(zmm28, word[rip + l_exp_approx_coeff]);
  vpbroadcastw(zmm29, word[rip + l_exp_approx_coeff + 2]);
  vpbroadcastw(zmm30, word[rip + l_exp_approx_coeff + 4]);
  mov(ebx, 0);
  mov(r11, src_addr);
  mov(r12, rsp);
  cmp(att_tile, 0);
  jle("load_sum_in_softmax_end", T_NEAR);
  L("load_sum_in_softmax");
  for (int i = 0; i < 16; i += 2) {
    vaddps(zmm24, Zmm(i), zword[r11 + i * 16 * 4]);  // subtract max
    vcvtps2ph(ymm24, zmm24, 0x08);

    vaddps(zmm25, Zmm(i + 1), zword[r11 + (i + 1) * 16 * 4]);  // subtract max
    vcvtps2ph(ymm25, zmm25, 0x08);

    vshufi64x2(zmm25, zmm24, zmm25, SHUFFLE(1, 0, 1, 0));
    exp_approx_f16(zmm25, zmm25, zmm26, zmm27, {zmm28, zmm29, zmm30}, {zmm24, zmm31});
    vmovaps(ptr[r12 + i * 16 * 2], zmm25);
    vaddph(Zmm(16 + (i >> 1)), Zmm(16 + (i >> 1)), zmm25);
  }
  add(r11, 16 * 16 * 4);
  add(r12, 16 * 16 * 2);
  add(ebx, 1);
  cmp(ebx, att_tile);
  jne("load_sum_in_softmax");
  L("load_sum_in_softmax_end");

  if (param_.att_tail) {
    mov(r15d, bit_cast<uint32_t>(-10000.f));
    for (int i = 0; i < 16; i += 2) {
      vpbroadcastd(zmm24, r15d);
      vaddps(zmm24 | mask, Zmm(i), zword[r11 + i * 16 * 4]);  // subtract max
      vcvtps2ph(ymm24, zmm24, 0x08);

      vpbroadcastd(zmm25, r15d);
      vaddps(zmm25 | mask, Zmm(i + 1), zword[r11 + (i + 1) * 16 * 4]);  // subtract max
      vcvtps2ph(ymm25, zmm25, 0x08);
      vshufi64x2(zmm25, zmm24, zmm25, SHUFFLE(1, 0, 1, 0));
      exp_approx_f16(zmm25, zmm25, zmm26, zmm27, {zmm28, zmm29, zmm30}, {zmm24, zmm31});
      vmovaps(ptr[r12 + i * 16 * 2], zmm25);
      vaddph(Zmm(16 + (i >> 1)), Zmm(16 + (i >> 1)), zmm25);
    }
  }
  // Zmm0 ~15 is free
  // calculate div and scale
  mov(r15w, word[reg_param + GET_OFF(softmax_rescale)]);
  vpbroadcastw(zmm31, r15w);  // zmm31 is voscale

  mov(r14, reinterpret_cast<int64_t>(perm_data));
  vmovups(zmm24, ptr[r14]);
  for (int i = 0; i < 8; ++i) {
    vpermw(zmm25, zmm24, Zmm(16 + i));
    vaddph(Zmm(16 + i), Zmm(16 + i), zmm25);
    vpermilps(zmm25, Zmm(16 + i), SHUFFLE(2, 3, 0, 1));
    vaddph(Zmm(16 + i), Zmm(16 + i), zmm25);
    vpermilps(zmm25, Zmm(16 + i), SHUFFLE(1, 0, 3, 2));
    vaddph(Zmm(16 + i), Zmm(16 + i), zmm25);
    vshuff32x4(zmm25, Zmm(16 + i), Zmm(16 + i), SHUFFLE(2, 3, 0, 1));
    vaddph(Zmm(16 + i), Zmm(16 + i), zmm25);
    vdivph(Zmm(16 + i), zmm31, Zmm(16 + i));
  }

  // convert fp16 to int8 and store
  mov(ebx, 0);
  mov(r13, dst_addr);
  mov(r14, rsp);
  mov(r11w, max.data);
  mov(r12w, min.data);
  vpbroadcastw(zmm29, r11w);
  vpbroadcastw(zmm30, r12w);
  mov(r11d, att_tile);  // r11d is att_tile16_in_tile64
  sar(r11d, 2);
  mov(r12d, att_tile);
  and_(r12d, 0x00000003);  // r12d is att_tile16_in_tile64_tail

  cmp(r11d, 0);
  je("tail_process", T_NEAR);

  L("store_in_softmax");
  for (int i = 0; i < 16; i += 2) {
    for (int j = 0; j < 4; j++) {
      vmovups(Zmm(j), ptr[r14 + j * 16 * 16 * 2 + i * 16 * 2]);
      scaleph2i16(Zmm(j), Zmm(16 + (i >> 1)), zmm29, zmm30);
    }
    cvtepi16_epi8_shuffle_storeu(r13 + i * 4 * 16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7);
  }
  add(ebx, 1);
  add(r13, 16 * 4 * 16);
  add(r14, 4 * 16 * 16 * 2);
  cmp(ebx, r11d);
  jne("store_in_softmax", T_NEAR);

  // Tail process
  L("tail_process");
  for (int i = 0; i < 4; ++i) vpxorq(Zmm(i), Zmm(i), Zmm(i));

  Xbyak::Label L_tail_tbl, L_tail_0, L_tail_1, L_tail_2, L_tail_3;
  mov(r11, L_tail_tbl);
  jmp(ptr[r11 + r12 * sizeof(void*)]);

  L(L_tail_0);
  if (param_.att_tail) {
    for (int i = 0; i < 16; i += 2) {
      vmovups(zmm0, ptr[r14 + i * 16 * 2]);
      scaleph2i16(zmm0, Zmm(16 + (i >> 1)), zmm29, zmm30);
      cvtepi16_epi8_shuffle_storeu(r13 + i * 4 * 16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7);
    }
  }
  jmp("tail_process_end", T_NEAR);

  L(L_tail_1);
  for (int i = 0; i < 16; i += 2) {
    vmovups(zmm0, ptr[r14 + i * 16 * 2]);
    scaleph2i16(zmm0, Zmm(16 + (i >> 1)), zmm29, zmm30);
    if (param_.att_tail) {
      vmovups(zmm1, ptr[r14 + 1 * 16 * 16 * 2 + i * 16 * 2]);
      scaleph2i16(zmm1, Zmm(16 + (i >> 1)), zmm29, zmm30);
    }
    cvtepi16_epi8_shuffle_storeu(r13 + i * 4 * 16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7);
  }
  jmp("tail_process_end", T_NEAR);

  L(L_tail_2);
  for (int i = 0; i < 16; i += 2) {
    vmovups(zmm0, ptr[r14 + i * 16 * 2]);
    scaleph2i16(zmm0, Zmm(16 + ((i >> 1))), zmm29, zmm30);
    vmovups(zmm1, ptr[r14 + 1 * 16 * 16 * 2 + i * 16 * 2]);
    scaleph2i16(zmm1, Zmm(16 + ((i >> 1))), zmm29, zmm30);
    if (param_.att_tail) {
      vmovups(zmm2, ptr[r14 + 2 * 16 * 16 * 2 + i * 16 * 2]);
      scaleph2i16(zmm2, Zmm(16 + ((i >> 1))), zmm29, zmm30);
    }
    cvtepi16_epi8_shuffle_storeu(r13 + i * 4 * 16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7);
  }
  jmp("tail_process_end", T_NEAR);

  L(L_tail_3);
  for (int i = 0; i < 16; i += 2) {
    vmovups(zmm0, ptr[r14 + i * 16 * 2]);
    scaleph2i16(zmm0, Zmm(16 + ((i >> 1))), zmm29, zmm30);
    vmovups(zmm1, ptr[r14 + 1 * 16 * 16 * 2 + i * 16 * 2]);
    scaleph2i16(zmm1, Zmm(16 + ((i >> 1))), zmm29, zmm30);
    vmovups(zmm2, ptr[r14 + 2 * 16 * 16 * 2 + i * 16 * 2]);
    scaleph2i16(zmm2, Zmm(16 + ((i >> 1))), zmm29, zmm30);
    if (param_.att_tail) {
      vmovups(zmm3, ptr[r14 + 3 * 16 * 16 * 2 + i * 16 * 2]);
      scaleph2i16(zmm3, Zmm(16 + ((i >> 1))), zmm29, zmm30);
    }
    cvtepi16_epi8_shuffle_storeu(r13 + i * 4 * 16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7);
  }
  jmp("tail_process_end");

  L("tail_process_end");
  add(ebx, 1);
  add(r13, 16 * 4 * 16);
  vpxorq(zmm0, zmm0, zmm0);
  cmp(ebx, param_.sl_pad64 / 64);
  jge("end");
  for (int i = 0; i < 16; ++i) vmovdqu32(ptr[r13 + i * 4 * 16], zmm0);
  jmp("tail_process_end");
  L("end");
  mov(rsp, rbp);
  this->postamble();

  align(sizeof(void*));
  L(L_tail_tbl);
  putL(L_tail_0);
  putL(L_tail_1);
  putL(L_tail_2);
  putL(L_tail_3);

  L(l_log2e);
  db(float16_t(std::log2f(std::exp(1.f))).data, sizeof(float16_t));
  L(l_ln2);
  db(float16_t(std::log(2.f)).data, sizeof(float16_t));
  L(l_exp_approx_coeff);
  db(reinterpret_cast<const uint8_t*>(exp_approx_f16_coeff.data()), sizeof(exp_approx_f16_coeff));
}

}  // namespace jd
