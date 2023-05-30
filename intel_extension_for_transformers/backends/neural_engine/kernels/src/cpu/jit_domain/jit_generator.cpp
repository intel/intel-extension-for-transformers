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

#include "jit_generator.hpp"

namespace jd {

bool dump_asm_flag = false;

int jit_generator::dump_idx = 0;

bool jit_generator::create_kernel() {
  generate();
  if (std::getenv("SPARSE_LIB_DUMP") != nullptr) dump_asm_flag = true;
  if (dump_asm_flag) dump_asm();
  jit_ker_ = get_code();
  return (jit_ker_ != nullptr);
}

const uint8_t* jit_generator::get_code() {
  this->ready();
  auto code = CodeGenerator::getCode();
  if (callee_functions_code_size_ == 0) {
    return code;
  }
  return code + callee_functions_code_size_;
}

template <typename T>
Xbyak::Address jit_generator::EVEX_compress_addr(Xbyak::Reg64 base, T raw_offt, bool bcast) {
  using Xbyak::Address;
  using Xbyak::Reg64;
  using Xbyak::RegExp;
  using Xbyak::Zmm;

  SPARSE_LOG_IF(FATAL, raw_offt > INT_MAX) << "raw_offt should be int";
  auto offt = static_cast<int>(raw_offt);

  int scale = 0;

  if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
    offt = offt - 2 * EVEX_max_8b_offt;
    scale = 1;
  } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
    offt = offt - 4 * EVEX_max_8b_offt;
    scale = 2;
  }

  auto re = RegExp() + base + offt;
  if (scale) re = re + reg_EVEX_max_8b_offt * scale;

  if (bcast)
    return zword_b[re];
  else
    return zword[re];
}

Xbyak::Address jit_generator::make_safe_addr(const Xbyak::Reg64& reg_out, size_t offt, const Xbyak::Reg64& tmp_reg,
                                             bool bcast) {
  if (offt > INT_MAX) {
    mov(tmp_reg, offt);
    return bcast ? ptr_b[reg_out + tmp_reg] : ptr[reg_out + tmp_reg];
  } else {
    return bcast ? ptr_b[reg_out + offt] : ptr[reg_out + offt];
  }
}

Xbyak::Address jit_generator::EVEX_compress_addr_safe(const Xbyak::Reg64& base, size_t raw_offt,
                                                      const Xbyak::Reg64& reg_offt, bool bcast) {
  if (raw_offt > INT_MAX) {
    return make_safe_addr(base, raw_offt, reg_offt, bcast);
  } else {
    return EVEX_compress_addr(base, raw_offt, bcast);
  }
}

void jit_generator::dump_asm() {
  std::string file_name("code_" + std::to_string(dump_idx++) + ".bin");
  std::ofstream out_file(file_name, std::ios::out | std::ios::binary);
  out_file.write(reinterpret_cast<const char*>(getCode()), getSize());
  out_file.close();
}

void jit_generator::transpose_16x16_ps(const std::array<Xbyak::Zmm, 16UL>& src, const std::array<Xbyak::Zmm, 16UL>& tmp,
                                       const int N) {
  for (int i = 0; i < 8; ++i) {
    vpunpckldq(tmp[2 * i + 0], src[2 * i], src[2 * i + 1]);
    vpunpckhdq(tmp[2 * i + 1], src[2 * i], src[2 * i + 1]);
  }

  for (int i = 0; i < 4; ++i) {
    vpunpcklqdq(src[4 * i + 0], tmp[4 * i + 0], tmp[4 * i + 2]);
    vpunpckhqdq(src[4 * i + 1], tmp[4 * i + 0], tmp[4 * i + 2]);
    vpunpcklqdq(src[4 * i + 2], tmp[4 * i + 1], tmp[4 * i + 3]);
    vpunpckhqdq(src[4 * i + 3], tmp[4 * i + 1], tmp[4 * i + 3]);
  }

  for (int i = 0; i < 2; ++i) {
    vshufi32x4(tmp[8 * i + 0], src[8 * i + 0], src[8 * i + 4], 0x88);
    vshufi32x4(tmp[8 * i + 1], src[8 * i + 1], src[8 * i + 5], 0x88);
    vshufi32x4(tmp[8 * i + 2], src[8 * i + 2], src[8 * i + 6], 0x88);
    vshufi32x4(tmp[8 * i + 3], src[8 * i + 3], src[8 * i + 7], 0x88);
    vshufi32x4(tmp[8 * i + 4], src[8 * i + 0], src[8 * i + 4], 0xdd);
    vshufi32x4(tmp[8 * i + 5], src[8 * i + 1], src[8 * i + 5], 0xdd);
    vshufi32x4(tmp[8 * i + 6], src[8 * i + 2], src[8 * i + 6], 0xdd);
    vshufi32x4(tmp[8 * i + 7], src[8 * i + 3], src[8 * i + 7], 0xdd);
  }

  // last step and move out
  for (int i = 0; i < N; ++i) {
    vshufi32x4(src[i], tmp[i % 8], tmp[8 + i % 8], i < 8 ? 0x88 : 0xdd);
  }
}
void jit_generator::tile_product_amx_bf16ps(const Xbyak::Operand& reg_ksize, const Reg64& src0, const Reg64& src1,
                                            const Reg64& src0_stride, const Reg64& src1_stride, const Reg64& reg_tmp0,
                                            const Reg64& reg_tmp1, std::function<Xbyak::Address(int, int)> dst_addr) {
  constexpr int tile_k = TMM_MAX_COL / sizeof(bfloat16_t), TH = 2, TW = 2;
  auto&& reg_iterk = reg_tmp0;
  auto&& reg_src1_t1 = reg_tmp1;
  imul(reg_src1_t1, src0_stride, 16);
  lea(reg_src1_t1, ptr[src0 + reg_src1_t1]);

  for (int i = 0; i < TH * TW; i++) tilezero(Tmm(i));
  xor_(reg_iterk.cvt32(), reg_iterk.cvt32());
  Xbyak::Label l_kloop;
  L(l_kloop);

  tileloaddt1(Tmm(6), ptr[src1 + src1_stride + 0]);
  tileloadd(Tmm(4), ptr[src0 + src0_stride]);
  add(src0, tile_k * sizeof(bfloat16_t));
  tdpbf16ps(Tmm(0), Tmm(4), Tmm(6));
  tileloaddt1(Tmm(7), ptr[src1 + src1_stride + 64]);
  add(src1, BYTES_TMM * TW);
  tdpbf16ps(Tmm(1), Tmm(4), Tmm(7));
  tileloadd(Tmm(5), ptr[reg_src1_t1 + src0_stride]);
  add(reg_src1_t1, tile_k * sizeof(bfloat16_t));
  tdpbf16ps(Tmm(2), Tmm(5), Tmm(6));
  tdpbf16ps(Tmm(3), Tmm(5), Tmm(7));

  add(reg_iterk, tile_k);
  cmp(reg_iterk, reg_ksize);
  jb(l_kloop);
  for (int i = 0; i < TH; i++)
    for (int j = 0; j < TW; j++) tilestored(dst_addr(i, j), Tmm(i * 2 + j));
}

const std::array<float16_t, 3> exp_approx_f16_coeff = {
    float16_t(exp_approx_f32_coeff[0]), float16_t(exp_approx_f32_coeff[1]), float16_t(exp_approx_f32_coeff[2])};

void jit_generator::vmov_avx2(const RegExp& dst, const RegExp& src, const int bytes, const Xmm& tmp_xmm,
                              const Reg64& tmp_r64) {
  SPARSE_LOG_IF(FATAL, bytes <= 0 || bytes >= 32) << "AVX2 tail length between 1 and 31";
  if (bytes >= 16) {
    vmovups(tmp_xmm, xword[src]);
    vmovups(xword[dst], tmp_xmm);
    if (bytes > 16) vmovups(tmp_xmm, xword[src + (bytes - 16)]);
    if (bytes > 16) vmovups(xword[dst + (bytes - 16)], tmp_xmm);
  } else if (bytes >= 8) {
    mov(tmp_r64, qword[src]);
    mov(qword[dst], tmp_r64);
    if (bytes > 8) mov(tmp_r64, qword[src + (bytes - 8)]);
    if (bytes > 8) mov(qword[dst + (bytes - 8)], tmp_r64);
  } else if (bytes >= 4) {
    const auto tmp_r32 = tmp_r64.cvt32();
    mov(tmp_r32, dword[src]);
    mov(dword[dst], tmp_r32);
    if (bytes > 4) mov(tmp_r32, dword[src + (bytes - 4)]);
    if (bytes > 4) mov(dword[dst + (bytes - 4)], tmp_r32);
  } else if (bytes >= 2) {
    const auto tmp_r32 = tmp_r64.cvt32();
    movzx(tmp_r32, word[src]);
    mov(word[dst], tmp_r32.cvt16());
    if (bytes > 2) movzx(tmp_r32, word[src + (bytes - 2)]);
    if (bytes > 2) mov(word[dst + (bytes - 2)], tmp_r32.cvt16());
  } else if (bytes == 1) {
    const auto tmp_r32 = tmp_r64.cvt32();
    movzx(tmp_r32, byte[src]);
    mov(byte[dst], tmp_r32.cvt8());
  }
}

}  // namespace jd
