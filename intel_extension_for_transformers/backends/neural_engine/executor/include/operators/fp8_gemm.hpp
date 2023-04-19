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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_FP8_GEMM_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_FP8_GEMM_HPP_
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <xbyak/xbyak_util.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include <utility>

#include "fp8.hpp"
#include "xbyak/xbyak.h"

#define OFFSET(field) offsetof(params, field)

namespace jit_avx512f_fp8_gemm {

inline int updiv(int a, int b) { return (a + b - 1) / b; }

inline int padto(int a, int b) { return updiv(a, b) * b; }

inline int padto_le(int a, int b) { return a / b * b; }

class Padding_PackBfp8_avx512f : protected Xbyak::CodeGenerator {
 public:
  struct params {
    void *srcptr, *dstptr;
    int row, col;
    int rowpad, colpad;
    int srcstride, dststride;
  };
  typedef long long (*func_t)(params*);  // NOLINT

 public:
  static Padding_PackBfp8_avx512f Instance;
  const int SrcBytes, RowTile;
  const int MinColLen;
  const int MinRowLen = 1;
  Padding_PackBfp8_avx512f() : SrcBytes(4), RowTile(4 / SrcBytes), MinColLen(64 / SrcBytes) {
    inLocalLabel();  // use local label for multiple instance
    int SF_TmpSize = 64;
    int SF_TmpPos = 16 * 10;
    Xbyak::util::StackFrame st(this, 1, 13, 16 * 10 + SF_TmpSize);
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_srcptr = st.t[0];
    const Xbyak::Reg64& reg_dstptr = st.t[1];
    const Xbyak::Reg64& reg_srcstride = st.t[2];
    const Xbyak::Reg64& reg_dststride = st.t[3];
    const Xbyak::Reg64& reg_rowsize = st.t[4];
    const Xbyak::Reg64& reg_colsize = st.t[5];
    const Xbyak::Reg64& reg_iterrow = st.t[6];
    const Xbyak::Reg64& reg_itercol = st.t[7];
    const Xbyak::Reg64& reg_colpadsize = st.t[11];
    const Xbyak::Reg64& reg_tmp = st.t[8];
    const Xbyak::Reg64& reg_tmp1 = st.t[9];
    const Xbyak::Reg64& reg_tmp2 = st.t[12];
    const Xbyak::Reg64& reg_tmp3 = st.t[10];
    const Xbyak::Reg64& reg_ret = rax;
    auto& mask_rd = k1;

    mov(reg_ret, 0);
    outLocalLabel();  // end of local label
    this->ready();
    mKernel = this->getCode<func_t>();
  }

  void forward(void* srcptr, void* dstptr, int row, int col, int rowpad, int colpad, int srcstride, int dststride) {
    auto param = params{srcptr, dstptr, row, col, rowpad, colpad, srcstride, dststride};
    mKernel(&param);
  }

  void reference(void* srcptr, void* dstptr, int row, int col, int rowpad, int colpad, int srcstride, int dststride) {
    int srcld = srcstride / 4;
    auto sptr = reinterpret_cast<float*>(srcptr);
    auto dptr = reinterpret_cast<uint8_t*>(dstptr);
    int RowPack = 1;
    int NTile = 16;
    for (int irow = 0; irow < rowpad; irow += RowPack) {
      for (int icol = 0; icol < colpad; icol += NTile) {
        for (int iin = 0; iin < NTile; iin++) {
          if (irow < row) {
            if (icol + iin < col) {
              *(dptr + irow * NTile + icol * dststride + iin) =
                  float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(*(sptr + irow * srcld + (icol + iin)));
            } else {
              *(dptr + irow * NTile + icol * dststride + iin) =
                  float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(static_cast<float>(0));
            }
          } else {
            *(dptr + irow * NTile + icol * dststride + iin) =
                float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(static_cast<float>(0));
          }
        }
      }
    }
  }

 private:
  func_t mKernel = nullptr;
};
Padding_PackBfp8_avx512f Padding_PackBfp8_avx512f::Instance;

class Padding_Transpose_PackBfp8_avx512f : protected Xbyak::CodeGenerator {
 public:
  struct params {
    void *srcptr, *dstptr;
    int row, col;
    int rowpad, colpad;
    int srcstride, dststride;
  };
  typedef long long (*func_t)(params*);  // NOLINT

 public:
  static Padding_Transpose_PackBfp8_avx512f Instance;
  const int SrcBytes, RowTile;
  const int MinColLen;
  const int MinRowLen = 1;
  Padding_Transpose_PackBfp8_avx512f() : SrcBytes(4), RowTile(4 / SrcBytes), MinColLen(64 / SrcBytes) {
    inLocalLabel();  // use local label for multiple instance
    int SF_TmpSize = 64;
    int SF_TmpPos = 16 * 10;
    Xbyak::util::StackFrame st(this, 1, 13, 16 * 10 + SF_TmpSize);
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_srcptr = st.t[0];
    const Xbyak::Reg64& reg_dstptr = st.t[1];
    const Xbyak::Reg64& reg_srcstride = st.t[2];
    const Xbyak::Reg64& reg_dststride = st.t[3];
    const Xbyak::Reg64& reg_rowsize = st.t[4];
    const Xbyak::Reg64& reg_colsize = st.t[5];
    const Xbyak::Reg64& reg_iterrow = st.t[6];
    const Xbyak::Reg64& reg_itercol = st.t[7];
    const Xbyak::Reg64& reg_colpadsize = st.t[11];
    const Xbyak::Reg64& reg_tmp = st.t[8];
    const Xbyak::Reg64& reg_tmp1 = st.t[9];
    const Xbyak::Reg64& reg_tmp2 = st.t[12];
    const Xbyak::Reg64& reg_tmp3 = st.t[10];
    const Xbyak::Reg64& reg_ret = rax;
    auto& mask_rd = k1;

    mov(reg_ret, 0);
    outLocalLabel();  // end of local label
    this->ready();
    mKernel = this->getCode<func_t>();
  }

  void forward(uint16_t* srcptr, uint8_t* dstptr, int row, int col, int rowpad, int colpad, int srcstride,
               int dststride) {
    auto param = params{srcptr, dstptr, row, col, rowpad, colpad, srcstride, dststride};
    mKernel(&param);
  }

  float bf162float(uint16_t a) {
    uint32_t tmp = *(reinterpret_cast<uint32_t*>(&a));
    tmp = tmp << 16;
    return *(reinterpret_cast<float*>(&tmp));
  }

  void reference(uint16_t* srcptr, uint8_t* dstptr, int row, int col, int rowpad, int colpad, int srcstride,
                 int dststride) {
    int srcld = srcstride / 2;
    auto sptr = reinterpret_cast<uint16_t*>(srcptr);
    auto dptr = reinterpret_cast<uint8_t*>(dstptr);
    int RowPack = 1;
    int NTile = 16;
    for (int irow = 0; irow < rowpad; irow += NTile) {
      for (int icol = 0; icol < colpad; icol += 1) {
        for (int iin = 0; iin < NTile; iin++) {
          if (irow + iin < row) {
            if (icol < col) {
              *(dptr + irow * dststride + icol * NTile + iin) = float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(
                  bf162float(*(sptr + (irow + iin) * srcld + icol)));
            } else {
              *(dptr + irow * dststride + icol * NTile + iin) =
                  float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(static_cast<float>(0));
            }
          } else {
            *(dptr + irow * dststride + icol * NTile + iin) =
                float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(static_cast<float>(0));
          }
        }
      }
    }
  }

 private:
  func_t mKernel = nullptr;
};
Padding_Transpose_PackBfp8_avx512f Padding_Transpose_PackBfp8_avx512f::Instance;

// matC fp32=alpha*(matA fp32*matB fp8)+beta*matD fp32


// matC bf16=alpha*(matA bf16*matB fp8)+beta*matD bf16
class Avx512f_Row_Abf16Bfp8 : protected Xbyak::CodeGenerator {
 public:
  struct params {
    uint16_t* matA;
    uint8_t* matB;
    uint16_t *matC, *matD;
    int k, n, astep, bstep, cstep, dstep;
    int kpos;
    float alpha, beta;
  };
  typedef long long (*func_t)(params*);  // NOLINT

  static int constexpr MTile = 4, NTile = 32, KTile = 2;
  int const NRegs = NTile / 16;
  int const CRegCount = NRegs * MTile, BRegCount = NRegs, ARegCount = 1;
  int const CReg = 0, BReg = CRegCount, AReg = BReg + BRegCount;
  int const ConstReg = 16;
  int const TmpReg = 24;
  // subc 24 zmms
  // subb 6  zmms
  // suba 2  zmms
 public:
  static Avx512f_Row_Abf16Bfp8 Instance;
  Avx512f_Row_Abf16Bfp8() {
    inLocalLabel();  // use local label for multiple instance

    Xbyak::util::StackFrame st(this, 1, 11, 16 * 10 + 32);
    int TTmpStart = 16 * 10;
    auto& parambase = st.p[0];
    auto& reg_matAptr = st.t[0];
    auto& reg_matBptr = st.t[1];
    auto& reg_ksize = st.t[2];
    auto& reg_cstep = st.t[3];
    auto& reg_astep = st.t[5];
    auto& reg_bstep = st.t[8];
    auto& reg_iterk = st.t[4];
    auto& reg_tmp = st.t[6];
    auto& reg_tmp1 = st.t[7];
    auto& reg_tmp2 = st.t[10];
    auto& reg_nsize = st.t[9];
    auto& reg_ret = rax;
    vreg_push(rsp);

    for (int i = 0; i < CRegCount; i++) {
      vpxorq(Xbyak::Zmm(CReg + i), Xbyak::Zmm(CReg + i), Xbyak::Zmm(CReg + i));
    }

    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
    load32(reg_astep, ptr[parambase + OFFSET(astep)]);
    load32(reg_bstep, ptr[parambase + OFFSET(bstep)]);
    imul(reg_bstep, reg_bstep, 16);
    xor_(reg_iterk, reg_iterk);

    mov(reg_tmp1, 0xaaaaaaaa);
    kmovq(k1, reg_tmp1);

    mov(reg_tmp2, 120 << 23);
    vpbroadcastd(Xbyak::Zmm(ConstReg + 0), reg_tmp2.cvt32());

    mov(reg_tmp2, 28);
    vpbroadcastd(Xbyak::Zmm(ConstReg + 1), reg_tmp2.cvt32());

    mov(reg_tmp2, 121);
    vpbroadcastd(Xbyak::Zmm(ConstReg + 2), reg_tmp2.cvt32());

    mov(reg_tmp2, 0x7);
    vpbroadcastd(Xbyak::Zmm(ConstReg + 3), reg_tmp2.cvt32());

    cmp(reg_nsize, NTile);
    jb(".lastloop", T_NEAR);
    L(".kloop");
    generate_fma(MTile, NRegs, KTile, reg_matAptr, reg_matBptr, reg_tmp, reg_tmp1, reg_tmp2, reg_astep, reg_bstep);
    add(reg_matAptr, KTile * 2);
    add(reg_matBptr, KTile * 16);
    add(reg_iterk, KTile);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    alphabeta_process(MTile, NRegs, parambase, reg_tmp, reg_tmp1);
    jmp(".retl", T_NEAR);

    L(".lastloop");
    L(".k1loop");
    generate_fma(MTile, 1, KTile, reg_matAptr, reg_matBptr, reg_tmp, reg_tmp1, reg_tmp2, reg_astep, reg_bstep);
    add(reg_matAptr, KTile * 2);
    add(reg_matBptr, KTile * 16);
    add(reg_iterk, KTile);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".k1loop");
    alphabeta_process(MTile, 1, parambase, reg_tmp, reg_tmp1);

    L(".retl");
    vreg_pop(rsp);

    mov(reg_ret, 0);
    outLocalLabel();  // end of local label

    this->ready();
    mKernel = this->getCode<func_t>();
  }

  void forward(uint16_t* matA, uint8_t* matB, uint16_t* matC, uint16_t* matD, int _k, int _n, int _astride,
               int _bstride, int _cstride, int _dstride, int kpos, float alpha, float beta, bool has_gelu_) {
    auto param = params{matA, matB, matC, matD, _k, _n, _astride, _bstride, _cstride, _dstride, kpos, alpha, beta};
    mKernel(&param);
    if (has_gelu_) {
      gelu_compute_vector_fwd(matC, _n, _cstride);
    }
  }

 protected:
  void gelu_compute_vector_fwd(uint16_t* matC, int _n, int _cstride) {
    __m512 srcs[MTile];
    float fconst;
    *(reinterpret_cast<uint32_t*>(&fconst)) = 0x3d372713;
    float ftwopi;
    *(reinterpret_cast<uint32_t*>(&ftwopi)) = 0x3f4c422a;
    __m512 fitting_const = _mm512_set1_ps(fconst);
    __m512 two_pi = _mm512_set1_ps(ftwopi);
    __m512 one = _mm512_set1_ps(1.f);
    __m512 half = _mm512_set1_ps(.5f);
    int ldc = _cstride / sizeof(matC[0]);
    for (int in = 0; in < _n; in += 16) {
      for (int i = 0; i < MTile; i++) {
        auto tmp = _mm256_loadu_si256(reinterpret_cast<__m256i*>(matC + i * ldc + in));
        srcs[i] = _mm512_castsi512_ps(_mm512_cvtepu16_epi32(tmp));
        srcs[i] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(srcs[i]), 16));
      }
      for (int i = 0; i < MTile; i++) {
        auto x = srcs[i];
        srcs[i] = _mm512_mul_ps(srcs[i], srcs[i]);
        srcs[i] = _mm512_fmadd_ps(srcs[i], fitting_const, one);
        srcs[i] = _mm512_mul_ps(srcs[i], x);
        srcs[i] = _mm512_mul_ps(srcs[i], two_pi);
#ifdef _WIN32
        srcs[i] = _mm512_tanh_ps(srcs[i]);
#else
        tanh_compute(srcs[i]);
#endif
        srcs[i] = _mm512_add_ps(srcs[i], one);
        srcs[i] = _mm512_mul_ps(srcs[i], half);
        srcs[i] = _mm512_mul_ps(srcs[i], x);
      }
      for (int i = 0; i < MTile; i++) {
        srcs[i] = _mm512_castsi512_ps(_mm512_srli_epi32(_mm512_castps_si512(srcs[i]), 16));
        auto tmp = _mm512_cvtepi32_epi16(_mm512_castps_si512(srcs[i]));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(matC + i * ldc + in), tmp);
      }
    }
  }

#ifndef _WIN32
  void tanh_compute(__m512& src) {  // NOLINT
    for (int i = 0; i < 16; i++) {
      src[i] = tanhf(src[i]);
    }
  }
#endif

  void generate_fma(int MTile, int _NRegs, int KTile, const Xbyak::Reg64& aptr, const Xbyak::Reg64& btpr,
                    const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_tmp1, const Xbyak::Reg64& reg_tmp2,
                    const Xbyak::Reg64& reg_astep, const Xbyak::Reg64& reg_bstep) {
    int kk = 0;
    Xbyak::Zmm tmps[] = {Xbyak::Zmm(TmpReg), Xbyak::Zmm(TmpReg + 1), Xbyak::Zmm(TmpReg + 2), Xbyak::Zmm(TmpReg + 3),
                         Xbyak::Zmm(TmpReg + 4)};
    for (; kk < KTile; kk++) {
      int mm = 0;
      lea(reg_tmp, ptr[aptr + kk * 2]);
      mov(reg_tmp1, btpr);
      for (int nn = 0; nn < _NRegs; nn++) {
        load_fp8_fp32_e4m3(Xbyak::Zmm(BReg + nn), reg_tmp2.cvt32(), tmps, ptr[reg_tmp1 + kk * 16]);
        add(reg_tmp1, reg_bstep);
      }
      for (; mm < MTile; mm++) {
        broadcast_bf16_fp32(Xbyak::Zmm(AReg), k1, ptr[reg_tmp]);
        add(reg_tmp, reg_astep);
        for (int nn = 0; nn < _NRegs; nn++) {
          vfmadd231ps(Xbyak::Zmm(CReg + mm * NRegs + nn), Xbyak::Zmm(BReg + nn), Xbyak::Zmm(AReg));
        }
      }
    }
  }
  // #define HIGH_PRECISION
  void load_fp8_fp32_e4m3(const Xbyak::Zmm& tar, const Xbyak::Reg32& tmp, Xbyak::Zmm* tmps,
                          const Xbyak::Address& addr) {
#ifdef HIGH_PRECISION
    auto& sign = tmps[0];
    auto& exp = tmps[1];
    auto& matissa = tmps[2];
    vpmovzxbd(tar, addr);
    vpsrad(sign, tar, 7);
    vpslld(sign, sign, 31);

    vpslld(exp, tar, 25);
    vpslld(matissa, exp, 4);
    vpsrld(matissa, matissa, 29);  // m3
    vpsrld(exp, exp, 28);          // e4

    vpaddd(tmps[3], exp, Xbyak::Zmm(ConstReg + 0));
    vpslld(tmps[3], tmps[3], 23);
    vorps(tar, sign, tmps[3]);
    vpslld(tmps[3], matissa, 20);
    vorps(tar, tmps[3]);
#if 1
    vxorps(tmps[3], tmps[3], tmps[3]);
    vpcmpud(k2, exp, tmps[3], 0);
    vpcmpud(k3, matissa, tmps[3], 4);
    kandw(k4, k2, k3);

    vplzcntd(tmps[4], matissa);
    vpsubd(tmps[4], tmps[4], Xbyak::Zmm(ConstReg + 1));

    vpsubd(tmps[3], Xbyak::Zmm(ConstReg + 2), tmps[4]);
    vpslld(tmps[3], tmps[3], 23);
    vorps(exp, sign, tmps[3]);

    // shift
    vprolvd(tmps[4], matissa, tmps[4]);
    vpandd(tmps[4], tmps[4], Xbyak::Zmm(ConstReg + 3));
    vpslld(tmps[4], tmps[4], 20);
    vorps(exp, tmps[4]);

    vmovdqu32(tar | k4, exp);
#endif
#else
    auto& sign = tmps[0];
    auto& exp = tmps[1];
    auto& matissa = tmps[2];
    vpmovzxbd(tar, addr);
    vpsrld(sign, tar, 7);
    vpslld(exp, tar, 25);
    vpslld(tar, sign, 31);
    vpsrld(exp, exp, 5);
    vpaddd(exp, exp, Xbyak::Zmm(ConstReg + 0));
    vorps(tar, exp);
#endif
  }

  void alphabeta_process(int MTile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_tmp,
                         const Xbyak::Reg64& reg_tmp1) {
    inLocalLabel();
    cmp(dword[parambase + OFFSET(alpha)], 0x3F800000);  // 1.f
    je(".afteralpha", T_NEAR);
    vbroadcastss(Xbyak::Zmm(AReg), zword[parambase + OFFSET(alpha)]);
    for (int i = 0; i < MTile; i++) {
      for (int j = 0; j < _NRegs; j++) {
        vmulps(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(AReg));
      }
    }
    L(".afteralpha");
    cmp(dword[parambase + OFFSET(kpos)], 0);
    jnz(".ntinitkl", T_NEAR);
    cmp(dword[parambase + OFFSET(beta)], 0);
    jz(".afterbeta", T_NEAR);
    mov(reg_tmp, ptr[parambase + OFFSET(matD)]);
    load32(reg_tmp1, ptr[parambase + OFFSET(dstep)]);
    vbroadcastss(Xbyak::Zmm(AReg), zword[parambase + OFFSET(beta)]);
    for (int i = 0; i < MTile; i++) {
      for (int j = 0; j < _NRegs; j++) {
        // vmovups(Xbyak::Zmm(BReg + j), ptr[reg_tmp + j * 64]);
        load_bf16_fp32(Xbyak::Zmm(BReg + j), ptr[reg_tmp + j * 32]);
        vmulps(Xbyak::Zmm(BReg + j), Xbyak::Zmm(AReg));
        vaddps(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(BReg + j));
      }
      add(reg_tmp, reg_tmp1);
    }

    L(".afterbeta");
    mov(reg_tmp, ptr[parambase + OFFSET(matC)]);
    load32(reg_tmp1, ptr[parambase + OFFSET(cstep)]);
    for (int i = 0; i < MTile; i++) {
      for (int j = 0; j < _NRegs; j++) {
        store_fp32_bf16(Xbyak::Zmm(CReg + i * NRegs + j), ptr[reg_tmp + j * 32]);
      }
      add(reg_tmp, reg_tmp1);
    }
    jmp(".retl", T_NEAR);

    L(".ntinitkl");
    mov(reg_tmp, ptr[parambase + OFFSET(matC)]);
    load32(reg_tmp1, ptr[parambase + OFFSET(cstep)]);
    for (int i = 0; i < MTile; i++) {
      for (int j = 0; j < _NRegs; j++) {
        load_bf16_fp32(Xbyak::Zmm(TmpReg), ptr[reg_tmp + j * 32]);
        vaddps(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(TmpReg));
        store_fp32_bf16(Xbyak::Zmm(CReg + i * NRegs + j), ptr[reg_tmp + j * 32]);
      }
      add(reg_tmp, reg_tmp1);
    }

    L(".retl");
    outLocalLabel();
  }

  void store_fp32_bf16(const Xbyak::Zmm& _fp32, const Xbyak::Address& _add) {
#if 0
#ifdef HIGH_PRECISION
			vcvtneps2bf16(Xbyak::Ymm(_fp32.getIdx()), _fp32); // NOLINT
			vmovups(_add, Xbyak::Ymm(_fp32.getIdx())); // NOLINT
#else
			vpsrld(_fp32, _fp32, 16); // NOLINT
			vpmovdw(_add, _fp32); // NOLINT
#endif
#else
    vcvtneps2bf16(Xbyak::Ymm(_fp32.getIdx()), _fp32);
    vmovups(_add, Xbyak::Ymm(_fp32.getIdx()));
#endif
  }

  void load_bf16_fp32(const Xbyak::Zmm& _fp32, const Xbyak::Address& _add) {
    vpmovzxwd(_fp32, _add);
    vpslld(_fp32, _fp32, 16);
  }

  void broadcast_bf16_fp32(const Xbyak::Zmm& _fp32, const Xbyak::Opmask& _mask, const Xbyak::Address& _add) {
    vpbroadcastw(_fp32 | _mask | T_z, _add);
  }

  void load32(const Xbyak::Reg64& reg, const Xbyak::Address& addr) {
    xor_(reg, reg);
    mov(reg.cvt32(), addr);
  }

  void vreg_push(const Xbyak::Reg64& baseaddr) {
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(xword[baseaddr + i * 16], Xbyak::Xmm(6 + i));
    }
#endif
  }

  void vreg_pop(const Xbyak::Reg64& baseaddr) {
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(Xbyak::Xmm(6 + i), xword[baseaddr + i * 16]);
    }
#endif
  }

 private:
  func_t mKernel = nullptr;
};
Avx512f_Row_Abf16Bfp8 Avx512f_Row_Abf16Bfp8::Instance;

namespace parallel {

static Xbyak::util::Cpu _cpu;
struct CpuDevice {
  CpuDevice() {
    numcores = _cpu.getNumCores(Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
    ompthreads = omp_get_max_threads();
    numthreads = std::min(numcores, ompthreads);
#ifdef FORCE_NUM_THREADS
    numthreads = FORCE_NUM_THREADS;
#endif
    omp_set_num_threads(numthreads);
    L1Cache = _cpu.getDataCacheSize(0);
    L2Cache = _cpu.getDataCacheSize(1);
    mHas512F = _cpu.has(_cpu.tAVX512F);
    mHasVNNI512 = _cpu.has(_cpu.tAVX512_VNNI);
    mHasAMXBF16 = _cpu.has(_cpu.tAMX_BF16);
    mHasAMXINT8 = _cpu.has(_cpu.tAMX_INT8);
  }

  int getThreads() { return numthreads; }
  int numthreads;
  uint32_t L2Cache, L1Cache;
  bool mHasVNNI512, mHasAMXINT8, mHasAMXBF16, mHas512F;
  int numcores;
  int ompthreads;
};
static CpuDevice _cpudevice;

template <typename _T>
struct GemmCacheAdpter {
  void update(int m, int n, int k, int cachesize, int minN, int preferedN, float cacheratio = 0.8f) {
    mKTotal = k;
    int csize = m * n;
    int bsize = n * k;
    int constexpr EleSize = sizeof(_T);
    mElesize = cachesize * cacheratio / EleSize;
    int n_start = mElesize / k;
    if (n_start >= n) {
      mNMax = padto_le(n_start, minN);
      mKBatch = k;
      return;
    }
    int k_min = mElesize / preferedN;
    float c_acccess_ratio = static_cast<float>(k) / k_min;
    float c_threshold = 1.5;  // merge C access by using small N
    if (c_acccess_ratio <= c_threshold) {
      mNMax = padto_le(n_start, minN);
      mKBatch = k;
      return;
    }
    mNMax = preferedN;
    mKBatch = k_min;
  }

  void set_N(int _N, bool _minm) {
    mNMax = _N;
    mKBatch = _minm ? mKTotal : mElesize / mNMax;
  }

  void print() { printf("KBatch:%d NMax:%d EleSize:%d\n", mKBatch, mNMax, mElesize); }
  int mKBatch;
  int mNMax;
  int mKTotal;
  int mElesize;
};

struct Parallel2D {
  void getIndex(int threadIdx, int* row, int* col, int* rowsize, int* colsize) {
    if (threadIdx >= mValidThreads) {
      *rowsize = 0;
      *colsize = 0;
      return;
    }
    int tx = threadIdx % mColThreads;
    int ty = threadIdx / mColThreads;
    *col = tx * mThdCol;
    *row = ty * mThdRow;
    *colsize = remainsize(*col, mCols, mThdCol);
    *rowsize = remainsize(*row, mRows, mThdRow);
  }

  void calc_valid_threads() { mValidThreads = mColThreads * std::ceil(static_cast<float>(mRows) / mThdRow); }

  void print() {
    printf("Thread Block:(%d,%d)\n", mThdRow, mThdCol);
    printf("Thread in use:%d of %d, Nx%d\n", mValidThreads, mThreadsCount, mColThreads);
  }
  int mThdRow, mThdCol;
  int mColThreads;
  int mRows, mCols;
  int mValidThreads, mThreadsCount;
};

struct Parallel2DRowMajor : Parallel2D {
  void update(int row, int col, int minrow, int mincol, int ncores) {
    mCols = col;
    mRows = row;
    int colnum = updiv(col, mincol);
    int rownum = updiv(row, minrow);
    float ratio = colnum * rownum / static_cast<float>(ncores);
    if (ratio <= 1) {
      mThdRow = minrow;
      mColThreads = colnum;
      mThdCol = mincol;
      calc_valid_threads();
      return;
    }
    float colratio = ratio > colnum ? colnum : ceil(ratio);
    mThdCol = colratio * mincol;
    mColThreads = ceil(static_cast<float>(colnum) / colratio);
    mThdRow = ceil(rownum / (static_cast<float>(ncores) / mColThreads)) * minrow;
    calc_valid_threads();
  }
};

template <typename _T>
struct Parallel2DGemm : Parallel2D {
  void update(int row, int col, int minrow, int mincol, int ncores,
              GemmCacheAdpter<_T>& _adapter) {  // NOLINT
    mCols = col;
    mRows = row;
    mThreadsCount = ncores;
    int colpad = padto(col, mincol);
    int NMax = padto_le(_adapter.mNMax, mincol);
    int startN = NMax < colpad ? NMax : colpad;
    int maxN = 0;
    int minloop = std::numeric_limits<int>::max();
    for (int i = startN; i >= mincol; i -= mincol) {
      generate_by_Nstep(i, minrow, mincol);
      int threadloop =
          std::ceil(mThdCol / static_cast<float>(mincol)) * std::ceil(mThdRow / static_cast<float>(minrow));
      if (minloop > threadloop) {
        minloop = threadloop;
        maxN = i;
      } else if (threadloop > minloop) {
        break;
      }
    }
    generate_by_Nstep(maxN, minrow, mincol);
  }

  void generate_by_Nstep(int n, int minrow, int mincol) {
    mNStep = n;
    int rownum = updiv(mRows, minrow);
    int icol_num = updiv(mCols, mNStep);
    if (icol_num >= mThreadsCount) {
      float colratio = std::ceil(icol_num / static_cast<float>(mThreadsCount));
      mThdCol = mNStep * colratio;
      mThdRow = mRows;
      mColThreads = std::ceil(static_cast<float>(mCols) / mThdCol);
      calc_valid_threads();
      return;
    }
    mColThreads = icol_num;
    int trow = floor(static_cast<float>(mThreadsCount) / mColThreads);
    mThdCol = mNStep;
    mThdRow = std::ceil(rownum / static_cast<float>(trow)) * minrow;
    calc_valid_threads();
  }

  void print() {
    Parallel2D::print();
    printf("GEMM NStep:%d\n", mNStep);
  }
  int mNStep;
};

template <typename _T>
struct Parallel2DGemmV2 : Parallel2D {
  void update(int row, int col, int minrow, int mincol, int ncores,
              GemmCacheAdpter<_T>& _adapter) {  // NOLINT
    mRows = row;
    mCols = col;
    mMinRow = minrow;
    mMinCol = mincol;
    mThreadsCount = ncores;
    mNStep = _adapter.mNMax;
    int rownum = updiv(mRows, mMinRow);
    int colnum = updiv(mCols, mMinCol);
    int NRow = updiv(_adapter.mNMax, mMinCol);
    int maxN = 1;
    float maxScore = std::numeric_limits<float>::min();
    int core_enum = std::sqrt(mThreadsCount);
    for (int i = 1; i <= core_enum; i += 1) {
      generate_by_cores(i, mThreadsCount / i, rownum, colnum);
      auto thdscore = calculate_score();
      if (maxScore < thdscore) {
        maxScore = thdscore;
        maxN = i;
      }
      generate_by_cores(mThreadsCount / i, i, rownum, colnum);
      thdscore = calculate_score();
      if (maxScore < thdscore) {
        maxScore = thdscore;
        maxN = ncores / i;
      }
    }
    generate_by_cores(maxN, mThreadsCount / maxN, rownum, colnum);
    mNStep = std::min(mNStep, mThdCol);
  }

  float calculate_score() {
    int tmpnstep = mThdCol < mNStep ? mThdCol : mNStep;
    float threadratio = static_cast<float>(mValidThreads) / mThreadsCount;
    float rowratio = static_cast<float>(mThdRow) / mRows;
    float tileratio = static_cast<float>(mThdCol) / padto(mThdCol, tmpnstep);
    float density = static_cast<float>(tmpnstep) * mThdRow / (tmpnstep + mThdRow);
    density /= tmpnstep;
    return threadratio * 4.f + density * tileratio * 0.2f;
  }

  void generate_by_cores(int ny, int nx, int rownum, int colnum) {
    mThdRow = updiv(rownum, ny) * mMinRow;
    mThdCol = updiv(colnum, nx) * mMinCol;
    mColThreads = updiv(mCols, mThdCol);
    mValidThreads = updiv(mRows, mThdRow) * mColThreads;
  }

  void print() {
    Parallel2D::print();
    printf("GEMM NStep:%d\n", mNStep);
  }
  int mNStep;
  int mMinCol, mMinRow;
};

}  // namespace parallel

class WrapperBase {
 public:
  WrapperBase(int _m, int _n, int _k) : mM(_m), mN(_n), mK(_k) {}
  int mM, mN, mK;
  std::vector<uint8_t> mTemp;
};

class IWrapperAvx512f_row_Abf16Bfp8 : public WrapperBase {
 public:
  IWrapperAvx512f_row_Abf16Bfp8(int m, int n, int k) : WrapperBase(m, n, k) {
    auto& kernelc = Padding_PackBfp8_avx512f::Instance;
    auto& kernelg = Avx512f_Row_Abf16Bfp8::Instance;
    int kpad = padto(k, kernelc.MinRowLen);
    int npad = padto(n, kernelc.MinColLen);
    mTemp.resize(kpad * npad * sizeof(uint8_t));
    mCacheAdapter.update(m, n, k, parallel::_cpudevice.L2Cache, 16, 240);
    mParallel.update(m, n, kernelg.MTile, kernelg.NTile, parallel::_cpudevice.getThreads(), mCacheAdapter);
    mCacheAdapter.set_N(mParallel.mNStep, mParallel.mThdRow <= kernelg.MTile);
  }

  void packB_NT(float* matB, int k, int n, int ldb) {
    auto& kernelc = Padding_PackBfp8_avx512f::Instance;
    int kpad = padto(k, kernelc.MinRowLen);
    int npad = padto(n, kernelc.MinColLen);
    int newsize = kpad * npad * sizeof(uint8_t);
    if (mTemp.size() < newsize) {
      mTemp.resize(newsize);
    }
    auto ncores = parallel::_cpudevice.getThreads();
    parallel::Parallel2DRowMajor _para;
    _para.update(kpad, npad, kernelc.MinRowLen, kernelc.MinColLen, ncores);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = remainsize(rowidx, k, rowsize);
        int colremain = remainsize(colidx, n, colsize);
        // printf("%d %d %d %d %d %d %d\n", tidx, rowidx, colidx, rowsize,
        // colsize, rowremain, colremain);
        kernelc.reference(matB + rowidx * ldb + colidx,
                          reinterpret_cast<uint8_t*>(mTemp.data()) + rowidx * 16 + colidx * kpad, rowremain, colremain,
                          rowsize, colsize, n * sizeof(float), kpad);
      }
    }
  }

  void packBbf16_T(uint16_t* matB, int n, int k, int ldb) {
    auto& kernelc = Padding_Transpose_PackBfp8_avx512f::Instance;
    int npad = padto(n, 16);
    int kpad = padto(k, 1);
    int newsize = kpad * npad * sizeof(uint8_t);
    if (mTemp.size() < newsize) {
      mTemp.resize(newsize);
    }
    auto ncores = parallel::_cpudevice.getThreads();
    parallel::Parallel2DRowMajor _para;
    _para.update(npad, kpad, 16, 1, ncores);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = remainsize(rowidx, n, rowsize);
        int colremain = remainsize(colidx, k, colsize);
        // printf("%d %d %d %d %d %d %d\n", tidx, rowidx, colidx, rowsize,
        // colsize, rowremain, colremain);
        kernelc.reference(matB + rowidx * ldb + colidx,
                          reinterpret_cast<uint8_t*>(mTemp.data()) + rowidx * kpad + colidx * 16, rowremain, colremain,
                          rowsize, colsize, k * sizeof(uint16_t), kpad);
      }
    }
  }

  void forward(uint16_t* matA, float* matB, uint16_t* matC, uint16_t* matD, int m, int n, int k, int lda, int ldb,
               int ldc, int ldd, float alpha, float beta, bool has_gelu) {
    auto ncores = parallel::_cpudevice.getThreads();
    auto& kernelg = Avx512f_Row_Abf16Bfp8::Instance;
    if (matB != NULL) {
      packB_NT(matB, k, n, ldb);
      mM = m;
      mN = n;
      mK = k;
      mCacheAdapter.update(m, n, k, parallel::_cpudevice.L2Cache, 16, 240);
      mParallel.update(m, n, kernelg.MTile, kernelg.NTile, ncores, mCacheAdapter);
      mCacheAdapter.set_N(mParallel.mNStep, mParallel.mThdRow <= kernelg.MTile);
    } else if (m != mM) {
      mM = m;
      mN = n;
      mK = k;
      mCacheAdapter.update(m, n, k, parallel::_cpudevice.L2Cache, 16, 240);
      mParallel.update(m, n, kernelg.MTile, kernelg.NTile, ncores, mCacheAdapter);
      mCacheAdapter.set_N(mParallel.mNStep, mParallel.mThdRow <= kernelg.MTile);
    }
    // override for gpt-j
    { mCacheAdapter.mKBatch = k; }
    /*static bool f = false;
    if (!f)
    {
    mCacheAdapter.print();
    mParallel.print();
    f = true;
    }*/
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      mParallel.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = remainsize(rowidx, m, rowsize);
        int colremain = remainsize(colidx, n, colsize);
        // printf("%d %d %d %d %d %d %d\n", tidx, rowidx, colidx, rowsize,
        // colsize, rowremain, colremain);

        int kbatch = padto_le(mCacheAdapter.mKBatch, kernelg.KTile);
        auto cptr = matC + colidx + rowidx * ldc;
        auto dptr = matD + colidx + rowidx * ldd;
        for (int iterk = 0; iterk < k; iterk += kbatch) {
          int kbatch_remain = iterk + kbatch <= k ? kbatch : k - iterk;
          auto aptr = matA + rowidx * lda + iterk;
          auto bptr = reinterpret_cast<uint8_t*>(mTemp.data()) + colidx * k + iterk * 16;
          for (int j = 0; j < colremain; j += mParallel.mNStep) {
            for (int i = 0; i < rowremain; i += kernelg.MTile) {
              for (int in = 0; in < mParallel.mNStep; in += kernelg.NTile) {
                int tmpcol = j + in;
                if (j + in < colremain) {
                  int nsize = remainsize(j + in, colremain, kernelg.NTile);
                  kernelg.forward(aptr + i * lda, bptr + tmpcol * k, cptr + i * ldc + tmpcol, dptr + i * ldd + tmpcol,
                                  kbatch_remain, nsize, lda * 2, k, ldc * 2, ldd, iterk, alpha, beta, has_gelu);
                }
              }
            }
          }
        }
      }
    }
  }

  parallel::GemmCacheAdpter<float> mCacheAdapter;
  parallel::Parallel2DGemmV2<float> mParallel;
};
}  // namespace jit_avx512f_fp8_gemm
#define FP8_PTR(voidptr) ((jit_avx512f_fp8_gemm::IWrapperAvx512f_row_Abf16Bfp8*)voidptr)
#undef OFFSET
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_FP8_GEMM_HPP_
