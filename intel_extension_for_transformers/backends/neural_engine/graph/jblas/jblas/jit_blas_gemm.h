//  Copyright (c) 2023 Intel Corporation
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
#pragma once
#include <array>

#include "jit_base.hpp"
#include "jit_blas_utils.h"

namespace jblas {
namespace gemm {
enum class GemmCoreType : int {
  Undef = 0,
  AVX2_4X24,
  AVX512F_8X48,
  AVX512_VNNI_8X48,
  AMX_BF16,
  AMX_INT8,
  AVX512_VNNI_3X48_KBLOCK,
};
class GemmCore_Row_NN_4x24_AVX2 {
 public:
  struct params {
    float *matA, *matB, *matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);
  typedef float AType;
  typedef float BType;
  typedef float CType;
  static JBLAS_ISA constexpr ISA = JblasAVX2;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX2_4X24;
  static int constexpr NTILE = 24, MTILE = 4, KTILE = 4 / sizeof(BType);
  static int constexpr KUNROLL = 2;
  static int constexpr PACK_ROW = 1;
  static int constexpr PREFERED_N = 144;
  class MicroKernel : protected jblas::xbyak::JitAvx2 {
   public:
    MicroKernel() {}
    static int constexpr VecBytes = 32;
    static int constexpr VecElements = VecBytes / sizeof(CType);
    int CRegCount = 12, BRegCount = 3, ARegCount = 1;
    int CReg = 0, BReg = 12, AReg = 15, TmpReg = BReg;
    int const NRegs = NTILE / VecElements;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxorq(Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n16", T_NEAR);
      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n16");
      cmp(reg_tmp, 16);
      jl(".n8", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n8");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 8);
      add(reg_matBptr, 8 * sizeof(BType));
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Ymm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(Xbyak::Ymm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vfmadd231ps(Xbyak::Ymm(CReg + mm * NRegs + i), Xbyak::Ymm(BReg + i), Xbyak::Ymm(AReg));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vaddps(Xbyak::Ymm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_4x24_AVX2() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(float* matA, float* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};
class GemmCore_Row_NN_8x48_AVX512F {
 public:
  struct params {
    float *matA, *matB, *matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);
  typedef float AType;
  typedef float BType;
  typedef float CType;
  static JBLAS_ISA constexpr ISA = JblasAVX512F;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX512F_8X48;
  static int constexpr NTILE = 48, MTILE = 8, KTILE = 4 / sizeof(BType);
  static int constexpr KUNROLL = 2;
  static int constexpr PACK_ROW = 1;
  static int constexpr PREFERED_N = 144;
  class MicroKernel : protected jblas::xbyak::JitAvx512f {
   public:
    MicroKernel() {}
    int CRegCount = 24, BRegCount = 6, ARegCount = 1;
    int CReg = 0, BReg = 24, AReg = 27, TmpReg = 28;
    int const NRegs = 3;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxorq(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n32", T_NEAR);
      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(Xbyak::Zmm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vfmadd231ps(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(BReg + i), Xbyak::Zmm(AReg));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vaddps(Xbyak::Zmm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_8x48_AVX512F() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(float* matA, float* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};
class GemmCore_Row_NN_8x48_AVX512_VNNI {
 public:
  struct params {
    uint8_t* matA;
    int8_t* matB;
    int32_t* matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef int32_t CType;
  static JBLAS_ISA constexpr ISA = JblasAVX512_VNNI;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX512_VNNI_8X48;
  static int constexpr NTILE = 48, MTILE = 8, KTILE = 4 / sizeof(BType);
  static int constexpr PACK_ROW = KTILE;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 192;

  class MicroKernel : protected jblas::xbyak::JitAvx512vnni {
   public:
    MicroKernel() {}
    int CRegCount = 24, BRegCount = 6, ARegCount = 1;
    int CReg = 0, BReg = 24, AReg = 27, TmpReg = 28;
    int const NRegs = 3;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxorq(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n32", T_NEAR);
      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(int8_t) * 4);
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(int8_t) * 4);
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KTILE * KUNROLL);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_tmp1, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& reg_tmp,
                      const Xbyak::Reg64& reg_matAptr, const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Zmm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(AReg), Xbyak::Zmm(BReg + i));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vpaddd(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
  };

 public:
  GemmCore_Row_NN_8x48_AVX512_VNNI() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

  void reference(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride,
                 int _cstride, int kpos) {
    int lda = _astride / sizeof(AType);
    int ldb = _bstride / sizeof(BType);
    int ldc = _cstride / sizeof(CType);
    for (int i = 0; i < _m; i++) {
      for (int j = 0; j < _n; j += NTILE) {
        for (int ij = 0; ij < NTILE; ij++) {
          int tmp = 0;
          for (int k = 0; k < _k; k += 4) {
            for (int ik = 0; ik < 4; ik++) {
              tmp += int(matA[i * lda + k + ik]) * int(matB[k * NTILE + ij * 4 + ik + j * ldb]);
            }
          }
          matC[i * ldc + j + ij] = tmp;
        }
      }
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};
class GemmCore_Row_NN_16x64_AMX_BF16 {
 public:
  typedef uint16_t AType;
  typedef uint16_t BType;
  typedef float CType;
  struct params {
    AType* matA;
    BType* matB;
    CType* matC;
    int k, msize, nsize;
    int astep, bstep, cstep;
    int kpos;
    void *workspace, *cfg;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAMX_BF16;
  static GemmCoreType constexpr TYPE = GemmCoreType::AMX_BF16;
  static int constexpr NTILE = 64, MTILE = 16, KTILE = 64 / sizeof(BType);
  static int constexpr PACK_ROW = 2;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 256;
  class MicroKernel : protected jblas::xbyak::JitAmxbf16 {
   public:
    friend GemmCore_Row_NN_16x64_AMX_BF16;
    MicroKernel() {}
    static int constexpr CReg = 0, TmpReg = 4;
    static int constexpr NRegs = 4;
    static int constexpr CRegCount = NRegs;
    static int constexpr C_tilenum = 4, A_tilenum = 1, B_tilenum = 3;
    static int constexpr CTile = 0, ATile = CTile + C_tilenum, BTile = ATile + A_tilenum;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code() {
      reset();
      generate_mtile();
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile() {
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);
      mov(reg_tmp, ptr[parambase + OFFSET(cfg)]);
      ldtilecfg(ptr[reg_tmp]);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < C_tilenum; i++) {
        tilezero(Xbyak::Tmm(CTile + i));
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n48", T_NEAR);
      generate_kloop(NRegs);
      write_back(MTILE, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n48");
      cmp(reg_tmp, 48);
      jl(".n32", T_NEAR);
      generate_kloop(3);
      write_back(MTILE, 3, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 48);
      add(reg_matBptr, 48 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(2);
      write_back(MTILE, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(1);
      write_back(MTILE, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NTile, int _kunroll, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      mov(reg_tmp, NTILE * 4);
      if (_NTile <= B_tilenum) {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile; i++) {
              tdpbf16ps(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + i));
            }
          }
        }
      } else {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile - 1; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile - 1; i++) {
              tdpbf16ps(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + i));
            }
            tileloaddt1(Xbyak::Tmm(BTile + 0), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + (_NTile - 1) * 64]);
            tdpbf16ps(Xbyak::Tmm(CTile + mm * C_tilenum + _NTile - 1), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + 0));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
      mov(reg_tmp1, NTILE * 4);
      for (int mm = 0; mm < 1; mm++) {
        for (int i = 0; i < _NRegs; i++) {
          tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * C_tilenum + i));
        }
      }
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vaddps(Xbyak::Zmm(CReg + j), ptr[reg_matCptr + j * VecBytes]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_16x64_AMX_BF16() {
    mCodes.generate_code();
    memset(&mCfg, 0, sizeof(mCfg));
    jblas::xbyak::JitAmxtile::configure_tiles(mCfg, 16, 16, 32, sizeof(BType), MicroKernel::A_tilenum,
                                              MicroKernel::B_tilenum, MicroKernel::C_tilenum);
  }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    char tmp[NTILE * MTILE * sizeof(CType)];
    auto param = params{matA, matB, matC, _k, _m, _n, _astride, _bstride, _cstride, kpos, tmp, &mCfg};
    if (_m <= MTILE) {
      jblas::xbyak::JitAmxtile::configure_tiles(mCfg, _m < 16 ? _m : 16, _n < 16 ? _n : 16, _k < KTILE ? _k : KTILE,
                                                sizeof(BType), MicroKernel::A_tilenum, MicroKernel::B_tilenum,
                                                MicroKernel::C_tilenum);
      mCodes.mKernel(&param);
    } else {
      assert(0);
    }
  }

  void reference(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride,
                 int _cstride, int kpos);

 private:
  MicroKernel::tileconfig_t mCfg;
  MicroKernel mCodes;
};
class GemmCore_Row_NN_16x64_AMX_INT8 {
 public:
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef int32_t CType;
  struct params {
    AType* matA;
    BType* matB;
    CType* matC;
    int k, msize, nsize;
    int astep, bstep, cstep;
    int kpos;
    void *workspace, *cfg;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAMX_INT8;
  static GemmCoreType constexpr TYPE = GemmCoreType::AMX_INT8;
  static int constexpr NTILE = 64, MTILE = 16, KTILE = 64 / sizeof(BType);
  static int constexpr PACK_ROW = 4;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 256;
  class MicroKernel : protected jblas::xbyak::JitAmxint8 {
   public:
    friend GemmCore_Row_NN_16x64_AMX_INT8;
    MicroKernel() {}
    static int constexpr CReg = 0, TmpReg = 4;
    static int constexpr NRegs = 4;
    static int constexpr CRegCount = NRegs;
    static int constexpr C_tilenum = 4, A_tilenum = 1, B_tilenum = 3;
    static int constexpr CTile = 0, ATile = CTile + C_tilenum, BTile = ATile + A_tilenum;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code() {
      reset();
      generate_mtile();
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile() {
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);
      mov(reg_tmp, ptr[parambase + OFFSET(cfg)]);
      ldtilecfg(ptr[reg_tmp]);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < C_tilenum; i++) {
        tilezero(Xbyak::Tmm(CTile + i));
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n48", T_NEAR);
      generate_kloop(NRegs);
      write_back(MTILE, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n48");
      cmp(reg_tmp, 48);
      jl(".n32", T_NEAR);
      generate_kloop(3);
      write_back(MTILE, 3, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 48);
      add(reg_matBptr, 48 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(2);
      write_back(MTILE, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(1);
      write_back(MTILE, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NTile, int _kunroll, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      mov(reg_tmp, NTILE * 4);
      if (_NTile <= B_tilenum) {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile; i++) {
              tdpbusd(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + i));
            }
          }
        }
      } else {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile - 1; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile - 1; i++) {
              tdpbusd(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + i));
            }
            tileloaddt1(Xbyak::Tmm(BTile + 0), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + (_NTile - 1) * 64]);
            tdpbusd(Xbyak::Tmm(CTile + mm * C_tilenum + _NTile - 1), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + 0));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
      mov(reg_tmp1, NTILE * 4);
      for (int mm = 0; mm < 1; mm++) {
        for (int i = 0; i < _NRegs; i++) {
          tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * C_tilenum + i));
        }
      }
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vpaddd(Xbyak::Zmm(CReg + j), Xbyak::Zmm(CReg + j), ptr[reg_matCptr + j * VecBytes]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_16x64_AMX_INT8() {
    mCodes.generate_code();
    memset(&mCfg, 0, sizeof(mCfg));
    jblas::xbyak::JitAmxint8::configure_tiles(mCfg, 16, 16, 64, sizeof(BType), MicroKernel::A_tilenum,
                                              MicroKernel::B_tilenum, MicroKernel::C_tilenum);
  }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    char tmp[NTILE * MTILE * sizeof(CType)];
    auto param = params{matA, matB, matC, _k, _m, _n, _astride, _bstride, _cstride, kpos, tmp, &mCfg};
    if (_m <= MTILE) {
      jblas::xbyak::JitAmxint8::configure_tiles(mCfg, _m < 16 ? _m : 16, _n < 16 ? _n : 16, _k < KTILE ? _k : KTILE,
                                                sizeof(BType), MicroKernel::A_tilenum, MicroKernel::B_tilenum,
                                                MicroKernel::C_tilenum);
      mCodes.mKernel(&param);
    } else {
      assert(0);
    }
  }

  void reference(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride,
                 int _cstride, int kpos);

 private:
  MicroKernel::tileconfig_t mCfg;
  MicroKernel mCodes;
};

// KBLKs= K/BlkSize
// Weight scale=KBLKs*N
// Activation zp=M*KBLKs scale=M*KBLKs
template <typename _OT, typename _ST>
class GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK {
 public:
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef _OT CType;
  typedef _ST SType;
  struct params {
    uint8_t* matA;
    int8_t* matB;
    CType* matC;
    uint8_t* zpA;
    float* scaleA;
    SType* scaleB;
    int ldsa, ldsb;
    int kblock;
    int k, nsize;
    int astep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAVX512_VNNI;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX512_VNNI_3X48_KBLOCK;
  static int constexpr NTILE = 48, MTILE = 3, KTILE = 4 / sizeof(BType);
  static int constexpr PACK_ROW = KTILE;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 192;

  class MicroKernel : protected jblas::xbyak::JitAvx512vnni {
   public:
    MicroKernel() {}
    int CRegCount = 9, BRegCount = 3, ARegCount = 1, ZpARegCount = MTILE;
    int CReg = 0, CF32Reg = 9, BReg = 18, AReg = 21, ZpAReg = 22, ZpTmp = 25;
    int const NRegs = 3;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      ZpARegCount = _mtile;
      BRegCount = NRegs;
      CF32Reg = CReg + CRegCount;
      BReg = CF32Reg + CRegCount;
      AReg = BReg + BRegCount;
      ZpAReg = AReg + ARegCount;
      ZpTmp = ZpAReg + ZpARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 13, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_cstep = st.t[3];
      reg_iterk = st.t[4];
      reg_astep = st.t[5];
      reg_kblock = st.t[6];
      reg_tmp = st.t[7];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[9];
      reg_zpAptr = st.t[10];
      reg_scaleAptr = st.t[11];
      reg_scaleBptr = st.t[12];
      reg_ret = rax;

      vreg_push(rsp);

      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_kblock, ptr[parambase + OFFSET(kblock)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxorq(Xbyak::Zmm(CF32Reg + i * NRegs + j), Xbyak::Zmm(CF32Reg + i * NRegs + j),
                 Xbyak::Zmm(CF32Reg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      mov(reg_zpAptr, ptr[parambase + OFFSET(zpA)]);
      mov(reg_scaleAptr, ptr[parambase + OFFSET(scaleA)]);
      mov(reg_scaleBptr, ptr[parambase + OFFSET(scaleB)]);
      xor_(reg_iterk, reg_iterk);

      load32(reg_tmp, ptr[parambase + OFFSET(nsize)]);
      cmp(reg_tmp, NTILE);
      jl(".n32", T_NEAR);
      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, reg_matCptr, reg_cstep);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2, reg_matCptr, reg_cstep);
      jmp(".nend", T_NEAR);

      L(".n16");
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1, reg_matCptr, reg_cstep);

      L(".nend");
      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _nregs; j++) {
          vpxorq(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j));
        }
      }
      mov(reg_tmp, reg_zpAptr);
      load32(reg_tmp1, ptr[parambase + OFFSET(ldsa)]);
      for (size_t i = 0; i < _mtile; i++) {
        vpbroadcastb(Xbyak::Zmm(ZpAReg + i), ptr[reg_tmp]);
        add(reg_tmp, reg_tmp1);
      }
      xor_(reg_tmp2, reg_tmp2);
      L(".kbloop");
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_matBptr, reg_astep);
      generate_zp_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matBptr);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_matBptr, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jge(".kbend");
      add(reg_tmp2, KTILE * KUNROLL);
      cmp(reg_tmp2.cvt32(), ptr[parambase + OFFSET(kblock)]);
      jb(".kbloop");
      L(".kbend");
      mov(reg_tmp, reg_scaleAptr);
      load32(reg_tmp1, ptr[parambase + OFFSET(ldsa)]);
      for (size_t i = 0; i < _mtile; i++) {
        vbroadcastss(Xbyak::Zmm(ZpAReg + i), ptr[reg_tmp]);
        lea(reg_tmp, ptr[reg_tmp + reg_tmp1 * sizeof(float)]);
      }
      for (size_t i = 0; i < _nregs; i++) {
        if (std::is_same<SType, float>::value) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_scaleBptr + i * VecBytes]);
        } else if (std::is_same<SType, utils::bf16>::value) {
          loadbf16_f32(Xbyak::Zmm(BReg + i), ptr[reg_scaleBptr + i * VecBytes / (sizeof(float) / sizeof(SType))]);
        }
      }
      generate_f32_accumulate(_mtile, _nregs);
      add(reg_zpAptr, sizeof(AType));
      add(reg_scaleAptr, sizeof(float));
      load32(reg_tmp, ptr[parambase + OFFSET(ldsb)]);
      lea(reg_scaleBptr, ptr[reg_scaleBptr + reg_tmp * sizeof(SType)]);
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& reg_tmp,
                      const Xbyak::Reg64& reg_matAptr, const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Zmm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(AReg), Xbyak::Zmm(BReg + i));
          }
        }
      }
    }

    void generate_zp_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& reg_tmp,
                         const Xbyak::Reg64& reg_matBptr) {
      for (int kk = 0; kk < _kunroll; kk++) {
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          for (int i = 0; i < _NRegs; i++) {
            vpxorq(Xbyak::Zmm(ZpTmp), Xbyak::Zmm(ZpTmp), Xbyak::Zmm(ZpTmp));
            vpdpbusds(Xbyak::Zmm(ZpTmp), Xbyak::Zmm(ZpAReg + mm), Xbyak::Zmm(BReg + i));
            vpsubd(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(ZpTmp));
          }
        }
      }
    }

    void generate_f32_accumulate(int _mtile, int _NRegs) {
      for (int mm = 0; mm < _mtile; mm++) {
        for (int i = 0; i < _NRegs; i++) {
          vcvtdq2ps(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(CReg + mm * NRegs + i));
          vmulps(Xbyak::Zmm(AReg), Xbyak::Zmm(ZpAReg + mm), Xbyak::Zmm(BReg + i));
          vmulps(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(AReg));
          vaddps(Xbyak::Zmm(CF32Reg + mm * NRegs + i), Xbyak::Zmm(CReg + mm * NRegs + i));
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& reg_matCptr, const Xbyak::Reg64& reg_cstep) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CF32Reg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vaddps(Xbyak::Zmm(CF32Reg + i * NRegs + j), Xbyak::Zmm(CF32Reg + i * NRegs + j),
                 ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CF32Reg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_zpAptr;
    Xbyak::Reg64 reg_scaleAptr;
    Xbyak::Reg64 reg_scaleBptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_kblock;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
  };

 public:
  GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, SType* scaleB, int _ldsb,
               int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    int ldb = _bstride / sizeof(BType);
    auto param = params{matA, matB, matC, zpA, scaleA, scaleB, _ldsa, _ldsb, _kblock, _k, _n, _astride, _cstride, kpos};
    if (_m <= MTILE) {
      for (int i = 0; i < _n; i += NTILE) {
        param.matB = matB + i * ldb;
        param.matC = matC + i;
        param.nsize = i + NTILE <= _n ? NTILE : _n - i;
        param.scaleB = scaleB + i;
        mCodes[_m - 1].mKernel(&param);
      }
    } else {
      assert(0);
    }
  }

  void reference(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, float* scaleB, int _ldsb,
                 int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    int lda = _astride / sizeof(matA[0]);
    int ldb = _bstride / sizeof(matB[0]);
    int ldc = _cstride / sizeof(matC[0]);
    for (int i = 0; i < _m; i++) {
      for (int j = 0; j < _n; j += NTILE) {
        for (int ij = 0; ij < NTILE; ij++) {
          if (j + ij >= _n) {
            break;
          }
          float tmpf = 0.f;
          for (int k = 0; k < _k; k += _kblock) {
            int tmp = 0;
            int zpval = int(zpA[i * _ldsa + k / _kblock]);
            for (int ik = 0; ik < _kblock; ik += 4) {
              if (k + ik >= _k) {
                break;
              }
              for (int ikk = 0; ikk < 4; ikk++) {
                tmp +=
                    (int(matA[i * lda + k + ik + ikk]) - zpval) * int(matB[(k + ik) * NTILE + ij * 4 + ikk + j * ldb]);
              }
            }
            tmpf += tmp * scaleA[i * _ldsa + k / _kblock] * scaleB[j + ij + k / _kblock * _ldsb];
          }
          matC[i * ldc + j + ij] = tmpf;
        }
      }
    }
  }
  void reference(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, utils::bf16* scaleB,
                 int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    int lda = _astride / sizeof(matA[0]);
    int ldb = _bstride / sizeof(matB[0]);
    int ldc = _cstride / sizeof(matC[0]);
    for (int i = 0; i < _m; i++) {
      for (int j = 0; j < _n; j += NTILE) {
        for (int ij = 0; ij < NTILE; ij++) {
          if (j + ij >= _n) {
            break;
          }
          float tmpf = 0.f;
          for (int k = 0; k < _k; k += _kblock) {
            int tmp = 0;
            int zpval = int(zpA[i * _ldsa + k / _kblock]);
            for (int ik = 0; ik < _kblock; ik += 4) {
              if (k + ik >= _k) {
                break;
              }
              for (int ikk = 0; ikk < 4; ikk++) {
                tmp +=
                    (int(matA[i * lda + k + ik + ikk]) - zpval) * int(matB[(k + ik) * NTILE + ij * 4 + ikk + j * ldb]);
              }
            }
            tmpf += tmp * scaleA[i * _ldsa + k / _kblock] * scaleB[j + ij + k / _kblock * _ldsb].tofloat();
          }
          matC[i * ldc + j + ij] = tmpf;
        }
      }
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};

}  // namespace gemm
}  // namespace jblas
