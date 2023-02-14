#pragma once
#include <stdint.h>
#include <algorithm>
#include "xbyak/xbyak.h"
#include <xbyak/xbyak_util.h>

using namespace Xbyak;

#define OFFSET(field) offsetof(params, field)

namespace jit_mha_vnni_pad32 {
typedef uint16_t __bf16;

// src nxk s8 => dst n//32xk//2x32x2 s8
class MHA_vnniw_transpose_1B_N : protected Xbyak::CodeGenerator {
 public:
  struct params {
    void *srcptr, *dstptr;
    int n, k;
    int srcstride;
  };
  typedef long long (*func_t)(params*);

 public:
  const int NTile;
  MHA_vnniw_transpose_1B_N(int _ntile, size_t size = 16 * 1024) : Xbyak::CodeGenerator(size, 0), NTile(_ntile) {
    inLocalLabel();  // use local label for multiple instance
    int SF_TmpSize = 64;
    util::StackFrame st(this, 1, 9, 16 * 10 + SF_TmpSize);
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_srcptr = st.t[0];
    const Xbyak::Reg64& reg_dstptr = st.t[1];
    const Xbyak::Reg64& reg_srcstride = st.t[2];
    const Xbyak::Reg64& reg_ksize = st.t[3];
    const Xbyak::Reg64& reg_nsize = st.t[4];
    const Xbyak::Reg64& reg_iterk = st.t[5];
    const Xbyak::Reg64& reg_itern = st.t[6];
    const Xbyak::Reg64& reg_tmp = st.t[7];
    const Xbyak::Reg64& reg_tmp1 = st.t[8];
    const Xbyak::Reg64& reg_ret = rax;
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(xword[rsp + i * 16], Xmm(6 + i));
    }
#endif
    mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
    mov(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
    xor_(reg_srcstride, reg_srcstride);
    mov(reg_srcstride.cvt32(), ptr[parambase + OFFSET(srcstride)]);

    xor_(reg_ksize, reg_ksize);
    mov(reg_ksize.cvt32(), ptr[parambase + OFFSET(k)]);
    xor_(reg_nsize, reg_nsize);
    mov(reg_nsize.cvt32(), ptr[parambase + OFFSET(n)]);

    int ZIDX_TranSrc = 0;
    int ZIDX_TransTmp = 8;
    std::vector<Ymm> inputs(8), tmp(8);
    for (size_t i = 0; i < inputs.size(); i++) {
      inputs[i] = Ymm(ZIDX_TranSrc + i);
    }
    for (size_t i = 0; i < tmp.size(); i++) {
      tmp[i] = Ymm(ZIDX_TransTmp + i);
    }

    xor_(reg_itern, reg_itern);
    L(".nloop");
    xor_(reg_iterk, reg_iterk);
    L(".kloop");
    lea(reg_tmp1, ptr[reg_srcptr + reg_iterk]);
    for (int i = 0; i < NTile; i += 8) {
      for (int ii = 0; ii < 8; ii++) {
        vpmovzxwd(Ymm(ii), ptr[reg_tmp1]);
        lea(reg_tmp1, ptr[reg_tmp1 + reg_srcstride]);
      }
      auto outputs = transpose8x4B(inputs.data(), tmp.data());
      for (int ii = 0; ii < 8; ii++) {
        vpmovdw(ptr[reg_dstptr + i * 2 + ii * NTile * 2], outputs[ii]);
      }
    }
    add(reg_dstptr, NTile * 16);
    add(reg_iterk, 16);
    cmp(reg_iterk, reg_ksize);
    jb(".kloop");

    imul(reg_tmp, reg_srcstride, NTile);
    lea(reg_srcptr, ptr[reg_srcptr + reg_tmp]);
    add(reg_itern, NTile);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");

    mov(reg_ret, 0);
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(Xmm(i + 6), xword[rsp + i * 16]);
    }
#endif
    outLocalLabel();  // end of local label

    this->ready();
    mKernel = this->getCode<func_t>();
  }

  void forward(int8_t* srcptr, int8_t* dstptr, int n, int k, int srcstride) {
    auto param = params{srcptr, dstptr, n, k, srcstride};
    mKernel(&param);
  }

 protected:
  std::vector<Ymm> transpose8x4B(Ymm* rows, Ymm* tmp) {
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

 private:
  func_t mKernel = nullptr;
};

// matA s8 * matB s8 * alpha + matD f32 = out bf16
// matC = expLut [out]
class MHA_s8s8s8_row_vnni_8x32_binary_expf32bf16_allsum : protected Xbyak::CodeGenerator {
 public:
  struct params {
    int8_t* matA;
    int8_t* matB;
    __bf16* matC;
    float* matD;
    float* expsum;
    int m, n, k, astep;
    float scaleAB;
  };
  typedef long long (*func_t)(params*);

  int const MTile = 8, NTile = 32;
  int const KTile = 8;

 public:
  MHA_s8s8s8_row_vnni_8x32_binary_expf32bf16_allsum(size_t size = 32 * 1024) : Xbyak::CodeGenerator(size, 0) {
    int const ZMM_PerROW = NTile / 16;

    inLocalLabel();  // use local label for multiple instance
    int XmmReserve = 16 * (10 + 2);
    int TmmReserve = 0;
    int TmpValueReserve = 64;
    int TmpSpace = XmmReserve + TmmReserve + TmpValueReserve;
    Label tmpfvariable;
    {
      util::StackFrame st(this, 1, 12, TmpSpace);
#ifdef _WIN32
      for (int i = 0; i < 10; i++) {
        movaps(xword[rsp + i * 16], Xmm(6 + i));
      }
#endif
      const Xbyak::Reg64& parambase = st.p[0];
      const Xbyak::Reg64& reg_matAptr = st.t[0];
      const Xbyak::Reg64& reg_matBptr = st.t[1];
      const Xbyak::Reg64& reg_astep = st.t[2];
      const Xbyak::Reg64& reg_matCptr = st.t[3];
      const Xbyak::Reg64& reg_matDptr = st.t[4];
      const Xbyak::Reg64& reg_iterk = st.t[5];
      const Xbyak::Reg64& reg_sumptr = st.t[6];
      const Xbyak::Reg64& reg_itern = st.t[7];
      const Xbyak::Reg64& reg_TmpPtr = st.t[8];
      const Xbyak::Reg64& reg_iterm = st.t[10];
      const Xbyak::Reg64& reg_ksize = st.t[11];
      const Xbyak::Reg64& reg_ret = rax;

      int ZIDX_ScaleAB = 31;
      vbroadcastss(Zmm(ZIDX_ScaleAB), ptr[parambase + OFFSET(scaleAB)]);

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

      xor_(reg_astep, reg_astep);

      xor_(reg_ksize, reg_ksize);
      mov(reg_ksize.cvt32(), dword[parambase + OFFSET(k)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < ZMM_PerROW; i++) {
        vxorps(Zmm(ZIDX_ExpSum + i), Zmm(ZIDX_ExpSum + i));
      }

      xor_(reg_iterm, reg_iterm);
      L(".mloop");
      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      {
        auto reg_tmp = reg_iterk;
        mov(reg_tmp, reg_ksize);
        imul(reg_tmp, reg_itern);
        lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_astep.cvt32(), dword[parambase + OFFSET(astep)]);
      {
        auto reg_tmp = reg_iterk;
        mov(reg_tmp, reg_astep);
        imul(reg_tmp, reg_iterm);
        lea(reg_matAptr, ptr[reg_matAptr + reg_tmp]);
      }
      for (int i = 0; i < CReg_Count; i++) {
        vxorps(Zmm(ZIDX_CReg + i), Zmm(ZIDX_CReg + i));
      }

      xor_(reg_iterk, reg_iterk);
      L(".kloop");
      for (int i = 0; i < KTile; i += 2) {
        vpmovsxbw(Zmm(ZIDX_BReg + 0), ptr[reg_matBptr + 0 + i * 32]);  // s8 => s16, src 32*2 s8
        vpmovsxbw(Zmm(ZIDX_BReg + 1), ptr[reg_matBptr + 32 + i * 32]);
        lea(reg_TmpPtr, ptr[reg_matAptr + reg_iterk + i]);
        for (int j = 0; j < MTile; j++) {
          vpbroadcastw(Ymm(ZIDX_AReg), ptr[reg_TmpPtr]);
          vpmovsxbw(Zmm(ZIDX_AReg), Ymm(ZIDX_AReg));
          vpdpwssds(Zmm(ZIDX_CReg + j * 2 + 0), Zmm(ZIDX_AReg), Zmm(ZIDX_BReg + 0));
          vpdpwssds(Zmm(ZIDX_CReg + j * 2 + 1), Zmm(ZIDX_AReg), Zmm(ZIDX_BReg + 1));
          add(reg_TmpPtr, reg_astep);
        }
      }
      add(reg_matBptr, KTile * 32);
      add(reg_iterk, KTile);
      cmp(reg_iterk, reg_ksize);
      jb(".kloop");

      mov(reg_matDptr, ptr[parambase + OFFSET(matD)]);
      lea(reg_matDptr, ptr[reg_matDptr + reg_iterm * sizeof(float)]);
      auto reg_cstep = reg_astep;
      xor_(reg_cstep, reg_cstep);
      imul(reg_cstep.cvt32(), dword[parambase + OFFSET(n)], sizeof(__bf16));
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(__bf16)]);
      {
        auto reg_tmp = reg_iterk;
        mov(reg_tmp, reg_iterm);
        imul(reg_tmp.cvt32(), reg_cstep);
        add(reg_matCptr, reg_tmp);
      }

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
      cmp(reg_iterm.cvt32(), dword[parambase + OFFSET(m)]);
      jb(".mloop");

      mov(reg_sumptr, ptr[parambase + OFFSET(expsum)]);
      {
        auto reg_tmp = reg_iterm;
        imul(reg_tmp, reg_itern, sizeof(float));
        add(reg_sumptr, reg_tmp);
      }
      for (int i = 0; i < ZMM_PerROW; i++) {
        vrcp14ps(Zmm(ZIDX_ExpSum + i), Zmm(ZIDX_ExpSum + i));
        vmulps(Zmm(ZIDX_ExpSum + i), Zmm(ZIDX_FF));
        vmovups(ptr[reg_sumptr + i * 64], Zmm(ZIDX_ExpSum + i));
      }

      add(reg_itern, NTile);
      cmp(reg_itern.cvt32(), dword[parambase + OFFSET(n)]);
      jb(".nloop");

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
    prepare();
  }

  void prepare() {
    this->ready();
    mKernel = this->getCode<func_t>();
  }

  void forward(int8_t* matA, int8_t* matB, __bf16* matC, float* matD, float* expsum, int _m, int _n, int _k,
               int _astride, float scaleAB, float alpha) {
    if (mKernel == nullptr) {
      return;
    }
    auto param = params{matA, matB, matC, matD, expsum, _m, _n, _k, _astride, scaleAB * alpha};
    mKernel(&param);
  }

 protected:
  void packedf32_bf16(int idx0, int idx1) {
    vpsrld(Zmm(idx0), Zmm(idx0), 16);
    vpmovdw(Ymm(idx0), Zmm(idx0));

    vpsrld(Zmm(idx1), Zmm(idx1), 16);
    vpmovdw(Ymm(idx1), Zmm(idx1));

    vshufi32x4(Zmm(idx0), Zmm(idx0), Zmm(idx1), (4 << 4) | 4);
  }

  // x and y can be same register
  // len(tmp)==2
  void exp_f32_lowprecision(Zmm log2e, Zmm ln2, Zmm c[3], Zmm x, Zmm y, Zmm tmp[]) {
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

 private:
  func_t mKernel = nullptr;
};

class SeqCopy_1B_avx512_Nx4_Temp : protected Xbyak::CodeGenerator {
 public:
  struct params {
    void *srcptr, *dstptr;
    int srcstride;
    int dststride, k;
  };
  typedef long long (*func_t)(params*);

 public:
  int const NTile;
  SeqCopy_1B_avx512_Nx4_Temp(int _N = 64, size_t size = 16 * 1024) : Xbyak::CodeGenerator(size, 0), NTile(_N) {
    inLocalLabel();  // use local label for multiple instance

    util::StackFrame st(this, 1, 8, 16 * 10);
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

    mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
    mov(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
    mov(reg_ksize.cvt32(), ptr[parambase + OFFSET(k)]);
    mov(reg_ksize, reg_ksize.cvt32());

    xor_(reg_iterk, reg_iterk);
    mov(reg_dststride.cvt32(), dword[parambase + OFFSET(dststride)]);
    mov(reg_srcstride.cvt32(), dword[parambase + OFFSET(srcstride)]);
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
    sub(reg_tmp, NTile);
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

    this->ready();
    mKernel = this->getCode<func_t>();
  }

  void forward(void* srcptr, void* dstptr, int srcstride, int dststride, int loopk) {
    auto param = params{srcptr, dstptr, srcstride, dststride, loopk};
    mKernel(&param);
  }

 protected:
  void vnni_interleave_load_6regs(int startIdx) {
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

 private:
  func_t mKernel = nullptr;
};

// src mxn bf16 => dst nxm u8
// scale n f32
class MHA_norm_quantize_transpose_bf16 : protected Xbyak::CodeGenerator {
 public:
  struct params {
    void *srcptr, *dstptr;
    void* scaleptr;
    int m, n;
  };
  typedef long long (*func_t)(params*);

 public:
  MHA_norm_quantize_transpose_bf16(size_t size = 16 * 1024) : Xbyak::CodeGenerator(size, 0) {
    inLocalLabel();  // use local label for multiple instance
    int SF_TmpSize = 64;
    util::StackFrame st(this, 1, 9, 16 * 10 + SF_TmpSize);
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_srcptr = st.t[0];
    const Xbyak::Reg64& reg_dstptr = st.t[1];
    const Xbyak::Reg64& reg_scaleptr = st.t[2];
    const Xbyak::Reg64& reg_msize = st.t[3];
    const Xbyak::Reg64& reg_nsize = st.t[4];
    const Xbyak::Reg64& reg_iterm = st.t[5];
    const Xbyak::Reg64& reg_itern = st.t[6];
    const Xbyak::Reg64& reg_tmp = st.t[7];
    const Xbyak::Reg64& reg_tmp1 = st.t[8];
    const Xbyak::Reg64& reg_ret = rax;
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(xword[rsp + i * 16], Xmm(6 + i));
    }
#endif

    mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
    mov(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
    mov(reg_scaleptr, ptr[parambase + OFFSET(scaleptr)]);
    xor_(reg_msize, reg_msize);
    mov(reg_msize.cvt32(), ptr[parambase + OFFSET(m)]);
    xor_(reg_nsize, reg_nsize);
    mov(reg_nsize.cvt32(), ptr[parambase + OFFSET(n)]);

    int ZIDX_Scale = 16;
    int ZIDX_TranSrc = 0;
    int ZIDX_TransTmp = 8;
    std::vector<Ymm> inputs(8), tmp(8);
    for (size_t i = 0; i < inputs.size(); i++) {
      inputs[i] = Ymm(ZIDX_TranSrc + i);
    }
    for (size_t i = 0; i < tmp.size(); i++) {
      tmp[i] = Ymm(ZIDX_TransTmp + i);
    }

    xor_(reg_itern, reg_itern);
    L(".nloop");
    vmovups(Ymm(ZIDX_Scale), ptr[reg_scaleptr]);  // 8
    add(reg_scaleptr, 8 * sizeof(float));
    lea(reg_tmp1, ptr[reg_srcptr + reg_itern * sizeof(__bf16)]);
    xor_(reg_iterm, reg_iterm);
    L(".mloop");
    lea(reg_tmp, ptr[reg_dstptr + reg_iterm]);
    for (int ii = 0; ii < 8; ii++) {
      loadbf16_norm<Ymm>(ii, reg_tmp1, 0, ZIDX_Scale);
      lea(reg_tmp1, ptr[reg_tmp1 + reg_nsize * sizeof(__bf16)]);
    }
    auto outputs = transpose8x4B(inputs.data(), tmp.data());
    for (int ii = 0; ii < 8; ii++) {
      vpmovdb(ptr[reg_tmp], outputs[ii]);
      lea(reg_tmp, ptr[reg_tmp + reg_msize * sizeof(uint8_t)]);
    }
    add(reg_iterm, 8);
    cmp(reg_iterm, reg_msize);
    jb(".mloop");
    lea(reg_dstptr, ptr[reg_dstptr + reg_msize * (sizeof(uint8_t) * 8)]);

    add(reg_itern, 8);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");

    mov(reg_ret, 0);
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(Xmm(i + 6), xword[rsp + i * 16]);
    }
#endif
    outLocalLabel();  // end of local label

    this->ready();
    mKernel = this->getCode<func_t>();
  }

  void forward(__bf16* srcptr, uint8_t* dstptr, float* sumptr, int m, int n) {
    auto param = params{srcptr, dstptr, sumptr, m, n};
    mKernel(&param);
  }

 protected:
  template <typename T>
  void loadbf16_norm(int idx, Reg64 reg_srcptr, int offset, int scaleidx) {
    vpmovzxwd(T(idx), yword[reg_srcptr + offset]);
    vpslld(T(idx), T(idx), 16);
    normalize<T>(idx, scaleidx);
  }

  // n=exp/max(sumexp,epsilon)  0~1
  // o=n*255  0~255
  template <typename T>
  void normalize(int idx, int scaleidx) {
    vmulps(T(idx), T(idx), T(scaleidx));
    vcvtps2dq(T(idx), T(idx));
  }

  std::vector<Ymm> transpose8x4B(Ymm* rows, Ymm* tmp) {
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

 private:
  func_t mKernel = nullptr;
};

class MHA_Matmul_u8s8u8_vnni_byte_8x32 : protected Xbyak::CodeGenerator {
 public:
  struct params {
    uint8_t* matA;
    int8_t* matB;
    uint8_t* matC;
    int m, n, k, astep, cstep;
    float scaleAB, scaleC;
    int zeropointC;
  };
  typedef long long (*func_t)(params*);

  int const MTile = 8, NTile = 32, KTile = 8;

 public:
  MHA_Matmul_u8s8u8_vnni_byte_8x32(size_t size = 32 * 1024) : Xbyak::CodeGenerator(size, 0) {
    inLocalLabel();  // use local label for multiple instance
    int XmmReserve = 16 * (10 + 2);
    int TmmReserve = 0;
    int TmpValueReserve = 64;
    int TmpSpace = XmmReserve + TmmReserve + TmpValueReserve;
    int TTmmStart = XmmReserve;
    int TValueStart = TTmmStart + TmmReserve;
    util::StackFrame st(this, 1, 10, TmpSpace);
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(xword[rsp + i * 16], Xmm(6 + i));
    }
#endif
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_matAptr = st.t[0];
    const Xbyak::Reg64& reg_matBptr = st.t[1];
    const Xbyak::Reg64& reg_matCptr = st.t[9];
    const Xbyak::Reg64& reg_astep = st.t[2];
    const Xbyak::Reg64& reg_cstep = st.t[2];
    const Xbyak::Reg64& reg_ksize = st.t[3];
    const Xbyak::Reg64& reg_iterk = st.t[4];
    const Xbyak::Reg64& reg_tmp = st.t[5];
    const Xbyak::Reg64& reg_TmpPtr = st.t[6];
    const Xbyak::Reg64& reg_itern = st.t[7];
    const Xbyak::Reg64& reg_iterm = st.t[8];
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
    mov(reg_ksize.cvt32(), ptr[parambase + OFFSET(k)]);
    xor_(reg_astep, reg_astep);

    xor_(reg_iterm, reg_iterm);
    L(".mloop");
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    mov(reg_astep.cvt32(), dword[parambase + OFFSET(astep)]);
    mov(reg_tmp, reg_iterm);
    imul(reg_tmp, reg_astep);
    add(reg_matAptr, reg_tmp);
    xor_(reg_iterk, reg_iterk);
    for (int i = 0; i < 16; i++) {
      vxorps(Zmm(ZIDX_CReg + i), Zmm(ZIDX_CReg + i));
    }

    L(".kloop");
    for (int i = 0; i < KTile; i += 4) {
      lea(reg_TmpPtr, ptr[reg_matAptr + reg_iterk + i]);
      vmovups(Zmm(ZIDX_BReg + 0), ptr[reg_matBptr + 0 + i * 32]);
      vmovups(Zmm(ZIDX_BReg + 1), ptr[reg_matBptr + 64 + i * 32]);
      for (int j = 0; j < MTile; j++) {
        vpbroadcastd(Zmm(ZIDX_AReg), ptr[reg_TmpPtr]);
        vpdpbusds(Zmm(ZIDX_CReg + j * 2 + 0), Zmm(ZIDX_AReg), Zmm(ZIDX_BReg + 0));
        vpdpbusds(Zmm(ZIDX_CReg + j * 2 + 1), Zmm(ZIDX_AReg), Zmm(ZIDX_BReg + 1));
        add(reg_TmpPtr, reg_astep);
      }
    }
    add(reg_matBptr, KTile * 32);
    add(reg_iterk, KTile);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");

    vbroadcastss(Zmm(ZIDX_ScaleABC), ptr[parambase + OFFSET(scaleAB)]);
    vbroadcastss(Zmm(ZIDX_ScaleC), ptr[parambase + OFFSET(scaleC)]);
    vrcp14ps(Zmm(ZIDX_ScaleC), Zmm(ZIDX_ScaleC));
    vmulps(Zmm(ZIDX_ScaleABC), Zmm(ZIDX_ScaleC));
    vxorps(Zmm(ZIDX_Zero), Zmm(ZIDX_Zero));
    vpbroadcastd(Zmm(ZIDX_FF), ptr[rsp + TValueStart]);
    vpbroadcastd(Zmm(ZIDX_ZeropointC), ptr[parambase + OFFSET(zeropointC)]);

    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    add(reg_matCptr, reg_itern);

    xor_(reg_cstep, reg_cstep);
    mov(reg_cstep.cvt32(), dword[parambase + OFFSET(cstep)]);
    mov(reg_tmp, reg_iterm);
    imul(reg_tmp, reg_cstep);
    add(reg_matCptr, reg_tmp);

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

    add(reg_itern, NTile);
    cmp(reg_itern.cvt32(), ptr[parambase + OFFSET(n)]);
    jb(".nloop");

    add(reg_iterm, MTile);
    cmp(reg_iterm.cvt32(), ptr[parambase + OFFSET(m)]);
    jb(".mloop");

    mov(reg_ret, 0);
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(Xmm(i + 6), xword[rsp + i * 16]);
    }
#endif
    outLocalLabel();  // end of local label

    prepare();
  }

  void prepare() {
    this->ready();
    mKernel = this->getCode<func_t>();
  }

  void forward(uint8_t* matA, int8_t* matB, uint8_t* matC, int _m, int _n, int _k, int _astride, int _cstride,
               float scaleA, float scaleB, float scaleC, int zeropointC) {
    if (mKernel == nullptr) {
      return;
    }
    auto param = params{matA, matB, matC, _m, _n, _k, _astride, _cstride, scaleA * scaleB, scaleC, zeropointC};
    mKernel(&param);
  }

 private:
  func_t mKernel = nullptr;
};

static MHA_vnniw_transpose_1B_N MHA_step1_trans_vnniw(32);
static MHA_s8s8s8_row_vnni_8x32_binary_expf32bf16_allsum MHA_step1NT_8x32_bf16_allsum;
static SeqCopy_1B_avx512_Nx4_Temp SeqVnniCopy32(32);
static MHA_norm_quantize_transpose_bf16 MHA_step2_trans;
static MHA_Matmul_u8s8u8_vnni_byte_8x32 MHA_step3NT_vnni_8x32;

static void forward32_bf16_vnni_NT(const int8_t* Q, const int8_t* K, const int* batch_seq_len, const int8_t* V,
                                   uint8_t* Ret, int batch, int head_num, int k, int seq_len, float scaleQ,
                                   float scaleK, float scaleV, float scaleRet, int zeropointRet, float alpha) {
  auto matA = const_cast<int8_t*>(K);
  auto matB = const_cast<int8_t*>(Q);
  auto matD = const_cast<int8_t*>(V);

#pragma omp parallel for collapse(1)
  for (int ibat = 0; ibat < batch * head_num; ibat++) {
    int _batch = ibat / head_num;
    int _b = ibat % head_num;
    char tempbuffer[512 * 1024];
    float mask[384];
    auto bseq = batch_seq_len[_batch];
    std::fill_n(mask, bseq, 0.f);
    std::fill_n(mask + bseq, seq_len - bseq, -10000.f);
    // step1:
    // step1_reBbuf + matA = step1_expout step1_expsum
    // step2:
    // step1_expout + step1_expsum = step2_reBbuf u8
    // matD = step2_reAbuf s8
    // step3:
    // step2_reAbuf + step2_reBbuf = matE u8
    // MaxMem= 384*384*sizeof(__bf16)+384*sizeof(float)+384*384*sizeof(u8)=432 KB

    auto step1_expout = (__bf16*)(tempbuffer);
    auto step1_expsum = (float*)(step1_expout + seq_len * seq_len);
    auto step2_reAbuf = (uint8_t*)(step1_expsum + seq_len);
    auto step1_reBbuf = (int8_t*)step2_reAbuf;
    auto step2_reBbuf = (int8_t*)step1_expout;  // released

    auto ioffset = _batch * seq_len * head_num * k + _b * k;
    auto abatchptr = matA + ioffset;
    auto bbatchptr = matB + ioffset;
    auto cbatchptr = mask;

    MHA_step1_trans_vnniw.forward(bbatchptr, step1_reBbuf, seq_len, k, head_num * k);

    float scaleAB = scaleQ * scaleK;

    auto& mha_step1 = MHA_step1NT_8x32_bf16_allsum;
    mha_step1.forward(abatchptr, step1_reBbuf, step1_expout, cbatchptr, step1_expsum, seq_len, seq_len, k, head_num * k,
                      scaleAB, alpha);
    // stage 1: exp(out)/sum of exp(out) +  (QK)*V
    auto dbatchptr = matD + ioffset;
    auto& mha_step2 = MHA_step2_trans;
    mha_step2.forward(step1_expout, step2_reAbuf, step1_expsum, seq_len, seq_len);
    for (int in = 0; in < seq_len; in += 4) {
      SeqVnniCopy32.forward(dbatchptr + in * head_num * k, step2_reBbuf + in * SeqVnniCopy32.NTile, head_num * k,
                            seq_len * SeqVnniCopy32.NTile, k);
    }
    // 2nd matmul
    auto ebatchptr = Ret + ioffset;
    auto& mha_step3 = MHA_step3NT_vnni_8x32;
    mha_step3.forward(step2_reAbuf, step2_reBbuf, ebatchptr, seq_len, k, seq_len, seq_len, head_num * k, scaleV,
                      1 / 255.f, scaleRet, zeropointRet);
  }
}

}  // namespace jit_mha_vnni_pad32
