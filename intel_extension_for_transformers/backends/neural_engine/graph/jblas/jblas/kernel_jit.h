#pragma once
#include "jit_blas_utils.h"
#include "jit_base.hpp"
namespace jblas {
namespace kernel {
namespace jit {

class DequanS8F32 {
 public:
  class MicroKernelAVX512F : protected jblas::xbyak::JitAvx512f {
   public:
    struct params {
      void *srcptr, *dstptr;
      int row, col;
      int srcstride, dststride;
      float *scales;
    };
    typedef long long (*func_t)(params *);
    static int constexpr VBytes = 64;
    static int constexpr RegScale = 0;
    static int constexpr RegTmp = RegScale + 4;
    MicroKernelAVX512F() {
      generate();
      this->ready();
      mKernel = this->getCode<func_t>();
    }

    void generate() {
      inLocalLabel();  // use local label for multiple instance
      int SF_TmpSize = 64;
      int SF_TmpPos = 16 * 10;
      Xbyak::util::StackFrame st(this, 1, 13, 16 * 10 + SF_TmpSize);
      parambase = st.p[0];
      reg_srcptr = st.t[0];
      reg_dstptr = st.t[1];
      reg_srcstride = st.t[2];
      reg_dststride = st.t[3];
      reg_rowsize = st.t[4];
      reg_colsize = st.t[5];
      reg_iterrow = st.t[6];
      reg_itercol = st.t[7];
      reg_tmp = st.t[8];
      reg_scaleptr = st.t[9];
      reg_tmpdst = st.t[10];
      reg_tmp1 = st.t[12];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
      mov(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
      mov(reg_scaleptr, ptr[parambase + OFFSET(scales)]);
      xor_(reg_srcstride, reg_srcstride);
      mov(reg_srcstride.cvt32(), ptr[parambase + OFFSET(srcstride)]);
      xor_(reg_dststride, reg_dststride);
      mov(reg_dststride.cvt32(), ptr[parambase + OFFSET(dststride)]);

      load32(reg_colsize, ptr[parambase + OFFSET(col)]);
      load32(reg_rowsize, ptr[parambase + OFFSET(row)]);
      xor_(reg_itercol, reg_itercol);

      L(".colloop");
      mov(reg_tmp, reg_colsize);
      sub(reg_tmp, reg_itercol);
      cmp(reg_tmp, 64);
      jl(".proc48", T_NEAR);
      generateNTile(4);
      add(reg_itercol, 64);
      add(reg_srcptr, 1 * 64);
      add(reg_dstptr, 4 * 64);
      add(reg_scaleptr, 4 * 64);
      jmp(".colend", T_NEAR);

      L(".proc48");
      cmp(reg_tmp, 48);
      jl(".proc32", T_NEAR);
      generateNTile(3);
      add(reg_itercol, 48);
      add(reg_srcptr, 1 * 48);
      add(reg_dstptr, 4 * 48);
      add(reg_scaleptr, 4 * 48);
      jmp(".colend", T_NEAR);

      L(".proc32");
      generateNTile(2);
      add(reg_itercol, 32);
      add(reg_srcptr, 1 * 32);
      add(reg_dstptr, 4 * 32);
      add(reg_scaleptr, 4 * 32);

      L(".colend");
      cmp(reg_itercol, reg_colsize);
      jb(".colloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);
      outLocalLabel();  // end of local label
    }

    void generateNTile(int N) {
      for (size_t i = 0; i < N; i++) {
        vmovups(Xbyak::Zmm(RegScale + i), ptr[reg_scaleptr + i * 64]);
      }
      inLocalLabel();
      xor_(reg_iterrow, reg_iterrow);
      mov(reg_tmp, reg_srcptr);
      mov(reg_tmp1, reg_dstptr);
      L(".rowloop");
      for (int i = 0; i < N; i++) {
        vpmovsxbd(Xbyak::Zmm(RegTmp), ptr[reg_tmp + i * 16]);
        vcvtdq2ps(Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegTmp));
        vmulps(Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegScale + i));
        vmovups(ptr[reg_tmp1 + i * 64], Xbyak::Zmm(RegTmp));
      }
      add(reg_tmp, reg_srcstride);
      add(reg_tmp1, reg_dststride);
      add(reg_iterrow, 1);
      cmp(reg_iterrow, reg_rowsize);
      jb(".rowloop");
      outLocalLabel();
    }
    func_t mKernel = nullptr;

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_srcptr;
    Xbyak::Reg64 reg_dstptr;
    Xbyak::Reg64 reg_srcstride;
    Xbyak::Reg64 reg_dststride;
    Xbyak::Reg64 reg_rowsize;
    Xbyak::Reg64 reg_colsize;
    Xbyak::Reg64 reg_iterrow;
    Xbyak::Reg64 reg_itercol;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_scaleptr;
    Xbyak::Reg64 reg_tmpdst;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_ret;
  };
  static void forward_avx512f(int8_t *srcptr, float *dstptr, int row, int col,
                              int ld_src, int ld_dst, float *scales) {
    static MicroKernelAVX512F mAVX512F;
    auto param = MicroKernelAVX512F::params{srcptr,
                                            dstptr,
                                            row,
                                            col,
                                            int(ld_src * sizeof(int8_t)),
                                            int(ld_dst * sizeof(float)),
                                            scales};
    mAVX512F.mKernel(&param);
  }
};

class JitMemcpy2DAvx512f : protected jblas::xbyak::JitAvx512f {
 public:
  struct params {
    void *srcptr, *dstptr;
    int row, col;
    int srcstride, dststride;
  };
  typedef long long (*func_t)(params *);

 public:
  static int constexpr VBytes = 64;
  JitMemcpy2DAvx512f() {
    inLocalLabel();  // use local label for multiple instance
    int SF_TmpSize = 64;
    int SF_TmpPos = 16 * 10;
    Xbyak::util::StackFrame st(this, 1, 13, 16 * 10 + SF_TmpSize);
    const Xbyak::Reg64 &parambase = st.p[0];
    const Xbyak::Reg64 &reg_srcptr = st.t[0];
    const Xbyak::Reg64 &reg_dstptr = st.t[1];
    const Xbyak::Reg64 &reg_srcstride = st.t[2];
    const Xbyak::Reg64 &reg_dststride = st.t[3];
    const Xbyak::Reg64 &reg_rowsize = st.t[4];
    const Xbyak::Reg64 &reg_colsize = st.t[5];
    const Xbyak::Reg64 &reg_iterrow = st.t[6];
    const Xbyak::Reg64 &reg_itercol = st.t[7];
    const Xbyak::Reg64 &reg_tmp = st.t[8];
    const Xbyak::Reg64 &reg_tmpsrc = st.t[9];
    const Xbyak::Reg64 &reg_tmpdst = st.t[10];
    const Xbyak::Reg64 &reg_tmp1 = st.t[12];
    const Xbyak::Reg64 &reg_ret = rax;

    vreg_push(rsp);

    mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
    mov(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
    xor_(reg_srcstride, reg_srcstride);
    mov(reg_srcstride.cvt32(), ptr[parambase + OFFSET(srcstride)]);
    xor_(reg_dststride, reg_dststride);
    mov(reg_dststride.cvt32(), ptr[parambase + OFFSET(dststride)]);

    load32(reg_colsize, ptr[parambase + OFFSET(col)]);
    load32(reg_rowsize, ptr[parambase + OFFSET(row)]);

    int const ColUnroll = 4;
    xor_(reg_iterrow, reg_iterrow);
    L(".rowloop");
    xor_(reg_itercol, reg_itercol);
    mov(reg_tmpsrc, reg_srcptr);
    mov(reg_tmpdst, reg_dstptr);
    L(".colloop");
    mov(reg_tmp, reg_colsize);
    sub(reg_tmp, reg_itercol);
    cmp(reg_tmp, ColUnroll * VBytes);
    jl(".maskproc", T_NEAR);

    for (int i = 0; i < ColUnroll; i++) {
      vmovups(Xbyak::Zmm(i), ptr[reg_srcptr + reg_itercol + i * VBytes]);
      vmovups(ptr[reg_dstptr + reg_itercol + i * VBytes], Xbyak::Zmm(i));
    }
    add(reg_itercol, ColUnroll * VBytes);
    jmp(".colend");
    L(".maskproc");
    generate_Nbitsmask(k1, reg_itercol, reg_colsize, reg_tmp, reg_tmp1, VBytes);
    vmovdqu8(Xbyak::Zmm(0) | k1, ptr[reg_srcptr + reg_itercol]);
    vmovdqu8(ptr[reg_dstptr + reg_itercol], Xbyak::Zmm(0) | k1);
    add(reg_itercol, VBytes);
    L(".colend");
    cmp(reg_itercol, reg_colsize);
    jb(".colloop");
    add(reg_iterrow, 1);
    lea(reg_srcptr, ptr[reg_srcptr + reg_srcstride]);
    lea(reg_dstptr, ptr[reg_dstptr + reg_dststride]);
    cmp(reg_iterrow, reg_rowsize);
    jb(".rowloop");

    mov(reg_ret, 0);
    vreg_pop(rsp);
    outLocalLabel();  // end of local label

    this->ready();
    mKernel = this->getCode<func_t>();
  }

  static JBLAS_CODE forward(void *srcptr, void *dstptr, int row, int col,
                      int srcstride, int dststride) {
    static JitMemcpy2DAvx512f instance;
    auto param = params{srcptr, dstptr, row, col, srcstride, dststride};
    instance.mKernel(&param);
    return JblasSuccess;
  }

 private:
  func_t mKernel = nullptr;
};

}  // namespace jit
}  // namespace kernel
}  // namespace jblas