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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_MHA_DENSE_BF16_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_MHA_DENSE_BF16_HPP_

#include <algorithm>
#include <vector>

#include "amx_utils.hpp"
#include "jit_generator.hpp"

namespace jd {
// src rowxcolbytes => dst rowbytespad//64xcolpadx64
// rowpad%N=0
class jit_padding_interleave4b_n : public jit_generator {
 public:
  struct rt_data_t {
    const void* srcptr;
    void* dstptr;
    int row;
    int col;
    int rowpad;
    int colpad;
    int srcstride;
    int dststride;
  };

  explicit jit_padding_interleave4b_n(int _ntile, int _srcbytes)
      : NTile(_ntile), SrcBytes(_srcbytes), RowTile(4 / SrcBytes) {}

 private:
  inline void generate() override {
    inLocalLabel();  // use local label for multiple instance

    int SF_TmpSize = 64;
    Xbyak::util::StackFrame st(this, 1, 12, 16 * 10 + SF_TmpSize);
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_srcptr = st.t[0];
    const Xbyak::Reg64& reg_dstptr = st.t[1];
    const Xbyak::Reg64& reg_srcstride = st.t[2];
    const Xbyak::Reg64& reg_dststride = st.t[3];
    const Xbyak::Reg64& reg_rowpadsize = st.t[4];
    const Xbyak::Reg64& reg_colsize = st.t[5];
    const Xbyak::Reg64& reg_iterrow = st.t[6];
    const Xbyak::Reg64& reg_itercol = st.t[7];
    const Xbyak::Reg64& reg_tmp = st.t[8];
    const Xbyak::Reg64& reg_tmp1 = st.t[9];
    const Xbyak::Reg64& reg_tmp2 = st.t[10];
    const Xbyak::Reg64& reg_colpadsize = st.t[11];
    const Xbyak::Reg64& reg_ret = rax;
    int ZmmEleSize = 64 / SrcBytes;
    auto shuf_masks = std::vector<Opmask>{k5, k6};
    mov(reg_tmp, 0xf0f0);
    kmovd(shuf_masks[0], reg_tmp.cvt32());
    mov(reg_tmp, 0x0f0f);
    kmovd(shuf_masks[1], reg_tmp.cvt32());
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(xword[rsp + i * 16], Xmm(6 + i));
    }
#endif
    mov(reg_srcptr, ptr[parambase + offsetof(rt_data_t, srcptr)]);
    mov(reg_dstptr, ptr[parambase + offsetof(rt_data_t, dstptr)]);
    mov(reg_srcstride.cvt32(), ptr[parambase + offsetof(rt_data_t, srcstride)]);
    mov(reg_dststride.cvt32(), ptr[parambase + offsetof(rt_data_t, dststride)]);

    mov(reg_colsize.cvt32(), ptr[parambase + offsetof(rt_data_t, col)]);

    mov(reg_colpadsize.cvt32(), ptr[parambase + offsetof(rt_data_t, colpad)]);
    mov(reg_rowpadsize.cvt32(), ptr[parambase + offsetof(rt_data_t, rowpad)]);

    int ZIDX_TranSrc = 0;
    int ZIDX_TransTmp = RowTile;
    std::vector<Zmm> reg_srcs(RowTile), reg_tmps(16);
    for (size_t i = 0; i < reg_srcs.size(); i++) {
      reg_srcs[i] = Zmm(ZIDX_TranSrc + i);
    }
    for (size_t i = 0; i < reg_tmps.size(); i++) {
      reg_tmps[i] = Zmm(ZIDX_TransTmp + i);
    }
    int NPerLoop = std::min(NTile, ZmmEleSize);
    int ZMM_PerNTile = std::min(1, NTile / ZmmEleSize);
    int Valid_NReg = NPerLoop / 16;

    xor_(reg_iterrow, reg_iterrow);
    L(".rowloop");
    xor_(reg_itercol, reg_itercol);
    mov(reg_tmp.cvt32(), ptr[parambase + offsetof(rt_data_t, row)]);
    sub(reg_tmp, reg_iterrow);
    cmp(reg_tmp, RowTile);
    jl(".tailloop", T_NEAR);

    L(".colloop");
    for (int i = 0; i < ZMM_PerNTile; i++) {
      int mskidx = 1 + i;
      generate_Nbitsmask(Opmask(mskidx), reg_itercol, reg_colsize, reg_tmp, reg_tmp1, NPerLoop);
      add(reg_itercol, NPerLoop);
    }
    sub(reg_itercol, NTile);
    mov(reg_tmp1, reg_itercol);
    imul(reg_tmp1, reg_dststride);
    lea(reg_tmp, ptr[reg_dstptr + reg_tmp1]);
    for (int i = 0; i < NTile; i += NPerLoop) {
      int mskidx = 1 + (i / NPerLoop);
      lea(reg_tmp1, ptr[reg_srcptr + reg_itercol * SrcBytes]);
      for (int j = 0; j < RowTile; j++) {
        if (SrcBytes == 1) {
          vmovdqu8(reg_srcs[j] | Opmask(mskidx) | T_z, ptr[reg_tmp1 + i * SrcBytes]);
        } else if (SrcBytes == 2) {
          vmovdqu16(reg_srcs[j] | Opmask(mskidx) | T_z, ptr[reg_tmp1 + i * SrcBytes]);
        } else if (SrcBytes == 4) {
          vmovdqu32(reg_srcs[j] | Opmask(mskidx) | T_z, ptr[reg_tmp1 + i * SrcBytes]);
        }
        add(reg_tmp1, reg_srcstride);
      }
      if (SrcBytes == 2) {
        interleave_2rows_4regs(reg_srcs.data(), reg_tmps.data());
      } else if (SrcBytes == 1) {
        assert(false);  // TODO(Yi): unify reorder V
      } else {
      }
      for (int j = 0; j < Valid_NReg; j++) {
        vmovups(ptr[reg_tmp + j * 64 + i * 4], reg_srcs[j]);
      }
    }
    add(reg_itercol, NTile);
    cmp(reg_itercol, reg_colpadsize);
    jb(".colloop");
    add(reg_iterrow, RowTile);
    imul(reg_tmp, reg_srcstride, RowTile);
    lea(reg_srcptr, ptr[reg_srcptr + reg_tmp]);
    lea(reg_dstptr, ptr[reg_dstptr + NTile * 4]);
    jmp(".rowend", T_NEAR);

    L(".tailloop");

    L(".tailcolloop");
    for (int i = 0; i < ZMM_PerNTile; i++) {
      int mskidx = 1 + i;
      generate_Nbitsmask(Opmask(mskidx), reg_itercol, reg_colsize, reg_tmp, reg_tmp1, NPerLoop);
      add(reg_itercol, NPerLoop);
    }
    sub(reg_itercol, NTile);
    mov(reg_tmp1, reg_itercol);
    imul(reg_tmp1, reg_dststride);
    lea(reg_tmp, ptr[reg_dstptr + reg_tmp1]);
    lea(reg_tmp1, ptr[reg_srcptr + reg_itercol * SrcBytes]);
    for (int i = 0; i < NTile; i += NPerLoop) {
      int mskidx = 1 + (i / NPerLoop);
      lea(reg_tmp1, ptr[reg_srcptr + reg_itercol * SrcBytes]);
      mov(reg_tmp2, reg_iterrow);
      for (int j = 0; j < RowTile; j++) vxorps(reg_srcs[j], reg_srcs[j]);
      inLocalLabel();
      for (int j = 0; j < RowTile; j++) {
        cmp(reg_tmp2.cvt32(), ptr[parambase + offsetof(rt_data_t, row)]);
        jge(".tailloop_skip", T_NEAR);
        if (SrcBytes == 1) {
          vmovdqu8(reg_srcs[j] | Opmask(mskidx) | T_z, ptr[reg_tmp1 + i * SrcBytes]);
        } else if (SrcBytes == 2) {
          vmovdqu16(reg_srcs[j] | Opmask(mskidx) | T_z, ptr[reg_tmp1 + i * SrcBytes]);
        } else if (SrcBytes == 4) {
          vmovdqu32(reg_srcs[j] | Opmask(mskidx) | T_z, ptr[reg_tmp1 + i * SrcBytes]);
        }
        add(reg_tmp1, reg_srcstride);
        add(reg_tmp2, 1);
      }
      L(".tailloop_skip");
      outLocalLabel();
      if (SrcBytes == 2) {
        interleave_2rows_4regs(reg_srcs.data(), reg_tmps.data());
      } else if (SrcBytes == 1) {
        assert(false);  // TODO(Yi): unify reorder V
      }
      for (int j = 0; j < Valid_NReg; j++) {
        vmovups(ptr[reg_tmp + j * 64 + i * 4], reg_srcs[j]);
      }
    }
    add(reg_itercol, NTile);
    cmp(reg_itercol, reg_colpadsize);
    jb(".tailcolloop");
    add(reg_iterrow, RowTile);
    lea(reg_dstptr, ptr[reg_dstptr + NTile * 4]);

    L(".rowend");
    cmp(reg_iterrow.cvt32(), reg_rowpadsize);
    jb(".rowloop");

    mov(reg_ret, 0);
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(Xmm(i + 6), xword[rsp + i * 16]);
    }
#endif
    outLocalLabel();  // end of local label
  }

  void generate_Nbitsmask(const Xbyak::Opmask& _msk, const Xbyak::Reg64& _pos, const Xbyak::Reg64& _total,
                          const Xbyak::Reg64& _tmp, const Xbyak::Reg64& _tmp1, int N) {
    inLocalLabel();
    mov(_tmp, _total);
    sub(_tmp, _pos);
    cmp(_tmp, N);
    jb(".maskflag");
    cmp(_tmp, 0);
    jl(".zeroflag");
    uint64_t allmask = ((uint64_t)1 << N) - 1;
    if (N == 64) {
      allmask = (uint64_t)-1;
    }
    mov(_tmp, allmask);
    kmovq(_msk, _tmp);
    jmp(".maskend");
    L(".maskflag");
    mov(_tmp1, 1);
    shlx(_tmp1, _tmp1, _tmp);
    sub(_tmp1, 1);
    kmovq(_msk, _tmp1);
    jmp(".maskend");
    L(".zeroflag");
    mov(_tmp1, 0);
    kmovq(_msk, _tmp1);
    L(".maskend");
    outLocalLabel();
  }
  void interleave_2rows_4regs(Xbyak::Zmm* src_2regs, Xbyak::Zmm* tmp_2reg) {
    vpunpcklwd(tmp_2reg[0], src_2regs[0], src_2regs[1]);
    vpunpckhwd(tmp_2reg[1], src_2regs[0], src_2regs[1]);
    vshuff32x4(src_2regs[0], tmp_2reg[0], tmp_2reg[1], 0 | (1 << 2) | (0 << 4) | (1 << 6));
    vshuff32x4(src_2regs[0], src_2regs[0], src_2regs[0], 0 | (2 << 2) | (1 << 4) | (3 << 6));
    vshuff32x4(src_2regs[1], tmp_2reg[0], tmp_2reg[1], 2 | (3 << 2) | (2 << 4) | (3 << 6));
    vshuff32x4(src_2regs[1], src_2regs[1], src_2regs[1], 0 | (2 << 2) | (1 << 4) | (3 << 6));
  }
  const int NTile;
  const int SrcBytes, RowTile;
};

//! src MxK bf16 => dst M//32 x K//2 x 32 x 2 bf16
class jit_padding_copy2d : public jit_generator {
 public:
  struct rt_data_t {
    const void* srcptr;
    void* dstptr;
    int row;
    int col;
    int rowpad;
    int colpad;
    int srcstride;
    int dststride;
  };

  jit_padding_copy2d() : jit_generator() {}

 private:
  inline void generate() override {
    inLocalLabel();  // use local label for multiple instance
    int SF_TmpSize = 64;
    Xbyak::util::StackFrame st(this, 1, 12, 16 * 10 + SF_TmpSize);
    const Reg64& parambase = st.p[0];
    const Reg64& reg_srcptr = st.t[0];
    const Reg64& reg_dstptr = st.t[1];
    const Reg64& reg_srcstride = st.t[2];
    const Reg64& reg_colsize = st.t[3];
    const Reg64& reg_rowsize = st.t[4];
    const Reg64& reg_itercol = st.t[5];
    const Reg64& reg_iterrow = st.t[6];
    const Reg64& reg_tmp = st.t[7];
    const Reg64& reg_tmp1 = st.t[8];
    const Reg64& reg_dststride = st.t[9];
    const Reg64& reg_colpadsize = st.t[10];
    const Reg64& reg_rowpadsize = st.t[11];
    const Reg64& reg_ret = rax;
    const Opmask& msk_rd = k1;

#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(xword[rsp + i * 16], Xmm(6 + i));
    }
#endif
    mov(reg_srcptr, ptr[parambase + offsetof(rt_data_t, srcptr)]);
    mov(reg_dstptr, ptr[parambase + offsetof(rt_data_t, dstptr)]);
    mov(reg_srcstride.cvt32(), ptr[parambase + offsetof(rt_data_t, srcstride)]);
    mov(reg_dststride.cvt32(), ptr[parambase + offsetof(rt_data_t, dststride)]);

    mov(reg_colsize.cvt32(), ptr[parambase + offsetof(rt_data_t, col)]);
    mov(reg_rowsize.cvt32(), ptr[parambase + offsetof(rt_data_t, row)]);

    mov(reg_colpadsize.cvt32(), ptr[parambase + offsetof(rt_data_t, colpad)]);
    mov(reg_rowpadsize.cvt32(), ptr[parambase + offsetof(rt_data_t, rowpad)]);

    int Level0_RowTile = 8;
    xor_(reg_iterrow, reg_iterrow);

    // TODO(Yu) padding by row
    L(".rowloop");
    xor_(reg_itercol, reg_itercol);

    mov(reg_tmp, reg_rowsize);
    sub(reg_tmp, reg_iterrow);
    cmp(reg_tmp, Level0_RowTile);
    jb(".tailloop", T_NEAR);

    // RowTile=16
    L(".colloop");
    generate_mask(msk_rd, reg_itercol, reg_colsize, reg_tmp, reg_tmp1, 64);
    lea(reg_tmp, ptr[reg_dstptr + reg_itercol]);
    lea(reg_tmp1, ptr[reg_srcptr + reg_itercol]);
    for (int i = 0; i < Level0_RowTile; i++) {
      vmovdqu8(Zmm(i) | msk_rd | T_z, ptr[reg_tmp1]);
      add(reg_tmp1, reg_srcstride);
      vmovups(ptr[reg_tmp], Zmm(i));
      add(reg_tmp, reg_dststride);
    }
    add(reg_itercol, 64);
    cmp(reg_itercol, reg_colpadsize);
    jb(".colloop");
    lea(reg_dstptr, ptr[reg_dstptr + reg_dststride * Level0_RowTile]);
    lea(reg_srcptr, ptr[reg_srcptr + reg_srcstride * Level0_RowTile]);
    add(reg_iterrow, Level0_RowTile);
    jmp(".colend");

    // RowTile=1
    L(".tailloop");
    L(".tailcolloop");
    generate_mask(msk_rd, reg_itercol, reg_colsize, reg_tmp, reg_tmp1, 64);
    lea(reg_tmp, ptr[reg_dstptr + reg_itercol]);
    lea(reg_tmp1, ptr[reg_srcptr + reg_itercol]);
    for (int i = 0; i < 1; i++) {
      vmovdqu8(Zmm(i) | msk_rd | T_z, ptr[reg_tmp1]);
      add(reg_tmp1, reg_srcstride);
      vmovups(ptr[reg_tmp], Zmm(i));
      add(reg_tmp, reg_dststride);
    }
    add(reg_itercol, 64);
    cmp(reg_itercol, reg_colpadsize);
    jb(".tailcolloop");
    lea(reg_dstptr, ptr[reg_dstptr + reg_dststride]);
    lea(reg_srcptr, ptr[reg_srcptr + reg_srcstride]);
    add(reg_iterrow, 1);

    L(".colend");
    cmp(reg_iterrow, reg_rowsize);
    jb(".rowloop");

    mov(reg_ret, 0);

#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(Xmm(i + 6), xword[rsp + i * 16]);
    }
#endif
    outLocalLabel();  // end of local label
  }

 protected:
  inline void generate_mask(const Opmask& _msk, const Reg64& _pos, const Reg64& _total, const Reg64& _tmp,
                            const Reg64& _tmp1, int N) {
    inLocalLabel();
    mov(_tmp, _total);
    sub(_tmp, _pos);
    cmp(_tmp, N);
    jb(".maskflag");
    mov(_tmp, uint64_t(-1));
    kmovq(_msk, _tmp);
    jmp(".maskend");
    L(".maskflag");
    cmp(_tmp, 0);
    jbe(".zeroflag");
    mov(_tmp1, 1);
    shlx(_tmp1, _tmp1, _tmp);
    sub(_tmp1, 1);
    kmovq(_msk, _tmp1);
    jmp(".maskend");
    L(".zeroflag");
    mov(_tmp1, 0);
    kmovq(_msk, _tmp1);
    L(".maskend");
    outLocalLabel();
  }
};

class jit_mha_bf16_row_amx_32x32_softmax : public jit_generator {
 public:
  struct rt_data_t {
    bfloat16_t* matA;
    bfloat16_t* matB;
    bfloat16_t* matC;
    const float* matD;
    int m;
    int n;
    int k;
    int astep;
    int dstep;
    float scaleAB;
  };

  static constexpr int MTile = 32, NTile = 32;
  static constexpr int KTile = 32;

  explicit jit_mha_bf16_row_amx_32x32_softmax(bool has_badd)
      : jit_generator(), binary_add(has_badd), mTC{tile_param_t(16, 16, KTile, true, 2)} {}

 private:
  inline void generate() override {
    // int CMTile = 2;
    int const ZMM_PerROW = NTile / 16;

    inLocalLabel();  // use local label for multiple instance
    int XmmReserve = 16 * (10 + 2);
    int TmmReserve = MTile * NTile * 4;
    int ExpSumReserve = MTile * 16 * 4;  // reserve for Expsum buffer
    int TmpSpace = XmmReserve + TmmReserve + ExpSumReserve;
    int TTmmStart = XmmReserve;
    int TExpsumStart = TTmmStart + TmmReserve;

    Xbyak::Label l_exp_approx_coeff;
    Xbyak::Label l_log2e;
    Xbyak::Label l_ln2;
    Xbyak::Label l_255f;
    {
      Xbyak::util::StackFrame st(this, 1, 13, TmpSpace);

#ifdef _WIN32
      for (int i = 0; i < 10; i++) {
        movaps(xword[rsp + i * 16], Xmm(6 + i));
      }
#endif
      const Reg64& parambase = st.p[0];
      const Reg64& reg_matAptr = st.t[0];
      const Reg64& reg_matBptr = st.t[1];
      const Reg64& reg_astep = st.t[2];
      const Reg64& reg_matCptr = st.t[3];
      const Reg64& reg_matDptr = st.t[4];
      const Reg64& reg_iterk = st.t[5];
      // const Reg64& reg_sumptr = st.t[6];
      const Reg64& reg_itern = st.t[7];
      const Reg64& reg_TmpPtr = st.t[8];
      const Reg64& reg_iterm = st.t[10];
      const Reg64& reg_ksize = st.t[11];
      const Reg64& reg_temp = st.t[12];
      const Reg64& reg_ret = rax;

      // ZMM:
      // [0-15] ZMMs for expsum [16-31] ZMMs for temp regs of transpose
      // [27-31] ZMMs for constant variables
      // [16-26] ZMMs for temperary use
      int ZIDX_LOG2E = 31;
      int ZIDX_LN2 = 30;
      int ZIDX_C0 = 29;
      int ZIDX_C1 = 28;
      int ZIDX_C2 = 27;

      const std::array<Zmm, 3> c = {
          Zmm(ZIDX_C0),
          Zmm(ZIDX_C1),
          Zmm(ZIDX_C2),
      };
      int ZIDX_TMP = 25;
      const std::array<Zmm, 2> tmp = {Zmm(ZIDX_TMP), Zmm(ZIDX_TMP + 1)};
      int ZIDX_ExpSum = 0;
      // int ZIDX_CReg = 16;
      std::array<Zmm, 16> regs_trans_src, regs_trans_tmp;
      for (int i = 0; i < 16; i++) {
        regs_trans_src[i] = Zmm(i);
        regs_trans_tmp[i] = Zmm(16 + i);
      }

      mov(reg_temp, reinterpret_cast<uint64_t>(&mTC));
      ldtilecfg(ptr[reg_temp]);

      xor_(reg_astep, reg_astep);

      mov(reg_ksize.cvt32(), dword[parambase + offsetof(rt_data_t, k)]);

      xor_(reg_iterm, reg_iterm);
      L(".mloop");
      for (int i = 0; i < 16; i++) {
        vxorps(Zmm(ZIDX_ExpSum + i), Zmm(ZIDX_ExpSum + i));
      }
      for (int i = 0; i < MTile; i++) {
        vmovups(ptr[rsp + TExpsumStart + i * 64], Zmm(ZIDX_ExpSum));
      }

      vbroadcastss(Zmm(ZIDX_LOG2E), ptr[rip + l_log2e]);
      vbroadcastss(Zmm(ZIDX_LN2), ptr[rip + l_ln2]);
      vbroadcastss(Zmm(ZIDX_C0), ptr[rip + l_exp_approx_coeff]);
      vbroadcastss(Zmm(ZIDX_C1), ptr[rip + l_exp_approx_coeff + 4]);
      vbroadcastss(Zmm(ZIDX_C2), ptr[rip + l_exp_approx_coeff + 8]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      mov(reg_matBptr, ptr[parambase + offsetof(rt_data_t, matB)]);
      {
        auto reg_tmp = reg_iterk;
        mov(reg_tmp, reg_ksize);
        imul(reg_tmp, reg_itern);
        lea(reg_matBptr, ptr[reg_matBptr + reg_tmp * sizeof(bfloat16_t)]);
      }
      mov(reg_matAptr, ptr[parambase + offsetof(rt_data_t, matA)]);
      mov(reg_astep.cvt32(), dword[parambase + offsetof(rt_data_t, astep)]);
      {  // reg_matAptr = reg_matAptr + reg_iterm * reg_astep
        auto reg_tmp = reg_iterk;
        mov(reg_tmp, reg_astep);
        imul(reg_tmp, reg_iterm);
        lea(reg_matAptr, ptr[reg_matAptr + reg_tmp]);
      }

      mov(reg_temp, NTile * 4);  // b,c stride
      tile_product_amx_bf16ps(reg_ksize, reg_matAptr, reg_matBptr, reg_astep, reg_temp, reg_iterk, reg_TmpPtr,
                              [&](int i, int j) { return ptr[rsp + reg_temp + TTmmStart + i * 16 * 64 * 2 + j * 64]; });

      if (binary_add) {
        mov(reg_matDptr.cvt32(), dword[parambase + offsetof(rt_data_t, dstep)]);
        imul(reg_matDptr, reg_iterm);
        add(reg_matDptr, qword[parambase + offsetof(rt_data_t, matD)]);
        lea(reg_matDptr, ptr[reg_matDptr + reg_itern * sizeof(float)]);
      }
      auto reg_cstep = reg_astep;
      imul(reg_cstep.cvt32(), dword[parambase + offsetof(rt_data_t, n)], sizeof(bfloat16_t));
      mov(reg_matCptr, ptr[parambase + offsetof(rt_data_t, matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(bfloat16_t)]);
      {  // reg_matCptr += reg_cstep * reg_iterm
        auto reg_tmp = reg_iterk;
        mov(reg_tmp, reg_iterm);
        imul(reg_tmp.cvt32(), reg_cstep);
        add(reg_matCptr, reg_tmp);
      }

      // calc f32 exp, accumulate exp to buffer
      for (int in = 0; in < NTile; in += 16) {
        lea(reg_TmpPtr, ptr[reg_matCptr + in * sizeof(bfloat16_t)]);
        for (int i = 0; i < MTile; i += 16) {
          for (int j = 0; j < 16; j++) {
            auto reg_tmp = reg_iterk;
            if (binary_add)
              (i == 0 && j == 0) ? xor_(reg_tmp.cvt32(), reg_tmp.cvt32())
                                 : add(reg_tmp.cvt32(), dword[parambase + offsetof(rt_data_t, dstep)]);
            if (binary_add) vmovups(regs_trans_tmp[0], zword[reg_matDptr + reg_tmp + in * 4]);

            vmovups(regs_trans_src[j], ptr[rsp + TTmmStart + in * 4 + (i + j) * NTile * 4]);
            !binary_add  // (optionally) add mask and scale
                ? vmulps(regs_trans_src[j], regs_trans_src[j], zword_b[parambase + offsetof(rt_data_t, scaleAB)])
                : vfmadd132ps(regs_trans_src[j], regs_trans_tmp[0], zword_b[parambase + offsetof(rt_data_t, scaleAB)]);
            exp_approx_f32(regs_trans_src[j], regs_trans_src[j], Zmm(ZIDX_LOG2E), Zmm(ZIDX_LN2), c, tmp);
            vpsrld(regs_trans_tmp[1], regs_trans_src[j], 16);
            vpmovdw(ptr[reg_TmpPtr], regs_trans_tmp[1]);
            vaddps(regs_trans_src[j], regs_trans_src[j], ptr[rsp + TExpsumStart + (i + j) * 64]);
            vmovups(ptr[rsp + TExpsumStart + (i + j) * 64], regs_trans_src[j]);
            add(reg_TmpPtr, reg_cstep);
          }
        }
      }
      add(reg_itern, NTile);
      cmp(reg_itern.cvt32(), dword[parambase + offsetof(rt_data_t, n)]);
      jb(".nloop");

      // normalize all temp exp values
      for (int im = 0; im < MTile; im += 16) {
        for (int j = 0; j < 16; j++) {
          vmovups(regs_trans_src[j], ptr[rsp + TExpsumStart + (im + j) * 64]);
        }
        transpose_16x16_ps(regs_trans_src, regs_trans_tmp);
        for (int i = 0; i < 16; i += 2) {
          vaddps(regs_trans_src[i], regs_trans_src[i], regs_trans_src[i + 1]);
        }
        for (int i = 0; i < 16; i += 4) {
          vaddps(regs_trans_src[i], regs_trans_src[i], regs_trans_src[i + 2]);
        }
        for (int i = 0; i < 16; i += 8) {
          vaddps(regs_trans_src[i], regs_trans_src[i], regs_trans_src[i + 4]);
        }
        for (int i = 0; i < 16; i += 16) {
          vaddps(regs_trans_src[i], regs_trans_src[i], regs_trans_src[i + 8]);
        }
        vrcp14ps(regs_trans_src[0], regs_trans_src[0]);
        vmovups(ptr[rsp + TExpsumStart + im * 4], regs_trans_src[0]);
      }

      imul(reg_cstep.cvt32(), dword[parambase + offsetof(rt_data_t, n)], sizeof(bfloat16_t));
      xor_(reg_itern, reg_itern);
      L(".nwrloop");
      mov(reg_matCptr, ptr[parambase + offsetof(rt_data_t, matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(bfloat16_t)]);
      {
        auto reg_tmp = reg_iterk;
        mov(reg_tmp, reg_iterm);
        imul(reg_tmp.cvt32(), reg_cstep);
        add(reg_matCptr, reg_tmp);
      }
      for (int i = 0; i < MTile; i++) {
        vbroadcastss(regs_trans_tmp[0], ptr[rsp + TExpsumStart + i * 4]);
        for (int in = 0; in < ZMM_PerROW; in++) {
          vpmovzxwd(regs_trans_src[in], ptr[reg_matCptr + in * 32]);
          vpslld(regs_trans_src[in], regs_trans_src[in], 16);
          vmulps(regs_trans_src[in], regs_trans_src[in], regs_trans_tmp[0]);
        }
        for (int in = 0; in < ZMM_PerROW; in += 2) {
          vcvtne2ps2bf16(regs_trans_src[in], regs_trans_src[in + 1], regs_trans_src[in]);
          vmovups(ptr[reg_matCptr + in * 32], regs_trans_src[in]);
        }
        add(reg_matCptr, reg_cstep);
      }
      add(reg_itern, NTile);
      cmp(reg_itern.cvt32(), dword[parambase + offsetof(rt_data_t, n)]);
      jb(".nwrloop");

      add(reg_iterm, MTile);
      cmp(reg_iterm.cvt32(), dword[parambase + offsetof(rt_data_t, m)]);
      jb(".mloop");

      mov(reg_ret, 0);

#ifdef _WIN32
      for (int i = 0; i < 10; i++) {
        movaps(Xmm(i + 6), xword[rsp + i * 16]);
      }
#endif
    }
    outLocalLabel();  // end of local label
    L(l_log2e);
    db(bit_cast<uint32_t>(std::log2f(std::exp(1.f))), sizeof(float));
    L(l_ln2);
    db(bit_cast<uint32_t>(std::log(2.f)), sizeof(float));
    L(l_exp_approx_coeff);
    db(reinterpret_cast<const uint8_t*>(exp_approx_f32_coeff.data()), sizeof(exp_approx_f32_coeff));
    L(l_255f);
    db(bit_cast<uint32_t>(255.f), sizeof(float));
  }

  const bool binary_add;
  const tileconfig_t mTC;
};

class jit_mha_bf16_row_amx_32x32 : public jit_generator {
 public:
  struct rt_data_t {
    const bfloat16_t* matA;
    const bfloat16_t* matB;
    bfloat16_t* matC;
    int m;
    int n;
    int k;
    int astep;
    int cstep;
    float alpha;
  };

  static constexpr int MTile = 32, NTile = 32;
  static constexpr int KTile = 32;

  jit_mha_bf16_row_amx_32x32() : jit_generator(), mTC{tile_param_t(16, 16, KTile, true, 2)} {}

 private:
  inline void generate() override {
    constexpr int ZMM_PerROW = NTile / 16;

    inLocalLabel();  // use local label for multiple instance
    constexpr int XmmReserve = 16 * (10 + 2);
    constexpr int TmmReserve = MTile * NTile * 4;
    constexpr int TmpValueReserve = 64;
    constexpr int TmpSpace = XmmReserve + TmmReserve + TmpValueReserve;
    constexpr int TTmmStart = XmmReserve;
    // int TValueStart = TTmmStart + TmmReserve;
    Xbyak::Label tmpfvariable;
    {
      Xbyak::util::StackFrame st(this, 1, 13, TmpSpace);
      mov(ptr[rsp], rdi);

#ifdef _WIN32
      for (int i = 0; i < 10; i++) {
        movaps(xword[rsp + i * 16], Xmm(6 + i));
      }
#endif
      const Reg64& parambase = st.p[0];
      const Reg64& reg_matAptr = st.t[0];
      const Reg64& reg_matBptr = st.t[1];
      const Reg64& reg_astep = st.t[2];
      const Reg64& reg_matCptr = st.t[3];
      const Reg64& reg_iterk = st.t[5];
      // const Reg64& reg_sumptr = st.t[6];
      const Reg64& reg_itern = st.t[7];
      const Reg64& reg_TmpPtr = st.t[8];
      const Reg64& reg_iterm = st.t[10];
      const Reg64& reg_ksize = st.t[11];
      const Reg64& reg_temp = st.t[12];
      const Reg64& reg_ret = rax;
      const Opmask& mask0 = k1;

      mov(reg_temp, reinterpret_cast<uint64_t>(&mTC));
      ldtilecfg(ptr[reg_temp]);

      int ZIDX_CReg = 0;

      mov(reg_ksize.cvt32(), dword[parambase + offsetof(rt_data_t, k)]);

      xor_(reg_iterm, reg_iterm);
      L(".mloop");

      xor_(reg_itern, reg_itern);
      L(".nloop");
      mov(reg_matBptr, ptr[parambase + offsetof(rt_data_t, matB)]);
      {
        auto reg_tmp = reg_iterk;
        mov(reg_tmp, reg_ksize);
        imul(reg_tmp, reg_itern);
        lea(reg_matBptr, ptr[reg_matBptr + reg_tmp * sizeof(bfloat16_t)]);
      }
      mov(reg_matAptr, ptr[parambase + offsetof(rt_data_t, matA)]);
      mov(reg_astep.cvt32(), dword[parambase + offsetof(rt_data_t, astep)]);
      {  // reg_matAptr = matA + reg_iterm * reg_astep
        auto reg_tmp = reg_iterk;
        mov(reg_tmp, reg_astep);
        imul(reg_tmp, reg_iterm);
        lea(reg_matAptr, ptr[reg_matAptr + reg_tmp]);
      }
      mov(reg_temp, NTile * 4);  // b,c stride
      tile_product_amx_bf16ps(reg_ksize, reg_matAptr, reg_matBptr, reg_astep, reg_temp, reg_iterk, reg_TmpPtr,
                              [&](int i, int j) { return ptr[rsp + reg_temp + TTmmStart + i * 16 * 64 * 2 + j * 64]; });

      auto reg_cstep = reg_astep;
      mov(reg_cstep.cvt32(), dword[parambase + offsetof(rt_data_t, cstep)]);
      mov(reg_matCptr, ptr[parambase + offsetof(rt_data_t, matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(bfloat16_t)]);
      {
        auto reg_tmp = reg_iterk;
        mov(reg_tmp, reg_iterm);
        imul(reg_tmp.cvt32(), reg_cstep);
        add(reg_matCptr, reg_tmp);
      }
      {
        auto reg_tmp = reg_iterk;
        mov(reg_tmp.cvt32(), dword[parambase + offsetof(rt_data_t, n)]);
        sub(reg_tmp, reg_itern);
        cmp(reg_tmp, NTile);
        jb(".maskflag");
        mov(reg_temp, 0xffff);
        kmovq(mask0, reg_temp);
        jmp(".maskend");
        L(".maskflag");
        shr(reg_tmp, 1);
        mov(reg_temp, 1);
        shlx(reg_temp, reg_temp, reg_tmp);
        sub(reg_temp, 1);
        kmovq(mask0, reg_temp);
        L(".maskend");
      }

      for (int i = 0; i < MTile; i++) {
        int zidx = 0;
        for (int j = 0; j < ZMM_PerROW; j++) {
          vmovups(Zmm(ZIDX_CReg + zidx * ZMM_PerROW + j), ptr[rsp + TTmmStart + j * 64 + i * NTile * 4]);
        }
        for (int j = 0; j < ZMM_PerROW; j += 2) {
          vcvtne2ps2bf16(Zmm(ZIDX_CReg + zidx * ZMM_PerROW + j), Zmm(ZIDX_CReg + zidx * ZMM_PerROW + j + 1),
                         Zmm(ZIDX_CReg + zidx * ZMM_PerROW + j));
          vmovups(ptr[reg_matCptr + j * 32] | mask0, Zmm(ZIDX_CReg + zidx * ZMM_PerROW + j));
        }
        add(reg_matCptr, reg_cstep);
      }

      add(reg_itern, NTile);
      cmp(reg_itern.cvt32(), dword[parambase + offsetof(rt_data_t, n)]);
      jb(".nloop");

      add(reg_iterm, MTile);
      cmp(reg_iterm.cvt32(), dword[parambase + offsetof(rt_data_t, m)]);
      jb(".mloop");
      mov(reg_ret, 0);

#ifdef _WIN32
      for (int i = 0; i < 10; i++) {
        movaps(Xmm(i + 6), xword[rsp + i * 16]);
      }
#endif
    }
    outLocalLabel();  // end of local label
  }

  const tileconfig_t mTC;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_MHA_DENSE_BF16_HPP_
