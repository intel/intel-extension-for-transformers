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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MHA_DENSE_BF16_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MHA_DENSE_BF16_HPP_

#include <algorithm>
#include <vector>
#include <memory>

#include "kernels/amx_utils.hpp"
#include "jit_generator.hpp"
#include "regs_pool.hpp"

/// @note jit classes in this file may not meet the coding convention and is to be refactored.

namespace jd {
// src row x colbytes => dst rowbytespad//64xcolpadx64
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
      : NTile(_ntile), SrcBytes(_srcbytes), RowTile(4 / SrcBytes) {
    SPARSE_LOG_IF(FATAL, SrcBytes != 1 && SrcBytes != 2) << "Unexpected SrcBytes!";
  }

 private:
  void process_row_tile(regs_pool* const rp, const Reg64 src_ptr, const Reg64 dst_ptr, const Reg64 src_stride,
                        const Reg64 dst_stride, const Reg64 iter_row, const Reg64 col_size, bool is_row_tail) {
    const auto ZmmEleSize = 64 / SrcBytes;
    const auto NPerLoop = std::min(NTile, ZmmEleSize);
    const auto ZMM_PerNTile = std::min(1, NTile / ZmmEleSize);
    const auto Valid_NReg = NPerLoop / 16;
    const auto& reg_itercol = rp->reg<Reg64>();
    xor_(reg_itercol.cvt32(), reg_itercol.cvt32());
    Xbyak::Label l_col_loop;
    L(l_col_loop);
    const auto masks = rp->regs<Opmask>(ZMM_PerNTile);
    for (int i = 0; i < ZMM_PerNTile; i++) {
      auto curr_itercol = rp->reg<Reg64>();  // curr_itercol = reg_itercol + i * NPerLoop;
      lea(curr_itercol, ptr[reg_itercol + i * NPerLoop]);
      generate_Nbitsmask(masks[i], curr_itercol, col_size, rp->reg<Reg64>(), rp->reg<Reg64>(), NPerLoop);
    }
    const auto& curr_dst = rp->reg<Reg64>();
    mov(curr_dst, reg_itercol);
    imul(curr_dst, dst_stride);
    lea(curr_dst, ptr[dst_ptr + curr_dst]);
    for (int i = 0; i < NTile; i += NPerLoop) {
      const auto reg_srcs = rp->regs<Zmm>(RowTile);
      if (is_row_tail) {
        for (int j = 0; j < RowTile; j++) vxorps(reg_srcs[j], reg_srcs[j]);
      }

      const auto curr_src = rp->reg<Reg64>();  // curr_src := reg_itercol * SrcBytes + j * reg_srcstride
      Xbyak::Label l_skip_row;
      for (int j = 0; j < RowTile; j++) {
        if (is_row_tail) {
          const auto& curr_iter_row = rp->reg<Reg64>();
          lea(curr_iter_row, ptr[iter_row + j]);
          cmp(curr_iter_row.cvt32(), dword[rp->p[0] + offsetof(rt_data_t, row)]);
          jge(l_skip_row, T_NEAR);
        }

        (j == 0) ? lea(curr_src, ptr[src_ptr + reg_itercol * SrcBytes]) : lea(curr_src, ptr[curr_src + src_stride]);
        if (SrcBytes == 1) {
          vmovdqu8(reg_srcs[j] | masks[i / NPerLoop] | T_z, zword[curr_src + i * SrcBytes]);
        } else if (SrcBytes == 2) {
          vmovdqu16(reg_srcs[j] | masks[i / NPerLoop] | T_z, zword[curr_src + i * SrcBytes]);
        }
      }
      if (is_row_tail) L(l_skip_row);
      if (SrcBytes == 1) {
        assert(false);  // TODO(Yi): unify reorder V
      } else if (SrcBytes == 2) {
        interleave_2rows_4regs(reg_srcs, rp->regs<Zmm>(16));
      } else {
        assert(false);
      }
      for (int j = 0; j < Valid_NReg; j++) vmovaps(ptr[curr_dst + j * BYTES_ZMM + i * 4], reg_srcs[j]);
    }
    lea(reg_itercol, ptr[reg_itercol + NTile]);
    cmp(reg_itercol, col_size);
    jb(l_col_loop);
  }

  inline void generate() override {
    inLocalLabel();  // use local label for multiple instance
    const int ZMM_PerNTile = ceil_div(NTile * SrcBytes, BYTES_ZMM);
    const int vmm_used = 16 + RowTile;
    regs_pool rp(this, 1, {11, vmm_used, ZMM_PerNTile});
    const auto& reg_srcptr = rp.reg<Reg64>();
    const auto& reg_dstptr = rp.reg<Reg64>();
    const auto& reg_srcstride = rp.reg<Reg64>();
    const auto& reg_dststride = rp.reg<Reg64>();
    const auto& reg_colsize = rp.reg<Reg64>();
    mov(reg_srcptr, ptr[rp.p[0] + offsetof(rt_data_t, srcptr)]);
    mov(reg_dstptr, ptr[rp.p[0] + offsetof(rt_data_t, dstptr)]);
    mov(reg_srcstride.cvt32(), ptr[rp.p[0] + offsetof(rt_data_t, srcstride)]);
    mov(reg_dststride.cvt32(), ptr[rp.p[0] + offsetof(rt_data_t, dststride)]);
    mov(reg_colsize.cvt32(), ptr[rp.p[0] + offsetof(rt_data_t, col)]);

    Xbyak::Label l_row_loop;
    const auto& reg_iterrow = rp.reg<Reg64>();
    xor_(reg_iterrow.cvt32(), reg_iterrow.cvt32());
    L(l_row_loop);

    const auto& reg_itercol = rp.reg<Reg64>();
    {  // if (reg_iterrow + RowTile > row) jmp to tail
      const auto& reg_tmp = rp.reg<Reg64>();
      lea(reg_tmp, ptr[reg_iterrow + RowTile]);
      cmp(reg_tmp.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, row)]);
    }
    jg(".tailloop", T_NEAR);

    process_row_tile(&rp, reg_srcptr, reg_dstptr, reg_srcstride, reg_dststride, reg_iterrow, reg_colsize, false);
    {  // reg_srcptr += RowTile * reg_srcstride
      const auto& reg_tmp = rp.reg<Reg64>();
      imul(reg_tmp, reg_srcstride, RowTile);
      lea(reg_srcptr, ptr[reg_srcptr + reg_tmp]);
    }
    jmp(".rowend", T_NEAR);

    L(".tailloop");
    process_row_tile(&rp, reg_srcptr, reg_dstptr, reg_srcstride, reg_dststride, reg_iterrow, reg_colsize, true);

    L(".rowend");
    lea(reg_dstptr, ptr[reg_dstptr + NTile * 4]);
    add(reg_iterrow, RowTile);
    cmp(reg_iterrow.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, rowpad)]);
    jb(l_row_loop);

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
  void interleave_2rows_4regs(std::vector<Xbyak::Zmm> src_2regs, std::vector<Xbyak::Zmm> tmp_2reg) {
    vpunpcklwd(tmp_2reg[0], src_2regs[0], src_2regs[1]);
    vpunpckhwd(tmp_2reg[1], src_2regs[0], src_2regs[1]);
    vshuff32x4(src_2regs[0], tmp_2reg[0], tmp_2reg[1], 0 | (1 << 2) | (0 << 4) | (1 << 6));
    vshuff32x4(src_2regs[0], src_2regs[0], src_2regs[0], 0 | (2 << 2) | (1 << 4) | (3 << 6));
    vshuff32x4(src_2regs[1], tmp_2reg[0], tmp_2reg[1], 2 | (3 << 2) | (2 << 4) | (3 << 6));
    vshuff32x4(src_2regs[1], src_2regs[1], src_2regs[1], 0 | (2 << 2) | (1 << 4) | (3 << 6));
  }
  const int NTile;  // in terms of #elemetns
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
    mov(reg_srcptr, ptr[st.p[0] + offsetof(rt_data_t, srcptr)]);
    mov(reg_dstptr, ptr[st.p[0] + offsetof(rt_data_t, dstptr)]);
    mov(reg_srcstride.cvt32(), ptr[st.p[0] + offsetof(rt_data_t, srcstride)]);
    mov(reg_dststride.cvt32(), ptr[st.p[0] + offsetof(rt_data_t, dststride)]);

    mov(reg_colsize.cvt32(), ptr[st.p[0] + offsetof(rt_data_t, col)]);
    mov(reg_rowsize.cvt32(), ptr[st.p[0] + offsetof(rt_data_t, row)]);

    mov(reg_colpadsize.cvt32(), ptr[st.p[0] + offsetof(rt_data_t, colpad)]);
    mov(reg_rowpadsize.cvt32(), ptr[st.p[0] + offsetof(rt_data_t, rowpad)]);

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

  explicit jit_mha_bf16_row_amx_32x32_softmax(bool has_badd, bool stable_softmax, const tile_param_t* pre_amx_cfg)
      : jit_generator(),
        binary_add(has_badd),
        stable_softmax(stable_softmax),
        pre_amx_cfg_(pre_amx_cfg),
        required_amx_cfg_{16, 16, KTile, true, 2},
        required_tile_cfg_(required_amx_cfg_) {}

 private:
  inline void generate() override {
    bool need_cfg_amx = pre_amx_cfg_ != nullptr && *pre_amx_cfg_ != required_amx_cfg_;
    std::shared_ptr<void> use_loacl_label = {(inLocalLabel(), nullptr), [&](...) { outLocalLabel(); }};

    int const ZMM_PerROW = NTile / 16;
    const int amx_config_size = need_cfg_amx ? sizeof(tileconfig_t) : 0;
    const int TmmReserve = MTile * NTile * sizeof(float);
    const int ExpSumReserve = MTile * BYTES_ZMM;  // reserve for Expsum buffer
    const int MaxValReserve = MTile * BYTES_ZMM;  // reserve for max_val buffer
    const int TmpSpace = amx_config_size + TmmReserve + ExpSumReserve + MaxValReserve;
    const int TTmmStart = amx_config_size;
    const int TExpsumStart = TTmmStart + TmmReserve;
    const int TMaxValStart = TTmmStart + TmmReserve + ExpSumReserve;

    Xbyak::Label l_exp_approx_coeff, l_log2e, l_ln2, l_neginf, l_amx_cfg;
    {
      regs_pool rp(this, 1, {9, 32, 0}, TmpSpace, regs_pool::DefaultFlags, 64);  // align 64

      std::shared_ptr<void> local_cfg;
      if (need_cfg_amx) {  // create a local amx config environment
        local_cfg = {(sttilecfg(ptr[rsp]), ldtilecfg(ptr[rip + l_amx_cfg]), nullptr),
                     [&](...) { ldtilecfg(ptr[rsp]); }};
      }

      const auto prepare_log2e = [&]() {
        const auto vreg = rp.reg<Zmm>();
        vbroadcastss(vreg, dword[rip + l_log2e]);
        return vreg;
      };
      const auto prepare_ln2 = [&]() {
        const auto vreg = rp.reg<Zmm>();
        vbroadcastss(vreg, dword[rip + l_ln2]);
        return vreg;
      };
      const auto prepare_c = [&]() {
        const auto c = rp.regs<Zmm, 3>();
        vbroadcastss(c[0], dword[rip + l_exp_approx_coeff]);
        vbroadcastss(c[1], dword[rip + l_exp_approx_coeff + 4]);
        vbroadcastss(c[2], dword[rip + l_exp_approx_coeff + 8]);
        return c;
      };

      const auto reg_ksize = rp.reg<Reg64>();
      mov(reg_ksize.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, k)]);

      const auto reg_iterm = rp.reg<Reg64>();
      xor_(reg_iterm, reg_iterm);
      L(".mloop");
      {
        const auto vreg_tmp = rp.reg<Zmm>();
        vxorps(vreg_tmp, vreg_tmp);
        for (int i = 0; i < MTile; i++) vmovaps(ptr[rsp + TExpsumStart + i * 64], vreg_tmp);
        if (stable_softmax) {
          vpbroadcastd(vreg_tmp, dword[rip + l_neginf]);
          for (int i = 0; i < MTile; i++) vmovaps(ptr[rsp + TMaxValStart + i * 64], vreg_tmp);
        }
      }

      {
        const auto vreg_log2e = prepare_log2e();
        const auto vreg_ln2 = prepare_ln2();
        const auto c = prepare_c();

        const auto reg_itern = rp.reg<Reg64>();
        xor_(reg_itern.cvt32(), reg_itern.cvt32());
        L(".nloop");
        {  // tile product
          const auto reg_matB = rp.reg<Reg64>();
          {  // reg_matB := matB + reg_ksize * reg_itern * sizeof(bf16)
            mov(reg_matB, ptr[rp.p[0] + offsetof(rt_data_t, matB)]);
            auto reg_tmp = rp.reg<Reg64>();
            mov(reg_tmp, reg_ksize);
            imul(reg_tmp, reg_itern);
            lea(reg_matB, ptr[reg_matB + reg_tmp * sizeof(bfloat16_t)]);
          }
          const auto reg_astep = rp.reg<Reg64>();
          mov(reg_astep.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, astep)]);
          const auto reg_matA = rp.reg<Reg64>();
          {  // reg_matA = matA + reg_iterm * reg_astep
            mov(reg_matA, ptr[rp.p[0] + offsetof(rt_data_t, matA)]);
            auto reg_tmp = rp.reg<Reg64>();
            mov(reg_tmp, reg_astep);
            imul(reg_tmp, reg_iterm);
            lea(reg_matA, ptr[reg_matA + reg_tmp]);
          }
          const auto stride_bc = rp.reg<Reg64>();
          mov(stride_bc, NTile * sizeof(float));  // b,c stride
          tile_product_amx_bf16ps(
              reg_ksize, reg_matA, reg_matB, reg_astep, stride_bc, rp.reg<Reg64>(), rp.reg<Reg64>(),
              [&](int i, int j) { return ptr[rsp + stride_bc + TTmmStart + i * 16 * 64 * 2 + j * 64]; });
        }
        const auto reg_matD = rp.reg<Reg64>();
        if (binary_add) {
          mov(reg_matD.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, dstep)]);
          imul(reg_matD, reg_iterm);
          add(reg_matD, qword[rp.p[0] + offsetof(rt_data_t, matD)]);
          lea(reg_matD, ptr[reg_matD + reg_itern * sizeof(float)]);
        }
        const auto reg_matC = rp.reg<Reg64>();
        const auto reg_cstep = rp.reg<Reg64>();
        imul(reg_cstep.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, n)], sizeof(bfloat16_t));
        mov(reg_matC, ptr[rp.p[0] + offsetof(rt_data_t, matC)]);
        lea(reg_matC, ptr[reg_matC + reg_itern * sizeof(bfloat16_t)]);
        {  // reg_matC += reg_cstep * reg_iterm
          auto reg_tmp = rp.reg<Reg64>();
          mov(reg_tmp, reg_iterm);
          imul(reg_tmp.cvt32(), reg_cstep);
          add(reg_matC, reg_tmp);
        }

        // calc f32 exp, accumulate exp to buffer
        for (int in = 0; in < NTile; in += 16) {
          const auto curr_matC = rp.reg<Reg64>();
          auto badd_ptr = rp.reg<Reg64>();
          for (int i = 0; i < MTile; i += 16) {
            const auto vregs_xs = rp.regs<Zmm, 16>();
            for (int ii = 0; ii < 16; ii++) {
              (i == 0 && ii == 0) ? lea(curr_matC, ptr[reg_matC + in * sizeof(bfloat16_t)])
                                  : lea(curr_matC, ptr[curr_matC + reg_cstep]);
              const auto vreg_badd = rp.reg<Zmm>();
              if (binary_add) {
                (i == 0 && ii == 0) ? xor_(badd_ptr.cvt32(), badd_ptr.cvt32())
                                    : add(badd_ptr.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, dstep)]);
                vmovups(vreg_badd, zword[reg_matD + badd_ptr + in * 4]);
              }

              vmovaps(vregs_xs[ii], zword[rsp + TTmmStart + in * 4 + (i + ii) * NTile * 4]);
              !binary_add  // (optionally) add mask and scale
                  ? vmulps(vregs_xs[ii], vregs_xs[ii], zword_b[rp.p[0] + offsetof(rt_data_t, scaleAB)])
                  : vfmadd132ps(vregs_xs[ii], vreg_badd, zword_b[rp.p[0] + offsetof(rt_data_t, scaleAB)]);
              if (!stable_softmax)
                exp_approx_f32(vregs_xs[ii], vregs_xs[ii], vreg_log2e, vreg_ln2, c, rp.regs<Zmm, 2>());
              vcvtneps2bf16(Ymm(vreg_badd.getIdx()), vregs_xs[ii]);
              vmovaps(yword[curr_matC], Ymm(vreg_badd.getIdx()));
              (!stable_softmax) ? vaddps(vregs_xs[ii], vregs_xs[ii], ptr[rsp + TExpsumStart + (i + ii) * 64])
                                : vmaxps(vregs_xs[ii], vregs_xs[ii], ptr[rsp + TMaxValStart + (i + ii) * 64]);
              vmovaps(zword[rsp + (stable_softmax ? TMaxValStart : TExpsumStart) + (i + ii) * 64], vregs_xs[ii]);
            }
          }
        }
        add(reg_itern, NTile);
        cmp(reg_itern.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, n)]);
        jb(".nloop");
      }
      if (stable_softmax) {
        for (int i = 0; i < MTile; i += 16) {  // round max to bf16 representable values
          const auto vregs_max = rp.regs<Zmm, 16>();
          for (int ii = 0; ii < 16; ++ii) vmovaps(vregs_max[ii], zword[rsp + TMaxValStart + (i + ii) * BYTES_ZMM]);
          transpose_16x16_ps(vregs_max, rp.regs<Zmm, 16>());
          reduce_vmms(vregs_max, &CodeGenerator::vmaxps);
          vcvtneps2bf16(Ymm(vregs_max[0].getIdx()), vregs_max[0]);
          vpmovzxwd(vregs_max[0], Ymm(vregs_max[0].getIdx()));
          vpslld(vregs_max[0], vregs_max[0], 16);
          vmovaps(zword[rsp + TMaxValStart + i * BYTES_ZMM], vregs_max[0]);
        }

        const auto vreg_log2e = prepare_log2e();
        const auto vreg_ln2 = prepare_ln2();
        const auto c = prepare_c();

        const auto reg_itern = rp.reg<Reg64>();
        xor_(reg_itern.cvt32(), reg_itern.cvt32());
        L(".nstableloop");
        const auto reg_matC = rp.reg<Reg64>();
        const auto reg_cstep = rp.reg<Reg64>();
        imul(reg_cstep.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, n)], sizeof(bfloat16_t));
        mov(reg_matC, ptr[rp.p[0] + offsetof(rt_data_t, matC)]);
        lea(reg_matC, ptr[reg_matC + reg_itern * sizeof(bfloat16_t)]);
        {  // reg_matC += reg_cstep * reg_iterm
          auto reg_tmp = rp.reg<Reg64>();
          mov(reg_tmp, reg_iterm);
          imul(reg_tmp.cvt32(), reg_cstep);
          add(reg_matC, reg_tmp);
        }
        // calc f32 exp, accumulate exp to buffer
        for (int in = 0; in < NTile; in += 16) {
          const auto curr_matC = rp.reg<Reg64>();
          auto badd_ptr = rp.reg<Reg64>();
          for (int i = 0; i < MTile; i += 16) {
            const auto vregs_xs = rp.regs<Zmm, 16>();
            for (int ii = 0; ii < 16; ii++) {
              (i == 0 && ii == 0) ? lea(curr_matC, ptr[reg_matC + in * sizeof(bfloat16_t)])
                                  : lea(curr_matC, ptr[curr_matC + reg_cstep]);
              const auto vreg_tmp = rp.reg<Zmm>();
              vpmovzxwd(vregs_xs[ii], ptr[curr_matC]);
              vpslld(vregs_xs[ii], vregs_xs[ii], 16);
              vsubps(vregs_xs[ii], vregs_xs[ii], zword_b[rsp + TMaxValStart + i * BYTES_ZMM + ii * sizeof(float)]);
              exp_approx_f32(vregs_xs[ii], vregs_xs[ii], vreg_log2e, vreg_ln2, c, rp.regs<Zmm, 2>());
              vcvtneps2bf16(Ymm(vreg_tmp.getIdx()), vregs_xs[ii]);
              vmovaps(yword[curr_matC], Ymm(vreg_tmp.getIdx()));
              vaddps(vregs_xs[ii], vregs_xs[ii], zword[rsp + TExpsumStart + (i + ii) * 64]);
              vmovaps(zword[rsp + TExpsumStart + (i + ii) * 64], vregs_xs[ii]);
            }
          }
        }
        add(reg_itern, NTile);
        cmp(reg_itern.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, n)]);
        jb(".nstableloop");
      }

      // normalize all temp exp values
      for (int im = 0; im < MTile; im += 16) {
        const auto vregs_xs = rp.regs<Zmm, 16>();
        for (int j = 0; j < 16; j++) vmovaps(vregs_xs[j], ptr[rsp + TExpsumStart + (im + j) * BYTES_ZMM]);
        transpose_16x16_ps(vregs_xs, rp.regs<Zmm, 16>());
        reduce_vmms(vregs_xs, &CodeGenerator::vaddps);
        vrcp14ps(vregs_xs[0], vregs_xs[0]);
        vmovaps(ptr[rsp + TExpsumStart + im * 4], vregs_xs[0]);
      }

      const auto reg_cstep = rp.reg<Reg64>();
      imul(reg_cstep.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, n)], sizeof(bfloat16_t));
      const auto reg_itern = rp.reg<Reg64>();
      xor_(reg_itern, reg_itern);
      L(".nwrloop");
      const auto reg_matC = rp.reg<Reg64>();  // reg_matC := matC + itern * bfloat16_t + reg_cstep * reg_iterm
      {
        mov(reg_matC, ptr[rp.p[0] + offsetof(rt_data_t, matC)]);
        lea(reg_matC, ptr[reg_matC + reg_itern * sizeof(bfloat16_t)]);
        auto reg_tmp = rp.reg<Reg64>();
        mov(reg_tmp, reg_iterm);
        imul(reg_tmp, reg_cstep);
        add(reg_matC, reg_tmp);
      }
      for (int i = 0; i < MTile; i++) {
        const auto vregs_xs = rp.regs<Zmm, 16>();
        const auto vreg_scale = rp.reg<Zmm>();
        vbroadcastss(vreg_scale, ptr[rsp + TExpsumStart + i * 4]);
        for (int in = 0; in < ZMM_PerROW; in++) {
          vpmovzxwd(vregs_xs[in], ptr[reg_matC + in * 32]);
          vpslld(vregs_xs[in], vregs_xs[in], 16);
          vmulps(vregs_xs[in], vregs_xs[in], vreg_scale);
        }
        for (int in = 0; in < ZMM_PerROW; in += 2) {
          vcvtne2ps2bf16(vregs_xs[in], vregs_xs[in + 1], vregs_xs[in]);
          vmovups(ptr[reg_matC + in * 32], vregs_xs[in]);
        }
        add(reg_matC, reg_cstep);
      }
      add(reg_itern, NTile);
      cmp(reg_itern.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, n)]);
      jb(".nwrloop");

      add(reg_iterm, MTile);
      cmp(reg_iterm.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, m)]);
      jb(".mloop");
    }

    L(l_log2e);
    db(bit_cast<uint32_t>(std::log2f(std::exp(1.f))), sizeof(float));
    L(l_ln2);
    db(bit_cast<uint32_t>(std::log(2.f)), sizeof(float));
    L(l_exp_approx_coeff);
    db(reinterpret_cast<const uint8_t*>(exp_approx_f32_coeff.data()), sizeof(exp_approx_f32_coeff));
    L(l_neginf);
    db(bit_cast<uint32_t>(-INFINITY), sizeof(float));
    if (need_cfg_amx) {
      align(sizeof(tileconfig_t));
      L(l_amx_cfg);
      db(reinterpret_cast<const uint8_t*>(&required_tile_cfg_), sizeof(tileconfig_t));
    }
  }

  const bool binary_add, stable_softmax;
  const tile_param_t* const pre_amx_cfg_;
  const tile_param_t required_amx_cfg_;
  const tileconfig_t required_tile_cfg_;
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

  explicit jit_mha_bf16_row_amx_32x32(const tile_param_t* pre_amx_cfg)
      : jit_generator(),
        pre_amx_cfg_(pre_amx_cfg),
        required_amx_cfg_{16, 16, KTile, true, 2},
        required_tile_cfg_(required_amx_cfg_) {}

 private:
  inline void generate() override {
    constexpr int ZMM_PerROW = NTile / 16;

    inLocalLabel();  // use local label for multiple instance
    bool need_cfg_amx = pre_amx_cfg_ != nullptr && *pre_amx_cfg_ != required_amx_cfg_;
    const int amx_config_size = need_cfg_amx ? sizeof(tileconfig_t) : 0;
    const int TmmReserve = MTile * NTile * sizeof(float);
    const int TmpSpace = amx_config_size + TmmReserve;
    const int TTmmStart = amx_config_size;
    Xbyak::Label tmpfvariable, l_amx_cfg;
    {
      regs_pool rp(this, 1, {9, ZMM_PerROW, 1}, TmpSpace, regs_pool::DefaultFlags, 64);  // align 64
      std::shared_ptr<void> local_cfg;
      if (need_cfg_amx) {  // create a local amx config environment
        local_cfg = {(sttilecfg(ptr[rsp]), ldtilecfg(ptr[rip + l_amx_cfg]), nullptr),
                     [&](...) { ldtilecfg(ptr[rsp]); }};
      }
      const Reg64& reg_ksize = rp.reg<Reg64>();
      mov(reg_ksize.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, k)]);

      const Reg64& reg_iterm = rp.reg<Reg64>();
      xor_(reg_iterm.cvt32(), reg_iterm.cvt32());
      L(".mloop");

      const Reg64& reg_itern = rp.reg<Reg64>();
      xor_(reg_itern.cvt32(), reg_itern.cvt32());
      L(".nloop");
      {  // tile product
        const Reg64& reg_astep = rp.reg<Reg64>();
        mov(reg_astep.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, astep)]);
        const Reg64& reg_matA = rp.reg<Reg64>();
        {  // reg_matA = matA + reg_iterm * reg_astep
          mov(reg_matA, ptr[rp.p[0] + offsetof(rt_data_t, matA)]);
          auto reg_tmp = rp.reg<Reg64>();
          mov(reg_tmp, reg_astep);
          imul(reg_tmp, reg_iterm);
          lea(reg_matA, ptr[reg_matA + reg_tmp]);
        }
        const Reg64& reg_matB = rp.reg<Reg64>();
        {  // reg_matB := matB + reg_ksize * reg_itern * sizeof(bf16)
          mov(reg_matB, ptr[rp.p[0] + offsetof(rt_data_t, matB)]);
          auto reg_tmp = rp.reg<Reg64>();
          mov(reg_tmp, reg_ksize);
          imul(reg_tmp, reg_itern);
          lea(reg_matB, ptr[reg_matB + reg_tmp * sizeof(bfloat16_t)]);
        }

        auto stride_bc = rp.reg<Reg64>();
        mov(stride_bc, NTile * 4);  // b,c stride
        tile_product_amx_bf16ps(
            reg_ksize, reg_matA, reg_matB, reg_astep, stride_bc, rp.reg<Reg64>(), rp.reg<Reg64>(),
            [&](int i, int j) { return ptr[rsp + stride_bc + TTmmStart + i * BYTES_TMM * 2 + j * 64]; });
      }
      auto reg_cstep = rp.reg<Reg64>();
      mov(reg_cstep.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, cstep)]);

      const Reg64& reg_matC = rp.reg<Reg64>();
      {  // reg_matC := matC + reg_itern * sizeof(bf16) + reg_iterm * reg_cstep
        mov(reg_matC, ptr[rp.p[0] + offsetof(rt_data_t, matC)]);
        lea(reg_matC, ptr[reg_matC + reg_itern * sizeof(bfloat16_t)]);
        auto reg_tmp = rp.reg<Reg64>();
        mov(reg_tmp, reg_iterm);
        imul(reg_tmp.cvt32(), reg_cstep);
        add(reg_matC, reg_tmp);
      }
      const auto mask0 = rp.reg<Opmask>();
      {  // prepare mask0
        const auto reg_temp = rp.reg<Reg64>();
        const auto reg_tmp = rp.reg<Reg64>();
        mov(reg_tmp.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, n)]);
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

      const auto vregs_xs = rp.regs<Zmm, ZMM_PerROW>();
      for (int i = 0; i < MTile; i++) {
        if (i != 0) lea(reg_matC, ptr[reg_matC + reg_cstep]);
        for (int j = 0; j < ZMM_PerROW; j++) {
          vmovaps(vregs_xs[j], ptr[rsp + TTmmStart + j * 64 + i * NTile * 4]);
          if (j % 2 == 1) {
            vcvtne2ps2bf16(vregs_xs[j - 1], vregs_xs[j], vregs_xs[j - 1]);
            vmovups(ptr[reg_matC + (j - 1) * 32] | mask0, vregs_xs[j - 1]);
          }
        }
      }

      add(reg_itern, NTile);
      cmp(reg_itern.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, n)]);
      jb(".nloop");

      add(reg_iterm, MTile);
      cmp(reg_iterm.cvt32(), dword[rp.p[0] + offsetof(rt_data_t, m)]);
      jb(".mloop");
    }

    if (need_cfg_amx) {
      align(sizeof(tileconfig_t));
      L(l_amx_cfg);
      db(reinterpret_cast<const uint8_t*>(&required_tile_cfg_), sizeof(tileconfig_t));
    }
    outLocalLabel();  // end of local label
  }

  const tile_param_t* const pre_amx_cfg_;
  const tile_param_t required_amx_cfg_;
  const tileconfig_t required_tile_cfg_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MHA_DENSE_BF16_HPP_
