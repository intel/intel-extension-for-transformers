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

#include "jit_matmul_avx512f_p2031_p2013.hpp"

namespace jd {
Xbyak::Zmm jit_matmul_avx512f_p2031_p2013_t::TW_Vmm(int i) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS;
  const int& alloc_idx = alloc_start - i;
  return Xbyak::Zmm(alloc_idx);
}

// {zmm0, zmm1, zmm2, zmm3, ...}
Xbyak::Zmm jit_matmul_avx512f_p2031_p2013_t::dst_tile_Vmm(int i, int j) {
  const int& alloc_start = 0;
  const int& alloc_idx = alloc_start + i * TW_ + j;
  return Xbyak::Zmm(alloc_idx);
}

void jit_matmul_avx512f_p2031_p2013_t::calc_THxkxTW() {
  // prologue
  if (param_.beta != 0.f) {
    auto& vreg_beta = vreg_temp;
    float badd_scale = static_cast<double>(param_.beta) / param_.alpha;
    uint32_t badd_scale_;
    memcpy(&badd_scale_, &badd_scale, sizeof(badd_scale));
    mov(reg_tmp.cvt32(), badd_scale_);  // all result will times alpha later
    vpbroadcastd(vreg_beta, reg_tmp.cvt32());

    for (int i = 0; i < TH_; i++)
      for (int j = 0; j < TW_; ++j)
        vmulps(dst_tile_Vmm(i, j), vreg_beta, zword[reg_src2 + i * ld_src2 + j * BYTES_ZMM]);
  } else {
    // clear reg for m tile
    for (int i = 0; i < TH_; ++i)
      for (int j = 0; j < TW_; ++j)  //
        vpxorq(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), dst_tile_Vmm(i, j));
  }

  xor_(reg_iterk.cvt32(), reg_iterk.cvt32());  // reg_iterk = 0
  Xbyak::Label L_k_loop;
  L(L_k_loop);

  for (int kk = 0; kk < UNROLL_K; kk++) {
    for (int i = 0; i < TH_; i++) {
      vpbroadcastd(vreg_temp, dword[reg_src0 + ld_src0 * kk + dsize_src0 * i]);  // load src0
      for (int j = 0; j < TW_; ++j) {
        // load src1
        if (i == 0) vmovups(TW_Vmm(j), ptr[reg_src1 + ld_src1 * kk + BYTES_ZMM * j]);  // 64 is #bytes of a zmm
        vfmadd231ps(dst_tile_Vmm(i, j), TW_Vmm(j), vreg_temp);
      }
    }
  }
  add(reg_src0, ld_src0 * UNROLL_K);
  add(reg_src1, ld_src1 * UNROLL_K);

  inc(reg_iterk);
  cmp(reg_iterk, k_iters);  // k iteration variable
  jb(L_k_loop);
  sub(reg_src0, ld_src0 * param_.K);
  sub(reg_src1, ld_src1 * param_.K);

  if (param_.alpha != 1.f) {
    uint32_t alpha_;
    memcpy(&alpha_, &param_.alpha, sizeof(alpha_));
    mov(reg_tmp.cvt32(), alpha_);
    vpbroadcastd(vreg_temp, reg_tmp.cvt32());
    for (int i = 0; i < TH_; ++i)
      for (int j = 0; j < TW_; ++j)  //
        vmulps(dst_tile_Vmm(i, j), vreg_temp);
  }

  // store result
  for (int i = 0; i < TH_; i++)
    for (int j = 0; j < TW_; ++j)  //
      vmovups(ptr[reg_dst + i * ld_src2 + j * BYTES_ZMM], dst_tile_Vmm(i, j));
}

void jit_matmul_avx512f_p2031_p2013_t::generate() {
  inLocalLabel();  // use local label for multiple instance
  preamble();

  mov(reg_src0, ptr[parambase + GET_OFF(src0)]);
  mov(reg_src0_end, param_.M * dsize_src0);
  add(reg_src0_end, reg_src0);

  mov(reg_dst, ptr[parambase + GET_OFF(dst)]);
  mov(reg_src2, ptr[parambase + GET_OFF(src2)]);

  Xbyak::Label L_m_loop;
  L(L_m_loop);
  // M loop begin
  {
    mov(reg_src1, ptr[parambase + GET_OFF(src1)]);
    mov(reg_src1_end, param_.N * dsize_src1);
    add(reg_src1_end, reg_src1);

    Xbyak::Label L_n_loop;
    L(L_n_loop);
    // N loop begin

    calc_THxkxTW();

    add(reg_dst, TW_ * BYTES_ZMM);
    add(reg_src2, TW_ * BYTES_ZMM);
    add(reg_src1, TW_ * BYTES_ZMM);
    cmp(reg_src1, reg_src1_end);
    jl(L_n_loop);
    // sub(reg_dst, dsize_dst * param_.N); -- combined to the next inst
    // sub(reg_src2, dsize_src2 * param_.N); -- combined to the next inst
    // N loop end
  }
  add(reg_dst, TH_ * ld_dst - dsize_dst * param_.N);
  add(reg_src2, TH_ * ld_src2 - dsize_src2 * param_.N);
  add(reg_src0, TH_ * dsize_src0);  // notice that src0 is transposed to it multiplies dsize rather than ld
  cmp(reg_src0, reg_src0_end);
  jl(L_m_loop);
  // M loop end

  postamble();
  outLocalLabel();  // end of local label
}
}  // namespace jd
