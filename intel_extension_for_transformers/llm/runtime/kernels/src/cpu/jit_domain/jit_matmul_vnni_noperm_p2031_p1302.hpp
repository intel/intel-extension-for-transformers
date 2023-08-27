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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_VNNI_NOPERM_P2031_P1302_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_VNNI_NOPERM_P2031_P1302_HPP_

#include <vector>

#include "jit_generator.hpp"
#include "kernels/matmul_types.hpp"
#include "src/utils.hpp"

namespace jd {
/**
 * @brief jit_matmul_vnni_noperm_p2031_p1302_t calculates this kind matmul: scale * src0 x src1 + zp = dst.
 *        scale(1) * src0(M, K) x src1(K, N) + zp(1) = dst(M, N)
 */
class jit_matmul_vnni_noperm_p2031_p1302_t : public jit_generator {
 public:
  explicit jit_matmul_vnni_noperm_p2031_p1302_t(const ssd::matmul_param_t& param)
      : jit_generator(),
        param_(param),
        TH_(param.m_tile),
        TW_(param.n_tile),
        ld_src0(param.K * dsize_src0),
        ld_src1(param.K * param.batch * dsize_src1),  // stride to load next row (before transpose)
        ld_dst(param.M * param.batch * dsize_dst),
        start_transpose_src0(0),
        size_transpose_src0((TH_ * BYTES_ZMM / VNNI_ADJ) * (TK_ * VNNI_ADJ) * dsize_src0),
        start_transpose_src1(size_transpose_src0),
        size_transpose_src1(TW_ * (TK_ * VNNI_ADJ) * dsize_src1),
        stack_size(size_transpose_src0 + size_transpose_src1) {}
  virtual ~jit_matmul_vnni_noperm_p2031_p1302_t() {}

 private:
  ssd::matmul_param_t param_;

  void generate() override;
  // transpose a 8x8 matrix of 4-byte-group in Ymm mat with help of 8 tmp vector registers
  void transpose8_ps(const Xbyak::Ymm mat[8], const Xbyak::Ymm tmp[8]);
  void calc_THxTKxTW();

  std::vector<Xbyak::Ymm> get_Ymm(int start, int num) const;  // generate a group of tmp YMM
  Xbyak::Zmm dst_tile_Vmm(int j = 0);                         // Reg alloc of DST tile

  const int TH_;                 // tile height (along m) in terms of #registers
  const int TW_;                 // tile width (along n) in terms of #registers
  static constexpr int TK_ = 8;  // tile size in reduction dim in terms of (unrolled) iterations
  static constexpr size_t dsize_src0 = sizeof(decltype(*ssd::matmul_u8_data_t::src0));
  static constexpr size_t dsize_src1 = sizeof(decltype(*ssd::matmul_u8_data_t::src1));
  static constexpr size_t dsize_dst = sizeof(decltype(*ssd::matmul_u8_data_t::dst));
  // leading dimension in #bytes
  const int ld_src0, ld_src1, ld_dst;
  const int start_transpose_src0, size_transpose_src0, start_transpose_src1, size_transpose_src1, stack_size;

  const Xbyak::Opmask& reg_k1 = k1;
  static constexpr int VREG_NUMS = 32;
  static constexpr int USED_VREGS = 0;
  static constexpr int VNNI_ADJ = 4;                        // reduction dim of VPDPBUSD
  static constexpr int VNNI_GROUPS = BYTES_ZMM / VNNI_ADJ;  // spacial dim of VPDPBUSD
#ifdef _WIN32
  const Xbyak::Reg64& parambase = rcx;
  const Xbyak::Reg64& reg_dst = rdi;
#else
  const Xbyak::Reg64& parambase = rdi;
  const Xbyak::Reg64& reg_dst = rcx;
#endif
  const Xbyak::Reg64& reg_src0 = rsi;
  const Xbyak::Reg64& reg_src1 = rdx;
  const Xbyak::Reg64& reg_ld_src0 = r8;
  const Xbyak::Reg64& reg_ksize = r9;
  const Xbyak::Reg64& reg_ld_src1 = r10;
  const Xbyak::Reg64& reg_iterk = r11;
  const Xbyak::Reg64& reg_tmp = rbx;

  Xbyak::Label kloop;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_VNNI_NOPERM_P2031_P1302_HPP_
