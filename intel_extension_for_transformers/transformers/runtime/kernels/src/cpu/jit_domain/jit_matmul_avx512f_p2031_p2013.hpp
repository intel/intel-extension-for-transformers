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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_AVX512F_P2031_p2013_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_AVX512F_P2031_p2013_HPP_

#include "jit_generator.hpp"
#include "kernels/matmul_types.hpp"
#include "src/utils.hpp"

#define GET_OFF(field) offsetof(ssd::matmul_data_t, field)

namespace jd {
/**
 * @brief jit_matmul_avx512f_p2031_p2013_t calculates this kind matmul: alpha * src0 x src1 + beta * src2 = dst.
 *        alpha * src0(M, K) x src1(K, N) + beta * scr2(M, N) = dst(M, N)
 */
class jit_matmul_avx512f_p2031_p2013_t : public jit_generator {
 public:
  explicit jit_matmul_avx512f_p2031_p2013_t(const ssd::matmul_param_t& param)
      : jit_generator(),
        param_(param),
        TH_(param.m_tile),
        TW_(param.n_tile),
        ld_src0(param.M * param.batch * dsize_src0),
        ld_src1(param.N * param.batch * dsize_src1),
        ld_src2(param.N * dsize_src2),
        ld_dst(param.N * dsize_dst),
        k_iters(param.K / UNROLL_K) {}
  virtual ~jit_matmul_avx512f_p2031_p2013_t() {}

 private:
  ssd::matmul_param_t param_;

  void generate() override;
  void calc_THxkxTW();
  Xbyak::Zmm TH_Vmm(int i = 0);                   // Register allocator of load weight. 1D shape=(TH)
  Xbyak::Zmm TW_Vmm(int i = 0);                   // Register allocator of load activation. 1D shape=(TW)
  Xbyak::Zmm dst_tile_Vmm(int i = 0, int j = 0);  // Reg alloc of DST tile. 2D shape=(TH,TW), stride=(TW,1)

  const int TH_;  // tile height (along m) in terms of #registers
  const int TW_;  // tile width (along n) in terms of #registers
  static constexpr size_t dsize_src0 = sizeof(decltype(*ssd::matmul_data_t::src0));
  static constexpr size_t dsize_src1 = sizeof(decltype(*ssd::matmul_data_t::src1));
  static constexpr size_t dsize_src2 = sizeof(decltype(*ssd::matmul_data_t::src2));
  static constexpr size_t dsize_dst = sizeof(decltype(*ssd::matmul_data_t::dst));
  // leading dimension in #bytes
  const int ld_src0, ld_src1, ld_src2, ld_dst;
  const int k_iters;

  const Xbyak::Zmm& vreg_temp = zmm31;
  static constexpr int VREG_NUMS = 32;
  static constexpr int USED_VREGS = 1;
  static constexpr int UNROLL_K = 8;
#ifdef _WIN32
  const Xbyak::Reg64& parambase = rcx;
  const Xbyak::Reg64& reg_src2 = rdi;
#else
  const Xbyak::Reg64& parambase = rdi;
  const Xbyak::Reg64& reg_src2 = rcx;
#endif
  const Xbyak::Reg64& reg_src0 = rsi;
  const Xbyak::Reg64& reg_src1 = rdx;
  const Xbyak::Reg64& reg_dst = r8;
  const Xbyak::Reg64& reg_src0_end = r9;
  const Xbyak::Reg64& reg_src1_end = r10;
  const Xbyak::Reg64& reg_iterk = r11;
  const Xbyak::Reg64& reg_tmp = rbx;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_AVX512F_P2031_p2013_HPP_
