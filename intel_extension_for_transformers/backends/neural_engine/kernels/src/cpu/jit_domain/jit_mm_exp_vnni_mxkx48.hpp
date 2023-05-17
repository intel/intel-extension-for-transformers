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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MM_EXP_VNNI_MXKX48_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MM_EXP_VNNI_MXKX48_HPP_

#include "jit_generator.hpp"
#include "jit_eltwise_injector.hpp"
#include "kernels/matmul_types.hpp"
#include "src/utils.hpp"

namespace jd {
/**
 * @brief jit_mm_exp_vnni_mxkx48_t calculates this kind matmul:
 * exp(scale * (src0 x src1 - bias << bias_lshift) + b0) = dst.
 * dst(m, 48) = exp( scale(1,) * ( src0(m, K) x src1(K, 48) - bias(1, 48) << bias_lshift ) + b0(m, 1) )
 * dst_scale(1, 48) = 255 / sum( dst(m, 48) )
 * src0: u8 | ABa8b4 | (M/8)x(K/4)x8x4;  src1: s8 | BAb48a4 | (K/4)x48x4
 * dst: bf16/fp32 | ab
 */
class jit_mm_exp_vnni_mxkx48_t : public jit_generator {
 public:
  struct param_t {
    uint8_t bias_lshift;
    bool binary_add;
    data_type dt_dst;
    dim_t dst_N;  // N that <= 48
  };

  template <typename dst_t>
  struct rt_data_t {
    const uint8_t* src0;
    const int8_t* src1;
    const int32_t* bias;
    const float* src_b0;
    dst_t* dst;
    float* dst_scale;
    int M;       // must be a multiple of 8
    int K;       // must be a multiple of 4
    int ld_dst;  // leading dimension in #bytes
    float scale;
  };

  explicit jit_mm_exp_vnni_mxkx48_t(const param_t& param);
  virtual ~jit_mm_exp_vnni_mxkx48_t() {}

 private:
  void generate() override;
  Xbyak::Zmm TW_Vmm(int j);               // Register allocator of load activation. 1D shape=(TW)
  Xbyak::Zmm dst_scale_Vmm(int j);        // Register allocator of load activation. 1D shape=(TW)
  Xbyak::Zmm dst_tile_Vmm(int i, int j);  // Reg alloc of DST tile. 2D shape=(TH,TW), stride=(TW,1)

  const dim_t N_ = 48;
  const int TH_ = 8;  // tile height (along m) in terms of #registers
  const int TW_ = 3;  // tile width (along n) in terms of #registers
  static constexpr size_t dsize_src0 = sizeof(decltype(*rt_data_t<void>::src0));
  static constexpr size_t dsize_src1 = sizeof(decltype(*rt_data_t<void>::src1));
  static constexpr size_t dsize_bias = sizeof(decltype(*rt_data_t<void>::bias));
  static constexpr size_t dsize_src_b0 = sizeof(decltype(*rt_data_t<void>::src_b0));
  const uint8_t bias_lshift;
  const bool binary_add;
  const data_type dt_dst;
  const size_t dsize_dst;

  const Xbyak::Zmm& vreg_temp = zmm31;
  static constexpr int VREG_NUMS = 32;
  static constexpr int USED_VREGS = 1;

#ifdef _WIN32
  const Xbyak::Reg64& parambase = rcx;
  const Xbyak::Reg64& reg_dst = rdi;
#else
  const Xbyak::Reg64& parambase = rdi;
  const Xbyak::Reg64& reg_dst = rcx;
#endif
  const Xbyak::Reg64& reg_src0 = rsi;
  const Xbyak::Reg64& reg_src1 = rdx;
  const Xbyak::Reg64& reg_iterk = r8;
  const Xbyak::Reg64& reg_ksize = r9;
  const Xbyak::Reg64& reg_tmp = r10;
  const Xbyak::Reg64& reg_src_b0 = r11;
  const Xbyak::Reg64& reg_ld_dst = r12;
  const Xbyak::Reg64& reg_iterm = r13;
  const Xbyak::Opmask& mask_n = k1;
  Xbyak::Label l_kloop, l_mloop, l_255f;
  Xbyak::Label l_log2e, l_ln2, l_exp_approx_coeff;
};
}  // namespace jd

#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MM_EXP_VNNI_MXKX48_HPP_
