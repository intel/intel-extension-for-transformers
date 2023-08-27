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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_VNNI_P2031_p2013_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_VNNI_P2031_p2013_HPP_

#include <vector>
#include "jit_generator.hpp"
#include "jit_eltwise_injector.hpp"
#include "kernels/matmul_types.hpp"
#include "src/utils.hpp"

namespace jd {
/**
 * @brief jit_matmul_vnni_8xkx48_t calculates this kind matmul:
 * scale * src0 x src1 - bias << bias_lshift + binary_add = dst.
 * dst(8, 48) = scale(1,) * ( src0(8, K) x src1(K, 48) - bias(1, 48) << bias_lshift ) + binary_add(8, 48)
 * src0: (K/4)x8x4;  src1: (K/4)x48x4
 */
class jit_matmul_vnni_8xkx48_t : public jit_generator {
 public:
  template <typename dst_t>
  struct rt_data_t {
    const uint8_t* src0;
    const int8_t* src1;
    const int32_t* bias;
    const float* src_b0;
    dst_t* dst;
  };

  struct param_t {
    dim_t K;
    dim_t ld_dst;
    float scale;
    uint8_t bias_lshift;
    bool binary_add;
    data_type dt_dst;
    std::vector<postop_attr> postop_attrs;
    dim_t dst_M;
    dim_t dst_N;
  };

  explicit jit_matmul_vnni_8xkx48_t(const param_t& param);
  virtual ~jit_matmul_vnni_8xkx48_t() {}

 private:
  const param_t param_;

  void generate() override;
  void calc_THxkxTW();
  Xbyak::Zmm TW_Vmm(int j);               // Register allocator of load activation. 1D shape=(TW)
  Xbyak::Zmm dst_tile_Vmm(int i, int j);  // Reg alloc of DST tile. 2D shape=(TH,TW), stride=(TW,1)

  const dim_t M_ = 8;
  const dim_t N_ = 48;
  const int TH_ = 8;  // tile height (along m) in terms of #registers
  const int TW_ = 3;  // tile width (along n) in terms of #registers
  static constexpr size_t dsize_src0 = sizeof(decltype(*rt_data_t<void>::src0));
  static constexpr size_t dsize_src1 = sizeof(decltype(*rt_data_t<void>::src1));
  static constexpr size_t dsize_bias = sizeof(decltype(*rt_data_t<void>::bias));
  static constexpr size_t dsize_src_b0 = sizeof(decltype(*rt_data_t<void>::src_b0));
  const size_t dsize_dst;
  // leading dimension in #bytes
  const int lb_src_b0, lb_dst;

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
  const Xbyak::Opmask& mask_n = k1;
  Xbyak::Label kloop;

  jit_eltwise_injector eltwise_inj_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_VNNI_P2031_p2013_HPP_
