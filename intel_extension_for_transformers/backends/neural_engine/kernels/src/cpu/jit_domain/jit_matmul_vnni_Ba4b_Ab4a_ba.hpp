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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_VNNI_BA4B_AB4A_BA_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_VNNI_BA4B_AB4A_BA_HPP_

#include <vector>

#include "jit_generator.hpp"
#include "kernels/matmul_types.hpp"
#include "src/utils.hpp"

namespace jd {
/**
 * @brief jit_matmul_vnni_Ba4b_Ab4a_ba_t calculates this kind matmul: scale * src0 x src1 + zp = dst.
 *        scale(1) * src0(M, K) x src1(K, N) + zp(1) = dst^T(N, M) where M=TH_x16 N=TW_x1
 */
class jit_matmul_vnni_Ba4b_Ab4a_ba_t : public jit_generator {
 public:
  explicit jit_matmul_vnni_Ba4b_Ab4a_ba_t(const ssd::matmul_param_t& param)
      : jit_generator(), param_(param), ld_dst(param.M * param.batch * dsize_dst) {}
  virtual ~jit_matmul_vnni_Ba4b_Ab4a_ba_t() {}

 private:
  ssd::matmul_param_t param_;

  void generate() override;
  void calc_THxTKxTW();

  std::vector<Xbyak::Ymm> get_Ymm(int start, int num) const;  // generate a group of tmp YMM

  // dst_tile should be in col-major
  Xbyak::Zmm dst_tile_Vmm(int i = 0, int j = 0);  // Reg alloc of DST tile
  Xbyak::Zmm src0_tile_Vmm(int i = 0);            // Reg alloc of src0 tile

  static constexpr int TH_ = 2;  // tile height (along m) in terms of #registers
  static constexpr int TW_ = 8;  // tile width (along n) in terms of #registers
  static constexpr int TK_ = 8;  // tile size in reduction dim in terms of (unrolled) iterations
  static constexpr size_t dsize_src0 = sizeof(decltype(*ssd::matmul_u8_data_t::src0));
  static constexpr size_t dsize_src1 = sizeof(decltype(*ssd::matmul_u8_data_t::src1));
  static constexpr size_t dsize_dst = sizeof(decltype(*ssd::matmul_u8_data_t::dst));
  // leading dimension in #bytes
  const int ld_dst;

  const Xbyak::Opmask& reg_k1 = k1;
  const Xbyak::Zmm vreg_temp[2] = {zmm31, zmm30};
  static constexpr int VREG_NUMS = 32;
  static constexpr int USED_VREGS = 2;
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
  const Xbyak::Reg64& reg_ksize = r8;
  const Xbyak::Reg64& reg_iterk = r9;
  const Xbyak::Reg64& reg_tmp = r10;
  // const Xbyak::Reg64&  = r11;
  // const Xbyak::Reg64&  = rbx;

  Xbyak::Label kloop;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMU_VNNI_BA4B_AB4A_BA_HPP_
