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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SPMM_AVX512F_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SPMM_AVX512F_HPP_

#include "jit_generator.hpp"
#include "kernels/sparse_data.hpp"
#include "kernels/spmm_types.hpp"
#include "src/utils.hpp"
#include "jit_eltwise_injector.hpp"

#define GET_OFF(field) offsetof(ssd::avx512_data_t, field)

namespace jd {
/**
 * @brief jit_spmm_avx512f_t calculates this kind matmul: dense x sparse = dst.
 *        weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
 */
class jit_spmm_avx512f_t : public jit_generator {
 public:
  explicit jit_spmm_avx512f_t(const ssd::avx512_fp32_params_t& param)
      : jit_generator(), param_(param), bsc_(param_.sparse_ptr) {
    eltwise_injector.eltwise_injector_init(this, param_.postop_attrs);
  }
  virtual ~jit_spmm_avx512f_t() {}
  bsc_data_t<float>* bsc_data() { return bsc_; }

 private:
  ssd::avx512_fp32_params_t param_;
  bsc_data_t<float>* bsc_;
  jit_eltwise_injector eltwise_injector;

  void generate() override;

  int64_t TW_ = 1;
  int64_t TH_ = 4;
#ifdef _WIN32
  const Xbyak::Reg64& reg_param = rcx;
  const Xbyak::Reg64& reg_sparse = rdi;  // the first argument which is packed nonzero values pointer
#else
  const Xbyak::Reg64& reg_param = rdi;
  const Xbyak::Reg64& reg_sparse = rcx;  // the first argument which is packed nonzero values pointer
#endif
  const Xbyak::Reg64& reg_dense = rsi;
  const Xbyak::Reg64& reg_dense_end = rdx;
  const Xbyak::Reg64& reg_bias = r8;
  const Xbyak::Reg64& reg_dst = r9;
  const Xbyak::Reg64& reg_n = r10;      // current n index
  const Xbyak::Reg64& reg_n_end = r11;  // end of n index

  // load params passed as avx512_data_t
  inline void load_params();

  // internal API of op kernel
  // Register allocator of load weight. 1D shape=(TH)
  Xbyak::Zmm TH_Vmm(int i = 0) {
    const int& alloc_start = VREG_NUMS - 1 - USED_VREGS;
    return Xbyak::Zmm(alloc_start - i);
  }
  // Register allocator of load activation. TW = 1
  Xbyak::Zmm TW_Vmm = Xbyak::Zmm(VREG_NUMS - 1 - USED_VREGS - TH_);
  // Reg alloc of DST tile.
  Xbyak::Zmm dst_tile_Vmm(int i = 0) { return Xbyak::Zmm(i); }

  static constexpr int stack_space_needed_ = 256;
  static constexpr int BYTE8 = 8;
  static constexpr int BYTE4 = 4;
  static constexpr int BYTE1 = 1;
  static constexpr int F32_BYTES = 4;
  static constexpr int ZMM_BYTES = 64;
  static constexpr int VREG_NUMS = 32;
  static constexpr int USED_VREGS = 0;
#ifdef XBYAK64
  static constexpr int PTR_SIZE = 8;
#else
  static constexpr int PTR_SIZE = 4;
#endif
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SPMM_AVX512F_HPP_
