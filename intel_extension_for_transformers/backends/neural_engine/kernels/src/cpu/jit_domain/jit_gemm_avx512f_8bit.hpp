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
#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_GEMM_AVX512F_8BIT_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_GEMM_AVX512F_8BIT_HPP_

#include <vector>

#include "jit_eltwise_injector.hpp"
#include "jit_generator.hpp"
#include "kernels/matmul_types.hpp"
#include "regs_pool.hpp"

namespace jd {
class jit_gemm_avx512f_8bit_t : public jit_generator {
 public:
  explicit jit_gemm_avx512f_8bit_t(const ssd::matmul_fp8_param_t& param) : jit_generator(), param_(param) {
    NRegs = NTile / 16;
    CRegCount = NRegs * MTile;
    BRegCount = NRegs;
    ARegCount = 1;
    TmpRegCount = 3;
    injector_.eltwise_injector_init(this, param_.postop_attrs);
  }
  void generate();

 public:
  static int constexpr MTile = 4, NTile = 32, KTile = 2;

 private:
  std::vector<Xbyak::Zmm> zmms_a_;
  std::vector<Xbyak::Zmm> zmms_b_;
  std::vector<Xbyak::Zmm> zmms_c_;
  std::vector<Xbyak::Zmm> zmms_tmp_;

  ssd::matmul_fp8_param_t param_;
  jit_eltwise_injector injector_;
  int NRegs;
  int ARegCount;
  int BRegCount;
  int CRegCount;
  int TmpRegCount;

 private:
  void broadcast_bf16_fp32(const Xbyak::Zmm& _fp32, const Xbyak::Opmask& _mask, const Xbyak::Address& _add) {
    vpbroadcastw(_fp32 | _mask | T_z, _add);
  }
  void load_bf16_fp32(const Xbyak::Zmm& _fp32, const Xbyak::Address& _add) {
    vpmovzxwd(_fp32, _add);
    vpslld(_fp32, _fp32, 16);
  }
  void store_fp32_bf16(const Xbyak::Zmm& _fp32, const Xbyak::Address& _add) {
    // post op
    injector_.vector_compute(_fp32, param_.postop_attrs);
    vcvtneps2bf16(Xbyak::Ymm(_fp32.getIdx()), _fp32);
    vmovups(_add, Xbyak::Ymm(_fp32.getIdx()));
  }

  void load_int8_fp32(const Xbyak::Zmm& tar, const Xbyak::Address& addr);
  void load_fp8_fp32(const Xbyak::Zmm& tar, const Xbyak::Address& addr);

  void generate_fma(int MTile, int _NRegs, int KTile, const Xbyak::Reg64& aptr, const Xbyak::Reg64& bptr,
                    const Xbyak::Reg64& reg_astep, const Xbyak::Reg64& reg_bstep, const Xbyak::Reg64& reg_tmp,
                    const Xbyak::Reg64& reg_tmp1);
  void alphabeta_process(int MTile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_tmp,
                         const Xbyak::Reg64& reg_tmp1, const Xbyak::Reg64& reg_tmp2);
  void vreg_push(const Xbyak::Reg64& baseaddr) {
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(xword[baseaddr + i * 16], Xbyak::Xmm(6 + i));
    }
#else
    (void)baseaddr;
#endif
  }

  void vreg_pop(const Xbyak::Reg64& baseaddr) {
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(Xbyak::Xmm(6 + i), xword[baseaddr + i * 16]);
    }
#else
    (void)baseaddr;
#endif
  }

  void load32(const Xbyak::Reg64& reg, const Xbyak::Address& addr) {
    xor_(reg, reg);
    mov(reg.cvt32(), addr);
  }
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_GEMM_AVX512F_8BIT_HPP_
