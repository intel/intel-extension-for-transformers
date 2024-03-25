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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANSPOSE_Nx8_4B_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANSPOSE_Nx8_4B_HPP_

#include <glog/logging.h>
#include <vector>
#include "src/utils.hpp"
#include "jit_generator.hpp"

namespace jd {

/**
 * @brief jit_transpose_nx8_4b transpose a matrix of bytes and write it to memory with tile size of
 * nx8 in terms of #elements: mk ==> Km4k
 *
 * @tparam tile_m The n in nx8_4b in terms of #elements
 */
template <int tile_m>
class jit_transpose_nx8_4b : public jit_generator {
 public:
  struct params {
    int K;
    int ld_src;
  };

  struct rt_data_t {
    const void* src;
    void* dst;
  };
  explicit jit_transpose_nx8_4b(const jit_transpose_nx8_4b::params& param) : jit_generator(), param_(param) {
    if (tile_m % dim_transpose != 0) {
      SPARSE_LOG(ERROR) << "tile_m must ve divided by dim_transpose" << std::endl;
    }
  }
  virtual ~jit_transpose_nx8_4b() {}

 private:
  void generate() override;
  // transpose a 8x8 matrix of 4-byte-group in Ymm mat with help of 8 tmp vector registers
  void transpose8_ps(const Xbyak::Ymm mat[8], const Xbyak::Ymm tmp[8]);

  static constexpr int dim_transpose = 8;
  static constexpr int VNNI_ADJ = 4;
  static constexpr int TK_ = 8;

  std::vector<Xbyak::Ymm> get_Ymm(int start, int num) const;  // generate a group of tmp YMM

  jit_transpose_nx8_4b::params param_;
  Xbyak::Label k_loop;
#ifdef _WIN32
  const Xbyak::Reg64& parambase = rcx;
  const Xbyak::Reg64& reg_srcstep = rdi;
#else
  const Xbyak::Reg64& parambase = rdi;
  const Xbyak::Reg64& reg_srcstep = rcx;
#endif
  const Xbyak::Reg64& reg_src = rsi;
  const Xbyak::Reg64& reg_dst = rdx;
  const Xbyak::Reg64& reg_ksize = r8;
  const Xbyak::Reg64& reg_iterk = r9;
  const Xbyak::Reg64& reg_tmp = r10;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANSPOSE_Nx8_4B_HPP_
