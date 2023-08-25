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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_AB16A4B_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_AB16A4B_HPP_

#include "jit_generator.hpp"
#include "regs_pool.hpp"
#include "src/utils.hpp"

namespace jd {

/**
 * @brief jit_trans_AB16a4b transpose a matrix of bytes of size mxn (ab) to (m/16)x(n/4)x16x4 (AB16a4b). Paddings are
 * set to zero.
 *
 * e.g. 384x64 => 24x16x16x4; 384x32 => 24x8x16x4
 */
class jit_trans_AB16a4b : public jit_generator {
 public:
  struct param_t {
    int M;
    int N;
    int ld_src;
    int pad_n;
  };

  struct rt_data_t {
    const int8_t* src;
    int8_t* dst;
  };

  explicit jit_trans_AB16a4b(const jit_trans_AB16a4b::param_t& param)
      : jit_generator(), M(param.M), N(param.N), ld_src(param.ld_src), pad_n(param.pad_n) {
    SPARSE_LOG_IF(ERROR, N > ld_src) << "N > ld_src which may lead to unexpected behavior!";
    SPARSE_LOG_IF(FATAL, N > 64) << "Only support N <= 64";
    SPARSE_LOG_IF(FATAL, pad_n % 4 != 0 || pad_n < N || pad_n > 64) << "Improper pad_n!";
    SPARSE_LOG_IF(WARNING, pad_n % 16 != 0) << "pad_n value of " << pad_n << " may lead to suboptimal performance";
  }
  virtual ~jit_trans_AB16a4b() {}

 private:
  void generate() override;
  void mem_trans_16x16_ps(regs_pool* const rp, const Xbyak::Reg64& src, const Xbyak::Reg64& dst,
                          const Xbyak::Opmask& mask, bool is_tail = false);

  const int M, N, ld_src, pad_n;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_AB16A4B_HPP_
