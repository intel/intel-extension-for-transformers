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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_BA16B4A_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_BA16B4A_HPP_

#include <vector>

#include "jit_generator.hpp"
#include "regs_pool.hpp"
#include "src/utils.hpp"

namespace jd {

/**
 * @brief jit_trans_BA16b4a transpose a matrix of bytes of size mxn (ab) to (n/16)x(m/4)x16x4 (BA16b4a). Paddings are
 * set to zero.
 */
class jit_trans_BA16b4a : public jit_generator {
 public:
  struct params {
    int M;
    int N;
    int ld_src;
  };

  struct rt_data_t {
    const int8_t* src;
    int8_t* dst;
    int ld_dst;
  };

  // calculate ld_dst from the M dim (outer dim of src)
  static inline int dst_stride(const int M) { return pad_to(M, 4) * VEC; }

  explicit jit_trans_BA16b4a(const jit_trans_BA16b4a::params& param)
      : jit_generator(), M(param.M), N(param.N), ld_src(param.ld_src) {
    SPARSE_LOG_IF(ERROR, N > ld_src) << "N > ld_src which may lead to unexpected behavior!";
    SPARSE_LOG_IF(FATAL, N > 64) << "Currently only support N <= 64";
  }
  virtual ~jit_trans_BA16b4a() {}

 private:
  void generate() override;
  void transpose_4x64(regs_pool* const rp, const Xbyak::Reg64& src, const Xbyak::Reg64& dst, const Xbyak::Reg64& ld_dst,
                      const Xbyak::Reg64& ld_dst3x, bool is_tail = false);

  const int M, N, ld_src;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_BA16B4A_HPP_
