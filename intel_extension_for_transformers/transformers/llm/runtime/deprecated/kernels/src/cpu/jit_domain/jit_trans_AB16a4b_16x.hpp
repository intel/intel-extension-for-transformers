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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_AB16A4B_16X_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_AB16A4B_16X_HPP_

#include "jit_generator.hpp"
#include "regs_pool.hpp"

namespace jd {

/**
 * @brief jit_trans_AB16a4b_16x transpose a matrix of bytes of size 16xn (ab) to (1)x(n/4)x16x4 (AB16a4b). Paddings are
 * set to zero.
 *
 * e.g. 384x64 => 24x16x16x4; 384x32 => 24x8x16x4
 *
 * @note Comparing to jit_trans_AB16a4b: it supports dynamic shape & loop along N
 */
class jit_trans_AB16a4b_16x : public jit_generator {
 public:
  struct param_t {
    int pad_n;      // pad the tail along N; should be no greater then 64 and be a multiple of 4
    bool cvt_s8u8;  // whether to convert s8 to u8
    int tile_m;     // number of 16-tiles over M. e.g. tile_m == 2 for AB32a4b
  };

  struct rt_data_t {
    const int8_t* src;
    void* dst;  // u8 / s8
    int ld_src;
    int M;  // less or equal to 16 * tile_m, will be padded with 0
    int N;
  };

  explicit jit_trans_AB16a4b_16x(const jit_trans_AB16a4b_16x::param_t& param)
      : jit_generator(), pad_n(param.pad_n), cvt_s8u8(param.cvt_s8u8), tile_m(param.tile_m) {
    SPARSE_LOG_IF(FATAL, pad_n % 4 != 0 || pad_n > 64) << "Improper pad_n!";
    SPARSE_LOG_IF(FATAL, tile_m <= 0) << "Improper tile_m!";
    SPARSE_LOG_IF(WARNING, pad_n % 16 != 0) << "pad_n value of " << pad_n << " may lead to suboptimal performance";
  }
  virtual ~jit_trans_AB16a4b_16x() {}

 private:
  void generate() override;
  void transpose_16x16_ps(  //
      regs_pool* const rp, const Xbyak::Reg64& src, const Xbyak::Reg64& dst, const Xbyak::Reg64& ld_src,
      const Xbyak::Reg64& reg_tmp, const int M, const Xbyak::Opmask& mask);

  const int pad_n;
  const bool cvt_s8u8;
  int tile_m;
  Xbyak::Label l_128x64, l_msize_tbl;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_AB16A4B_16X_HPP_
