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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SEQ_CPY_2x8x8_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SEQ_CPY_2x8x8_HPP_

#include <glog/logging.h>
#include <vector>
#include "src/utils.hpp"
#include "jit_generator.hpp"

namespace jd {
class jit_seq_cpy_2x8x8 : public jit_generator {
 public:
  // calculate ld_dst from the M dim (outer dim of src)
  static inline int dst_step(const int M) { return 8 * (ceil_div(M, 8) * 8); }

  struct param_t {
    uint8_t val_offset;  // add an offset to every elements
  };

  struct rt_data_t {
    const void* src;
    void* dst;
    int N;       // inner dim of src: loop dimention
    int ld_src;  // leading dim / bytes of src
    int ld_dst;  // leading dim / bytes of dst
  };
  /**
   * jit_seq_cpy_2x8x8 reorders matrix in the following way:
   *  src(8xN) ==> dst((n/8)x2x8x4)
   *
   * define jit_seq_cpy_2x8x8(M, N, src, dst, ld_src):
   *   for (int i=0; i < 8; i+=4)
   *     for (int j=0; j < N; j+=8)
   *       reorder_4x8(src + j + i*ld, dst + j*M + i*8, ld_src)
   *
   * +------ src ------<ld_src>
   * | 0 1 2 3 4 5 6 7 <ld_src>
   * | 8 9 a b c d e f <ld_src>
   * | g h i j k l m n <ld_src>
   * | o p q r s t u v <ld_src>
   * +-----------------<ld_src>
   * ===== reorder_4x8(src, dst, ld_src) ====>
   * +-- dst --+
   * | 0 8 g o |
   * | 1 9 h p |
   * | 2 a i q |
   * | 3 b j r |
   * | 4 c k s |
   * | 5 d l t |
   * | 6 e m u |
   * | 7 f n v |
   * +---------+
   */
  explicit jit_seq_cpy_2x8x8(const jit_seq_cpy_2x8x8::param_t& param) : jit_generator(), val_offset(param.val_offset) {}
  virtual ~jit_seq_cpy_2x8x8() {}

 private:
  void generate() override;

  const uint8_t val_offset;

#ifdef _WIN32
  const Xbyak::Reg64& parambase = rcx;
  const Xbyak::Reg64& reg_nsize = rdi;
#else
  const Xbyak::Reg64& parambase = rdi;
  const Xbyak::Reg64& reg_nsize = rcx;
#endif
  const Xbyak::Reg64& reg_src = rsi;
  const Xbyak::Reg64& reg_dst = rdx;
  const Xbyak::Reg64& reg_itern = r8;
  const Xbyak::Reg64& reg_tmp = r9;
  const Xbyak::Reg64& reg_ld_dst = r10;
  const Xbyak::Reg64& reg_ld_src = r11;
  const Xbyak::Reg64& reg_3ld_src = r12;

  const Xbyak::Zmm& vpermt2d_arg_idx = zmm31;
  const Xbyak::Zmm& vpshufb_arg_b = zmm30;
  const Xbyak::Zmm& vreg_val_offset = zmm29;
  const Xbyak::Zmm& vreg_t1 = zmm28;
  const Xbyak::Zmm& vreg_t2 = zmm27;
  const Xbyak::Opmask& reg_k = k1;
  const Xbyak::Opmask& reg_k_tail = k2;

  Xbyak::Label k_loop;
  Xbyak::Label l_vpermt2d_control;
  Xbyak::Label l_vpshufb_control;
  Xbyak::Label l_n_loop;
  Xbyak::Label l_n_loop_end;
  Xbyak::Label l_ntail_end;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SEQ_CPY_2x8x8_HPP_
