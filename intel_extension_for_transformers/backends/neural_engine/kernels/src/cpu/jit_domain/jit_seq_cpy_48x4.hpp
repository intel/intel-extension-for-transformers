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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SEQ_CPY_48X4_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SEQ_CPY_48X4_HPP_

#include <glog/logging.h>
#include <vector>
#include "src/utils.hpp"
#include "jit_generator.hpp"

namespace jd {
class jit_seq_cpy_48x4 : public jit_generator {
 public:
  // calculate ld_dst from the M dim (outer dim of src)
  static inline int dst_step(const int M) { return 48 * (ceil_div(M, 4) * 4); }

  struct param_t {
    bool sum_m;           // calculate sum along the M axis
    bool is_unsigned;     // datatype of either signed or unsigned; only useful when calculating sum
    int32_t sum_pad_val;  // padding value for the sum dst

    // param_t() = default;
  };

  struct rt_data_t {
    const void* src;
    void* dst;
    int32_t* dst_sum;
    bool sum_append;  // whether the sum should overwrite or accumulate
    int N;            // inner dim of src: loop dimention
    int ld_src;       // leading dim / bytes of src
    int ld_dst;       // leading dim / bytes of dst
  };

  /**
   * jit_seq_cpy_48x4 reorders matrix in the following way:
   *  src(4xN) ==> dst((N/48)x48x4)
   *
   * define jit_seq_cpy_48x4(M, N, src, dst, ld_src):
   *   for (int j=0; j < N; j+=48)
   *     reorder_4x48(src + j, dst + j*M, ld_src)
   *     if (sum_m) sum_4x48(src + k * ld_src, dst_sum + j, ld_src)
   *
   * +--------- src -----------<ld_src>
   * |   0   1   2   3 ...  47 <ld_src>
   * |  48  49  50  51 ...  95 <ld_src>
   * |  96  97  98  99 ... 143 <ld_src>
   * | 144 145 146 147 ... 191 <ld_src>
   * +-------------------------<ld_src>
   * ===== reorder_4x8(src, dst, ld_src) ====>
   * +------ dst -----+
   * |  0  48  96 144 |
   * |  1  49  97 145 |
   * |  2  50  98 146 |
   * |  3  51  99 147 |
   * |  .   .   .   . |
   * |  .   .   .   . |
   * |  .   .   .   . |
   * | 47  95 143 191 |
   * +----------------+
   * ====== sum_4x48(src, dst_sum, ld_src) ======>
   * +------------------------------------ dst_sum ---------------------------------------+
   * | +=0+48+96+144   +=1+49+97+145   +=2+50+98+146   +=3+51+99+147 ...  +=47+95+143+191 |
   * +------------------------------------------------------------------------------------+
   */
  explicit jit_seq_cpy_48x4(const jit_seq_cpy_48x4::param_t& param)
      : jit_generator(), sum_m(param.sum_m), is_unsigned(param.is_unsigned), sum_pad_val(param.sum_pad_val) {}
  virtual ~jit_seq_cpy_48x4() {}

 private:
  void generate() override;
  void trans_4x16(const Zmm& zreg_res, const Zmm& zreg_sum, const Xbyak::Operand& op0, const Xbyak::Operand& op1,
                  const Xbyak::Operand& op2, const Xbyak::Operand& op3, bool is_tail = false);

  static constexpr int VNNI_ADJ = 4;

// Note: reserve rdx for division
#ifdef _WIN32
  const Xbyak::Reg64& parambase = rcx;
  const Xbyak::Reg64& reg_sum = rdi;
#else
  const Xbyak::Reg64& parambase = rdi;
  const Xbyak::Reg64& reg_sum = rcx;
#endif
  const Xbyak::Reg64& reg_src = rsi;
  const Xbyak::Reg64& reg_dst = r8;
  const Xbyak::Reg64& reg_nsize = r9;
  const Xbyak::Reg64& reg_itern = r10;
  const Xbyak::Reg8& reg_sum_append = r11b;
  const Xbyak::Reg64& reg_ld_src = r12;
  const Xbyak::Reg64& reg_3ld_src = r13;
  const Xbyak::Reg64& reg_ld_dst = r14;
  const Xbyak::Reg64& reg_tmp = r15;

  const Xbyak::Zmm& vpermt2d_arg_idx = zmm31;
  const Xbyak::Zmm& vpshufb_arg_b = zmm30;
  const Xbyak::Zmm& vreg_oneb = zmm29;
  const Xbyak::Zmm& vreg_t1 = zmm28;
  const Xbyak::Zmm& vreg_t2 = zmm27;
  const Xbyak::Zmm& vreg_sum_pad_val = zmm26;
  const Xbyak::Zmm& vreg_zero = zmm25;
  const Xbyak::Opmask& reg_k = k1;
  const Xbyak::Opmask& reg_k_tail = k2;

  Xbyak::Label k_loop;
  Xbyak::Label l_vpermt2d_control;
  Xbyak::Label l_vpshufb_control;
  Xbyak::Label l_n_loop;
  Xbyak::Label l_n_loop_end;
  Xbyak::Label l_ntail_end[3];

  const bool sum_m, is_unsigned;
  int32_t sum_pad_val;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SEQ_CPY_48X4_HPP_
