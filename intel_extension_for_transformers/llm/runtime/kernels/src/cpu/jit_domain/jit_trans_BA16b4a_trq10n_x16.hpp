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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_BA16B4A_TRQ10N_X16_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_BA16B4A_TRQ10N_X16_HPP_

#include "jit_generator.hpp"
#include "regs_pool.hpp"

namespace jd {

/**
 * @brief jit_trans_BA16b4a_trq10n_x16 transpose a matrix of bytes of size mx16 (ab) to (1)x(m/4)x16x4 (BA16b4a) while
 * performing de-q10n and re-q10n in along "transpsoed channels"(dst_scale size: 1x16). Paddings are set to zero.
 */
class jit_trans_BA16b4a_trq10n_x16 : public jit_generator {
 public:
  struct rt_data_t {
    const int8_t* src;
    int8_t* dst;
    const float* src_scale;
    float* dst_scale;
    int ld_src;
    int M;
    int N;  // less or equal to 16, will be padded with 0
  };

  jit_trans_BA16b4a_trq10n_x16() : jit_generator() {}
  virtual ~jit_trans_BA16b4a_trq10n_x16() {}

 private:
  void generate() override;

  static constexpr uint64_t vpshufb_mask_bw = 0x2222222222222222;
  static constexpr uint64_t vpshufb_mask_wd = 0xcccccccccccccccc;

  Xbyak::Label l_vpshufb_bw_ctl, l_vpshufb_wd_ctl, L_int32_max, L_127f;
  Xbyak::Label l_nsize_tbl, l_nsize_case[17];
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_TRANS_BA16B4A_TRQ10N_X16_HPP_
