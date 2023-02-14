//  Copyright (c) 2021 Intel Corporationit_softmax_Ab16a
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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SOFTMAX_AB16A_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SOFTMAX_AB16A_HPP_

#include <glog/logging.h>
#include <vector>
#include <array>
#include "utils.hpp"
#include "jit_generator.hpp"

namespace jd {

class jit_softmax_Ab16a : public jit_generator {
 public:
  struct params {
    int att_tail;
    int sl_pad64_;
    std::string output_type = "s8";

    params() = default;
    params(int att_tail, int sl_pad64_, std::string output_type)
        : att_tail(att_tail), sl_pad64_(sl_pad64_), output_type(output_type) {}
  };

  struct rt_data_t {
    const int32_t* src;
    uint8_t* dst;
    int32_t att_tile;
    uint16_t softmax_rescale;
    float QK_rescale;
  };

  explicit jit_softmax_Ab16a(const params& param) : jit_generator(), param_(param) {
    if (param_.output_type == "u8") {
      min = fp32_to_fp16(0.0f);
      max = fp32_to_fp16(255.0f);
    } else {
      min = fp32_to_fp16(-127.0f);
      max = fp32_to_fp16(127.0f);
    }

    assign_regs();
  }
  virtual ~jit_softmax_Ab16a() {}

 private:
  void cvtepi16_epi8_shuffle_storeu(RegExp dst, const Zmm& zmm0, const Zmm& zmm1, const Zmm& zmm2, const Zmm& zmm3,
                                    const Zmm& tmp0, const Zmm& tmp1, const Zmm& tmp2, const Zmm& tmp3);
  void exp_ph_0_1(const Zmm& src, const Zmm& dst, const Zmm& magic_number0, const Zmm& magic_number1,
                  const Zmm& magic_number2, const Zmm& magic_number3, const Zmm& magic_number4, const Reg64 bd);
  void scaleph2i16(const Zmm& zmm, const Zmm& zmm_scale, const Zmm& tmp1, const Zmm& tmp2);
  void assign_regs() {
    src_addr = r8;
    dst_addr = r9;
    att_tile = r10d;
    mask = k2;
#ifdef _WIN32
    reg_param = rcx;
#else
    reg_param = rdi;
#endif
  }

  void generate() override;
  Reg64 reg_param;
  params param_;
  int i_tile_w = 16, i_tile_h = 16;
  Reg64 src_addr;
  Reg64 dst_addr;
  Xbyak::Reg32 att_tile;
  Opmask mask;
  uint16_t min;
  uint16_t max;
  const uint16_t magic_number[5] = {
      fp32_to_fp16(1.442695f),    fp32_to_fp16(0.240226507f), fp32_to_fp16(0.452920674f),
      fp32_to_fp16(0.713483036f), fp32_to_fp16(0.5f),
  };
  const int16_t perm_data[32] = {1,  0,  3,  2,  5,  4,  7,  6,  9,  8,  11, 10, 13, 12, 15, 14,
                                 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30};
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SOFTMAX_AB16A_HPP_
