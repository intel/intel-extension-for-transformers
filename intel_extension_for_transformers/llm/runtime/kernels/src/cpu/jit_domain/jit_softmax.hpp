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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SOFTMAX_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SOTFMAX_HPP_

#include <utility>
#include <vector>
#include <map>
#include <set>
#include "src/cpu/cpu_isa.hpp"
#include "jit_generator.hpp"
#include "src/utils.hpp"
#include "kernels/softmax_types.hpp"
#include "jit_eltwise_injector.hpp"

#define CUSTSM_GET_OFF(field) offsetof(ssd::softmax_data_t, field)

namespace jd {
class jit_softmax_t : public jit_generator {
 public:
  explicit jit_softmax_t(const ssd::softmax_param_t& param) : jit_generator(), param_(param) {
    assign_regs();
    eltwise_injector.eltwise_injector_init(this, param_.postop_attrs);
    if (param_.sepc_type == ssd::spec_softmax_type::lut)
      get_lut_exp_injector.eltwise_injector_init(this, param_.get_lut_exp_attrs);
  }
  virtual ~jit_softmax_t() {}

 private:
  void generate() override;
  void assign_regs();
  void prepare_mask();
  void get_unroll();
  void lut_softmax_kernel_gen();
  void lut_int8_cvt_int16(Zmm dst, Reg64 src);
  void lut_store_data(int simd_idx, Reg64 dst, int offset = 0, bool mask = false);
  void lut_handle_exp(bool tail = false);

 private:
  ssd::softmax_param_t param_;
  jit_eltwise_injector eltwise_injector;
  jit_eltwise_injector get_lut_exp_injector;
  std::map<reg_type, std::set<int>> reg_map;
  const size_t xmm_byte_size = 16;
  const size_t ymm_byte_size = 32;
  const size_t zmm_byte_size = 64;
  const size_t process_element_16bit = 32;
  const size_t process_element_32bit = 16;
  int unroll = 8;

  Reg64 reg_param;
  Reg64 src_addr;
  Reg64 src_addr_volatile;
  Reg64 dst_addr;
  Reg64 dst_addr_volatile;
  Reg64 vec_num;
  Reg64 vec_offset;  // load/sotre offset
  Reg64 reg_tmp;     // store max/sum
  Opmask bit16_mask;
  Opmask bit32_mask;
  Zmm zmm_vec;
  Ymm ymm_vec;
  Zmm zmm_tmp;
  Zmm zmm_exp_neg_max;
  Zmm zmm_one_bf16;
  Zmm zmm_one_fp32;
  Ymm ymm_exp_neg_max;
  Xmm xmm_exp_neg_max;
  Zmm zmm_denominator;  // broadcast sum to this zmm reg and then mul e^-M.

  Xbyak::Label process_vec_loop;
  Xbyak::Label max_reduction_loop;
  Xbyak::Label max_reduction_end;
  Xbyak::Label sum_reduction_loop;
  Xbyak::Label sum_reduction_end;
  Xbyak::Label softmax_loop;
  Xbyak::Label softmax_end;
};  // namespace jd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SOFTMAX_HPP_
