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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_ELTWISEOP_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_ELTWISEOP_HPP_

#include <map>
#include "jit_generator.hpp"
#include "src/utils.hpp"
#include "kernels/eltwiseop_types.hpp"
#include "jit_eltwise_injector.hpp"

#define ELT_GET_OFF(field) offsetof(ssd::eltwiseop_data_t, field)

namespace jd {
class jit_eltwiseop_t : public jit_generator {
  using Zmm = Xbyak::Zmm;
  using Ymm = Xbyak::Ymm;
  using Xmm = Xbyak::Xmm;

 public:
  explicit jit_eltwiseop_t(const ssd::eltwiseop_param_t& param) : jit_generator(), param_(param) {
    eltwise_injector.eltwise_injector_init(this, param_.postop_attrs);
    assign_regs();
  }
  virtual ~jit_eltwiseop_t() {}

 private:
  void generate() override;
  void assign_regs();
  void store_dst(Xbyak::Zmm reg_src, Xbyak::Reg64 dst_addr);
  void store_tail(Xbyak::Zmm reg_src, Xbyak::Reg64 dst_addr);
  void load_src(Xbyak::Zmm reg_src, Xbyak::Reg64 src_addr);
  void load_tail(Xbyak::Zmm reg_src, Xbyak::Reg64 src_addr);
  void prepare_mask();

 private:
  ssd::eltwiseop_param_t param_;
  jit_eltwise_injector eltwise_injector;

  /*labels*/
  Xbyak::Label vectorized_loop_start;
  Xbyak::Label vectorized_loop_end;
  Xbyak::Label reminder_loop_start;
  Xbyak::Label reminder_loop_end;

  /* registers for fwd*/
#ifdef _WIN32
  Xbyak::Reg64 reg_param = rcx;
#else
  Xbyak::Reg64 reg_param = rdi;
#endif
  Zmm reg_src;
  Xbyak::Reg64 addr_src = r15;
  Xbyak::Reg64 addr_dst = r14;
  Xbyak::Reg64 remain_element_num = rsi;

  /* registers for bf16 tasks*/
  Xbyak::Opmask remain_task_mask;
  Xbyak::Reg64 scratch_;

  size_t load_offset() {
    if (param_.postop_attrs[0].op_alg == postop_alg::eltop_int_lut && param_.postop_attrs[0].alpha == 8) {
      return 64u;  // special case:bit8_lut
    }
    if (param_.postop_attrs[0].op_alg == postop_alg::eltop_int_lut && param_.postop_attrs[0].alpha == 16) {
      return 32u;  // special case:bit16_lut
    }
    if (param_.postop_attrs[0].op_alg == postop_alg::quantize) {
      return 64u;  // special case:direct quantize
    }
    auto head_dt = param_.postop_attrs.front().dt;
    switch (head_dt) {
      case data_type::fp32:
      case data_type::bf16:
        return 64u;
      case data_type::u8:  // dequantize case
      case data_type::s8:
        return 16u;
      default:
        SPARSE_LOG(ERROR) << "wrong head data type, expect fp32/bf16/u8" << std::endl;
        return 0u;
    }
  }

  size_t store_offset() {
    // todo:except dequantize case, our zmm always full of result data and needs to be stored.
    if (param_.postop_attrs.front().op_alg == postop_alg::eltop_int_lut) return 64u;  // lut case;
    if (param_.postop_attrs.back().op_alg == postop_alg::quantize) return 16u;        // quantize case.
    if (param_.postop_attrs.back().op_alg == postop_alg::dequantize) return 64u;      // dequantize case.
    switch (param_.out_dt) {
      case data_type::fp32:
      case data_type::bf16:
        return 64u;
      default:
        SPARSE_LOG(ERROR) << "wrong output data type, expect fp32/bf16" << std::endl;
        return 0u;
    }
  }

  size_t process_element_num() {
    auto front_attr = param_.postop_attrs.front();
    if (front_attr.op_alg == postop_alg::eltop_int_lut && front_attr.alpha == 8) return 64;   // special case:bit8_lut
    if (front_attr.op_alg == postop_alg::eltop_int_lut && front_attr.alpha == 16) return 32;  // sepcial case:bit16_lut
    switch (param_.in_dt) {
      case data_type::fp32:
        return 16;
      case data_type::bf16:
        return 32;
      case data_type::u8:  // dequantize case
      case data_type::s8:
        return 16;
      default:
        SPARSE_LOG(ERROR) << "wrong input data type, expect fp32/bf16/u8" << std::endl;
        return 0u;
    }
  }

  void load_params() {
    mov(addr_dst, ptr[reg_param + ELT_GET_OFF(dst)]);
    mov(addr_src, ptr[reg_param + ELT_GET_OFF(src)]);
    mov(remain_element_num, ptr[reg_param + ELT_GET_OFF(element_num)]);
  }
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_ELTWISEOP_HPP_
