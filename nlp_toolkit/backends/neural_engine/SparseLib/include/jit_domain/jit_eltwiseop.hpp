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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_ELTWISEOP_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_ELTWISEOP_HPP_

#include <map>
#include "jit_generator.hpp"
#include "utils.hpp"
#include "kernels/eltwiseop_types.hpp"
#include "jit_domain/jit_eltwise_injector.hpp"

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
  Xbyak::Reg64 reg_param = rdi;
  Zmm reg_src;
  Xbyak::Reg64 addr_src = r15;
  Xbyak::Reg64 addr_dst = r14;
  Xbyak::Reg64 remain_element_num = rsi;

  /* registers for bf16 tasks*/
  Xbyak::Opmask remain_task_mask;
  Xbyak::Reg64 scratch_;

  size_t dtype_size(postop_attr attr) {
    // quantize only happend on first postop,we load the data from memory to zmm,in tail case,the offset is one byte;
    // dequantize only happend on last postop,we store the data from zmm to memory,in taill case,the offset is also one
    // byte.
    if (attr.op_alg == postop_alg::quantize || attr.op_alg == postop_alg::dequantize) return 1u;
    switch (attr.dt) {
      case data_type::fp32:
        return 4u;
      case data_type::bf16:
        return 2u;
    }
  }

  size_t load_offset() {
    auto head_dt = param_.postop_attrs.front().dt;
    switch (head_dt) {
      case data_type::fp32:
      case data_type::bf16:
        return 64u;
      case data_type::u8:  // dequantize case
        return 16u;
    }
  }

  size_t store_offset() {
    if (param_.postop_attrs.back().op_alg == postop_alg::quantize) return 16u;    // quantize case.
    if (param_.postop_attrs.back().op_alg == postop_alg::dequantize) return 64u;  // dequantize case.
    auto tail_dt = param_.postop_attrs.back().dt;
    switch (tail_dt) {
      case data_type::fp32:
      case data_type::bf16:
        return 64u;
    }
  }

  size_t process_element_num() {
    auto front_attr = param_.postop_attrs.front();
    switch (front_attr.dt) {
      case data_type::fp32:
        return 16;
      case data_type::bf16:
        return 32;
      case data_type::u8:  // dequantize case
        return 16;
    }
  }

  void load_params() {
    mov(addr_dst, ptr[reg_param + ELT_GET_OFF(dst)]);
    mov(addr_src, ptr[reg_param + ELT_GET_OFF(src)]);
    mov(remain_element_num, ptr[reg_param + ELT_GET_OFF(element_num)]);
  }
};
}  // namespace jd
#endif
