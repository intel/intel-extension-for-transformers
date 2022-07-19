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

#include "jit_domain/jit_eltwiseop.hpp"
#include "kernels/eltwiseop_types.hpp"
namespace jd {

void jit_eltwiseop_t::generate() {
  this->preamble();
  load_params();
  prepare_mask();
  cmp(remain_element_num, process_element_num());
  jl(reminder_loop_start, T_NEAR);
  L(vectorized_loop_start);
  load_src(reg_src, addr_src);
  eltwise_injector.vector_compute(reg_src, param_.postop_attrs);
  store_dst(reg_src, addr_dst);

  add(addr_src, load_offset());
  add(addr_dst, store_offset());
  sub(remain_element_num, process_element_num());
  cmp(remain_element_num, process_element_num());
  jge(vectorized_loop_start, T_NEAR);

  L(vectorized_loop_end);
  L(reminder_loop_start);
  cmp(remain_element_num, 0);
  jle(reminder_loop_end, T_NEAR);
  load_tail(reg_src, addr_src);
  eltwise_injector.vector_compute(reg_src, param_.postop_attrs);
  store_tail(reg_src, addr_dst);

  add(addr_src, dtype_size(param_.postop_attrs.front()));
  add(addr_dst, dtype_size(param_.postop_attrs.back()));
  dec(remain_element_num);
  jmp(reminder_loop_start, T_NEAR);

  L(reminder_loop_end);
  this->postamble();

  eltwise_injector.prepare_table();
}

void jit_eltwiseop_t::assign_regs() {
  remain_task_mask = Xbyak::Opmask(6);
  scratch_ = Xbyak::Reg64(r10);
  reg_src = Zmm(6);

  auto escape_mock_regs = [&](reg_type type, const std::set<int>& idxs) {
    for (auto&& i : idxs) eltwise_injector.escape_regs(type, i);
  };

  eltwise_injector.escape_regs(reg_type::mask, remain_task_mask.getIdx());
  eltwise_injector.escape_regs(reg_type::reg64, scratch_.getIdx());
  eltwise_injector.escape_regs(reg_type::zmm, reg_src.getIdx());
  eltwise_injector.escape_regs(reg_type::zmm, addr_src.getIdx());
  eltwise_injector.escape_regs(reg_type::zmm, addr_dst.getIdx());
}

void jit_eltwiseop_t::prepare_mask() {
  mov(scratch_.cvt32(), 0x1);
  kmovd(remain_task_mask, scratch_.cvt32());
}

void jit_eltwiseop_t::store_dst(Xbyak::Zmm reg_src, Xbyak::Reg64 addr_dst) {
  auto last_attr = param_.postop_attrs.back();
  if (last_attr.op_alg == postop_alg::quantize) {
    vmovups(ptr[addr_dst], Xmm(reg_src.getIdx()));
  } else {
    vmovups(ptr[addr_dst], reg_src);
  }
}

void jit_eltwiseop_t::load_src(Xbyak::Zmm reg_src, Xbyak::Reg64 addr_src) {
  auto first_attr = param_.postop_attrs.front();
  if (first_attr.op_alg == postop_alg::dequantize) {
    vmovups(Xmm(reg_src.getIdx()), ptr[addr_src]);
  } else {
    vmovups(reg_src, ptr[addr_src]);
  }
}

void jit_eltwiseop_t::load_tail(Xbyak::Zmm reg_src, Xbyak::Reg64 addr_src) {
  auto first_attr = param_.postop_attrs.front();
  if (first_attr.op_alg == postop_alg::dequantize) {
    vmovdqu8(Xmm(reg_src.getIdx()), ptr[addr_src]);
  } else if (first_attr.dt == data_type::fp32) {
    vmovss(Xmm(reg_src.getIdx()), ptr[addr_src]);
  } else if (first_attr.dt == data_type::bf16) {
    vmovups(reg_src, ptr[addr_src]);
  }
}

void jit_eltwiseop_t::store_tail(Xbyak::Zmm reg_src, Xbyak::Reg64 addr_dst) {
  auto last_attr = param_.postop_attrs.back();
  if (last_attr.op_alg == postop_alg::quantize) {
    vmovdqu8(ptr[addr_dst] | remain_task_mask, Xmm(reg_src.getIdx()));
  } else if (last_attr.dt == data_type::fp32) {
    vmovss(ptr[addr_dst] | remain_task_mask, Xmm(reg_src.getIdx()));
  } else if (last_attr.dt == data_type::bf16) {
    Ymm ymm_bf16 = Ymm(reg_src.getIdx());
    vmovdqu16(ptr[addr_dst] | remain_task_mask, ymm_bf16);
  }
}
}  // namespace jd