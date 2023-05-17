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

#include "jit_gather.hpp"
#include "kernels/gather_types.hpp"

namespace jd {
void jit_gather_t::generate() {
  this->preamble();
  load_params();
  mov(gather_idx.cvt32(), dword[idx_addr]);
  mov(next_gather_idx.cvt32(), dword[idx_addr + 4]);
  imul(gather_idx, gather_idx, param_.inner_size * param_.dt_size);
  imul(next_gather_idx, next_gather_idx, param_.inner_size * param_.dt_size);
  prefetchnta(ptr[next_gather_idx + src_addr]);
  add(src_addr, gather_idx);
  int offset;
  for (int m = 0; m < param_.loops; ++m) {
    offset = m * 512 / 8;
    vmovdqu32(Zmm(m % 16), ptr[src_addr + offset]);
    for (size_t i = 0; i < param_.binaryop_attrs.size(); i++) {
      RegExp offset_exp = binaryop_addr[i] + offset;  //
      binary_injector.compute_vector(Zmm(m % 16), offset_exp, param_.binaryop_attrs[i]);
    }
    vmovdqu32(ptr[dst_addr + offset], Zmm(m % 16));
  }
  // tail
  if (param_.mask != 0) {
    if (param_.binaryop_attrs.size() > 0) {
      mov(r10, param_.mask);
      kmovq(tail_mask, r10);
      binary_injector.set_mask(tail_mask);
    }
    mov(r10, param_.extend_mask);
    kmovq(extend_tail_mask, r10);
    offset = param_.loops * 512 / 8;
    vmovdqu8(Zmm(0) | extend_tail_mask, ptr[src_addr + offset]);
    for (size_t i = 0; i < param_.binaryop_attrs.size(); i++) {
      RegExp offset_exp = binaryop_addr[i] + offset;
      binary_injector.compute_vector(Zmm(0), offset_exp, param_.binaryop_attrs[i], true);
    }
    vmovdqu8(ptr[dst_addr + offset] | extend_tail_mask, Zmm(0));
  }
  this->postamble();
}

void jit_gather_t::assign_regs() {
  src_addr = r15;
  idx_addr = r14;
  dst_addr = r13;
  gather_idx = r12;
  next_gather_idx = r11;
  for (size_t i = 0; i < param_.binaryop_attrs.size(); i++) {
    binaryop_addr.push_back(Reg64(i));
  }
  tail_mask = Xbyak::Opmask(2);
  extend_tail_mask = Xbyak::Opmask(3);
#ifdef _WIN32
  reg_param = rcx;
#else
  reg_param = rdi;
#endif
}

}  // namespace jd
