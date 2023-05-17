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

#include "jit_slice.hpp"
#include "kernels/slice_types.hpp"

namespace jd {

void jit_slice_t::copy_continuously() {
  int loops = param_.copy_size / (512 / 8);
  int offset;
  for (int m = 0; m < loops; ++m) {
    offset = m * 512 / 8;
    vmovdqu32(Zmm(m % 16), ptr[src_addr + offset]);
    vmovdqu32(ptr[dst_addr + offset], Zmm(m % 16));
  }

  int remain = param_.copy_size % (512 / 8) / param_.dt_size;
  int mask = (1LL << remain) - 1, extend_mask = (1LL << (remain * param_.dt_size)) - 1;
  // tail
  if (mask != 0) {
    mov(r10, extend_mask);
    kmovq(extend_tail_mask, r10);
    offset = loops * 512 / 8;
    vmovdqu8(Zmm(0) | extend_tail_mask, ptr[src_addr + offset]);
    vmovdqu8(ptr[dst_addr + offset] | extend_tail_mask, Zmm(0));
  }
}

void jit_slice_t::copy_by_step() {
  int loops = param_.copy_size / (512 / 8);
  int src_offset;
  int dst_offset;
  for (int m = 0; m < loops; ++m) {
    src_offset = m * 512 / 8;
    dst_offset = m * 256 / 8;
    vmovdqu32(Zmm(m % 16), ptr[src_addr + src_offset]);
    switch (param_.dt_size) {
      case 4:
        vpmovqd(ptr[dst_addr + dst_offset], Zmm(m % 16));
        break;
      case 2:
        vpmovdw(ptr[dst_addr + dst_offset], Zmm(m % 16));
        break;
      case 1:
        vpmovwb(ptr[dst_addr + dst_offset], Zmm(m % 16));
        break;
      default:
        break;
    }
  }
  int remain = (param_.copy_size % (512 / 8)) / param_.dt_size;
  int mask = (1LL << remain) - 1;
  // tail
  if (mask != 0) {
    mov(r10.cvt32(), mask);
    kmovd(extend_tail_mask, r10.cvt32());
    src_offset = loops * 512 / 8;
    dst_offset = loops * 256 / 8;
    vmovdqu32(Zmm(0), ptr[src_addr + src_offset]);
    switch (param_.dt_size) {
      case 4:
        vpmovqd(ptr[dst_addr + dst_offset] | extend_tail_mask, Zmm(0));
        break;
      case 2:
        vpmovdw(ptr[dst_addr + dst_offset] | extend_tail_mask, Zmm(0));
        break;
      case 1:
        vpmovwb(ptr[dst_addr + dst_offset] | extend_tail_mask, Zmm(0));
        break;
      default:
        break;
    }
  }
}

void jit_slice_t::generate() {
  this->preamble();
  load_params();
  if (param_.inner_size > 1 && param_.step > 1)
    prefetchnta(ptr[param_.inner_size * param_.step * param_.dt_size + src_addr]);
  else
    prefetchnta(ptr[param_.inner_size * param_.src_axis_size * param_.dt_size + src_addr]);
  if (param_.inner_size == 1 && param_.step > 1)
    copy_by_step();
  else
    copy_continuously();
  this->postamble();
}

void jit_slice_t::assign_regs() {
  src_addr = r15;
  dst_addr = r13;
  extend_tail_mask = Xbyak::Opmask(3);
#ifdef _WIN32
  reg_param = rcx;
#else
  reg_param = rdi;
#endif
}

}  // namespace jd
