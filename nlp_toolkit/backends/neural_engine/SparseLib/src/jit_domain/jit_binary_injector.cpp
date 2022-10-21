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

#include "jit_domain/jit_binary_injector.hpp"

namespace jd {

void jit_binary_injector::binary_injector_init(jit_generator* ptr) { h = ptr; }
void jit_binary_injector::set_mask(Opmask mask) { this->mask = mask; }
void jit_binary_injector::compute_vector(Zmm zmm_src1, RegExp src2, binaryop_attr op_attr, data_type op_dt,
                                         bool enable_mask, bool broadcast) {
  switch (op_attr.op_alg) {
    case binaryop_alg::add:
      add(zmm_src1, src2, op_dt, enable_mask, broadcast);
      break;

    default:

      break;
  }
}

void jit_binary_injector::add(Zmm src1, RegExp src2, data_type op_dt, bool enable_mask, bool broadcast) {
  if (op_dt == data_type::fp32)
    h->vaddps(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  if (op_dt == data_type::s32)
    h->vpaddd(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  if (op_dt == data_type::s8)
    h->vpaddb(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  if (op_dt == data_type::u8)
    h->vpaddusb(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
}

void jit_binary_injector::get_addr(Reg64 reg, binaryop_attr op_attr) {
  int64_t addr = reinterpret_cast<int64_t>(op_attr.src_addr);
  h->mov(reg, addr);
}

}  // namespace jd
