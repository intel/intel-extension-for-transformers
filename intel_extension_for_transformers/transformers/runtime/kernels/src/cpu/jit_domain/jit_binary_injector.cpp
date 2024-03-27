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

#include "jit_binary_injector.hpp"

namespace jd {

void jit_binary_injector::binary_injector_init(jit_generator* ptr) { h = ptr; }
void jit_binary_injector::set_mask(Opmask mask) { this->mask = mask; }
void jit_binary_injector::compute_vector(const Xmm& zmm_src1, const RegExp& src2, const binaryop_attr& op_attr,
                                         bool enable_mask, bool broadcast) {
  switch (op_attr.op_alg) {
    case binaryop_alg::add:
      add(zmm_src1, src2, op_attr.op_dt, enable_mask, broadcast);
      break;
    case binaryop_alg::sub:
      sub(zmm_src1, src2, op_attr.op_dt, enable_mask, broadcast);
      break;
    case binaryop_alg::mul:
      mul(zmm_src1, src2, op_attr.op_dt, enable_mask, broadcast);
      break;
    case binaryop_alg::per_channel_quant:
      per_channel_quant(zmm_src1, src2, op_attr);
      break;
    case binaryop_alg::per_channel_dequant:
      per_channel_dequant(zmm_src1, src2, op_attr);
      break;
    default:
      SPARSE_LOG(FATAL) << "unsupported binaryalg type.";
      break;
  }
}

void jit_binary_injector::per_channel_quant(const Xmm& src1, const RegExp& src2, const binaryop_attr& op_attr) {
  get_addr(reg_tmp, op_attr, addr_type::scale);
  h->vbroadcastss(zmm_tmp, h->dword[reg_tmp + src2]);
  get_addr(reg_tmp, op_attr, addr_type::zp);
  h->vfmadd213ps(src1, zmm_tmp, h->ptr_b[reg_tmp + src2]);
  if (op_attr.op_dt == data_type::s8) {
    h->vcvtps2dq(src1, src1);  // fp32->s32
  } else if (op_attr.op_dt == data_type::u8) {
    h->vxorps(zmm_tmp, zmm_tmp, zmm_tmp);
    h->vcvtps2udq(src1, src1);  // fp32->u32
    h->vpmaxsd(src1, src1, zmm_tmp);
  } else {
    SPARSE_LOG(FATAL) << "per_channel_quant op only support int8 as input data_type";
  }
}

void jit_binary_injector::per_channel_dequant(const Xmm& src1, const RegExp& src2, const binaryop_attr& op_attr) {
  if (op_attr.op_dt == data_type::u8)
    h->vpmovzxbd(src1, Xmm(src1.getIdx()));  // u8->s32
  else if (op_attr.op_dt == data_type::s8)
    h->vpmovsxbd(src1, Xmm(src1.getIdx()));  // s8->s32
  else
    SPARSE_LOG(FATAL) << "per_channel_dequant op only support int8 as input data_type";
  h->vcvtdq2ps(src1, src1);
  get_addr(reg_tmp, op_attr, addr_type::zp);
  sub(src1, reg_tmp + src2, data_type::fp32, false, true);
  get_addr(reg_tmp, op_attr, addr_type::scale);
  mul(src1, reg_tmp + src2, data_type::fp32, false, true);
}

void jit_binary_injector::mul(const Xmm& src1, const RegExp& src2, data_type op_dt, bool enable_mask, bool broadcast) {
  if (op_dt == data_type::fp16) {
    h->vmulph(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  } else if (op_dt == data_type::fp32) {
    h->vmulps(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  } else {
    SPARSE_LOG(FATAL) << "mul op in binary injector only support floating point arithmetic";
  }
}

void jit_binary_injector::add(const Xmm& src1, const RegExp& src2, data_type op_dt, bool enable_mask, bool broadcast) {
  if (op_dt == data_type::fp32)
    h->vaddps(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  if (op_dt == data_type::s32)
    h->vpaddd(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  if (op_dt == data_type::s8)
    h->vpaddb(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  if (op_dt == data_type::u8)
    h->vpaddusb(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
}

void jit_binary_injector::sub(const Xmm& src1, const RegExp& src2, data_type op_dt, bool enable_mask, bool broadcast) {
  if (op_dt == data_type::fp32)
    h->vsubps(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  if (op_dt == data_type::s32)
    h->vpsubd(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  if (op_dt == data_type::s8)
    h->vpsubb(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
  if (op_dt == data_type::u8)
    h->vpsubusb(enable_mask ? src1 | mask : src1, src1, broadcast ? h->ptr_b[src2] : h->ptr[src2]);
}

void jit_binary_injector::init_quantization(const Xmm& zmm, const Reg64& reg) {
  zmm_tmp = zmm;
  reg_tmp = reg;
}

void jit_binary_injector::get_addr(const Reg64& reg, binaryop_attr op_attr, addr_type type) {
  int64_t addr;
  switch (type) {
    case addr_type::normal:
      addr = reinterpret_cast<int64_t>(op_attr.static_addr);
      break;
    case addr_type::scale:
      addr = reinterpret_cast<int64_t>(op_attr.scale);
      break;
    case addr_type::zp:
      addr = reinterpret_cast<int64_t>(op_attr.zp);
      break;
    default:
      SPARSE_LOG(FATAL) << "unsupported binary_addr type.";
      break;
  }
  h->mov(reg, addr);
}

}  // namespace jd
