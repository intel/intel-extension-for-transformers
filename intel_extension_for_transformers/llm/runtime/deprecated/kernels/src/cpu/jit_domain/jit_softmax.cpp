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

#include "jit_softmax.hpp"
#include "kernels/softmax_types.hpp"
namespace jd {
void jit_softmax_t::generate() {
  this->preamble();

  switch (param_.sepc_type) {
    case ssd::spec_softmax_type::lut:
      lut_softmax_kernel_gen();
      break;
    default:
      break;
  }

  this->postamble();
  if (param_.sepc_type == ssd::spec_softmax_type::lut) get_lut_exp_injector.prepare_table();
  eltwise_injector.prepare_table();
}

void jit_softmax_t::lut_softmax_kernel_gen() {
  // for eltwise injector regs allocating, we need to escape the regs which we care.
  for (auto&& i : reg_map) {
    for (auto&& j : i.second) {
      eltwise_injector.escape_regs(i.first, j);
      get_lut_exp_injector.escape_regs(i.first, j);
    }
  }

  prepare_mask();
  mov(reg_tmp, ptr[reg_param + CUSTSM_GET_OFF(one)]);
  vpbroadcastw(zmm_one_bf16, Xbyak::Reg16(reg_tmp.getIdx()));
  vpmovzxwd(zmm_one_fp32, Ymm(zmm_one_bf16.getIdx()));
  vpslld(zmm_one_fp32, zmm_one_fp32, 16);  // bf16->fp32
  mov(src_addr, ptr[reg_param + CUSTSM_GET_OFF(src)]);
  mov(dst_addr, ptr[reg_param + CUSTSM_GET_OFF(dst)]);
  mov(reg_tmp, ptr[reg_param + CUSTSM_GET_OFF(tmp)]);
  mov(vec_num, 0);
  L(process_vec_loop);
  vxorps(zmm_denominator, zmm_denominator, zmm_denominator);
  mov(vec_offset, 0);
  // loop one:max reduction.
  mov(src_addr_volatile, src_addr);
  lut_int8_cvt_int16(zmm_exp_neg_max, src_addr_volatile);
  cmp(vec_offset, param_.vec_align_len);
  je(max_reduction_end, T_NEAR);
  L(max_reduction_loop);
  lut_int8_cvt_int16(zmm_vec, src_addr_volatile);
  vpmaxsw(zmm_exp_neg_max, zmm_exp_neg_max, zmm_vec);
  add(src_addr_volatile, ymm_byte_size);
  add(vec_offset, process_element_16bit);
  cmp(vec_offset, param_.vec_align_len);
  jl(max_reduction_loop);
  L(max_reduction_end);
  if (param_.vec_tail_len != 0) {
    lut_int8_cvt_int16(zmm_vec, src_addr_volatile);
    vpmaxsw(zmm_exp_neg_max | bit16_mask, zmm_exp_neg_max, zmm_vec);
  }
  vshuff32x4(zmm_tmp, zmm_exp_neg_max, zmm_exp_neg_max, 0x4E);  //  horizontal max
  vpmaxsw(zmm_exp_neg_max, zmm_exp_neg_max, zmm_tmp);           // ymm_exp_neg_max contain the max value(s16)
  vpmovsxwd(zmm_exp_neg_max, ymm_exp_neg_max);                  // s16 cvt s32 and store into the zmm_exp_neg_max
  reduce_dwords(zmm_exp_neg_max, zmm_tmp, &CodeGenerator::vpmaxsd);
  if (param_.input_dt == data_type::s8)
    vpmovdb(xmm_exp_neg_max, zmm_exp_neg_max);
  else
    vpmovusdb(xmm_exp_neg_max, zmm_exp_neg_max);
  vpbroadcastb(zmm_exp_neg_max, xmm_exp_neg_max);  // zmm_exp_neg_max contain all max int8 value

  // loop two:sum reduction & stroe exp value.
  mov(src_addr_volatile, src_addr);
  mov(dst_addr_volatile, reg_tmp);
  mov(vec_offset, 0);
  cmp(vec_offset, param_.vec_align_len);
  je(sum_reduction_end, T_NEAR);
  L(sum_reduction_loop);
  lut_handle_exp();
  add(src_addr_volatile, isa_available(avx512_core_bf16) ? ymm_byte_size : xmm_byte_size);
  add(dst_addr_volatile, zmm_byte_size);
  add(vec_offset, isa_available(avx512_core_bf16) ? process_element_16bit : process_element_32bit);
  cmp(vec_offset, param_.vec_align_len);
  jl(sum_reduction_loop);
  L(sum_reduction_end);
  if (param_.vec_tail_len != 0) lut_handle_exp(true);

  reduce_dwords(zmm_denominator, zmm_tmp, &CodeGenerator::vaddps);  // horizontal sum
  vdivps(zmm_denominator, zmm_one_fp32, zmm_denominator);           // calculate denominator

  // loop3: calculate softmax,apply unroll optimization.
  mov(dst_addr_volatile, dst_addr);
  mov(src_addr_volatile, reg_tmp);
  mov(vec_offset, 0);
  cmp(vec_offset, param_.vec_align_len);
  je(softmax_end, T_NEAR);
  L(softmax_loop);
  get_unroll();
  if (isa_available(avx512_core_bf16)) {
    for (int i = 0; i < unroll; i++)
      vmovups(Ymm(10 + i), ptr[src_addr_volatile + i * ymm_byte_size]);  // exp value(bf16)
    for (int i = 0; i < unroll; i++) bf16_cvt_fp32(Zmm(10 + i));
  } else {
    for (int i = 0; i < unroll; i++)
      vmovups(Zmm(10 + i), ptr[src_addr_volatile + i * zmm_byte_size]);  // exp value(bf16)
  }
  for (int i = 0; i < unroll; i++) vmulps(Zmm(10 + i), Zmm(10 + i), zmm_denominator);

  // apply postops
  if (param_.postop_attrs.size() != 0) {
    for (int i = 0; i < unroll; i++) eltwise_injector.vector_compute(Zmm(10 + i), param_.postop_attrs);
  }

  // store data.
  for (int i = 0; i < unroll; i++) lut_store_data(10 + i, dst_addr_volatile, i);
  add(src_addr_volatile, isa_available(avx512_core_bf16) ? unroll * ymm_byte_size : unroll * zmm_byte_size);
  add(dst_addr_volatile, unroll * 16 * get_data_size(param_.output_dt));
  add(vec_offset, process_element_32bit * unroll);
  cmp(vec_offset, param_.vec_align_len);
  jl(softmax_loop);
  L(softmax_end);
  if (param_.vec_tail_len != 0) {
    if (isa_available(avx512_core_bf16)) {
      vmovups(ymm_vec, ptr[src_addr_volatile]);  // exp value(bf16)
      bf16_cvt_fp32(zmm_vec);
    } else {
      vmovups(zmm_vec, ptr[src_addr_volatile]);  // exp value(bf16)
    }
    vmulps(zmm_vec, zmm_vec, zmm_denominator);
    if (param_.postop_attrs.size() != 0) eltwise_injector.vector_compute(zmm_vec, param_.postop_attrs);
    lut_store_data(zmm_vec.getIdx(), dst_addr_volatile, 0, true);
  }

  add(src_addr, param_.vec_align_len + param_.vec_tail_len);
  add(dst_addr, get_data_size(param_.output_dt) * (param_.vec_align_len + param_.vec_tail_len));
  if (isa_available(avx512_core_bf16)) {
    add(reg_tmp, get_data_size(data_type::bf16) * (param_.vec_align_len + param_.vec_tail_len));
  } else {
    add(reg_tmp, get_data_size(data_type::fp32) * (param_.vec_align_len + param_.vec_tail_len));
  }
  inc(vec_num);
  cmp(vec_num, ptr[reg_param + CUSTSM_GET_OFF(process_vec_num)]);
  jl(process_vec_loop);
}

void jit_softmax_t::lut_int8_cvt_int16(Zmm dst, Reg64 src) {
  if (param_.input_dt == data_type::s8)
    vpmovsxbw(dst, ptr[src]);  // s8->s16
  else
    vpmovzxbw(dst, ptr[src]);  // u8->s16
}

void jit_softmax_t::lut_handle_exp(bool tail) {
  vmovups(ymm_vec, ptr[src_addr_volatile]);
  vpsubsb(ymm_vec, ymm_vec, ymm_exp_neg_max);                              // x-max
  get_lut_exp_injector.vector_compute(zmm_vec, param_.get_lut_exp_attrs);  // e^(x-max)
  if (!tail) {
    if (isa_available(avx512_core_bf16)) {
      vdpbf16ps(zmm_denominator, zmm_one_bf16, zmm_vec);
    } else {
      bf16_cvt_fp32(zmm_vec);
      vaddps(zmm_denominator, zmm_denominator, zmm_vec);
    }
    vmovups(ptr[dst_addr_volatile], zmm_vec);  // now tmp store the bf16/fp32 exp value
  } else {
    if (isa_available(avx512_core_bf16)) vmovdqu16(ptr[dst_addr_volatile] | bit16_mask, zmm_vec);
    if (param_.vec_tail_len < 16) {
      bf16_cvt_fp32(zmm_vec);
      vaddps(zmm_denominator | bit32_mask, zmm_denominator, zmm_vec);
      if (!isa_available(avx512_core_bf16)) vmovups(ptr[dst_addr_volatile] | bit16_mask, zmm_vec);
    } else {
      vextractf32x8(Ymm(zmm_tmp.getIdx()), zmm_vec, 1);
      bf16_cvt_fp32(zmm_vec);
      vaddps(zmm_denominator, zmm_denominator, zmm_vec);
      bf16_cvt_fp32(zmm_tmp);
      vaddps(zmm_denominator | bit32_mask, zmm_denominator, zmm_tmp);
      if (!isa_available(avx512_core_bf16)) {
        vmovups(ptr[dst_addr_volatile], zmm_vec);
        vmovups(ptr[dst_addr_volatile + zmm_byte_size] | bit16_mask, zmm_tmp);
      }
    }
  }
}

void jit_softmax_t::lut_store_data(int simd_idx, Reg64 dst, int offset, bool mask) {
  if (param_.output_dt == data_type::u8) {
    vpmovusdb(mask ? ptr[dst + offset * xmm_byte_size] | bit32_mask : ptr[dst + offset * xmm_byte_size], Zmm(simd_idx));
  } else if (param_.output_dt == data_type::s8) {
    vpmovsdb(mask ? ptr[dst + offset * xmm_byte_size] | bit32_mask : ptr[dst + offset * xmm_byte_size], Zmm(simd_idx));
  } else if (param_.output_dt == data_type::bf16) {
    fp32_cvt_bf16(Zmm(simd_idx));
    vmovdqu16(mask ? ptr[dst + offset * ymm_byte_size] | bit32_mask : ptr[dst + offset * ymm_byte_size], Ymm(simd_idx));
  } else {
    vmovups(mask ? ptr[dst + offset * zmm_byte_size] | bit32_mask : ptr[dst + offset * zmm_byte_size], Zmm(simd_idx));
  }
}

void jit_softmax_t::get_unroll() {
  unroll = 8;
  while (param_.vec_align_len % (unroll * process_element_32bit) != 0) unroll--;
  if (param_.vec_tail_len >= 16) unroll += 1;
}

void jit_softmax_t::prepare_mask() {
  int mask = 0x0;
  for (size_t i = 0; i < param_.vec_tail_len; i++) mask = (mask << 1) + 1;
  mov(reg_tmp.cvt32(), mask);
  kmovd(bit16_mask, reg_tmp.cvt32());
  mask = 0x0;
  for (size_t i = 0; i < param_.vec_tail_len % 16; i++) mask = (mask << 1) + 1;
  mov(reg_tmp.cvt32(), mask);
  kmovd(bit32_mask, reg_tmp.cvt32());
}

void jit_softmax_t::assign_regs() {
#ifdef _WIN32
  reg_param = rcx;
#else
  reg_param = rdi;
#endif
  src_addr = r8;
  dst_addr = r9;
  vec_num = r10;
  vec_offset = r11;
  reg_tmp = r12;
  src_addr_volatile = r13;
  dst_addr_volatile = r14;
  reg_map.insert(std::pair<reg_type, std::set<int>>(
      reg_type::reg64, {src_addr.getIdx(), dst_addr.getIdx(), vec_num.getIdx(), vec_offset.getIdx(), reg_tmp.getIdx(),
                        src_addr_volatile.getIdx(), dst_addr_volatile.getIdx()}));

  bit16_mask = Opmask(6);
  bit32_mask = Opmask(7);
  reg_map.insert(std::pair<reg_type, std::set<int>>(reg_type::mask, {bit16_mask.getIdx(), bit16_mask.getIdx()}));

  zmm_vec = Zmm(0);
  ymm_vec = ymm0;
  zmm_denominator = Zmm(1);
  zmm_exp_neg_max = Zmm(2);
  ymm_exp_neg_max = ymm2;
  xmm_exp_neg_max = xmm2;
  zmm_one_fp32 = Zmm(4);
  zmm_one_bf16 = Zmm(5);
  zmm_tmp = Zmm(3);
  reg_map.insert(std::pair<reg_type, std::set<int>>(
      reg_type::zmm, {zmm_vec.getIdx(), zmm_denominator.getIdx(), zmm_exp_neg_max.getIdx(), zmm_one_bf16.getIdx(),
                      zmm_one_fp32.getIdx()}));
}
}  // namespace jd
