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

#include "jit_domain/jit_softmax.hpp"
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
  eltwise_injector.prepare_table();
}

void jit_softmax_t::lut_softmax_kernel_gen() {
  prepare_mask();
  mov(src_addr, ptr[reg_param + CUSTSM_GET_OFF(src)]);
  mov(dst_addr, ptr[reg_param + CUSTSM_GET_OFF(dst)]);
  // for eltwise injector regs allocating, we need to escape the regs which we care.
  for (auto&& i : reg_map) {
    for (auto&& j : i.second) eltwise_injector.escape_regs(i.first, j);
  }

  vxorps(zmm_exp_neg_max, zmm_exp_neg_max, zmm_exp_neg_max);  // risk:max<0
  mov(vec_num, 0);
  mov(vec_offset, 0);
  L(process_vec_loop);
  // loop one:max reduction.
  mov(src_addr_volatile, src_addr);
  cmp(vec_offset, param_.vec_align_len);
  je(max_reduction_end, T_NEAR);
  L(max_reduction_loop);
  vpmovsxbw(zmm_vec, ptr[src_addr_volatile]);  // s8/u8->s16
  vpmaxsw(zmm_exp_neg_max, zmm_exp_neg_max, zmm_vec);
  add(src_addr_volatile, ymm_byte_size);
  add(vec_offset, process_element_16bit);
  cmp(vec_offset, param_.vec_align_len);
  jl(max_reduction_loop);
  L(max_reduction_end);
  if (param_.vec_tail_len != 0) {
    vpmovsxbw(zmm_vec, ptr[src_addr_volatile]);
    vpmaxsw(zmm_exp_neg_max | tail_mask, zmm_exp_neg_max, zmm_vec);
  }
  //  horizontal max
  vshuff32x4(zmm_tmp, zmm_exp_neg_max, zmm_exp_neg_max, 0x4E);
  vpmaxsw(zmm_exp_neg_max, zmm_exp_neg_max, zmm_tmp);  // ymm_exp_neg_max contain the max value(s16)
  vpmovsxwd(zmm_exp_neg_max, ymm_exp_neg_max);         // s16 cvt s32 and store into the zmm_exp_neg_max
  get_horizontal_op(zmm_exp_neg_max, zmm_tmp, op_t::max);
  if (param_.input_dt == data_type::s8)
    vpmovsdb(xmm_exp_neg_max, zmm_exp_neg_max);
  else
    vpmovusdb(xmm_exp_neg_max, zmm_exp_neg_max);                          // cvt to s8/u8 for look up
  eltwise_injector.vector_compute(zmm_exp_neg_max, param_.postop_attrs);  // e^M(bf16)
  vpmovzxwd(zmm_exp_neg_max, ymm_exp_neg_max);
  vpslld(zmm_exp_neg_max, zmm_exp_neg_max, 16);         // bf16->fp32
  rcpss(xmm_exp_neg_max, xmm_exp_neg_max);              // first element in xmm_exp_neg_max is fp32 e^(-M)
  vpbroadcastd(zmm_exp_neg_max_fp32, xmm_exp_neg_max);  // all element in zmm_exp_neg_max_fp32 are fp32 e^(-M)
  vcvtneps2bf16(ymm_exp_neg_max, zmm_exp_neg_max);      // fp32 e^(-M) cvt bf16 e^(-M)
  vpbroadcastw(zmm_exp_neg_max, xmm_exp_neg_max);       // all element in zmm_exp_neg_max are bf16 e^(-M)

  // loop two:sum reduction & stroe exp value.
  mov(src_addr_volatile, src_addr);
  mov(dst_addr_volatile, dst_addr);
  mov(vec_offset, 0);
  cmp(vec_offset, param_.vec_align_len);
  je(sum_reduction_end, T_NEAR);
  L(sum_reduction_loop);
  vmovups(ymm_vec, ptr[src_addr_volatile]);
  eltwise_injector.vector_compute(zmm_vec, param_.postop_attrs);
  vmovups(ptr[dst_addr_volatile], zmm_vec);  // now dst store the bf16 exp value
  vdpbf16ps(zmm_scale, zmm_exp_neg_max, zmm_vec);
  add(src_addr_volatile, ymm_byte_size);
  add(dst_addr_volatile, zmm_byte_size);
  add(vec_offset, process_element_16bit);
  cmp(vec_offset, param_.vec_align_len);
  jl(sum_reduction_loop);
  L(sum_reduction_end);
  if (param_.vec_tail_len != 0) {
    vmovups(ymm_vec, src_addr_volatile);
    eltwise_injector.vector_compute(zmm_vec, param_.postop_attrs);
    vdpbf16ps(zmm_scale | tail_mask, zmm_exp_neg_max, zmm_vec);
  }
  // horizontal sum
  get_horizontal_op(zmm_scale, zmm_tmp, op_t::sum);
  // calculate the scale
  vdivps(zmm_scale, zmm_exp_neg_max_fp32, zmm_scale);

  // loop3: calculate softmax,apply unroll optimization.
  mov(dst_addr_volatile, dst_addr);
  mov(src_addr_volatile, dst_addr);
  mov(vec_offset, 0);
  cmp(vec_offset, param_.vec_align_len);
  je(softmax_end, T_NEAR);
  L(softmax_loop);
  get_unroll();
  for (int i = 0; i < unroll; i++) vmovups(Ymm(10 + i), ptr[src_addr_volatile + i * ymm_byte_size]);  // exp value(bf16)

  for (int i = 0; i < unroll; i++) vpmovzxwd(Zmm(10 + i), Ymm(10 + i));

  for (int i = 0; i < unroll; i++) vpslld(Zmm(10 + i), Zmm(10 + i), 16);  // bf16->fp32

  for (int i = 0; i < unroll; i++) vmulps(Zmm(10 + i), Zmm(10 + i), zmm_scale);

  for (int i = 0; i < unroll; i++) vcvtneps2bf16(Ymm(10 + i), Zmm(10 + i));

  for (int i = 0; i < unroll; i++) vmovups(ptr[dst_addr_volatile + i * ymm_byte_size], Ymm(10 + i));
  add(src_addr_volatile, unroll * ymm_byte_size);
  add(dst_addr_volatile, unroll * ymm_byte_size);
  add(vec_offset, process_element_32bit * unroll);
  cmp(vec_offset, param_.vec_align_len);
  jl(softmax_loop);
  L(softmax_end);
  if (param_.vec_tail_len != 0) {
    vmovups(ymm_vec, ptr[dst_addr_volatile]);  // exp value(bf16)
    vpmovzxwd(zmm_vec, ymm_vec);
    vpslld(zmm_vec, zmm_vec, 16);  // bf16->fp32
    vmulps(zmm_vec, zmm_vec, zmm_scale);
    vcvtneps2bf16(ymm_vec, zmm_vec);
    vmovups(ptr[dst_addr_volatile] | tail_mask, ymm_vec);
  }

  // move src_addr ptr & dst_addr ptr
  // inc vec_num
  add(src_addr, param_.vec_align_len + param_.vec_tail_len);
  add(dst_addr, get_data_size(param_.output_dt) * (param_.vec_align_len + param_.vec_tail_len));
  inc(vec_num);
  cmp(vec_num, ptr[reg_param + CUSTSM_GET_OFF(process_vec_num)]);
  jl(process_vec_loop);
}

void jit_softmax_t::get_unroll() {
  unroll = 8;
  while (param_.vec_align_len % (unroll * process_element_32bit) != 0) unroll--;
}

void jit_softmax_t::prepare_mask() {
  int mask = 0x0;
  for (int i = 0; i < param_.vec_tail_len; i++) mask = (mask << 1) + 1;
  mov(reg_tmp.cvt32(), mask);
  kmovd(tail_mask, reg_tmp.cvt32());
}

void jit_softmax_t::get_horizontal_op(const Zmm& v, const Zmm& vtmp, op_t op) {
  vshuff32x4(vtmp, v, v, 0x4E);  // 256-bit shuffle
  perform_op(v, vtmp, op);
  vshuff32x4(vtmp, v, v, 0xB1);  // 128/256-bit shuffle
  perform_op(v, vtmp, op);
  vshufps(vtmp, v, v, 0x4E);  // 64/128-bit shuffle
  perform_op(v, vtmp, op);
  vshufps(vtmp, v, v, 0xB1);  // 32/64-bit shuffle
  perform_op(v, vtmp, op);
}

void jit_softmax_t::perform_op(Zmm v, Zmm vtmp, op_t op) {
  if (op == op_t::max)
    vpmaxsd(v, v, vtmp);
  else if (op == op_t::sum)
    vaddps(v, v, vtmp);
}

void jit_softmax_t::assign_regs() {
  reg_param = rdi;
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

  tail_mask = Opmask(6);
  reg_map.insert(std::pair<reg_type, std::set<int>>(reg_type::mask, {tail_mask.getIdx()}));

  zmm_vec = Zmm(0);
  ymm_vec = ymm0;
  zmm_scale = Zmm(1);
  zmm_exp_neg_max = Zmm(2);
  ymm_exp_neg_max = ymm2;
  xmm_exp_neg_max = xmm2;
  zmm_exp_neg_max_fp32 = Zmm(4);
  zmm_tmp = Zmm(3);
  reg_map.insert(std::pair<reg_type, std::set<int>>(
      reg_type::zmm, {zmm_vec.getIdx(), zmm_scale.getIdx(), zmm_exp_neg_max.getIdx(), zmm_exp_neg_max_fp32.getIdx()}));
}
}  // namespace jd
