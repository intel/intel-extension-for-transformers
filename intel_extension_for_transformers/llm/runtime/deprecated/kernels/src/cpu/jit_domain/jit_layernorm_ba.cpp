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

#include "jit_layernorm_ba.hpp"
#include "kernels/layernorm_ba_types.hpp"
namespace jd {
void jit_layernorm_ba_t::generate() {
  this->preamble();
  switch (param_.spec_type) {
    case ssd::spec_translnorm_type::normal:
      normal_gen();
      break;
    case ssd::spec_translnorm_type::direct:
      direct_gen();
      break;
    default:
      break;
  }
  this->postamble();
  eltwise_injector.prepare_table();
}

void jit_layernorm_ba_t::direct_gen() {
  direct_load_params();
  // get unroll_degree
  unroll_degree = 10;
  int tail = param_.col_num % 16;
  auto align_col = param_.col_num - tail;
  while ((unroll_degree > 0) && ((align_col / (unroll_degree * 16) == 0) || (align_col % (unroll_degree * 16) != 0)))
    unroll_degree--;
  // reg alisa
  Reg64 reg_mean = reg_src_offset;
  Reg64 reg_var = reg_dst_offset;
  Reg64 reg_tmp = reg_batch, src_addr_cp = reg_batch;
  Reg64 dst_addr_cp = reg_affine_offset;

  // calculate norm loop nums
  int col_loop = 0;
  if (unroll_degree != 0) col_loop = param_.col_num / (unroll_degree * 16);
  // set mask
  if (tail > 0) {
    unsigned int mask = 0;
    for (int i = 0; i < tail; i++) mask = (mask << 1) + 1;
    mov(reg_tmp.cvt32(), mask);
    kmovd(tail_mask, reg_tmp.cvt32());
  }
  mov(reg_col, 0);
  if (col_loop > 0) {
    L(col_loop_start);
    direct_handel_var(unroll_degree, reg_var);
    mov(reg_row, 0);
    mov(src_addr_cp, src_addr);
    mov(dst_addr_cp, dst_addr);
    if (param_.split_output) mov(dst2_addr_cp, dst2_addr);
    escape_regs(unroll_degree * 3);
    L(norm_loop_start);
    direct_handel_norm(unroll_degree, src_addr_cp, dst_addr_cp, reg_mean);
    add(reg_row, 1);
    add(src_addr_cp, param_.col_num * get_data_size(param_.input_dt));
    add(dst_addr_cp, param_.col_num * get_data_size(param_.output_dt));
    if (param_.split_output) add(dst2_addr_cp, param_.col_num * get_data_size(param_.output2_dt));
    cmp(Xbyak::Reg32(reg_row.getIdx()), ptr[reg_param + LNBA_GET_OFF(process_row)]);
    jl(norm_loop_start, T_NEAR);

    // handle col_loop
    add(reg_col, 1);
    add(src_addr, unroll_degree * 16 * get_data_size(param_.input_dt));
    add(dst_addr, unroll_degree * 16 * get_data_size(param_.output_dt));
    if (param_.split_output) add(dst2_addr, unroll_degree * 16 * get_data_size(param_.output2_dt));
    add(reg_mean, unroll_degree * 16 * get_data_size(data_type::fp32));
    add(reg_var, unroll_degree * 16 * get_data_size(data_type::fp32));
    cmp(Xbyak::Reg32(reg_col.getIdx()), col_loop);
    jl(col_loop_start, T_NEAR);
  }
  if (tail > 0) {
    direct_handel_var(1, reg_var);
    mov(reg_row, 0);
    mov(src_addr_cp, src_addr);
    mov(dst_addr_cp, dst_addr);
    if (param_.split_output) mov(dst2_addr_cp, dst2_addr);
    escape_regs(3);
    L(tail_loop_start);
    direct_handel_norm(1, src_addr_cp, dst_addr_cp, reg_mean, true);
    add(reg_row, 1);
    add(src_addr_cp, param_.col_num * get_data_size(param_.input_dt));
    add(dst_addr_cp, param_.col_num * get_data_size(param_.output_dt));
    if (param_.split_output) add(dst2_addr_cp, param_.col_num * get_data_size(param_.output2_dt));
    cmp(Xbyak::Reg32(reg_row.getIdx()), ptr[reg_param + LNBA_GET_OFF(process_row)]);
    jl(tail_loop_start, T_NEAR);
  }
}

void jit_layernorm_ba_t::direct_handel_var(int degree, Reg64 reg_var) {
  for (int i = 0; i < degree; i++) {
    vmovups(Zmm(i), dword[reg_var + i * zmm_byte_size]);
    vaddps(Zmm(i), Zmm(i), ptr_b[reg_param + LNBA_GET_OFF(eps)]);
    vsqrtps(Zmm(i), Zmm(i));
  }
}

void jit_layernorm_ba_t::direct_handel_norm(int degree, Reg64 src_addr, Reg64 dst_addr, Reg64 reg_mean, bool tail) {
  vbroadcastss(Zmm(2 * degree), dword[reg_alpha + get_data_size(data_type::fp32) * reg_row]);
  for (int i = 0; i < degree; i++) vdivps(Zmm(i + degree), Zmm(degree * 2), Zmm(i));
  for (int i = 0; i < degree; i++) {
    vmovups(Zmm(i + degree * 2), dword[src_addr + i * zmm_byte_size]);
    // dt convert when input is int32
    if (param_.input_dt == data_type::s32) vcvtdq2ps(Zmm(i + degree * 2), Zmm(i + degree * 2));
    vsubps(Zmm(i + degree * 2), Zmm(i + degree * 2), dword[reg_mean + i * zmm_byte_size]);
    vfmadd213ps(Zmm(i + degree * 2), Zmm(i + degree), ptr_b[reg_beta + get_data_size(data_type::fp32) * reg_row]);
    if (param_.split_output) {
      vmovups(tail ? ptr[dst_addr + i * zmm_byte_size] | tail_mask : ptr[dst_addr + i * zmm_byte_size],
              Zmm(degree * 2 + i));
      eltwise_injector.vector_compute(Zmm(degree * 2 + i), param_.postop_attrs);
      switch (param_.output2_dt) {
        case data_type::s8:
          vpmovsdb(tail ? ptr[dst2_addr_cp + i * xmm_byte_size] | tail_mask : ptr[dst2_addr_cp + i * xmm_byte_size],
                   Zmm(degree * 2 + i));
          break;

        case data_type::u8:
          vpmovusdb(tail ? ptr[dst2_addr_cp + i * xmm_byte_size] | tail_mask : ptr[dst2_addr_cp + i * xmm_byte_size],
                    Zmm(degree * 2 + i));
          break;
        default:
          SPARSE_LOG(FATAL) << "only support u8/s8 in quantop";
          break;
      }
    } else {
      // apply postops.
      if (!param_.postop_attrs.empty()) eltwise_injector.vector_compute(Zmm(degree * 2 + i), param_.postop_attrs);
      // store data
      switch (param_.output_dt) {
        case data_type::fp32:
          vmovups(tail ? ptr[dst_addr + i * zmm_byte_size] | tail_mask : ptr[dst_addr + i * zmm_byte_size],
                  Zmm(degree * 2 + i));
          break;
        case data_type::s8:
          vpmovsdb(tail ? ptr[dst_addr + i * xmm_byte_size] | tail_mask : ptr[dst_addr + i * zmm_byte_size],
                   Zmm(degree * 2 + i));
          break;
        case data_type::u8:
          vpmovusdb(tail ? ptr[dst_addr + i * xmm_byte_size] | tail_mask : ptr[dst_addr + i * xmm_byte_size],
                    Zmm(degree * 2 + i));
          break;
        default:
          SPARSE_LOG(FATAL) << "unsupported output data type in direct translnorm.";
          break;
      }
    }
  }
}

void jit_layernorm_ba_t::normal_gen() {
  normal_gen_load_offset();
  normal_load_params();
  vbroadcastss(zmm_one, ptr[reg_param + LNBA_GET_OFF(one)]);
  mov(reg_batch, 0);
  L(batch_loop_start);
  mov(reg_col, 0);
  L(col_loop_start);
  vxorps(zmm_mean, zmm_mean, zmm_mean);
  vxorps(zmm_var, zmm_var, zmm_var);
  vxorps(zmm_mean_pow, zmm_mean_pow, zmm_mean_pow);
  vxorps(zmm_powx_mean, zmm_powx_mean, zmm_powx_mean);
  mov(reg_src_offset, reg_col);
  shl(reg_src_offset, log2(get_data_size(param_.input_dt)));
  mov(reg_dst_offset, reg_col);
  shl(reg_dst_offset, log2(get_data_size(param_.output_dt)));
  mov(reg_row, param_.row_num);
  L(mean_loop_start);
  // loop1:compute mean.
  // load unroll rows data.
  for (int k = 0; k < unroll_degree; k++) vmovups(Zmm(k), dword[src_addr + reg_src_offset + src_load_offset[k]]);
  // calculate the sum of x
  normal_binary_add(unroll_degree, zmm_mean);
  // calculate the sum of x^2 .
  for (int k = 0; k < unroll_degree; k++) {
    vmovups(Zmm(k), dword[src_addr + reg_src_offset + src_load_offset[k]]);
    vmulps(Zmm(k), Zmm(k), Zmm(k));
  }
  normal_binary_add(unroll_degree, zmm_powx_mean);
  add(reg_src_offset, unroll_degree * param_.col_num * get_data_size(param_.input_dt));
  add(reg_dst_offset, unroll_degree * param_.col_num * get_data_size(param_.output_dt));
  sub(reg_row, unroll_degree);
  cmp(reg_row, 0);
  jg(mean_loop_start, T_NEAR);

  // then calculate the mean of x & mean of x^2.
  vdivps(zmm_mean, zmm_mean, ptr_b[reg_param + LNBA_GET_OFF(n)]);
  vdivps(zmm_powx_mean, zmm_powx_mean, ptr_b[reg_param + LNBA_GET_OFF(n)]);
  vmulps(zmm_mean_pow, zmm_mean, zmm_mean);
  vsubps(zmm_var, zmm_powx_mean, zmm_mean_pow);
  sub(reg_src_offset, unroll_degree * param_.col_num * get_data_size(param_.input_dt));
  sub(reg_dst_offset, unroll_degree * param_.col_num * get_data_size(param_.output_dt));
  vaddps(zmm_var, zmm_var, ptr_b[reg_param + LNBA_GET_OFF(eps)]);
  vsqrtps(zmm_var, zmm_var);
  vdivps(zmm_var, zmm_one, zmm_var);

  // loop2:(x-mean)*var*α+β
  mov(reg_affine_offset, (param_.row_num - unroll_degree) * get_data_size(data_type::fp32));  // dt(affine)==fp32
  L(norm_loop_start);
  // positive sqeuence load rows data for cache performance.
  for (int k = 0; k < unroll_degree; k++) {
    vmovups(Zmm(k), dword[src_addr + reg_src_offset + src_load_offset[k]]);
    vsubps(Zmm(k), Zmm(k), zmm_mean);
    vmulps(Zmm(k), Zmm(k), zmm_var);
    vbroadcastss(
        zmm_alpha,
        dword[reg_alpha + reg_affine_offset + k * get_data_size(data_type::fp32)]);  // dt(aplha)==dt(beta)==fp32
    vbroadcastss(zmm_beta, dword[reg_beta + reg_affine_offset + k * get_data_size(data_type::fp32)]);
    vfmadd213ps(Zmm(k), zmm_alpha, zmm_beta);
  }
  escape_regs(unroll_degree);
  // store the value.
  for (int k = 0; k < unroll_degree; k++) {
    eltwise_injector.vector_compute(Zmm(k), param_.postop_attrs);
    RegExp binarop_offset;
    for (auto&& attr : param_.binaryop_attrs) {
      if (attr.op_alg == binaryop_alg::per_channel_quant || attr.op_alg == binaryop_alg::per_channel_dequant) {
        binary_injector.init_quantization(Zmm(23), r15);
        binarop_offset = reg_affine_offset + k * get_data_size(data_type::fp32);
      }
      binary_injector.compute_vector(Zmm(k), binarop_offset, attr);
    }
    if (param_.output_dt == data_type::fp32) {
      vmovups(ptr[dst_addr + reg_dst_offset + dst_load_offset[k]], Zmm(k));
    } else {
      if (param_.output_dt == data_type::u8) vpmovusdb(ptr[dst_addr + reg_dst_offset + dst_load_offset[k]], Zmm(k));
      if (param_.output_dt == data_type::s8) vpmovsdb(ptr[dst_addr + reg_dst_offset + dst_load_offset[k]], Zmm(k));
    }
  }

  sub(reg_affine_offset, unroll_degree * get_data_size(data_type::fp32));
  sub(reg_src_offset, unroll_degree * param_.col_num * get_data_size(param_.input_dt));
  sub(reg_dst_offset, unroll_degree * param_.col_num * get_data_size(param_.output_dt));
  add(reg_row, unroll_degree);
  cmp(reg_row, param_.row_num);
  jl(norm_loop_start);

  add(reg_col, zmm_byte_size / get_data_size(param_.input_dt));
  cmp(reg_col, param_.process_col);
  jl(col_loop_start, T_NEAR);

  add(reg_batch, 1);
  add(src_addr, param_.col_num * param_.row_num * get_data_size(param_.input_dt));
  add(dst_addr, param_.col_num * param_.row_num * get_data_size(param_.output_dt));
  cmp(reg_batch, param_.process_batch_per_ker);
  jl(batch_loop_start, T_NEAR);
}

// for pipline performance.
void jit_layernorm_ba_t::normal_binary_add(int degree, Zmm dst) {
  normal_reset_unroll_reg_idxs(degree);
  int first_idx = 0, second_idx = 0, begin_idx = 0;
  while (!normal_check_unroll_add_done()) {
    auto idx_pair = normal_get_unroll_add_idx(begin_idx);
    first_idx = idx_pair.first;
    second_idx = idx_pair.second;
    begin_idx = first_idx + 1;
    vaddps(Zmm(first_idx), Zmm(first_idx), Zmm(second_idx));
  }
  vaddps(dst, dst, Zmm(first_idx));
}

void jit_layernorm_ba_t::escape_regs(int degree) {
  for (auto&& i : reg_map) {
    for (auto&& j : i.second) eltwise_injector.escape_regs(i.first, j);
  }
  for (int i = 0; i < degree; i++) eltwise_injector.escape_regs(reg_type::zmm, i);
}

void jit_layernorm_ba_t::assign_regs() {
  zmm_mean_pow = Zmm(25);
  zmm_powx_mean = Zmm(26);
  zmm_alpha = Zmm(27);
  zmm_beta = Zmm(28);
  zmm_one = Zmm(29);
  zmm_mean = Zmm(30);
  zmm_var = Zmm(31);
  // when apply postop,all zmm can be free except zmm_one & zmm_eps.
  reg_map.insert(std::pair<reg_type, std::set<int>>(reg_type::zmm, {zmm_one.getIdx()}));

#ifdef _WIN32
  reg_param = rcx;
  reg_affine_offset = rdi;
#else
  reg_param = rdi;
  reg_affine_offset = rcx;
#endif
  reg_alpha = rax;
  reg_beta = rbx;
  reg_col = r8;
  reg_src_offset = r9;
  reg_row = r10;
  src_addr = r11;
  dst_addr = r12;
  reg_dst_offset = r13;
  reg_batch = r14;
  dst2_addr = rsi;  // Don't need to notify injector cause injector will skip these special usage registers(e.g.
                    // rsi,rdx) when reg allocating.
  dst2_addr_cp = rdx;
  reg_map.insert(std::pair<reg_type, std::set<int>>(
      reg_type::reg64,
      {reg_affine_offset.getIdx(), reg_row.getIdx(), reg_col.getIdx(), src_addr.getIdx(), reg_alpha.getIdx(),
       reg_src_offset.getIdx(), reg_dst_offset.getIdx(), reg_beta.getIdx(), dst_addr.getIdx(), reg_batch.getIdx()}));

  tail_mask = Opmask(1);
  reg_map.insert(std::pair<reg_type, std::set<int>>(reg_type::mask, {tail_mask.getIdx()}));
}

void jit_layernorm_ba_t::normal_gen_load_offset() {
  size_t src_offset = param_.thread_elt_offset * get_data_size(param_.input_dt);
  size_t dst_offset = param_.thread_elt_offset * get_data_size(param_.output_dt);
  for (int i = 0; i < unroll_degree; i++) {
    src_load_offset.insert(std::make_pair(i, src_offset));
    dst_load_offset.insert(std::make_pair(i, dst_offset));
    src_offset += param_.col_num * get_data_size(param_.input_dt);
    dst_offset += param_.col_num * get_data_size(param_.output_dt);
  }
}

void jit_layernorm_ba_t::normal_reset_unroll_reg_idxs(int degree) {
  unroll_reg_idxs.clear();
  for (int i = 0; i < degree; i++) unroll_reg_idxs.push_back(true);
}

std::pair<int, int> jit_layernorm_ba_t::normal_get_unroll_add_idx(int begin) {
  int first_idx = -1, second_idx = -1;
  int iter = begin;
  while (first_idx == -1 || second_idx == -1) {
    // TODO(zhe1wang): memory access violation
    if (unroll_reg_idxs[iter] == true && first_idx == -1) {
      first_idx = iter;
    } else if (unroll_reg_idxs[iter] == true && first_idx != -1 && second_idx == -1) {
      second_idx = iter;
      unroll_reg_idxs[iter] = false;
    }
    iter = (iter + 1) % unroll_reg_idxs.size();
  }
  return std::pair<int, int>(first_idx, second_idx);
}

bool jit_layernorm_ba_t::normal_check_unroll_add_done() {
  int count = 0;
  for (auto&& i : unroll_reg_idxs)
    if (i) count++;
  if (count == 1) return true;
  return false;
}

}  // namespace jd
