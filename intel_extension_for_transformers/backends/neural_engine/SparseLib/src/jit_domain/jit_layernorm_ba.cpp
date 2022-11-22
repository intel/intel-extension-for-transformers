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

#include "jit_domain/jit_layernorm_ba.hpp"
#include "kernels/layernorm_ba_types.hpp"
namespace jd {
void jit_layernorm_ba_t::generate() {
  this->preamble();
  load_params();
  vbroadcastss(zmm_one, ptr[reg_param + LNBA_GET_OFF(one)]);
  vbroadcastss(zmm_eps, ptr[reg_param + LNBA_GET_OFF(eps)]);
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
  binary_add(unroll_degree, zmm_mean);
  // calculate the sum of x^2 .
  for (int k = 0; k < unroll_degree; k++) {
    vmovups(Zmm(k), dword[src_addr + reg_src_offset + src_load_offset[k]]);
    vmulps(Zmm(k), Zmm(k), Zmm(k));
  }
  binary_add(unroll_degree, zmm_powx_mean);
  add(reg_src_offset, unroll_degree * param_.col_num * get_data_size(param_.input_dt));
  add(reg_dst_offset, unroll_degree * param_.col_num * get_data_size(param_.output_dt));
  sub(reg_row, unroll_degree);
  cmp(reg_row, 0);
  jg(mean_loop_start, T_NEAR);

  // then calculate the mean of x & mean of x^2.
  vmulps(zmm_mean, zmm_mean, dword[one_div_n]);
  vmulps(zmm_powx_mean, zmm_powx_mean, dword[one_div_n]);
  vmulps(zmm_mean_pow, zmm_mean, zmm_mean);
  vsubps(zmm_var, zmm_powx_mean, zmm_mean_pow);
  sub(reg_src_offset, unroll_degree * param_.col_num * get_data_size(param_.input_dt));
  sub(reg_dst_offset, unroll_degree * param_.col_num * get_data_size(param_.output_dt));
  vaddps(zmm_var, zmm_var, zmm_eps);
  vsqrtps(zmm_var, zmm_var);
  vdivps(zmm_var, zmm_one, zmm_var);

  // loop2:(x-mean)*var*α+β
  mov(reg_affine_offset, (param_.row_num - unroll_degree) * get_data_size(param_.affine_dt));
  L(norm_loop_start);
  // positive sqeuence load rows data for cache performance.
  for (int k = 0; k < unroll_degree; k++) {
    vmovups(Zmm(k), dword[src_addr + reg_src_offset + src_load_offset[k]]);
    vsubps(Zmm(k), Zmm(k), zmm_mean);
    vmulps(Zmm(k), Zmm(k), zmm_var);
    vbroadcastss(zmm_alpha, dword[reg_alpha + reg_affine_offset + k * get_data_size(param_.affine_dt)]);
    vbroadcastss(zmm_beta, dword[reg_beta + reg_affine_offset + k * get_data_size(param_.affine_dt)]);
    vfmadd213ps(Zmm(k), zmm_alpha, zmm_beta);
  }
  sub(reg_affine_offset, unroll_degree * get_data_size(param_.affine_dt));
  escape_regs(unroll_degree);
  // store the value.
  for (int k = 0; k < unroll_degree; k++) {
    eltwise_injector.vector_compute(Zmm(k), param_.postop_attrs);
    if (param_.output_dt == data_type::fp32) {
      vmovups(ptr[dst_addr + reg_dst_offset + dst_load_offset[k]], Zmm(k));
    } else {
      if (param_.output_dt == data_type::u8) vpmovusdb(ptr[dst_addr + reg_dst_offset + dst_load_offset[k]], Zmm(k));
      if (param_.output_dt == data_type::s8) vpmovsdb(ptr[dst_addr + reg_dst_offset + dst_load_offset[k]], Zmm(k));
    }
  }

  sub(reg_src_offset, unroll_degree * param_.col_num * get_data_size(param_.input_dt));
  sub(reg_dst_offset, unroll_degree * param_.col_num * get_data_size(param_.output_dt));
  add(reg_row, unroll_degree);
  cmp(reg_row, param_.row_num);
  jl(norm_loop_start);

  add(reg_col, zmm_byte_size / get_data_size(param_.input_dt));
  cmp(reg_col, param_.process_col);
  jl(col_loop_start, T_NEAR);

  this->postamble();
  eltwise_injector.prepare_table();
}

// for pipline performance.
void jit_layernorm_ba_t::binary_add(int degree, Zmm dst) {
  reset_unroll_reg_idxs(degree);
  int first_idx = 0, second_idx = 0, begin_idx = 0;
  while (!check_unroll_add_done()) {
    auto idx_pair = get_unroll_add_idx(begin_idx);
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
  zmm_eps = Zmm(24);
  zmm_mean_pow = Zmm(25);
  zmm_powx_mean = Zmm(26);
  zmm_alpha = Zmm(27);
  zmm_beta = Zmm(28);
  zmm_one = Zmm(29);
  zmm_mean = Zmm(30);
  zmm_var = Zmm(31);
  // when apply postop,all zmm can be free except zmm_one & zmm_eps.
  reg_map.insert(std::pair<reg_type, std::set<int>>(reg_type::zmm, {zmm_one.getIdx(), zmm_eps.getIdx()}));

  reg_param = rdi;
  reg_alpha = rax;
  reg_beta = rbx;
  reg_affine_offset = rcx;
  reg_col = r8;
  reg_src_offset = r9;
  reg_dst_offset = r14;
  reg_row = r10;
  src_addr = r11;
  dst_addr = r12;
  one_div_n = r13;
  reg_map.insert(std::pair<reg_type, std::set<int>>(
      reg_type::reg64,
      {reg_affine_offset.getIdx(), reg_row.getIdx(), reg_col.getIdx(), src_addr.getIdx(), reg_alpha.getIdx(),
       reg_src_offset.getIdx(), reg_dst_offset.getIdx(), reg_beta.getIdx(), dst_addr.getIdx(), one_div_n.getIdx()}));
}

void jit_layernorm_ba_t::gen_load_offset() {
  size_t src_offset = param_.thread_elt_offset * get_data_size(param_.input_dt);
  size_t dst_offset = param_.thread_elt_offset * get_data_size(param_.output_dt);
  for (int i = 0; i < unroll_degree; i++) {
    src_load_offset.insert(std::make_pair(i, src_offset));
    dst_load_offset.insert(std::make_pair(i, dst_offset));
    src_offset += param_.col_num * get_data_size(param_.input_dt);
    dst_offset += param_.col_num * get_data_size(param_.output_dt);
  }
}

void jit_layernorm_ba_t::reset_unroll_reg_idxs(int degree) {
  unroll_reg_idxs.clear();
  for (int i = 0; i < degree; i++) unroll_reg_idxs.push_back(true);
}

std::pair<int, int> jit_layernorm_ba_t::get_unroll_add_idx(int begin) {
  int first_idx = -1, second_idx = -1;
  int iter = begin;
  while (first_idx == -1 || second_idx == -1) {
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

bool jit_layernorm_ba_t::check_unroll_add_done() {
  int count = 0;
  for (auto&& i : unroll_reg_idxs)
    if (i) count++;
  if (count == 1) return true;
  return false;
}

}  // namespace jd
