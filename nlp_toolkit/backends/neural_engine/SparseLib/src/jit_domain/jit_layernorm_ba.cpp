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
  prepare_mask();
  vbroadcastss(zmm_one, dword[one]);
  for (int col_idx = 0; col_idx < param_.process_col; col_idx += 16) {
    vxorps(reg_mean, reg_mean, reg_mean);
    vxorps(reg_var, reg_var, reg_var);
    int remain_rows = param_.row_num;
    int row_idx = 0;

    // loop1:compute mean.
    while (remain_rows > 0) {
      auto final_unroll_degree = remain_rows < unroll_degree ? remain_rows : unroll_degree;
      remain_rows -= final_unroll_degree;
      // load unroll rows data.
      for (int k = 0; k < final_unroll_degree; k++) vmovups(Zmm(k), dword[src_addr + get_offset(col_idx, row_idx++)]);
      // calculate the sum of mean.
      binary_add(final_unroll_degree, reg_mean);
    }
    // then calculate the mean.
    vmulps(reg_mean, reg_mean, dword[one_div_n]);

    // loop2:compute var.
    remain_rows = param_.row_num;
    while (remain_rows > 0) {
      auto final_unroll_degree = remain_rows < unroll_degree ? remain_rows : unroll_degree;
      remain_rows -= final_unroll_degree;
      // reverse sequence load rows data for cache performance.
      for (int k = 0; k < final_unroll_degree; k++) vmovups(Zmm(k), dword[src_addr + get_offset(col_idx, --row_idx)]);
      for (int k = 0; k < final_unroll_degree; k++) vsubps(Zmm(k), Zmm(k), reg_mean);  // sub mean
      for (int k = 0; k < final_unroll_degree; k++) vmulps(Zmm(k), Zmm(k), Zmm(k));    // pow(2)
      // calculate the sum of var of unroll rows data.
      binary_add(final_unroll_degree, reg_var);
    }
    // calculate 1/[sqrt(var+eps)].
    vmulps(reg_var, reg_var, dword[one_div_n]);
    vaddps(reg_var, reg_var, dword[eps]);
    vsqrtps(reg_var, reg_var);
    vdivps(reg_var, zmm_one, reg_var);

    // loop3:(x-mean)*var*α+β
    remain_rows = param_.row_num;
    while (remain_rows > 0) {
      auto final_unroll_degree = remain_rows < unroll_degree ? remain_rows : unroll_degree;
      remain_rows -= final_unroll_degree;
      int row_idx_affine = row_idx;
      int row_idx_store = row_idx;
      // positive sqeuence load rows data for cache performance.
      for (int k = 0; k < final_unroll_degree; k++) vmovups(Zmm(k), dword[src_addr + get_offset(col_idx, row_idx++)]);
      for (int k = 0; k < final_unroll_degree; k++) vsubps(Zmm(k), Zmm(k), reg_mean);  // sub mean
      for (int k = 0; k < final_unroll_degree; k++) vmulps(Zmm(k), Zmm(k), reg_var);   // mul 1/[sqrt(var+eps)]

      // prepare for the op-fusion.
      escape_regs(final_unroll_degree);

      // affine
      if (param_.affine) {
        for (int k = 0; k < final_unroll_degree; k++)
          eltwise_injector.vector_compute(Zmm(k), param_.postop_attrs, {row_idx_affine++});
      }

      // if contain postops,apply them.
      if (!param_.affine || param_.postop_attrs.size() > param_.row_num) {
        std::vector<int> postop_idxs;
        for (int i = param_.row_num; i < param_.postop_attrs.size(); i++) postop_idxs.push_back(i);
        for (int k = 0; k < final_unroll_degree; k++)
          eltwise_injector.vector_compute(Zmm(k), param_.postop_attrs, postop_idxs);
      }

      // store the value.
      for (int k = 0; k < final_unroll_degree; k++) {
        if (col_idx + 16 <= param_.process_col)
          vmovups(dword[dst_addr + get_offset(col_idx, row_idx_store++)], Zmm(k));
        else
          vmovups(dword[dst_addr + get_offset(col_idx, row_idx_store++)] | remain_task_mask, Zmm(k));
      }
    }
  }
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
    eltwise_injector.escape_erase(i.first);
    for (auto&& j : i.second) eltwise_injector.escape_erase(i.first, j);
  }
  for (int i = 0; i < degree; i++) eltwise_injector.escape_regs(reg_type::zmm, i);
}

void jit_layernorm_ba_t::assign_regs() {
  zmm_one = Zmm(29);
  reg_mean = Zmm(30);
  reg_var = Zmm(31);
  // when apply affine&postop,the reg_mean&reg_var can be free.
  reg_map.insert(std::pair<reg_type, std::set<int>>(reg_type::zmm, {zmm_one.getIdx()}));

  remain_task_mask = Opmask(6);
  reg_map.insert(std::pair<reg_type, std::set<int>>(reg_type::mask, {remain_task_mask.getIdx()}));

  reg_param = rdi;
  reg64_tmp = r10;
  src_addr = r11;
  dst_addr = r12;
  one_div_n = r13;
  one = r14;
  eps = r15;
  reg_map.insert(std::pair<reg_type, std::set<int>>(
      reg_type::reg64,
      {reg64_tmp.getIdx(), src_addr.getIdx(), dst_addr.getIdx(), one_div_n.getIdx(), one.getIdx(), eps.getIdx()}));
}

void jit_layernorm_ba_t::prepare_mask() {
  // even if dt is bf16, we will cvt it to fp32 first,then do next step.
  int tail_task = param_.process_col % 16;
  int mask = 0x0;
  for (int i = 0; i < tail_task; i++) mask = (mask << 1) + 1;

  mov(reg64_tmp.cvt32(), mask);
  kmovd(remain_task_mask, reg64_tmp.cvt32());
}

// In bytes.
size_t jit_layernorm_ba_t::get_offset(int col, int row) {
  size_t offset = row * param_.col_num * get_data_size(param_.dt);
  offset += param_.thread_offset;
  offset += col * get_data_size(param_.dt);
  return offset;
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

