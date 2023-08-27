//  Copyright (c) 2021 Intel Corporation
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

#include "jit_spmm_amx_bf16_x16.hpp"

#define GET_OFF_FP32(field) offsetof(ssd::amx_bf16f32_inputs_t, field)
#define GET_OFF_BF16(field) offsetof(ssd::amx_bf16bf16_inputs_t, field)

namespace jd {

void jit_spmm_amx_bf16_x16_t::read_inputs() {
  if (bf16_out) {
    mov(reg_weight, ptr[reg_param + GET_OFF_BF16(weight)]);
    mov(reg_src, ptr[reg_param + GET_OFF_BF16(src)]);
    mov(reg_bia, ptr[reg_param + GET_OFF_BF16(bias)]);
    mov(reg_dst, ptr[reg_param + GET_OFF_BF16(dst)]);
  } else {
    mov(reg_weight, ptr[reg_param + GET_OFF_FP32(weight)]);
    mov(reg_src, ptr[reg_param + GET_OFF_FP32(src)]);
    mov(reg_bia, ptr[reg_param + GET_OFF_FP32(bias)]);
    mov(reg_dst, ptr[reg_param + GET_OFF_FP32(dst)]);
  }
}

void jit_spmm_amx_bf16_x16_t::handle_postop_escape_vmms() {
  for (int i = 0; i < TILE_N; ++i) {
    eltwise_injector.escape_regs(reg_type::zmm, Xbyak::Zmm(i).getIdx());
    for (int j = 0; j < NUM_M * TILE_M; j += TILE_M) {
      eltwise_injector.escape_regs(reg_type::zmm, Xbyak::Zmm(j / TILE_M + TILE_N).getIdx());
    }
  }
  eltwise_injector.escape_regs(reg_type::zmm, reg_mask.getIdx());
  if (bf16_out) {
    eltwise_injector.escape_regs(reg_type::zmm, reg_bf16.getIdx());
  }
}

void jit_spmm_amx_bf16_x16_t::handle_postop_escape_regs() {
  eltwise_injector.escape_regs(reg_type::reg64, reg_weight.getIdx());
  eltwise_injector.escape_regs(reg_type::reg64, reg_bia.getIdx());
  eltwise_injector.escape_regs(reg_type::reg64, reg_m.getIdx());
  eltwise_injector.escape_regs(reg_type::reg64, reg_mstart.getIdx());
  eltwise_injector.escape_regs(reg_type::reg64, reg_tileM.getIdx());
  eltwise_injector.escape_regs(reg_type::reg64, reg_stride.getIdx());
}

void jit_spmm_amx_bf16_x16_t::postop_and_store_dst(int i, int j) {
  if (bf16_out) {
    eltwise_injector.vector_compute(Xbyak::Zmm(j / TILE_M + TILE_N), param_.postop_attrs);
    vcvtneps2bf16(reg_bf16, Xbyak::Zmm(j / TILE_M + TILE_N));
    vmovdqu16(ptr[reg_dst + (j + i * tileM) * size_of_dst_t], reg_bf16);
  } else {
    eltwise_injector.vector_compute(Xbyak::Zmm(j / TILE_M + TILE_N), param_.postop_attrs);
    vmovdqu32(ptr[reg_dst + (j + i * tileM) * size_of_dst_t], Xbyak::Zmm(j / TILE_M + TILE_N));
  }
}

void jit_spmm_amx_bf16_x16_t::handle_dst_offset(int b_row) {
  mov(r12, b_row * 16 * tileM * size_of_dst_t);
  mov(reg_temp, reg_mstart);
  if (!bf16_out) {
    add(reg_temp, reg_temp);  // sizeof(dst_t) = sizeof(src_t) + sizeof(src_t)
  }
  add(r12, reg_temp);
  add(reg_dst, r12);
}

void jit_spmm_amx_bf16_x16_t::main_compute() {
  for (int b_row = 0; b_row < nrowptr - 1; ++b_row) {
    if (group_rowptr[b_row] == group_rowptr[b_row + 1]) {  // the row is all zeros
      handle_dst_offset(b_row);
      for (int i = 0; i < TILE_N; ++i) {
        for (int j = 0; j < NUM_M * TILE_M; j += TILE_M) {
          vpbroadcastd(Xbyak::Zmm(j / TILE_M + TILE_N), ptr[reg_bia + (b_row * TILE_N + i) * sizeof(dst_t)]);
          postop_and_store_dst(i, j);
        }
      }
      sub(reg_dst, r12);
      continue;
    }

    tilezero(tmm0);
    tilezero(tmm1);
    tilezero(tmm2);
    tilezero(tmm3);

    for (int group = group_rowptr[b_row]; group < group_rowptr[b_row + 1]; ++group) {
      dim_t* my_rows = colidxs + group * TILE_K;

      tileloadd(tmm6, ptr[reg_weight + reg_stride + group * 512 * size_of_src_t]);

      add(reg_m, reg_mstart);
      for (int m = 0; m < NUM_M * TILE_M; m += TILE_M) {
        for (int k = 0; k < TILE_K; k += 2) {
          vmovdqu(Xbyak::Ymm(k % 16), ptr[reg_m + (m + my_rows[k] * tileM) * size_of_src_t]);
          vmovdqu(Xbyak::Ymm(k % 16 + 1), ptr[reg_m + (m + my_rows[k + 1] * tileM) * size_of_src_t]);
          vinserti32x8(Xbyak::Zmm(k % 16), Xbyak::Zmm(k % 16), Xbyak::Ymm(k % 16 + 1), 1);
          vpermw(Xbyak::Zmm(k % 16), reg_mask, Xbyak::Zmm(k % 16));
          vmovdqu32(qword[rsp + 0x40 + (m / TILE_M * TILE_N * TILE_K + k / 2 * TILE_K) * 2], Xbyak::Zmm(k % 16));
        }
      }
      sub(reg_m, reg_mstart);

      tileloaddt1(tmm4, ptr[rsp + reg_stride + (0x40)]);
      tdpbf16ps(tmm0, tmm6, tmm4);
      if (group == group_rowptr[b_row + 1] - 1) {
        tilestored(ptr[rsp + reg_stride + (0x40)], tmm0);
      }
      tileloaddt1(tmm4, ptr[rsp + reg_stride + (TILE_M * TILE_N * size_of_out_t + 0x40)]);
      tdpbf16ps(tmm1, tmm6, tmm4);
      if (group == group_rowptr[b_row + 1] - 1) {
        tilestored(ptr[rsp + reg_stride + (0x40 + TILE_M * TILE_N * size_of_out_t)], tmm1);
      }
      tileloaddt1(tmm4, ptr[rsp + reg_stride + (TILE_M * TILE_N * size_of_out_t * 2 + 0x40)]);
      tdpbf16ps(tmm2, tmm6, tmm4);
      if (group == group_rowptr[b_row + 1] - 1) {
        tilestored(ptr[rsp + reg_stride + (0x40 + TILE_M * TILE_N * size_of_out_t * 2)], tmm2);
      }
      tileloaddt1(tmm4, ptr[rsp + reg_stride + (TILE_M * TILE_N * size_of_out_t * 3 + 0x40)]);
      tdpbf16ps(tmm3, tmm6, tmm4);
      if (group == group_rowptr[b_row + 1] - 1) {
        tilestored(ptr[rsp + reg_stride + (0x40 + TILE_M * TILE_N * size_of_out_t * 3)], tmm3);
      }
    }
    handle_dst_offset(b_row);
    for (int i = 0; i < TILE_N; ++i) {
      vpbroadcastd(Xbyak::Zmm(i), ptr[reg_bia + (b_row * TILE_N + i) * size_of_out_t]);
      for (int j = 0; j < NUM_M * TILE_M; j += TILE_M) {
        vmovdqu32(Xbyak::Zmm(j / TILE_M + TILE_N), ptr[rsp + (0x40) + (j * TILE_N + i * TILE_M) * size_of_out_t]);
        vaddps(Xbyak::Zmm(j / TILE_M + TILE_N), Xbyak::Zmm(j / TILE_M + TILE_N), Xbyak::Zmm(i));
        postop_and_store_dst(i, j);
      }
    }
    sub(reg_dst, r12);
  }
}

void jit_spmm_amx_bf16_x16_t::loop_M() {
  mov(reg_tileM, tileM * size_of_src_t);
  L(lM);
  main_compute();
  add(reg_mstart, NUM_M * TILE_M * size_of_src_t);
  cmp(reg_mstart, reg_tileM);
  jl(lM);
}

void jit_spmm_amx_bf16_x16_t::init_param() {
  mov(reg_mstart, 0);
  mov(reg_temp, loopMask);
  mov(eax, reg_src);
  mov(reg_m, eax);
  mov(eax, 0xffff);
  kmovw(ktail_mask, eax);
  mov(reg_stride, TILE_K * size_of_src_t);
  vmovups(reg_mask, zword[reg_temp]);
  handle_postop_escape_vmms();
  handle_postop_escape_regs();
}

void jit_spmm_amx_bf16_x16_t::generate() {
  {
    sub(rsp, stack_space_needed_);

    mov(ptr[rsp + 0x00], rbx);
    mov(ptr[rsp + 0x08], rbp);
    mov(ptr[rsp + 0x10], r12);
    mov(ptr[rsp + 0x18], r13);
    mov(ptr[rsp + 0x20], r14);
    mov(ptr[rsp + 0x28], r15);

    read_inputs();
    init_param();
    loop_M();

    mov(rbx, ptr[rsp + 0x00]);
    mov(rbp, ptr[rsp + 0x08]);
    mov(r12, ptr[rsp + 0x10]);
    mov(r13, ptr[rsp + 0x18]);
    mov(r14, ptr[rsp + 0x20]);
    mov(r15, ptr[rsp + 0x28]);

    add(rsp, stack_space_needed_);

    ret();
  }
  align(64);
  L(loopMask);
  int num = 32;
  int wordlen = 2;
  const src_t mask[32] = {0, 16, 1, 17, 2,  18, 3,  19, 4,  20, 5,  21, 6,  22, 7,  23,
                          8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
  for (int i = 0; i < num; ++i) {
    db(mask[i], wordlen);
  }
  eltwise_injector.prepare_table();
}

}  // namespace jd
