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

#include "jit_domain/jit_spmm_amx_bf16_x16.hpp"

#define TILE_M 16  // Number of rows in an A or C tile
#define TILE_K 32  // Number of columns in an A tile or rows in a B tile
#define TILE_N 16  // Number of columns in a B or C tile
#define KPACK 2    // Vertical K packing into dword
#define MZ 64      // (M / MT)
#define NUM_M 4    // (MZ / TILE_N)

#define GET_OFF(field) offsetof(ssd::amx_bf16f32_inputs_t, field)

namespace jd {
void jit_spmm_amx_bf16_x16_t::read_inputs() {
  mov(reg_weight, ptr[reg_param + GET_OFF(weight)]);
  mov(reg_src, ptr[reg_param + GET_OFF(src)]);
  mov(reg_bia, ptr[reg_param + GET_OFF(bias)]);
  mov(reg_dst, ptr[reg_param + GET_OFF(dst)]);
}

void jit_spmm_amx_bf16_x16_t::main_compute() {
  for (int b_row = 0; b_row < nrowptr - 1; ++b_row) {
    tilezero(tmm0);
    tilezero(tmm1);
    tilezero(tmm2);
    tilezero(tmm3);

    for (int group = group_rowptr[b_row]; group < group_rowptr[b_row + 1]; ++group) {
      dim_t* my_rows = colidxs + group * 32;

      tileloadd(tmm6, ptr[reg_weight + reg_stride + group * 512 * sizeof(src_t)]);

      for (int m = 0; m < 4 * TILE_M; m += TILE_M) {
        add(reg_m, reg_mstart);
        for (int k = 0; k < 32; k += 2) {
          vmovdqu(ymm0, ptr[reg_m + (m + my_rows[k] * tileM) * sizeof(src_t)]);
          vmovdqu(ymm1, ptr[reg_m + (m + my_rows[k + 1] * tileM) * sizeof(src_t)]);
          vinserti32x8(zmm0, zmm0, ymm1, 1);
          vpermw(zmm0, reg_mask, zmm0);
          vmovdqu32(qword[rsp + 0x40 + (m / TILE_M * 512 + k / 2 * 32) * 2], zmm0);
        }
        sub(reg_m, reg_mstart);
      }
      tileloadd(tmm4, ptr[rsp + reg_stride + (0x40)]);
      tdpbf16ps(tmm0, tmm6, tmm4);
      tileloadd(tmm4, ptr[rsp + reg_stride + (1024 + 0x40)]);
      tdpbf16ps(tmm1, tmm6, tmm4);
      tileloadd(tmm4, ptr[rsp + reg_stride + (2048 + 0x40)]);
      tdpbf16ps(tmm2, tmm6, tmm4);
      tileloadd(tmm4, ptr[rsp + reg_stride + (3072 + 0x40)]);
      tdpbf16ps(tmm3, tmm6, tmm4);
    }
    tilestored(ptr[rsp + reg_stride + (0x40)], tmm0);
    tilestored(ptr[rsp + reg_stride + (0x40 + 1024)], tmm1);
    tilestored(ptr[rsp + reg_stride + (0x40 + 2048)], tmm2);
    tilestored(ptr[rsp + reg_stride + (0x40 + 3072)], tmm3);
    mov(r12, b_row * 16 * tileM * size_of_dst_t);
    mov(reg_temp, reg_mstart);
    if (!bf16_out) {
      add(reg_temp, reg_temp);  // sizeof(dst_t) = sizeof(src_t) + sizeof(src_t)
    }
    add(r12, reg_temp);
    add(reg_dst, r12);
    for (int i = 0; i < TILE_N; ++i) {
      vpbroadcastd(zmm0, ptr[reg_bia + (b_row * TILE_N + i) * sizeof(dst_t)]);
      for (int j = 0; j < 4 * TILE_M; j += TILE_M) {
        vmovdqu32(zmm1, ptr[rsp + (0x40) + (j * TILE_N + i * TILE_M) * sizeof(dst_t)]);
        vaddps(zmm1, zmm1, zmm0);
        if (bf16_out) {
          vcvtneps2bf16(reg_bf16, zmm1);
          vmovdqu32(ptr[reg_dst + (j + i * tileM) * size_of_dst_t], reg_bf16);
        } else {
          vmovdqu32(ptr[reg_dst + (j + i * tileM) * size_of_dst_t], zmm1);
        }
      }
    }
    sub(reg_dst, r12);
  }
}

void jit_spmm_amx_bf16_x16_t::loop_M() {
  mov(reg_tileM, tileM * sizeof(src_t));
  L(lM);
  main_compute();
  add(reg_mstart, 4 * TILE_M * sizeof(src_t));
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
  mov(reg_stride, 64);
  vmovups(reg_mask, zword[reg_temp]);
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
}
}  // namespace jd
