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

#include "jit_spmm_avx512f.hpp"

namespace jd {
inline void jit_spmm_avx512f_t::load_params() {
  mov(reg_dense, ptr[reg_param + GET_OFF(dense)]);
  mov(reg_sparse, ptr[reg_param + GET_OFF(sparse)]);
  mov(reg_bias, ptr[reg_param + GET_OFF(bias)]);
  mov(reg_dst, ptr[reg_param + GET_OFF(dst)]);
}

void jit_spmm_avx512f_t::generate() {
#ifdef _WIN32
  const int nonvolatile_reg_size = 8 * 8;
#else
  const int nonvolatile_reg_size = 8 * 6;
#endif
  const auto& sparse_indptr = param_.sparse_ptr->indptr();
  const auto& sparse_indices = param_.sparse_ptr->indices();
  inLocalLabel();  // use local label for multiple instance
  // TODO(yi1ding): sub_function
  {
    sub(rsp, nonvolatile_reg_size);
    mov(ptr[rsp + 0x00], rbx);
    mov(ptr[rsp + 0x08], rbp);
    mov(ptr[rsp + 0x10], r12);
    mov(ptr[rsp + 0x18], r13);
    mov(ptr[rsp + 0x20], r14);
    mov(ptr[rsp + 0x28], r15);
#ifdef _WIN32
    mov(ptr[rsp + 0x30], rdi);
    mov(ptr[rsp + 0x38], rsi);
#endif

    load_params();
    mov(reg_dense_end, reg_dense);

    add(reg_dense, param_.im_start * F32_BYTES * param_.K);
    add(reg_dense_end, param_.im_end * F32_BYTES * param_.K);
    add(reg_dst, param_.im_start * F32_BYTES * param_.N);

    // Loop-m: asm loop
    Xbyak::Label L_m_loop;
    L(L_m_loop);
    // Loop-n: CPP loop fop each BLOCKED column
    int seq_data_tidx = 0;
    for (size_t j = 0; j < sparse_indptr.size() - 1; ++j) {
      // Load bias or clear registers
      if (param_.has_bias) {
        vmovups(dst_tile_Vmm(0), dword[reg_bias + j * ZMM_BYTES]);
        for (int ti = 1; ti < TH_; ++ti) {
          vmovaps(dst_tile_Vmm(ti), dst_tile_Vmm(0));
        }
      } else {
        for (int ti = 0; ti < TH_; ++ti) {
          vxorps(dst_tile_Vmm(ti), dst_tile_Vmm(ti), dst_tile_Vmm(ti));
        }
      }
      // Loop-k: CPP loop over K
      for (int k_idx = sparse_indptr[j]; k_idx < sparse_indptr[j + 1]; ++k_idx) {
        dim_t k = sparse_indices[k_idx];
        // Load dense
        for (int ti = 0; ti < TH_; ++ti) {
          vpbroadcastd(TH_Vmm(ti), dword[reg_dense + (param_.K * ti + k) * F32_BYTES]);
        }
        // Load sparse
        vmovups(TW_Vmm, dword[reg_sparse + (seq_data_tidx++) * ZMM_BYTES]);
        // MADD
        for (int ti = 0; ti < TH_; ++ti) {
          vfmadd231ps(dst_tile_Vmm(ti), TH_Vmm(ti), TW_Vmm);
        }
      }

      for (int ti = 0; ti < TH_; ++ti) {
        eltwise_injector.escape_regs(reg_type::zmm, dst_tile_Vmm(ti).getIdx());
      }
      eltwise_injector.escape_regs(reg_type::reg64, reg_bias.getIdx());
      eltwise_injector.escape_regs(reg_type::reg64, reg_dst.getIdx());
      eltwise_injector.escape_regs(reg_type::reg64, reg_n.getIdx());
      eltwise_injector.escape_regs(reg_type::reg64, reg_n_end.getIdx());
      // storeu
      for (int ti = 0; ti < TH_; ++ti) {
        eltwise_injector.vector_compute(dst_tile_Vmm(ti), param_.postop_attrs);
        vmovups(dword[reg_dst + (param_.N * ti) * F32_BYTES + j * ZMM_BYTES], dst_tile_Vmm(ti));
      }
    }

    add(reg_dense, TH_ * param_.K * F32_BYTES);  // TODO(yi1ding): handel edge cases where desc is not a multiple of TH_
    add(reg_dst, TH_ * param_.N * F32_BYTES);
    cmp(reg_dense, reg_dense_end);
    jl(L_m_loop, T_NEAR);  // End of loop-m: asm loop

    mov(rbx, ptr[rsp + 0x00]);
    mov(rbp, ptr[rsp + 0x08]);
    mov(r12, ptr[rsp + 0x10]);
    mov(r13, ptr[rsp + 0x18]);
    mov(r14, ptr[rsp + 0x20]);
    mov(r15, ptr[rsp + 0x28]);
#ifdef _WIN32
    mov(rdi, ptr[rsp + 0x30]);
    mov(rsi, ptr[rsp + 0x38]);
#endif
    add(rsp, nonvolatile_reg_size);
    ret();
  }
  outLocalLabel();  // end of local label
  eltwise_injector.prepare_table();
}
}  // namespace jd
