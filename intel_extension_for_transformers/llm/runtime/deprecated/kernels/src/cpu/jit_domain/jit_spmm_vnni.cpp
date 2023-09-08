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

#include "jit_spmm_vnni.hpp"
#include <memory>

#define GET_OFF(field) offsetof(ssd::vnni_data_t<void>, field)

namespace jd {
const data_type& jit_spmm_vnni_t::output_type() { return param_.output_type; }

// {zmm28, zmm27, zmm26, zmm25, ...}
Xbyak::Zmm jit_spmm_vnni_t::TH_Vmm(int i) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS;
  const int& alloc_idx = alloc_start - i;
  return Xbyak::Zmm(alloc_idx);
}

// {zmm24, zmm23, zmm22, zmm21, ...}
Xbyak::Zmm jit_spmm_vnni_t::TW_Vmm(int i) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS - TH();
  const int& alloc_idx = alloc_start - i;
  return Xbyak::Zmm(alloc_idx);
}

// using TH VMM as dst temp register
Xbyak::Zmm jit_spmm_vnni_t::Temp_Vmm(int i) { return TH_Vmm(i); }

// {zmm0, zmm1, zmm2, zmm3, ...}
Xbyak::Zmm jit_spmm_vnni_t::dst_tile_Vmm(int i, int j) {
  const int& alloc_start = 0;
  const int& alloc_idx = alloc_start + i * TW() + j;
  return Xbyak::Zmm(alloc_idx);
}

void jit_spmm_vnni_t::load_bias(dim_t m_start) {
  for (int i = 0; i < TH(); ++i) {
    vpbroadcastd(dst_tile_Vmm(i, 0), ptr[reg_bias + (m_start + i) * BYTE4]);
    for (int j = 1; j < TW(); ++j) {
      vmovdqa32(dst_tile_Vmm(i, j), dst_tile_Vmm(i, 0));
    }
  }
}

void jit_spmm_vnni_t::clear_dst_tile() {
  for (int i = 0; i < TH(); ++i) {
    for (int j = 0; j < TW(); ++j) {
      vxorps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), dst_tile_Vmm(i, j));
    }
  }
}

void jit_spmm_vnni_t::load_intermediate_dst(dim_t m_start) {
  for (int i = 0; i < TH(); ++i) {
    for (int j = 0; j < TW(); ++j) {
      int sliced_dst_idx = (m_start + i) * ld_dst() + j * VEC;
      vmovdqu32(dst_tile_Vmm(i, j), ptr[reg_dst + reg_n_idx * BYTE4 + sliced_dst_idx * BYTE4]);
    }
  }
}

void jit_spmm_vnni_t::handle_dst_buffer_init(int kb_idx, dim_t m_start) {
  // Note that m_indices length is processed.
  if (kb_idx == 0) {
    if (param_.has_bias) {
      load_bias(m_start);
    } else {
      clear_dst_tile();
    }
  } else {
    load_intermediate_dst(m_start);
  }
}

void jit_spmm_vnni_t::tile_product(int tile_height, int tile_width) {
  for (int i = 0; i < tile_height; ++i) {
    for (int j = 0; j < tile_width; ++j) {
      vpdpbusd(dst_tile_Vmm(i, j), TW_Vmm(j), TH_Vmm(i));
    }
  }
}

void jit_spmm_vnni_t::load_dense(const std::vector<int64_t>& k_indices) {
  // e.g.: when tile_shape = {4, 4}, zmm24 = b[[0, 3, 4, 9], 0:16], zmm23 = b[[0, 3, 4, 9], 16:32],
  // ..., zmm21 = b[[0, 3, 4, 9], 48:64]. zmm24 is shared by each row in a TH, so the TH's blocked.
  for (int j = 0; j < TW(); ++j) {
    int vreg_dst_idx = TW_Vmm(j).getIdx();
    Xbyak::Xmm TW_xmm(vreg_dst_idx);
    Xbyak::Ymm TW_ymm = Xbyak::Ymm(vreg_dst_idx) | reg_k1;
    int vreg_temp_idx = vreg_temp.getIdx();
    Xbyak::Xmm temp_xmm(vreg_temp_idx);
    Xbyak::Ymm temp_ymm = Xbyak::Ymm(vreg_temp_idx) | reg_k1;
    SPARSE_LOG_IF(FATAL, !(k_indices.size() > 0 && k_indices.size() <= spns::ADJ))
        << "k_indices.size() > 0 && k_indices.size() <= spns::ADJ";
    if (k_indices.size() == spns::ADJ) {
      vmovdqu8(TW_xmm, ptr[reg_dense + reg_n_idx * BYTE1 + k_indices[0] * ld_dst() + j * VEC]);
      vbroadcasti32x4(TW_ymm, ptr[reg_dense + reg_n_idx * BYTE1 + k_indices[1] * ld_dst() + j * VEC]);
      vmovdqu8(temp_xmm, ptr[reg_dense + reg_n_idx * BYTE1 + k_indices[2] * ld_dst() + j * VEC]);
      vbroadcasti32x4(temp_ymm, ptr[reg_dense + reg_n_idx * BYTE1 + k_indices[3] * ld_dst() + j * VEC]);
    } else {
      vxorps(vreg_temp, vreg_temp, vreg_temp);
      vxorps(TW_Vmm(j), TW_Vmm(j), TW_Vmm(j));
      vmovdqu8(TW_xmm, ptr[reg_dense + reg_n_idx * BYTE1 + k_indices[0] * ld_dst() + j * VEC]);
      if (k_indices.size() > 1) {
        vbroadcasti32x4(TW_ymm, ptr[reg_dense + reg_n_idx * BYTE1 + k_indices[1] * ld_dst() + j * VEC]);
      }
      if (k_indices.size() > 2) {
        vmovdqu8(temp_xmm, ptr[reg_dense + reg_n_idx * BYTE1 + k_indices[2] * ld_dst() + j * VEC]);
      }
      if (k_indices.size() > 3) {
        vbroadcasti32x4(temp_ymm, ptr[reg_dense + reg_n_idx * BYTE1 + k_indices[3] * ld_dst() + j * VEC]);
      }
    }
    vpermt2d(TW_Vmm(j), vpermt2d_arg_idx, vreg_temp);
    vpshufb(TW_Vmm(j), TW_Vmm(j), vpshufb_arg_b);
  }
}

void jit_spmm_vnni_t::load_sparse(const Xbyak::Reg64& reg_addr, uint64_t offset) {
  for (int i = 0; i < TH(); ++i) {
    vpbroadcastd(TH_Vmm(i), ptr[reg_addr + offset + i * spns::ADJ * sizeof(decltype(*param_.weight))]);
  }
}

void jit_spmm_vnni_t::repeat_THx4xTW_matmal(dim_t m_start) {
  int need_regs = TH() + TW() + TH() * TW() + USED_VREGS;
  SPARSE_LOG_IF(FATAL, need_regs >= VREG_NUMS) << "loading weight's REGs (TH=" << TH()
                                               << "), loading "
                                                  "activation's REGs (TW="
                                               << TW() << "), dst tile's REGs (TH*TW=" << (TH() * TW())
                                               << "). "
                                                  "Their sum "
                                               << need_regs << " mustn't exceed 32zmm.";
  const dim_t imb = (param_.im_start + m_start) / TH();  // global index of m-block

  // ADJ=4 means 4 S8 combine a DST_S32. As ADJ repeats in K-dim, a DST_S32 also accumulates.
  // Note that a whole k-dim(segment) is processed.
  // Terminology:
  const dim_t indptr_kernel_start = param_.indptr[param_.im_start / TH()] * spns::ADJ;
  const dim_t indptr_lo = param_.indptr[imb] * spns::ADJ;      // min offset of index pointer
  const dim_t indptr_hi = param_.indptr[imb + 1] * spns::ADJ;  // max offset of index pointer
  const dim_t nnz = indptr_hi - indptr_lo;

  auto idx_begin = param_.indices.begin();
  const std::vector<dim_t> k_indices(idx_begin + indptr_lo, idx_begin + indptr_hi);

  switch (param_.sub_func) {
    case ssd::subfunc_level::non_kdims:
    case ssd::subfunc_level::kdims:
      mov(reg_seq_indices, reinterpret_cast<uint64_t>(dense_load_offsets.data() + indptr_lo - indptr_kernel_start));
      mov(reg_wei, reinterpret_cast<uint64_t>(param_.weight + param_.blocksize[0] * param_.blocksize[1] * indptr_lo));
      break;
    default:
      break;
  }
  switch (param_.sub_func) {
    case ssd::subfunc_level::none:
    case ssd::subfunc_level::non_kdims:
      // kp (k-idx pointer is the idx of nnz blocks of the current row)
      for (int64_t kp_lo = 0; kp_lo < nnz; kp_lo += spns::ADJ) {
        const int64_t kp_hi = std::min(kp_lo + spns::ADJ, nnz);  // end of k-index pointer (noninclusive)
        dim_t element_offset = param_.blocksize[0] * param_.blocksize[1] * indptr_lo + kp_lo * TH();

        // Step 1: load dense (activation). Note that k_indices length is processed.00-00
        // Step 2: load sparse (weight) and reorder data for that.
        // Step 3: tile product. Note that k_indices length is processed.
        // A tile product can calculate at least 1 row and 16 columns of DST.
        // Min tile calculation: Tile width/height is 1, compute (1, ADJ) x (ADJ, 16) = (1, 16) matmul.
        switch (param_.sub_func) {
          case ssd::subfunc_level::none:
            load_dense({k_indices.begin() + kp_lo, k_indices.begin() + kp_hi});
            load_sparse(reg_wei, element_offset * sizeof(decltype(*param_.weight)));
            tile_product(TH(), TW());
            break;
          case ssd::subfunc_level::non_kdims:
            call(func_load_and_prod_);
            break;
          default:
            break;
        }
      }
      break;
    case ssd::subfunc_level::kdims:
      if (nnz > 0) {  // at least one iteration
        xor_(reg_k_ptr, reg_k_ptr);
        add(reg_dense, reg_n_idx);  // reg_dense += reg_n_idx * BYTE1

        Xbyak::Label L_adj_k_loop;
        L(L_adj_k_loop);
        load_dense_sparse_prod();
        add(reg_k_ptr, spns::ADJ);
        cmp(reg_k_ptr, static_cast<int>(nnz));
        jl(L_adj_k_loop);  // Loop-N2 end.

        sub(reg_dense, reg_n_idx);  // reg_dense = reg_n_idx * BYTE1
        break;
      }
  }
}

void jit_spmm_vnni_t::handle_postop_escape_vmms() {
  eltwise_injector_.escape_regs(reg_type::zmm, vpermt2d_arg_idx.getIdx());
  eltwise_injector_.escape_regs(reg_type::zmm, vpshufb_arg_b.getIdx());
  eltwise_injector_.escape_regs(reg_type::zmm, vreg_temp.getIdx());
  for (int i = 0; i < TH(); ++i) {
    for (int j = 0; j < TW(); ++j) {
      eltwise_injector_.escape_regs(reg_type::zmm, dst_tile_Vmm(i, j).getIdx());
    }
  }
  if (param_.welford) {
    for (int j = 0; j < TW(); ++j) {
      eltwise_injector_.escape_regs(reg_type::zmm, Temp_Vmm(j).getIdx());
    }
  }
}

void jit_spmm_vnni_t::handle_postop_escape_regs() {
  eltwise_injector_.escape_regs(reg_type::reg64, param1.getIdx());
  eltwise_injector_.escape_regs(reg_type::reg64, reg_dst.getIdx());
  eltwise_injector_.escape_regs(reg_type::reg64, reg_scale.getIdx());
  eltwise_injector_.escape_regs(reg_type::reg64, reg_dst_idx.getIdx());
  eltwise_injector_.escape_regs(reg_type::reg64, reg_n_idx.getIdx());
  eltwise_injector_.escape_regs(reg_type::reg64, reg_seq_indices.getIdx());
  eltwise_injector_.escape_regs(reg_type::reg64, reg_m_idx.getIdx());
  eltwise_injector_.escape_regs(reg_type::reg64, reg_dst_idx.getIdx());
  eltwise_injector_.escape_regs(reg_type::mask, reg_k1.getIdx());
  if (param_.welford) {
    eltwise_injector_.escape_regs(reg_type::reg64, reg_dst_m1.getIdx());
    eltwise_injector_.escape_regs(reg_type::reg64, reg_dst_m2.getIdx());
  }
}

void jit_spmm_vnni_t::store_intermediate_dst(dim_t m_start) {
  SPARSE_LOG(FATAL) << "K-blocking is not implemented.";
  for (int i = 0; i < TH(); ++i) {
    for (int j = 0; j < TW(); ++j) {
      int sliced_dst_idx = (m_start + i) * ld_dst() + j * VEC;
      vmovdqu32(ptr[reg_dst + reg_n_idx * BYTE4 + sliced_dst_idx * BYTE4], dst_tile_Vmm(i, j));
    }
  }
}

void jit_spmm_vnni_t::handle_dst_buffer_epilogue_sub(bool set_zero) {
  mov(reg_dst, ptr[param1 + GET_OFF(ptr_dst)]);
  add(reg_dst, reg_dst_idx);
  std::shared_ptr<void> rec_vregs;
  if (param_.welford) {
    // restore vpshufb_arg_b and vpermt2d_arg_idx
    // which are used as reg_dst_m1 and reg_dst_m1 in line 266&267
    rec_vregs = {nullptr, [&](...) {
                   vbroadcasti32x4(vpshufb_arg_b, ptr[rip + L_vpshufb_arg]);
                   vpmovzxbd(vpermt2d_arg_idx, ptr[rip + L_vpermt2d_arg]);
                 }};
    vpbroadcastd(vreg_m_idx, reg_m_idx.cvt32());
    vcvtdq2ps(vreg_m_idx, vreg_m_idx);
    mov(reg_dst_m1, ptr[param1 + GET_OFF(ptr_dst_m1)]);
    mov(reg_dst_m2, ptr[param1 + GET_OFF(ptr_dst_m2)]);
  }

  for (int i = 0; i < TH(); ++i) {
    for (int j = 0; j < TW(); ++j) {
      vcvtdq2ps(dst_tile_Vmm(i, j) | T_rn_sae, dst_tile_Vmm(i, j));
      vmulps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), zword_b[reg_scale + reg_m_idx * BYTE4 + i * BYTE4]);  // *= scale.
      if (output_type() == data_type::fp32 && param_.append_sum) {
        vaddps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j),
               ptr[reg_dst + reg_n_idx * BYTE4 + i * ld_dst() * BYTE4 + j * VEC * BYTE4]);
      }
      if (param_.postop_attrs.size() != 0) {
        eltwise_injector_.vector_compute(dst_tile_Vmm(i, j), param_.postop_attrs);
      }

      // move out
      if (output_type() == data_type::u8) {
        if (param_.postop_attrs.size() == 0 || param_.postop_attrs.back().op_alg != postop_alg::quantize)
          vcvtps2udq(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j));
        vpmovusdb(ptr[reg_dst + reg_n_idx * BYTE1 + i * ld_dst() * BYTE1 + j * VEC * BYTE1], dst_tile_Vmm(i, j));
      } else if (output_type() == data_type::s8) {
        if (param_.postop_attrs.size() == 0 || param_.postop_attrs.back().op_alg != postop_alg::quantize)
          vcvtps2dq(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j));
        vpmovsdb(ptr[reg_dst + reg_n_idx * BYTE1 + i * ld_dst() * BYTE1 + j * VEC * BYTE1], dst_tile_Vmm(i, j));
      } else if (output_type() == data_type::fp32) {
        vmovups(ptr[reg_dst + reg_n_idx * BYTE4 + i * ld_dst() * BYTE4 + j * VEC * BYTE4], dst_tile_Vmm(i, j));
      }

      if (param_.welford) {
        calc_mean_variance(i, j, set_zero);
      }
    }
  }
}

void jit_spmm_vnni_t::calc_mean_variance(int i, int j, bool set_zero) {
  /**
   * mean_{n}= mean_{n-1} + (xn - mean_{n-1}) / n
   * m2_{n} = m2_{n-1} + (xn - mean_{n-1})(xn - mean_{n})
   *
   * mean_{n}= mean_{n-1} + A/n
   * m2_{n} = m2_{n-1} + A (A - A/n) = m2_{n-1} + (A*A - A*A/n)
   *
   * means are accumulate in Temp_Vmm(j) / TW_Vmm(j)
   * M2 are accumulate in dst_tile_Vmm(0, j)
   */
  const Xbyak::Address& zword_dst_m1 = ptr[reg_dst_m1 + reg_n_idx * BYTE4 + j * VEC * BYTE4];
  const Xbyak::Address& zword_dst_m2 = ptr[reg_dst_m2 + reg_n_idx * BYTE4 + j * VEC * BYTE4];
  if (j == 0) vaddps(vreg_m_idx, vreg_m_idx, zword_b[rip + L_m512_1f]);  // ({m+1}_idx_, n) := m_idx + 1
  if (j == 0) vrcp14ps(vreg_temp2, vreg_m_idx);                          // get 1/n
  if (i == 0) {
    if (set_zero == true) {
      vxorps(Temp_Vmm(j), Temp_Vmm(j), Temp_Vmm(j));  // move in M0(mean)
    } else {
      vmovups(Temp_Vmm(j), zword_dst_m1);  // load from memory
    }
  }

  vsubps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), Temp_Vmm(j));         // A:= x - mean;
  vfmadd231ps(Temp_Vmm(j), dst_tile_Vmm(i, j), vreg_temp2);            // mean += A/n
  vmulps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), dst_tile_Vmm(i, j));  // A2 = A * A
  vfnmadd231ps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), vreg_temp2);    // A2 - A2 / N

  if (i == 0) {
    if (set_zero == true) {
      vxorps(dst_tile_Vmm(0, j), dst_tile_Vmm(0, j), dst_tile_Vmm(0, j));  // update m2
    } else {
      vaddps(dst_tile_Vmm(0, j), dst_tile_Vmm(i, j), zword_dst_m2);  // update m2
    }
  } else {
    vaddps(dst_tile_Vmm(0, j), dst_tile_Vmm(i, j), dst_tile_Vmm(0, j));  // update m2
  }

  if (i == TH() - 1) {  // move out
    vmovups(zword_dst_m1, Temp_Vmm(j));
    vmovups(zword_dst_m2, dst_tile_Vmm(0, j));
  }
}

void jit_spmm_vnni_t::gen_subfunc_dst_epilogue() {
  L(func_dst_epilogue_start_);
  {
    Xbyak::util::StackFrame callee1_sf(this, 0);
    handle_dst_buffer_epilogue_sub(true);
  }
  L(func_dst_epilogue_);
  {
    Xbyak::util::StackFrame callee1_sf(this, 0);
    handle_dst_buffer_epilogue_sub(false);
  }
}

void jit_spmm_vnni_t::read_params() {
  mov(reg_dense, ptr[param1 + GET_OFF(ptr_dense)]);
  mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
  mov(reg_scale, ptr[param1 + GET_OFF(ptr_scales)]);
}

/**
 * Required registers:
 *  reg_dense - the start of the current row of dense matrix
 *  reg_seq_indices - the start of offset for each TW, it will be updated after read
 *  reg_wei - the start of weight matrix, it will be updated after read
 */
void jit_spmm_vnni_t::load_dense_sparse_prod() {
  constexpr size_t idx_size = sizeof(decltype(param_.indices)::value_type);
  mov(reg_addr_tmp[0], qword[reg_seq_indices + 0 * idx_size]);
  mov(reg_addr_tmp[1], qword[reg_seq_indices + 1 * idx_size]);
  mov(reg_addr_tmp[2], qword[reg_seq_indices + 2 * idx_size]);
  mov(reg_addr_tmp[3], qword[reg_seq_indices + 3 * idx_size]);
  add(reg_seq_indices, 4 * idx_size);

  constexpr size_t wei_size = sizeof(decltype(*param_.weight));

  for (int j = 0; j < TW(); ++j) {
    int vreg_dst_idx = TW_Vmm(j).getIdx();
    Xbyak::Xmm TW_xmm(vreg_dst_idx);
    Xbyak::Ymm TW_ymm = Xbyak::Ymm(vreg_dst_idx) | reg_k1;
    int vreg_temp_idx = vreg_temp.getIdx();
    Xbyak::Xmm temp_xmm(vreg_temp_idx);
    Xbyak::Ymm temp_ymm = Xbyak::Ymm(vreg_temp_idx) | reg_k1;
    // assert(k_indices.size() > 0 && k_indices.size() <= spns::ADJ);

    vmovdqu8(TW_xmm, ptr[reg_dense + reg_addr_tmp[0] + j * VEC]);
    vbroadcasti32x4(TW_ymm, ptr[reg_dense + reg_addr_tmp[1] + j * VEC]);
    vmovdqu8(temp_xmm, ptr[reg_dense + reg_addr_tmp[2] + j * VEC]);
    vbroadcasti32x4(temp_ymm, ptr[reg_dense + reg_addr_tmp[3] + j * VEC]);

    vpermt2d(TW_Vmm(j), vpermt2d_arg_idx, vreg_temp);
    vpshufb(TW_Vmm(j), TW_Vmm(j), vpshufb_arg_b);

    for (int i = 0; i < TH(); ++i) {
      // load sparse
      if (j == 0) vpbroadcastd(TH_Vmm(i), ptr[reg_wei + i * spns::ADJ * wei_size]);

      // tile prod
      vpdpbusd(dst_tile_Vmm(i, j), TW_Vmm(j), TH_Vmm(i));
    }
    // update reg_wei in the middle
    if (j == TW() / 2) add(reg_wei, TH() * spns::ADJ * wei_size);
  }
}

void jit_spmm_vnni_t::gen_subfunc_load_and_prod() {
  L(func_load_and_prod_);
  add(reg_dense, reg_n_idx);  // reg_dense += reg_n_idx * BYTE1

  load_dense_sparse_prod();

  sub(reg_dense, reg_n_idx);  // reg_dense = reg_n_idx * BYTE1
  ret();
}

void jit_spmm_vnni_t::generate() {
#ifdef _WIN32
  const int nonvolatile_reg_size = 8 * 8;
#else
  const int nonvolatile_reg_size = 8 * 6;
#endif
  handle_postop_escape_vmms();
  handle_postop_escape_regs();
  inLocalLabel();  // use local label for multiple instance
  gen_subfunc_dst_epilogue();
  switch (param_.sub_func) {
    case ssd::subfunc_level::none:
    case ssd::subfunc_level::kdims:
      break;
    case ssd::subfunc_level::non_kdims:
      gen_subfunc_load_and_prod();
      break;
    default:
      SPARSE_LOG(FATAL) << "Unexpected subfunc_level: " << static_cast<uint8_t>(param_.sub_func);
      break;
  }
  callee_functions_code_size_ = getSize();

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
    read_params();

    mov(reg_wei, reinterpret_cast<uint64_t>(param_.weight));

    // initialize the control reg which we are going to use for permutes and shuffles.
    vpmovzxbd(vpermt2d_arg_idx, ptr[rip + L_vpermt2d_arg]);
    vbroadcasti32x4(vpshufb_arg_b, ptr[rip + L_vpshufb_arg]);
    auto temp_r32 = Xbyak::Reg32(reg_tmp.getIdx());
    mov(temp_r32, 0xf0);
    kmovb(reg_k1, temp_r32);

    xor_(reg_dst_idx, reg_dst_idx);
    // When K-dim is cut into k_blocks parts, it'll produce same number of DST intermediate results.
    // Loop-M2: CPP loop for each blocked row. Asm code unroll.
    for (dim_t im = 0; im < param_.BM; im += mt_size()) {
      xor_(reg_n_idx, reg_n_idx);  // reg_n_idx = 0
      mov(reg_m_idx, im);          // reg_m_idx = im

      // n_blocks and n_tiles are N-dim loops, and can be used for N-dim multithread.
      // n_blocks is asm outer loop, with longer assembly context span. n_tiles is inner loop.
      // Loop-N2: Assembly loop at "n_tiles". Asm code fold.
      Xbyak::Label L_nt_loop;
      L(L_nt_loop);
      // init dst buffer, like init the value to bias or the previous intermediate result.
      handle_dst_buffer_init(0, im);
      repeat_THx4xTW_matmal(im);
      // generate the epilogue logic. This is different depending on B_blocks value (should we
      // cache intermediate results or write results with post-op to output)
      if (im == 0) {
        call(func_dst_epilogue_start_);  // zeroing registers for the first iteration
      } else {
        call(func_dst_epilogue_);  // load from memory for the rest iterations
      }

      add(reg_n_idx, nt_size());
      cmp(reg_n_idx, param_.BN);
      jl(L_nt_loop, T_NEAR);  // Loop-N2 end.
      add(reg_dst_idx, mt_size() * get_data_size(output_type()) * ld_dst());
    }  // Loop-M2 end.

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
  int word_size = 1;
  int num_size = 16;
  const uint8_t vpermt2d_control[16] = {0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23};
  const uint8_t vpshufb_control[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
  L(L_vpermt2d_arg);
  for (int i = 0; i < num_size; ++i) {
    db(vpermt2d_control[i], word_size);
  }
  L(L_vpshufb_arg);
  for (int i = 0; i < num_size; ++i) {
    db(vpshufb_control[i], word_size);
  }
  L(L_m512_1f);
  for (int i = 0; i < num_size; ++i) {
    db(bit_cast<uint32_t, float>(1.f), sizeof(float));
  }
  outLocalLabel();  // end of local label
  eltwise_injector_.prepare_table();
}
}  // namespace jd
