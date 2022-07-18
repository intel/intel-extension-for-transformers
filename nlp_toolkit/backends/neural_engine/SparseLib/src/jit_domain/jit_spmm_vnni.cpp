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

#include "jit_domain/jit_spmm_vnni.hpp"

#define GET_OFF(field) offsetof(ssd::flat_data_t, field)

namespace jd {
jit_spmm_vnni_t::jit_spmm_vnni_t(const ssd::flat_param_t& param)
    : jit_generator(), param_(param), bsr_(param_.sparse_ptr) {
  n_blocks_ = param.mkn_blocks[2];
  nb_size_ = ceil_div(param.N, n_blocks_);

  TW_ = param.tile_shape[1];
  nt_size_ = TW_ * VEC;
  n_tiles_ = ceil_div(nb_size_, nt_size_);

  TH_ = param.tile_shape[0];
  mt_size_ = TH_ * 1;
  m_tiles_ = ceil_div(param.M, mt_size_);

  k_blocks_ = 1;
  kb_size_ = ceil_div(param.K, k_blocks_);
  dst_stride_ = {param.N, 1};
  output_type_ = param.output_type;
}

// {zmm28, zmm27, zmm26, zmm25, ...}
Xbyak::Zmm jit_spmm_vnni_t::TH_Vmm(int i) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS;
  const int& alloc_idx = alloc_start - i;
  return Xbyak::Zmm(alloc_idx);
}

// {zmm24, zmm23, zmm22, zmm21, ...}
Xbyak::Zmm jit_spmm_vnni_t::TW_Vmm(int i) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS - TH_;
  const int& alloc_idx = alloc_start - i;
  return Xbyak::Zmm(alloc_idx);
}

// {zmm0, zmm1, zmm2, zmm3, ...}
Xbyak::Zmm jit_spmm_vnni_t::dst_tile_Vmm(int i, int j) {
  const int& alloc_start = 0;
  const int& alloc_idx = alloc_start + i * TW_ + j;
  return Xbyak::Zmm(alloc_idx);
}

void jit_spmm_vnni_t::load_bias(int64_t m_start) {
  for (int i = 0; i < TH_; ++i) {
    vpbroadcastd(dst_tile_Vmm(i, 0), ptr[reg_bias + (m_start + i) * BYTE4]);
    for (int j = 1; j < TW_; ++j) {
      vmovdqa32(dst_tile_Vmm(i, j), dst_tile_Vmm(i, 0));
    }
  }
}

void jit_spmm_vnni_t::clear_dst_tile() {
  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      vxorps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), dst_tile_Vmm(i, j));
    }
  }
}

void jit_spmm_vnni_t::load_intermediate_dst(int64_t m_start) {
  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      int sliced_dst_idx = (m_start + i) * dst_stride_[0] + j * VEC;
      vmovdqu32(dst_tile_Vmm(i, j), ptr[reg_dst + reg_nt_absolute_idx * BYTE4 + sliced_dst_idx * BYTE4]);
    }
  }
}

void jit_spmm_vnni_t::handle_dst_buffer_init(int kb_idx, int64_t m_start) {
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
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < tile_width; ++j) {
      vpdpbusd(dst_tile_Vmm(i, j), TW_Vmm(j), TH_Vmm(i));
    }
  }
}

void jit_spmm_vnni_t::load_dense(const std::vector<int64_t>& k_indices) {
  // e.g.: when tile_shape = {4, 4}, zmm24 = b[[0, 3, 4, 9], 0:16], zmm23 = b[[0, 3, 4, 9], 16:32],
  // ..., zmm21 = b[[0, 3, 4, 9], 48:64]. zmm24 is shared by each row in a TH, so the TH's blocked.
  for (int j = 0; j < TW_; ++j) {
    int vreg_idx = TW_Vmm(j).getIdx();
    Xbyak::Xmm TW_xmm(vreg_idx);
    Xbyak::Ymm TW_ymm = Xbyak::Ymm(vreg_idx) | reg_k1;
    int vreg_temp_idx = vreg_temp.getIdx();
    Xbyak::Xmm temp_xmm(vreg_temp_idx);
    Xbyak::Ymm temp_ymm = Xbyak::Ymm(vreg_temp_idx) | reg_k1;
    assert(k_indices.size() > 0 && k_indices.size() <= spns::ADJ);
    if (k_indices.size() == spns::ADJ) {
      vmovdqu8(TW_xmm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + k_indices[0] * dst_stride_[0] + j * VEC]);
      vbroadcasti32x4(TW_ymm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + k_indices[1] * dst_stride_[0] + j * VEC]);
      vmovdqu8(temp_xmm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + k_indices[2] * dst_stride_[0] + j * VEC]);
      vbroadcasti32x4(temp_ymm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + k_indices[3] * dst_stride_[0] + j * VEC]);
    } else {
      vxorps(vreg_temp, vreg_temp, vreg_temp);
      vxorps(TW_Vmm(j), TW_Vmm(j), TW_Vmm(j));
      vmovdqu8(TW_xmm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + k_indices[0] * dst_stride_[0] + j * VEC]);
      if (k_indices.size() > 1) {
        vbroadcasti32x4(TW_ymm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + k_indices[1] * dst_stride_[0] + j * VEC]);
      }
      if (k_indices.size() > 2) {
        vmovdqu8(temp_xmm, ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + k_indices[2] * dst_stride_[0] + j * VEC]);
      }
      if (k_indices.size() > 3) {
        vbroadcasti32x4(temp_ymm,
                        ptr[reg_dense + reg_nt_absolute_idx * BYTE1 + k_indices[3] * dst_stride_[0] + j * VEC]);
      }
    }
    vpermt2d(TW_Vmm(j), vpermt2d_arg_idx, vreg_temp);
    vpshufb(TW_Vmm(j), TW_Vmm(j), vpshufb_arg_b);
  }
}

void jit_spmm_vnni_t::load_sparse(const int8_t* bsr_data, int64_t kp_lo, int64_t kp_hi) {
  for (int i = 0; i < TH_; ++i) {
    for (int kp = kp_lo; kp < kp_lo + spns::ADJ; ++kp) {
      if (kp < kp_hi)
        seq_vals_.push_back(*(bsr_data + kp * TH_ + i));
      else
        seq_vals_.push_back(0);
    }
    vpbroadcastd(TH_Vmm(i), ptr[reg_seq_vals + (seq_pos++) * spns::ADJ * BYTE1]);
  }
}

void jit_spmm_vnni_t::repeat_THx4xTW_matmal(int64_t m_start) {
  int need_regs = TH_ + TW_ + TH_ * TW_ + USED_VREGS;
  LOG_IF(FATAL, need_regs >= VREG_NUMS) << "loading weight's REGs (TH=" << TH_
                                        << "), loading "
                                           "activation's REGs (TW="
                                        << TW_ << "), dst tile's REGs (TH*TW=" << (TH_ * TW_)
                                        << "). "
                                           "Their sum "
                                        << need_regs << " mustn't exceed 32zmm.";
  const int64_t imb = m_start / TH_;  // index of m-block
  // ADJ=4 means 4 S8 combine a DST_S32. As ADJ repeats in K-dim, a DST_S32 also accumulates.
  // Note that a whole k-dim(segment) is processed.
  // Terminology:
  const int64_t nnz = bsr_->getnnz(imb);
  auto idx_begin = bsr_->indices().begin();
  const std::vector<int64_t> k_indices(idx_begin + bsr_->indptr()[imb], idx_begin + bsr_->indptr()[imb + 1]);
  const int8_t* bsr_data = bsr_->data().data() + spns::ADJ * bsr_->indptr()[imb];
  // kp (k-idx pointer is the idx of nnz blocks of the current row)
  for (int64_t kp_lo = 0; kp_lo < nnz; kp_lo += spns::ADJ) {
    const int64_t kp_hi = std::min(kp_lo + spns::ADJ, nnz);  // end of k-index pointer (noninclusive)
    // Step 1: load dense (activation). Note that k_indices length is processed.
    load_dense({k_indices.begin() + kp_lo, k_indices.begin() + kp_hi});
    // Step 2: load sparse (weight) and reorder data for that.
    load_sparse(bsr_data, kp_lo, kp_hi);
    //  Step 3: tile product. Note that k_indices length is processed.
    //  A tile product can calculate at least 1 row and 16 columns of DST.
    //  Min tile calculation: Tile width/height is 1, compute (1, ADJ) x (ADJ, 16) = (1, 16) matmul.
    if (!param_.sub_func || TH_ != param_.tile_shape[0]) {
      tile_product(TH_, TW_);
    } else {
      call(sub_func_fptr_);
    }
  }
}

void jit_spmm_vnni_t::mul_scale(int i) {
  for (int j = 0; j < TW_; ++j) {
    vcvtdq2ps(dst_tile_Vmm(i, j) | T_rn_sae, dst_tile_Vmm(i, j));
    vmulps(dst_tile_Vmm(i, j), vreg_dst_temp, dst_tile_Vmm(i, j));
  }
}

void jit_spmm_vnni_t::move_out(int i, int j, int row_idx, int bytes) {
  int sliced_dst_idx = row_idx * dst_stride_[0] + j * VEC;
  if (bytes == BYTE1) {
    vpmovsdb(ptr[reg_dst + reg_nt_absolute_idx * bytes + sliced_dst_idx * bytes], dst_tile_Vmm(i, j));
  } else if (bytes == BYTE4) {
    vmovdqu32(ptr[reg_dst + reg_nt_absolute_idx * bytes + sliced_dst_idx * bytes], dst_tile_Vmm(i, j));
  }
}

void jit_spmm_vnni_t::store_intermediate_dst(int64_t m_start) {
  LOG(FATAL) << "K-blocking is not implemented.";
  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      int sliced_dst_idx = (m_start + i) * dst_stride_[0] + j * VEC;
      vmovdqu32(ptr[reg_dst + reg_nt_absolute_idx * BYTE4 + sliced_dst_idx * BYTE4], dst_tile_Vmm(i, j));
    }
  }
}

void jit_spmm_vnni_t::handle_dst_buffer_epilogue(int kb_idx, int64_t m_start) {
  if (kb_idx == k_blocks_ - 1) {
    for (int i = 0; i < TH_; ++i) {
      int row_idx = m_start + i;
      vbroadcastss(vreg_dst_temp, ptr[reg_scale + row_idx * BYTE4]);  // move in scale.
      mul_scale(i);
      for (int j = 0; j < TW_; ++j) {
        if (output_type_ == data_type::u8 || output_type_ == data_type::s8) {
          vcvtps2dq(dst_tile_Vmm(i, j) | T_rn_sae, dst_tile_Vmm(i, j));
          move_out(i, j, row_idx, BYTE1);
        } else if (output_type_ == data_type::fp32) {
          if (param_.append_sum) {
            int sliced_dst_idx = row_idx * dst_stride_[0] + j * VEC;
            vmovups(vreg_dst_temp, ptr[reg_dst + reg_nt_absolute_idx * BYTE4 + sliced_dst_idx * BYTE4]);
            vaddps(dst_tile_Vmm(i, j), vreg_dst_temp, dst_tile_Vmm(i, j));
          }
          move_out(i, j, row_idx, BYTE4);
        }
      }
    }
  } else {  // direct move out intermediate results
    store_intermediate_dst(m_start);
  }
}

void jit_spmm_vnni_t::read_params() {
  mov(reg_seq_vals, ptr[param1 + GET_OFF(ptr_seq_vals)]);
  mov(reg_dense, ptr[param1 + GET_OFF(ptr_dense)]);
  mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
  mov(reg_dst, ptr[param1 + GET_OFF(ptr_dst)]);
  mov(reg_scale, ptr[param1 + GET_OFF(ptr_scales)]);
}

void jit_spmm_vnni_t::gen_sub_function() {
  Xbyak::util::StackFrame callee1_sf(this, 0);
  tile_product(param_.tile_shape[0], param_.tile_shape[1]);
}

void jit_spmm_vnni_t::generate() {
  inLocalLabel();  // use local label for multiple instance
  if (param_.sub_func) {
    sub_func_fptr_ = getCurr();
    gen_sub_function();
    callee_functions_code_size_ = getSize();
  }
  Xbyak::Label g_label1;
  Xbyak::Label g_label2;
  {
    Xbyak::util::StackFrame spmm_sf(this, 1, 0, stack_space_needed_);
    read_params();
    mov(reg_nb_start, param_.in_start);
    mov(reg_nb_end, param_.in_end);

    // initialize the control avx vectors which we are going to use for permutes and shuffles.
    vpmovzxbd(vpermt2d_arg_idx, ptr[rip + g_label1]);
    vbroadcasti32x4(vpshufb_arg_b, ptr[rip + g_label2]);
    auto temp_r32 = Xbyak::Reg32(param1.getIdx());
    mov(temp_r32, 0xf0);  // param1 can be used as temp reg when all params are loaded
    kmovb(reg_k1, temp_r32);

    // Loop-N1: Assembly loop at "n_blocks". Asm code fold.
    Xbyak::Label L_nb_loop;
    L(L_nb_loop);
    // When K-dim is cut into k_blocks parts, it'll produce same number of DST intermediate results.
    LOG_IF(FATAL, k_blocks_ > 1) << "k-blocking is not fully implemented";
    for (int kb_idx = 0; kb_idx < k_blocks_; ++kb_idx) {
      // Loop-M2: CPP loop for each blocked row. Asm code unroll.
      for (int im = 0; im < param_.M; im += TH_) {
        xor_(reg_nt_relative_idx, reg_nt_relative_idx);
        mov(reg_nt_absolute_idx, reg_nb_start);

        // n_blocks and n_tiles are N-dim loops, and can be used for N-dim multithread.
        // n_blocks is asm outer loop, with longer assembly context span. n_tiles is inner loop.
        // Loop-N2: Assembly loop at "n_tiles". Asm code fold.
        Xbyak::Label L_nt_loop;
        L(L_nt_loop);
        // init dst buffer, like init the value to bias or the previous intermediate result.
        handle_dst_buffer_init(kb_idx, im);
        repeat_THx4xTW_matmal(im);
        // generate the epilogue logic. This is different depending on B_blocks value (should we
        // cache intermediate results or write results with post-op to output)
        handle_dst_buffer_epilogue(kb_idx, im);

        add(reg_nt_absolute_idx, nt_size_);
        add(reg_nt_relative_idx, nt_size_);
        cmp(reg_nt_relative_idx, nb_size_);
        jl(L_nt_loop, T_NEAR);  // Loop-N2 end.
      }                         // Loop-M2 end.
    }
    add(reg_nb_start, nb_size_);
    cmp(reg_nb_start, reg_nb_end);
    jl(L_nb_loop, T_NEAR);  // Loop-N1 end.
  }
  int word_size = 1;
  int num_size = 16;
  const uint8_t vpermt2d_control[16] = {0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23};
  const uint8_t vpshufb_control[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
  L(g_label1);
  for (int i = 0; i < num_size; ++i) {
    db(vpermt2d_control[i], word_size);
  }
  L(g_label2);
  for (int i = 0; i < num_size; ++i) {
    db(vpshufb_control[i], word_size);
  }
  outLocalLabel();  // end of local label
}
}  // namespace jd
