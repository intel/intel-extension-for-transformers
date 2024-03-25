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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SPMM_VNNI_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SPMM_VNNI_HPP_

#include <omp.h>
#include <glog/logging.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include "src/utils.hpp"
#include "jit_generator.hpp"
#include "kernels/sparse_data.hpp"
#include "kernels/spmm_types.hpp"
#include "jit_eltwise_injector.hpp"

namespace jd {
/**
 * @brief jit_spmm_vnni_t calculates this kind matmul: sparse x dense = dst.
 *        weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
 */
class jit_spmm_vnni_t : public jit_generator {
 public:
  explicit jit_spmm_vnni_t(const ssd::vnni_param_t& param) : jit_generator(), param_(param) {
    const dim_t imb_lo = param_.im_start / TH();
    const dim_t imb_hi = (param_.im_start + param.BM) / TH();
    const dim_t indptr_lo = param_.indptr[imb_lo] * spns::ADJ;
    const dim_t indptr_hi = param_.indptr[imb_hi] * spns::ADJ;
    const dim_t blk_size = param_.blocksize[0] * param_.blocksize[1];
    dense_load_offsets.resize((indptr_hi - indptr_lo) * blk_size);

    std::transform(param_.indices.begin() + indptr_lo, param_.indices.begin() + indptr_hi, dense_load_offsets.begin(),
                   [&](decltype(param_.indices)::value_type k) { return k * ld_dst(); });
    eltwise_injector_.eltwise_injector_init(this, param_.postop_attrs);
  }
  virtual ~jit_spmm_vnni_t() {}

 private:
  ssd::vnni_param_t param_;
  std::vector<dim_t, aligned_allocator_t<dim_t>> dense_load_offsets;  // param_.indices * ld_dst
  jit_eltwise_injector eltwise_injector_;

 private:
  void generate() override;

 private:
  // internal API of op kernel
  Xbyak::Zmm TH_Vmm(int i = 0);           // Register allocator of load weight. 1D shape=(TH)
  Xbyak::Zmm TW_Vmm(int i = 0);           // Register allocator of load activation. 1D shape=(TW)
  Xbyak::Zmm Temp_Vmm(int i = 0);         // Register allocator of load activation. 1D shape=(TW)
  Xbyak::Zmm dst_tile_Vmm(int i, int j);  // Reg alloc of DST tile. 2D shape=(TH,TW), stride=(TW,1)
  void params_alias(const ssd::vnni_param_t& param);
  void read_params();
  void load_bias(dim_t m_start);
  void load_dense(const std::vector<int64_t>& k_indices);
  void load_sparse(const Xbyak::Reg64& reg_addr, uint64_t offset);
  void tile_product(int tile_height, int tile_width);
  void handle_dst_buffer_init(int kb_idx, dim_t m_start);
  void handle_dst_buffer_epilogue_sub(bool set_zero);
  void repeat_THx4xTW_matmal(dim_t m_start);
  void clear_dst_tile();
  void load_intermediate_dst(dim_t m_start);
  void store_intermediate_dst(dim_t m_start);
  void load_dense_sparse_prod();
  void gen_subfunc_load_and_prod();
  void gen_subfunc_dst_epilogue();
  void handle_postop_escape_vmms();
  void handle_postop_escape_regs();
  void calc_mean_variance(int i, int j, bool set_zero);

  inline int TH() const { return param_.blocksize[0]; }
  inline int TW() const { return param_.tile_w; }
  inline int nt_size() const { return TW() * VEC; }
  inline int mt_size() const { return TH(); }
  inline int n_tiles() const { return param_.BN / nt_size(); }
  inline int m_tiles() const { return param_.BM / mt_size(); }
  const data_type& output_type();
  inline int ld_dst() const { return param_.BN; }  // leading dimension of dst matrix

 private:
  const int64_t PADDED_NEG_ONE = -1;
  const int64_t PADDED_ZERO = 0;
  Xbyak::Label func_load_and_prod_;       // subfunction for dense load & sparse load & tile product
  Xbyak::Label func_dst_epilogue_;        // subfunction for dst handling
  Xbyak::Label func_dst_epilogue_start_;  // subfunction for the first iteration of dst handling

  Xbyak::Label L_m512_1f;  // address where 16x 1.f stored
  Xbyak::Label L_vpermt2d_arg;
  Xbyak::Label L_vpshufb_arg;

 private:
  static constexpr int stack_space_needed_ = 200;
  static constexpr int BYTE8 = 8;
  static constexpr int BYTE4 = 4;
  static constexpr int BYTE1 = 1;
  static constexpr int VREG_NUMS = 32;
#ifdef XBYAK64
  static constexpr int PTR_SIZE = 8;
#else
  static constexpr int PTR_SIZE = 4;
#endif
  // Register decomposition
#ifdef _WIN32
  const Xbyak::Reg64& param1 = rcx;
  const Xbyak::Reg64& reg_wei = rdi;  // the first argument which is packed nonzero values pointer
#else
  const Xbyak::Reg64& param1 = rdi;
  const Xbyak::Reg64& reg_wei = rcx;  // the first argument which is packed nonzero values pointer
#endif
  const Xbyak::Reg64& reg_dense = rdx;  // the second argument which is input matrix pointer
  const Xbyak::Reg64& reg_bias = rsi;   // the third argument which is bias values pointer
  const Xbyak::Reg64& reg_dst = rax;    // the fourth argument which is output matrix pointer
  const Xbyak::Reg64& reg_scale = rbx;  // the scale
  const Xbyak::Opmask& reg_k1 = k1;

  const Xbyak::Reg64& reg_k_ptr = reg_dst;
  const Xbyak::Reg64& reg_tmp = r9;
  const Xbyak::Reg64& reg_dst_idx = r8;
  const Xbyak::Reg64& reg_m_idx = reg_tmp;
  const Xbyak::Reg64& reg_n_idx = r10;
  const Xbyak::Reg64& reg_seq_indices = r11;
  const Xbyak::Reg64 reg_addr_tmp[4] = {r12, r13, r14, r15};

  const Xbyak::Reg64& reg_dst_m1 = reg_addr_tmp[0];
  const Xbyak::Reg64& reg_dst_m2 = reg_addr_tmp[1];

  const Xbyak::Zmm& vreg_temp = zmm29;
  const Xbyak::Zmm& vpshufb_arg_b = zmm30;
  const Xbyak::Zmm& vpermt2d_arg_idx = zmm31;
  const Xbyak::Zmm& vreg_m_idx = vpshufb_arg_b;
  const Xbyak::Zmm& vreg_temp2 = vpermt2d_arg_idx;

  static constexpr int USED_VREGS = 3;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SPMM_VNNI_HPP_
