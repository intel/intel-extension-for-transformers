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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SPMM_VNNI_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SPMM_VNNI_HPP_

#include <omp.h>
#include <glog/logging.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "jit_generator.hpp"
#include "../kernels/sparse_data.hpp"
#include "../kernels/spmm_types.hpp"
#include "utils.hpp"

namespace jd {
/**
 * @brief jit_spmm_vnni_t calculates this kind matmul: sparse x dense = dst.
 *        weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
 */
class jit_spmm_vnni_t : public jit_generator {
 public:
  explicit jit_spmm_vnni_t(const ssd::vnni_param_t& param) : jit_generator(), param_(param){};
  virtual ~jit_spmm_vnni_t() {}

 public:
  const int8_t* sequence_vals() const { return seq_vals_.data(); }

 private:
  ssd::vnni_param_t param_;
  std::vector<int8_t> seq_vals_;

 private:
  void generate() override;

 private:
  // internal API of op kernel
  Xbyak::Zmm TH_Vmm(int i = 0);           // Register allocator of load weight. 1D shape=(TH)
  Xbyak::Zmm TW_Vmm(int i = 0);           // Register allocator of load activation. 1D shape=(TW)
  Xbyak::Zmm dst_tile_Vmm(int i, int j);  // Reg alloc of DST tile. 2D shape=(TH,TW), stride=(TW,1)
  void params_alias(const ssd::vnni_param_t& param);
  void read_params();
  void load_bias(int64_t m_start);
  void load_dense(const std::vector<int64_t>& k_indices);
  void load_sparse(const int8_t* bsr_data, int64_t kp_lo, int64_t kp_hi);
  void tile_product(int tile_height, int tile_width);
  void handle_dst_buffer_init(int kb_idx, int64_t m_start);
  void handle_dst_buffer_epilogue(int kb_idx, int64_t m_start);
  void mul_scale(int i);
  void move_out(int i, int j, int row_idx, int bytes = 1);
  std::unordered_map<int64_t, std::vector<int64_t>> get_idx_balanced(int64_t m_start,
                                                                     const std::vector<int64_t>& sparse_indptr,
                                                                     const std::vector<int64_t>& sparse_indices, int lo,
                                                                     int hi);
  std::unordered_map<int64_t, std::vector<int8_t>> get_val_balanced(int64_t m_start,
                                                                    const std::vector<int64_t>& sparse_indptr,
                                                                    const std::vector<int64_t>& sparse_indices, int lo,
                                                                    int hi, const std::vector<int8_t>& sparse_inddata);
  void repeat_THx4xTW_matmal(int64_t imb);
  void clear_dst_tile();
  void load_intermediate_dst(int64_t m_start);
  void store_intermediate_dst(int64_t m_start);
  void gen_sub_function();

  inline int TH() const { return param_.blocksize[0]; }
  inline int TW() const { return param_.tile_w; }
  inline int nt_size() const { return TW() * VEC; }
  inline int mt_size() const { return TH(); }
  inline int n_tiles() const { return param_.BN / nt_size(); }
  inline int m_tiles() const { return param_.BM / mt_size(); }
  inline data_type output_type() const { return param_.output_type; };
  inline int ld_dst() const { return param_.BN; }  // leading dimension of dst matrix

 private:
  const int64_t PADDED_NEG_ONE = -1;
  const int64_t PADDED_ZERO = 0;
  int64_t seq_pos = 0;
  const uint8_t* sub_func_fptr_ = nullptr;

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
  const Xbyak::Reg64& param1 = rdi;
  const Xbyak::Reg64& reg_seq_vals = rcx;  // the first argument which is packed nonzero values pointer
  const Xbyak::Reg64& reg_dense = rdx;     // the second argument which is input matrix pointer
  const Xbyak::Reg64& reg_bias = rsi;      // the third argument which is bias values pointer
  const Xbyak::Reg64& reg_dst = rax;       // the fourth argument which is output matrix pointer
  const Xbyak::Reg64& reg_scale = rbx;     // the scale
  const Xbyak::Opmask& reg_k1 = k1;

  const Xbyak::Reg64& reg_n_idx = r10;

  const Xbyak::Zmm& vpermt2d_arg_idx = zmm31;
  const Xbyak::Zmm& vpshufb_arg_b = zmm30;
  const Xbyak::Zmm& vreg_temp = zmm29;
  const Xbyak::Zmm& vreg_dst_temp = vreg_temp;
  static constexpr int USED_VREGS = 3;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SPMM_VNNI_HPP_
