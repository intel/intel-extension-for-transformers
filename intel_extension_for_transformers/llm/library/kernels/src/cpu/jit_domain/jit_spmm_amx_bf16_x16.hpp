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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SPMM_AMX_BF16_X16_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SPMM_AMX_BF16_X16_HPP_

#include <glog/logging.h>
#include <omp.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "kernels/sparse_data.hpp"
#include "kernels/spmm_types.hpp"
#include "jit_generator.hpp"
#include "src/utils.hpp"
#include "jit_eltwise_injector.hpp"

#define TILE_M 16  // Number of rows in an A or C tile
#define TILE_K 32  // Number of columns in an A tile or rows in a B tile
#define TILE_N 16  // Number of columns in a B or C tile
#define KPACK 2    // Vertical K packing into dword
#define MZ 64      // (M / MT)
#define NUM_M 4    // (MZ / TILE_N)

namespace jd {
typedef uint16_t src_t;
typedef float dst_t;
/**
 * @brief jit_spmm_amx_bf16_x16_t calculates this kind matmul: sparse x dense =
 * dst. weight(N, K) * activation(K, M) + bias(N, 1) = dst(N, M)
 */
class jit_spmm_amx_bf16_x16_t : public jit_generator {
 public:
  explicit jit_spmm_amx_bf16_x16_t(const ssd::amx_bf16_params_t& param) : jit_generator(), param_(param) {
    eltwise_injector.eltwise_injector_init(this, param_.postop_attrs);
    N = param_.shape[0];
    K = param_.shape[1];
    nnz_group = param_.nnz_group;
    nrowptr = param_.nrowptr;
    colidxs = param_.colidxs;
    group_rowptr = param_.group_rowptr;
    weight_ = param_.weight;
    tileM = param_.tileM;
    bf16_out = param_.same_src_dtype;
    size_of_dst_t = size_of_out_t;
    if (bf16_out) {
      size_of_dst_t = size_of_src_t;
    }
  }
  virtual ~jit_spmm_amx_bf16_x16_t() {}

 public:
  const bfloat16_t* weight() const { return weight_; }

 private:
  ssd::amx_bf16_params_t param_;
  jit_eltwise_injector eltwise_injector;

  void generate() override;

  const Xbyak::uint8* jit_ker_ = nullptr;
#ifdef _WIN32
  const Xbyak::Reg64& reg_param = rcx;
  const Xbyak::Reg64& reg_dst = rdi;
#else
  const Xbyak::Reg64& reg_param = rdi;
  const Xbyak::Reg64& reg_dst = rcx;
#endif
  const Xbyak::Reg64& reg_weight = r15;
  const Xbyak::Reg64& reg_src = rdx;
  const Xbyak::Reg64& reg_bia = rbx;
  const Xbyak::Reg64& reg_m = rbp;
  const Xbyak::Reg64& reg_mstart = r9;
  const Xbyak::Reg64& reg_tileM = r10;
  const Xbyak::Reg64& reg_temp = rsi;
  const Xbyak::Reg64& reg_stride = r8;
  const Xbyak::Zmm& reg_mask = zmm31;
  const Xbyak::Ymm& reg_bf16 = ymm30;
  const Xbyak::Opmask& ktail_mask = k2;
  Xbyak::Label loopMask, lM;

  dim_t N;
  dim_t K;
  const dim_t blocks_per_group = TILE_K;
  dim_t nnz_group;
  dim_t nrowptr;
  dim_t* colidxs;
  dim_t* group_rowptr;
  bfloat16_t* weight_;
  dim_t tileM;
  bool bf16_out;
  dim_t size_of_dst_t;
  dim_t size_of_src_t = sizeof(src_t);  // size of bfloat16
  dim_t size_of_out_t = sizeof(dst_t);  // size of float since bf16 x bd16 = fp32

  static constexpr int stack_space_needed_ = 5120;

  void handle_postop_escape_vmms();
  void handle_postop_escape_regs();
  void postop_and_store_dst(int i, int j);
  void handle_dst_offset(int b_row);
  void read_inputs();
  void main_compute();
  void loop_M();
  void init_param();
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SPMM_AMX_BF16_X16_HPP_
