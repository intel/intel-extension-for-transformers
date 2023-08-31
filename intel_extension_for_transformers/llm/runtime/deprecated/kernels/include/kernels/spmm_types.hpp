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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_TYPES_HPP_
#include <cstdint>
#include <vector>

#include "param_types.hpp"

namespace jd {
template <typename T>
class csrp_data_t;
template <typename T>
class bsc_data_t;
template <typename T>
class bsr_data_t;
namespace ssd {
/**
 * @brief tensors index configuration of this kernel.
 * TODO(Yi): potential confliction with indices of other op types
 */
static constexpr int WEI = 0;
static constexpr int SRC = 1; /*  bs*seq, 768  */
static constexpr int BIAS = 2;
static constexpr int DST = 3;
static constexpr int SCALES = 4;
static constexpr int DST_M1 = 5;      // m in Welford's online algorithm
static constexpr int DST_M2 = 6;      // m2 in Welford's online algorithm
static constexpr int WORK_SPACE = 7;  // memory be used in computing

/**
 * @brief Scenarios supported by spmm_vnni kernel/algorithm.
 */
enum class sparse_scheme : uint8_t {
  undef,
  sparse_x_dense,
  dense_x_sparse,
  sparse_x_sparse,
};

enum class subfunc_level : uint8_t {
  none,       // No sub-function
  non_kdims,  // fold all except on K-dimension
  kdims,      // a whole THxKxTW tile generates a constent size of code
  subfunc_level_MAX = kdims
};

/**
 * @brief kernel parameters passed between kernel/primitive and jit_domain.
 */
struct vnni_param_t {
  dim_t BN;  // size on N-dim; used as the leading dim of dense / dst, aka micro_bs
  dim_t BM;  // size on M-dim for the blocks computed by this kernel, aka micro_oc;
  bool has_bias;
  bool append_sum;
  data_type output_type;
  int tile_w;  // width of a tile in terms of #registers; Note that the height is determined by the height of BSR block
  subfunc_level sub_func;
  dim_t im_start;  // start m-idx of dest to be calculated; used as the m offset when reading sparse
  // sparse weight related
  dim_t blocksize[2] = {4, 1};
  std::vector<dim_t> indptr;
  std::vector<dim_t> indices;
  const int8_t* weight;
  std::vector<postop_attr> postop_attrs;
  bool welford;
};

/**
 * @brief kernel data at runtime.
 */
template <typename dst_t>
struct vnni_data_t {
  const uint8_t* ptr_dense;  // activation(K, N).
  const int32_t* ptr_bias;   // bias(M, 1).
  dst_t* ptr_dst;            // dst(M, N).
  const float* ptr_scales;   // bias(M, 1)
  float* ptr_dst_m1;         // m in Welford's online algorithm
  float* ptr_dst_m2;         // m2 in Welford's online algorithm
};

/**
 * @brief kernel parameters for kernel initialization
 */
template <typename T>
struct amx_params_t {
  dim_t num_tileM;
  dim_t tileM;
  dim_t tileN;
  dim_t shape[2];
  dim_t blocksize[2] = {16, 1};
  dim_t blocks_per_group = 64 / sizeof(T);
  dim_t nnz_group;
  dim_t nrowptr;
  dim_t* colidxs;
  dim_t* group_rowptr;
  T* weight;
  bool has_bias;
  bool same_src_dtype;
  std::vector<postop_attr> postop_attrs;
};

typedef amx_params_t<bfloat16_t> amx_bf16_params_t;
typedef amx_params_t<int8_t> amx_int8_params_t;

/**
 * @brief kernel inputs for kernel runtime
 */
template <typename src_t, typename wgt_t, typename dst_t, typename bia_t>
struct amx_inputs_t {
  src_t* weight;
  wgt_t* src;
  bia_t* bias;
  dst_t* dst;
};

typedef amx_inputs_t<bfloat16_t, bfloat16_t, float, float> amx_bf16f32_inputs_t;
typedef amx_inputs_t<bfloat16_t, bfloat16_t, bfloat16_t, float> amx_bf16bf16_inputs_t;

struct avx512_fp32_params_t {
  int64_t M;
  int64_t K;
  int64_t N;
  bool has_bias;
  bsc_data_t<float>* sparse_ptr;
  int64_t im_start;  // start m-idx of dest to be calculated
  int64_t im_end;    // end m-idx of dest to be calculated
  int64_t in_start;  // start n-idx of dest to be calculated
  int64_t in_end;    // end n-idx of dest to be calculated
  std::vector<postop_attr> postop_attrs;
};

/**
 * @brief kernel data at runtime.
 */
struct avx512_data_t {
  const float* dense;
  const float* sparse;
  const float* bias;
  float* dst;
};

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_TYPES_HPP_
