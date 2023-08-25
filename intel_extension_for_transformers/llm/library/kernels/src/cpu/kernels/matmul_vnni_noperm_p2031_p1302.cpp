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

#include "matmul_vnni_noperm_p2031_p1302.hpp"
namespace jd {
using io = ssd::matmul_io::io;

/**
 * Dimension details:
 *   src0: bs0 bs1 m   k ===========> bs0 bs1 m k
 *   src1: bs1 n   bs0 k ==perm2031=> bs0 bs1 k n
 *   dst:  bs1 n   bs0 m <=perm1302== bs0 bs1 m n
 *
 * Entries of op_desc_.attrs:
 *   unified: transpose input while doing gemm. This optimization usually benefits small input.
 */

// Part1: class matmul_vnni_noperm_p2031_p1302_kd_t

bool matmul_vnni_noperm_p2031_p1302_kd_t::init() {
  if (!isa_available(avx512_core_vnni)) return false;
  auto shapes = op_desc_.tensor_shapes();
  auto dtypes = op_desc_.tensor_dtypes();

  for (auto mat : {io::SRC0, io::SRC1, io::SRC2, io::DST0})
    if (shapes[mat].size() != 4 && shapes[mat].size() != 0) {
      SPARSE_LOG(WARNING) << "All operand should be 4D matrix";
      return false;
    }

  std::vector<dim_t>& src0_perm_shape = shapes[io::SRC0];  // src0 perm none
  std::vector<dim_t> src1_perm_shape = {
      shapes[io::SRC1][2],  // bs0
      shapes[io::SRC1][0],  // bs1
      shapes[io::SRC1][3],  // K
      shapes[io::SRC1][1],  // N
  };
  std::vector<dim_t> dst0_perm_shape = {
      // reverse of 1302 is 2031
      shapes[io::DST0][2],  // bs0
      shapes[io::DST0][0],  // bs1
      shapes[io::DST0][3],  // M
      shapes[io::DST0][1],  // N
  };
  if (!shapes[io::SRC2].empty()) {
    SPARSE_LOG(WARNING) << "Does not support binary add";
    return false;
  }
  bool scaler_scale = shapes[io::SCALE0] == std::vector<dim_t>({1}) && shapes[io::ZP0] == std::vector<dim_t>({1});
  bool is_supported = (op_desc_.kernel_prop() == kernel_prop::forward_inference) &&
                      is_any_of({data_type::u8}, [&](const data_type& a) { return dtypes[io::SRC0] == a; }) &&
                      is_any_of({data_type::s8}, [&](const data_type& a) { return dtypes[io::SRC1] == a; }) &&
                      is_any_of({data_type::u8}, [&](const data_type& a) { return dtypes[io::DST0] == a; }) &&  //
                      scaler_scale && dtypes[io::SCALE0] == data_type::fp32 && dtypes[io::ZP0] == data_type::fp32;

  if (!is_supported) {
    SPARSE_LOG(WARNING) << "Skip as dtype not matched";
    return false;
  }

  if (src0_perm_shape[3] != src1_perm_shape[2]) {
    SPARSE_LOG(WARNING) << "Skip as src0 k-dim (" << src0_perm_shape[3] << ") doesn't match src1 k-dim ("
                        << src1_perm_shape[2] << ").";
    return false;
  }

  for (auto idx : {0, 1}) {
    for (auto shape_perm : {src0_perm_shape, src1_perm_shape}) {
      if (shape_perm.empty()) continue;
      if (shape_perm[idx] != dst0_perm_shape[idx]) {
        SPARSE_LOG(WARNING) << "First 2 dimensions of all tensors after permutation should be the same";
        return false;
      }
    }
  }

  matmul_params_init();
  return true;
}

bool matmul_vnni_noperm_p2031_p1302_kd_t::matmul_params_init() {
  const auto shapes = op_desc_.tensor_shapes();
  auto attrs = op_desc_.attrs();

  jit_param_.M = shapes[io::SRC0][2];      // aka src0_perm_shape[2]
  jit_param_.K = shapes[io::SRC0][3];      // aka src0_perm_shape[3]
  jit_param_.N = shapes[io::SRC1][1];      // aka src1_perm_shape[3]
  jit_param_.batch = shapes[io::SRC0][0];  // bs0

  // note that this kernel writes dst in col-major
  jit_param_.m_tile = 1;
  jit_param_.n_tile = 16;

  if (attrs["unified"] == "1") using_unified_kernel_ = true;
  return true;
}

// Part2: class matmul_vnni_noperm_p2031_p1302_k_t

matmul_vnni_noperm_p2031_p1302_k_t::matmul_vnni_noperm_p2031_p1302_k_t(const std::shared_ptr<const kd_t>& kd)
    : kernel_t(kd),
      t_shapes_(kd->get_operator_desc().tensor_shapes()),
      src0_perm_shape_(t_shapes_[io::SRC0]),  // src0 perm none
      src1_perm_shape_({
          t_shapes_[io::SRC1][2],
          t_shapes_[io::SRC1][0],
          t_shapes_[io::SRC1][3],
          t_shapes_[io::SRC1][1],
      }),
      dst1_perm_shape_({
          // reverse of 1302 is 2031
          t_shapes_[io::DST0][2],
          t_shapes_[io::DST0][0],
          t_shapes_[io::DST0][3],
          t_shapes_[io::DST0][1],
      }),
      M_(src0_perm_shape_[2]),
      K_(src0_perm_shape_[3]),
      N_(src1_perm_shape_[3]),
      bs0_(dst1_perm_shape_[0]),
      bs1_(dst1_perm_shape_[1]),
      using_unified_kernel_(derived_kd()->using_unified_kernel()) {}

bool matmul_vnni_noperm_p2031_p1302_k_t::init() {
  auto& ker_param = derived_kd()->jit_param();
  if (using_unified_kernel_) {
    auto ker = new jit_matmul_vnni_noperm_p2031_p1302_t(ker_param);
    if (ker == nullptr) return false;
    if (!ker->create_kernel()) {
      safe_delete(ker);
      return false;
    }
    jit_ker_noperm_p2031_p1302_ = ker;
  } else {
    auto K = static_cast<int>(K_), bs0 = static_cast<int>(bs0_);
    auto ker_tr_src0 = new jit_transpose_nx8_4b<32>({K, K});
    if (ker_tr_src0 == nullptr) return false;
    if (!ker_tr_src0->create_kernel()) {
      safe_delete(ker_tr_src0);
      return false;
    }
    jit_trans_src0_ = ker_tr_src0;

    auto ker_tr_src1 = new jit_transpose_nx8_4b<8>({K, K * bs0});
    if (ker_tr_src1 == nullptr) return false;
    if (!ker_tr_src1->create_kernel()) {
      safe_delete(ker_tr_src1);
      return false;
    }
    jit_trans_src1_ = ker_tr_src1;

    auto ker = new jit_matmul_vnni_Ba4b_Ab4a_ba_t(ker_param);
    if (ker == nullptr) return false;
    if (!ker->create_kernel()) {
      safe_delete(ker);
      return false;
    }
    jit_ker_Ba4b_Ab4a_ba_ = ker;

    // allocate memory on heap for large task (more than half MB)
    if ((M_ + N_) * K_ > 512 * 1024) {
      src0_tmp_ = reinterpret_cast<uint8_t*>(aligned_alloc(64, M_ * K_));
      src1_tmp_ = reinterpret_cast<int8_t*>(aligned_alloc(64, N_ * K_));
    }
  }
  return true;
}

void matmul_vnni_noperm_p2031_p1302_k_t::thread_exec(const std::vector<const void*>& rt_data, const dim_t ibs0,
                                                     const dim_t ibs1) const {
  constexpr int n_tile = 8;
  constexpr int m_tile = 32;
  constexpr int mb_size = 4 * m_tile;
  auto base_src0 = static_cast<const uint8_t*>(rt_data[io::SRC0]);
  auto base_src1 = static_cast<const int8_t*>(rt_data[io::SRC1]);
  auto base_dst = const_cast<uint8_t*>(static_cast<const uint8_t*>(rt_data[io::DST0]));
  auto base_scale = static_cast<const float*>(rt_data[io::SCALE0]);
  auto base_zp = reinterpret_cast<const float*>(rt_data[io::ZP0]);

  uint8_t* src0_tmp;
  int8_t* src1_tmp;
  if (src0_tmp_ != nullptr) {
    src0_tmp = src0_tmp_ + (ibs0 * bs1_ + ibs1) * M_ * K_;
    src1_tmp = src1_tmp_ + (ibs0 * bs1_ + ibs1) * N_ * K_;
  } else {
    size_t tmp_total_len = (M_ + N_) * K_ + 64 + 4;  // 64 for alignment; 4 for extra access due to ping-pong loading
    void* mem_tmp = alloca(tmp_total_len);
    char* mem_tmp_aligned = static_cast<char*>(std::align(64, (M_ + N_) * K_, mem_tmp, tmp_total_len));
    src0_tmp = reinterpret_cast<uint8_t*>(mem_tmp_aligned);
    src1_tmp = reinterpret_cast<int8_t*>(mem_tmp_aligned + M_ * K_);
  }

  for (dim_t i = 0; i < M_; i += m_tile) {
    // src0: bs0_ bs1_ M_ K_
    jit_transpose_nx8_4b<32>::rt_data_t rt_param;
    rt_param.src = base_src0 + ibs0 * bs1_ * M_ * K_ + ibs1 * M_ * K_ + i * K_;
    rt_param.dst = src0_tmp + i * K_;
    (*jit_trans_src0_)(&rt_param);
  }
  for (dim_t j = 0; j < N_; j += n_tile) {
    // src1: bs1_ N_ bs0 K_
    jit_transpose_nx8_4b<8>::rt_data_t rt_param;
    rt_param.src = base_src1 + ibs0 * K_ + ibs1 * N_ * bs0_ * K_ + j * bs0_ * K_;
    rt_param.dst = src1_tmp + j * K_;
    (*jit_trans_src1_)(&rt_param);
  }

  for (dim_t i = 0; i < M_; i += mb_size)
    for (dim_t j = 0; j < N_; j += n_tile) {
      const dim_t max_ii = std::min(M_, i + mb_size);
      for (dim_t ii = i; ii < max_ii; ii += m_tile) {
        ssd::matmul_u8_data_t rt_param;
        // dst: bs1_ N bs0 M
        rt_param.src0 = src0_tmp + ii * K_;
        rt_param.src1 = src1_tmp + j * K_;
        rt_param.dst = base_dst + ibs1 * bs0_ * N_ * M_ + j * bs0_ * M_ + ibs0 * M_ + ii;
        rt_param.scale = base_scale;
        rt_param.zp = base_zp;

        // each jit kernel calculates 32xKx8, where K should be multiple of 32 (VNNI_R * dim_transpose)
        (*jit_ker_Ba4b_Ab4a_ba_)(&rt_param);
      }
    }
}

bool matmul_vnni_noperm_p2031_p1302_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto base_src0 = reinterpret_cast<const uint8_t*>(rt_data[io::SRC0]);
  auto base_src1 = reinterpret_cast<const int8_t*>(rt_data[io::SRC1]);
  auto base_dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(rt_data[io::DST0]));
  auto base_scale = reinterpret_cast<const float*>(rt_data[io::SCALE0]);
  auto base_zp = reinterpret_cast<const float*>(rt_data[io::ZP0]);
  if (using_unified_kernel_) {
#pragma omp parallel for collapse(2)
    for (dim_t ibs1 = 0; ibs1 < bs1_; ++ibs1)
      for (dim_t j = 0; j < N_; j += 16)
        for (dim_t ibs0 = 0; ibs0 < bs0_; ++ibs0)
          for (dim_t i = 0; i < M_; i += 16) {  // call the kernel
            ssd::matmul_u8_data_t rt_param;
            rt_param.src0 = base_src0 + ibs0 * bs1_ * M_ * K_ + ibs1 * M_ * K_ + i * K_;
            rt_param.src1 = base_src1 + ibs1 * bs0_ * N_ * K_ + j * bs0_ * K_ + ibs0 * K_;
            rt_param.dst = base_dst + ibs1 * bs0_ * N_ * M_ + j * bs0_ * M_ + ibs0 * M_ + i;
            rt_param.scale = base_scale;
            rt_param.zp = base_zp;

            // each jit kernel calculates 16xKx16, where K should be multiple of 32 (VNNI_R * dim_transpose)
            (*jit_ker_noperm_p2031_p1302_)(&rt_param);
          }
  } else {
#pragma omp parallel for collapse(2)
    for (dim_t ibs1 = 0; ibs1 < bs1_; ++ibs1)
      for (dim_t ibs0 = 0; ibs0 < bs0_; ++ibs0) {
        thread_exec(rt_data, ibs0, ibs1);
      }
  }
  return true;
}
}  // namespace jd
