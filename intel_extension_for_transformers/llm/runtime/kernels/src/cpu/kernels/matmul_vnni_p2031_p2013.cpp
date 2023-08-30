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

#include "matmul_vnni_p2031_p2013.hpp"

namespace jd {
using io = ssd::matmul_io::io;

/**
 * Dimension details:
 *   src0:     bs1 k   bs0 m  ==perm2031=> bs0 bs1 m k
 *   src1:     bs1 k   bs0 n  ==perm2013=> bs0 bs1 k n
 *   src2/dst: bs0 bs1 m   n  ===========> bs0 bs1 m n
 *
 * Entries of op_desc_.attrs:
 *   src0_scale: scale of the left quantized tensor
 *   src1_scale: scale of the right quantized tensor
 *   out_scale:  output scale
 *
 * ouput_f32
 *   = src0_f32 * src1_f32 * out_scale + src2_f32
 *   = (src0_s8 / src0_scale) * (src1_s8 / src1_scale) * out_scale + src2_f32
 *   = ((src0p128_u8 - 128) / src0_scale) * (src1_s8 / src1_scale) * out_scale + src2_f32
 *   = (src0p128_u8 * src1_s8 - 128 * sum(src1_s8, axis=K)) / (src0_scale * src1_scale) * out_scale + src2_f32
 * where
 *   src0p128_u8 = src0_s8 + 128
 */

// Part1: class matmul_vnni_p2031_p2013_kd_t

bool matmul_vnni_p2031_p2013_kd_t::init() {
  if (!isa_available(avx512_core_vnni)) return false;
  auto shapes = op_desc_.tensor_shapes();
  auto dtypes = op_desc_.tensor_dtypes();
  auto& attrs = op_desc_.attrs();

  for (auto mat : {io::SRC0, io::SRC1, io::SRC2, io::DST0})
    if (shapes[mat].size() != 4 && shapes[mat].size() != 0) {
      SPARSE_LOG(WARNING) << "All operand should be 4D matrix";
      return false;
    }

  std::vector<dim_t> src0_perm_shape = {
      shapes[io::SRC0][2],  // bs0
      shapes[io::SRC0][0],  // bs1
      shapes[io::SRC0][3],  // m
      shapes[io::SRC0][1],  // k
  };

  std::vector<dim_t> src1_perm_shape = {
      shapes[io::SRC1][2],  // bs0
      shapes[io::SRC1][0],  // bs1
      shapes[io::SRC1][1],  // k
      shapes[io::SRC1][3],  // n
  };

  if (shapes[io::SRC0][3] % 8 != 0) {
    SPARSE_LOG(WARNING) << "M must be a multiple of 8";
    return false;
  }
  if (shapes[io::SRC0][1] % 8 != 0) {
    SPARSE_LOG(WARNING) << "K must be a multiple of 8";
    return false;
  }

  bool has_binary_add = !shapes[io::SRC2].empty();
  if (has_binary_add && dtypes[io::SRC2] != data_type::fp32) {
    SPARSE_LOG(WARNING) << "Argument of binary_add must be of type f32!";
    return false;
  }

  bool is_supported = (op_desc_.kernel_prop() == kernel_prop::forward_inference) &&
                      is_any_of({data_type::s8}, [&](const data_type& a) { return dtypes[io::SRC0] == a; }) &&
                      is_any_of({data_type::s8}, [&](const data_type& a) { return dtypes[io::SRC1] == a; }) &&
                      is_any_of({data_type::s8, data_type::u8, data_type::fp32},
                                [&](const data_type& a) { return dtypes[io::DST0] == a; });
  if (!is_supported) {
    SPARSE_LOG(WARNING) << "Argument type not supported by matmul_vnni_p2031_p2013_kd_t!";
    return false;
  }

  auto& postops = op_desc_.apply_postops_list();
  if (dtypes[io::DST0] != (postops.size() != 0 ? postops.back().dt : data_type::fp32)) {
    SPARSE_LOG(WARNING) << "DST type does not match the last postop's type!";
    return false;
  }

  if (src0_perm_shape[3] != src1_perm_shape[2]) {
    SPARSE_LOG(WARNING) << "Skip as src0 k-dim (" << src0_perm_shape[3] << ") and src1 k-dim (" << src1_perm_shape[2]
                        << ") don't match!";
    return false;
  }

  for (auto idx : {0, 1}) {
    for (auto shape_perm : {src0_perm_shape, src1_perm_shape, shapes[io::SRC2]}) {
      if (shape_perm.empty()) continue;
      if (shape_perm[idx] != shapes[io::DST0][idx]) {
        SPARSE_LOG(WARNING) << "First 2 dimensions of all tensors after permutation should be the same";
        return false;
      }
    }
  }

  if (attrs.find("src0_scale") != attrs.end())
    src0_scale_ = str_to_num<float>(attrs.at("src0_scale"));
  else
    SPARSE_LOG(ERROR) << "Missing attr src0_scale, default to " << src0_scale_;
  if (attrs.find("src1_scale") != attrs.end())
    src1_scale_ = str_to_num<float>(attrs.at("src1_scale"));
  else
    SPARSE_LOG(ERROR) << "Missing attr src1_scale, default to " << src1_scale_;
  if (attrs.find("out_scale") != attrs.end()) out_scale_ = str_to_num<float>(attrs.at("out_scale"));
  return true;
}

// Part2: class matmul_vnni_p2031_p2013_k_t

matmul_vnni_p2031_p2013_k_t::matmul_vnni_p2031_p2013_k_t(const std::shared_ptr<const kd_t>& kd)
    : kernel_t(kd),
      t_shapes_(kd->get_operator_desc().tensor_shapes()),
      src0_perm_shape_({
          t_shapes_[io::SRC0][2],
          t_shapes_[io::SRC0][0],
          t_shapes_[io::SRC0][3],
          t_shapes_[io::SRC0][1],
      }),
      src1_perm_shape_({
          t_shapes_[io::SRC1][2],
          t_shapes_[io::SRC1][0],
          t_shapes_[io::SRC1][1],
          t_shapes_[io::SRC1][3],
      }),
      M_(src0_perm_shape_[2]),
      K_(src0_perm_shape_[3]),
      N_(src1_perm_shape_[3]),
      bs0_(t_shapes_[io::DST0][0]),
      bs1_(t_shapes_[io::DST0][1]),
      M_pad_(pad_to(M_, 8L)),
      K_pad_(pad_to(K_, 8L)),
      N_pad_(pad_to(N_, 48L)),
      tmp0_bytes(M_pad_ * K_pad_ * sizeof(uint8_t)),
      tmp1_bytes(N_pad_ * K_pad_ * sizeof(int8_t)),
      tmp_sum_bytes(N_pad_ * sizeof(int32_t)),
      dst_type(kd->get_operator_desc().tensor_descs()[io::DST0].dtype()),
      has_binary_add(t_shapes_[io::SRC2].size() != 0),
      use_thread_exec_((M_ + N_) * K_ < 512 * 1024) {}

bool matmul_vnni_p2031_p2013_k_t::init() {
  auto& dkd = *derived_kd();
  try {
    // The kernel for preprocessing src0
    add128_2x8x8_ker_ = new jit_seq_cpy_2x8x8(jit_seq_cpy_2x8x8::param_t{128});
    if (!add128_2x8x8_ker_->create_kernel()) return false;

    // The kernel for preprocessing src1
    constexpr bool src1_is_unsigned = false;
    cpy_48x4_ker_ = new jit_seq_cpy_48x4(jit_seq_cpy_48x4::param_t{true, src1_is_unsigned, 0});
    if (!cpy_48x4_ker_->create_kernel()) return false;

    // The kernel for compute matmul
    if (N_ >= 48) {
      matmul_ker_ = new jit_matmul_vnni_8xkx48_t(jit_matmul_vnni_8xkx48_t::param_t{
          K_,
          N_,
          dkd.out_scale() / dkd.src0_scale() / dkd.src1_scale(),
          7,  // bias shift left 7 == mul 128
          has_binary_add,
          dst_type,
          kd_->get_operator_desc().apply_postops_list(),
          8,
          48,
      });
      if (!matmul_ker_->create_kernel()) return false;
    }
    if (N_ % 48 != 0) {
      matmul_tile_n_ker_ = new jit_matmul_vnni_8xkx48_t(jit_matmul_vnni_8xkx48_t::param_t{
          K_,
          N_,
          dkd.out_scale() / dkd.src0_scale() / dkd.src1_scale(),
          7,  // bias shift left 7 == mul 128
          has_binary_add,
          dst_type,
          kd_->get_operator_desc().apply_postops_list(),
          8,
          N_ % 48,
      });
      if (!matmul_tile_n_ker_->create_kernel()) return false;
    }
    if (!use_thread_exec_) {
      src0_tmp_ = aligned_allocator_t<uint8_t>::allocate(bs0_ * bs1_ * M_pad_ * K_pad_);
      src1_tmp_ = aligned_allocator_t<int8_t>::allocate(bs0_ * bs1_ * N_pad_ * K_pad_);
      sum_tmp_ = aligned_allocator_t<int32_t>::allocate(bs0_ * bs1_ * N_pad_ * max_cpy_nthr);
    }
  } catch (const std::exception& e) {
    SPARSE_LOG(WARNING) << e.what();
    return false;
  }
  return true;
}
template <typename dt_dst>
inline bool matmul_vnni_p2031_p2013_k_t::thread_exec(const std::vector<const void*>& rt_data, const dim_t ibs0,
                                                     const dim_t ibs1) const {
  const auto curr_src0 = reinterpret_cast<const uint8_t*>(rt_data[io::SRC0]) + ibs0 * M_ + ibs1 * bs0_ * M_ * K_;
  const auto curr_src1 = reinterpret_cast<const int8_t*>(rt_data[io::SRC1]) + ibs0 * N_ + ibs1 * bs0_ * N_ * K_;
  const auto curr_dst_offset = (ibs0 * bs1_ + ibs1) * M_ * N_;
  const auto curr_dst = const_cast<dt_dst*>(reinterpret_cast<const dt_dst*>(rt_data[io::DST0])) + curr_dst_offset;
  const auto curr_src2 = reinterpret_cast<const float*>(rt_data[io::SRC2]) + curr_dst_offset;
  const size_t tmp_total_len = tmp0_bytes + tmp1_bytes + tmp_sum_bytes;
  size_t tmp_total_len_raw = tmp_total_len + 64;
  void* mem_tmp = alloca(tmp_total_len_raw);  // 64 for alignment
  std::align(64, tmp_total_len, mem_tmp, tmp_total_len_raw);
  const auto curr_src0_tmp = reinterpret_cast<uint8_t*>(mem_tmp);
  const auto curr_src1_tmp = reinterpret_cast<int8_t*>(reinterpret_cast<char*>(mem_tmp) + tmp0_bytes);
  const auto curr_sum_tmp = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(mem_tmp) + tmp0_bytes + tmp1_bytes);

  // reorder src1 and compute sum over K
  jit_seq_cpy_48x4::rt_data_t rt_cpy_src1{
      nullptr,                         // src
      nullptr,                         // dst
      curr_sum_tmp,                    // dst_sum
      false,                           // sum_append
      static_cast<int>(N_),            // N
      static_cast<int>(N_ * bs0_),     // ld_src
      jit_seq_cpy_48x4::dst_step(K_),  // ld_dst
  };
  for (dim_t k = 0; k < K_; k += 4) {
    rt_cpy_src1.src = curr_src1 + k * bs0_ * N_;
    rt_cpy_src1.dst = curr_src1_tmp + k * 48;
    rt_cpy_src1.sum_append = (k != 0);
    (*cpy_48x4_ker_)(&rt_cpy_src1);
  }

  // reorder src0 + 128
  jit_seq_cpy_2x8x8::rt_data_t rt_cpy_src0{
      nullptr,                          // src
      nullptr,                          // dst
      static_cast<int>(M_),             // N
      static_cast<int>(M_ * bs0_),      // ld_src
      jit_seq_cpy_2x8x8::dst_step(K_),  // ld_dst
  };
  for (dim_t k = 0; k < K_; k += 8) {
    rt_cpy_src0.src = curr_src0 + k * bs0_ * M_;
    rt_cpy_src0.dst = curr_src0_tmp + k * 8;
    (*add128_2x8x8_ker_)(&rt_cpy_src0);
  }

  jit_matmul_vnni_8xkx48_t::rt_data_t<dt_dst> rt_matmul;
  for (dim_t i = 0; i < M_; i += 8) {
    for (dim_t j = 0; j < N_; j += 48) {
      rt_matmul.src0 = curr_src0_tmp + i * K_;
      rt_matmul.src1 = curr_src1_tmp + j * K_;
      rt_matmul.bias = curr_sum_tmp + j;
      rt_matmul.src_b0 = curr_src2 + i * N_ + j;
      rt_matmul.dst = curr_dst + i * N_ + j;
      (j <= N_ - 48 ? *matmul_ker_ : *matmul_tile_n_ker_)(&rt_matmul);
    }
  }
  return true;
}

template <typename dt_dst>
inline bool matmul_vnni_p2031_p2013_k_t::execute_(const std::vector<const void*>& rt_data) const {
  if (use_thread_exec_) {
    bool succeed = true;
#pragma omp parallel for collapse(2)
    for (dim_t ibs1 = 0; ibs1 < bs1_; ++ibs1)
      for (dim_t ibs0 = 0; ibs0 < bs0_; ++ibs0) thread_exec<dt_dst>(rt_data, ibs0, ibs1);
    return succeed;
  }
  const auto base_src0 = static_cast<const uint8_t*>(rt_data[io::SRC0]);
  const auto base_src1 = static_cast<const int8_t*>(rt_data[io::SRC1]);
  const auto base_dst = const_cast<dt_dst*>(static_cast<const dt_dst*>(rt_data[io::DST0]));
  const auto base_src2 = static_cast<const float*>(rt_data[io::SRC2]);

#pragma omp parallel for collapse(2)
  for (dim_t ibs1 = 0; ibs1 < bs1_; ++ibs1)
    for (dim_t ibs0 = 0; ibs0 < bs0_; ++ibs0) {
      for (dim_t k = 0; k < K_; k += 8) {
        jit_seq_cpy_2x8x8::rt_data_t rt_cpy_src0;
        const auto curr_src0 = base_src0 + ibs0 * M_ + ibs1 * bs0_ * M_ * K_;
        const auto curr_src0_tmp = src0_tmp_ + (ibs0 * bs1_ + ibs1) * M_pad_ * K_pad_;
        rt_cpy_src0.src = curr_src0 + k * bs0_ * M_;
        rt_cpy_src0.dst = curr_src0_tmp + k * 8;
        (*add128_2x8x8_ker_)(&rt_cpy_src0);
      }
    }

#pragma omp parallel for collapse(2)
  for (dim_t ibs1 = 0; ibs1 < bs1_; ++ibs1)
    for (dim_t ibs0 = 0; ibs0 < bs0_; ++ibs0) {
      for (dim_t k = 0; k < K_; k += 4) {
        jit_seq_cpy_48x4::rt_data_t rt_cpy_src1;
        const auto curr_src1 = base_src1 + ibs0 * N_ + ibs1 * bs0_ * N_ * K_;
        const auto curr_src1_tmp = src1_tmp_ + (ibs0 * bs1_ + ibs1) * N_pad_ * K_pad_;
        const auto curr_sum_tmp = sum_tmp_ + (ibs0 * bs1_ + ibs1) * N_pad_ * max_cpy_nthr;
        rt_cpy_src1.src = curr_src1 + k * bs0_ * N_;
        rt_cpy_src1.dst = curr_src1_tmp + k * 48;
        rt_cpy_src1.dst_sum = curr_sum_tmp;
        rt_cpy_src1.sum_append = (k != 0);
        (*cpy_48x4_ker_)(&rt_cpy_src1);
      }
    }

#pragma omp parallel for collapse(2)
  for (dim_t ibs1 = 0; ibs1 < bs1_; ++ibs1)
    for (dim_t ibs0 = 0; ibs0 < bs0_; ++ibs0) {
      const auto curr_dst_offset = (ibs0 * bs1_ + ibs1) * M_ * N_;
      const auto curr_dst = base_dst + curr_dst_offset;
      const auto curr_src2 = base_src2 + curr_dst_offset;
      const auto curr_src0_tmp = src0_tmp_ + (ibs0 * bs1_ + ibs1) * M_pad_ * K_pad_;
      const auto curr_src1_tmp = src1_tmp_ + (ibs0 * bs1_ + ibs1) * N_pad_ * K_pad_;
      const auto curr_sum_tmp = sum_tmp_ + (ibs0 * bs1_ + ibs1) * N_pad_ * max_cpy_nthr;

      // reorder src1 and compute sum over K
      for (dim_t i = 0; i < M_; i += 8) {
        for (dim_t j = 0; j < N_; j += 48) {
          jit_matmul_vnni_8xkx48_t::rt_data_t<dt_dst> rt_matmul;
          rt_matmul.src0 = curr_src0_tmp + i * K_;
          rt_matmul.src1 = curr_src1_tmp + j * K_;
          rt_matmul.bias = curr_sum_tmp + j;
          rt_matmul.src_b0 = curr_src2 + i * N_ + j;
          rt_matmul.dst = curr_dst + i * N_ + j;
          (j <= N_ - 48 ? *matmul_ker_ : *matmul_tile_n_ker_)(&rt_matmul);
        }
      }
    }
  return true;
}

bool matmul_vnni_p2031_p2013_k_t::execute(const std::vector<const void*>& rt_data) const {
  switch (dst_type) {
    case data_type::fp32:
      return execute_<float>(rt_data);
    case data_type::u8:
      return execute_<uint8_t>(rt_data);
    case data_type::s8:
      return execute_<int8_t>(rt_data);
    default:
      SPARSE_LOG(ERROR) << "";
      return false;
  }
}
}  // namespace jd
