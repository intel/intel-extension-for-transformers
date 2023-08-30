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

#include "spmm_vnni.hpp"

#define TH 4
#define TW 4
#define VEC 16

#define TILE_SIZE_M 4
#define TILE_SIZE_N 64

#define KERNEL_INIT_CHECK(f)                                         \
  if (!(f)) {                                                        \
    SPARSE_LOG(ERROR) << "Spmm VNNI kernel requires `" << #f << "`"; \
    return false;                                                    \
  }

namespace jd {
//// Part1: class spmm_vnni_kd_t

void auto_blocking(dim_t& BM, dim_t BN, const dim_t M, const dim_t N) {  // NOLINT
  if (BM > M) {
    BM = M;
  } else if (BM <= 0) {  // try to get optimized block size
    int cores = omp_get_num_procs();
    const dim_t blocks_n = N / BN;

    BM = ceil_div(M, ceil_div(cores, blocks_n));
    BM = ceil_div(BM, TILE_SIZE_M) * TILE_SIZE_M;  // round to a multiple of 4
    SPARSE_LOG(INFO) << "BM (micro output channel) automatically configured: BM=" << BM;
  }
}

/**
 * Entries of op_desc_.attrs:
 *   sparse_ptr: pointer of the sparse data
 *   append_sum: if the kenrel add the result to the destination tensor instead of overwriting it. Set to "true"
 *               to enable.
 *   tile_n: n-size of a tile in terms of #registers; default is 4
 *   sub_func: use -1 / 0 / 1 / 2 to specify sub_func folding level
 *    -1. use max available value
 *     0. no subfunction
 *     1. subfunction for dense loading & sparse loading & tile product
 *     2. use cmp/jp to replace subfunction calls
 *   micro_oc: m-size of a block; default is the whole M
 */

// Part1: class spmm_vnni_kd_t

bool spmm_vnni_kd_t::init() {
  if (!isa_available(avx512_core_vnni)) return false;

  const auto& wei_desc = op_desc_.tensor_descs()[ssd::WEI];
  const auto& src_desc = op_desc_.tensor_descs()[ssd::SRC];
  const auto& bias_desc = op_desc_.tensor_descs()[ssd::BIAS];
  const auto& dst_desc = op_desc_.tensor_descs()[ssd::DST];
  bool has_bias = !bias_desc.shape().empty();
  bool is_supported =
      (op_desc_.kernel_prop() == kernel_prop::forward_inference) &&
      is_any_of({data_type::s8, data_type::fp32}, [&](const data_type& a) { return wei_desc.dtype() == a; }) &&
      is_any_of({data_type::u8, data_type::fp32}, [&](const data_type& a) { return src_desc.dtype() == a; }) &&
      (!has_bias ||
       is_any_of({data_type::s32, data_type::fp32}, [&](const data_type& a) { return bias_desc.dtype() == a; })) &&
      is_any_of({data_type::s8, data_type::u8, data_type::fp32},
                [&](const data_type& a) { return dst_desc.dtype() == a; });
  if (!is_supported) {
    return false;
  }
  bool shape_matched = wei_desc.shape().size() == 2 && (src_desc.shape().size() == 2 || src_desc.shape().size() == 3) &&
                       (dst_desc.shape().size() == src_desc.shape().size()) &&
                       wei_desc.shape().back() != src_desc.shape()[src_desc.shape().size() - 2];
  if (shape_matched) {
    return false;
  }

  auto op_attrs = op_desc_.attrs();
  BM_ = str_to_num<dim_t>(op_attrs["micro_oc"]);  // block m
  auto_blocking(BM_, BN(), M(), N());
  SPARSE_LOG_IF(FATAL, BM_ % TILE_SIZE_M != 0) << "BM must be a multiple of TILE_SIZE_M";
  if (op_attrs["welford"] == "true") {
    KERNEL_INIT_CHECK(op_desc_.tensor_descs().size() > ssd::DST_M2);
    for (size_t welford_idx : {ssd::DST_M1, ssd::DST_M2}) {
      auto& ds_src = op_desc_.tensor_descs()[ssd::SRC].shape();
      KERNEL_INIT_CHECK(op_desc_.tensor_descs()[welford_idx].dtype() == data_type::fp32);
      if (op_desc_.tensor_descs()[welford_idx].shape().size() != 0) {
        if (ds_src.size() == 3) {
          KERNEL_INIT_CHECK((op_desc_.tensor_descs()[welford_idx].shape() == std::vector<dim_t>{ds_src[0], ds_src[2]}));
        } else {
          KERNEL_INIT_CHECK(op_desc_.tensor_descs()[welford_idx].shape() == std::vector<dim_t>{N()});
        }
      }
    }
    apply_welford_ = true;
  }
  spmm_params_init();
  return true;
}

bool spmm_vnni_kd_t::spmm_params_init() {
  auto op_attrs = op_desc_.attrs();
  const uint64_t data_addr = str_to_num<uint64_t>(op_attrs["sparse_ptr"]);
  bsr_data_t<int8_t>* bsr_data = reinterpret_cast<bsr_data_t<int8_t>*>(data_addr);

  ssd::subfunc_level sub_func = ssd::subfunc_level::subfunc_level_MAX;
  if (op_attrs["sub_func"].length() != 0) {
    sub_func = static_cast<ssd::subfunc_level>(atoi(op_attrs["sub_func"].c_str()));
  }

  dim_t num_mblock = ceil_div(M(), BM());
  params_.resize(num_mblock);
  SPARSE_LOG_IF(FATAL, bsr_data->block_size().size() != 2 || bsr_data->block_size()[0] != params_[0].blocksize[0] ||
                           bsr_data->block_size()[1] != params_[0].blocksize[1])
      << "different block sizes between sparse encoding and param";
  SPARSE_LOG_IF(FATAL,
                op_attrs["append_sum"] != "true" && op_attrs["append_sum"] != "false" && op_attrs["append_sum"] != "")
      << "append_sum must only be true/false";
  int tile_w = atoi(op_attrs["tile_n"].c_str());
  if (tile_w == 0) {
    tile_w = 4;
    while (BN() % (tile_w * 16) != 0) tile_w--;
  }

  for (int i = 0, im_start = 0; i < num_mblock; ++i, im_start += BM()) {
    params_[i].BN = BN();
    params_[i].BM = std::min(BM(), M() - im_start);
    params_[i].has_bias = has_bias();
    params_[i].append_sum = op_attrs["append_sum"] == "true";
    params_[i].output_type = dst_type();
    params_[i].tile_w = tile_w;
    params_[i].sub_func = sub_func;
    params_[i].im_start = im_start;
    params_[i].indptr = bsr_data->indptr();
    params_[i].indices = bsr_data->indices();
    params_[i].weight = bsr_data->data().data();
    params_[i].postop_attrs = op_desc_.apply_postops_list();
    params_[i].welford = op_attrs["welford"] == "true";
  }
  return true;
}

// Part2: class spmm_vnni_k_t
bool spmm_vnni_k_t::init() {
  dim_t num_mblock = ceil_div(M_, BM_);
  jit_spmm_kers_.resize(num_mblock);
  for (int i = 0; i < num_mblock; ++i) {
    jit_spmm_vnni_t* ker = new jit_spmm_vnni_t(derived_kd()->params()[i]);
    if (ker == nullptr) return false;
    if (!(ker->create_kernel())) return false;
    jit_spmm_kers_[i] = ker;
  }
  ssd::mean_var_reduce_param_t param{ceil_div(M_, BM_), M_, N_, BM_, BN_};
  jit_mean_var_reduce_kers_.resize(ceil_div(N_, 16));
  for (size_t i = 0; i < jit_mean_var_reduce_kers_.size(); ++i) {
    jit_mean_var_reduce_t* ker = new jit_mean_var_reduce_t(param);
    if (ker == nullptr) return false;
    if (!(ker->create_kernel())) return false;
    jit_mean_var_reduce_kers_[i] = ker;
  }

  if (derived_kd()->welford()) {
// alloc  M/BM x N
#ifndef WORKSPACE
    tmp_mem_mean_ = aligned_allocator_t<float>::allocate(ceil_div(M_, BM_) * N_);
    tmp_mem_var_ = aligned_allocator_t<float>::allocate(ceil_div(M_, BM_) * N_);
#endif
  }
  return true;
}

template <typename dst_t>
bool spmm_vnni_k_t::execute_(const std::vector<const void*>& rt_data) const {
  float* tmp_mem_mean = nullptr;
  float* tmp_mem_var = nullptr;
  if (derived_kd()->welford()) {
#ifdef WORKSPACE
    tmp_mem_mean = const_cast<float*>(static_cast<const float*>(rt_data[ssd::WORK_SPACE]));
    tmp_mem_var = const_cast<float*>(static_cast<const float*>(rt_data[ssd::WORK_SPACE])) + ceil_div(M_, BM_) * N_;
#else
    tmp_mem_mean = tmp_mem_mean_;
    tmp_mem_var = tmp_mem_var_;
#endif
  }
#pragma omp parallel for collapse(2)
  for (dim_t im = 0; im < M_; im += BM_) {
    for (dim_t in = 0; in < N_; in += BN_) {
      const jit_spmm_vnni_t* jit_impl = jit_spmm_kers_[im / BM_];
      ssd::vnni_data_t<dst_t> data;
      data.ptr_dense = static_cast<const uint8_t*>(rt_data[ssd::SRC]) + in * K_;
      data.ptr_bias = static_cast<const int32_t*>(rt_data[ssd::BIAS]) + im;
      data.ptr_scales = static_cast<const float*>(rt_data[ssd::SCALES]) + im;
      data.ptr_dst = const_cast<dst_t*>(static_cast<const dst_t*>(rt_data[ssd::DST])) + in * M_ + im * BN_;
      if (derived_kd()->welford()) {
        data.ptr_dst_m1 = tmp_mem_mean + in * ceil_div(M_, BM_) + im / BM_ * BN_;
        data.ptr_dst_m2 = tmp_mem_var + in * ceil_div(M_, BM_) + im / BM_ * BN_;
      }
      (*jit_impl)(&data);
    }
  }
  if (derived_kd()->welford()) {
    // int index = 0;
#pragma omp parallel for collapse(2)
    for (dim_t idx_mbs = 0; idx_mbs < N_ / BN_; ++idx_mbs) {
      for (dim_t j = 0; j < BN_; j += 16) {
        size_t index = (idx_mbs * BN_ + j) / 16;
        const jit_mean_var_reduce_t* jit_impl = jit_mean_var_reduce_kers_[index];
        ssd::mean_var_reduce_data_t data;
        data.mean_in = tmp_mem_mean + idx_mbs * BN_ * ceil_div(M_, BM_) + j;
        data.var_in = tmp_mem_var + idx_mbs * BN_ * ceil_div(M_, BM_) + j;
        data.mean_out = reinterpret_cast<float*>(const_cast<void*>(rt_data[ssd::DST_M1])) + idx_mbs * BN_ + j;
        data.var_out = reinterpret_cast<float*>(const_cast<void*>(rt_data[ssd::DST_M2])) + idx_mbs * BN_ + j;
        (*jit_impl)(&data);
      }
    }
  }
  return true;
}

bool spmm_vnni_k_t::execute(const std::vector<const void*>& rt_data) const {
  switch (dst_type()) {
    case data_type::fp32:
      return execute_<float>(rt_data);
    case data_type::s8:
      return execute_<int8_t>(rt_data);
    case data_type::u8:
      return execute_<uint8_t>(rt_data);
    default:
      SPARSE_LOG(ERROR) << "Unexpected dst_type: " << static_cast<uint8_t>(dst_type());
      break;
  }
  return false;
}

}  // namespace jd
