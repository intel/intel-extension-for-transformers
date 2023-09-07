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

#include "spmm_ref.hpp"

#define TH 4
#define TW 4
#define VEC 16

#define TILE_SIZE_M 4
#define TILE_SIZE_N 64

namespace jd {
bool spmm_ref_kd_t::init() {
  const auto& wei_desc = op_desc_.tensor_descs()[ssd::WEI];
  const auto& src_desc = op_desc_.tensor_descs()[ssd::SRC];
  const auto& dst_desc = op_desc_.tensor_descs()[ssd::DST];

  bool shape_matched = true;
  if (wei_desc.dtype() != data_type::fp32) {
    shape_matched = wei_desc.shape().size() == 2 && (src_desc.shape().size() == 2 || src_desc.shape().size() == 3) &&
                    (dst_desc.shape().size() == src_desc.shape().size()) &&
                    wei_desc.shape().back() != src_desc.shape()[src_desc.shape().size() - 2];
  } else {
    shape_matched = wei_desc.shape().front() != src_desc.shape().back();
  }

  if (shape_matched) {
    return false;
  }

  auto op_attrs = op_desc_.attrs();
  BM_ = str_to_num<dim_t>(op_attrs["micro_oc"]);   // block m
  if (BM_ == 0 && wei_desc.dtype() != data_type::fp32) {  // try to get optimized block size
    int cores = omp_get_num_procs();
    const dim_t blocks_n = N() / BN();

    BM_ = ceil_div(M(), ceil_div(cores, blocks_n));
    BM_ = ceil_div(BM_, TILE_SIZE_M) * TILE_SIZE_M;
    SPARSE_LOG(INFO) << "BM (micro output channel) automatically configured: BM=" << BM_;
  }
  SPARSE_LOG_IF(FATAL, BM_ % TILE_SIZE_M != 0) << "BM must be a multiple of TILE_SIZE_M\n";
  return true;
}

//// Part2: class spmm_ref_k_t
bool spmm_ref_k_t::init() { return true; }

bool spmm_ref_k_t::execute_s8_(const std::vector<const void*>& rt_data) const {
  const auto& dst_dt = dst_type();
  bool has_bias = derived_kd()->has_bias();
  auto attrs_map = derived_kd()->get_operator_desc().attrs();
  bool append_sum = (attrs_map["append_sum"] == "true");
  bool welford = (attrs_map["welford"] == "true");
  auto num_BN = N_ / BN_;
  std::vector<dim_t> left_stride = {K_, 1};
  std::vector<dim_t> right_stride = {BN_ * K_, BN_, 1};
  std::vector<dim_t> dst_stride = {BN_ * M_, BN_, 1};

  // runtime data alias
  const auto left_data = rt_data[ssd::WEI];
  const auto right_data = rt_data[ssd::SRC];
  const auto bias_data = static_cast<const int32_t*>(rt_data[ssd::BIAS]);
  auto dst_data = const_cast<void*>(rt_data[ssd::DST]);
  const auto scales_data = static_cast<const float*>(rt_data[ssd::SCALES]);

  // weight data
  auto left_s8 = static_cast<const int8_t*>(left_data);

  // src data
  auto right_u8 = static_cast<const uint8_t*>(right_data);

  // dst data
  auto dst_fp32 = static_cast<float*>(dst_data);  // ptr alias
  auto dst_s32 = static_cast<int32_t*>(dst_data);
  auto dst_s8 = static_cast<int8_t*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);

  // TODO(zhe1wang): add per channel support for post-op;
  auto postop_list = derived_kd()->get_operator_desc().apply_postops_list();
  if ((dst_dt == data_type::s8 || dst_dt == data_type::u8) && (postop_list.size() == 0)) {
    if (postop_list.size() == 0 || postop_list.back().op_alg != postop_alg::quantize)
      postop_list.emplace_back(dst_dt, postop_type::eltwise, postop_alg::quantize, 0., 0., 1.);
  }
// Computing the kernel
#pragma omp parallel for collapse(3)
  for (dim_t idx_mbs = 0; idx_mbs < num_BN; ++idx_mbs) {
    for (dim_t i = 0; i < M_; ++i) {
      for (dim_t j = 0; j < BN_; ++j) {
        float value = 0;  // Consistent with the actual precision (float or double) of cpu instructions.
#pragma omp simd
        for (dim_t k = 0; k < K_; ++k) {
          dim_t l_idx = i * left_stride[0] + k * left_stride[1];
          dim_t r_idx = idx_mbs * right_stride[0] + k * right_stride[1] + j * right_stride[2];
          auto left_k = left_s8[l_idx];
          auto right_k = right_u8[r_idx];
          value += left_k * right_k;
        }

        // Accumulate bias or post sum
        if (has_bias) {
          value += bias_data[i];
        }
        dim_t dst_idx = idx_mbs * dst_stride[0] + i * dst_stride[1] + j * dst_stride[2];
        dim_t scale_idx = i;
        if (dst_dt != data_type::s32) {
          value = value * scales_data[scale_idx];
        }
        if (append_sum) {
          value += dst_fp32[dst_idx];
        }
        value = apply_postop_list(value, postop_list);
        // Quantize dst data
        if (dst_dt == data_type::fp32) {
          dst_fp32[dst_idx] = static_cast<float>(value);
        } else if (dst_dt == data_type::s32) {
          dst_s32[dst_idx] = static_cast<int32_t>(value);
        } else if (dst_dt == data_type::s8) {
          dst_s8[dst_idx] = static_cast<int8_t>(value);
        } else if (dst_dt == data_type::u8) {
          dst_u8[dst_idx] = static_cast<uint8_t>(value);
        }
      }
    }
  }
  if (rt_data.size() >= ssd::DST_M2 + 1 && dst_dt == data_type::fp32 && welford) {
    auto mean_fp32 = static_cast<float*>(const_cast<void*>(rt_data[ssd::DST_M1]));
    auto var_fp32 = static_cast<float*>(const_cast<void*>(rt_data[ssd::DST_M2]));

    for (dim_t idx_mbs = 0; idx_mbs < num_BN; ++idx_mbs) {
      for (dim_t j = 0; j < BN_; ++j) {
        double sum = 0.f;
        for (dim_t i = 0; i < M_; ++i) {
          sum += dst_fp32[idx_mbs * BN_ * M_ + i * BN_ + j];
        }
        mean_fp32[idx_mbs * BN_ + j] = sum / M_;
      }
    }
    for (dim_t idx_mbs = 0; idx_mbs < num_BN; ++idx_mbs) {
      for (dim_t j = 0; j < BN_; ++j) {
        double M2 = 0.f;
        for (dim_t i = 0; i < M_; ++i) {
          M2 += pow(dst_fp32[idx_mbs * BN_ * M_ + i * BN_ + j] - mean_fp32[idx_mbs * BN_ + j], 2);
        }
        var_fp32[idx_mbs * BN_ + j] = M2 / M_;
      }
    }
  }
  return true;
}

bool spmm_ref_k_t::execute_bf16_(const std::vector<const void*>& rt_data) const {
  const auto& dst_dt = dst_type();
  auto num_BN = N_ / BN_;
  std::vector<dim_t> wei_stride = {K_, 1};
  std::vector<dim_t> src_stride = {BN_ * K_, BN_, 1};
  std::vector<dim_t> dst_stride = {BN_ * M_, BN_, 1};

  bool has_bias = derived_kd()->has_bias();
  auto attrs_map = derived_kd()->get_operator_desc().attrs();
  auto postop_list = derived_kd()->get_operator_desc().apply_postops_list();

  // runtime data alias
  const auto wei_data = static_cast<const bfloat16_t*>(rt_data[0]);
  const auto src_data = static_cast<const bfloat16_t*>(rt_data[1]);
  const auto bia_data = static_cast<const float*>(rt_data[2]);
  void* dst_data = const_cast<void*>(rt_data[3]);

  std::vector<float> float_dst_data(M_ * N_, 0);
  bfloat16_t* bf_dst_data = static_cast<bfloat16_t*>(dst_data);
  float* fp_dst_data = static_cast<float*>(dst_data);

  // Computing the kernelfor (int num_n = 0; num_n < NUM_BN; ++num_n) {
  for (int num_n = 0; num_n < num_BN; ++num_n) {
    for (int m = 0; m < M_; ++m) {
#pragma omp parallel for
      for (int n = 0; n < BN_; ++n) {
        auto dst_idx = num_n * dst_stride[0] + m * dst_stride[1] + n * dst_stride[2];
        for (int k = 0; k < K_; ++k) {
          float_dst_data[dst_idx] +=
              static_cast<float>(wei_data[m * wei_stride[0] + k * wei_stride[1]]) *
              static_cast<float>(src_data[num_n * src_stride[0] + k * src_stride[1] + n * src_stride[2]]);
        }
        if (has_bias) {
          float_dst_data[dst_idx] += bia_data[m];
        }
        float_dst_data[dst_idx] = apply_postop_list(float_dst_data[dst_idx], postop_list);
        if (dst_dt == data_type::bf16) {
          bf_dst_data[dst_idx] = float_dst_data[dst_idx];
        } else {
          fp_dst_data[dst_idx] = float_dst_data[dst_idx];
        }
      }
    }
  }
  return true;
}

bool spmm_ref_k_t::execute_f32_(const std::vector<const void*>& rt_data) const {
  const auto& ts_descs = derived_kd()->get_operator_desc().tensor_descs();
  const auto& postops_list = derived_kd()->get_operator_desc().apply_postops_list();
  const auto& wei_desc = ts_descs[ssd::WEI];
  const auto& src_desc = ts_descs[ssd::SRC];
  const auto& bias_desc = ts_descs[ssd::BIAS];
  int dims = wei_desc.shape().size();
  int M = src_desc.shape()[0];
  int K = wei_desc.shape()[0];
  int N = wei_desc.shape()[1];
  bool has_bias = !bias_desc.shape().empty();
  std::vector<int64_t> left_stride = {K, 1};
  std::vector<int64_t> right_stride = {N, 1};
  std::vector<int64_t> dst_stride = {N, 1};

  // runtime data alias
  const auto left_fp32 = static_cast<const float*>(rt_data[ssd::SRC]);
  const auto right_fp32 = static_cast<const float*>(rt_data[ssd::WEI]);
  const auto bias_fp32 = static_cast<const float*>(rt_data[ssd::BIAS]);
  auto dst_fp32 = static_cast<float*>(const_cast<void*>(rt_data[ssd::DST]));

  // Computing the kernel
  SPARSE_LOG_IF(FATAL, dims != 2) << "dim should be 2";
  for (int i = 0; i < M; ++i) {
#pragma omp parallel for
    for (int j = 0; j < N; ++j) {
      float value = 0;  // Consistent with the actual precision (float or double) of cpu instructions.
#pragma omp simd
      for (int k = 0; k < K; ++k) {
        auto left_k = left_fp32[i * left_stride[0] + k * left_stride[1]];
        auto right_k = right_fp32[k * right_stride[0] + j * right_stride[1]];
        value += left_k * right_k;
      }

      // Accumulate bias or post sum
      if (has_bias) {
        value += bias_fp32[j];
      }

      value = apply_postop_list(value, postops_list);

      dst_fp32[i * dst_stride[0] + j * dst_stride[1]] = static_cast<float>(value);
    }
  }
  return true;
}

bool spmm_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  switch (wei_type()) {
    case data_type::fp32:
      return execute_f32_(rt_data);
    case data_type::s8:
      return execute_s8_(rt_data);
    case data_type::bf16:
      return execute_bf16_(rt_data);
    default:
      SPARSE_LOG(ERROR) << "Unexpected dst_type: " << static_cast<uint8_t>(dst_type());
      break;
  }
  return false;
}

}  // namespace jd
