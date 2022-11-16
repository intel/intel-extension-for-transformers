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

#include "kernels/matmul_ref.hpp"

namespace jd {

inline std::vector<std::vector<dim_t>> get_tensor_shapes(const std::vector<tensor_desc>& descs) {
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  return shapes;
}

// Part1: class matmul_ref_kd_t

bool matmul_ref_kd_t::init() {
  auto& descs = op_desc_.tensor_descs();
  auto shapes = get_tensor_shapes(descs);

  for (auto mat : {ssd::SRC0, ssd::SRC1, ssd::SRC2, ssd::DST0})
    if (shapes[mat].size() != 4 && shapes[mat].size() != 0) {
      SPARSE_LOG(WARNING) << "All operand of transpose matmul op should be 4D matrix";
      return false;
    }

  std::vector<dim_t> src0_perm_shape = {
      shapes[ssd::SRC0][is_f32() ? 2 : 0],
      shapes[ssd::SRC0][is_f32() ? 0 : 1],
      shapes[ssd::SRC0][is_f32() ? 3 : 2],
      shapes[ssd::SRC0][is_f32() ? 1 : 3],
  };
  std::vector<dim_t> src1_perm_shape = {
      shapes[ssd::SRC1][is_f32() ? 2 : 2],
      shapes[ssd::SRC1][is_f32() ? 0 : 0],
      shapes[ssd::SRC1][is_f32() ? 1 : 3],
      shapes[ssd::SRC1][is_f32() ? 3 : 1],
  };
  std::vector<dim_t> dst0_perm_shape = {
      shapes[ssd::DST0][is_f32() ? 0 : 2],
      shapes[ssd::DST0][is_f32() ? 1 : 0],
      shapes[ssd::DST0][is_f32() ? 2 : 3],
      shapes[ssd::DST0][is_f32() ? 3 : 1],
  };
  if (is_f32()) {
    if (!shapes[ssd::SRC2].empty() && shapes[ssd::SRC2] != dst0_perm_shape) return false;
  } else {
    if (shapes[ssd::SCALE0] != std::vector<dim_t>({1})) return false;
  }

  for (auto idx : {0, 1}) {  // for bs0 and bs1
    for (auto shape_perm : {src0_perm_shape, src1_perm_shape}) {
      if (shape_perm[idx] != dst0_perm_shape[idx]) {
        SPARSE_LOG(WARNING) << "First 2 dimensions of all tensors after permutation should be the same";
        return false;
      }
    }
  }
  bool mkn_matches = src0_perm_shape[2] == dst0_perm_shape[2] &&  // M
                     src0_perm_shape[3] == src1_perm_shape[2] &&  // K
                     src1_perm_shape[3] == dst0_perm_shape[3];    // N
  if (!mkn_matches) {
    SPARSE_LOG(WARNING) << "M / K / N from src0 / src1 dst0 should match";
    return false;
  }

  return true;
}

//// Part2: class matmul_ref_k_t
bool matmul_ref_k_t::init() { return true; }

bool matmul_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  using dt = jd::data_type;
  // configure alias
  const matmul_ref_kd_t& ref_kd = *derived_kd();
  auto& op_desc = ref_kd.get_operator_desc();
  auto& descs = op_desc.tensor_descs();
  auto attrs = op_desc.attrs();
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  std::vector<jd::data_type> dtypes(descs.size());
  std::transform(descs.begin(), descs.end(), dtypes.begin(), [&](tensor_desc d) { return d.dtype(); });
  bool has_binary_add = ref_kd.is_f32() && !shapes[ssd::SRC2].empty();

  const auto& left_dt = dtypes[ssd::SRC0];
  const auto& right_dt = dtypes[ssd::SRC1];
  const auto& dst_dt = dtypes[ssd::DST0];

  // alpha * src0 x src1 + beta * src2 = dst.
  // TBD(yi): change naming of matmul variables
  float alpha = 1.f, beta = 1.f;
  const float* zp = nullptr;
  if (attrs["alpha"] != "") alpha = str_to_num<float>(attrs["alpha"]);
  if (attrs["beta"] != "") beta = str_to_num<float>(attrs["beta"]);
  if (!ref_kd.is_f32()) {
    const auto scale_data = rt_data[ssd::SCALE0];
    auto scale_f32 = static_cast<const float*>(scale_data);
    auto& scale_value = scale_f32[0];
    alpha = scale_value;
    zp = static_cast<const float*>(rt_data[ssd::ZP0]);
  }

  // stride for dims index afte perm. e.g. first elements is always for bs0
  std::vector<dim_t> left_perm_stride, right_perm_stride, dst_perm_stride;
  if (ref_kd.is_f32()) {
    left_perm_stride = {M_, K_ * bs0_ * M_, 1, bs0_ * M_};   // src0:     bs1 k   bs0 m
    right_perm_stride = {N_, K_ * bs0_ * N_, bs0_ * N_, 1};  // src1:     bs1 k   bs0 n
    dst_perm_stride = {bs1_ * M_ * N_, M_ * N_, N_, 1};      // src2/dst: bs0 bs1 m   n
  } else {
    left_perm_stride = {bs1_ * M_ * K_, M_ * K_, K_, 1};     // src0: bs0 bs1 m   k ===========> bs0 bs1 m k
    right_perm_stride = {K_, N_ * bs0_ * K_, 1, bs0_ * K_};  // src1: bs1 n   bs0 k ==perm2031=> bs0 bs1 k n
    dst_perm_stride = {M_, N_ * bs0_ * M_, 1, bs0_ * M_};    // dst:  bs1 n   bs0 m <=perm1302== bs0 bs1 m n
  }

  // runtime data alias
  const auto left_data = rt_data[ssd::SRC0];
  const auto right_data = rt_data[ssd::SRC1];
  auto dst_data = const_cast<void*>(rt_data[ssd::DST0]);
  const auto badd_data = ref_kd.is_f32() ? rt_data[ssd::SRC2] : nullptr;

  // ptr alias
  auto left_fp32 = static_cast<const float*>(left_data);
  auto left_u8 = static_cast<const uint8_t*>(left_data);
  auto right_fp32 = static_cast<const float*>(right_data);
  auto right_s8 = static_cast<const int8_t*>(right_data);
  auto dst_fp32 = static_cast<float*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);
  auto badd_fp32 = static_cast<const float*>(badd_data);

  // Computing the kernel
  // #pragma omp parallel for collapse(4)
  for (dim_t ibs0 = 0; ibs0 < bs0_; ++ibs0)
    for (dim_t ibs1 = 0; ibs1 < bs1_; ++ibs1)
      for (dim_t i = 0; i < M_; ++i)
        for (dim_t j = 0; j < N_; ++j) {
          float value = 0;
          dim_t dst_idx =
              ibs0 * dst_perm_stride[0] + ibs1 * dst_perm_stride[1] + i * dst_perm_stride[2] + j * dst_perm_stride[3];
          // #pragma omp simd
          for (dim_t k = 0; k < K_; ++k) {
            dim_t l_idx = ibs0 * left_perm_stride[0] + ibs1 * left_perm_stride[1] + i * left_perm_stride[2] +
                          k * left_perm_stride[3];
            dim_t r_idx = ibs0 * right_perm_stride[0] + ibs1 * right_perm_stride[1] + k * right_perm_stride[2] +
                          j * right_perm_stride[3];
            auto l_value = left_dt == dt::fp32 ? left_fp32[l_idx] : left_dt == dt::u8 ? left_u8[l_idx] : 0.f;
            auto r_value = right_dt == dt::fp32 ? right_fp32[r_idx] : right_dt == dt::s8 ? right_s8[r_idx] : 0.f;
            value += l_value * r_value;
          }
          float badd_value = 0;
          if (has_binary_add) badd_value = dtypes[ssd::SRC2] == dt::fp32 ? badd_fp32[dst_idx] : 0;
          value = alpha * value + beta * badd_value;

          // Quantize dst data
          if (dst_dt == dt::fp32) {
            dst_fp32[dst_idx] = value;
          } else if (dst_dt == dt::u8) {
            jd::postop_attr quantize{
                dt::u8, postop_type::eltwise, postop_alg::quantize,
                zp[0],  // alpha
                0,  // beta
                1,  // scale already applied in the previous step
            };
            float quantized_value = apply_postop_list(value, {quantize});
            dst_u8[dst_idx] = static_cast<uint8_t>(quantized_value);
          } else {
            SPARSE_LOG(FATAL) << "unsupported dst type";
          }
        }
  return true;
}

}  // namespace jd
