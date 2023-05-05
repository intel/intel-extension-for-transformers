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

#include <algorithm>

#include "fp8.hpp"

namespace jd {
using io = ssd::matmul_io::io;

static const std::vector<dim_t> perm_plain4{0, 1, 2, 3};
static const std::vector<std::vector<std::vector<dim_t>>> perm_list = {
    {{2, 0, 3, 1}, {2, 0, 1, 3}, perm_plain4},
    {perm_plain4, {2, 0, 3, 1}, {1, 3, 0, 2}},
    {perm_plain4, {0, 1, 3, 2}, perm_plain4},
};
// Part1: class matmul_ref_kd_t

bool matmul_ref_kd_t::init() {
  auto shapes = op_desc_.tensor_shapes();
  std::transform(shapes.begin(), shapes.end(), shapes.begin(),
                 [](auto&& x) { return (x.size() != 1 || x[0] != 1) ? pre_pad1<dim_t>(4, x) : x; });

  for (auto mat : {io::SRC0, io::SRC1, io::SRC2, io::DST0}) {
    if (shapes[mat].size() != 4 && shapes[mat].size() != 0) {
      SPARSE_LOG(WARNING) << "All operand of transpose matmul op should be 4D matrix";
      return false;
    }
  }

  for (const std::vector<std::vector<jd::dim_t>>& perm : perm_list) {
    std::vector<dim_t> src0_perm_shape = {
        shapes[io::SRC0][perm[0][0]],
        shapes[io::SRC0][perm[0][1]],
        shapes[io::SRC0][perm[0][2]],
        shapes[io::SRC0][perm[0][3]],
    };
    std::vector<dim_t> src1_perm_shape = {
        shapes[io::SRC1][perm[1][0]],
        shapes[io::SRC1][perm[1][1]],
        shapes[io::SRC1][perm[1][2]],
        shapes[io::SRC1][perm[1][3]],
    };
    const std::vector<jd::dim_t> dst0_perm_inv = perm_inv(perm[2]);
    std::vector<dim_t> dst0_perm_shape = {
        shapes[io::DST0][dst0_perm_inv[0]],
        shapes[io::DST0][dst0_perm_inv[1]],
        shapes[io::DST0][dst0_perm_inv[2]],
        shapes[io::DST0][dst0_perm_inv[3]],
    };
    // check bs0 and bs1
    for (auto idx : {0, 1}) {
      for (auto shape_perm : {src0_perm_shape, src1_perm_shape}) {
        if (shape_perm[idx] != dst0_perm_shape[idx]) {
          SPARSE_LOG(WARNING)
              << "Cannot match perm as first 2 dimensions of all tensors after permutation are different";
          continue;
        }
      }
    }

    bool mkn_matches = src0_perm_shape[2] == dst0_perm_shape[2] &&  // M
                       src0_perm_shape[3] == src1_perm_shape[2] &&  // K
                       src1_perm_shape[3] == dst0_perm_shape[3];    // N
    if (!mkn_matches) {
      SPARSE_LOG(WARNING) << "Cannot match perm as M / K / N from src0 / src1 dst0 does not match";
      continue;
    }

    if (shapes.size() > io::SRC2 && !shapes[io::SRC2].empty()) {
      for (size_t idx = 0; idx < shapes[io::SRC2].size(); idx++) {
        if (shapes[io::SRC2][idx] != shapes[io::DST0][idx] && shapes[io::SRC2][idx] != 1) {
          SPARSE_LOG(ERROR) << "Shape SRC2 don't match DST0";
          return false;
        }
      }
    }
    if (shapes.size() > io::SCALE0 && !shapes[io::SCALE0].empty()) {
      if (shapes[io::SCALE0] != std::vector<dim_t>{1}) {
        SPARSE_LOG(ERROR) << "SCALE0 is not scaler";
        return false;
      }
    }
    if (shapes.size() > io::ZP0 && !shapes[io::ZP0].empty()) {
      if (shapes[io::ZP0] != std::vector<dim_t>{1}) {
        SPARSE_LOG(ERROR) << "ZP0 is not scaler";
        return false;
      }
    }

    perm_ptr_ = &perm;
    shape_[0] = src0_perm_shape[0];  // bs0
    shape_[1] = src0_perm_shape[1];  // bs1
    shape_[2] = src0_perm_shape[2];  // m
    shape_[3] = src0_perm_shape[3];  // k
    shape_[4] = src1_perm_shape[3];  // n
    return true;
  }
  return false;
}

// Part2: class matmul_ref_k_t
matmul_ref_k_t::matmul_ref_k_t(const std::shared_ptr<const kd_t>& kd)
    : kernel_t(kd),
      bs0_(derived_kd()->bs0()),
      bs1_(derived_kd()->bs1()),
      M_(derived_kd()->M()),
      K_(derived_kd()->K()),
      N_(derived_kd()->N()) {}

bool matmul_ref_k_t::init() { return true; }

bool matmul_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  using dt = jd::data_type;
  // configure alias
  const matmul_ref_kd_t& ref_kd = *derived_kd();
  auto& op_desc = ref_kd.get_operator_desc();
  auto& descs = op_desc.tensor_descs();
  auto& attrs = op_desc.attrs();
  auto shapes = op_desc.tensor_shapes();
  std::transform(shapes.begin(), shapes.end(), shapes.begin(),
                 [](auto&& x) { return (x.size() != 1 || x[0] != 1) ? pre_pad1<dim_t>(4, x) : x; });
  std::vector<jd::data_type> dtypes(descs.size());
  std::transform(descs.begin(), descs.end(), dtypes.begin(), [&](tensor_desc d) { return d.dtype(); });
  bool has_binary_add = shapes.size() > io::SRC2 && !shapes[io::SRC2].empty();

  const auto& left_dt = dtypes[io::SRC0];
  const auto& right_dt = dtypes[io::SRC1];
  const auto& dst_dt = dtypes[io::DST0];

  // alpha * src0 x src1 + beta * src2 = dst.
  // TBD(yi): change naming of matmul variables
  float alpha = 1.f, beta = 1.f, zp = 0.f;
  if (attrs.find("alpha") != attrs.end()) alpha = str_to_num<float>(attrs.at("alpha"));
  if (attrs.find("beta") != attrs.end()) beta = str_to_num<float>(attrs.at("beta"));
  if (shapes.size() > io::SCALE0 && !shapes[io::SCALE0].empty())
    alpha = static_cast<const float*>(rt_data[io::SCALE0])[0];
  if (shapes.size() > io::ZP0 && !shapes[io::ZP0].empty()) zp = static_cast<const float*>(rt_data[io::ZP0])[0];
  if (attrs.find("src0_scale") != attrs.end()) alpha /= str_to_num<float>(attrs.at("src0_scale"));
  if (attrs.find("src1_scale") != attrs.end()) alpha /= str_to_num<float>(attrs.at("src1_scale"));
  if (attrs.find("out_scale") != attrs.end()) alpha *= str_to_num<float>(attrs.at("out_scale"));

  // stride for dims index afte perm. e.g. first elements is always for bs0
  const std::vector<dim_t> left_dim{bs0_, bs1_, M_, K_};
  const std::vector<dim_t> right_dim{bs0_, bs1_, K_, N_};
  const std::vector<dim_t> dst_dim{bs0_, bs1_, M_, N_};
  const auto perm0 = perm()[0];
  const auto perm0left = apply_perm(left_dim, perm0);
  const std::vector<dim_t> left_perm_stride = dim2stride(perm0left);
  const std::vector<dim_t> right_perm_stride = dim2stride(apply_perm(right_dim, perm()[1]));
  const std::vector<dim_t> dst_perm_stride = dim2stride(apply_perm(dst_dim, perm_inv(perm()[2])));
  const std::vector<dim_t> left_stride = apply_perm(left_perm_stride, perm_inv(perm()[0]));
  const std::vector<dim_t> right_stride = apply_perm(right_perm_stride, perm_inv(perm()[1]));
  const std::vector<dim_t> dst_stride = apply_perm(dst_perm_stride, perm()[2]);

  // runtime data alias
  const auto left_data = rt_data[io::SRC0];
  const auto right_data = rt_data[io::SRC1];
  auto dst_data = const_cast<void*>(rt_data[io::DST0]);
  const auto badd_data = rt_data.size() > io::SRC2 ? rt_data[io::SRC2] : nullptr;

  // ptr alias
  auto left_fp32 = static_cast<const float*>(left_data);
  auto left_bf16 = static_cast<const bfloat16_t*>(left_data);
  auto left_u8 = static_cast<const uint8_t*>(left_data);
  auto left_s8 = static_cast<const int8_t*>(left_data);
  auto right_fp32 = static_cast<const float*>(right_data);
  auto right_u8 = static_cast<const uint8_t*>(right_data);
  auto right_s8 = static_cast<const int8_t*>(right_data);
  auto right_bf16 = static_cast<const bfloat16_t*>(right_data);
  auto dst_fp32 = static_cast<float*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);
  auto dst_s8 = static_cast<int8_t*>(dst_data);
  auto dst_bf16 = static_cast<bfloat16_t*>(dst_data);
  auto badd_fp32 = static_cast<const float*>(badd_data);
  auto badd_bf16 = static_cast<const bfloat16_t*>(badd_data);

  std::vector<jd::postop_attr> post_attr = op_desc.apply_postops_list();
  if (dst_dt != dt::fp32 && (post_attr.size() == 0 || post_attr.back().dt != dst_dt)) {
    post_attr.emplace_back(dst_dt, postop_type::eltwise, postop_alg::quantize, 0, 0, 1);
  }
  std::vector<dim_t> bias_stride = dim2step(shapes[io::SRC2]);
  // Computing the kernel
#pragma omp parallel for collapse(4)
  for (dim_t ibs0 = 0; ibs0 < bs0_; ++ibs0)
    for (dim_t ibs1 = 0; ibs1 < bs1_; ++ibs1)
      for (dim_t i = 0; i < M_; ++i)
        for (dim_t j = 0; j < N_; ++j) {
          float value = 0;
          dim_t dst_idx = ibs0 * dst_stride[0] + ibs1 * dst_stride[1] + i * dst_stride[2] + j * dst_stride[3];
          dim_t bias_idx = ibs0 * bias_stride[0] + ibs1 * bias_stride[1] + i * bias_stride[2] + j * bias_stride[3];
          for (dim_t k = 0; k < K_; ++k) {
            dim_t l_idx = ibs0 * left_stride[0] + ibs1 * left_stride[1] + i * left_stride[2] + k * left_stride[3];
            dim_t r_idx = ibs0 * right_stride[0] + ibs1 * right_stride[1] + k * right_stride[2] + j * right_stride[3];

            auto l_value = left_dt == dt::fp32   ? left_fp32[l_idx]
                           : left_dt == dt::u8   ? left_u8[l_idx]
                           : left_dt == dt::s8   ? left_s8[l_idx]
                           : left_dt == dt::bf16 ? bf16_to_fp32(left_bf16[l_idx])
                           : left_dt == dt::f8_e4m3
                               ? float8_base<FloatEncoding::E4M3>::convert_fp8_to_float(left_u8[l_idx])
                               : 0.f;

            auto r_value = right_dt == dt::fp32   ? right_fp32[r_idx]
                           : right_dt == dt::u8   ? right_u8[r_idx]
                           : right_dt == dt::s8   ? right_s8[r_idx]
                           : right_dt == dt::bf16 ? bf16_to_fp32(right_bf16[r_idx])
                           : right_dt == dt::f8_e4m3
                               ? float8_base<FloatEncoding::E4M3>::convert_fp8_to_float(right_u8[r_idx])
                               : 0.f;
            value += l_value * r_value;
          }
          float badd_value = 0;
          if (badd_data != nullptr && has_binary_add)
            badd_value = dtypes[io::SRC2] == dt::fp32   ? badd_fp32[bias_idx]
                         : dtypes[io::SRC2] == dt::bf16 ? bf16_to_fp32(badd_bf16[bias_idx])
                                                        : 0;
          value = apply_postop_list(alpha * value + beta * badd_value + zp, post_attr);

          // Quantize dst data
          if (dst_dt == dt::fp32) {
            dst_fp32[dst_idx] = value;
          } else if (dst_dt == dt::u8) {
            dst_u8[dst_idx] = static_cast<uint8_t>(value);
          } else if (dst_dt == dt::s8) {
            dst_s8[dst_idx] = static_cast<int8_t>(value);
          } else if (dst_dt == dt::bf16) {
            dst_bf16[dst_idx] = fp32_to_bf16(value);
          } else {
            SPARSE_LOG(FATAL) << "unsupported dst type";
          }
        }
  return true;
}

}  // namespace jd
