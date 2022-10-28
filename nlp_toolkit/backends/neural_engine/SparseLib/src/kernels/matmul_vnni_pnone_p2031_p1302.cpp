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

#include "kernels/matmul_vnni_noperm_p2031_p1302.hpp"

namespace jd {

/**
 * Dimension details:
 *   src0: bs0 bs1 m   k ===========> bs0 bs1 m k
 *   src1: bs1 n   bs0 k ==perm2031=> bs0 bs1 k n
 *   dst:  bs1 n   bs0 m <=perm1302== bs0 bs1 m n
 *
 * Entries of op_desc_.attrs:
 *   No attrs supported yet!
 */

inline std::vector<std::vector<dim_t>> get_tensor_shapes(const std::vector<tensor_desc>& descs) {
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  return shapes;
}

// Part1: class matmul_vnni_noperm_p2031_p1302_kd_t

bool matmul_vnni_noperm_p2031_p1302_kd_t::init() {
  using dt = jd::data_type;
  if (!isa_available(avx512_core_vnni)) return false;
  auto& descs = op_desc_.tensor_descs();
  auto shapes = get_tensor_shapes(descs);

  for (auto mat : {ssd::SRC0, ssd::SRC1, ssd::SRC2, ssd::DST0})
    if (shapes[mat].size() != 4 && shapes[mat].size() != 0) {
      SPARSE_LOG(WARNING) << "All operand should be 4D matrix";
      return false;
    }

  std::vector<dim_t>& src0_perm_shape = shapes[ssd::SRC0];  // src0 perm none
  std::vector<dim_t> src1_perm_shape = {
      shapes[ssd::SRC1][2],  // bs0
      shapes[ssd::SRC1][0],  // bs1
      shapes[ssd::SRC1][3],  // K
      shapes[ssd::SRC1][1],  // N
  };
  std::vector<dim_t> dst0_perm_shape = {
      // reverse of 1302 is 2031
      shapes[ssd::DST0][2],  // bs0
      shapes[ssd::DST0][0],  // bs1
      shapes[ssd::DST0][3],  // M
      shapes[ssd::DST0][1],  // N
  };

  bool has_binary_add = !shapes[ssd::SRC2].empty();
  bool scaler_scale = shapes[ssd::SCALE0] == std::vector<dim_t>({1});

  bool is_supported = (op_desc_.kernel_prop() == kernel_prop::forward_inference) &&
                      is_any_of({dt::u8}, [&](const dt& a) { return descs[ssd::SRC0].dtype() == a; }) &&
                      is_any_of({dt::s8}, [&](const dt& a) { return descs[ssd::SRC1].dtype() == a; }) &&
                      is_any_of({dt::u8}, [&](const dt& a) { return descs[ssd::DST0].dtype() == a; }) &&
                      !has_binary_add && scaler_scale;

  if (!is_supported) return false;

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

  matmul_params_init(op_desc_);
  return true;
}

bool matmul_vnni_noperm_p2031_p1302_kd_t::matmul_params_init(const jd::operator_desc& op_desc) {
  auto& descs = op_desc_.tensor_descs();
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  auto attrs = op_desc_.attrs();

  jit_param_.M = shapes[ssd::SRC0][2];  // aka src0_perm_shape[2]
  jit_param_.K = shapes[ssd::SRC0][3];  // aka src0_perm_shape[3]
  jit_param_.N = shapes[ssd::SRC1][1];  // aka src1_perm_shape[3]
  jit_param_.batch = shapes[ssd::SRC0][0];

  // note that this kernel writes dst in col-major
  jit_param_.m_tile = 1;
  jit_param_.n_tile = 16;
  return true;
}

// Part2: class matmul_vnni_noperm_p2031_p1302_k_t

matmul_vnni_noperm_p2031_p1302_k_t::matmul_vnni_noperm_p2031_p1302_k_t(const std::shared_ptr<const kd_t>& kd)
    : kernel_t(kd),
      t_shapes_(get_tensor_shapes(kd->operator_desc().tensor_descs())),
      src0_perm_shape_(t_shapes_[ssd::SRC0]),  // src0 perm none
      src1_perm_shape_({
          t_shapes_[ssd::SRC1][2],
          t_shapes_[ssd::SRC1][0],
          t_shapes_[ssd::SRC1][3],
          t_shapes_[ssd::SRC1][1],
      }),
      dst1_perm_shape_({
          // reverse of 1302 is 2031
          t_shapes_[ssd::DST0][2],
          t_shapes_[ssd::DST0][0],
          t_shapes_[ssd::DST0][3],
          t_shapes_[ssd::DST0][1],
      }),
      M_(src0_perm_shape_[2]),
      K_(src0_perm_shape_[3]),
      N_(src1_perm_shape_[3]),
      bs0_(dst1_perm_shape_[0]),
      bs1_(dst1_perm_shape_[1]) {}

bool matmul_vnni_noperm_p2031_p1302_k_t::init() {
  auto& ker_param = derived_kd()->jit_param();
  auto ker = new jit_matmul_vnni_noperm_p2031_p1302_t(ker_param);
  if (ker == nullptr) return false;
  if (!ker->create_kernel()) return false;
  jit_ker_ = ker;
  return true;
}

bool matmul_vnni_noperm_p2031_p1302_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto base_src0 = static_cast<const uint8_t*>(rt_data[ssd::SRC0]);
  auto base_src1 = static_cast<const int8_t*>(rt_data[ssd::SRC1]);
  auto base_dst = const_cast<uint8_t*>(static_cast<const uint8_t*>(rt_data[ssd::DST0]));
  auto base_scale = static_cast<const float*>(rt_data[ssd::SCALE0]);

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

          // each jit kernel calculates 16xKx16, where K should be multiple of 32 (VNNI_R * dim_transpose)
          (*jit_ker_)(rt_param);
        }

  return true;
}
}  // namespace jd
