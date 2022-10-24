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

#include "kernels/matmul_avx512f_p2031_p2013.hpp"

namespace jd {

/**
 * Dimension details:
 *   src0:     bs1 k   bs0 m  ==perm2031=> bs0 bs1 m k
 *   src1:     bs1 k   bs0 n  ==perm2013=> bs0 bs1 k n
 *   src2/dst: bs0 bs1 m   n  ===========> bs0 bs1 m n
 *
 * Entries of op_desc_.attrs:
 *   alpha:  coefficient for matmul results
 *   beta:   coefficient for the operand of binary add
 *   m_tile: m-size of a tile in terms of #registers; default is 8
 *   n_tile: n-size of a tile in terms of #registers; default is 2
 */

inline std::vector<std::vector<dim_t>> get_tensor_shapes(const std::vector<tensor_desc>& descs) {
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  return shapes;
}

// Part1: class matmul_avx512f_p2031_p2013_kd_t

bool matmul_avx512f_p2031_p2013_kd_t::init() {
  using dt = jd::data_type;
  if (!isa_available(avx512_core)) return false;
  auto& descs = op_desc_.tensor_descs();
  auto shapes = get_tensor_shapes(descs);

  for (auto mat : {ssd::SRC0, ssd::SRC1, ssd::SRC2, ssd::DST0})
    if (shapes[mat].size() != 4 && shapes[mat].size() != 0) {
      LOG(WARNING) << "All operand should be 4D matrix";
      return false;
    }

  std::vector<dim_t> src0_perm_shape = {
      shapes[ssd::SRC0][2],
      shapes[ssd::SRC0][0],
      shapes[ssd::SRC0][3],
      shapes[ssd::SRC0][1],
  };

  std::vector<dim_t> src1_perm_shape = {
      shapes[ssd::SRC1][2],
      shapes[ssd::SRC1][0],
      shapes[ssd::SRC1][1],
      shapes[ssd::SRC1][3],
  };

  bool has_binary_add = !shapes[ssd::SRC2].empty();

  bool is_supported =
      (op_desc_.kernel_prop() == kernel_prop::forward_inference) &&
      is_any_of({dt::fp32}, [&](const dt& a) { return descs[ssd::SRC0].dtype() == a; }) &&
      is_any_of({dt::fp32}, [&](const dt& a) { return descs[ssd::SRC1].dtype() == a; }) &&
      is_any_of({dt::fp32}, [&](const dt& a) { return descs[ssd::DST0].dtype() == a; }) &&
      (!has_binary_add || is_any_of({dt::fp32}, [&](const dt& a) { return descs[ssd::SRC2].dtype() == a; }));

  if (!is_supported) return false;

  if (src0_perm_shape[3] != src1_perm_shape[2]) {
    LOG(WARNING) << "Skip as src0 k-dim (" << src0_perm_shape[3] << ") and src1 k-dim (" << src1_perm_shape[2]
                 << ") don't match!";
    return false;
  }

  for (auto idx : {0, 1}) {
    for (auto shape_perm : {src0_perm_shape, src1_perm_shape, shapes[ssd::SRC2]}) {
      if (shape_perm.empty()) continue;
      if (shape_perm[idx] != shapes[ssd::DST0][idx]) {
        LOG(WARNING) << "First 2 dimensions of all tensors after permutation should be the same";
        return false;
      }
    }
  }

  matmul_params_init(op_desc_);
  return true;
}

bool matmul_avx512f_p2031_p2013_kd_t::matmul_params_init(const jd::operator_desc& op_desc) {
  auto& descs = op_desc_.tensor_descs();
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  auto attrs = op_desc_.attrs();

  dim_t M = shapes[ssd::SRC0][3];  // aka src0_perm_shape[2]
  dim_t K = shapes[ssd::SRC0][1];  // aka src0_perm_shape[3]
  dim_t N = shapes[ssd::SRC1][3];  // aka src1_perm_shape[3]
  dim_t bs0 = shapes[ssd::DST0][0];
  jit_param_.M = M;
  jit_param_.K = K;
  jit_param_.N = N;
  jit_param_.batch = bs0;

  bool has_binary_add = !shapes[ssd::SRC2].empty();

  if (attrs["alpha"] != "") jit_param_.alpha = str_to_num<float>(attrs["alpha"]);
  LOG_IF(WARNING, jit_param_.alpha == 0.f)
      << "Alpha for matmul is set to 0 meaning that the base result will be discarded";

  if (has_binary_add) {
    if (attrs["beta"] != "") jit_param_.beta = str_to_num<float>(attrs["beta"]);
    LOG_IF(WARNING, has_binary_add && jit_param_.beta == 0.f)
        << "Beta for matmul is set to 0 meaning the binary-add does nothing";
  } else {
    jit_param_.beta = 0;  // set beta to 0 to avoid generate unnecessary asm ascode
  }

  int m_tile = str_to_num<int>(attrs["m_tile"]);
  int n_tile = str_to_num<int>(attrs["n_tile"]);
  if (m_tile > 0) jit_param_.m_tile = m_tile;
  if (n_tile > 0) jit_param_.n_tile = n_tile;

  return true;
}

// Part2: class matmul_avx512f_p2031_p2013_k_t

matmul_avx512f_p2031_p2013_k_t::matmul_avx512f_p2031_p2013_k_t(const std::shared_ptr<const kd_t>& kd)
    : kernel_t(kd),
      t_shapes_(get_tensor_shapes(kd->operator_desc().tensor_descs())),
      src0_perm_shape_({
          t_shapes_[ssd::SRC0][2],
          t_shapes_[ssd::SRC0][0],
          t_shapes_[ssd::SRC0][3],
          t_shapes_[ssd::SRC0][1],
      }),
      src1_perm_shape_({
          t_shapes_[ssd::SRC1][2],
          t_shapes_[ssd::SRC1][0],
          t_shapes_[ssd::SRC1][1],
          t_shapes_[ssd::SRC1][3],
      }),
      M_(src0_perm_shape_[2]),
      K_(src0_perm_shape_[3]),
      N_(src1_perm_shape_[3]),
      bs0_(t_shapes_[ssd::DST0][0]),
      bs1_(t_shapes_[ssd::DST0][1]) {}

bool matmul_avx512f_p2031_p2013_k_t::init() {
  auto& ker_param = derived_kd()->jit_param();
  auto ker = new jit_matmul_avx512f_p2031_p2013_t(ker_param);
  if (ker == nullptr) return false;
  if (!ker->create_kernel()) return false;
  jit_ker_ = ker;
  return true;
}

bool matmul_avx512f_p2031_p2013_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto base_src0 = static_cast<const float*>(rt_data[ssd::SRC0]);
  auto base_src1 = static_cast<const float*>(rt_data[ssd::SRC1]);
  auto base_dst = const_cast<float*>(static_cast<const float*>(rt_data[ssd::DST0]));
  auto base_src2 = static_cast<const float*>(rt_data[ssd::SRC2]);

#pragma omp parallel for collapse(2)
  for (dim_t ibs1 = 0; ibs1 < bs1_; ++ibs1)
    for (dim_t ibs0 = 0; ibs0 < bs0_; ++ibs0) {
      ssd::matmul_data_t rt_param;
      dim_t dst_offset = (ibs0 * bs1_ + ibs1) * M_ * N_;
      rt_param.dst = base_dst + dst_offset;
      rt_param.src2 = base_src2 + dst_offset;
      rt_param.src0 = base_src0 + ibs0 * M_ + bs0_ * ibs1 * M_ * K_;
      rt_param.src1 = base_src1 + ibs0 * N_ + bs0_ * ibs1 * N_ * K_;

      (*jit_ker_)(rt_param);
    }

  return true;
}
}  // namespace jd
