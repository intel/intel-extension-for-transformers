//  Copyright (c) 2023 Intel Corporation
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

#include "kernels/dynamic_quantize_mha_ref.hpp"

#include <algorithm>

namespace jd {
using io = ssd::dynamic_quantize_mha_io::io;
using dt = jd::data_type;
namespace {
inline std::vector<std::vector<dim_t>> get_tensor_shapes(const std::vector<tensor_desc>& descs) {
  std::vector<std::vector<dim_t>> shapes(io::dynamic_quantize_mha_io_MAX + 1);
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  return shapes;
}
inline std::vector<dt> get_tensor_dtypes(const std::vector<tensor_desc>& descs) {
  std::vector<dt> shapes(io::dynamic_quantize_mha_io_MAX + 1);
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.dtype(); });
  return shapes;
}
}  // namespace

#define KERNEL_INIT_CHECK(f)                                                     \
  if (!(f)) {                                                                    \
    SPARSE_LOG(ERROR) << "Dynamic _quantize ref kernel requires `" << #f << "`"; \
    return false;                                                                \
  }
bool jd::dynamic_quantize_mha_ref_kd_t::init() {
  const auto& descs = op_desc_.tensor_descs();
  const auto shapes = get_tensor_shapes(descs);
  const auto dtypes = get_tensor_dtypes(descs);

  const auto batch_size = shapes[io::Q][0];
  const auto head_num = shapes[io::Q][2];
  const auto M = shapes[io::Q][1];
  const auto head_size = shapes[io::Q][3];
  const auto N = shapes[io::K][1];

  KERNEL_INIT_CHECK((batch_size > 0 || shapes[io::BATCH_SIZE] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((head_num > 0 || shapes[io::HEAD_NUM] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((M > 0 || shapes[io::HEAD_SIZE] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((head_size > 0 || shapes[io::M] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((N > 0 || shapes[io::N] == std::vector<dim_t>{1}));

  KERNEL_INIT_CHECK((shapes[io::Q] == std::vector<dim_t>{batch_size, M, head_num, head_size}));
  KERNEL_INIT_CHECK((shapes[io::K] == std::vector<dim_t>{batch_size, N, head_num, head_size}));
  KERNEL_INIT_CHECK((shapes[io::V] == std::vector<dim_t>{batch_size, N, head_num, head_size}));
  KERNEL_INIT_CHECK((shapes[io::DST] == std::vector<dim_t>{batch_size, M, head_num, head_size}));
  KERNEL_INIT_CHECK((shapes[io::MASK] == std::vector<dim_t>{batch_size, N}));

  KERNEL_INIT_CHECK((shapes[io::Q_SCALE] == std::vector<dim_t>{batch_size, M}));
  KERNEL_INIT_CHECK((shapes[io::K_SCALE] == std::vector<dim_t>{batch_size, N}));
  KERNEL_INIT_CHECK((shapes[io::V_SCALE] == std::vector<dim_t>{batch_size, N}));
  KERNEL_INIT_CHECK((shapes[io::DST_SCALE] == std::vector<dim_t>{batch_size, M}));

  // currently only support s8
  KERNEL_INIT_CHECK((shapes[io::Q_ZP].empty()));
  KERNEL_INIT_CHECK((shapes[io::K_ZP].empty()));
  KERNEL_INIT_CHECK((shapes[io::V_ZP].empty()));
  KERNEL_INIT_CHECK((shapes[io::DST_ZP].empty()));

  // dtype
  KERNEL_INIT_CHECK(is_all_of(
      {
          dtypes[io::Q],
          dtypes[io::K],
          dtypes[io::V],
          dtypes[io::DST],
      },
      [&](const dt t) { return t == dt::s8; }));
  KERNEL_INIT_CHECK(is_all_of(
      {
          dtypes[io::MASK],
          dtypes[io::Q_SCALE],
          dtypes[io::K_SCALE],
          dtypes[io::V_SCALE],
          dtypes[io::DST_SCALE],
      },
      [&](const dt t) { return t == dt::fp32; }));

  return true;
}
#undef KERNEL_INIT_CHECK

dynamic_quantize_mha_ref_k_t::dynamic_quantize_mha_ref_k_t(const std::shared_ptr<const kernel_desc_t>& kd)
    : kernel_t(kd),
      t_shapes_(get_tensor_shapes(derived_kd()->get_operator_desc().tensor_descs())),
      batch_size_(t_shapes_[io::Q][0]),
      head_num_(t_shapes_[io::Q][2]),
      M_(t_shapes_[io::Q][1]),
      head_size_(t_shapes_[io::Q][3]),
      N_(t_shapes_[io::K][1]) {}

bool dynamic_quantize_mha_ref_k_t::init() { return true; }

bool dynamic_quantize_mha_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  const auto src_q = reinterpret_cast<const int8_t*>(rt_data[io::Q]);
  const auto src_k = reinterpret_cast<const int8_t*>(rt_data[io::K]);
  const auto mask = reinterpret_cast<const float*>(rt_data[io::MASK]);
  const auto src_v = reinterpret_cast<const int8_t*>(rt_data[io::V]);
  const auto dst = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data[io::DST]));
  // const auto tmp_mem = reinterpret_cast<char*>(const_cast<void*>(rt_data[io::TMP]));
  const auto q_scale = reinterpret_cast<const float*>(rt_data[io::Q_SCALE]);
  const auto k_scale = reinterpret_cast<const float*>(rt_data[io::K_SCALE]);
  const auto v_scale = reinterpret_cast<const float*>(rt_data[io::V_SCALE]);
  const auto dst_scale = reinterpret_cast<float*>(const_cast<void*>(rt_data[io::DST_SCALE]));

  const auto batch_size = batch_size_ > 0 ? batch_size_ : reinterpret_cast<const int32_t*>(rt_data[io::BATCH_SIZE])[0];
  const auto head_num = head_num_ > 0 ? head_num_ : reinterpret_cast<const int32_t*>(rt_data[io::HEAD_NUM])[0];
  const auto head_size = head_size_ > 0 ? head_size_ : reinterpret_cast<const int32_t*>(rt_data[io::HEAD_SIZE])[0];
  const auto M = M_ > 0 ? M_ : reinterpret_cast<const int32_t*>(rt_data[io::M])[0];
  const auto N = N_ > 0 ? N_ : reinterpret_cast<const int32_t*>(rt_data[io::N])[0];

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < batch_size; ++ibs) {
    for (int i = 0; i < M; ++i) {
      const auto batch_q = src_q + ibs * M * head_size * head_num;
      const auto batch_k = src_k + ibs * N * head_size * head_num;
      const auto batch_v = src_v + ibs * N * head_size * head_num;
      const auto batch_dst = dst + ibs * M * head_size * head_num;
      const auto batch_q_scale = q_scale + ibs * M;
      const auto batch_k_scale = k_scale + ibs * N;
      const auto batch_v_scale = v_scale + ibs * N;
      const auto batch_dst_scale = dst_scale + ibs * M;
      const auto batch_mask = mask + ibs * N;

      const auto axv_f32 = std::unique_ptr<float[]>(new float[head_size * head_num]);
      float axv_f32_absmax = 0;
      for (int ihn = 0; ihn < head_num; ++ihn) {
        const auto mm_softmax = std::unique_ptr<float[]>(new float[N]);
        float expsum = 0.f;
        for (int j = 0; j < N; ++j) {
          mm_softmax[j] = batch_mask[j];  // binary add mask
          for (int k = 0; k < head_size; ++k)
            mm_softmax[j] += (batch_q_scale[i] * batch_q[i * head_size * head_num + ihn * head_size + k]) *
                             (batch_k_scale[j] * batch_k[j * head_size * head_num + ihn * head_size + k]);
          mm_softmax[j] = std::exp(mm_softmax[j]);
          expsum += mm_softmax[j];
        }
        const auto mm_softmax_u8 = std::unique_ptr<uint8_t[]>(new uint8_t[N]);
#pragma omp simd
        for (int j = 0; j < N; ++j)
          mm_softmax_u8[j] = static_cast<uint8_t>(std::roundf((mm_softmax[j] / expsum) * UINT8_MAX));

        for (int j = 0; j < head_size; ++j) {
          // requant v to simulate the AMX kernel
          const auto src_v_f32 = std::unique_ptr<float[]>(new float[N]);
          float src_v_f32_absmax = 0.f;
          for (int k = 0; k < N; ++k) {
            src_v_f32[k] = batch_v_scale[k] * batch_v[k * head_size * head_num + ihn * head_size + j];
            src_v_f32_absmax = std::max(src_v_f32_absmax, std::abs(src_v_f32[k]));
          }
          const auto src_v_requant_scale = src_v_f32_absmax / INT8_MAX;
          const auto src_v_requant_s8 = std::unique_ptr<int8_t[]>(new int8_t[N]);
#pragma omp simd
          for (int k = 0; k < N; ++k) src_v_requant_s8[k] = std::round(src_v_f32[k] / src_v_requant_scale);

          // a x v
          float axv_value = 0.f;
          for (int k = 0; k < N; ++k)
            axv_value += (1.f / UINT8_MAX * mm_softmax_u8[k]) * (src_v_requant_scale * src_v_requant_s8[k]);

          // update axv_f32
          axv_f32[ihn * head_size + j] = axv_value;
          axv_f32_absmax = std::max(axv_f32_absmax, std::abs(axv_value));
        }
      }

      // dynamic _quantize
      batch_dst_scale[i] = axv_f32_absmax / INT8_MAX;
      for (int j = 0; j < head_size * head_num; ++j)
        batch_dst[i * head_size * head_num + j] = std::roundf(axv_f32[j] / batch_dst_scale[i]);
    }
  }
  return true;
}

}  // namespace jd
