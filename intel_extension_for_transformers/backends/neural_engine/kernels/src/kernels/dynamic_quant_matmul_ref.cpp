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

#include "kernels/dynamic_quant_matmul_ref.hpp"

namespace jd {

enum prob_size_idx { batch, m, n, k };

using io = ssd::dynamic_quant_matmul_io::io;

dynamic_quant_matmul_ref_kd_t::dynamic_quant_matmul_ref_kd_t(const jd::operator_desc& op_desc)
    : kernel_desc_t(kernel_kind::dynamic_quant_matmul), op_desc_(op_desc) {
  prob_size_.resize(4);
  auto ts_desc = op_desc.tensor_descs();
  auto activation_shape = ts_desc[0].shape();
  auto weight_shape = ts_desc[1].shape();
  auto dst_shape = ts_desc[2].shape();
  prob_size_[batch] = activation_shape.size() == 3 ? activation_shape[0] : 1;
  prob_size_[m] = activation_shape.size() == 3 ? activation_shape[1] : activation_shape[0];
  prob_size_[n] = dst_shape[2];
  prob_size_[k] = weight_shape[0];
}

bool dynamic_quant_matmul_ref_kd_t::init() {
  auto ts_desc = op_desc_.tensor_descs();
  if (ts_desc[0].dtype() != data_type::s8 || ts_desc[1].dtype() != data_type::s8 || ts_desc[2].dtype() != data_type::s8)
    SPARSE_LOG(FATAL) << "activation, weight, dst should be s8 in dynamic_quant_matmul";
  SPARSE_LOG_IF(FATAL, prob_size_[k] % 4 != 0) << "k must pad with 4.";
  has_bias = (ts_desc.size() - 1) == static_cast<int>(io::BIAS) ? true : false;
  return true;
}

template <typename T, typename DST>
void gemm(T* a, T* b, DST* c, int B, int M, int N, int K) {
  for (int batch = 0; batch < B; batch++) {
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        DST tmp = 0;
        for (int k = 0; k < K; k++) {
          tmp += (DST) * (a + batch * M * K + i * K + k) * (DST) * (b + k * N + j);
        }
        *(c + batch * M * N + i * N + j) = tmp;
      }
    }
  }
}

void dequant_add_bias(float* mat, const float* scale_a, const float* scale_w, int b, int m, int n,
                      bool add_bias = false, const float* bias = nullptr) {
  for (int batch = 0; batch < b; batch++)
#pragma omp parallel for
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        mat[batch * m * n + i * n + j] = mat[batch * m * n + i * n + j] * scale_a[batch * m + i] * scale_w[j];
        if (add_bias) mat[batch * m * n + i * n + j] += bias[j];
      }
}

void get_dynamic_quant_scale(float* mat, float* scale, int b, int m, int n) {
  for (int batch = 0; batch < b; batch++) {
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
      float max = 0.f;
      for (int j = 0; j < n; j++)
        max = max < abs(mat[batch * m * n + i * n + j]) ? abs(mat[batch * m * n + i * n + j]) : max;
      scale[batch * m + i] = max / 127.f;
    }
  }
}

void s8_quant_mat(int8_t* dst_mat, const std::vector<float>& src_mat, float* scale, int b, int m, int n) {
  for (int batch = 0; batch < b; batch++) {
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        int ans = nearbyint(src_mat[batch * m * n + i * n + j] / scale[batch * m + i]);
        ans = ans > 127 ? 127 : ans;
        ans = ans < -128 ? -128 : ans;
        dst_mat[batch * m * n + i * n + j] = ans;
      }
    }
  }
}

int8_t* reorder_back(const int8_t* reorder_mat, int k, int n) {
  // step1. transpose back.
  auto pad_n = ceil_div(n, 16) * 16;
  int tile_k = 64;
  while (k % tile_k != 0) tile_k -= 4;
  auto trans_block_row = pad_n / 16;
  auto trans_block_col = k / tile_k;
  auto block_size = tile_k * 16;
  std::vector<int8_t> trans_back_buf(k * pad_n, 0);
#pragma omp parallel
  for (int n_loop = 0; n_loop < trans_block_row; n_loop++)
    for (int k_loop = 0; k_loop < trans_block_col; k_loop++)
      for (int i = 0; i < tile_k / 4; i++)
        for (int j = 0; j < 64; j++)
          trans_back_buf[k_loop * trans_block_row * block_size + n_loop * 64 + i * pad_n * 4 + j] =
              reorder_mat[n_loop * trans_block_col * block_size + k_loop * 64 + i * trans_block_col * 64 + j];
  // step2. reorder back.
  int8_t* reorder_back_mat = reinterpret_cast<int8_t*>(malloc(k * n));
  if (reorder_back_mat == nullptr) {
    SPARSE_LOG(FATAL) << "Failed to malloc for reorder_back_mat";
    return nullptr;
  }
#pragma omp parallel for
  for (int i = 0; i < k / 4; i++)
    for (int j = 0; j < n * 4; j++)
      *(reorder_back_mat + j % 4 * n + j / 4 + i * 4 * n) = trans_back_buf[i * 4 * pad_n + j];
  return reorder_back_mat;
}

bool dynamic_quant_matmul_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto prob_size = derived_kd()->get_prob_size();
  bool add_bias = derived_kd()->has_bias;
  auto l_mat = static_cast<const int8_t*>(rt_data[0]);
  auto r_mat = static_cast<const int8_t*>(rt_data[1]);
  int8_t* reorder_back_mat = reorder_back(r_mat, prob_size[k], prob_size[n]);
  auto dst_mat = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data[2]));
  auto scale_a = static_cast<const float*>(rt_data[3]);
  auto scale_w = static_cast<const float*>(rt_data[4]);
  auto scale_dst = reinterpret_cast<float*>(const_cast<void*>(rt_data[5]));
  auto* bias = add_bias ? static_cast<const float*>(rt_data[7]) : nullptr;
  std::vector<float> fp32_dst_mat(prob_size[batch] * prob_size[m] * prob_size[n], 0);
  gemm(l_mat, const_cast<const int8_t*>(reorder_back_mat), fp32_dst_mat.data(), prob_size[batch], prob_size[m],
       prob_size[n], prob_size[k]);
  dequant_add_bias(fp32_dst_mat.data(), scale_a, scale_w, prob_size[batch], prob_size[m], prob_size[n], add_bias, bias);
  get_dynamic_quant_scale(fp32_dst_mat.data(), scale_dst, prob_size[batch], prob_size[m], prob_size[n]);
  s8_quant_mat(dst_mat, fp32_dst_mat, scale_dst, prob_size[batch], prob_size[m], prob_size[n]);
  free(reorder_back_mat);
  return true;
}
}  // namespace jd
