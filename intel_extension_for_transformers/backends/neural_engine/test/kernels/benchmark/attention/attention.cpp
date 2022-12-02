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

#include <algorithm>
#include "utils.hpp"
#include "attention/attention.hpp"

#include "kernels/matmul_types.hpp"

namespace {
void get_true_ip(const std::vector<jd::tensor_desc>& ts_descs, const std::vector<const void*>& rt_data,
                 const bool append_sum = false, const std::vector<jd::postop_attr>& postop_list = {}) {
  // shape configure alias
  const auto& wei_shape = ts_descs[jd::ssd::WEI].shape();
  const auto& wei_type = ts_descs[jd::ssd::WEI].dtype();
  const auto& src_type = ts_descs[jd::ssd::SRC].dtype();
  const auto& src_shape = ts_descs[jd::ssd::SRC].shape();
  const auto& dst_type = ts_descs[jd::ssd::DST].dtype();
  const auto& dst_shape = ts_descs[jd::ssd::DST].shape();
  SPARSE_LOG_IF(FATAL, src_shape.size() != 2 && src_shape.size() != 3 || src_shape.size() != dst_shape.size())
      << "Invalid shape";

  int oc = wei_shape[0];
  int ic = wei_shape[1];
  int micro_bs = src_shape.back();
  int num_mbs = src_shape.size() == 2 ? 1 : src_shape[0];

  const auto& left_dt = wei_type;
  const auto& right_dt = src_type;
  const auto& dst_dt = dst_type;
  bool has_bias = !ts_descs[jd::ssd::BIAS].shape().empty();
  std::vector<jd::dim_t> left_stride = {ic, 1};
  std::vector<jd::dim_t> right_stride = {micro_bs * ic, micro_bs, 1};
  std::vector<jd::dim_t> dst_stride = {micro_bs * oc, micro_bs, 1};

  // runtime data alias
  const auto left_data = rt_data[jd::ssd::WEI];
  const auto right_data = rt_data[jd::ssd::SRC];
  const auto bias_data = static_cast<const int32_t*>(rt_data[jd::ssd::BIAS]);
  auto dst_data = const_cast<void*>(rt_data[jd::ssd::DST]);
  const auto scales_data = static_cast<const float*>(rt_data[jd::ssd::SCALES]);

  // buffer data
  auto left_fp32 = static_cast<const float*>(left_data);  // ptr alias
  auto left_u8 = static_cast<const uint8_t*>(left_data);
  auto left_s8 = static_cast<const int8_t*>(left_data);

  auto right_fp32 = static_cast<const float*>(right_data);  // ptr alias
  auto right_u8 = static_cast<const uint8_t*>(right_data);
  auto right_s8 = static_cast<const int8_t*>(right_data);

  auto dst_fp32 = static_cast<float*>(dst_data);  // ptr alias
  auto dst_s32 = static_cast<int32_t*>(dst_data);
  auto dst_s8 = static_cast<int8_t*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);

// Computing the kernel
#pragma omp parallel for collapse(3)
  for (jd::dim_t idx_mbs = 0; idx_mbs < num_mbs; ++idx_mbs) {
    for (jd::dim_t i = 0; i < oc; ++i) {
      for (jd::dim_t j = 0; j < micro_bs; ++j) {
        float value = 0;  // Consistent with the actual precision (float or double) of cpu instructions.
#pragma omp simd
        for (jd::dim_t k = 0; k < ic; ++k) {
          jd::dim_t l_idx = i * left_stride[0] + k * left_stride[1];
          jd::dim_t r_idx = idx_mbs * right_stride[0] + k * right_stride[1] + j * right_stride[2];
          auto left_k = (left_dt == jd::data_type::fp32)
                            ? left_fp32[l_idx]
                            : ((left_dt == jd::data_type::u8) ? left_u8[l_idx]
                                                              : ((left_dt == jd::data_type::s8) ? left_s8[l_idx] : 0));
          auto right_k =
              (right_dt == jd::data_type::fp32)
                  ? right_fp32[r_idx]
                  : ((right_dt == jd::data_type::u8) ? right_u8[r_idx]
                                                     : ((right_dt == jd::data_type::s8) ? right_s8[r_idx] : 0));
          value += left_k * right_k;
        }

        // Accumulate bias or post sum
        if (has_bias) {
          value += bias_data[i];
        }
        jd::dim_t dst_idx = idx_mbs * dst_stride[0] + i * dst_stride[1] + j * dst_stride[2];
        jd::dim_t scale_idx = i;
        if (dst_dt != jd::data_type::s32) {
          value = value * scales_data[scale_idx];
        }
        if (append_sum) {
          value += dst_fp32[dst_idx];
        }
        value = jd::apply_postop_list(value, postop_list);
        if (dst_dt != jd::data_type::fp32) {
          value = jd::apply_postop_list(value, {{dst_dt, jd::postop_type::eltwise, jd::postop_alg::quantize, 0, 0, 1}});
        }
        // Quantize dst data
        if (dst_dt == jd::data_type::fp32) {
          dst_fp32[dst_idx] = static_cast<float>(value);
        } else if (dst_dt == jd::data_type::s32) {
          dst_s32[dst_idx] = static_cast<int32_t>(value);
        } else if (dst_dt == jd::data_type::s8) {
          dst_s8[dst_idx] = static_cast<int8_t>(value);
        } else if (dst_dt == jd::data_type::u8) {
          dst_u8[dst_idx] = static_cast<uint8_t>(value);
        }
      }
    }
  }
}

void get_true_softmax(const std::vector<jd::tensor_desc>& ts_descs, const std::vector<const void*>& rt_data,
                      const std::vector<jd::postop_attr>& postop_list = {}) {
  auto src_s8 = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data[0]));
  auto src_u8 = reinterpret_cast<uint8_t*>(const_cast<void*>(rt_data[0]));
  auto dst_dt = ts_descs[1].dtype();
  void* dst = const_cast<void*>(rt_data[1]);

  std::vector<jd::postop_attr> dequant_list = {postop_list.front()};
  std::vector<jd::postop_attr> quant_list;
  if (postop_list.back().op_alg == jd::postop_alg::quantize) quant_list.push_back(postop_list.back());
  auto src_tensor = ts_descs[0];
  auto src_dt = src_tensor.dtype();
  auto tensor_shape = src_tensor.shape();
  int row = src_tensor.reduce_rows();
  int col = tensor_shape.back();
  std::vector<float> float_dst_data(row * col, 0);
  for (int i = 0; i < row; i++) {
    // step1. find max
    float max = -256;
    for (int j = 0; j < col; j++) {
      int src_idx = i * col + j;
      if (src_dt == jd::data_type::s8) {
        max = static_cast<float>(src_s8[src_idx]) > max ? static_cast<float>(src_s8[src_idx]) : max;
      } else {
        max = static_cast<float>(src_u8[src_idx]) > max ? static_cast<float>(src_u8[src_idx]) : max;
      }
    }
    // get e^M
    max = jd::apply_postop_list(max, dequant_list);
    // step2. compute sum of exp
    float exp_sum = 0;
    for (int j = 0; j < col; j++) {
      float value = 0;
      if (src_dt == jd::data_type::s8) {
        value = jd::apply_postop_list(static_cast<float>(src_s8[i * col + j]), dequant_list);
      } else {
        value = jd::apply_postop_list(static_cast<float>(src_u8[i * col + j]), dequant_list);
      }
      value = jd::get_exp(value - max);
      float_dst_data[i * col + j] = value;
      exp_sum += value;
    }

    float scale = 1 / exp_sum;
    // step3. compute softmax
    if (dst_dt == jd::data_type::bf16) {
      for (int j = 0; j < col; j++)
        reinterpret_cast<jd::bfloat16_t*>(dst)[i * col + j] = jd::make_bf16(float_dst_data[i * col + j] * scale);
    } else if (dst_dt == jd::data_type::u8) {
      for (int j = 0; j < col; j++) {
        reinterpret_cast<uint8_t*>(dst)[i * col + j] =
            (uint8_t)apply_postop_list(float_dst_data[i * col + j] * scale, quant_list);
      }
    } else if (dst_dt == jd::data_type::s8) {
      for (int j = 0; j < col; j++)
        reinterpret_cast<int8_t*>(dst)[i * col + j] =
            (int8_t)apply_postop_list(float_dst_data[i * col + j] * scale, quant_list);
    }
  }
}

void get_true_matmul_p2031_p2013(const std::vector<jd::tensor_desc>& ts_descs, const std::vector<const void*>& rt_data,
                                 float alpha, float beta, const std::vector<jd::postop_attr>& postop_list = {}) {
  // configure alias
  std::vector<std::vector<jd::dim_t>> shapes(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), shapes.begin(), [&](jd::tensor_desc d) { return d.shape(); });
  std::vector<jd::data_type> dtypes(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), dtypes.begin(), [&](jd::tensor_desc d) { return d.dtype(); });

  const jd::dim_t M = shapes[jd::ssd::SRC0][3];  // aka src0_perm_shape[2]
  const jd::dim_t K = shapes[jd::ssd::SRC0][1];  // aka src0_perm_shape[3]
  const jd::dim_t N = shapes[jd::ssd::SRC1][3];  // aka src1_perm_shape[3]
  const jd::dim_t bs0 = shapes[jd::ssd::DST0][0];
  const jd::dim_t bs1 = shapes[jd::ssd::DST0][1];
  bool has_binary_add = !shapes[jd::ssd::SRC2].empty();

  // alpha * src0 x src1 + beta * src2 = dst.
  const auto& left_dt = dtypes[jd::ssd::SRC0];
  const auto& right_dt = dtypes[jd::ssd::SRC1];
  const auto& dst_dt = dtypes[jd::ssd::DST0];

  std::vector<jd::dim_t> left_stride = {K * bs0 * M, bs0 * M, M, 1};
  std::vector<jd::dim_t> right_stride = {K * bs0 * N, bs0 * N, N, 1};
  std::vector<jd::dim_t> dst_stride = {bs1 * M * N, M * N, N, 1};

  // runtime data alias
  const auto left_data = rt_data[jd::ssd::SRC0];
  const auto right_data = rt_data[jd::ssd::SRC1];
  const auto badd_data = rt_data[jd::ssd::SRC2];
  auto dst_data = const_cast<void*>(rt_data[jd::ssd::DST0]);

  // buffer data
  auto left_fp32 = static_cast<const float*>(left_data);  // ptr alias
  auto left_s8 = static_cast<const int8_t*>(left_data);
  auto right_fp32 = static_cast<const float*>(right_data);
  auto right_s8 = static_cast<const int8_t*>(right_data);
  auto dst_fp32 = static_cast<float*>(dst_data);
  auto dst_s8 = static_cast<int8_t*>(dst_data);
  auto badd_fp32 = static_cast<const float*>(badd_data);

  SPARSE_LOG_IF(ERROR, postop_list.empty() || postop_list.back().dt != jd::data_type::s8)
      << "s8/u8 output must has postop";
  // Computing the kernel
#pragma omp parallel for collapse(4)
  for (jd::dim_t ibs0 = 0; ibs0 < bs0; ++ibs0)
    for (jd::dim_t ibs1 = 0; ibs1 < bs1; ++ibs1)
      for (jd::dim_t i = 0; i < M; ++i)
        for (jd::dim_t j = 0; j < N; ++j) {
          float value = 0;
          jd::dim_t dst_idx = ibs0 * dst_stride[0] + ibs1 * dst_stride[1] + i * dst_stride[2] + j * dst_stride[3];
#pragma omp simd
          for (jd::dim_t k = 0; k < K; ++k) {
            /**
             *   src0:     bs1 k   bs0 m
             *   src1:     bs1 k   bs0 n
             *   src2/dst: bs0 bs1 m   n
             */
            jd::dim_t l_idx = ibs1 * left_stride[0] + k * left_stride[1] + ibs0 * left_stride[2] + i * left_stride[3];
            jd::dim_t r_idx =
                ibs1 * right_stride[0] + k * right_stride[1] + ibs0 * right_stride[2] + j * right_stride[3];
            auto l_value = left_dt == jd::data_type::fp32 ? left_fp32[l_idx]
                           : left_dt == jd::data_type::s8 ? left_s8[l_idx]
                                                          : 0;
            auto r_value = right_dt == jd::data_type::fp32 ? right_fp32[r_idx]
                           : right_dt == jd::data_type::s8 ? right_s8[r_idx]
                                                           : 0;
            value += l_value * r_value;
          }
          float badd_value = 0;
          if (has_binary_add) badd_value = dtypes[jd::ssd::SRC2] == jd::data_type::fp32 ? badd_fp32[dst_idx] : 0;
          value = alpha * value + beta * badd_value;
          value = jd::apply_postop_list(value, postop_list);
          // Quantize dst data
          if (dst_dt == jd::data_type::fp32) {
            dst_fp32[dst_idx] = static_cast<float>(value);
          } else if (dst_dt == jd::data_type::s8) {
            dst_s8[dst_idx] = static_cast<int8_t>(value);
          } else {
            LOG(FATAL) << "unsupported dst type";
          }
        }
}

void get_true_matmul_noperm_p2013_p1302(const std::vector<jd::tensor_desc>& ts_descs,
                                        const std::vector<const void*>& rt_data) {
  // configure alias
  std::vector<std::vector<jd::dim_t>> shapes(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), shapes.begin(), [&](jd::tensor_desc d) { return d.shape(); });
  std::vector<jd::data_type> dtypes(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), dtypes.begin(), [&](jd::tensor_desc d) { return d.dtype(); });

  const jd::dim_t M = shapes[jd::ssd::SRC0][2];  // aka src0_perm_shape[2]
  const jd::dim_t K = shapes[jd::ssd::SRC0][3];  // aka src0_perm_shape[3]
  const jd::dim_t N = shapes[jd::ssd::SRC1][1];  // aka src1_perm_shape[3]
  const jd::dim_t bs0 = shapes[jd::ssd::SRC0][0];
  const jd::dim_t bs1 = shapes[jd::ssd::SRC0][1];

  // scale * src0 x src1 = dst.

  const auto& left_dt = dtypes[jd::ssd::SRC0];
  const auto& right_dt = dtypes[jd::ssd::SRC1];
  const auto& dst_dt = dtypes[jd::ssd::DST0];

  /**
   *   src0: bs0 bs1 m   k ===========> bs0 bs1 m k
   *   src1: bs1 n   bs0 k ==perm2031=> bs0 bs1 k n
   *   dst:  bs1 n   bs0 m <=perm1302== bs0 bs1 m n
   */
  std::vector<jd::dim_t> left_stride = {bs1 * M * K, M * K, K, 1};
  std::vector<jd::dim_t> right_stride = {N * bs0 * K, bs0 * K, K, 1};
  std::vector<jd::dim_t> dst_stride = {N * bs0 * M, bs0 * M, M, 1};

  // runtime data alias
  const auto left_data = rt_data[jd::ssd::SRC0];
  const auto right_data = rt_data[jd::ssd::SRC1];
  auto dst_data = const_cast<void*>(rt_data[jd::ssd::DST0]);
  const auto scale_data = rt_data[jd::ssd::SCALE0];
  const auto zp_data = rt_data[jd::ssd::ZP0];

  // ptr alias
  auto left_u8 = static_cast<const uint8_t*>(left_data);
  auto right_s8 = static_cast<const int8_t*>(right_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);
  auto scale_f32 = static_cast<const float*>(scale_data);
  auto zp_f32 = static_cast<const float*>(zp_data);
  const auto scale_value = scale_f32[0];
  const auto zp_value = zp_f32[0];

  // Computing the kernel
#pragma omp parallel for collapse(4)
  for (jd::dim_t ibs0 = 0; ibs0 < bs0; ++ibs0)
    for (jd::dim_t ibs1 = 0; ibs1 < bs1; ++ibs1)
      for (jd::dim_t i = 0; i < M; ++i)
        for (jd::dim_t j = 0; j < N; ++j) {
          float value = 0;
          jd::dim_t dst_idx = ibs1 * dst_stride[0] + j * dst_stride[1] + ibs0 * dst_stride[2] + i * dst_stride[3];
#pragma omp simd
          for (jd::dim_t k = 0; k < K; ++k) {
            jd::dim_t l_idx = ibs0 * left_stride[0] + ibs1 * left_stride[1] + i * left_stride[2] + k * left_stride[3];
            jd::dim_t r_idx =
                ibs1 * right_stride[0] + j * right_stride[1] + ibs0 * right_stride[2] + k * right_stride[3];
            auto l_value = left_dt == jd::data_type::u8 ? left_u8[l_idx] : 0.f;
            auto r_value = right_dt == jd::data_type::s8 ? right_s8[r_idx] : 0.f;
            value += l_value * r_value;
          }

          // Quantize dst data
          if (dst_dt == jd::data_type::u8) {
            jd::postop_attr quantize{
                jd::data_type::u8,
                jd::postop_type::eltwise,
                jd::postop_alg::quantize,
                zp_value,         // zp
                0,                // beta
                1 / scale_value,  // scale
            };
            float quantized_value = jd::apply_postop_list(value, {quantize});
            dst_u8[dst_idx] = static_cast<uint8_t>(quantized_value);
          } else {
            LOG(FATAL) << "unsupported dst type";
          }
        }
}

template <typename T>
void prepare_sparse_data(T* vector_data, jd::dim_t rows, jd::dim_t cols, jd::dim_t blk_row, jd::dim_t blk_col,
                         float sparsity, uint32_t* seed = nullptr) {
  uint32_t default_seed = 123;
  if (seed == nullptr) seed = &default_seed;
  for (int i = 0; i < rows; i += blk_row) {
    for (int j = 0; j < cols; j += blk_col) {
      bool fill_zero = rand_r(seed) % 100 <= (sparsity * 100);
      if (fill_zero) {
        for (int bi = i; bi < i + blk_row; ++bi) {
          for (int bj = j; bj < j + blk_col; ++bj) {
            vector_data[bi * cols + bj] = 0;
          }
        }
      }
    }
  }
}

const void* make_data_obj(const std::vector<int64_t>& a_shape, const jd::data_type& a_dt,
                          const void* src_data = nullptr, bool is_clear = false, float sparsity = 0.f,
                          const std::vector<float>& ranges = {-10, 10}) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int bytes_size = elem_num * jd::type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else if (src_data != nullptr) {
    data_ptr = new uint8_t[bytes_size];
    memcpy(data_ptr, src_data, bytes_size);
  } else {
    if (a_dt == jd::data_type::fp32) {
      data_ptr = new float[elem_num];
      jd::init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::s32) {
      data_ptr = new int32_t[elem_num];
      jd::init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::u8) {
      data_ptr = new uint8_t[elem_num];
      jd::init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::s8) {
      data_ptr = new int8_t[elem_num];
      jd::init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
      if (sparsity != 0.f) {
        int8_t* s8_ptr = static_cast<int8_t*>(data_ptr);
        prepare_sparse_data(s8_ptr, a_shape[0], a_shape[1], 4, 1, sparsity);
      }
    }
  }
  return data_ptr;
}

std::vector<float> make_output_scale(jd::dim_t size, const std::vector<float>& ranges = {0, 10}) {
  std::vector<float> output_scale(size, 0);
  jd::init_vector(output_scale.data(), size, ranges[0], ranges[1]);
  return output_scale;
}

std::vector<float> make_output_zo(jd::dim_t size, const std::vector<float>& ranges = {-100, -1}) {
  std::vector<float> output_zo(size, 0);
  jd::init_vector(output_zo.data(), size, ranges[0], ranges[1]);
  return output_zo;
}

}  // namespace
namespace jd {

bench_res_t attention_bench::set_config(int argc, char** argv) {
  if (argc < ATTENTION_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";

    return {bench_status::wrong_input};
  }
  LOG(INFO) << "attention\n";
  head_num = str_to_num<int64_t>(argv[0]);
  head_size = str_to_num<int64_t>(argv[1]);
  batch_size = str_to_num<int64_t>(argv[2]);
  seq_len = str_to_num<int64_t>(argv[3]);
  sparsity = str_to_num<float>(argv[4]);
  dt_dst = strcmp(argv[5], "1") == 0 ? jd::data_type::fp32 : jd::data_type::s8;
  return {bench_status::success};
}

void attention_bench::get_true_data() {
  auto& op_desc = args.second.op_desc;
  auto& rt_data = args.second.rt_data;
  const auto& ts_descs = op_desc.tensor_descs();
  auto op_attrs = op_desc.attrs();
  std::vector<std::vector<jd::dim_t>> ts_shapes(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), ts_shapes.begin(), [](tensor_desc d) { return d.shape(); });
  std::vector<data_type> ts_types(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), ts_types.begin(), [](tensor_desc d) { return d.dtype(); });

  const jd::dim_t head_num = ts_shapes[jd::ssd::MERGE_DST][0];
  const jd::dim_t head_size = ts_shapes[jd::ssd::MERGE_DST][1];
  const jd::dim_t batch_size = ts_shapes[jd::ssd::MERGE_DST][2];
  const jd::dim_t seq_len = ts_shapes[jd::ssd::MERGE_DST][3];

  const void* q_weight_ptr = reinterpret_cast<void*>(str_to_num<uint64_t>(op_attrs["q_weight_ptr"]));
  const void* q_bias_ptr = reinterpret_cast<void*>(str_to_num<uint64_t>(op_attrs["q_bias_ptr"]));
  const void* q_scales_ptr = reinterpret_cast<void*>(str_to_num<uint64_t>(op_attrs["q_scales_ptr"]));
  const void* k_weight_ptr = reinterpret_cast<void*>(str_to_num<uint64_t>(op_attrs["k_weight_ptr"]));
  const void* k_bias_ptr = reinterpret_cast<void*>(str_to_num<uint64_t>(op_attrs["k_bias_ptr"]));
  const void* k_scales_ptr = reinterpret_cast<void*>(str_to_num<uint64_t>(op_attrs["k_scales_ptr"]));
  const void* v_weight_ptr = reinterpret_cast<void*>(str_to_num<uint64_t>(op_attrs["v_weight_ptr"]));
  const void* v_bias_ptr = reinterpret_cast<void*>(str_to_num<uint64_t>(op_attrs["v_bias_ptr"]));
  const void* v_scales_ptr = reinterpret_cast<void*>(str_to_num<uint64_t>(op_attrs["v_scales_ptr"]));
  const float softmax_in_zp = str_to_num<float>(op_attrs["softmax_in_zero_point"]);
  const float softmax_in_scale = str_to_num<float>(op_attrs["softmax_in_scale"]);
  const float softmax_out_zp = str_to_num<float>(op_attrs["softmax_out_zero_point"]);
  const float softmax_out_scale = str_to_num<float>(op_attrs["softmax_out_scale"]);

  const tensor_desc qkv_dst_desc =
      tensor_desc(ts_descs[jd::ssd::MERGE_SRC].shape(), jd::data_type::s8, ts_descs[jd::ssd::MERGE_SRC].ftype());

  int8_t* q_dst_data = new int8_t[ts_descs[jd::ssd::MERGE_SRC].size()];
  const std::vector<jd::tensor_desc> q_ip_desc = {
      ts_descs[jd::ssd::Q_WEIGHT],   // WEI
      ts_descs[jd::ssd::MERGE_SRC],  // SRC
      ts_descs[jd::ssd::Q_BIAS],     // BIAS
      qkv_dst_desc,                  // DST
      ts_descs[jd::ssd::Q_SCALES],   // SCALE
  };
  const std::vector<const void*> q_ip_rt = {
      q_weight_ptr, rt_data[jd::ssd::MERGE_SRC], q_bias_ptr, q_dst_data, q_scales_ptr,
  };

  get_true_ip(q_ip_desc, q_ip_rt);
  int8_t* k_dst_data = new int8_t[ts_descs[jd::ssd::MERGE_SRC].size()];
  get_true_ip(
      {
          ts_descs[jd::ssd::K_WEIGHT],   // WEI
          ts_descs[jd::ssd::MERGE_SRC],  // SRC
          ts_descs[jd::ssd::K_BIAS],     // BIAS
          qkv_dst_desc,                  // DST
          ts_descs[jd::ssd::K_SCALES],   // SCALE
      },
      {
          k_weight_ptr,
          rt_data[jd::ssd::MERGE_SRC],
          k_bias_ptr,
          k_dst_data,
          k_scales_ptr,
      });
  int8_t* v_dst_data = new int8_t[ts_descs[jd::ssd::MERGE_SRC].size()];
  get_true_ip(
      {
          ts_descs[jd::ssd::V_WEIGHT],   // WEI
          ts_descs[jd::ssd::MERGE_SRC],  // SRC
          ts_descs[jd::ssd::V_BIAS],     // BIAS
          qkv_dst_desc,                  // DST
          ts_descs[jd::ssd::V_SCALES],   // SCALE
      },
      {
          v_weight_ptr,
          rt_data[jd::ssd::MERGE_SRC],
          v_bias_ptr,
          v_dst_data,
          v_scales_ptr,
      });

  // Q mul K
  // alpha * src0 x src1 + beta * src2 = dst.
  float alpha = 1.f, beta = 1.f;
  if (op_attrs["alpha"] != "") alpha = str_to_num<float>(op_attrs["alpha"]);
  if (op_attrs["beta"] != "") beta = str_to_num<float>(op_attrs["beta"]);
  const tensor_desc qk_dst_desc = {ts_shapes[jd::ssd::Q_K_SRC2], jd::data_type::s8, format_type::ab};
  int8_t* qk_dst_data = new int8_t[qk_dst_desc.size()];
  get_true_matmul_p2031_p2013(
      {
          {ts_shapes[jd::ssd::MERGE_DST], jd::data_type::s8, format_type::ab},  // src0
          {ts_shapes[jd::ssd::MERGE_DST], jd::data_type::s8, format_type::ab},  // src1
          qk_dst_desc,                                                          // dst0
          ts_descs[jd::ssd::Q_K_SRC2],                                          // src2
      },
      {
          q_dst_data,
          k_dst_data,
          qk_dst_data,
          rt_data[jd::ssd::Q_K_SRC2],
      },
      alpha, beta,
      {
          {data_type::s8, postop_type::eltwise, postop_alg::quantize, softmax_in_zp, 0, softmax_in_scale},
      });

  // softmax
  uint8_t* qk_softmax_dst = new uint8_t[qk_dst_desc.size()];
  const tensor_desc softmax_dst_desc = {qk_dst_desc.shape(), jd::data_type::u8, qk_dst_desc.ftype()};
  postop_attr qk_softmax_i(data_type::s8, postop_type::eltwise, postop_alg::dequantize, softmax_in_zp, 0,
                           softmax_in_scale);
  postop_attr qk_softmax_o(data_type::u8, postop_type::eltwise, postop_alg::quantize, softmax_out_zp, 0,
                           softmax_out_scale);
  get_true_softmax(
      {
          qk_dst_desc,       // src
          softmax_dst_desc,  // dst
      },
      {
          qk_dst_data,
          qk_softmax_dst,
      },
      {qk_softmax_i, qk_softmax_o});

  // softmax(QK) mul V
  get_true_matmul_noperm_p2013_p1302(
      {
          softmax_dst_desc,                                                     // src0
          {ts_shapes[jd::ssd::MERGE_DST], jd::data_type::s8, format_type::ab},  // src1
          ts_descs[jd::ssd::MERGE_DST],                                         // dst0
          {{}, jd::data_type::fp32, format_type::ab},                           // src2
          ts_descs[jd::ssd::QK_V_OUTPUT_SCALES],                                // scale
          ts_descs[jd::ssd::QK_V_OUTPUT_ZERO_POINT],                            // zp
      },
      {
          qk_softmax_dst,
          v_dst_data,
          rt_data[jd::ssd::MERGE_DST],
          nullptr,
          rt_data[jd::ssd::QK_V_OUTPUT_SCALES],
          rt_data[jd::ssd::QK_V_OUTPUT_ZERO_POINT],
      });

  delete[] q_dst_data;
  delete[] k_dst_data;
  delete[] v_dst_data;
  delete[] qk_dst_data;
  delete[] qk_softmax_dst;
}

bool attention_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;

  get_true_data();

  auto buf1 = p.rt_data[jd::ssd::MERGE_DST];
  auto size1 = p.op_desc.tensor_descs()[jd::ssd::MERGE_DST].size();
  auto buf2 = q.rt_data[jd::ssd::MERGE_DST];
  auto size2 = q.op_desc.tensor_descs()[jd::ssd::MERGE_DST].size();
  // Should compare buffer with different addresses
  const auto& dst_type = p.op_desc.tensor_descs()[jd::ssd::MERGE_DST].dtype();
  if (dst_type == jd::data_type::fp32) {
    return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == jd::data_type::s32) {
    return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == jd::data_type::u8) {
    return compare_data<uint8_t>(buf1, size1, buf2, size2, 8e-3);
  } else if (dst_type == jd::data_type::s8) {
    return compare_data<int8_t>(buf1, size1, buf2, size2, 8e-3);
  }
  return false;
}

void attention_bench::gen_case() {
  op_attrs = {{"alpha", "0.125"},
              {"beta", "1"},
              {"softmax_in_zero_point", "140"},
              {"softmax_in_scale", "500"},
              {"softmax_out_zero_point", "0"},
              {"softmax_out_scale", "0.00324144"}};
  // Step 1: Construct runtime data for equivalent merged spmm
  const jd::dim_t ip_chanel = head_num * head_size;  // channel width for any of the three linear layer
  std::vector<tensor_desc> ts_descs_qkv_ip = {
      {{ip_chanel * 3, ip_chanel}, jd::data_type::s8, format_type::bsr},        // WEI
      {{ip_chanel, batch_size * seq_len}, jd::data_type::u8, format_type::ab},  // SRC
      {{ip_chanel * 3, 1}, jd::data_type::s32, format_type::ab},                // BIAS
      {{ip_chanel * 3, batch_size * seq_len}, dt_dst, format_type::ab},         // DST
      {{ip_chanel * 3, 1}, jd::data_type::fp32, format_type::ab},               // SCALE
  };
  std::vector<const void*> rt_data_qkv_ip(ts_descs_qkv_ip.size(), nullptr);
  for (size_t index = 0; index < ts_descs_qkv_ip.size(); ++index) {
    auto& tsd = ts_descs_qkv_ip[index];
    bool is_clear = index == ssd::DST;
    float data_sparsity = (index == ssd::WEI) ? sparsity : 0;
    auto ranges = (index == ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    const void* data_addr = make_data_obj(tsd.shape(), tsd.dtype(), nullptr, is_clear, data_sparsity, ranges);
    rt_data_qkv_ip[index] = data_addr;
  }

  tensor_desc attention_src_desc = {{ip_chanel, batch_size * seq_len}, jd::data_type::u8, format_type::ab};
  tensor_desc attention_dst_desc = {{head_num, head_size, batch_size, seq_len}, jd::data_type::u8, format_type::ab};
  tensor_desc q_weight_desc = {{ip_chanel, ip_chanel}, jd::data_type::s8, format_type::bsr};
  tensor_desc k_weight_desc = {{ip_chanel, ip_chanel}, jd::data_type::s8, format_type::bsr};
  tensor_desc v_weight_desc = {{ip_chanel, ip_chanel}, jd::data_type::s8, format_type::bsr};
  tensor_desc q_bias_desc = {{ip_chanel, 1}, jd::data_type::s32, format_type::ab};
  tensor_desc k_bias_desc = {{ip_chanel, 1}, jd::data_type::s32, format_type::ab};
  tensor_desc v_bias_desc = {{ip_chanel, 1}, jd::data_type::s32, format_type::ab};
  tensor_desc q_scales_desc = {{ip_chanel, 1}, jd::data_type::fp32, format_type::ab};
  tensor_desc k_scales_desc = {{ip_chanel, 1}, jd::data_type::fp32, format_type::ab};
  tensor_desc v_scales_desc = {{ip_chanel, 1}, jd::data_type::fp32, format_type::ab};
  tensor_desc reshape_desc = {{batch_size, seq_len}, jd::data_type::fp32, format_type::ab};
  tensor_desc q_k_src2_desc = {{batch_size, head_num, seq_len, seq_len}, jd::data_type::fp32, format_type::ab};
  tensor_desc q_k_scales_desc = {{}, jd::data_type::undef, format_type::undef};  // currently pass by static value
  tensor_desc qk_v_scales_desc = {{1}, jd::data_type::fp32, format_type::a};
  tensor_desc qk_v_zp_desc = {{1}, jd::data_type::fp32, format_type::a};

  ts_descs = {attention_src_desc, attention_dst_desc, q_weight_desc, k_weight_desc,   v_weight_desc, q_bias_desc,
              k_bias_desc,        v_bias_desc,        q_scales_desc, k_scales_desc,   v_scales_desc, reshape_desc,
              q_k_src2_desc,      q_k_scales_desc,    qk_v_zp_desc,  qk_v_scales_desc};

  std::vector<std::vector<jd::dim_t>> ts_shapes(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), ts_shapes.begin(), [](tensor_desc d) { return d.shape(); });
  std::vector<data_type> ts_types(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), ts_types.begin(), [](tensor_desc d) { return d.dtype(); });

  std::vector<const void*> rt_data_p(ssd::QK_V_OUTPUT_SCALES + 1, nullptr);
  std::vector<const void*> rt_data_q(ssd::QK_V_OUTPUT_SCALES + 1, nullptr);

  rt_data_p[jd::ssd::MERGE_SRC] =
      make_data_obj(ts_shapes[jd::ssd::MERGE_SRC], ts_types[jd::ssd::MERGE_SRC], rt_data_qkv_ip[jd::ssd::SRC]);
  rt_data_q[jd::ssd::MERGE_SRC] =
      make_data_obj(ts_shapes[jd::ssd::MERGE_SRC], ts_types[jd::ssd::MERGE_SRC], rt_data_qkv_ip[jd::ssd::SRC]);

  rt_data_p[jd::ssd::MERGE_DST] =
      make_data_obj(ts_shapes[jd::ssd::MERGE_DST], ts_types[jd::ssd::MERGE_DST], nullptr, true);
  rt_data_q[jd::ssd::MERGE_DST] =
      make_data_obj(ts_shapes[jd::ssd::MERGE_DST], ts_types[jd::ssd::MERGE_DST], nullptr, true);

  // for binary add of QxK matmul
  rt_data_p[jd::ssd::Q_K_SRC2] = make_data_obj(ts_shapes[jd::ssd::Q_K_SRC2], ts_types[jd::ssd::Q_K_SRC2], nullptr);
  rt_data_q[jd::ssd::Q_K_SRC2] =
      make_data_obj(ts_shapes[jd::ssd::Q_K_SRC2], ts_types[jd::ssd::Q_K_SRC2], rt_data_p[jd::ssd::Q_K_SRC2]);

  // for output scale & zp of QKxV matmul
  rt_data_p[jd::ssd::QK_V_OUTPUT_ZERO_POINT] = new float(113);
  rt_data_q[jd::ssd::QK_V_OUTPUT_ZERO_POINT] = new float(113);
  rt_data_p[jd::ssd::QK_V_OUTPUT_SCALES] = new float(.003f);
  rt_data_q[jd::ssd::QK_V_OUTPUT_SCALES] = new float(.003f);

  // Merge weight
  const size_t wei_bytes = ts_descs[jd::ssd::Q_WEIGHT].size() * type_size[ts_descs[jd::ssd::Q_WEIGHT].dtype()];
  char* q_weight_addr = new char[wei_bytes];
  char* k_weight_addr = new char[wei_bytes];
  char* v_weight_addr = new char[wei_bytes];
  const char* rt_data_qkv_ip_wei = static_cast<const char*>(rt_data_qkv_ip[jd::ssd::WEI]);
  memcpy(q_weight_addr, rt_data_qkv_ip_wei, wei_bytes);
  memcpy(k_weight_addr, rt_data_qkv_ip_wei + wei_bytes, wei_bytes);
  memcpy(v_weight_addr, rt_data_qkv_ip_wei + wei_bytes * 2, wei_bytes);
  op_attrs["q_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_weight_addr));
  op_attrs["k_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_weight_addr));
  op_attrs["v_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_weight_addr));

  // Merge bias
  const size_t bias_bytes = ts_descs[jd::ssd::Q_BIAS].size() * type_size[ts_descs[jd::ssd::Q_BIAS].dtype()];
  char* q_bias_addr = new char[bias_bytes];
  char* k_bias_addr = new char[bias_bytes];
  char* v_bias_addr = new char[bias_bytes];
  const char* rt_data_qkv_ip_bias = static_cast<const char*>(rt_data_qkv_ip[jd::ssd::BIAS]);
  memcpy(q_bias_addr, rt_data_qkv_ip_bias, bias_bytes);
  memcpy(k_bias_addr, rt_data_qkv_ip_bias + bias_bytes, bias_bytes);
  memcpy(v_bias_addr, rt_data_qkv_ip_bias + bias_bytes * 2, bias_bytes);
  op_attrs["q_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_bias_addr));
  op_attrs["k_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_bias_addr));
  op_attrs["v_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_bias_addr));

  // Merge scales
  const size_t scale_bytes = ts_descs[jd::ssd::Q_SCALES].size() * type_size[ts_descs[jd::ssd::Q_SCALES].dtype()];
  char* q_scales_addr = new char[scale_bytes];
  char* k_scales_addr = new char[scale_bytes];
  char* v_scales_addr = new char[scale_bytes];
  const char* rt_data_qkv_ip_scale = static_cast<const char*>(rt_data_qkv_ip[jd::ssd::SCALES]);
  memcpy(q_scales_addr, rt_data_qkv_ip_scale, scale_bytes);
  memcpy(k_scales_addr, rt_data_qkv_ip_scale + scale_bytes, scale_bytes);
  memcpy(v_scales_addr, rt_data_qkv_ip_scale + scale_bytes * 2, scale_bytes);
  op_attrs["q_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_scales_addr));
  op_attrs["k_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_scales_addr));
  op_attrs["v_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_scales_addr));

  float scale = make_output_scale(1)[0];
  float zero_point = make_output_zo(1)[0];
  operator_desc op_desc(kernel_kind::attention, kernel_prop::forward_inference, engine_kind::cpu, ts_descs, op_attrs);

  // Step 3: op_args_t testcase pair
  op_args_t op_args_p = {op_desc, rt_data_p};
  op_args_t op_args_q = {op_desc, rt_data_q};

  for (size_t index = 0; index < ts_descs_qkv_ip.size(); ++index) {
    delete[] reinterpret_cast<const char*>(rt_data_qkv_ip[index]);
  }

  args = {op_args_p, op_args_q};
}

}  // namespace jd
