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

#include "dynamic_quant_matmul.hpp"
namespace jd {

using io = exposed_enum::dynamic_quant_matmul::io;

enum prob_size_idx { batch, m, n, k };

dynamic_quant_matmul_kd_t::dynamic_quant_matmul_kd_t(const operator_desc& op_desc)
    : kernel_desc_t(kernel_kind::dynamic_quant_matmul), op_desc_(op_desc) {
  prob_size_.resize(4);
  auto ts_desc = op_desc.tensor_descs();
  auto op_attrs = op_desc.attrs();
  auto activation_shape = ts_desc[0].shape();
  auto weight_shape = ts_desc[1].shape();
  auto dst_shape = ts_desc[2].shape();
  prob_size_[batch] = activation_shape.size() == 3 ? activation_shape[0] : 1;
  prob_size_[m] = activation_shape.size() == 3 ? activation_shape[1] : activation_shape[0];
  prob_size_[n] = dst_shape[2];
  prob_size_[k] = weight_shape[0];
  int reuse_data_size = prob_size_[n] * (prob_size_[k] + 16);
  float large_wei_threshold = 1.f;
  if (op_attrs.count("large_wei_threshold") != 0)
    large_wei_threshold = std::stof(op_attrs["large_wei_threshold"]);  // tunable param for engine.
  if (reuse_data_size > L2_size_ * large_wei_threshold || ts_desc[2].dtype() != data_type::s8) split_execute_ = true;
}

bool dynamic_quant_matmul_kd_t::split_execute_init() {
  auto core_num = omp_get_max_threads();
  auto ts_descs = op_desc_.tensor_descs();
  // first integer in pairs represent cores assigned to the weight, second represent cores assigned to the activation.
  std::vector<std::pair<int, int>> assign_cores_list;
  for (int i = 1; i <= core_num; i++)
    if (core_num % i == 0) assign_cores_list.push_back({i, core_num / i});

  auto pad_n = ceil_div(prob_size_[n], 16) * 16;
  int activation_cores = core_num, weight_cores = 1;
  for (auto&& i : assign_cores_list) {
    if (pad_n % i.first != 0) continue;
    if ((pad_n / i.first) % 16 != 0) continue;
    auto N_dim_per_core = pad_n / i.first;
    if (N_dim_per_core * prob_size_[k] < L2_size_) {
      weight_cores = i.first;
      activation_cores = i.second;
      break;
    }
  }
  assign_cores_ = {weight_cores, activation_cores};
  auto m_per_core = prob_size_[m] / activation_cores;
  auto remain_m = prob_size_[m] % activation_cores;
  bool add_bias = ts_descs[io::BIAS].size() != 0;
  bool append_sum = op_desc_.attrs().count("append_sum") != 0 ? true : false;
  auto dst_dt = ts_descs[2].dtype();
  if (append_sum)
    SPARSE_LOG_IF(FATAL, dst_dt != data_type::fp32)
        << "only support fp32 dst data type when append sum feature enable.";
  for (int i = 0; i < activation_cores; i++) {
    for (int j = 0; j < weight_cores; j++) {
      ssd::dynamic_quant_matmul_param_t p;
      p.m = i < remain_m ? m_per_core + 1 : m_per_core;
      p.n = pad_n / weight_cores;
      p.k = prob_size_[k];
      p.align_build_block_num = 3;
      auto align_build_dst_n = p.align_build_block_num * 16;
      p.align_n_loop = p.n / align_build_dst_n;
      if (j == weight_cores - 1 && prob_size_[n] % 16 != 0) p.write_mask = 16 - prob_size_[n] % 16;
      if (p.n % align_build_dst_n != 0) p.tail_n_loop = (p.n % align_build_dst_n) / 16;
      p.tile_k = 64;
      while (p.k % p.tile_k != 0) p.tile_k -= 4;

      p.align_m_loop = p.m / 16;
      p.tail_m = p.m % 16;
      p.add_bias = add_bias;
      p.append_sum = append_sum;
      p.dst_dt = dst_dt;
      p.postop_attrs = op_desc_.apply_postops_list();

      auto config_amx = [&](int M_tile, int N_tile, int K_tile, tileconfig_t* cfg) {
        tile_param_t p = {M_tile, N_tile, K_tile, false, 4, 3, 1, 4};
        configure_tiles(p, cfg);
      };

      config_amx(16, 16, p.tile_k, &p.m_align_cfg);
      config_amx(p.tail_m, 16, p.tile_k, &p.m_tail_cfg);
      params_.push_back(p);
    }
  }
  // init quant_param.
  quant_param_.input_dt = data_type::bf16;  // make it configable?
  quant_param_.output_dt = data_type::s8;
  quant_param_.quantized_dim_elt_num = prob_size_[n];
  quant_param_.quantized_dim_tail_elt_num = prob_size_[n] % 16;
  quant_param_.channel_num = prob_size_[batch] * prob_size_[m];
  return true;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
bool dynamic_quant_matmul_kd_t::init() {
  if (!isa_available(amx_int8)) return false;
  auto ts_descs = op_desc_.tensor_descs();
  bool add_bias = ts_descs[io::BIAS].size() != 0;
  if (ts_descs[0].dtype() != data_type::s8 || ts_descs[1].dtype() != data_type::s8)
    SPARSE_LOG(FATAL) << "activation, weight should be s8 in dynamic_quant_matmul";
  SPARSE_LOG_IF(FATAL, prob_size_[k] % 4 != 0) << "k must pad with 4.";
  if (split_execute_) return split_execute_init();
  SPARSE_LOG_IF(FATAL, ts_descs[2].dtype() != data_type::s8) << "dst must be s8 in non-split-execute mode.";
  auto core_num = omp_get_max_threads();
  auto m_per_core = prob_size_[m] / core_num;
  auto remain_m = prob_size_[m] % core_num;
  // assign m for each core
  for (int i = 0; i < core_num; i++) {
    ssd::dynamic_quant_matmul_param_t p;
    p.m = i < remain_m ? m_per_core + 1 : m_per_core;
    params_.push_back(p);
  }
  for (auto&& p : params_) {
    p.n = prob_size_[n];
    p.k = prob_size_[k];
    p.pad_n = ceil_div(prob_size_[n], 16) * 16;
    p.align_build_block_num = 3;
    auto align_build_dst_n = p.align_build_block_num * 16;
    p.align_n_loop = p.pad_n / align_build_dst_n;
    if (p.n % 16 != 0) p.write_mask = 16 - p.n % 16;
    if (p.pad_n % align_build_dst_n != 0) p.tail_n_loop = (p.pad_n % align_build_dst_n) / 16;

    p.tile_k = 64;
    while (p.k % p.tile_k != 0) p.tile_k -= 4;

    p.align_m_loop = p.m / 16;
    p.tail_m = p.m % 16;
    p.add_bias = add_bias;
    p.postop_attrs = op_desc_.apply_postops_list();

    auto config_amx = [&](int M_tile, int N_tile, int K_tile, tileconfig_t* cfg) {
      tile_param_t p = {M_tile, N_tile, K_tile, false, 4, 3, 1, 4};
      configure_tiles(p, cfg);
    };

    config_amx(16, 16, p.tile_k, &p.m_align_cfg);
    config_amx(p.tail_m, 16, p.tile_k, &p.m_tail_cfg);
  }

  return true;
}
#pragma GCC diagnostic pop

bool dynamic_quant_matmul_k_t::split_execute_init() {
  auto prob_size = derived_kd()->shape();
  single_tmp_buf_size_ = 16 * 16 * 3 * sizeof(float);
  auto enable_thr = omp_get_max_threads();
  bf16_tmp_buf_offset_ = enable_thr * single_tmp_buf_size_;
  total_tmp_buf_size_ = bf16_tmp_buf_offset_ + prob_size[batch] * prob_size[m] * prob_size[n] * sizeof(bfloat16_t);
  split_execute_ = true;

  auto quant_param = derived_kd()->get_quant_param();
  auto remain_channel = quant_param.channel_num % enable_thr;
  auto channel_per_thr = quant_param.channel_num / enable_thr;
  auto params = derived_kd()->params();
  auto dst_dt = params[0].dst_dt;
  SPARSE_LOG_IF(FATAL, !(dst_dt == data_type::fp32 || dst_dt == data_type::bf16 || dst_dt == data_type::s8));
  has_bias_ = params[0].add_bias;
  if (dst_dt == data_type::fp32 || dst_dt == data_type::bf16) {
    quant_stage_ = false;
    total_tmp_buf_size_ = bf16_tmp_buf_offset_;
  }
  auto assign_cores = derived_kd()->get_assign_cores();
  auto activation_cores = assign_cores.second;
  auto weight_cores = assign_cores.first;
  int m_offset = 0;
  int channel_offset = 0;
  for (int i = 0; i < activation_cores; i++) {
    int n_offset = 0;
    for (int j = 0; j < weight_cores; j++) {
      auto p = params[i * weight_cores + j];
      m_offset_list_.push_back(m_offset);
      n_offset_list_.push_back(n_offset);
      n_offset += p.n;
      auto process_channel =
          (i * weight_cores + j) < static_cast<int>(remain_channel) ? channel_per_thr + 1 : channel_per_thr;
      jit_quant_kers_.push_back(new jit_dynamic_quant_t(quant_param, process_channel));
      quant_channel_offset_list_.push_back(channel_offset);
      channel_offset += process_channel * quant_param.quantized_dim_elt_num;
      jit_s8s8_dynamic_dequant_kers_.push_back(
          new jit_amx_s8s8_dynamic_dequant_matmul_t(p, quant_param.quantized_dim_elt_num));
    }
    m_offset += params[i * weight_cores].m;
  }

  for (auto&& ker : jit_s8s8_dynamic_dequant_kers_) {
    if (!ker->create_kernel()) return false;
  }

  if (quant_stage_) {
    for (auto&& ker : jit_quant_kers_) {
      if (!ker->create_kernel()) return false;
    }
  }

  return true;
}

bool dynamic_quant_matmul_k_t::init() {
  if (derived_kd()->check_split_execute()) return split_execute_init();
  int m_offset = 0;
  auto params = derived_kd()->params();
  single_tmp_buf_size_ = 16 * params[0].pad_n * sizeof(float);  // build 16xpad_n tile by default.
  total_tmp_buf_size_ = omp_get_max_threads() * single_tmp_buf_size_;
  has_bias_ = params[0].add_bias;
  for (auto&& i : params) {
    m_offset_list_.push_back(m_offset);
    m_offset += i.m;
    jit_kers_.push_back(new jit_amx_s8s8_dynamic_quant_matmul_t(i));
    jit_kers_.back()->create_kernel();
  }

  return true;
}

size_t dynamic_quant_matmul_k_t::get_workspace_size() const { return total_tmp_buf_size_; }

bool dynamic_quant_matmul_k_t::split_execute(const std::vector<const void*>& rt_data) const {
  auto prob_size = derived_kd()->shape();
  auto dst_dt = derived_kd()->params()[0].dst_dt;
  // gemm stage
  for (int batch = 0; batch < prob_size[prob_size_idx::batch]; batch++) {
#pragma omp parallel for
    for (int ker_idx = 0; ker_idx < static_cast<int>(m_offset_list_.size()); ker_idx++) {
      auto ker = jit_s8s8_dynamic_dequant_kers_[ker_idx];
      ssd::dynamic_quant_matmul_data_t data;
      data.activation = get_data_ptr(rt_data[io::ACTIVATION],
                                     batch * prob_size[m] * prob_size[k] + m_offset_list_[ker_idx] * prob_size[k]);
      data.reordered_weight = get_data_ptr(rt_data[io::WEIGHT], n_offset_list_[ker_idx] * prob_size[k]);
      if (quant_stage_) {
        data.dst = get_data_ptr(rt_data[io::WORKSPACE],
                                bf16_tmp_buf_offset_ + sizeof(bfloat16_t) * (batch * prob_size[m] * prob_size[n] +
                                                                             m_offset_list_[ker_idx] * prob_size[n] +
                                                                             n_offset_list_[ker_idx]));
      } else {
        data.dst = get_data_ptr(rt_data[io::DST], get_data_size(dst_dt) * (batch * prob_size[m] * prob_size[n] +
                                                                           m_offset_list_[ker_idx] * prob_size[n] +
                                                                           n_offset_list_[ker_idx]));
      }
      data.scale_a =
          get_data_ptr(rt_data[io::SCALE_A], sizeof(float) * (m_offset_list_[ker_idx] + batch * prob_size[m]));
      data.scale_w = get_data_ptr(rt_data[io::SCALE_W], n_offset_list_[ker_idx] * sizeof(float));
      data.tmp_buf = get_data_ptr(rt_data[io::WORKSPACE], ker_idx * single_tmp_buf_size_);
      if (has_bias_) data.bias = get_data_ptr(rt_data[io::BIAS], n_offset_list_[ker_idx] * sizeof(float));
      (*ker)(&data);
    }
  }
  if (quant_stage_) {
    // quant stage.
    auto quant_param = derived_kd()->get_quant_param();
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(jit_quant_kers_.size()); i++) {
      auto offset = quant_channel_offset_list_[i];
      dynamic_quant_data_t data;
      auto ker = jit_quant_kers_[i];
      data.src = get_data_ptr(rt_data[io::WORKSPACE], bf16_tmp_buf_offset_ + offset * get_data_size(data_type::bf16));
      data.mat_dst = reinterpret_cast<char*>(const_cast<void*>(rt_data[io::DST])) + offset * sizeof(int8_t);
      data.scale_dst = reinterpret_cast<char*>(const_cast<void*>(rt_data[io::SCALE_DST])) +
                       offset / quant_param.quantized_dim_elt_num * sizeof(float);
      (*ker)(&data);
    }
  }
  return true;
}

bool dynamic_quant_matmul_k_t::execute(const std::vector<const void*>& rt_data) const {
  if (split_execute_) return split_execute(rt_data);
  auto prob_size = derived_kd()->shape();
  for (int batch = 0; batch < prob_size[prob_size_idx::batch]; batch++) {
#pragma omp parallel for
    for (int ker_idx = 0; ker_idx < static_cast<int>(m_offset_list_.size()); ker_idx++) {
      auto ker = jit_kers_[ker_idx];
      ssd::dynamic_quant_matmul_data_t data;
      data.activation = get_data_ptr(rt_data[io::ACTIVATION],
                                     batch * prob_size[m] * prob_size[k] + m_offset_list_[ker_idx] * prob_size[k]);
      data.reordered_weight = const_cast<void*>(rt_data[io::WEIGHT]);
      data.dst =
          get_data_ptr(rt_data[io::DST], batch * prob_size[m] * prob_size[n] + m_offset_list_[ker_idx] * prob_size[n]);
      data.scale_a =
          get_data_ptr(rt_data[io::SCALE_A], sizeof(float) * (m_offset_list_[ker_idx] + batch * prob_size[m]));
      data.scale_w = const_cast<void*>(rt_data[io::SCALE_W]);
      data.scale_dst =
          get_data_ptr(rt_data[io::SCALE_DST], sizeof(float) * (batch * prob_size[m] + m_offset_list_[ker_idx]));
      data.tmp_buf = get_data_ptr(rt_data[io::WORKSPACE], ker_idx * single_tmp_buf_size_);
      if (has_bias_) data.bias = const_cast<void*>(rt_data[io::BIAS]);
      (*ker)(&data);
    }
  }
  return true;
}
}  // namespace jd
