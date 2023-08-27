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
#include "attention.hpp"

#include <algorithm>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "src/cpu/kernels/attention_ref.hpp"
#include "kernels/spmm_types.hpp"

namespace bench {
template <typename T>
void prepare_sparse_data(T* vector_data, dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, float sparsity,
                         uint32_t seed = 123) {
  std::srand(seed);
  for (int i = 0; i < rows; i += blk_row) {
    for (int j = 0; j < cols; j += blk_col) {
      bool fill_zero = std::rand() % 100 <= (sparsity * 100);
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
                          const void* src_data = nullptr, bool is_clear = false, float sparse_ratio = 0.f,
                          const std::vector<float>& ranges = {-10, 10}) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int bytes_size = elem_num * jd::type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = aligned_allocator_t<uint8_t, 64>::allocate(bytes_size);
    memset(data_ptr, 0, bytes_size);
  } else if (src_data != nullptr) {
    data_ptr = aligned_allocator_t<uint8_t, 64>::allocate(bytes_size);
    memcpy(data_ptr, src_data, bytes_size);
  } else {
    if (a_dt == jd::data_type::fp32) {
      data_ptr = aligned_allocator_t<float, 64>::allocate(elem_num);
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::s32) {
      data_ptr = aligned_allocator_t<int32_t, 64>::allocate(elem_num);
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::u8) {
      data_ptr = aligned_allocator_t<uint8_t, 64>::allocate(elem_num);
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::s8) {
      data_ptr = aligned_allocator_t<int8_t, 64>::allocate(elem_num);
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
      if (sparse_ratio != 0.f) {
        int8_t* s8_ptr = static_cast<int8_t*>(data_ptr);
        prepare_sparse_data(s8_ptr, a_shape[0], a_shape[1], 4, 1, sparse_ratio);
      }
    }
  }
  return data_ptr;
}

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
bool attention_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;
  bool ret = false;
  std::shared_ptr<const jd::kernel_desc_t> attention_ref_desc;
  jd::kernel_desc_t::create<jd::attention_ref_kd_t>(attention_ref_desc, q.op_desc);
  std::shared_ptr<const jd::kernel_t> attention_ref_kernel;
  jd::kernel_t::create<jd::attention_ref_k_t, jd::attention_ref_kd_t>(attention_ref_kernel, attention_ref_desc);
  attention_ref_kernel->execute(q.rt_data);
  auto buf1 = p.rt_data[jd::attention_io::MERGE_DST];
  auto size1 = p.op_desc.tensor_descs()[jd::attention_io::MERGE_DST].size();
  auto buf2 = q.rt_data[jd::attention_io::MERGE_DST];
  auto size2 = q.op_desc.tensor_descs()[jd::attention_io::MERGE_DST].size();
  // Should compare buffer with different addresses
  const auto& dst_type = p.op_desc.tensor_descs()[jd::attention_io::MERGE_DST].dtype();
  if (dst_type == jd::data_type::fp32) {
    ret = compare_data<float>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == jd::data_type::s32) {
    ret = compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == jd::data_type::u8) {
    ret = compare_data<uint8_t>(buf1, size1, buf2, size2, 8e-3);
  } else if (dst_type == jd::data_type::s8) {
    ret = compare_data<int8_t>(buf1, size1, buf2, size2, 8e-3);
  }
  return ret;
}
void attention_bench::gen_case() {
  op_attrs = {{"out_scale", "0.125"},
              {"scr0_scale", "1"},
              {"scr1_scale", "1"},
              {"softmax_in_zero_point", "140"},
              {"softmax_in_scale", "500"},
              {"softmax_out_zero_point", "0"},
              {"softmax_out_scale", "0.00324144"}};
  // Step 1: Construct runtime data for equivalent merged spmm
  const dim_t ip_chanel = head_num * head_size;  // channel width for any of the three linear layer
  std::vector<jd::tensor_desc> ts_descs_qkv_ip = {
      {{ip_chanel * 3, ip_chanel}, jd::data_type::s8, jd::format_type::bsr},        // WEI
      {{ip_chanel, batch_size * seq_len}, jd::data_type::u8, jd::format_type::ab},  // SRC
      {{ip_chanel * 3, 1}, jd::data_type::s32, jd::format_type::ab},                // BIAS
      {{ip_chanel * 3, batch_size * seq_len}, dt_dst, jd::format_type::ab},         // DST
      {{ip_chanel * 3, 1}, jd::data_type::fp32, jd::format_type::ab},               // SCALE
  };
  std::vector<const void*> rt_data_qkv_ip(ts_descs_qkv_ip.size(), nullptr);
  for (size_t index = 0; index < ts_descs_qkv_ip.size(); ++index) {
    auto& tsd = ts_descs_qkv_ip[index];
    bool is_clear = index == jd::ssd::DST;
    float data_sparsity = (index == jd::ssd::WEI) ? sparsity : 0;
    auto ranges = (index == jd::ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    const void* data_addr = make_data_obj(tsd.shape(), tsd.dtype(), nullptr, is_clear, data_sparsity, ranges);
    rt_data_qkv_ip[index] = data_addr;
  }
  jd::tensor_desc attention_src_desc = {{batch_size, ip_chanel, seq_len}, jd::data_type::u8, jd::format_type::ab};
  jd::tensor_desc attention_dst_desc = {
      {head_num, head_size, batch_size, seq_len}, jd::data_type::u8, jd::format_type::ab};
  jd::tensor_desc q_weight_desc = {{ip_chanel, ip_chanel}, jd::data_type::s8, jd::format_type::bsr};
  jd::tensor_desc k_weight_desc = {{ip_chanel, ip_chanel}, jd::data_type::s8, jd::format_type::bsr};
  jd::tensor_desc v_weight_desc = {{ip_chanel, ip_chanel}, jd::data_type::s8, jd::format_type::bsr};
  jd::tensor_desc q_bias_desc = {{ip_chanel, 1}, jd::data_type::s32, jd::format_type::ab};
  jd::tensor_desc k_bias_desc = {{ip_chanel, 1}, jd::data_type::s32, jd::format_type::ab};
  jd::tensor_desc v_bias_desc = {{ip_chanel, 1}, jd::data_type::s32, jd::format_type::ab};
  jd::tensor_desc q_scales_desc = {{ip_chanel, 1}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc k_scales_desc = {{ip_chanel, 1}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc v_scales_desc = {{ip_chanel, 1}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc reshape_desc = {{batch_size, seq_len}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc q_k_src2_desc = {{batch_size, head_num, seq_len, seq_len}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc q_k_scales_desc = {
      {}, jd::data_type::undef, jd::format_type::undef};  // currently pass by static value
  jd::tensor_desc qk_v_scales_desc = {{1}, jd::data_type::fp32, jd::format_type::a};
  jd::tensor_desc qk_v_zp_desc = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs = {attention_src_desc, attention_dst_desc, q_weight_desc, k_weight_desc,   v_weight_desc, q_bias_desc,
              k_bias_desc,        v_bias_desc,        q_scales_desc, k_scales_desc,   v_scales_desc, reshape_desc,
              q_k_src2_desc,      q_k_scales_desc,    qk_v_zp_desc,  qk_v_scales_desc};
  std::vector<std::vector<dim_t>> ts_shapes(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), ts_shapes.begin(), [](jd::tensor_desc d) { return d.shape(); });
  std::vector<jd::data_type> ts_types(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), ts_types.begin(), [](jd::tensor_desc d) { return d.dtype(); });
  std::vector<const void*> rt_data_p(jd::attention_io::QK_V_OUTPUT_SCALES + 1, nullptr);
  std::vector<const void*> rt_data_q(jd::attention_io::QK_V_OUTPUT_SCALES + 1, nullptr);
  rt_data_p[jd::attention_io::MERGE_SRC] = make_data_obj(
      ts_shapes[jd::attention_io::MERGE_SRC], ts_types[jd::attention_io::MERGE_SRC], rt_data_qkv_ip[jd::ssd::SRC]);
  rt_data_q[jd::attention_io::MERGE_SRC] = make_data_obj(
      ts_shapes[jd::attention_io::MERGE_SRC], ts_types[jd::attention_io::MERGE_SRC], rt_data_qkv_ip[jd::ssd::SRC]);
  rt_data_p[jd::attention_io::MERGE_DST] =
      make_data_obj(ts_shapes[jd::attention_io::MERGE_DST], ts_types[jd::attention_io::MERGE_DST], nullptr, true);
  rt_data_q[jd::attention_io::MERGE_DST] =
      make_data_obj(ts_shapes[jd::attention_io::MERGE_DST], ts_types[jd::attention_io::MERGE_DST], nullptr, true);
  // for binary add of QxK matmul
  rt_data_p[jd::attention_io::Q_K_SRC2] =
      make_data_obj(ts_shapes[jd::attention_io::Q_K_SRC2], ts_types[jd::attention_io::Q_K_SRC2], nullptr);
  rt_data_q[jd::attention_io::Q_K_SRC2] =
      make_data_obj(ts_shapes[jd::attention_io::Q_K_SRC2], ts_types[jd::attention_io::Q_K_SRC2],
                    rt_data_p[jd::attention_io::Q_K_SRC2]);
  // for output scale & zp of QKxV matmul
  rt_data_p[jd::attention_io::QK_V_OUTPUT_ZERO_POINT] = new float(113);
  rt_data_q[jd::attention_io::QK_V_OUTPUT_ZERO_POINT] = new float(113);
  rt_data_p[jd::attention_io::QK_V_OUTPUT_SCALES] = new float(.003f);
  rt_data_q[jd::attention_io::QK_V_OUTPUT_SCALES] = new float(.003f);
  // Merge weight
  const size_t wei_bytes =
      ts_descs[jd::attention_io::Q_WEIGHT].size() * jd::type_size[ts_descs[jd::attention_io::Q_WEIGHT].dtype()];
  void* q_weight_addr = const_cast<void*>(
      make_data_obj(ts_descs[jd::attention_io::Q_WEIGHT].shape(), ts_descs[jd::attention_io::Q_WEIGHT].dtype()));
  void* k_weight_addr = const_cast<void*>(
      make_data_obj(ts_descs[jd::attention_io::K_WEIGHT].shape(), ts_descs[jd::attention_io::K_WEIGHT].dtype()));
  void* v_weight_addr = const_cast<void*>(
      make_data_obj(ts_descs[jd::attention_io::V_WEIGHT].shape(), ts_descs[jd::attention_io::V_WEIGHT].dtype()));
  const char* rt_data_qkv_ip_wei = static_cast<const char*>(rt_data_qkv_ip[jd::ssd::WEI]);
  memcpy(q_weight_addr, rt_data_qkv_ip_wei, wei_bytes);
  memcpy(k_weight_addr, rt_data_qkv_ip_wei + wei_bytes, wei_bytes);
  memcpy(v_weight_addr, rt_data_qkv_ip_wei + wei_bytes * 2, wei_bytes);
  op_attrs["q_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_weight_addr));
  op_attrs["k_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_weight_addr));
  op_attrs["v_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_weight_addr));
  // Merge bias
  const size_t bias_bytes =
      ts_descs[jd::attention_io::Q_BIAS].size() * jd::type_size[ts_descs[jd::attention_io::Q_BIAS].dtype()];
  void* q_bias_addr = const_cast<void*>(
      make_data_obj(ts_descs[jd::attention_io::Q_BIAS].shape(), ts_descs[jd::attention_io::Q_BIAS].dtype()));
  void* k_bias_addr = const_cast<void*>(
      make_data_obj(ts_descs[jd::attention_io::K_BIAS].shape(), ts_descs[jd::attention_io::K_BIAS].dtype()));
  void* v_bias_addr = const_cast<void*>(
      make_data_obj(ts_descs[jd::attention_io::V_BIAS].shape(), ts_descs[jd::attention_io::V_BIAS].dtype()));
  const char* rt_data_qkv_ip_bias = static_cast<const char*>(rt_data_qkv_ip[jd::ssd::BIAS]);
  memcpy(q_bias_addr, rt_data_qkv_ip_bias, bias_bytes);
  memcpy(k_bias_addr, rt_data_qkv_ip_bias + bias_bytes, bias_bytes);
  memcpy(v_bias_addr, rt_data_qkv_ip_bias + bias_bytes * 2, bias_bytes);
  op_attrs["q_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_bias_addr));
  op_attrs["k_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_bias_addr));
  op_attrs["v_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_bias_addr));
  // Merge scales
  const size_t scale_bytes =
      ts_descs[jd::attention_io::Q_SCALES].size() * jd::type_size[ts_descs[jd::attention_io::Q_SCALES].dtype()];
  void* q_scales_addr = const_cast<void*>(
      make_data_obj(ts_descs[jd::attention_io::Q_SCALES].shape(), ts_descs[jd::attention_io::Q_SCALES].dtype()));
  void* k_scales_addr = const_cast<void*>(
      make_data_obj(ts_descs[jd::attention_io::K_SCALES].shape(), ts_descs[jd::attention_io::K_SCALES].dtype()));
  void* v_scales_addr = const_cast<void*>(
      make_data_obj(ts_descs[jd::attention_io::V_SCALES].shape(), ts_descs[jd::attention_io::V_SCALES].dtype()));
  const char* rt_data_qkv_ip_scale = static_cast<const char*>(rt_data_qkv_ip[jd::ssd::SCALES]);
  memcpy(q_scales_addr, rt_data_qkv_ip_scale, scale_bytes);
  memcpy(k_scales_addr, rt_data_qkv_ip_scale + scale_bytes, scale_bytes);
  memcpy(v_scales_addr, rt_data_qkv_ip_scale + scale_bytes * 2, scale_bytes);
  op_attrs["q_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_scales_addr));
  op_attrs["k_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_scales_addr));
  op_attrs["v_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_scales_addr));
  jd::operator_desc op_desc(jd::kernel_kind::attention, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs);
  // Step 3: op_args_t testcase pair
  op_args_t op_args_p = {op_desc, rt_data_p};
  op_args_t op_args_q = {op_desc, rt_data_q};
  for (size_t index = 0; index < ts_descs_qkv_ip.size(); ++index) {
    aligned_allocator_t<uint8_t, 64>::deallocate(const_cast<void*>(rt_data_qkv_ip[index]));
  }
  args = {op_args_p, op_args_q};
}
}  // namespace bench
