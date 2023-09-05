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
#include "transpose_mha.hpp"
#include <algorithm>
#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "kernels/spmm_types.hpp"

namespace bench {
bench_res_t transpose_mha_bench::set_config(int argc, char** argv) {
  if (argc < TRANSPOSE_MHA_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  LOG(INFO) << "transpose_mha\n";
  batch_size = str_to_num<dim_t>(argv[0]);
  head_num = str_to_num<dim_t>(argv[1]);
  head_size = str_to_num<dim_t>(argv[2]);
  seq_len = str_to_num<dim_t>(argv[3]);
  if (argc > 4) impl = argv[4];
  return {bench_status::success};
}
bool transpose_mha_bench::check_result() {
  // const auto& p = args.first;
  // const auto& q = args.second;
  SPARSE_LOG(ERROR) << "Result checking not implemented: transpose_mha";
  return true;
}
static void matf32_quantize_perlayer_s8(float* srcmat, int8_t* tarmat, int size, float* scale) {
  float minval = std::numeric_limits<float>::max();
  float maxval = std::numeric_limits<float>::min();
  for (int i = 0; i < size; i++) {
    if (srcmat[i] < minval) {
      minval = srcmat[i];
    }
    if (srcmat[i] > maxval) {
      maxval = srcmat[i];
    }
  }
  *scale = std::max(abs(minval), abs(maxval)) / 127;
  for (int i = 0; i < size; i++) {
    tarmat[i] = static_cast<char>(srcmat[i] / scale[0]);
  }
}
void transpose_mha_bench::gen_case() {
  op_attrs = {};
  if (impl != "") op_attrs["impl"] = impl;
  // Step 1: Construct runtime data for equivalent merged spmm
  const std::vector<dim_t> spmm_shape = {batch_size, head_num, head_size, seq_len};
  const jd::tensor_desc src_k_desc = {spmm_shape, jd::data_type::s8, jd::format_type::undef};
  const jd::tensor_desc src_q_desc = {spmm_shape, jd::data_type::s8, jd::format_type::undef};
  const jd::tensor_desc mask_desc = {spmm_shape, jd::data_type::fp32, jd::format_type::undef};
  const jd::tensor_desc src_v_desc = {spmm_shape, jd::data_type::s8, jd::format_type::undef};
  const jd::tensor_desc dst_desc = {spmm_shape, jd::data_type::u8, jd::format_type::undef};
  ts_descs = {src_k_desc, src_q_desc, mask_desc, src_v_desc, dst_desc};
  const int spmm_size = std::accumulate(spmm_shape.begin(), spmm_shape.end(), 1LL, std::multiplies<dim_t>());
  const int mask_size = batch_size * seq_len;
  const auto src_k = aligned_allocator_t<int8_t>::allocate(spmm_size);
  const auto src_k_f32 = aligned_allocator_t<float>::allocate(spmm_size);
  const auto src_q = aligned_allocator_t<int8_t>::allocate(spmm_size);
  const auto src_q_f32 = aligned_allocator_t<float>::allocate(spmm_size);
  const auto src_mask = aligned_allocator_t<float>::allocate(mask_size);
  const auto src_v = aligned_allocator_t<int8_t>::allocate(spmm_size);
  const auto src_v_f32 = aligned_allocator_t<float>::allocate(spmm_size);
  const auto dst = aligned_allocator_t<uint8_t>::allocate(spmm_size, true);
  init_vector(src_mask, mask_size, -.5f, .5f);
  init_vector(src_k_f32, spmm_size, -.5f, .5f);
  init_vector(src_q_f32, spmm_size, 0.f, 1.f);
  init_vector(src_v_f32, spmm_size, -.005f, .995f);
  float scale_q, scale_k, scale_v;
  matf32_quantize_perlayer_s8(src_k_f32, src_k, spmm_size, &scale_q);
  matf32_quantize_perlayer_s8(src_q_f32, src_q, spmm_size, &scale_k);
  matf32_quantize_perlayer_s8(src_v_f32, src_v, spmm_size, &scale_v);
  aligned_allocator_t<float>::deallocate(src_k_f32);
  aligned_allocator_t<float>::deallocate(src_q_f32);
  aligned_allocator_t<float>::deallocate(src_v_f32);
  std::vector<const void*> rt_data_p(io::transpose_mha_io_MAX + 1, nullptr);
  std::vector<const void*> rt_data_q(io::transpose_mha_io_MAX + 1, nullptr);
  rt_data_p[io::SRC_K] = src_k;
  rt_data_p[io::SRC_Q] = src_q;
  rt_data_p[io::MASK] = src_mask;
  rt_data_p[io::SRC_V] = src_v;
  rt_data_p[io::DST] = dst;
  rt_data_p[io::TMP2M] = aligned_allocator_t<uint8_t>::allocate(omp_get_max_threads() * (1 << 21));
  rt_data_p[io::SL_PAD] = new int(seq_len);
  rt_data_p[io::BATCH] = new int(batch_size);
  rt_data_p[io::HEAD_NUM] = new int(head_num);
  rt_data_p[io::HEAD_SIZE] = new int(head_size);
  rt_data_p[io::SEQ_LEN] = new int(seq_len);
  rt_data_p[io::SCALE_Q] = new float(scale_q);
  rt_data_p[io::SCALE_K] = new float(scale_k);
  rt_data_p[io::SCALE_V] = new float(scale_v);
  rt_data_p[io::SCALE_DST] = new float(scale_v);  // every element is some kind of mix of a row in src_v
  rt_data_p[io::ZP_DST] = new int(0);
  // TODO(zhe): implement ref kernel and fill rt_data_p
  jd::operator_desc op_desc(jd::kernel_kind::transpose_mha, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs);
  // Step 3: op_args_t testcase pair
  op_args_t op_args_p = {op_desc, rt_data_p};
  op_args_t op_args_q = {op_desc, rt_data_q};
  args = {op_args_p, op_args_q};
}
}  // namespace bench
