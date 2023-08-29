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
#include "mha_dense_dynamic.hpp"

#include <functional>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "src/cpu/kernels/mha_dense_ref.hpp"

namespace bench {
using io = jd::exposed_enum::mha_dense::io;
const void* make_data_obj(const jd::tensor_desc desc, const float min_val, const float max_val) {
  int elem_num = std::accumulate(desc.shape().begin(), desc.shape().end(), dim_t{1}, std::multiplies<dim_t>());
  const int bytes_size = elem_num * jd::type_size[desc.dtype()];
  void* data_ptr = nullptr;
  if (min_val == 0 && max_val == 0) {
    data_ptr = aligned_allocator_t<uint8_t>::allocate(pad_to(bytes_size, 64), true);
  } else {
    static std::mt19937 rand_gen(1);
    const auto seed = std::uniform_int_distribution<>()(rand_gen);
    if (desc.dtype() == jd::data_type::fp32) {
      data_ptr = aligned_allocator_t<float>::allocate(pad_to(elem_num, 16));
      init_vector(static_cast<float*>(data_ptr), elem_num, min_val, max_val, seed);
    } else if (desc.dtype() == jd::data_type::s32) {
      data_ptr = aligned_allocator_t<int32_t>::allocate(pad_to(elem_num, 16));
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, min_val, max_val, seed);
    } else if (desc.dtype() == jd::data_type::u8) {
      data_ptr = aligned_allocator_t<uint8_t>::allocate(pad_to(elem_num, 64));
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, min_val, max_val, seed);
    } else if (desc.dtype() == jd::data_type::s8) {
      data_ptr = aligned_allocator_t<int8_t>::allocate(pad_to(elem_num, 64));
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, min_val, max_val, seed);
    } else {
      SPARSE_LOG(FATAL) << "Unexpected dt!";
    }
  }
  return data_ptr;
}
const void* copy_data_obj(const jd::tensor_desc desc, const void* src) {
  int elem_num = std::accumulate(desc.shape().begin(), desc.shape().end(), dim_t{1}, std::multiplies<dim_t>());
  const int bytes_size = elem_num * jd::type_size[desc.dtype()];
  void* data_ptr = aligned_allocator_t<char>::allocate(pad_to(bytes_size, 64));
  memcpy(data_ptr, src, bytes_size);
  return data_ptr;
}
void mha_dense_dynamic_bench::get_true_data() {
  const auto& q = args.second;
  std::shared_ptr<const jd::kernel_desc_t> mha_dense_ref_desc;
  jd::kernel_desc_t::create<jd::mha_dense_ref_kd_t>(mha_dense_ref_desc, q.op_desc);
  std::shared_ptr<const jd::kernel_t> mha_dense_ref_kernel;
  jd::kernel_t::create<jd::mha_dense_ref_k_t, jd::mha_dense_ref_kd_t>(mha_dense_ref_kernel, mha_dense_ref_desc);
  const auto workspace_q = aligned_allocator_t<char>::allocate(mha_dense_ref_kernel->get_workspace_size());
  std::vector<const void*> data_q(q.rt_data);
  data_q[io::WORKSPACE] = workspace_q;
  mha_dense_ref_kernel->execute(data_q);
  aligned_allocator_t<char>::deallocate(workspace_q);
}
bench_res_t mha_dense_dynamic_bench::set_config(int argc, char** argv) {
  if (argc < MIN_ARG_NUM) {
    SPARSE_LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  SPARSE_LOG(INFO) << "mha_dense_dynamic\n";
  batch_size = str_to_num<dim_t>(argv[0]);
  head_num = str_to_num<dim_t>(argv[1]);
  sl_M = str_to_num<dim_t>(argv[2]);
  head_size = str_to_num<dim_t>(argv[3]);
  sl_N = str_to_num<dim_t>(argv[4]);
  return {bench_status::success};
}
bool mha_dense_dynamic_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;
  std::shared_ptr<const jd::kernel_desc_t> mha_dense_ref_desc;
  jd::kernel_desc_t::create<jd::mha_dense_ref_kd_t>(mha_dense_ref_desc, q.op_desc);
  std::shared_ptr<const jd::kernel_t> mha_dense_dynamic_ref_kernel;
  jd::kernel_t::create<jd::mha_dense_ref_k_t, jd::mha_dense_ref_kd_t>(mha_dense_dynamic_ref_kernel, mha_dense_ref_desc);
  auto data_q = q.rt_data;
  data_q[io::WORKSPACE] = aligned_allocator_t<char>::allocate(mha_dense_dynamic_ref_kernel->get_workspace_size());
  mha_dense_dynamic_ref_kernel->execute(data_q);
  aligned_allocator_t<char>::deallocate(const_cast<void*>(data_q[io::WORKSPACE]));
  const auto dst = p.rt_data[io::DST];
  const auto dst_ref = data_q[io::DST];
  const auto dst_scale = p.rt_data[io::DST_SCALE];
  const auto dst_scale_ref = data_q[io::DST_SCALE];
  // Should compare buffer with different addresses
  if (dst == dst_ref) return false;
  if (dst_scale == dst_scale_ref) return false;
  const auto dst_size = batch_size * head_num * head_size * sl_M;
  const auto dst_scale_size = batch_size * sl_M;
  return compare_data<int8_t>(dst, dst_size, dst_ref, dst_size, 8e-2f) &&
         compare_data<float>(dst_scale, dst_scale_size, dst_scale_ref, dst_scale_size, 1e-3);
}
void mha_dense_dynamic_bench::gen_case() {
  op_attrs.clear();
  op_attrs["approx_exp"] = "True";
  op_attrs["stable_softmax"] = "False";
  op_attrs["softmax_rescale"] = "dynamic";
  // Step 2: Set tensor shapes
  ts_descs = std::vector<jd::tensor_desc>(io::SIZE, {{}, jd::data_type::undef, jd::format_type::undef});
  ts_descs[io::SRC_Q] = {{batch_size, sl_M, head_num, head_size}, jd::data_type::s8, jd::format_type::abcd};
  ts_descs[io::SRC_K] = {{batch_size, sl_N, head_num, head_size}, jd::data_type::s8, jd::format_type::abcd};
  ts_descs[io::SRC_V] = {{batch_size, sl_N, head_num, head_size}, jd::data_type::s8, jd::format_type::abcd};
  ts_descs[io::DST] = {{batch_size, sl_M, head_num, head_size}, jd::data_type::s8, jd::format_type::abcd};
  ts_descs[io::BINARY_ADD] = {{batch_size, 1, 1, sl_N}, jd::data_type::fp32, jd::format_type::ab};
  ts_descs[io::ATT_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::Q_SCALE] = {{batch_size, sl_M}, jd::data_type::fp32, jd::format_type::ab};
  ts_descs[io::K_SCALE] = {{batch_size, sl_N}, jd::data_type::fp32, jd::format_type::ab};
  ts_descs[io::V_SCALE] = {{batch_size, sl_N}, jd::data_type::fp32, jd::format_type::ab};
  ts_descs[io::DST_SCALE] = {{batch_size, sl_M}, jd::data_type::fp32, jd::format_type::ab};
  // Step 2: Construct runtime data
  std::vector<const void*> rt_data(io::SIZE, nullptr);
  rt_data[io::SRC_Q] = make_data_obj(ts_descs[io::SRC_Q], -128, 127);
  rt_data[io::SRC_K] = make_data_obj(ts_descs[io::SRC_K], -128, 127);
  rt_data[io::SRC_V] = make_data_obj(ts_descs[io::SRC_V], -128, 127);
  rt_data[io::BINARY_ADD] = make_data_obj(ts_descs[io::BINARY_ADD], 0, 0);
  rt_data[io::ATT_SCALE] = make_data_obj(ts_descs[io::ATT_SCALE], 1, 1);  // TODO(Yi): 1/sqrt(qk)
  rt_data[io::Q_SCALE] = make_data_obj(ts_descs[io::Q_SCALE], 0.001, 0.01);
  rt_data[io::K_SCALE] = make_data_obj(ts_descs[io::K_SCALE], 0.001, 0.01);
  rt_data[io::V_SCALE] = make_data_obj(ts_descs[io::V_SCALE], 0.001, 0.01);
  // workspace memory to be allocate just before execuation
  rt_data[io::DST] = make_data_obj(ts_descs[io::DST], -128, 127);  // random dst and scale to be overwrite
  rt_data[io::DST_SCALE] = make_data_obj(ts_descs[io::DST_SCALE], INT32_MIN, INT32_MAX);
  std::vector<const void*> rt_data_cpy(io::SIZE, nullptr);
  for (std::underlying_type<io>::type idx = 0; idx < io::SIZE; ++idx)
    if (rt_data[idx] != nullptr) rt_data_cpy[idx] = copy_data_obj(ts_descs[idx], rt_data[idx]);
  // Step 3: op_args_t testcase pair
  jd::operator_desc op_desc(jd::kernel_kind::mha_dense, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs);
  op_args_t op_args_p = {op_desc, rt_data};
  op_args_t op_args_q = {op_desc, rt_data_cpy};
  args = {op_args_p, op_args_q};
}
}  // namespace bench
