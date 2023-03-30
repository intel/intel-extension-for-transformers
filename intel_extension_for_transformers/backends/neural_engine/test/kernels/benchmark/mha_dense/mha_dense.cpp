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

#include "mha_dense/mha_dense.hpp"

#include <algorithm>
#include <functional>
#include <utility>

#include "kernels/mha_dense_ref.hpp"
#include "kernels/mha_dense_types.hpp"
#include "utils.hpp"

namespace jd {
using dt = data_type;
std::pair<const void*, const void*> make_tensor_obj(const tensor_desc& ts_desc, float min_value = -10,
                                                    float max_value = 10) {
  int64_t elem_num = std::accumulate(ts_desc.shape().begin(), ts_desc.shape().end(), 1LL, std::multiplies<int64_t>());
  int bytes_size = elem_num * type_size[ts_desc.dtype()];
  void* data_ptr = nullptr;
  if (min_value == 0.f && max_value == 0.f) {
    data_ptr = aligned_allocator_t<uint8_t>::allocate(bytes_size, true);
    memset(data_ptr, 0, bytes_size);
  } else {
    if (ts_desc.dtype() == dt::fp32) {
      data_ptr = aligned_allocator_t<float>::allocate(pad_to(elem_num, 16));
      init_vector(static_cast<float*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == dt::bf16) {
      data_ptr = aligned_allocator_t<bfloat16_t>::allocate(pad_to(elem_num, 32));
      init_vector(static_cast<bfloat16_t*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == dt::s32) {
      data_ptr = aligned_allocator_t<int32_t>::allocate(pad_to(elem_num, 16));
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == dt::u8) {
      data_ptr = aligned_allocator_t<uint8_t>::allocate(pad_to(elem_num, 64));
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == dt::s8) {
      data_ptr = aligned_allocator_t<int8_t>::allocate(pad_to(elem_num, 64));
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, min_value, max_value);
    } else {
      SPARSE_LOG(FATAL) << "Unexpected dtype!";
    }
  }

  void* data_ptr_copy = aligned_allocator_t<uint8_t>::allocate(pad_to(bytes_size, 64), true);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

double mha_dense_bench::calc_flop() const {
  float flops = 0;
  flops += 2 * sl_m * head_size * sl_n;  // Q x K
  flops += 6 * sl_m * sl_n;              // softmax: 1max + 3reduction + 2softmax  ??copied from softmax benchmark
  flops += 2 * sl_n * sl_m * head_size;  // A x V

  flops *= head_num * batch_size;
  return flops;
}

bench_res_t mha_dense_bench::set_config(int argc, char** argv) {
  if (argc < MHA_DENSE_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";

    return {bench_status::wrong_input};
  }
  LOG(INFO) << "mha_dense\n";
  batch_size = str_to_num<int64_t>(argv[0]);
  sl_m = str_to_num<int64_t>(argv[1]);
  head_num = str_to_num<int64_t>(argv[2]);
  head_size = str_to_num<int64_t>(argv[3]);
  dt_dst = (argc <= 4)                    ? dt::u8
           : strcmp(argv[4], "fp32") == 0 ? dt::fp32
           : strcmp(argv[4], "s8") == 0   ? dt::s8
           : strcmp(argv[4], "u8") == 0   ? dt::u8
           : strcmp(argv[4], "bf16") == 0 ? dt::bf16
                                          : dt::undef;
  if (argc > 5) mask = str_to_num<int32_t>(argv[5]);
  if (argc > 6) badd_dim = str_to_num<int32_t>(argv[6]);
  if (argc > 7) sl_n = str_to_num<int32_t>(argv[7]);

  if (argc > 8) return {bench_status::wrong_input};
  if (sl_n <= 0) sl_n = sl_m;
  if (mask <= 0) mask = sl_n;
  if (dt_dst == dt::undef) return {bench_status::wrong_input};
  if (mask > sl_n) return {bench_status::wrong_input};
  if (badd_dim > 4) return {bench_status::wrong_input};

  return {bench_status::success};
}

void mha_dense_bench::get_true_data() {
  const auto& q = args.second;
  std::shared_ptr<const kernel_desc_t> mha_dense_ref_desc;
  kernel_desc_t::create<mha_dense_ref_kd_t>(mha_dense_ref_desc, q.op_desc);
  std::shared_ptr<const kernel_t> mha_dense_ref_kernel;
  kernel_t::create<mha_dense_ref_k_t, mha_dense_ref_kd_t>(mha_dense_ref_kernel, mha_dense_ref_desc);
  mha_dense_ref_kernel->execute(q.rt_data);
}

bool mha_dense_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;

  std::shared_ptr<const kernel_desc_t> mha_dense_ref_desc;
  kernel_desc_t::create<mha_dense_ref_kd_t>(mha_dense_ref_desc, q.op_desc);
  std::shared_ptr<const kernel_t> mha_dense_ref_kernel;
  kernel_t::create<mha_dense_ref_k_t, mha_dense_ref_kd_t>(mha_dense_ref_kernel, mha_dense_ref_desc);
  mha_dense_ref_kernel->execute(q.rt_data);

  auto buf1 = p.rt_data[mha_dense_io::DST];
  auto size1 = p.op_desc.tensor_descs()[mha_dense_io::DST].size();
  auto buf2 = q.rt_data[mha_dense_io::DST];
  auto size2 = q.op_desc.tensor_descs()[mha_dense_io::DST].size();
  // Should compare buffer with different addresses
  if (buf1 == buf2) return false;

  switch (p.op_desc.tensor_descs()[mha_dense_io::DST].dtype()) {
    case dt::fp32:
      return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    case dt::bf16:
      return compare_data<bfloat16_t>(buf1, size1, buf2, size2, 5e-3);
    case dt::s32:
      return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
    case dt::u8:
      return compare_data<uint8_t>(buf1, size1, buf2, size2, 8e-3);
    case dt::s8:
      return compare_data<int8_t>(buf1, size1, buf2, size2, 8e-3);
    default:
      SPARSE_LOG(ERROR) << "Unexpected dst type";
  }
  return false;
}

void mha_dense_bench::gen_case() {
  op_attrs.clear();
  op_attrs["approx_exp"] = "True";
  op_attrs["QK_rescale"] = "1.1";
  op_attrs["softmax_rescale"] = "255";
  op_attrs["QKV_rescale"] = "0.002";
  op_attrs["QKV_dstzp"] = "10";

  // Step 1: Construct runtime data for equivalent merged spmm
  std::vector<dim_t> badd_full = {batch_size, head_num, sl_m, sl_n};
  ts_descs.assign(mha_dense_io::mha_dense_io_MAX + 1, tensor_desc{{}, dt::undef, format_type::undef});
  ts_descs[mha_dense_io::SRC_Q] = {{batch_size, sl_m, head_num, head_size}, dt::s8, format_type::abcd};
  ts_descs[mha_dense_io::SRC_K] = {{batch_size, sl_n, head_num, head_size}, dt::s8, format_type::abcd};
  ts_descs[mha_dense_io::SRC_V] = {{batch_size, sl_n, head_num, head_size}, dt::s8, format_type::abcd};
  ts_descs[mha_dense_io::MASK] = {{batch_size}, dt::s32, format_type::ab};
  ts_descs[mha_dense_io::DST] = {{batch_size, sl_m, head_num, head_size}, dt_dst, format_type::abcd};
  if (badd_dim > 0) {
    SPARSE_LOG_IF(FATAL, badd_dim > 4) << "Unsupported binary add dimension";
    ts_descs[mha_dense_io::BINARY_ADD] = {std::vector<dim_t>(badd_full.cend() - badd_dim, badd_full.cend()), dt::fp32,
                                          format_type::a};
  }

  // Step 2: Construct Tensor ptr
  auto Qs = make_tensor_obj(ts_descs[mha_dense_io::SRC_Q]);
  auto Ks = make_tensor_obj(ts_descs[mha_dense_io::SRC_K]);
  auto Vs = make_tensor_obj(ts_descs[mha_dense_io::SRC_V]);
  auto masks = make_tensor_obj(ts_descs[mha_dense_io::MASK], mask, mask);
  auto dsts = make_tensor_obj(ts_descs[mha_dense_io::DST], 0, 0);
  auto badds = badd_dim > 0 ? make_tensor_obj(ts_descs[mha_dense_io::BINARY_ADD], -1.f, 1.f)
                            : std::pair<const void*, const void*>{nullptr, nullptr};

  std::vector<const void*> data_p(mha_dense_io::mha_dense_io_MAX + 1, nullptr);
  data_p[mha_dense_io::SRC_Q] = Qs.first;
  data_p[mha_dense_io::SRC_K] = Ks.first;
  data_p[mha_dense_io::SRC_V] = Vs.first;
  data_p[mha_dense_io::MASK] = masks.first;
  data_p[mha_dense_io::DST] = dsts.first;
  data_p[mha_dense_io::BINARY_ADD] = badds.first;

  std::vector<const void*> data_q(mha_dense_io::mha_dense_io_MAX + 1, nullptr);
  data_q[mha_dense_io::SRC_Q] = Qs.second;
  data_q[mha_dense_io::SRC_K] = Ks.second;
  data_q[mha_dense_io::SRC_V] = Vs.second;
  data_q[mha_dense_io::MASK] = masks.second;
  data_q[mha_dense_io::DST] = dsts.second;
  data_q[mha_dense_io::BINARY_ADD] = badds.second;

  operator_desc op_desc(kernel_kind::mha_dense, kernel_prop::forward_inference, engine_kind::cpu, ts_descs, op_attrs);

  // Step 3: op_args_t testcase pair
  args = {{op_desc, data_p}, {op_desc, data_q}};
}

}  // namespace jd
