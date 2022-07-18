//  Copyright (c) 2021 Intel Corporation
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

#include "kernels/spmm_vnni.hpp"

namespace jd {
//// Part1: class spmm_vnni_kd_t
bool spmm_vnni_kd_t::init() {
  using dt = jd::data_type;
  const auto& wei_desc = op_desc_.tensor_descs()[ssd::WEI];
  const auto& src_desc = op_desc_.tensor_descs()[ssd::SRC];
  const auto& bias_desc = op_desc_.tensor_descs()[ssd::BIAS];
  const auto& dst_desc = op_desc_.tensor_descs()[ssd::DST];
  bool has_bias = !bias_desc.shape().empty();
  bool is_supported =
      (op_desc_.kernel_prop() == kernel_prop::forward_inference) &&
      is_any_of({dt::s8, dt::fp32}, [&](const dt& a) { return wei_desc.dtype() == a; }) &&
      is_any_of({dt::u8, dt::fp32}, [&](const dt& a) { return src_desc.dtype() == a; }) &&
      (!has_bias || is_any_of({dt::s32, dt::fp32}, [&](const dt& a) { return bias_desc.dtype() == a; })) &&
      is_any_of({dt::s8, dt::fp32}, [&](const dt& a) { return dst_desc.dtype() == a; });
  if (!is_supported) {
    return false;
  }
  if (wei_desc.shape().back() != src_desc.shape().front()) {
    return false;
  }

  int nthr = op_desc_.impl_nthr();
  params_.resize(nthr);
  for (int idx = 0; idx < nthr; ++idx) {
    ssd::flat_param_t& param = params_[idx];
    spmm_params_init(param, op_desc_, nthr, idx);
  }
  return true;
}

bool spmm_vnni_kd_t::spmm_params_init(ssd::flat_param_t& param_ref, const jd::operator_desc& op_desc, int nthr,
                                      int ithr) {
  const auto& wei_desc = op_desc.tensor_descs()[ssd::WEI];
  const auto& src_desc = op_desc.tensor_descs()[ssd::SRC];
  const auto& bias_desc = op_desc.tensor_descs()[ssd::BIAS];
  const auto& dst_desc = op_desc.tensor_descs()[ssd::DST];
  param_ref.M = wei_desc.shape()[0];
  param_ref.K = wei_desc.shape()[1];
  param_ref.N = src_desc.shape()[1];
  param_ref.has_bias = !bias_desc.shape().empty();
  auto op_attrs = op_desc.attrs();
  param_ref.append_sum = (op_attrs["post_op"] == "append_sum");
  param_ref.output_type = dst_desc.dtype();
  if (op_attrs["sparse_scheme"] == "dense_x_sparse") {
    param_ref.scheme = ssd::sparse_scheme::dense_x_sparse;
  } else if (op_attrs["sparse_scheme"] == "sparse_x_sparse") {
    param_ref.scheme = ssd::sparse_scheme::sparse_x_sparse;
  } else {
    param_ref.scheme = ssd::sparse_scheme::sparse_x_dense;
  }
  const auto& temp1 = split_str<int64_t>(op_attrs["mkn_blocks"]);
  param_ref.mkn_blocks = temp1.empty() ? std::vector<int64_t>{1, 1, 1} : temp1;
  const auto& temp2 = split_str<int64_t>(op_attrs["tile_shape"]);
  param_ref.tile_shape = temp2.empty() ? std::vector<int64_t>{4, 4} : temp2;
  param_ref.sub_func = (op_attrs["sub_func"] != "false");
  param_ref.in_start = 0;  // TODO: partition with fixed size m/n blocks
  param_ref.in_end = param_ref.N;
  int64_t th_mb = ceil_div(param_ref.mkn_blocks[0], nthr);
  int64_t MB = param_ref.M / param_ref.mkn_blocks[0];
  param_ref.im_start = th_mb * ithr * MB;
  param_ref.im_end = (param_ref.im_start + th_mb) * MB;

  const auto& temp_addr = str_to_num<uint64_t>(op_attrs["sparse_ptr"]);
  param_ref.sparse_ptr = reinterpret_cast<bsr_data_t<int8_t>*>(temp_addr);
  return true;
}

//// Part2: class spmm_vnni_k_t
bool spmm_vnni_k_t::init() {
  int nthr = kd()->operator_desc().impl_nthr();
  jit_kers_.resize(nthr);
  for (int idx = 0; idx < nthr; ++idx) {
    jit_spmm_vnni_t* ker = new jit_spmm_vnni_t(derived_kd()->params()[idx]);
    if (ker == nullptr) return false;
    if (!(ker->create_kernel())) return false;
    jit_kers_[idx] = ker;
  }
  return true;
}

bool spmm_vnni_k_t::execute(const std::vector<const void*>& rt_data) const {
  int nthr = kd()->operator_desc().impl_nthr();
  std::vector<ssd::flat_data_t> td(nthr);
#pragma omp parallel for num_threads(nthr)
  for (int idx = nthr - 1; idx >= 0; --idx) {
    const jit_spmm_vnni_t* jit_impl = jit_kers_[idx];
    td[idx].ptr_seq_vals = jit_impl->sequence_vals();
    td[idx].ptr_dense = rt_data[ssd::SRC];
    td[idx].ptr_bias = rt_data[ssd::BIAS];
    td[idx].ptr_dst = const_cast<void*>(rt_data[ssd::DST]);
    td[idx].ptr_scales = rt_data[ssd::SCALES];
    (*jit_impl)(&(td[idx]));
  }
  return true;
}
}  // namespace jd
