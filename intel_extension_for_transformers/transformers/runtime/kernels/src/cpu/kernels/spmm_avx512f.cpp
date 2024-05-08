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

#include "spmm_avx512f.hpp"
namespace jd {

// Part1: class spmm_avx512f_kd_t
bool spmm_avx512f_kd_t::init() {
  if (!isa_available(avx512_core)) return false;

  const auto& wei_desc = op_desc_.tensor_descs()[ssd::WEI];
  const auto& src_desc = op_desc_.tensor_descs()[ssd::SRC];
  const auto& bias_desc = op_desc_.tensor_descs()[ssd::BIAS];
  const auto& dst_desc = op_desc_.tensor_descs()[ssd::DST];
  bool has_bias = !bias_desc.shape().empty();

  bool is_supported =
      (op_desc_.kernel_prop() == kernel_prop::forward_inference) &&
      is_any_of({data_type::fp32}, [&](const data_type& a) { return wei_desc.dtype() == a; }) &&
      is_any_of({data_type::fp32}, [&](const data_type& a) { return src_desc.dtype() == a; }) &&
      (!has_bias || is_any_of({data_type::fp32}, [&](const data_type& a) { return bias_desc.dtype() == a; })) &&
      is_any_of({data_type::fp32}, [&](const data_type& a) { return dst_desc.dtype() == a; });
  if (!is_supported) return false;

  if (wei_desc.shape().front() != src_desc.shape().back()) {
    SPARSE_LOG(WARNING) << "Skip as weight shape (" << wei_desc.shape().front() << ") and source shape ("
                        << wei_desc.shape().back() << ") don't match!";
    return false;
  }

  spmm_params_init(op_desc_);
  return true;
}

bool spmm_avx512f_kd_t::spmm_params_init(const operator_desc& op_desc) {
  const auto& wei_desc = op_desc.tensor_descs()[ssd::WEI];
  const auto& src_desc = op_desc.tensor_descs()[ssd::SRC];
  const auto& bias_desc = op_desc.tensor_descs()[ssd::BIAS];
  const auto& has_bias = !bias_desc.shape().empty();
  dim_t M = src_desc.shape()[0];
  dim_t K = src_desc.shape()[1];
  dim_t N = wei_desc.shape()[1];

  auto op_attrs = op_desc.attrs();
  const auto& sparse_addr = str_to_num<uint64_t>(op_attrs["sparse_ptr"]);
  const auto sparse_ptr = reinterpret_cast<bsc_data_t<float>*>(sparse_addr);
  int num_mblock = ceil_div(M, block_m_);
  params_.resize(num_mblock);
  for (int i = 0; i < num_mblock; ++i) {
    params_[i].M = M;
    params_[i].K = K;
    params_[i].N = N;
    params_[i].has_bias = has_bias;
    params_[i].im_start = i * block_m_;
    params_[i].im_end = std::min((i + 1) * block_m_, M);
    params_[i].sparse_ptr = sparse_ptr;
    params_[i].in_start = 0;
    params_[i].in_end = N;
    params_[i].postop_attrs = op_desc.apply_postops_list();
  }

  return true;
}

// Part2: class spmm_avx512f_k_t
bool spmm_avx512f_k_t::init() {
  auto& ker_params = derived_kd()->params();
  jit_kers_.clear();
  jit_kers_.reserve(ker_params.size());
  for (auto& param : ker_params) {
    jit_spmm_avx512f_t* ker = new jit_spmm_avx512f_t(param);
    if (ker == nullptr) return false;
    if (!ker->create_kernel()) return false;
    jit_kers_.emplace_back(ker);
  }
  return true;
}

bool spmm_avx512f_k_t::execute(const std::vector<const void*>& rt_data) const {
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(jit_kers_.size()); ++i) {
    auto& jit_impl = jit_kers_[i];
    ssd::avx512_data_t rt_param;
    rt_param.sparse = jit_impl->bsc_data()->data().data();
    rt_param.dense = reinterpret_cast<const float*>(rt_data[ssd::SRC]);
    rt_param.bias = reinterpret_cast<const float*>(rt_data[ssd::BIAS]);
    rt_param.dst = const_cast<float*>(reinterpret_cast<const float*>(rt_data[ssd::DST]));
    (*jit_impl)(&(rt_param));
  }
  return true;
}
}  // namespace jd
