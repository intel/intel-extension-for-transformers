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

#include <immintrin.h>
#include "spmm_amx_bf16_x16.hpp"
#include "src/singleton.hpp"

namespace jd {
//// Part1: class spmm_amx_bf16_x16_kd_t
bool spmm_amx_bf16_x16_kd_t::init() {
  if (!isa_available(avx512_core_bf16_amx_bf16)) return false;

  const auto& wei_desc = op_desc_.tensor_descs()[ssd::WEI];
  const auto& src_desc = op_desc_.tensor_descs()[ssd::SRC];
  const auto& bias_desc = op_desc_.tensor_descs()[ssd::BIAS];
  const auto& dst_desc = op_desc_.tensor_descs()[ssd::DST];
  bool has_bias = !bias_desc.shape().empty();
  // TBD(hengyu): int8 support
  bool is_supported =
      (op_desc_.kernel_prop() == kernel_prop::forward_inference) &&
      is_any_of({data_type::bf16}, [&](const data_type& a) { return wei_desc.dtype() == a; }) &&
      is_any_of({data_type::bf16}, [&](const data_type& a) { return src_desc.dtype() == a; }) &&
      (!has_bias || is_any_of({data_type::fp32}, [&](const data_type& a) { return bias_desc.dtype() == a; })) &&
      is_any_of({data_type::bf16, data_type::fp32}, [&](const data_type& a) { return dst_desc.dtype() == a; });
  if (!is_supported) {
    return false;
  }
  if (wei_desc.shape().back() != src_desc.shape()[1]) {
    return false;
  }

  return spmm_params_init(params_, op_desc_);
}

bool spmm_amx_bf16_x16_kd_t::spmm_params_init(std::vector<ssd::amx_bf16_params_t>& param_ref,
                                              const operator_desc& op_desc) {
  const auto& wei_desc = op_desc.tensor_descs()[0];
  const auto& src_desc = op_desc.tensor_descs()[1];
  const auto& bias_desc = op_desc.tensor_descs()[2];
  const auto& dst_desc = op_desc.tensor_descs()[3];
  auto op_attrs = op_desc.attrs();
  const dim_t microOC = str_to_num<dim_t>(op_attrs["micro_oc"]);
  const auto& temp_addr = str_to_num<uint64_t>(op_attrs["sparse_ptr"]);
  const auto& all_bsr_data = reinterpret_cast<std::vector<bsr_data_t<bfloat16_t>*>*>(temp_addr);
  num_kernels_ = wei_desc.shape()[0] / microOC;
  param_ref.resize(num_kernels_);
  for (dim_t i = 0; i < num_kernels_; ++i) {
    const auto& bsr_data = (*all_bsr_data)[i];
    param_ref[i].num_tileM = src_desc.shape()[0];
    param_ref[i].tileM = src_desc.shape()[2];
    param_ref[i].shape[0] = wei_desc.shape()[0];
    param_ref[i].shape[1] = wei_desc.shape()[1];
    param_ref[i].nrowptr = wei_desc.shape()[1] + 1;
    param_ref[i].tileN = microOC;
    param_ref[i].nnz_group = bsr_data->nnz_group();
    param_ref[i].nrowptr = bsr_data->indptr().size();
    param_ref[i].colidxs = const_cast<dim_t*>(bsr_data->indices().data());
    param_ref[i].group_rowptr = const_cast<dim_t*>(bsr_data->indptr().data());
    param_ref[i].weight = const_cast<bfloat16_t*>(bsr_data->data().data());
    param_ref[i].has_bias = !bias_desc.shape().empty();
    param_ref[i].same_src_dtype = dst_desc.dtype() == data_type::bf16;
    param_ref[i].postop_attrs = op_desc.apply_postops_list();
  }
  return true;
}

//// Part2: class spmm_amx_bf16_x16_k_t
bool spmm_amx_bf16_x16_k_t::init() {
  dim_t num_kernels = derived_kd()->num_kernels();
  auto param = derived_kd()->params()[0];
  IC = param.shape[1];
  OC = param.shape[0];
  num_tileBS = param.num_tileM;
  tileBS = param.tileM;
  tileOC = param.tileN;
  num_tileOC = OC / tileOC;

  jit_kers_.resize(num_kernels);
  weights_.resize(num_kernels);
  for (dim_t i = 0; i < num_kernels; ++i) {
    jit_spmm_amx_bf16_x16_t* ker = nullptr;
    param = derived_kd()->params()[i];
    bool status = spmm_kernel_create(&ker, param);
    if (!status) return false;
    jit_kers_[i] = ker;
    weights_[i] = param.weight;
  }
  amx_config_ = Singleton<amx_tile_config_t>::GetInstance();
  return true;
}

bool spmm_amx_bf16_x16_k_t::spmm_kernel_create(jit_spmm_amx_bf16_x16_t** ker_pp, const ssd::amx_bf16_params_t& param) {
  *ker_pp = new jit_spmm_amx_bf16_x16_t(param);
  if (*ker_pp == nullptr) {
    return false;
  }
  auto status = (*ker_pp)->create_kernel();
  return status;
}

bool spmm_amx_bf16_x16_k_t::execute(const std::vector<const void*>& rt_data) const {
  bool bf16_out = derived_kd()->params()[0].same_src_dtype;
  if (!bf16_out) {
#pragma omp parallel for collapse(2)
    for (dim_t micro_oc = 0; micro_oc < num_tileOC; micro_oc++) {
      for (dim_t micro_bs = 0; micro_bs < num_tileBS; micro_bs++) {
        int thread_idx = omp_get_thread_num();
        amx_config_->amx_tile_configure(thread_idx, tile_param_);
        ssd::amx_bf16f32_inputs_t inputs;
        inputs.weight = weights_[micro_oc];
        inputs.src = static_cast<bfloat16_t*>(const_cast<void*>(rt_data[1])) + micro_bs * tileBS * IC;
        inputs.bias = static_cast<float*>(const_cast<void*>(rt_data[2])) + micro_oc * tileOC;
        inputs.dst =
            static_cast<float*>(const_cast<void*>(rt_data[3])) + micro_bs * tileBS * OC + micro_oc * tileOC * tileBS;
        (*jit_kers_[micro_oc])(&inputs);
      }
    }
  } else {
#pragma omp parallel for collapse(2)
    for (dim_t micro_oc = 0; micro_oc < num_tileOC; micro_oc++) {
      for (dim_t micro_bs = 0; micro_bs < num_tileBS; micro_bs++) {
        int thread_idx = omp_get_thread_num();
        amx_config_->amx_tile_configure(thread_idx, tile_param_);
        ssd::amx_bf16bf16_inputs_t inputs;
        inputs.weight = weights_[micro_oc];
        inputs.src = static_cast<bfloat16_t*>(const_cast<void*>(rt_data[1])) + micro_bs * tileBS * IC;
        inputs.bias = static_cast<float*>(const_cast<void*>(rt_data[2])) + micro_oc * tileOC;
        inputs.dst = static_cast<bfloat16_t*>(const_cast<void*>(rt_data[3])) + micro_bs * tileBS * OC +
                     micro_oc * tileOC * tileBS;
        (*jit_kers_[micro_oc])(&inputs);
      }
    }
  }
  return true;
}
}  // namespace jd
