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

#include <map>
#include <tuple>

#include "src/cpu/engine/cpu_engine.hpp"
#include "impl_list_item.hpp"
#include "spmm_amx_bf16_x16.hpp"
#include "spmm_vnni.hpp"
#include "spmm_avx512f.hpp"
#include "spmm_ref.hpp"
#include "param_types.hpp"

namespace jd {
using map_key_t = std::tuple<kernel_prop, data_type, data_type, data_type>;
/**
 * @param kernel_kind, point to this file "cpu_sparse_matmul_list.cpp".
 * @param kernel_prop, point to [KEY] of impl_list_map. A specific function or
 * scenario.
 * @param kernel_algorithm, point to [VAL] of impl_list_map. Different
 * algorithms of a specific function, e.g.: gemm, brgemm.
 * @note Use (kernel_kind->kernel_prop->kernel_algorithm) to denote a
 * specific/derived kernel. Ref onednn's cpu_inner_product_list.cpp, a map's
 * [VAL] is a derived "struct primitive_t", e.g.: gemm_inner_product_fwd_t<f32>.
 */
static const std::map<map_key_t, std::vector<impl_list_item_t>> impl_list_map = {
    {{kernel_prop::forward_inference, data_type::bf16, data_type::bf16, data_type::fp32},
     {CPU_INSTANCE(spmm_amx_bf16_x16_k_t), CPU_INSTANCE(spmm_ref_k_t), NULL_INSTANCE()}},
    {{kernel_prop::forward_inference, data_type::bf16, data_type::bf16, data_type::bf16},
     {CPU_INSTANCE(spmm_amx_bf16_x16_k_t), CPU_INSTANCE(spmm_ref_k_t), NULL_INSTANCE()}},
    {{kernel_prop::forward_inference, data_type::s8, data_type::u8, data_type::s8},
     {CPU_INSTANCE(spmm_vnni_k_t), CPU_INSTANCE(spmm_ref_k_t), NULL_INSTANCE()}},
    {{kernel_prop::forward_inference, data_type::s8, data_type::u8, data_type::fp32},
     {CPU_INSTANCE(spmm_vnni_k_t), CPU_INSTANCE(spmm_ref_k_t), NULL_INSTANCE()}},
    {{kernel_prop::forward_inference, data_type::s8, data_type::u8, data_type::u8},
     {CPU_INSTANCE(spmm_vnni_k_t), CPU_INSTANCE(spmm_ref_k_t), NULL_INSTANCE()}},
    {{kernel_prop::forward_inference, data_type::fp32, data_type::fp32, data_type::fp32},
     {CPU_INSTANCE(spmm_avx512f_k_t), CPU_INSTANCE(spmm_ref_k_t), NULL_INSTANCE()}},
};

const std::vector<impl_list_item_t>* get_sparse_matmul_impl_list(const operator_desc& op_desc) {
  const auto& tensor_descs = op_desc.tensor_descs();
  const auto& src0_dtype = tensor_descs[ssd::WEI].dtype();
  const auto& src1_dtype = tensor_descs[ssd::SRC].dtype();
  const auto& dst_dtype = tensor_descs[ssd::DST].dtype();
  map_key_t key{op_desc.kernel_prop(), src0_dtype, src1_dtype, dst_dtype};
  const auto impl_list_it = impl_list_map.find(key);
  return (impl_list_it != impl_list_map.end()) ? &(impl_list_it->second) : &cpu_engine_t::empty_list;
}
}  // namespace jd
