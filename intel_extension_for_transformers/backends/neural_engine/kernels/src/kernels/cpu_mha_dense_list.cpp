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

#include <map>
#include <tuple>
#include "cpu_engine.hpp"
#include "param_types.hpp"
#include "impl_list_item.hpp"
#include "kernels/mha_dense.hpp"
#include "kernels/mha_dense_bf16.hpp"
#include "kernels/mha_dense_ref.hpp"
#include "kernels/dynamic_quant_mha.hpp"
#include "kernels/mha_dense_types.hpp"
namespace jd {
static const std::vector<impl_list_item_t> bf16_impl_list{
    CPU_INSTANCE(mha_dense_bf16_k_t),
    CPU_INSTANCE(mha_dense_ref_k_t),
    NULL_INSTANCE(),
};
static const std::vector<impl_list_item_t> static_impl_list{
    CPU_INSTANCE(mha_dense_k_t),
    CPU_INSTANCE(mha_dense_ref_k_t),
    NULL_INSTANCE(),
};
static const std::vector<impl_list_item_t> dynamic_impl_list{
    CPU_INSTANCE(dynamic_quant_mha_k_t),
    CPU_INSTANCE(mha_dense_ref_k_t),
    NULL_INSTANCE(),
};

const std::vector<impl_list_item_t>* get_mha_dense_impl_list(const operator_desc& op_desc) {
  return op_desc.tensor_descs()[mha_dense_io::SRC_Q].dtype() == data_type::bf16        ? &bf16_impl_list
         : op_desc.tensor_descs()[mha_dense_io::DST_SCALE].dtype() == data_type::undef ? &static_impl_list
                                                                                       : &dynamic_impl_list;
}
}  // namespace jd
