/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#include "subgroup/subgroup.hpp"

namespace gpu::xetla {

/// @brief
///
enum class ln_fwd_fused_kind : uint8_t {
    none = 0,
    bias_dropout_resAdd_ln = 1,
    ln_dropout = 2,
    //fused with random number generator kernel
    bias_rng_dropout_resAdd_ln = 3,
    //fused with random number generator kernel
    ln_rng_dropout = 4,
};

/// @brief
///
enum class ln_bwd_fused_kind : uint8_t {
    none = 0,
    bias_dropout_resAdd_ln = 1,
    ln_dropout_gradAdd = 2,
    ln_dropout = 3,
};

namespace group {

/// @brief
///
/// @tparam fused_op_kind_
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
/// @tparam arch_
template <ln_fwd_fused_kind fused_op_kind_, typename dtype_in_,
        typename dtype_out_, typename dtype_acc_, typename layer_norm_attr_,
        gpu_arch arch_ = gpu_arch::Xe>
struct ln_fwd_fused_op_t {};

/// @brief
///
/// @tparam fused_op_kind_
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
/// @tparam arch_
template <ln_bwd_fused_kind fused_op_kind_, typename dtype_in_,
        typename dtype_out_, typename dtype_acc_, typename layer_norm_attr_,
        gpu_arch arch_ = gpu_arch::Xe>
struct ln_bwd_fused_op_t {};

} // namespace group
} // namespace gpu::xetla
