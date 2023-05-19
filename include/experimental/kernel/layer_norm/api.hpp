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

#include "experimental/kernel/layer_norm/common.hpp"
#include "experimental/kernel/layer_norm/config.hpp"

namespace gpu::xetla::kernel {

/// @brief
///
/// @tparam dtype_x_
/// @tparam dtype_y_
/// @tparam dtype_weight_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
/// @tparam store_for_bwd_
/// @tparam arch_
/// @tparam ln_fwd_fused_op_
template <typename dtype_x_, typename dtype_y_, typename dtype_weight_,
        typename dtype_acc_, typename layer_norm_attr_,
        bool store_for_bwd_ = true, gpu_arch arch_ = gpu_arch::Xe,
        typename ln_fwd_fused_op_
        = group::ln_fwd_fused_op_t<ln_fwd_fused_kind::none, dtype_x_, dtype_y_,
                dtype_acc_, layer_norm_attr_, arch_>>
struct layer_norm_fwd_t {};

/// @brief
///
/// @tparam dtype_x_
/// @tparam dtype_y_
/// @tparam dtype_weight_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
/// @tparam arch_
/// @tparam ln_bwd_fused_op_
template <typename dtype_x_, typename dtype_y_, typename dtype_weight_,
        typename dtype_acc_, typename layer_norm_attr_,
        gpu_arch arch_ = gpu_arch::Xe,
        typename ln_bwd_fused_op_
        = group::ln_bwd_fused_op_t<ln_bwd_fused_kind::none, dtype_y_, dtype_x_,
                /*in bwd, y is input, x is output*/ dtype_acc_,
                layer_norm_attr_, arch_>>
struct layer_norm_bwd_t {};

} // namespace gpu::xetla::kernel
