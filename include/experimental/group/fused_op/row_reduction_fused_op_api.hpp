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
enum class reduction_fused_kind : uint8_t {
    none = 0,
    bias_gelu_w_bwd = 1,
    bias_dropout_bwd = 2
};

namespace group {

/// @brief Additional Ops that can be fused with row reduction processing flow.
///
/// @tparam fused_op_kind_ Is the type of the fused op.
/// @tparam dtype_in_ Is the data type of input.
/// @tparam dtype_out_ Is the data type of output.
/// @tparam dtype_acc_ Is the accumulation data type.
/// @tparam reduction_attr_ Is the tile size for each group to do the reduction.
/// @tparam arch_ Is the HW generation.
template <reduction_fused_kind fused_op_kind_, typename dtype_in_,
        typename dtype_out_, typename dtype_acc_, typename reduction_attr_,
        gpu_arch arch_ = gpu_arch::Xe>
struct row_reduction_fused_op_t {};

} // namespace group
} // namespace gpu::xetla
