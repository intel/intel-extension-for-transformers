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

#include "group/epilogue/common.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_epilogue
/// @{

/// @brief Default epilogue policy for store C.
/// @tparam update_method_ Is the store method of matC.
/// @tparam arch_ Is the HW architecture.
template <typename update_method_ = result_overwrite,
        gpu_arch arch_ = gpu_arch::Xe>
struct epilogue_policy_default {
    using update_method = update_method_;
    static constexpr gpu_arch arch_tag = arch_;
    static_assert(std::is_same<update_method, result_overwrite>::value
                    || std::is_same<update_method, result_reduce_sum>::value,
            "The result can be either overwrite or reduce_sum");
};

/// @brief Epilogue policy for tile_op + store C fusion.
/// @tparam tile_op_t_ Is the tile_op functor.
/// @tparam arch_ Is the HW architecture.
template <typename tile_op_t_, gpu_arch arch_ = gpu_arch::Xe>
struct epilogue_policy_tile_op {
    using tile_op = tile_op_t_;
    using update_method = result_overwrite;
    static constexpr gpu_arch arch_tag = arch_;
};

/// @brief Epilogue functor, specialized for quantization operator.
/// @tparam tile_op_t_ is the tile op type.
/// @tparam quant_op_t_ is the quantization op type
/// @tparam arch_ Is the HW architecture.
template <typename tile_op_t_, typename quant_op_t_,
        gpu_arch arch_ = gpu_arch::Xe>
struct epilogue_policy_quant_op {
    using tile_op = tile_op_t_;
    using quant_op = quant_op_t_;
    using update_method = result_overwrite;
    static constexpr gpu_arch arch_tag = arch_;
};
/// @} xetla_epilogue

} // namespace gpu::xetla::group
