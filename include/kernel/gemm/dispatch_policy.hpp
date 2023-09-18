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

#include "kernel/gemm/common.hpp"

namespace gpu::xetla::kernel {

/// @addtogroup xetla_gemm_universal
/// @{

/// @brief Default GEMM_UNIVERSAL implementation.
/// A general GEMM_UNIVERSAL implementation to provide a composition point of gemm_universal and epilogue.
/// @tparam arch_tag_ Is the HW architecture.
template <gpu_arch arch_tag_>
struct dispatch_policy_default {
    static constexpr gpu_arch arch_tag = arch_tag_;
};

/// @brief Kslicing GEMM_UNIVERSAL implementation.
/// A special GEMM_UNIVERSAL implementation to increase the hardware occupancy by splitting the GEMM_UNIVERSAL task along k dimension.
/// It includes inter-group reduction (by using global atomic) and intra-group reduction (by using local memory for data exchange).
/// @tparam num_global_kslicing_ Is the k dim split ratio between groups.
/// @tparam num_local_kslicing_ Is the k dim split ratio within a group.
/// @tparam arch_tag_ Is the HW architecture.
template <int num_global_kslicing_, int num_local_kslicing_, gpu_arch arch_tag_>
struct dispatch_policy_kslicing {
    static constexpr int num_global_kslicing = num_global_kslicing_;
    static constexpr int num_local_kslicing = num_local_kslicing_;
    static constexpr gpu_arch arch_tag = arch_tag_;
};

/// @brief Blocked dispatch GEMM_UNIVERSAL implementation.
/// A GEMM_UNIVERSAL implementation to provide a composition point of gemm and epilogue.
/// @tparam wg_num_n_ Is the x-dir workgroup number of repeat block.
/// @tparam arch_tag_ Is the HW architecture.
template <int wg_num_n_, gpu_arch arch_tag_>
struct dispatch_policy_block {
    static constexpr gpu_arch arch_tag = arch_tag_;
    static constexpr uint32_t max_wg_num = arch_attr_t<arch_tag>::max_wg_num;
    static constexpr int wg_num_n = wg_num_n_;
    static_assert(!(max_wg_num % wg_num_n),
            "max_wg_num cannot be divisible by given wg_num_n!");
    static constexpr int wg_num_m = max_wg_num / wg_num_n;
};

/// @} xetla_gemm_universal

} // namespace gpu::xetla::kernel
