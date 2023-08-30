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

/// @addtogroup xetla_gemm
/// @{

/// @brief Default GEMM implementation.
/// A general GEMM implementation to provide a composition point of brgemm and epilogue.
/// @tparam arch_tag_ Is the HW architecture.
template <gpu_arch arch_tag_ = gpu_arch::Xe>
struct dispatch_policy_default {
    static constexpr gpu_arch arch_tag = arch_tag_;
};

/// @brief Kslicing GEMM implementation.
/// A special GEMM implementation to increase the hardware occupancy by splitting the GEMM task along k dimension.
/// It includes inter-group reduction (by using global atomic) and intra-group reduction (by using local memory for data exchange).
/// @tparam global_ratio_ Is the k dim split ratio between groups.
/// @tparam local_ratio_ Is the k dim split ratio within a group.
/// @tparam arch_tag_ Is the HW architecture.
template <int global_ratio_ = 1, int local_ratio_ = 1,
        gpu_arch arch_tag_ = gpu_arch::Xe>
struct dispatch_policy_kslicing {
    static constexpr int global_ratio = global_ratio_;
    static constexpr int local_ratio = local_ratio_;
    static constexpr gpu_arch arch_tag = arch_tag_;
};

/// @brief Persistent-thread GEMM implementation.
/// A GEMM implementation to provide a composition point of brgemm and epilogue.
/// @tparam wg_num_n_ Is the x-dir workgroup number of repeat block.
/// @tparam arch_tag_ Is the HW architecture.
template <int wg_num_n_ = 8, gpu_arch arch_tag_ = gpu_arch::Xe>
struct dispatch_policy_block {
    static constexpr gpu_arch arch_tag = arch_tag_;
    static constexpr uint32_t max_wg_num = arch_attr_t<arch_tag>::max_wg_num;
    static constexpr int wg_num_n = wg_num_n_;
    static_assert(!(max_wg_num % wg_num_n),
            "max_wg_num cannot be divisible by given wg_num_n!");
    static constexpr int wg_num_m = max_wg_num / wg_num_n;
};

/// @} xetla_gemm

} // namespace gpu::xetla::kernel
