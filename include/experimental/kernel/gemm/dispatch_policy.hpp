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

#include "experimental/kernel/gemm/common.hpp"

namespace gpu::xetla::kernel {

/// @addtogroup xetla_gemm
/// @{

/// @brief 4bit kslicing GEMM implementation.
/// A special GEMM implementation to increase the hardware occupancy by splitting the GEMM task along k dimension.
/// It includes inter-group reduction (by using global atomic) and intra-group reduction (by using local memory for data exchange).
/// @note The difference compare with dispatch_policy_kslicing is we will add additional handling for 4bit.
/// @tparam num_global_kslicing_ Is the k dim split ratio between groups.
/// @tparam num_local_kslicing_ Is the k dim split ratio within a group.
/// @tparam arch_tag_ Is the HW architecture.
template <typename group_swizzle_policy_, int num_global_kslicing_ = 1,
        int num_local_kslicing_ = 1>
struct dispatch_policy_int4_dequantize_kslicing {
    using group_swizzle_policy = group_swizzle_policy_;
    static constexpr int num_global_kslicing = num_global_kslicing_;
    static constexpr int num_local_kslicing = num_local_kslicing_;
    static constexpr gpu_arch arch_tag = group_swizzle_policy::arch_tag;
};

/// @} xetla_gemm

} // namespace gpu::xetla::kernel
