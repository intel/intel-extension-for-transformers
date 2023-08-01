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
/// @tparam global_ratio_ Is the k dim split ratio between groups.
/// @tparam local_ratio_ Is the k dim split ratio within a group.
/// @tparam arch_tag_ Is the HW architecture.
template <int global_ratio_ = 1, int local_ratio_ = 1,
        gpu_arch arch_tag_ = gpu_arch::Xe>
struct dispatch_policy_int4_dequantize_kslicing {
    static constexpr int global_ratio = global_ratio_;
    static constexpr int local_ratio = local_ratio_;
    static constexpr gpu_arch arch_tag = arch_tag_;
};

/// @} xetla_gemm

} // namespace gpu::xetla::kernel
