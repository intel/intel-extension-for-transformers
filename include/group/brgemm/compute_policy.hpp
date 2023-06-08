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

#include "group/brgemm/common.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_brgemm
/// @{

/// @brief Compute policy for xmx engine.
/// @tparam compute_attr_ Is compute-related attributes.
/// @tparam perf_tuning_knob_ Is performance-related knobs.
/// @tparam arch_tag_ Is the HW architecture.
template <typename compute_attr_, typename perf_tuning_knob_,
        gpu_arch arch_tag_ = gpu_arch::Xe>
struct compute_policy_default_xmx {};

/// @brief Specialized for Xe architecture.
template <typename compute_attr_, typename perf_tuning_knob_>
struct compute_policy_default_xmx<compute_attr_, perf_tuning_knob_,
        gpu_arch::Xe> {
    using compute_attr = compute_attr_;
    using perf_tuning_knob = perf_tuning_knob_;
    static constexpr int k_stride = perf_tuning_knob::k_stride;
    static constexpr int stages = perf_tuning_knob::stages;
    static constexpr int sync_freq = perf_tuning_knob::sync_freq;
    static constexpr gpu_arch arch_tag = gpu_arch::Xe;
    using dtype_mma_acc = typename compute_attr::dtype_acc;
    using dtype_mma_a = typename compute_attr::dtype_a;
    using dtype_mma_b = typename compute_attr::dtype_b;

    static constexpr uint32_t block_bytes_x_a = 32;
    static constexpr uint32_t block_size_y_a = 16;

    static constexpr uint32_t block_size_x_b = 16;
    static constexpr uint32_t block_bytes_y_b = 32;
    static_assert(block_bytes_x_a == block_bytes_y_b,
            "mat_a x need to match with mat_b y");
};

/// @brief Compute policy for fpu engine.
/// @tparam compute_attr_ Is compute-related attributes.
/// @tparam perf_tuning_knob_ Is performance-related knobs.
/// @tparam arch_tag_ Is the HW architecture.
template <typename compute_attr_, typename perf_tuning_knob_,
        gpu_arch arch_tag_ = gpu_arch::Xe>
struct compute_policy_default_fpu {};

/// @brief Specialized for Xe architecture.
template <typename compute_attr_, typename perf_tuning_knob_>
struct compute_policy_default_fpu<compute_attr_, perf_tuning_knob_,
        gpu_arch::Xe> {
    using compute_attr = compute_attr_;
    using perf_tuning_knob = perf_tuning_knob_;
    static constexpr int k_stride = perf_tuning_knob::k_stride;
    static constexpr int stages = perf_tuning_knob::stages;
    static constexpr int sync_freq = perf_tuning_knob::sync_freq;
    static constexpr gpu_arch arch_tag = gpu_arch::Xe;
    using dtype_mma_acc = typename compute_attr::dtype_acc;
    using dtype_mma_a = typename compute_attr::dtype_a;
    using dtype_mma_b = typename compute_attr::dtype_b;

    static constexpr uint32_t block_bytes_x_a = 64;
    static constexpr uint32_t block_size_y_a = 16;
    static constexpr uint32_t block_bytes_x_b = 64;
    static constexpr uint32_t block_size_y_b = 16;
};

/// @} xetla_brgemm

} // namespace gpu::xetla::group
