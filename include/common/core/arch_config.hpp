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

#include "common/core/common.hpp"

namespace gpu::xetla {

/// @addtogroup xetla_core_arch_config
/// @{
enum class gpu_arch : uint8_t { Xe = 0, Xe2 = 1 };
enum class grf_mode : uint8_t { normal = 0, double_grf = 1 };

template <gpu_arch ARCH_>
struct load_store_attr_t {};
template <>
struct load_store_attr_t<gpu_arch::Xe> {
    /// HW limitation checks https://gfxspecs.intel.com/Predator/Home/Index/55490
    static constexpr uint32_t max_load_height_in_elem = 32;
    static constexpr uint32_t max_load_width_in_bytes = 64;
    static constexpr uint32_t max_trans_load_width_in_bytes = 32;
    static constexpr uint32_t max_vnni_load_width_in_elems = 16;
    static constexpr uint32_t min_vnni_load_height_in_bytes = 4;

    static constexpr uint32_t max_store_height_in_elem = 8;
    static constexpr uint32_t max_store_width_in_bytes = 64;

    static constexpr uint32_t max_load_size_in_bytes = 2048;
    static constexpr uint32_t max_store_size_in_bytes = 512;

    static constexpr uint32_t cache_line_size_in_bytes = 64;
    static constexpr uint32_t alignment_in_bytes = 8;
};

template <gpu_arch ARCH_>
struct mma_attr_t {};
template <>
struct mma_attr_t<gpu_arch::Xe> {
    static constexpr uint32_t mma_m_in_elem = 8;
    static constexpr uint32_t mma_n_in_elem = 16;
    static constexpr uint32_t mma_k_in_bytes = 32;
};

template <gpu_arch ARCH_, grf_mode GRF_MODE_>
struct register_attr_t {};
template <>
struct register_attr_t<gpu_arch::Xe, grf_mode::double_grf> {
    static constexpr uint32_t acc_reg_in_bytes = 64 * 8;
    static constexpr uint32_t grf_in_bytes = 64 * 256;
    static constexpr uint32_t reg_in_bytes = 64;
};
template <gpu_arch ARCH_>
struct arch_attr_t {
    using load_store_attr = load_store_attr_t<ARCH_>;
    using mma_attr = mma_attr_t<ARCH_>;
    using register_attr = register_attr_t<ARCH_, grf_mode::double_grf>;
};

/// @} xetla_core_arch_config

} // namespace gpu::xetla
