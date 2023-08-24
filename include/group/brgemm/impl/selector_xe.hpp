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
#include "group/brgemm/compute_policy.hpp"

namespace gpu::xetla::group {
namespace detail {

using load_store_attr_xe = typename arch_attr_t<gpu_arch::Xe>::load_store_attr;
constexpr int alignment_bytes_xe = load_store_attr_xe::alignment_in_bytes;

} // namespace detail

/// @addtogroup xetla_brgemm
/// @{

/// @brief Selects 2d block && xmx based brgemm.
template <typename dtype_a, typename dtype_b, mem_layout mem_layout_a,
        mem_layout mem_layout_b, mem_space mem_space_a, mem_space mem_space_b,
        int alignment_a, int alignment_b, typename dtype_acc,
        typename tile_shape, int k_stride, int stages, int sync_freq>
class brgemm_selector_t<dtype_a, dtype_b, mem_layout_a, mem_layout_b,
        mem_space_a, mem_space_b, alignment_a, alignment_b, dtype_acc,
        tile_shape, k_stride, mma_engine::xmx, gpu_arch::Xe, stages, sync_freq,
        std::enable_if_t<((sizeof(dtype_a) * alignment_a)
                                         % detail::alignment_bytes_xe
                                 == 0)
                && ((sizeof(dtype_b) * alignment_b) % detail::alignment_bytes_xe
                        == 0)>> {
    using mem_desc_a = mem_desc_t<dtype_a, mem_layout_a, mem_space_a>;
    using mem_desc_b = mem_desc_t<dtype_b, mem_layout_b, mem_space_b>;
    using compute_attr = compute_attr_t<dtype_a, dtype_b, dtype_acc>;
    using perf_tuning_knob = perf_tuning_knob_t<k_stride, stages, sync_freq>;
    using compute_policy = compute_policy_default_xmx<compute_attr,
            perf_tuning_knob, gpu_arch::Xe>;
    using pre_processing = pre_processing_default_t<tile_shape, gpu_arch::Xe>;

public:
    using brgemm = brgemm_t<compute_policy, tile_shape, mem_desc_a, mem_desc_b,
            pre_processing>;
};

/// @brief Selects 2d block && fpu based brgemm.
template <typename dtype_a, typename dtype_b, mem_layout mem_layout_a,
        mem_layout mem_layout_b, mem_space mem_space_a, mem_space mem_space_b,
        int alignment_a, int alignment_b, typename dtype_acc,
        typename tile_shape, int k_stride, int stages, int sync_freq>
class brgemm_selector_t<dtype_a, dtype_b, mem_layout_a, mem_layout_b,
        mem_space_a, mem_space_b, alignment_a, alignment_b, dtype_acc,
        tile_shape, k_stride, mma_engine::fpu, gpu_arch::Xe, stages, sync_freq,
        std::enable_if_t<((sizeof(dtype_a) * alignment_a)
                                         % detail::alignment_bytes_xe
                                 == 0)
                && ((sizeof(dtype_b) * alignment_b) % detail::alignment_bytes_xe
                        == 0)>> {
    static_assert(std::is_same<dtype_a, dtype_acc>::value
                    && std::is_same<dtype_b, dtype_acc>::value,
            "When use brgemm_selector, dtype_a and dtype_b in fpu based gemm"
            "should be the same as dtype_acc");
    using mem_desc_a = mem_desc_t<dtype_a, mem_layout_a, mem_space_a>;
    using mem_desc_b = mem_desc_t<dtype_b, mem_layout_b, mem_space_b>;
    using compute_attr = compute_attr_t<dtype_a, dtype_b, dtype_acc>;
    using perf_tuning_knob = perf_tuning_knob_t<k_stride, stages, sync_freq>;
    using compute_policy = compute_policy_default_fpu<compute_attr,
            perf_tuning_knob, gpu_arch::Xe>;
    using pre_processing = pre_processing_default_t<tile_shape, gpu_arch::Xe>;

public:
    using brgemm = brgemm_t<compute_policy, tile_shape, mem_desc_a, mem_desc_b,
            pre_processing>;
};

/// @} xetla_brgemm
} // namespace gpu::xetla::group