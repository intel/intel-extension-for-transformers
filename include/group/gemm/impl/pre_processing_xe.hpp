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

#include "group/gemm/api.hpp"
#include "group/gemm/common.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_gemm
/// @{

/// @brief gemm default pre_processing functor. Specialized for Xe architecture.
template <typename tile_shape_, gpu_arch arch_tag>
class pre_processing_default_t<tile_shape_, arch_tag,
        std::enable_if_t<(arch_tag == gpu_arch::Xe)
                || (arch_tag == gpu_arch::Arc)>> {
    using tile_shape = tile_shape_;
    using work_group_t = typename tile_shape::work_group_t;

public:
    struct arguments_t {};

    inline pre_processing_default_t() = default;

    inline pre_processing_default_t(work_group_t &g, arguments_t &args) {}

    inline void init(work_group_t &g, arguments_t &args) {}

    template <typename matA_acc_t, typename matB_acc_t, typename matA_t,
            typename matB_t>
    inline KERNEL_FUNC void operator()(matA_acc_t &matA_acc,
            matB_acc_t &matB_acc, matA_t &matA, matB_t &matB) {}
};

/// @brief gemm pre_processing functor with applying relu op to matA. Specialized for Xe architecture.
template <typename tile_shape_, gpu_arch arch_tag>
class pre_processing_matA_neg_filter_t<tile_shape_, arch_tag,
        std::enable_if_t<(arch_tag == gpu_arch::Xe)>> {
    using tile_shape = tile_shape_;
    using work_group_t = typename tile_shape::work_group_t;

public:
    struct arguments_t {};

    inline pre_processing_matA_neg_filter_t() = default;

    inline pre_processing_matA_neg_filter_t(
            work_group_t &g, arguments_t &args) {}

    inline void init(work_group_t &g, arguments_t &args) {}

    template <typename matA_acc_t, typename matB_acc_t, typename matA_t,
            typename matB_t>
    inline KERNEL_FUNC void operator()(matA_acc_t &matA_acc,
            matB_acc_t &matB_acc, matA_t &matA, matB_t &matB) {

        using data_t = typename matA_acc_t::dtype;
        if constexpr (sizeof(data_t) == 2) {
            xetla_mask<matA_acc_t::tile_elems> mask
                    = matA_acc.reg.xetla_format<int16_t>() < 0;
            matA_acc.reg.xetla_format<int16_t>().xetla_merge(0, mask);
        }
        if constexpr (sizeof(data_t) == 1) {
            xetla_mask<matA_acc_t::tile_elems> mask
                    = matA_acc.reg.xetla_format<int8_t>() < 0;
            matA_acc.reg.xetla_format<int8_t>().xetla_merge(0, mask);
        }
        if constexpr (sizeof(data_t) == 4) {
            xetla_mask<matA_acc_t::tile_elems> mask
                    = matA_acc.reg.xetla_format<int32_t>() < 0;
            matA_acc.reg.xetla_format<int32_t>().xetla_merge(0, mask);
        }
    }
};

/// @} xetla_gemm

} // namespace gpu::xetla::group
