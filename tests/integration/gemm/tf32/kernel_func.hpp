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

#pragma once

#include "xetla.hpp"

using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc, uint32_t wg_m, uint32_t wg_n, uint32_t sg_m,
        uint32_t sg_n, uint32_t sg_k, mem_layout layout_a, mem_layout layout_b,
        uint32_t l3_kslicing, uint32_t slm_kslicing>
struct gemm_test_func {
    using tile_shape = tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
    static constexpr uint32_t periodic_sync_interval = 8;
    static constexpr uint32_t prefetch_distance = 3;
    using brgemm_t = typename brgemm_selector_t<dtype_a, dtype_b, layout_a,
            layout_b, mem_space::global, mem_space::global,
            sizeof(dtype_a) == 4 ? 4 : 8, sizeof(dtype_a) == 4 ? 4 : 8,
            dtype_acc, tile_shape, sg_k, mma_engine::xmx, gpu_arch::Xe,
            prefetch_distance, periodic_sync_interval>::brgemm;

    using update_method = typename std::conditional<(l3_kslicing > 1),
            result_reduce_sum, result_overwrite>::type;
    using epilogue_t = epilogue_t<
            epilogue_policy_default<update_method, gpu_arch::Xe>, tile_shape,
            mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global>>;

    using gemm_op_t = gemm_t<
            dispatch_policy_kslicing<l3_kslicing, slm_kslicing, gpu_arch::Xe>,
            brgemm_t, epilogue_t>;
};
