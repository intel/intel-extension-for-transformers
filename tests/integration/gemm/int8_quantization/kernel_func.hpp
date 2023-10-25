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
using namespace gpu::xetla::subgroup;

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_param, int wg_m, int wg_n, int sg_m, int sg_n, int sg_k,
        mem_layout mem_layout_a, mem_layout mem_layout_b>
struct igemm_quantize_func {
    static constexpr uint32_t periodic_sync_interval = 8;
    static constexpr uint32_t prefetch_distance = 3;
    using dtype_acc = int32_t;
    using tile_shape = tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
    using gemm_t = typename gemm_selector_t<dtype_a, dtype_b, mem_layout_a,
            mem_layout_b, mem_space::global, mem_space::global, 8, 8, dtype_acc,
            tile_shape, sg_k, mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
            periodic_sync_interval>::gemm;
    using dequant_op_t = subgroup::dequant_op_t<
            scale_v_offset_v_op_t<dtype_param, dtype_param, gpu_arch::Xe>,
            gpu_arch::Xe>;
    using quant_op_t = subgroup::quant_op_t<none_op_t, gpu_arch::Xe>;
    using epilogue_t = gpu::xetla::group::epilogue_t<
            gpu::xetla::group::epilogue_policy_quant_op<dequant_op_t, none_op_t,
                    quant_op_t, gpu_arch::Xe>,
            tile_shape,
            mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global>>;

    using group_swizzle
            = gpu::xetla::kernel::group_swizzle_default<gpu_arch::Xe>;

    using dispatch_policy
            = gpu::xetla::kernel::dispatch_policy_default<group_swizzle>;
    using gemm_op_t = gpu::xetla::kernel::gemm_universal_t<dispatch_policy,
            gemm_t, epilogue_t>;

    static constexpr uint32_t barrier_count = gemm_op_t::get_barrier_count();
    static constexpr uint32_t slm_size = gemm_op_t::get_slm_size();

    static inline void run(sycl::nd_item<3> &item, dtype_a *A, dtype_b *B,
            dtype_c *C, dtype_param *scale, dtype_param *offset, uint32_t mat_m,
            uint32_t mat_n, uint32_t mat_k) {
        typename gemm_op_t::arguments_t arg(mat_m, mat_k, mat_n, A,
                mem_layout_a == mem_layout::col_major ? mat_m : mat_k, B,
                mem_layout_b == mem_layout::col_major ? mat_k : mat_n, C, mat_n,
                typename epilogue_t::arguments_t(
                        typename dequant_op_t::arguments_t {
                                {{scale}, {mat_n, 1, mat_n}, {offset},
                                        {mat_n, 1, mat_n}}},
                        typename none_op_t::arguments_t {},
                        typename quant_op_t::arguments_t {}));
        gemm_op_t gemm_op;
        gemm_op(item, arg);
    }
};
