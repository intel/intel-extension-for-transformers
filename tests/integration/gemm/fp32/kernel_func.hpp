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

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc, int wg_m, int wg_n, int sg_m, int sg_n, int sg_k,
        mem_layout layout_a, mem_layout layout_b, uint32_t l3_kslicing,
        uint32_t slm_kslicing, mma_engine engine>
struct fp32_gemm_test_func {

    static const char *func_name() { return "fp32_gemm_test_func"; }

    static inline void run(xetla_exec_item<3> &ei, dtype_a *A, dtype_b *B,
            dtype_c *C, uint32_t mat_m, uint32_t mat_n, uint32_t mat_k) {
        using namespace gpu::xetla::group;
        using namespace gpu::xetla::kernel;
        using namespace gpu::xetla::subgroup;
        using tile_shape = tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
        static constexpr uint32_t periodic_sync_interval = 0;
        static constexpr uint32_t prefetch_distance = 1;

        using compute_attr = compute_attr_t<dtype_acc, dtype_acc, dtype_acc>;
        using perf_tuning_knob = perf_tuning_knob_t<sg_k, prefetch_distance,
                periodic_sync_interval>;
        using compute_policy =
                typename std::conditional<(engine == mma_engine::fpu),
                        compute_policy_default_fpu<compute_attr,
                                perf_tuning_knob, gpu_arch::Xe>,
                        compute_policy_default_xmx<compute_attr,
                                perf_tuning_knob, gpu_arch::Xe>>::type;
        using mem_desc_input_a
                = mem_desc_t<dtype_a, layout_a, mem_space::global>;
        using mem_desc_input_b
                = mem_desc_t<dtype_b, layout_b, mem_space::global>;
        using mem_desc_output_c
                = mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global>;

        using brgemm_t = brgemm_t<compute_policy, tile_shape, mem_desc_input_a,
                mem_desc_input_b>;

        using update_method = typename std::conditional<(l3_kslicing > 1),
                result_reduce_sum, result_overwrite>::type;
        using epilogue_t = epilogue_t<
                epilogue_policy_default<update_method, gpu_arch::Xe>,
                tile_shape, mem_desc_output_c>;

        using gemm_op_t = gemm_t<dispatch_policy_kslicing<l3_kslicing,
                                         slm_kslicing, gpu_arch::Xe>,
                brgemm_t, epilogue_t>;
        typename gemm_op_t::arguments_t arg(mat_m, mat_k, mat_n, A,
                layout_a == mem_layout::col_major ? mat_m : mat_k, B,
                layout_b == mem_layout::col_major ? mat_k : mat_n, C, mat_n);
        gemm_op_t gemm_op;
        gemm_op(ei, arg);
    }
};
