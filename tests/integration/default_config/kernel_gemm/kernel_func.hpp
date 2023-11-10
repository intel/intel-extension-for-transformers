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
        uint32_t global_kslicing, uint32_t local_kslicing, mma_engine engine>
struct default_config_kernel_gemm_test_func {
    using tune_option
            = dict_t<elem_v_t<tune_key::PARAM_OPTIMZER_TYPE,
                             tune_key_value::PARAM_OPTIMZER_DECISION_TREE>,
                    elem_v_t<tune_key::DISPATCH_POLICY,
                            tune_key_value::DISPATCH_POLICY_KSLICING>,
                    elem_v_t<tune_key::GLOBAL_KSLICING_RATIO, global_kslicing>,
                    elem_v_t<tune_key::LOCAL_KSLICING_RATIO, local_kslicing>,
                    elem_v_t<tune_key::MMA_ENGINE, engine>,
                    elem_t_t<tune_key::WG_TILE_SHAPE, shape<wg_n, wg_m>>,
                    elem_t_t<tune_key::SG_TILE_SHAPE, shape<sg_n, sg_m>>>;
    using gemm_op_t = gpu::xetla::kernel::default_gemm_t<
            dtype_a, // input datatype for A
            layout_a, // memory layout for A
            8, // leading dimension alignment for A, in unit of element
            dtype_b, // input datatype for B
            layout_b, // memory layout for B
            8, // leading dimension alignment for B, in unit of element
            dtype_c, // output datatype for C
            mem_layout::row_major, // memory layout for C
            8, // leading dimension alignment for C, in unit of element
            dtype_acc, // accumulator data type for intermediate resutls
            gpu_arch::Xe, // GPU arch
            tune_option>;

    static const char *func_name() {
        return "default_config_kernel_gemm_test_func";
    }

    static inline void run(sycl::nd_item<3> &item, dtype_a *A, dtype_b *B,
            dtype_c *C, uint32_t mat_m, uint32_t mat_n, uint32_t mat_k,
            dtype_acc *Acc, uint32_t *Cnt) {
        typename gemm_op_t::arguments_t arg(mat_m, mat_k, mat_n, A,
                layout_a == mem_layout::col_major ? mat_m : mat_k, B,
                layout_b == mem_layout::col_major ? mat_k : mat_n, C, mat_n,
                Acc, Cnt);
        gemm_op_t gemm_op;
        gemm_op(item, arg);
    }
};
