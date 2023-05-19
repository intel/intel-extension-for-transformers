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

template <typename dtype_in, typename dtype_out, typename dtype_acc,
        typename dtype_x, typename dtype_w, typename dtype_d, int wg_n,
        int wg_m, int sg_n, int sg_m, bool is_dynamic,
        reduction_fused_kind fused_op_kind>
struct row_reduction_func_t {
    using reduction_attr = gpu::xetla::kernel::row_reduction_attr_t<wg_n, wg_m,
            sg_n, sg_m, is_dynamic>;
    using fused_op_t
            = gpu::xetla::group::row_reduction_fused_op_t<fused_op_kind,
                    dtype_w, dtype_x, dtype_acc, reduction_attr, gpu_arch::Xe>;
    using row_reduction = gpu::xetla::kernel::xetla_row_reduction_t<dtype_in,
            dtype_out, dtype_acc, reduction_attr, gpu_arch::Xe, fused_op_t>;

    static constexpr uint32_t slm_size = row_reduction::get_slm_size::size;
    static constexpr uint32_t barrier_count
            = row_reduction::get_barrier_count::count;

    static inline void call(xetla_exec_item<3> &ei, dtype_in *buffer_in,
            dtype_out *buffer_out, int matrix_m, int matrix_n,
            dtype_w *buffer_w, dtype_x *buffer_x, dtype_d *buffer_d,
            uint8_t *buffer_mask, float drop_out_prob, float drop_out_scale) {

        typename row_reduction::arguments_t args;
        args.mat_in_ptr = buffer_in;
        args.mat_out_ptr = buffer_out;
        args.matrix_m = matrix_m;
        args.matrix_n = matrix_n;
        args.mat_in_ld = matrix_n;

        typename row_reduction::fused_op_arguments_t fused_op_args;
        fused_op_args.gelu_bwd_w_ptr = buffer_w;
        fused_op_args.gelu_bwd_x_ptr = buffer_x;
        fused_op_args.matrix_m = matrix_m;
        fused_op_args.matrix_n = matrix_n;
        fused_op_args.mat_in_ld = matrix_n;
        fused_op_args.mat_out_ld = matrix_n;
        fused_op_args.dropout_bwd_ptr = buffer_d;
        fused_op_args.mask_ptr = buffer_mask;
        fused_op_args.dropout_prob = drop_out_prob;
        fused_op_args.dropout_scale_inv = drop_out_scale;

        row_reduction::call(ei, &args, &fused_op_args);
    }
};
