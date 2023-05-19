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

template <typename dtype_y, typename dtype_x, typename dtype_weight,
        typename dtype_acc, int wg_n, int wg_m, int sg_n, int sg_m,
        int wg_num_m, int wg_num_n, ln_bwd_fused_kind fused_op_kind>
struct ln_bwd_func_t {
    using layer_norm_attr = gpu::xetla::kernel::layer_norm_attr_t<wg_n, wg_m,
            sg_n, sg_m, wg_num_m, wg_num_n>;
    using ln_fused_op = gpu::xetla::group::ln_bwd_fused_op_t<fused_op_kind,
            dtype_y, dtype_x, dtype_acc, layer_norm_attr, gpu_arch::Xe>;
    using layer_norm_bwd = gpu::xetla::kernel::layer_norm_bwd_t<dtype_x,
            dtype_y, dtype_weight, dtype_acc, layer_norm_attr, gpu_arch::Xe,
            ln_fused_op>;
    static constexpr uint32_t slm_size = layer_norm_bwd::get_slm_size::size;
    static constexpr uint32_t barrier_count
            = layer_norm_bwd::get_barrier_count::count;

    static inline void call(xetla_exec_item<3> &ei, dtype_y *dy_in,
            dtype_x *x_in, dtype_weight *gamma_in, int matrix_m, int matrix_n,
            int mat_ld, dtype_acc *buffer_rs, dtype_acc *buffer_mu,
            dtype_x *dx_out, dtype_acc *dgamma, dtype_acc *dbeta,
            uint8_t *buffer_mask, dtype_x *dx_resAdd_out, dtype_acc *dbias,
            int mask_ld, float drop_out_scale_inv, float drop_out_prob,
            dtype_y *grad_in) {
        typename layer_norm_bwd::arguments_t args;
        args.dy_in_ptr = dy_in;
        args.x_in_ptr = x_in;
        args.gamma_in_ptr = gamma_in;
        args.matrix_m = matrix_m;
        args.matrix_n = matrix_n;
        args.mat_ld = mat_ld;
        args.rs_ptr = buffer_rs;
        args.mu_ptr = buffer_mu;
        args.dx_out_ptr = dx_out;
        args.dgamma_acc_ptr = dgamma;
        args.dbeta_acc_ptr = dbeta;

        typename ln_fused_op::arguments_t ln_fused_args;
        ln_fused_args.mask_ld = mask_ld;
        ln_fused_args.mat_ld = mat_ld;
        ln_fused_args.matrix_m = matrix_m;
        ln_fused_args.matrix_n = matrix_n;
        ln_fused_args.dx_resAdd_ptr = dx_resAdd_out;
        ln_fused_args.dbias_acc_ptr = dbias;
        ln_fused_args.mask_ptr = buffer_mask;
        ln_fused_args.dropout_scale_inv = drop_out_scale_inv;
        ln_fused_args.dropout_prob = drop_out_prob;
        ln_fused_args.gradAdd_ptr = grad_in;

        layer_norm_bwd::call(ei, &args, 0, 0, &ln_fused_args);
    }
};

template <typename dtype_x, typename dtype_weight, typename dtype_acc,
        int final_wg_n, int final_wg_m, int final_sg_n, int final_sg_m>
struct ln_bwd_final_func_t {
    using reduction_attr = gpu::xetla::kernel::row_reduction_attr_t<final_wg_n,
            final_wg_m, final_sg_n, final_sg_m, false>;
    using row_reduction0 = gpu::xetla::kernel::xetla_row_reduction_t<dtype_acc,
            dtype_weight, dtype_acc, reduction_attr, gpu_arch::Xe>;
    using row_reduction1 = gpu::xetla::kernel::xetla_row_reduction_t<dtype_acc,
            dtype_weight, dtype_acc, reduction_attr, gpu_arch::Xe>;
    using row_reduction2 = gpu::xetla::kernel::xetla_row_reduction_t<dtype_acc,
            dtype_x, dtype_acc, reduction_attr, gpu_arch::Xe>;
    static constexpr uint32_t slm_size = row_reduction0::get_slm_size::size;
    static constexpr uint32_t barrier_count
            = row_reduction0::get_barrier_count::count;

    static inline void call(xetla_exec_item<3> &ei, dtype_acc *dgamma_acc,
            dtype_acc *dbeta_acc, dtype_weight *dgamma, dtype_weight *dbeta,
            int matrix_m, int matrix_n, dtype_acc *dbias_acc, dtype_x *dbias) {
        typename row_reduction0::arguments_t args0;
        args0.mat_in_ptr = dgamma_acc;
        args0.mat_out_ptr = dgamma;
        args0.matrix_m = matrix_m;
        args0.matrix_n = matrix_n;
        args0.mat_in_ld = matrix_n;
        typename row_reduction1::arguments_t args1;
        args1.mat_in_ptr = dbeta_acc;
        args1.mat_out_ptr = dbeta;
        args1.matrix_m = matrix_m;
        args1.matrix_n = matrix_n;
        args1.mat_in_ld = matrix_n;
        typename row_reduction2::arguments_t args2;
        args2.mat_in_ptr = dbias_acc;
        args2.mat_out_ptr = dbias;
        args2.matrix_m = matrix_m;
        args2.matrix_n = matrix_n;
        args2.mat_in_ld = matrix_n;

        if (ei.get_group(0) == 0) {
            row_reduction0::call(ei, &args0);
        } else if (ei.get_group(0) == 1) {
            row_reduction1::call(ei, &args1);
        } else if (ei.get_group(0) == 2) {
            row_reduction2::call(ei, &args2);
        }
    }
};
