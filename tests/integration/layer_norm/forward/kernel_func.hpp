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

template <typename dtype_x, typename dtype_y, typename dtype_weight,
        typename dtype_acc, int wg_n, int wg_m, int sg_n, int sg_m,
        int wg_num_m, int wg_num_n, ln_fwd_fused_kind fused_op_kind,
        bool store_for_bwd>
struct ln_fwd_func_t {
    using layer_norm_attr = gpu::xetla::kernel::layer_norm_attr_t<wg_n, wg_m,
            sg_n, sg_m, wg_num_m, wg_num_n>;
    using ln_fused_op = gpu::xetla::group::ln_fwd_fused_op_t<fused_op_kind,
            dtype_x, dtype_x, dtype_acc, layer_norm_attr, gpu_arch::Xe>;
    using layer_norm_fwd = gpu::xetla::kernel::layer_norm_fwd_t<dtype_x,
            dtype_y, dtype_weight, dtype_acc, layer_norm_attr, store_for_bwd,
            gpu_arch::Xe, ln_fused_op>;

    static constexpr uint32_t slm_size = layer_norm_fwd::get_slm_size::size;
    static constexpr uint32_t barrier_count
            = layer_norm_fwd::get_barrier_count::count;

    static inline void call(xetla_exec_item<3> &ei, dtype_x *x_in,
            dtype_y *y_out, int matrix_m, int matrix_n,
            dtype_weight *buffer_gamma, dtype_weight *buffer_beta,
            dtype_acc *buffer_rs, dtype_acc *buffer_mu, int mat_ld, int mask_ld,
            dtype_x *buffer_bias, dtype_x *buffer_resAdd, uint8_t *buffer_mask,
            uint64_t *buffer_rand_offset, float drop_out_ratio,
            float drop_out_scale, dtype_x *res_dropout_res_out) {

        typename layer_norm_fwd::arguments_t args;
        args.x_in_ptr = x_in;
        args.y_out_ptr = y_out;
        args.matrix_m = matrix_m;
        args.matrix_n = matrix_n;
        args.gamma_ptr = buffer_gamma;
        args.beta_ptr = buffer_beta;
        args.rs_ptr = buffer_rs;
        args.mu_ptr = buffer_mu;
        args.mat_ld = mat_ld;

        typename ln_fused_op::arguments_t ln_fused_args;
        ln_fused_args.mask_ld = mask_ld;
        ln_fused_args.mat_ld = matrix_n;
        ln_fused_args.matrix_m = matrix_m;
        ln_fused_args.matrix_n = matrix_n;
        ln_fused_args.bias_ptr = buffer_bias;
        ln_fused_args.res_add_ptr = buffer_resAdd;
        ln_fused_args.mask_ptr = buffer_mask;
        ln_fused_args.rand_offset_ptr = buffer_rand_offset;
        ln_fused_args.dropout_prob = drop_out_ratio;
        ln_fused_args.dropout_scale = drop_out_scale;
        ln_fused_args.bias_dropout_res_ptr = res_dropout_res_out;

        layer_norm_fwd::call(ei, &args, 0, 0, &ln_fused_args);
    }
};
