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

#include "utils/buff_compare.hpp"
#include "xetla.hpp"
#include "gtest/gtest.h"

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;

template <typename data_type_x, typename data_type_weight, typename data_type_y,
        typename data_type_acc = float>
int ln_bwd_result_validate(data_type_y *dy_in, data_type_y *grad_in,
        data_type_x *x_in, data_type_weight *gamma_in, data_type_acc *mu_in,
        data_type_acc *rs_in, data_type_x *dx_out, data_type_weight *dgamma_out,
        data_type_weight *dbeta_out, int m, int n, int mask_n, int sg_size_n,
        data_type_x *dx_resAdd_out, data_type_x *dbias_out,
        uint8_t *buffer_mask, float drop_out_scale_inv,
        ln_bwd_fused_kind ln_fused_op_kind, data_type_acc epsilon = 1e-5) {

    bool is_bias_dropout_resAdd_fuse
            = ln_fused_op_kind == ln_bwd_fused_kind::bias_dropout_resAdd_ln;
    bool is_ln_dropout_gradAdd_fuse
            = ln_fused_op_kind == ln_bwd_fused_kind::ln_dropout_gradAdd;
    std::vector<data_type_acc> dgamma_acc(n, 0);
    std::vector<data_type_acc> dbeta_acc(n, 0);
    std::vector<data_type_acc> dx_acc(m * n, 0);
    std::vector<data_type_acc> dy_acc(m * n, 0);
    int sg_mask_len = (sg_size_n + 31) / 32;
    std::vector<data_type_acc> dbias_acc(n, 0);
    std::vector<data_type_acc> dx_resAdd_acc(m * n, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dy_acc[i * n + j] = (data_type_acc)dy_in[i * n + j];
        }
    }
    if (is_ln_dropout_gradAdd_fuse) {
        int positive_num = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dy_acc[i * n + j]
                        = dy_acc[i * n + j] + (data_type_acc)grad_in[i * n + j];
            }

            // drop out
            for (int j = 0; j < n; j++) {
                bool set_zero = buffer_mask[i * n + j] != 0;
                positive_num += set_zero ? 1 : 0;
                dy_acc[i * n + j]
                        = set_zero ? 0 : dy_acc[i * n + j] * drop_out_scale_inv;
            }
        }
        std::cout << "positive ratio is: "
                  << float(positive_num) / float(m) / float(n) * 100.f << "%\n";
    }
    for (int i = 0; i < m; i++) {
        data_type_acc sum_temp0 = 0;
        data_type_acc sum_temp1 = 0;
        for (int j = 0; j < n; j++) {
            data_type_acc x_temp
                    = rs_in[i] * ((data_type_acc)x_in[i * n + j] - mu_in[i]);
            data_type_acc y_temp
                    = dy_acc[i * n + j] * (data_type_acc)gamma_in[j];

            dgamma_acc[j] += x_temp * dy_acc[i * n + j];
            dbeta_acc[j] += dy_acc[i * n + j];

            sum_temp0 += y_temp;
            sum_temp1 += y_temp * x_temp;
        }
        sum_temp0 /= (data_type_acc)n;
        sum_temp1 /= (data_type_acc)n;
        for (int j = 0; j < n; j++) {
            data_type_acc x_temp
                    = rs_in[i] * ((data_type_acc)x_in[i * n + j] - mu_in[i]);
            data_type_acc y_temp
                    = dy_acc[i * n + j] * (data_type_acc)gamma_in[j];
            dx_acc[i * n + j]
                    = rs_in[i] * (y_temp - (sum_temp1 * x_temp + sum_temp0));
        }
    }
    if (is_bias_dropout_resAdd_fuse) {
        int positive_num = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dx_resAdd_acc[i * n + j] = dx_acc[i * n + j];
            }

            for (int j = 0; j < n; j++) {
                bool set_zero = buffer_mask[i * n + j] != 0;
                positive_num += set_zero ? 1 : 0;
                dx_acc[i * n + j]
                        = set_zero ? 0 : dx_acc[i * n + j] * drop_out_scale_inv;
                dbias_acc[j] += dx_acc[i * n + j];
            }
        }
        std::cout << "positive ratio is: "
                  << float(positive_num) / float(m) / float(n) * 100.f << "%\n";
    }

    bool result = true;
    buff_cmp::buff_vals<data_type_x> gpu_out(dx_out, m, n, n);
    buff_cmp::buff_vals<data_type_x, data_type_acc> cpu_out(
            dx_acc.data(), m, n, n);
    result &= buff_cmp::xetla_buff_cmp(gpu_out, cpu_out, "compare ln_bwd out");

    buff_cmp::buff_vals<data_type_weight> gpu_dgamma(dgamma_out, 1, n, n);
    buff_cmp::buff_vals<data_type_weight, data_type_acc> cpu_dgamma(
            dgamma_acc.data(), 1, n, n);
    result &= buff_cmp::xetla_buff_cmp(
            gpu_dgamma, cpu_dgamma, "compare dgamma");

    buff_cmp::buff_vals<data_type_weight> gpu_dbeta(dbeta_out, 1, n, n);
    buff_cmp::buff_vals<data_type_weight, data_type_acc> cpu_dbeta(
            dbeta_acc.data(), 1, n, n);
    result &= buff_cmp::xetla_buff_cmp(gpu_dbeta, cpu_dbeta, "compare dbeta");
    if (is_bias_dropout_resAdd_fuse) {
        buff_cmp::buff_vals<data_type_x> gpu_mask_out(dx_resAdd_out, m, n, n);
        buff_cmp::buff_vals<data_type_x, data_type_acc> cpu_mask_out(
                dx_resAdd_acc.data(), m, n, n);
        result &= buff_cmp::xetla_buff_cmp(
                gpu_mask_out, cpu_mask_out, "compare ln_bwd resAdd out");

        buff_cmp::buff_vals<data_type_x> gpu_dbias(dbias_out, 1, n, n);
        buff_cmp::buff_vals<data_type_x, data_type_acc> cpu_dbias(
                dbias_acc.data(), 1, n, n);
        result &= buff_cmp::xetla_buff_cmp(
                gpu_dbias, cpu_dbias, "compare dbias");
    }

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

class ln_bwd_0_bf16 {
public:
    static constexpr size_t mat_m = 128;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 256;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    static constexpr size_t final_mat_m = wg_num_m;
    static constexpr size_t final_mat_n = mat_n;
    static constexpr size_t final_wg_m = 8;
    static constexpr size_t final_wg_n = 32;
    static constexpr size_t final_sg_m = 4;
    static constexpr size_t final_sg_n = 16;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = bf16;
    using data_type_acc = float;
    static constexpr ln_bwd_fused_kind ln_fused_op_kind
            = ln_bwd_fused_kind::ln_dropout_gradAdd;
};

class ln_bwd_1_bf16 {
public:
    static constexpr size_t mat_m = 512;
    static constexpr size_t mat_n = 512;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 256;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    static constexpr size_t final_mat_m = wg_num_m;
    static constexpr size_t final_mat_n = mat_n;
    static constexpr size_t final_wg_m = 8;
    static constexpr size_t final_wg_n = 32;
    static constexpr size_t final_sg_m = 4;
    static constexpr size_t final_sg_n = 32;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = bf16;
    using data_type_acc = float;
    static constexpr ln_bwd_fused_kind ln_fused_op_kind
            = ln_bwd_fused_kind::none;
};

class ln_bwd_2_bf16 {
public:
    static constexpr size_t mat_m = 512;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 1;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 256;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    static constexpr size_t final_mat_m = wg_num_m;
    static constexpr size_t final_mat_n = mat_n;
    static constexpr size_t final_wg_m = 8;
    static constexpr size_t final_wg_n = 16;
    static constexpr size_t final_sg_m = 4;
    static constexpr size_t final_sg_n = 16;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = bf16;
    using data_type_acc = float;
    static constexpr ln_bwd_fused_kind ln_fused_op_kind
            = ln_bwd_fused_kind::none;
};

class ln_bwd_3_bf16 {
public:
    static constexpr size_t mat_m = 512;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 256;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    static constexpr size_t final_mat_m = wg_num_m;
    static constexpr size_t final_mat_n = mat_n;
    static constexpr size_t final_wg_m = 8;
    static constexpr size_t final_wg_n = 32;
    static constexpr size_t final_sg_m = 4;
    static constexpr size_t final_sg_n = 16;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = bf16;
    using data_type_acc = float;
    static constexpr ln_bwd_fused_kind ln_fused_op_kind
            = ln_bwd_fused_kind::bias_dropout_resAdd_ln;
};

class ln_bwd_4_bf16 {
public:
    static constexpr size_t mat_m = 512;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 256;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    static constexpr size_t final_mat_m = wg_num_m;
    static constexpr size_t final_mat_n = mat_n;
    static constexpr size_t final_wg_m = 8;
    static constexpr size_t final_wg_n = 32;
    static constexpr size_t final_sg_m = 4;
    static constexpr size_t final_sg_n = 16;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = float;
    using data_type_acc = float;
    static constexpr ln_bwd_fused_kind ln_fused_op_kind
            = ln_bwd_fused_kind::none;
};

class ln_bwd_0_fp16 {
public:
    static constexpr size_t mat_m = 128;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 256;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    static constexpr size_t final_mat_m = wg_num_m;
    static constexpr size_t final_mat_n = mat_n;
    static constexpr size_t final_wg_m = 8;
    static constexpr size_t final_wg_n = 32;
    static constexpr size_t final_sg_m = 4;
    static constexpr size_t final_sg_n = 16;

    using data_type_x = fp16;
    using data_type_y = fp16;
    using data_type_weight = fp16;
    using data_type_acc = float;
    static constexpr ln_bwd_fused_kind ln_fused_op_kind
            = ln_bwd_fused_kind::ln_dropout_gradAdd;
};

class ln_bwd_1_fp16 {
public:
    static constexpr size_t mat_m = 512;
    static constexpr size_t mat_n = 512;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 256;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    static constexpr size_t final_mat_m = wg_num_m;
    static constexpr size_t final_mat_n = mat_n;
    static constexpr size_t final_wg_m = 8;
    static constexpr size_t final_wg_n = 32;
    static constexpr size_t final_sg_m = 4;
    static constexpr size_t final_sg_n = 32;

    using data_type_x = fp16;
    using data_type_y = fp16;
    using data_type_weight = fp16;
    using data_type_acc = float;
    static constexpr ln_bwd_fused_kind ln_fused_op_kind
            = ln_bwd_fused_kind::none;
};

class ln_bwd_2_fp16 {
public:
    static constexpr size_t mat_m = 512;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 1;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 256;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    static constexpr size_t final_mat_m = wg_num_m;
    static constexpr size_t final_mat_n = mat_n;
    static constexpr size_t final_wg_m = 8;
    static constexpr size_t final_wg_n = 16;
    static constexpr size_t final_sg_m = 4;
    static constexpr size_t final_sg_n = 16;

    using data_type_x = fp16;
    using data_type_y = fp16;
    using data_type_weight = fp16;
    using data_type_acc = float;
    static constexpr ln_bwd_fused_kind ln_fused_op_kind
            = ln_bwd_fused_kind::none;
};

class ln_bwd_3_fp16 {
public:
    static constexpr size_t mat_m = 512;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 256;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    static constexpr size_t final_mat_m = wg_num_m;
    static constexpr size_t final_mat_n = mat_n;
    static constexpr size_t final_wg_m = 8;
    static constexpr size_t final_wg_n = 32;
    static constexpr size_t final_sg_m = 4;
    static constexpr size_t final_sg_n = 16;

    using data_type_x = fp16;
    using data_type_y = fp16;
    using data_type_weight = fp16;
    using data_type_acc = float;
    static constexpr ln_bwd_fused_kind ln_fused_op_kind
            = ln_bwd_fused_kind::bias_dropout_resAdd_ln;
};
