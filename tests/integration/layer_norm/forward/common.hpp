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

#include "kernel_func.hpp"
#include "utils/buff_compare.hpp"
#include "gtest/gtest.h"

using namespace cl::sycl;

template <typename data_type_x, typename data_type_weight, typename data_type_y,
        typename data_type_acc = float>
int ln_fwd_result_validate(data_type_x *In, data_type_y *Out,
        data_type_weight *Gamma, data_type_weight *Beta, data_type_acc *Mu,
        data_type_acc *Rs, int m, int n, int mask_n, int sg_size_n,
        data_type_x *buffer_bias, data_type_x *buffer_resAdd,
        uint8_t *buffer_mask, data_type_x *buffer_bias_dropout_res,
        float drop_out_scale, ln_fwd_fused_kind ln_fused_op_kind,
        data_type_acc epsilon = 1e-5) {
    bool is_bias_dropout_resAdd_fuse
            = (ln_fused_op_kind == ln_fwd_fused_kind::bias_dropout_resAdd_ln)
            || (ln_fused_op_kind
                    == ln_fwd_fused_kind::bias_rng_dropout_resAdd_ln);
    bool is_ln_dropout_fuse
            = (ln_fused_op_kind == ln_fwd_fused_kind::ln_dropout)
            || (ln_fused_op_kind == ln_fwd_fused_kind::ln_rng_dropout);
    std::vector<data_type_acc> acc(m * n, 0);
    std::vector<data_type_acc> rs(m, 0);
    std::vector<data_type_acc> mu(m, 0);
    std::vector<data_type_acc> sigma2(m, 0);
    std::vector<data_type_acc> input(m * n, 0);
    std::vector<data_type_acc> bias_dropout_res(m * n, 0);
    if (is_bias_dropout_resAdd_fuse) {
        int positive_num = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                input[i * n + j] = (data_type_acc)In[i * n + j]
                        + (data_type_acc)buffer_bias[j];
            }
            // drop out
            for (int j = 0; j < n; j++) {
                bool set_zero = buffer_mask[i * n + j] != 0;
                positive_num += set_zero ? 1 : 0;
                input[i * n + j]
                        = set_zero ? 0 : input[i * n + j] * drop_out_scale;
            }

            for (int j = 0; j < n; j++) {
                input[i * n + j] += (data_type_acc)buffer_resAdd[i * n + j];
                bias_dropout_res[i * n + j] = input[i * n + j];
            }
        }
        std::cout << "positive ratio is: "
                  << float(positive_num) / float(m) / float(n) * 100.f << "%\n";
    } else {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                input[i * n + j] = (data_type_acc)In[i * n + j];
            }
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mu[i] += input[i * n + j];
        }
        mu[i] *= 1.f / (data_type_acc)n;
        for (int j = 0; j < n; j++) {
            data_type_acc diff = input[i * n + j] - mu[i];
            sigma2[i] += diff * diff;
        }
        sigma2[i] *= 1.f / (data_type_acc)n;
        rs[i] = 1.f / std::sqrt(sigma2[i] + epsilon);
        for (int j = 0; j < n; j++) {
            acc[i * n + j] = (data_type_acc)Gamma[j]
                            * (rs[i] * (input[i * n + j] - mu[i]))
                    + (data_type_acc)Beta[j];
        }
    }

    if (is_ln_dropout_fuse) {
        int positive_num = 0;
        for (int i = 0; i < m; i++) {
            // drop out
            for (int j = 0; j < n; j++) {
                bool set_zero = buffer_mask[i * n + j] != 0;
                positive_num += set_zero ? 1 : 0;
                acc[i * n + j] = set_zero ? 0 : acc[i * n + j] * drop_out_scale;
            }
        }
        std::cout << "positive ratio is: "
                  << float(positive_num) / float(m) / float(n) * 100.f << "%\n";
    }

    buff_cmp::buff_vals<data_type_y> gpu_out(Out, m, n, n);
    buff_cmp::buff_vals<data_type_y, data_type_acc> cpu_out(
            acc.data(), m, n, n);
    bool result0
            = buff_cmp::xetla_buff_cmp(gpu_out, cpu_out, "compare ln_fwd out");

    buff_cmp::buff_vals<data_type_acc> gpu_mu(Mu, m, 1, 1);
    buff_cmp::buff_vals<data_type_acc> cpu_mu(mu.data(), m, 1, 1);
    bool result1
            = buff_cmp::xetla_buff_cmp(gpu_mu, cpu_mu, "compare ln_fwd mu");

    buff_cmp::buff_vals<data_type_acc> gpu_rs(Rs, m, 1, 1);
    buff_cmp::buff_vals<data_type_acc> cpu_rs(rs.data(), m, 1, 1);
    bool result2
            = buff_cmp::xetla_buff_cmp(gpu_rs, cpu_rs, "compare ln_fwd rs");

    bool result = (result0 && result1 && result2);

    if (is_bias_dropout_resAdd_fuse) {
        buff_cmp::buff_vals<data_type_x> gpu_bias_dropout_res(
                buffer_bias_dropout_res, m, n, n);
        buff_cmp::buff_vals<data_type_x, data_type_acc> cpu_bias_dropout_res(
                bias_dropout_res.data(), m, n, n);
        result &= buff_cmp::xetla_buff_cmp(gpu_bias_dropout_res,
                cpu_bias_dropout_res, "compare bias_dropout_res out");
    }

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

class ln_fwd_0 {
public:
    static constexpr size_t mat_m = 32;
    static constexpr size_t mat_n = 128;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 128;

    static constexpr size_t wg_num_m = 4;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = bf16;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::ln_dropout;
};

class ln_fwd_1 {
public:
    static constexpr size_t mat_m = 128;
    static constexpr size_t mat_n = 560;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 140;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = bf16;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::none;
};

class ln_fwd_2 {
public:
    static constexpr size_t mat_m = 32;
    static constexpr size_t mat_n = 96;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 96;

    static constexpr size_t wg_num_m = 4;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = bf16;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::bias_dropout_resAdd_ln;
};

class ln_fwd_3 {
public:
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_n = 128;
    static constexpr size_t wg_m = 2;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 64;

    static constexpr size_t wg_num_m = 2;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = bf16;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::bias_dropout_resAdd_ln;
};

class ln_fwd_4 {
public:
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 2;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 128;

    static constexpr size_t wg_num_m = 2;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = float;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::none;
};

class ln_fwd_5 {
public:
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 2;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 128;

    static constexpr size_t wg_num_m = 2;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = bf16;
    using data_type_y = float;
    using data_type_weight = float;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::none;
};

class ln_fwd_6 {
public:
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 2;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 128;

    static constexpr size_t wg_num_m = 2;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = float;
    using data_type_y = float;
    using data_type_weight = float;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::none;
};

class ln_fwd_7 {
public:
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_n = 128;
    static constexpr size_t wg_m = 2;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 64;

    static constexpr size_t wg_num_m = 2;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = bf16;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::bias_rng_dropout_resAdd_ln;
};

class ln_fwd_8 {
public:
    static constexpr size_t mat_m = 32;
    static constexpr size_t mat_n = 128;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 128;

    static constexpr size_t wg_num_m = 4;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_weight = bf16;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::ln_rng_dropout;
};

class ln_fwd_0_fp16 {
public:
    static constexpr size_t mat_m = 32;
    static constexpr size_t mat_n = 128;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 128;

    static constexpr size_t wg_num_m = 4;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = fp16;
    using data_type_y = fp16;
    using data_type_weight = fp16;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::ln_dropout;
};
class ln_fwd_1_fp16 {
public:
    static constexpr size_t mat_m = 128;
    static constexpr size_t mat_n = 512;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 128;

    static constexpr size_t wg_num_m = 16;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = fp16;
    using data_type_y = fp16;
    using data_type_weight = fp16;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::none;
};

class ln_fwd_2_fp16 {
public:
    static constexpr size_t mat_m = 32;
    static constexpr size_t mat_n = 96;
    static constexpr size_t wg_m = 4;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 96;

    static constexpr size_t wg_num_m = 4;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = fp16;
    using data_type_y = fp16;
    using data_type_weight = fp16;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::bias_dropout_resAdd_ln;
};

class ln_fwd_3_fp16 {
public:
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_n = 128;
    static constexpr size_t wg_m = 2;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 64;

    static constexpr size_t wg_num_m = 2;
    static constexpr size_t wg_num_n = 1;

    using data_type_x = fp16;
    using data_type_y = fp16;
    using data_type_weight = fp16;
    using data_type_acc = float;
    static constexpr ln_fwd_fused_kind ln_fused_op_kind
            = ln_fwd_fused_kind::bias_dropout_resAdd_ln;
};
