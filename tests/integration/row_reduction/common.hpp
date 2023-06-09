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
#include "utils/common.hpp"
#include "xetla.hpp"
#include "gtest/gtest.h"

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;

template <typename data_type_in, typename data_type_out, typename data_type_w,
        typename data_type_x, typename data_type_d, typename data_type_acc>
int reduction_result_validate(data_type_in *device_in,
        data_type_out *device_out, data_type_w *device_w_in,
        data_type_x *device_x_out, data_type_d *device_d_out,
        uint8_t *device_mask_in, int m, int n, float scale,
        reduction_fused_kind fused_op, sycl::queue queue) {
    int err_cnt = 0;
    bool is_bias_gelu_bwd = fused_op == reduction_fused_kind::bias_gelu_w_bwd;
    bool is_bias_dropout_bwd
            = fused_op == reduction_fused_kind::bias_dropout_bwd;

    std::vector<data_type_acc> acc(n, 0);
    std::vector<data_type_x> dgelu_x(m * n, 0);
    std::vector<data_type_x> dropout_out(m * n, 0);

    int size_in = m * n;
    int size_out = n;
    int size_mask = m * n;
    int size_w = m * n;
    int size_x = m * n;
    int size_d = m * n;

    auto in = alloc_host_and_copy<data_type_in>(device_in, size_in, queue);
    auto out = alloc_host_and_copy<data_type_out>(device_out, size_out, queue);
    auto w_in = alloc_host_and_copy<data_type_w>(device_w_in, size_w, queue);
    auto x_out = alloc_host_and_copy<data_type_x>(device_x_out, size_x, queue);
    auto d_out = alloc_host_and_copy<data_type_d>(device_d_out, size_d, queue);
    auto mask_in
            = alloc_host_and_copy<uint8_t>(device_mask_in, size_mask, queue);

    if (is_bias_gelu_bwd) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dgelu_x[j + i * n] = data_type_acc(in[j + i * n])
                        * data_type_acc(w_in[j + i * n]);
                acc[j] += dgelu_x[j + i * n];
            }
        }
    } else if (is_bias_dropout_bwd) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                bool set_zero = mask_in[i * n + j] != 0;
                dropout_out[i * n + j]
                        = set_zero ? 0 : data_type_acc(in[j + i * n]) * scale;
                acc[j] += dropout_out[i * n + j];
            }
        }

    } else {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                acc[j] += data_type_acc(in[j + i * n]);
            }
        }
    }

    buff_cmp::buff_vals<data_type_out> GPU_output(out, 1, n, n);
    buff_cmp::buff_vals<data_type_out, data_type_acc> CPU_output(
            acc.data(), 1, n, n);
    bool result = buff_cmp::xetla_buff_cmp(GPU_output, CPU_output, "dbias");
    if (is_bias_gelu_bwd) {
        buff_cmp::buff_vals<data_type_x> dgelu_x_GPU(x_out, m, n, n);
        buff_cmp::buff_vals<data_type_x> dgelu_x_CPU(dgelu_x.data(), m, n, n);
        result &= buff_cmp::xetla_buff_cmp(
                dgelu_x_GPU, dgelu_x_CPU, "bias_gelu_w_bwd");
    }
    if (is_bias_dropout_bwd) {
        buff_cmp::buff_vals<data_type_x> dropout_GPU(d_out, m, n, n);
        buff_cmp::buff_vals<data_type_x> dropout_CPU(
                dropout_out.data(), m, n, n);
        result &= buff_cmp::xetla_buff_cmp(
                dropout_GPU, dropout_CPU, "bias_dropout_bwd");
    }

    free(in);
    free(out);
    free(w_in);
    free(x_out);
    free(d_out);
    free(mask_in);

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

class test_bf16_0 {
public:
    using data_type_in = bf16;
    using data_type_out = bf16;
    using data_type_w = bf16;
    using data_type_x = bf16;
    using data_type_d = bf16;
    using data_type_acc = float;
    static constexpr uint32_t matrix_m = 1024;
    static constexpr uint32_t matrix_n = 3072;

    static constexpr uint32_t wg_m = 128;
    static constexpr uint32_t wg_n = 48;
    static constexpr uint32_t sg_m = 32;
    static constexpr uint32_t sg_n = 48;

    static constexpr bool is_dynamic = true;
    static constexpr reduction_fused_kind fused_op = reduction_fused_kind::none;
};

class test_bf16_1 {
public:
    using data_type_in = bf16;
    using data_type_out = bf16;
    using data_type_w = bf16;
    using data_type_x = bf16;
    using data_type_d = bf16;
    using data_type_acc = float;
    static constexpr uint32_t matrix_m = 1024;
    static constexpr uint32_t matrix_n = 3072;

    static constexpr uint32_t wg_m = 128;
    static constexpr uint32_t wg_n = 48;
    static constexpr uint32_t sg_m = 32;
    static constexpr uint32_t sg_n = 48;

    static constexpr bool is_dynamic = false;
    static constexpr reduction_fused_kind fused_op = reduction_fused_kind::none;
};

class test_bf16_2 {
public:
    using data_type_in = bf16;
    using data_type_out = bf16;
    using data_type_w = bf16;
    using data_type_x = bf16;
    using data_type_d = bf16;
    using data_type_acc = float;
    static constexpr uint32_t matrix_m = 1024;
    static constexpr uint32_t matrix_n = 1024;

    static constexpr uint32_t wg_m = 128;
    static constexpr uint32_t wg_n = 64;
    static constexpr uint32_t sg_m = 32;
    static constexpr uint32_t sg_n = 32;

    static constexpr bool is_dynamic = true;
    static constexpr reduction_fused_kind fused_op = reduction_fused_kind::none;
};

class test_bf16_3 {
public:
    using data_type_in = bf16;
    using data_type_out = bf16;
    using data_type_w = bf16;
    using data_type_x = bf16;
    using data_type_d = bf16;
    using data_type_acc = float;
    static constexpr uint32_t matrix_m = 1024;
    static constexpr uint32_t matrix_n = 1024;

    static constexpr uint32_t wg_m = 128;
    static constexpr uint32_t wg_n = 64;
    static constexpr uint32_t sg_m = 16;
    static constexpr uint32_t sg_n = 32;

    static constexpr bool is_dynamic = false;
    static constexpr reduction_fused_kind fused_op
            = reduction_fused_kind::bias_gelu_w_bwd;
};

class test_bf16_4 {
public:
    using data_type_in = bf16;
    using data_type_out = bf16;
    using data_type_w = bf16;
    using data_type_x = bf16;
    using data_type_d = bf16;
    using data_type_acc = float;
    static constexpr uint32_t matrix_m = 1024;
    static constexpr uint32_t matrix_n = 3072;

    static constexpr uint32_t wg_m = 128;
    static constexpr uint32_t wg_n = 48;
    static constexpr uint32_t sg_m = 16;
    static constexpr uint32_t sg_n = 48;

    static constexpr bool is_dynamic = true;
    static constexpr reduction_fused_kind fused_op
            = reduction_fused_kind::bias_dropout_bwd;
};

class test_fp16_0 {
public:
    using data_type_in = fp16;
    using data_type_out = fp16;
    using data_type_w = fp16;
    using data_type_x = fp16;
    using data_type_d = fp16;
    using data_type_acc = float;
    static constexpr uint32_t matrix_m = 1024;
    static constexpr uint32_t matrix_n = 3072;

    static constexpr uint32_t wg_m = 128;
    static constexpr uint32_t wg_n = 48;
    static constexpr uint32_t sg_m = 32;
    static constexpr uint32_t sg_n = 48;

    static constexpr bool is_dynamic = true;
    static constexpr reduction_fused_kind fused_op = reduction_fused_kind::none;
};

class test_fp16_1 {
public:
    using data_type_in = fp16;
    using data_type_out = fp16;
    using data_type_w = fp16;
    using data_type_x = fp16;
    using data_type_d = fp16;
    using data_type_acc = float;
    static constexpr uint32_t matrix_m = 1024;
    static constexpr uint32_t matrix_n = 1024;

    static constexpr uint32_t wg_m = 128;
    static constexpr uint32_t wg_n = 64;
    static constexpr uint32_t sg_m = 32;
    static constexpr uint32_t sg_n = 32;

    static constexpr bool is_dynamic = true;
    static constexpr reduction_fused_kind fused_op = reduction_fused_kind::none;
};

class test_fp16_2 {
public:
    using data_type_in = fp16;
    using data_type_out = fp16;
    using data_type_w = fp16;
    using data_type_x = fp16;
    using data_type_d = fp16;
    using data_type_acc = float;
    static constexpr uint32_t matrix_m = 1024;
    static constexpr uint32_t matrix_n = 1024;

    static constexpr uint32_t wg_m = 128;
    static constexpr uint32_t wg_n = 64;
    static constexpr uint32_t sg_m = 16;
    static constexpr uint32_t sg_n = 32;

    static constexpr bool is_dynamic = true;
    static constexpr reduction_fused_kind fused_op
            = reduction_fused_kind::bias_gelu_w_bwd;
};
