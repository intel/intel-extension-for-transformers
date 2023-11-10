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

#include "utils/utils.hpp"
#include "xetla.hpp"

namespace gpu {
namespace xetla {

class mat0_96x2048x2048_bf16 {
public:
    static constexpr size_t mat_n = 64;
    static constexpr size_t mat_m = 16;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t wg_m = 4;
    static constexpr size_t sg_n = 32;
    static constexpr size_t sg_m = 2;
    using data_type_in = bf16;
    using data_type_out = bf16;
    using data_type_coff_in = bf16;
    using data_type_acc = float;
};

template <typename data_type_in, typename data_type_coff_in,
        typename data_type_out, typename data_type_acc>
int bwd_reduction_result_validate(data_type_in *in_ptr,
        data_type_coff_in *coff_in_ptr, data_type_out *out_ptr, int m, int n,
        data_type_acc sqrt_dk_inv) {
    int err_cnt = 0;
    buff_cmp::buff_vals<data_type_out> softmax_gpu(out_ptr, m, n, n);
    std::vector<data_type_acc> softmax_acc(m * n, 0);
    std::vector<data_type_acc> softmax_max(m, 0);
    std::vector<data_type_acc> softmax_sum(m, 0);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            softmax_acc[i * n + j] = data_type_acc(in_ptr[i * n + j])
                    * data_type_acc(coff_in_ptr[i * n + j]);
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            softmax_sum[i] += softmax_acc[i * n + j];
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            softmax_acc[i * n + j] = (softmax_acc[i * n + j] - softmax_sum[i])
                    * data_type_acc(coff_in_ptr[i * n + j]) * sqrt_dk_inv;
        }
    }

    buff_cmp::buff_vals<data_type_out, data_type_acc> softmax_cpu(
            softmax_acc.data(), m, n, n);
    bool result = buff_cmp::xetla_buff_cmp(softmax_gpu, softmax_cpu, "softmax");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

template <typename data_type_in, typename data_type_out, typename data_type_acc>
int fwd_reduction_result_validate(data_type_in *in_ptr, data_type_out *out_ptr,
        int m, int n, data_type_acc sqrt_dk_inv) {
    int err_cnt = 0;
    buff_cmp::buff_vals<data_type_out> softmax_gpu(out_ptr, m, n, n);
    std::vector<data_type_acc> softmax_acc(m * n, 0);
    std::vector<data_type_acc> softmax_max(m, 0);
    std::vector<data_type_acc> softmax_sum(m, 0);

    for (int i = 0; i < m; i++) {
        softmax_max[i] = in_ptr[i * n];
        for (int j = 1; j < n; j++) {
            softmax_max[i] = (in_ptr[i * n + j] > softmax_max[i])
                    ? data_type_acc(in_ptr[i * n + j])
                    : softmax_max[i];
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            softmax_acc[i * n + j] = std::exp(
                    (data_type_acc(in_ptr[i * n + j]) - softmax_max[i])
                    * sqrt_dk_inv);
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            softmax_sum[i] += softmax_acc[i * n + j];
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            softmax_acc[i * n + j] /= softmax_sum[i];
        }
    }

    buff_cmp::buff_vals<data_type_out, data_type_acc> softmax_cpu(
            softmax_acc.data(), m, n, n);
    bool result = buff_cmp::xetla_buff_cmp(softmax_gpu, softmax_cpu, "softmax");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

} // namespace xetla
} // namespace gpu