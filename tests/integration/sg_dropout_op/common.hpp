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
#include "utils/common.hpp"
#include "xetla.hpp"
#include "gtest/gtest.h"

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;

template <typename data_type_x, typename data_type_y,
        typename data_type_acc = float>
int dropout_result_validate(data_type_x *In, data_type_y *Out, int m, int n,
        uint8_t *buffer_mask, float drop_out_scale) {

    std::vector<data_type_acc> acc(m * n, 0);

    int positive_num = 0;
    for (int i = 0; i < m; i++) {
        // drop out
        for (int j = 0; j < n; j++) {
            bool set_zero = buffer_mask[i * n + j] != 0;
            positive_num += set_zero ? 1 : 0;
            acc[i * n + j] = set_zero
                    ? 0
                    : data_type_acc(In[i * n + j]) * drop_out_scale;
        }
    }
    std::cout << "positive ratio is: "
              << float(positive_num) / float(m) / float(n) * 100.f << "%\n";

    buff_cmp::buff_vals<data_type_y> gpu_out(Out, m, n, n);
    buff_cmp::buff_vals<data_type_y, data_type_acc> cpu_out(
            acc.data(), m, n, n);
    bool result
            = buff_cmp::xetla_buff_cmp(gpu_out, cpu_out, "compare dropout out");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

class dropout_normal {
public:
    static constexpr size_t mat_m = 256;
    static constexpr size_t mat_n = 512;
    static constexpr size_t wg_m = 128;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 32;
    static constexpr dropout_op dropout_kind = dropout_op::normal;
    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_acc = float;
};
class dropout_rng {
public:
    static constexpr size_t mat_m = 32;
    static constexpr size_t mat_n = 32;
    static constexpr size_t wg_m = 32;
    static constexpr size_t wg_n = 32;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 32;
    static constexpr dropout_op dropout_kind = dropout_op::mask_gen;
    using data_type_x = bf16;
    using data_type_y = bf16;
    using data_type_acc = float;
};
