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
using namespace cl::sycl;

template <typename data_type_in, typename data_type_out, typename data_type_acc>
int data_transformer_result_validate(data_type_in *in, data_type_out *out,
        size_t mat_m, size_t mat_n, bool is_transposed, int need_fp8_op,
        data_type_acc *amax_ptr, data_type_acc *scale) {
    int err_num = 0;
    data_type_acc cpu_max = (data_type_acc)0;
    data_type_out res = data_type_out(0);
    data_type_out ref = data_type_out(0);
    for (int i = 0; i < mat_m; i++) {
        for (int j = 0; j < mat_n; j++) {
            int idx = i * mat_n + j;

            cpu_max = (cpu_max > abs(in[idx])) ? cpu_max
                                               : abs((data_type_acc)in[idx]);

            res = out[idx];

            if (!is_transposed) {
                ref = need_fp8_op ? (data_type_out)(in[idx] * scale[0])
                                  : (data_type_out)(in[idx]);
            } else {
                ref = need_fp8_op
                        ? (data_type_out)(in[j * mat_m + i] * scale[0])
                        : (data_type_out)(in[j * mat_m + i]);
            }

            if (abs(res - ref) > abs(0.01 * res)) {
                std::cout << "i: " << i << " j: " << j << " idx: " << idx
                          << " in: " << in[idx] << " cpu: " << ref
                          << " gpu: " << res << std::endl;
                err_num++;
                if (err_num > 32) { return 1; }
            }
        }
    }

    cpu_max = cpu_max * scale[0];

    if (need_fp8_op) {
        if (abs(cpu_max - amax_ptr[0]) > abs(0.01 * cpu_max)) {
            std::cout << "cpu_max: " << cpu_max << " gpu_max: " << amax_ptr[0]
                      << std::endl;
            return 1;
        }
    }

    if (err_num == 0) { std::cout << "Test Passed!!!" << std::endl; }
    return err_num;
}

class TestBase {
public:
    static constexpr size_t mat_m = 128;
    static constexpr size_t mat_n = 128;
    static constexpr size_t mat_in_ld = 128;
    static constexpr size_t mat_out_ld = 128;

    static constexpr uint32_t wg_m = 32;
    static constexpr uint32_t wg_n = 32;
    static constexpr uint32_t sg_m = 8;
    static constexpr uint32_t sg_n = 32;

    static constexpr int need_fp8_op = false;
    static constexpr mem_layout layout_in = mem_layout::row_major;

    using data_type_in = float;
    using data_type_out = bf16;
    using data_type_acc = float;
};

class Test_fp32tobf16_128_64 : public TestBase {
public:
    static constexpr size_t mat_m = 128;
    static constexpr size_t mat_n = 64;
    static constexpr size_t mat_in_ld = 64;
    static constexpr size_t mat_out_ld = 64;

    static constexpr uint32_t wg_m = 32;
    static constexpr uint32_t wg_n = 32;
    static constexpr uint32_t sg_m = 8;
    static constexpr uint32_t sg_n = 32;
};

class Test_fp32tobf16_64_128_need_fp8_op : public TestBase {
public:
    static constexpr size_t mat_m = 64;
    static constexpr size_t mat_n = 128;
    static constexpr size_t mat_in_ld = 128;
    static constexpr size_t mat_out_ld = 128;

    static constexpr uint32_t wg_m = 32;
    static constexpr uint32_t wg_n = 32;
    static constexpr uint32_t sg_m = 32;
    static constexpr uint32_t sg_n = 32;

    static constexpr int need_fp8_op = true;
};

class Test_fp32_128_64_transpose : public TestBase {
public:
    static constexpr size_t mat_m = 128;
    static constexpr size_t mat_n = 64;
    static constexpr size_t mat_in_ld = 128;
    static constexpr size_t mat_out_ld = 64;

    static constexpr uint32_t wg_m = 32;
    static constexpr uint32_t wg_n = 32;
    static constexpr uint32_t sg_m = 8;
    static constexpr uint32_t sg_n = 32;

    using data_type_in = float;
    using data_type_out = float;
    using data_type_acc = float;

    static constexpr mem_layout layout_in = mem_layout::col_major;
};

class Test_fp32_64_128_transpose_need_fp8_op : public TestBase {
public:
    static constexpr size_t mat_m = 64;
    static constexpr size_t mat_n = 128;
    static constexpr size_t mat_in_ld = 64;
    static constexpr size_t mat_out_ld = 128;

    static constexpr uint32_t wg_m = 32;
    static constexpr uint32_t wg_n = 32;
    static constexpr uint32_t sg_m = 8;
    static constexpr uint32_t sg_n = 32;

    using data_type_in = float;
    using data_type_out = float;
    using data_type_acc = float;

    static constexpr mem_layout layout_in = mem_layout::col_major;
    static constexpr int need_fp8_op = true;
};

class Test_fp16tofp32_64_128_transpose_need_fp8_op : public TestBase {
public:
    static constexpr size_t mat_m = 64;
    static constexpr size_t mat_n = 128;
    static constexpr size_t mat_in_ld = 64;
    static constexpr size_t mat_out_ld = 128;

    static constexpr uint32_t wg_m = 32;
    static constexpr uint32_t wg_n = 32;
    static constexpr uint32_t sg_m = 8;
    static constexpr uint32_t sg_n = 32;

    using data_type_in = fp16;
    using data_type_out = float;
    using data_type_acc = float;

    static constexpr mem_layout layout_in = mem_layout::col_major;
    static constexpr int need_fp8_op = true;
};
