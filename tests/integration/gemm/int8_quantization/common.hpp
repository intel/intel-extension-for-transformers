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

#include "utils/common.hpp"
#include "xetla.hpp"
#include <gtest/gtest.h>

class Test0 {
public:
    static constexpr size_t mat_m = 4000;
    static constexpr size_t mat_k = 64;
    static constexpr size_t mat_n = 512;
    static constexpr size_t wg_m = 128;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 32;
    static constexpr size_t sg_k = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = uint8_t;
    using data_type_acc = int32_t;
    using data_type_param = float;
};

class Test1 {
public:
    static constexpr size_t mat_m = 4000;
    static constexpr size_t mat_k = 512;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 128;
    static constexpr size_t wg_n = 128;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 32;
    static constexpr size_t sg_k = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = uint8_t;
    using data_type_acc = int32_t;
    using data_type_param = float;
};

class Test2 {
public:
    static constexpr size_t mat_m = 4000;
    static constexpr size_t mat_k = 256;
    static constexpr size_t mat_n = 128;
    static constexpr size_t wg_m = 128;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = uint8_t;
    using data_type_acc = int32_t;
    using data_type_param = float;
};

class Test3 {
public:
    static constexpr size_t mat_m = 4000;
    static constexpr size_t mat_k = 3456;
    static constexpr size_t mat_n = 512;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 128;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 32;
    static constexpr size_t sg_k = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = uint8_t;
    using data_type_acc = int32_t;
    using data_type_param = float;
};

class Test4 {
public:
    static constexpr size_t mat_m = 4000;
    static constexpr size_t mat_k = 512;
    static constexpr size_t mat_n = 3456;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = uint8_t;
    using data_type_acc = int32_t;
    using data_type_param = float;
};

class Test5 {
public:
    static constexpr size_t mat_m = 4000;
    static constexpr size_t mat_k = 3456;
    static constexpr size_t mat_n = 1024;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = uint8_t;
    using data_type_acc = int32_t;
    using data_type_param = float;
};

class Test6 {
public:
    static constexpr size_t mat_m = 4000;
    static constexpr size_t mat_k = 1024;
    static constexpr size_t mat_n = 1024;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = uint8_t;
    using data_type_acc = int32_t;
    using data_type_param = float;
};

class Test7 {
public:
    static constexpr size_t mat_m = 4000;
    static constexpr size_t mat_k = 1024;
    static constexpr size_t mat_n = 512;
    static constexpr size_t wg_m = 128;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 32;
    static constexpr size_t sg_k = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = uint8_t;
    using data_type_acc = int32_t;
    using data_type_param = float;
};

class Test8 {
public:
    static constexpr size_t mat_m = 4000;
    static constexpr size_t mat_k = 512;
    static constexpr size_t mat_n = 256;
    static constexpr size_t wg_m = 128;
    static constexpr size_t wg_n = 128;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 32;
    static constexpr size_t sg_k = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = uint8_t;
    using data_type_acc = int32_t;
    using data_type_param = float;
};

class Test9 {
public:
    static constexpr size_t mat_m = 4096;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = uint8_t;
    using data_type_acc = int32_t;
    using data_type_param = float;
};

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc, typename data_type_param>
int gemm_result_validate(data_type_a *A_device, data_type_b *B_device,
        data_type_c *C_device, data_type_param *scale_device,
        data_type_param *offset_device, int m, int k, int n,
        gpu::xetla::mem_layout mem_layout_a,
        gpu::xetla::mem_layout mem_layout_b, sycl::queue &queue) {

    auto A = alloc_host_and_copy<data_type_a>(A_device, m * k, queue);
    auto B = alloc_host_and_copy<data_type_b>(B_device, k * n, queue);
    auto C = alloc_host_and_copy<data_type_c>(C_device, m * n, queue);
    auto scale = alloc_host_and_copy<data_type_param>(scale_device, n, queue);
    auto offset = alloc_host_and_copy<data_type_param>(offset_device, n, queue);

    bool is_col_major_a = mem_layout_a == gpu::xetla::mem_layout::col_major;
    bool is_col_major_b = mem_layout_b == gpu::xetla::mem_layout::col_major;
    int err_cnt = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            data_type_acc result = 0;
            for (int kk = 0; kk < k; kk++) {
                data_type_a a_temp
                        = is_col_major_a ? A[i + kk * m] : A[i * k + kk];
                data_type_b b_temp
                        = is_col_major_b ? B[kk + j * k] : B[kk * n + j];
                result = result + data_type_acc(a_temp) * data_type_acc(b_temp);
            }
            //Quantization with saturation
            float result_fp32 = result;
            float result_scaled
                    = std::round(result_fp32 * scale[j] + offset[j]);
            uint8_t result_quant;

            if (result_scaled <= 0) {

                result_quant = 0;
            } else if (result_scaled >= 255) {

                result_quant = 255;
            } else {

                result_quant = result_scaled;
            }
            //allow 1 ulp
            if (abs(result_quant - C[i * n + j]) > 1) {
                if (++err_cnt < 100) {
                    std::cout << "failed at (" << i << ", " << j << "), "
                              << " golden: " << uint(result_quant)
                              << " != GPU: " << float(C[i * n + j]) << "\n";
                }
            }
        }
    }

    free(A);
    free(B);
    free(C);
    free(offset);
    free(scale);

    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(m * n - err_cnt) / (float)(m * n)) * 100.0f
                  << "% (" << (m * n - err_cnt) << "/" << m * n << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
