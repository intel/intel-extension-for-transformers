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
#include <gtest/gtest.h>

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_param>
int gemm_result_validate(data_type_a *A, data_type_b *B, data_type_c *C,
        data_type_param *scale, data_type_param *offset, int m, int k, int n,
        gpu::xetla::mem_layout mem_layout_a,
        gpu::xetla::mem_layout mem_layout_b) {
    bool is_col_major_a = mem_layout_a == gpu::xetla::mem_layout::col_major;
    bool is_col_major_b = mem_layout_b == gpu::xetla::mem_layout::col_major;
    int err_cnt = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int result = 0;
            for (int kk = 0; kk < k; kk++) {
                data_type_a a_temp
                        = is_col_major_a ? A[i + kk * m] : A[i * k + kk];
                data_type_b b_temp
                        = is_col_major_b ? B[kk + j * k] : B[kk * n + j];
                result = result + data_type_c(a_temp) * data_type_c(b_temp);
            }
            //Quantization with saturation
            float result_fp32 = result;
            float result_scaled = result_fp32 * scale[j] + offset[j];
            uint8_t result_quant;

            if (result_scaled <= 0) {

                result_quant = 0;
            } else if (result_scaled >= 255) {

                result_quant = 255;
            } else {

                result_quant = result_scaled;
            }

            if (result_quant != C[i * n + j]) {
                if (++err_cnt < 100) {
                    std::cout << "failed at (" << i << ", " << j << "), "
                              << " golden: " << uint(result_quant)
                              << " != GPU: " << float(C[i * n + j]) << "\n";
                }
            }
        }
    }

    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(m * n - err_cnt) / (float)(m * n)) * 100.0f
                  << "% (" << (m * n - err_cnt) << "/" << m * n << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
