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

template <typename T>
int tile_mma_result_validate(T *A, T *B, T *C, int m, int n, int k,
        gpu::xetla::mem_layout mem_layout_a,
        gpu::xetla::mem_layout mem_layout_b) {
    bool is_col_major_a = mem_layout_a == gpu::xetla::mem_layout::col_major;
    bool is_col_major_b = mem_layout_b == gpu::xetla::mem_layout::col_major;
    int err_cnt = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float result = 0;
            for (int kk = 0; kk < k; kk++) {
                float a_temp = is_col_major_a ? A[i + kk * m] : A[i * k + kk];
                float b_temp = is_col_major_b ? B[kk + j * k] : B[kk * n + j];
                result = result + a_temp * b_temp;
            }
            if (std::abs((T(result) - C[i * n + j]) / C[i * n + j]) > 0.01) {
                if (++err_cnt < 33) {
                    std::cout << "failed at (" << i << ", " << j << "), "
                              << " golden: " << result
                              << " != GPU: " << C[i * n + j] << "\n";
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
