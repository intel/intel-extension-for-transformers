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

template <typename dtype>
int kernel_validation(
        dtype *A, dtype *B, dtype *C, uint32_t size_x, uint32_t size_y) {
    int err_cnt = 0;
    for (int i = 0; i < size_y; ++i) {
        for (int j = 0; j < size_x; ++j) {
            int offset = i * size_x + j;
            if ((A[offset] + j) != C[offset]) {
                if (++err_cnt < 100) {
                    std::cout << "failed at [" << i << ", " << j
                              << "], CPU: " << A[offset] + j
                              << ", GPU: " << C[offset] << std::endl;
                }
            }
        }
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
