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
struct reduce_sum {
    static dtype run(dtype a, dtype b) { return a + b; }
};

template <typename dtype>
struct reduce_prod {
    static dtype run(dtype a, dtype b) { return a * b; }
};

template <typename dtype>
struct reduce_min {
    static dtype run(dtype a, dtype b) { return a > b ? b : a; }
};

template <typename dtype>
struct reduce_max {
    static dtype run(dtype a, dtype b) { return a > b ? a : b; }
};

template <typename dtype, typename op>
int kernel_validation(float *A, float *B, float *C, unsigned Size) {
    int err_cnt = 0;
    dtype dst = A[0] + 5;
    for (unsigned i = 1; i < Size; ++i) {
        dtype src = A[i] + 5;
        dst = op::run(src, dst);
    }
    if ((dst - dtype(C[0])) / dst > 0.01) {
        if (++err_cnt < 16) {
            std::cout << "Test failed " << dst << " != " << C[0] << "\n";
        }
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
