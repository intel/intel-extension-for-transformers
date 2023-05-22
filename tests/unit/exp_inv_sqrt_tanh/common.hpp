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

#include <cmath>
#include "kernel_func.hpp"

template <typename dtype>
struct exp_op {
    static dtype run(dtype a) { return std::exp(a); }
};

template <typename dtype>
struct exp2_op {
    static dtype run(dtype a) { return std::exp2(a); }
};

template <typename dtype>
struct inv_op {
    static dtype run(dtype a) { return 1.0f / a; }
};

template <typename dtype>
struct sqrt_op {
    static dtype run(dtype a) { return std::sqrt(a); }
};

template <typename dtype>
struct sqrt_ieee_op {
    static dtype run(dtype a) { return std::sqrt(a); }
};

template <typename dtype>
struct rsqrt_op {
    static dtype run(dtype a) { return 1.0f / std::sqrt(a); }
};

template <typename dtype>
struct tanh_op {
    // in order to match with the kernel implementation
    static dtype run(dtype a) { return std::tanh(a - 10); }
};

template <typename dtype, typename op>
int kernel_validation(float *A, float *B, float *C, unsigned Size) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Size; ++i) {
        dtype src = A[i] + 5;
        dtype dst = op::run(src);
        if ((dst - dtype(C[i])) / dst > 0.01) {
            if (++err_cnt < 16) {
                std::cout << "vector version failed at index " << i << ", "
                          << dst << " != " << C[i] << "\n";
            }
        }
    }
    {
        dtype src = A[0] + 5;
        dtype dst = op::run(src);
        if ((dst - dtype(B[0])) / dst > 0.01) {
            if (++err_cnt < 16) {
                std::cout << "scalar version failed, " << dst << " != " << B[0]
                          << "\n";
            }
        }
    }

    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size + 1 - err_cnt) / ((float)Size + 1)) * 100.0f
                  << "% (" << (Size + 1 - err_cnt) << "/" << Size + 1 << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
////for exp
