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

enum class math_op {
    abs_vector = 0,
    abs_scalar = 1,
    min_vector = 2,
    min_scalar = 3,
    max_vector = 4,
    max_scalar = 5
};

template <typename datatype>
int math_result_validate(
        datatype *A, datatype *B, datatype *C, uint32_t Size, math_op op) {
    int err_cnt = 0;
    for (uint32_t i = 0; i < Size; ++i) {
        auto lambda = [&, i]() -> std::tuple<datatype, datatype> {
            switch (op) {
                case math_op::abs_vector: return {std::abs(A[i] - 8), C[i]};
                case math_op::abs_scalar: return {std::abs(A[0] - 8), C[0]};
                case math_op::min_vector: return {std::min(A[i], 8), C[i]};
                case math_op::min_scalar: return {std::min(A[0], 8), C[0]};
                case math_op::max_vector: return {std::max(A[i], 8), C[i]};
                case math_op::max_scalar: return {std::max(A[0], 8), C[0]};

                default: {
                    std::cerr << "Not supported Op"
                              << "\n";
                    exit(-1);
                }
            }
        };
        auto [e1, e2] = lambda();
        if (e1 != e2) {
            if (++err_cnt < 10) {
                std::cout << "failed at index " << i << ", " << e2 << "!=" << e1
                          << "\n";
            }
        }
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
