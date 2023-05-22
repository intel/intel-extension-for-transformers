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

enum class bit_shift_op {
    shl_vector = 0,
    shl_scalar = 1,
    shr_vector = 2,
    shr_scalar = 3,
    rol_vector = 4,
    rol_scalar = 5,
    ror_vector = 6,
    ror_scalar = 7,
    lsr_vector = 8,
    lsr_scalar = 9,
    asr_vector = 10,
    asr_scalar = 11
};

template <typename datatype>
int bit_shift_result_validate(
        datatype *A, datatype *B, datatype *C, uint32_t Size, bit_shift_op op) {
    int err_cnt = 0;
    for (uint32_t i = 0; i < Size; ++i) {
        auto lambda = [&, i]() -> std::tuple<datatype, datatype> {
            switch (op) {
                case bit_shift_op::shl_vector: return {A[i] << 1, C[i]};
                case bit_shift_op::shl_scalar: return {A[0] << 1, C[i]};
                case bit_shift_op::shr_vector: return {A[i] >> 1, C[i]};
                case bit_shift_op::shr_scalar: return {A[0] >> 1, C[i]};
                case bit_shift_op::rol_vector:
                    return {(A[i] << 1) | (A[i] >> 63), C[i]};
                case bit_shift_op::rol_scalar:
                    return {(A[0] << 1) | (A[0] >> 63), C[i]};
                case bit_shift_op::ror_vector:
                    return {(A[i] >> 1) | (A[i] << 63), C[i]};
                case bit_shift_op::ror_scalar:
                    return {(A[0] >> 1) | (A[0] << 63), C[i]};
                case bit_shift_op::lsr_vector: return {A[i] >> 1, C[i]};
                case bit_shift_op::lsr_scalar: return {A[0] >> 1, C[i]};
                case bit_shift_op::asr_vector: return {A[i] >> 1, C[i]};
                case bit_shift_op::asr_scalar: return {A[0] >> 1, C[i]};

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
