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

using namespace gpu;
using namespace gpu::xetla;

template <typename datatype>
int global_atomic_result_validate(
        datatype *A, datatype *B, datatype *C, uint32_t Size, atomic_op op) {
    int err_cnt = 0;
    for (uint32_t i = 0; i < Size; ++i) {
        auto lambda = [&, i]() -> std::tuple<datatype, datatype> {
            switch (op) {
                case atomic_op::iinc: return {A[i], B[i] + 1};
                case atomic_op::idec: return {A[i], B[i] - 1};
                case atomic_op::iadd:
                case atomic_op::fadd:
                    return {A[i], B[i] + 3}; // Here we use 3 as add element
                case atomic_op::isub:
                case atomic_op::fsub:
                    return {A[i], B[i] - 3}; // Here we use 3 as sub element
                case atomic_op::smin:
                case atomic_op::fmin:
                case atomic_op::umin: return {A[i], std::min(B[i], A[i])};
                case atomic_op::smax:
                case atomic_op::fmax:
                case atomic_op::umax: return {A[i], std::max(B[i], A[i])};
                case atomic_op::load: return {A[i], C[i]};
                case atomic_op::store:
                case atomic_op::cmpxchg:
                case atomic_op::fcmpxchg: return {A[i], 3};
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

int global_atomic_bit_op_result_validate(
        uint32_t *A, uint32_t *B, uint32_t *C, uint32_t Size, atomic_op op) {
    int err_cnt = 0;
    for (uint32_t i = 0; i < Size; ++i) {

        auto lambda = [&, i]() -> std::tuple<uint32_t, uint32_t> {
            switch (op) {
                case atomic_op::bit_and: return {A[i], (B[i] & 3)};
                case atomic_op::bit_or: return {A[i], (B[i] | 3)};
                case atomic_op::bit_xor: return {A[i], (B[i] ^ 3)};
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

template <typename datatype>
int global_atomic_with_ret_result_validate(
        datatype *A, datatype *B, datatype *C, uint32_t Size, atomic_op op) {
    int err_cnt = 0;
    for (uint32_t i = 0; i < Size; ++i) {
        auto lambda = [&, i]() -> std::tuple<datatype, datatype> {
            switch (op) {
                case atomic_op::iinc: return {A[i], B[i] + 1};
                case atomic_op::iadd: return {A[i], B[i] + 3};
                case atomic_op::cmpxchg: return {A[i], 3};
                default: {
                    std::cerr << "Not supported Op"
                              << "\n";
                    exit(-1);
                }
            }
        };
        auto [e1, e2] = lambda();
        if ((e1 != e2) || C[i] != B[i]) {
            if (++err_cnt < 10) {
                std::cout << "failed at index " << i << ", " << e2 << "!=" << e1
                          << " or " << C[i] << " != " << B[i] << "\n";
            }
        }
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}

template <typename datatype>
int global_atomic_with_mask_result_validate(datatype *A, datatype *B,
        datatype *C, uint32_t Size, atomic_op op, int mask) {
    int err_cnt = 0;

    for (uint32_t i = 0; i < Size; ++i) {
        auto lambda = [&, i]() -> std::tuple<datatype, datatype> {
            if ((mask >> i) & 0x1) {
                switch (op) {
                    case atomic_op::iinc: return {A[i], B[i] + 1};
                    case atomic_op::iadd: return {A[i], B[i] + 3};
                    case atomic_op::cmpxchg: return {A[i], 3};
                    default: {
                        std::cerr << "Not supported Op"
                                  << "\n";
                        exit(-1);
                    }
                }
            } else {
                return {A[i], B[i]};
            }
        };

        auto [e1, e2] = lambda();
        if ((e1 != e2)) {
            if (++err_cnt < 10) {
                std::cout << "failed at index " << i << ", " << e2 << "!=" << e1
                          << "\n";
            }
        }
    }

    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
