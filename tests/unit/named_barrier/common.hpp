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

int named_barrier_result_validate(int *A, int *B, int *C, unsigned Size) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Size; ++i) {
        if (A[i] != C[i] * 2) {
            if (++err_cnt < 16) {
                std::cout << "failed at index " << i << ", " << C[i] * 2
                          << " != " << A[i] << "\n";
            }
        }
    }

    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
                  << (Size - err_cnt) << "/" << Size << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}

int named_barrier_split_validate(
        int *A, int *B, int *C, unsigned Size, unsigned times) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Size; ++i) {
        if (C[i] != A[i] * times) {
            if (++err_cnt < 16) {
                std::cout << "failed at index " << i << ", " << C[i]
                          << " != " << A[i] * times << "\n";
            }
        }
    }

    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
                  << (Size - err_cnt) << "/" << Size << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
