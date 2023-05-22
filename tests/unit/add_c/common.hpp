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

int add_update_carry_result_validate(
        uint32_t *A, uint32_t *B, uint32_t *C, unsigned Size) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Size; ++i) {
        uint32_t src1 = -1;
        uint64_t dst = (uint64_t)src1 + (uint64_t)A[i];
        uint32_t dst_lo = dst & 0xffffffff;
        uint32_t dst_hi = dst >> 32;
        if (dst_lo != C[i]) {
            if (++err_cnt < 16) {
                std::cout << "dst failed at index " << i << ", " << dst_lo
                          << " != " << C[i] << "\n";
            }
        }
        if (dst_hi != B[i]) {
            if (++err_cnt < 16) {
                std::cout << "carry failed at index " << i << ", " << dst_hi
                          << " != " << B[i] << "\n";
            }
        }
    }

    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size * 2 - err_cnt) / ((float)Size * 2)) * 100.0f
                  << "% (" << (Size * 2 - err_cnt) << "/" << Size * 2 << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
