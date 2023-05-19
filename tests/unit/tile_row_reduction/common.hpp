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
int tile_row_reduction_result_validate(dtype *A, dtype *B, dtype *C,
        unsigned Sizex, unsigned Blockx, unsigned Blocky) {
    int err_cnt = 0;
    std::vector<float> acc(Blockx, 0);
    for (unsigned i = 0; i < Blocky; ++i) {
        for (unsigned j = 0; j < Blockx; ++j) {
            acc[j] += A[i * Sizex + j];
        }
    }

    for (unsigned j = 0; j < Blockx; ++j) {
        if ((dtype(acc[j]) - C[j]) / C[j] > 0.001) {
            if (++err_cnt < 100) {
                std::cout << "failed at index " << j << ", " << C[j]
                          << " != " << acc[j] << "\n";
            }
        }
    }

    unsigned Size = Blockx;
    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
                  << (Size - err_cnt) << "/" << Size << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
