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

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;

class Test1;
#define data_type bf16

int vadd_result_validate(data_type *A, data_type *B, data_type *C,
        unsigned Size, unsigned pitch) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Size; ++i) {
        for (unsigned j = 0; j < Size; ++j) {
            if (A[i * pitch + j] + B[i * pitch + j] != C[i * pitch + j]) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index " << i * pitch + j << ", "
                              << C[i * pitch + j] << " != " << A[i * pitch + j]
                              << " + " << B[i * pitch + j] << "\n";
                }
            }
        }
    }
    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)((Size * Size) - err_cnt) / (float)(Size * Size))
                        * 100.0f
                  << "% (" << ((Size * Size) - err_cnt) << "/" << (Size * Size)
                  << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
