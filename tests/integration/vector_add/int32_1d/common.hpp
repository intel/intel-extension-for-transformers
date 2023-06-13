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
#include "xetla.hpp"

using namespace gpu;
using namespace gpu::xetla;

#define data_type int
using namespace cl::sycl;

int vadd_result_validate(data_type *A_device, data_type *B_device,
        data_type *C_device, unsigned Size, sycl::queue &queue) {
    auto A = alloc_host_and_copy<data_type>(A_device, Size, queue);
    auto B = alloc_host_and_copy<data_type>(B_device, Size, queue);
    auto C = alloc_host_and_copy<data_type>(C_device, Size, queue);

    int err_cnt = 0;
    for (unsigned i = 0; i < Size; ++i) {
        if (A[i] + B[i] != C[i]) {
            if (++err_cnt < 10) {
                std::cout << "failed at index " << i << ", " << C[i]
                          << " != " << A[i] << " + " << B[i] << "\n";
            }
        }
    }

    free(A);
    free(B);
    free(C);

    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
                  << (Size - err_cnt) << "/" << Size << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
