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

#include "common.hpp"
#include "kernel_func.hpp"
#include "utils/utils.hpp"
#include <gtest/gtest.h>

static void vadd_run() {
    constexpr unsigned VL = 16;
    constexpr unsigned BL = 64;
    constexpr unsigned Size = BL * BL;
    constexpr unsigned GroupSize = 1;
    queue Queue {};
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    data_type *A = static_cast<data_type *>(
            malloc_shared(Size * sizeof(data_type), Device, Context));
    data_type *B = static_cast<data_type *>(
            malloc_shared(Size * sizeof(data_type), Device, Context));
    data_type *C = static_cast<data_type *>(
            malloc_shared(Size * sizeof(data_type), Device, Context));

    for (unsigned i = 0; i < Size; ++i) {
        A[i] = B[i] = i;
    }
    // each thread process 16x16 block
    cl::sycl::range<1> GlobalRange {Size / BL / BL};
    cl::sycl::range<1> LocalRange {GroupSize};
    cl::sycl::nd_range<1> Range(GlobalRange, LocalRange);

    try {
        auto e_esimd = Queue.submit([&](handler &cgh) {
            cgh.parallel_for<Test1>(
                    Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                        xetla_exec_item ei(ndi);
                        vector_add_func<data_type, VL, BL>(&ei, A, B, C);
                    });
        });
        e_esimd.wait();
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    // validation
    ASSERT_EQ(0, vadd_result_validate(A, B, C, VL, BL));

    free(A, Context);
    free(B, Context);
    free(C, Context);
}

TEST(vadd, esimd) {
    vadd_run();
}
