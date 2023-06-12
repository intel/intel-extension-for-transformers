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
#include <gtest/gtest.h>

static void vadd_run() {
    constexpr unsigned VL = 8;
    constexpr unsigned BL = 16;
    constexpr unsigned Size = BL * BL * 8;
    constexpr unsigned GroupSize = 1;
    queue queue {};
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto A = alloc_device_and_init<data_type>(
            Size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(idx);
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type>(
            Size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(idx);
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type>(
            Size, [](data_type *data, size_t idx) {}, queue, device, context);

    // each thread process 16x16 block
    cl::sycl::range<1> Globalrange {Size / BL / BL};
    cl::sycl::range<1> local_range {GroupSize};
    cl::sycl::nd_range<1> nd_range(Globalrange, local_range);

    try {
        auto e_esimd = queue.submit([&](handler &cgh) {
            cgh.parallel_for<Test1>(
                    nd_range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
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
    ASSERT_EQ(0, vadd_result_validate(A, B, C, Size, queue));

    free(A, context);
    free(B, context);
    free(C, context);
}

TEST(vadd_v2, esimd) {
    vadd_run();
}
