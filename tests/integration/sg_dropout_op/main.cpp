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
#include "utils/buff_compare.hpp"
#include "utils/common.hpp"
#include "gtest/gtest.h"

using namespace cl::sycl;

template <typename test>
static void dropout_op_run() {
    size_t matrix_m = test::mat_m;
    size_t matrix_n = test::mat_n;
    float drop_out_ratio = 0.2f;
    bool is_mask_gen = test::dropout_kind == dropout_op::mask_gen;
    float drop_out_scale = 1.f / (1.f - drop_out_ratio);
    int size_in = matrix_m * matrix_n;
    int size_out = matrix_m * matrix_n;
    int size_mask = matrix_m * matrix_n;

    using data_type_x = typename test::data_type_x;
    using data_type_y = typename test::data_type_y;
    using data_type_acc = typename test::data_type_acc;

    queue queue {};
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto buffer_x = alloc_device_and_init<data_type_x>(
            size_in,
            [](data_type_x *data, size_t idx) {
                data[idx]
                        = static_cast<data_type_x>((random_float() - 0.5) * 10);
            },
            queue, device, context);
    auto buffer_y = alloc_device_and_init<data_type_y>(
            size_out,
            [](data_type_y *data, size_t idx) {
                data[idx] = static_cast<data_type_y>(0);
            },
            queue, device, context);

    uint32_t drop_threshold = drop_out_ratio * double(RAND_MAX);
    auto buffer_mask = alloc_device_and_init<uint8_t>(
            size_mask,
            [&is_mask_gen, &drop_threshold](uint8_t *data, size_t idx) {
                if (is_mask_gen) {
                    data[idx] = static_cast<uint8_t>(0);
                } else {
                    data[idx] = (random_float() * double(RAND_MAX)
                                        > drop_threshold)
                            ? 0
                            : 1;
                }
            },
            queue, device, context);
    auto buffer_rand_offset = alloc_device_and_init<uint64_t>(
            1,
            [](uint64_t *data, size_t idx) {
                data[idx] = static_cast<uint64_t>(0);
            },
            queue, device, context);

    cl::sycl::range<3> group_range {1,
            (test::mat_m + test::wg_m - 1) / test::wg_m,
            (test::mat_n + test::wg_n - 1) / test::wg_n};
    cl::sycl::range<3> local_range {1,
            (test::wg_m + test::sg_m - 1) / test::sg_m,
            (test::wg_n + test::sg_n - 1) / test::sg_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    try {
        auto e_esimd = queue.submit([&](handler &cgh) {
            cgh.parallel_for<test>(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                using dropout_func = dropout_func_t<data_type_x, data_type_y,
                        data_type_acc, test::wg_n, test::wg_m, test::sg_n,
                        test::sg_m, test::dropout_kind>;

                dropout_func::run(item, buffer_x, buffer_mask, buffer_y,
                        buffer_rand_offset, matrix_m, matrix_n, matrix_n,
                        drop_out_ratio, drop_out_scale);
            });
        });
        e_esimd.wait();
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    // validation
    ASSERT_EQ(0,
            (dropout_result_validate<data_type_x, data_type_y, data_type_acc>(
                    buffer_x, buffer_y, matrix_m, matrix_n, buffer_mask,
                    drop_out_scale, queue)));

    free(buffer_x, context);
    free(buffer_y, context);
    free(buffer_mask, context);
    free(buffer_rand_offset, context);
}

TEST(dropout_normal, esimd) {
    dropout_op_run<dropout_normal>();
}

TEST(dropout_rng, esimd) {
    dropout_op_run<dropout_rng>();
}
