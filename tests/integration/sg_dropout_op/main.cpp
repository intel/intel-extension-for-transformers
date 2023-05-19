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

    queue Queue {};
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    data_type_x *buffer_x = static_cast<data_type_x *>(
            malloc_shared(size_in * sizeof(data_type_x), Device, Context));
    data_type_y *buffer_y = static_cast<data_type_y *>(
            malloc_shared(size_out * sizeof(data_type_y), Device, Context));
    uint8_t *buffer_mask = static_cast<uint8_t *>(
            malloc_shared(size_mask * sizeof(uint8_t), Device, Context));
    uint64_t *buffer_rand_offset = static_cast<uint64_t *>(
            malloc_shared(sizeof(uint64_t), Device, Context));
    buffer_rand_offset[0] = 0;
    for (unsigned i = 0; i < size_in; ++i) {
        buffer_x[i] = (random_float() - 0.5) * 10;
    }
    for (unsigned i = 0; i < size_out; ++i) {
        buffer_y[i] = data_type_y(0);
    }
    if (is_mask_gen) {
        for (unsigned i = 0; i < size_mask; ++i) {
            buffer_mask[i] = 0;
        }
    } else {
        uint32_t drop_threshold = drop_out_ratio * double(RAND_MAX);
        for (unsigned i = 0; i < size_mask; ++i) {
            buffer_mask[i]
                    = (random_float() * double(RAND_MAX) > drop_threshold) ? 0
                                                                           : 1;
        }
    }

    cl::sycl::range<3> GroupRange {1,
            (test::mat_m + test::wg_m - 1) / test::wg_m,
            (test::mat_n + test::wg_n - 1) / test::wg_n};
    cl::sycl::range<3> LocalRange {1,
            (test::wg_m + test::sg_m - 1) / test::sg_m,
            (test::wg_n + test::sg_n - 1) / test::sg_n};
    cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

    try {
        auto e_esimd = Queue.submit([&](handler &cgh) {
            cgh.parallel_for<test>(
                    Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                        xetla_exec_item<3> ei(item);
                        using dropout_func = dropout_func_t<data_type_x,
                                data_type_y, data_type_acc, test::wg_n,
                                test::wg_m, test::sg_n, test::sg_m,
                                test::dropout_kind>;

                        dropout_func::run(ei, buffer_x, buffer_mask, buffer_y,
                                buffer_rand_offset, matrix_m, matrix_n,
                                matrix_n, drop_out_ratio, drop_out_scale);
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
                    drop_out_scale)));

    free(buffer_x, Context);
    free(buffer_y, Context);
    free(buffer_mask, Context);
    free(buffer_rand_offset, Context);
}

TEST(dropout_normal, esimd) {
    dropout_op_run<dropout_normal>();
}

TEST(dropout_rng, esimd) {
    dropout_op_run<dropout_rng>();
}
