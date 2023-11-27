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

template <typename Test>
static void row_reduction_run() {
    using namespace gpu::xetla::subgroup;

    size_t matrix_m = Test::matrix_m;
    size_t matrix_n = Test::matrix_n;

    constexpr uint32_t wg_m = Test::wg_m;
    constexpr uint32_t wg_n = Test::wg_n;
    constexpr uint32_t sg_m = Test::sg_m;
    constexpr uint32_t sg_n = Test::sg_n;
    constexpr bool is_dynamic = Test::is_dynamic;

    constexpr reduction_fused_kind fused_op_kind = Test::fused_op;

    using data_type_in = typename Test::data_type_in;
    using data_type_out = typename Test::data_type_out;
    using data_type_acc = typename Test::data_type_acc;
    using data_type_w = typename Test::data_type_w;
    using data_type_x = typename Test::data_type_x;
    using data_type_d = typename Test::data_type_d;

    int size_in = matrix_m * matrix_n;
    int size_out = matrix_n;
    int size_mask = matrix_m * matrix_n;
    int size_w = matrix_m * matrix_n;
    int size_x = matrix_m * matrix_n;
    int size_d = matrix_m * matrix_n;
    float drop_out_prob = 0.2f;
    float drop_out_scale = (1.f - drop_out_prob);

    queue queue {};
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto buffer_in = alloc_device_and_init<data_type_in>(
            size_in,
            [](data_type_in *data, size_t idx) {
                data[idx] = static_cast<data_type_in>(random_float());
            },
            queue, device, context);

    auto buffer_w = alloc_device_and_init<data_type_w>(
            size_w,
            [](data_type_w *data, size_t idx) {
                data[idx] = static_cast<data_type_w>(random_float());
            },
            queue, device, context);

    auto buffer_x = alloc_device_and_init<data_type_x>(
            size_x, [](data_type_x *data, size_t idx) { data[idx] = 0; }, queue,
            device, context);

    auto buffer_d = alloc_device_and_init<data_type_d>(
            size_d, [](data_type_d *data, size_t idx) { data[idx] = 0; }, queue,
            device, context);

    auto buffer_out = alloc_device_and_init<data_type_out>(
            size_out, [](data_type_out *data, size_t idx) { data[idx] = 0; },
            queue, device, context);

    uint32_t drop_threshold = drop_out_prob * double(RAND_MAX);
    auto buffer_mask = alloc_device_and_init<uint8_t>(
            size_mask,
            [drop_threshold](uint8_t *data, size_t idx) {
                data[idx] = (random_float() * double(RAND_MAX) > drop_threshold)
                        ? 0
                        : 1;
            },
            queue, device, context);

    cl::sycl::range<3> group_range {1, 1, (matrix_n + wg_n - 1) / wg_n};
    cl::sycl::range<3> local_range {
            1, (wg_m + sg_m - 1) / sg_m, (wg_n + sg_n - 1) / sg_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    try {
        auto e_esimd = queue.submit([&](handler &cgh) {
            cgh.parallel_for<Test>(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                using row_reduction_func = row_reduction_func_t<data_type_in,
                        data_type_out, data_type_acc, data_type_x, data_type_w,
                        data_type_d, wg_n, wg_m, sg_n, sg_m, is_dynamic,
                        fused_op_kind>;
                constexpr uint32_t slm_size = row_reduction_func::slm_size;
                constexpr uint32_t barrier_count
                        = row_reduction_func::barrier_count;
                if constexpr (slm_size != 0) { xetla_local_init<slm_size>(); }
                if constexpr (barrier_count != 0) {
                    __ESIMD_ENS::named_barrier_init<barrier_count>();
                }

                row_reduction_func::call(item, buffer_in, buffer_out, matrix_m,
                        matrix_n, buffer_w, buffer_x, buffer_d, buffer_mask,
                        drop_out_prob, drop_out_scale);
            });
        });
        e_esimd.wait();
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    // validation
    int err_cnt;
    ASSERT_EQ(0,
            (reduction_result_validate<data_type_in, data_type_out, data_type_w,
                    data_type_x, data_type_d, data_type_acc>(buffer_in,
                    buffer_out, buffer_w, buffer_x, buffer_d, buffer_mask,
                    matrix_m, matrix_n, drop_out_scale, fused_op_kind, queue)));

    free(buffer_in, context);
    free(buffer_w, context);
    free(buffer_x, context);
    free(buffer_d, context);
    free(buffer_out, context);
    free(buffer_mask, context);
}

TEST(test_bf16_0, esimd) {
    row_reduction_run<test_bf16_0>();
}

TEST(test_bf16_1, esimd) {
    row_reduction_run<test_bf16_1>();
}

TEST(test_bf16_2, esimd) {
    row_reduction_run<test_bf16_2>();
}

TEST(test_bf16_3, esimd) {
    row_reduction_run<test_bf16_3>();
}

TEST(test_bf16_4, esimd) {
    row_reduction_run<test_bf16_4>();
}

TEST(test_fp16_0, esimd) {
    row_reduction_run<test_fp16_0>();
}

TEST(test_fp16_1, esimd) {
    row_reduction_run<test_fp16_1>();
}

TEST(test_fp16_2, esimd) {
    row_reduction_run<test_fp16_2>();
}
