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

    queue Queue {};
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    data_type_in *buffer_in = static_cast<data_type_in *>(
            malloc_shared(size_in * sizeof(data_type_in), Device, Context));
    data_type_w *buffer_w = static_cast<data_type_w *>(
            malloc_shared(size_w * sizeof(data_type_w), Device, Context));
    data_type_x *buffer_x = static_cast<data_type_x *>(
            malloc_shared(size_x * sizeof(data_type_x), Device, Context));
    data_type_d *buffer_d = static_cast<data_type_d *>(
            malloc_shared(size_d * sizeof(data_type_d), Device, Context));
    data_type_out *buffer_out = static_cast<data_type_out *>(
            malloc_shared(size_out * sizeof(data_type_out), Device, Context));
    uint8_t *buffer_mask = static_cast<uint8_t *>(
            malloc_shared(size_mask * sizeof(uint8_t), Device, Context));

    for (unsigned i = 0; i < size_in; ++i) {
        buffer_in[i] = random_float();
    }
    for (unsigned i = 0; i < size_w; ++i) {
        buffer_w[i] = random_float();
    }
    for (unsigned i = 0; i < size_out; ++i) {
        buffer_out[i] = data_type_out(0);
    }
    for (unsigned i = 0; i < size_x; ++i) {
        buffer_x[i] = 0;
    }
    for (unsigned i = 0; i < size_d; ++i) {
        buffer_d[i] = 0;
    }
    uint32_t drop_threshold = drop_out_prob * double(RAND_MAX);
    for (unsigned i = 0; i < size_mask; ++i) {
        buffer_mask[i]
                = (random_float() * double(RAND_MAX) > drop_threshold) ? 0 : 1;
    }

    cl::sycl::range<3> GroupRange {1, 1, (matrix_n + wg_n - 1) / wg_n};
    cl::sycl::range<3> LocalRange {
            1, (wg_m + sg_m - 1) / sg_m, (wg_n + sg_n - 1) / sg_n};
    cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

    try {
        auto e_esimd = Queue.submit([&](handler &cgh) {
            cgh.parallel_for<Test>(
                    Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                        using row_reduction_func = row_reduction_func_t<
                                data_type_in, data_type_out, data_type_acc,
                                data_type_x, data_type_w, data_type_d, wg_n,
                                wg_m, sg_n, sg_m, is_dynamic, fused_op_kind>;
                        constexpr uint32_t slm_size
                                = row_reduction_func::slm_size;
                        constexpr uint32_t barrier_count
                                = row_reduction_func::barrier_count;
                        if constexpr (slm_size != 0) {
                            xetla_local_init<slm_size>();
                        }
                        if constexpr (barrier_count != 0) {
                            __ESIMD_ENS::named_barrier_init<barrier_count>();
                        }

                        xetla_exec_item<3> ei(item);
                        row_reduction_func::call(ei, buffer_in, buffer_out,
                                matrix_m, matrix_n, buffer_w, buffer_x,
                                buffer_d, buffer_mask, drop_out_prob,
                                drop_out_scale);
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
                    matrix_m, matrix_n, drop_out_scale, fused_op_kind)));

    free(buffer_in, Context);
    free(buffer_out, Context);
    free(buffer_w, Context);
    free(buffer_x, Context);
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
