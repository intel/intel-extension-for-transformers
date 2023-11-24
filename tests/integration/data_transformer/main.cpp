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

using namespace gpu::xetla;
using namespace cl::sycl;

template <class Test>
static void data_transformer_run() {
    size_t matrix_m = Test::mat_m;
    size_t matrix_n = Test::mat_n;
    size_t matrix_in_ld = Test::mat_in_ld;
    size_t matrix_out_ld = Test::mat_out_ld;

    constexpr uint32_t wg_tile_m = Test::wg_m;
    constexpr uint32_t wg_tile_n = Test::wg_n;
    constexpr uint32_t sg_tile_m = Test::sg_m;
    constexpr uint32_t sg_tile_n = Test::sg_n;

    using data_type_in = typename Test::data_type_in;
    using data_type_out = typename Test::data_type_out;
    using data_type_acc = typename Test::data_type_acc;

    constexpr int need_fp8_op = Test::need_fp8_op;

    //Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    int size = matrix_m * matrix_n;

    auto buffer_in = alloc_device_and_init<data_type_in>(
            size,
            [](data_type_in *data, size_t idx) {
                data[idx] = static_cast<data_type_in>(
                        (random_float() - 0.5) * 10);
            },
            queue, device, context);
    auto buffer_out = alloc_device_and_init<data_type_out>(
            size,
            [](data_type_out *data, size_t idx) {
                data[idx] = static_cast<data_type_out>(0);
            },
            queue, device, context);
    auto amax_ptr = alloc_device_and_init<data_type_acc>(
            1,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0);
            },
            queue, device, context);
    auto scale = alloc_device_and_init<data_type_acc>(
            1,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0.8);
            },
            queue, device, context);

    cl::sycl::range<3> group_range {1, (matrix_m + wg_tile_m - 1) / wg_tile_m,
            (matrix_n + wg_tile_n - 1) / wg_tile_n};
    cl::sycl::range<3> local_range {1, (wg_tile_m + sg_tile_m - 1) / sg_tile_m,
            (wg_tile_n + sg_tile_n - 1) / sg_tile_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    std::vector<kernel_id> kernelId = {get_kernel_id<Test>()};
    auto inputBundle
            = get_kernel_bundle<bundle_state::input>(context, kernelId);
    setenv("SYCL_PROGRAM_COMPILE_OPTIONS",
            " -vc-codegen -Xfinalizer ' "
            "-printregusage -enableBCR  "
            "-DPASTokenReduction '",
            1);
    kernel_bundle<bundle_state::executable> exeBundle = build(inputBundle);
    unsetenv("SYCL_PROGRAM_COMPILE_OPTIONS");

    try {
        auto e_esimd = queue.submit([&](handler &cgh) {
            cgh.use_kernel_bundle(exeBundle);
            cgh.parallel_for<
                    Test>(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                using data_transformer_attr
                        = gpu::xetla::kernel::data_transformer_attr_t<wg_tile_n,
                                wg_tile_m, sg_tile_n, sg_tile_m>;
                using data_transformer
                        = gpu::xetla::kernel::xetla_data_transformer<
                                data_type_in, data_type_out, data_type_acc,
                                data_transformer_attr, Test::layout_in,
                                need_fp8_op, gpu_arch::Xe>;

                constexpr uint32_t barrier_count
                        = data_transformer::get_barrier_count::count;
                if constexpr (barrier_count != 0) {
                    xetla_nbarrier_init<barrier_count>();
                }

                constexpr uint32_t slm_size
                        = data_transformer::get_slm_size::size;
                if constexpr (slm_size != 0) { xetla_local_init<slm_size>(); }

                data_transformer_func<data_type_in, data_type_out,
                        data_type_acc, wg_tile_n, wg_tile_m, sg_tile_n,
                        sg_tile_m, Test::layout_in, need_fp8_op>(item,
                        buffer_in, buffer_out, matrix_m, matrix_n, matrix_in_ld,
                        matrix_out_ld, scale, amax_ptr);
            });
        });
        e_esimd.wait();

    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    // validation
    constexpr bool is_transposed = Test::layout_in == mem_layout::col_major;
    ASSERT_EQ(0,
            (data_transformer_result_validate<data_type_in, data_type_out>(
                    buffer_in, buffer_out, Test::mat_m, Test::mat_n,
                    is_transposed, need_fp8_op, amax_ptr, scale, queue)));

    free(buffer_in, context);
    free(buffer_out, context);
    free(amax_ptr, context);
    free(scale, context);
}

TEST(TestBase, esimd) {
    data_transformer_run<TestBase>();
}

TEST(Test_fp32tobf16_128_64, esimd) {
    data_transformer_run<Test_fp32tobf16_128_64>();
}

TEST(Test_fp32tobf16_64_128_need_fp8_op, esimd) {
    data_transformer_run<Test_fp32tobf16_64_128_need_fp8_op>();
}

TEST(Test_fp32_128_64_transpose, esimd) {
    data_transformer_run<Test_fp32_128_64_transpose>();
}

TEST(Test_fp32_64_128_transpose_need_fp8_op, esimd) {
    data_transformer_run<Test_fp32_64_128_transpose_need_fp8_op>();
}

TEST(Test_fp16tofp32_64_128_transpose_need_fp8_op, esimd) {
    data_transformer_run<Test_fp16tofp32_64_128_transpose_need_fp8_op>();
}
