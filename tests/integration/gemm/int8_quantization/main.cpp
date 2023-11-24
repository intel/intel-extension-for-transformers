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

template <typename Test>
static void igemm_quantize_run(int iter = 100) {
    using namespace gpu::xetla::subgroup;

    size_t matrix_m = Test::mat_m;
    size_t matrix_n = Test::mat_n;
    size_t matrix_k = Test::mat_k;
    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;
    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_c = typename Test::data_type_c;
    using data_type_acc = typename Test::data_type_acc;
    using data_type_param = typename Test::data_type_param;

    constexpr gpu::xetla::mem_layout mem_layout_a = Test::layout_a;
    constexpr gpu::xetla::mem_layout mem_layout_b = Test::layout_b;

    constexpr bool is_col_major_a
            = mem_layout_a == gpu::xetla::mem_layout::col_major;
    constexpr bool is_col_major_b
            = mem_layout_b == gpu::xetla::mem_layout::col_major;

    int size_a = matrix_m * matrix_k;
    int size_b = matrix_k * matrix_n;
    int size_c = matrix_m * matrix_n;

    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto A = alloc_device_and_init<data_type_a>(
            size_a,
            [](data_type_a *data, size_t idx) {
                data[idx] = (random_float() - 0.5f) * 256;
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type_b>(
            size_b,
            [](data_type_b *data, size_t idx) {
                data[idx] = (random_float() - 0.5f) * 256;
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type_c>(
            size_c,
            [](data_type_c *data, size_t idx) { data[idx] = (data_type_c)0; },
            queue, device, context);

    auto offset = alloc_device_and_init<data_type_param>(
            matrix_n,
            [](data_type_param *data, size_t idx) {
                data[idx] = (random_float() - 0.5f);
            },
            queue, device, context);
    auto scale = alloc_device_and_init<data_type_param>(
            matrix_n,
            [](data_type_param *data, size_t idx) {
                data[idx] = (random_float() - 0.5f);
            },
            queue, device, context);

    size_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
    size_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;
    size_t subgroup_range_m = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
    size_t subgroup_range_n = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    cl::sycl::range<3> group_range {1, group_range_m, group_range_n};
    cl::sycl::range<3> local_range {1, subgroup_range_m, subgroup_range_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);
    std::cout << "group_num_x: " << group_range_n
              << ", group_num_y: " << group_range_m << ", group_num_z: " << 1
              << "\n";
    std::cout << "group_size_x: " << subgroup_range_n
              << ", group_size_y: " << subgroup_range_m << std::endl;

    size_t ops = 2 * matrix_m * matrix_n * matrix_k;
    profiling_helper prof("igemm_quantize", ops, "gflops");

    try {
        for (int i = 0; i < iter; i++) {
            prof.cpu_start();
            auto e_esimd = queue.submit([&](handler &cgh) {
                cgh.parallel_for<
                        Test>(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                    using igemm_quantize_functor
                            = igemm_quantize_func<data_type_a, data_type_b,
                                    data_type_c, data_type_param, wg_tile_m,
                                    wg_tile_n, sg_tile_m, sg_tile_n, sg_tile_k,
                                    mem_layout_a, mem_layout_b>;

                    constexpr uint32_t barrier_count
                            = igemm_quantize_functor::barrier_count;
                    constexpr uint32_t slm_size
                            = igemm_quantize_functor::slm_size;
                    if constexpr (barrier_count != 0) {
                        xetla_nbarrier_init<barrier_count>();
                    }
                    if constexpr (slm_size != 0) {
                        xetla_local_init<slm_size>();
                    }
                    igemm_quantize_functor::run(item, A, B, C, scale, offset,
                            matrix_m, matrix_n, matrix_k);
                });
            });
            e_esimd.wait();
            prof.cpu_end();
            prof.add_gpu_event(e_esimd);
        }
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    // validation
    int err_cnt;
    ASSERT_EQ(0,
            (gemm_result_validate<data_type_a, data_type_b, data_type_c,
                    data_type_acc, data_type_param>(A, B, C, scale, offset,
                    matrix_m, matrix_k, matrix_n, mem_layout_a, mem_layout_b,
                    queue)));

    free(A, context);
    free(B, context);
    free(C, context);
    free(scale, context);
    free(offset, context);
}

template <typename T>
class igemm_quantize_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(igemm_quantize_test);

TYPED_TEST_P(igemm_quantize_test, esimd) {
    igemm_quantize_run<TypeParam>(100);
}

REGISTER_TYPED_TEST_SUITE_P(igemm_quantize_test, esimd);
using tests = ::testing::Types<Test0>;

INSTANTIATE_TYPED_TEST_SUITE_P(
        igemm_quantize_test_suite, igemm_quantize_test, tests);
