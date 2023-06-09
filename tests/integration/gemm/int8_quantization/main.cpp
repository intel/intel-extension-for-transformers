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
#include "utils/common.hpp"
#include <gtest/gtest.h>

class Test1;

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_param>
static void igemm_quantize_run() {
    using namespace gpu::xetla::subgroup;

    size_t matrix_m = 512;
    size_t matrix_n = 512;
    size_t matrix_k = 256;
    constexpr size_t wg_tile_m = 256;
    constexpr size_t wg_tile_n = 256;
    constexpr size_t sg_tile_m = 32;
    constexpr size_t sg_tile_n = 64;
    constexpr size_t sg_tile_k = 32;

    constexpr gpu::xetla::mem_layout mem_layout_a
            = gpu::xetla::mem_layout::row_major;
    constexpr gpu::xetla::mem_layout mem_layout_b
            = gpu::xetla::mem_layout::row_major;

    std::string string_mem_layout_a = "gpu::xetla::mem_layout::row_major";
    std::string string_mem_layout_b = "gpu::xetla::mem_layout::row_major";

    constexpr bool is_col_major_a
            = mem_layout_a == gpu::xetla::mem_layout::col_major;
    constexpr bool is_col_major_b
            = mem_layout_b == gpu::xetla::mem_layout::col_major;

    int size_a = matrix_m * matrix_k;
    int size_b = matrix_k * matrix_n;
    int size_c = matrix_m * matrix_n;
    queue queue {};
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto A = alloc_device_and_init<data_type_a>(
            size_a,
            [](data_type_a *data, size_t idx) {
                data[idx] = (data_type_a)((idx * 3) % 17);
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type_b>(
            size_b,
            [](data_type_b *data, size_t idx) {
                data[idx] = (data_type_b)((idx * 5) % 19);
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

    // here keep the same dim in CM and esimd, diff the index in kernel code
    cl::sycl::range<3> GroupRange {1, (matrix_m + wg_tile_m - 1) / wg_tile_m,
            (matrix_n + wg_tile_n - 1) / wg_tile_n};
    cl::sycl::range<3> LocalRange {1, (wg_tile_m + sg_tile_m - 1) / sg_tile_m,
            (wg_tile_n + sg_tile_n - 1) / sg_tile_n};
    cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

    try {
        auto e_esimd = queue.submit([&](handler &cgh) {
            cgh.parallel_for<Test1>(
                    Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                        xetla_exec_item<3> ei(item);
                        using igemm_quantize_functor
                                = igemm_quantize_func<data_type_a, data_type_b,
                                        data_type_c, data_type_param, wg_tile_m,
                                        wg_tile_n, sg_tile_m, sg_tile_n,
                                        sg_tile_k, mem_layout_a, mem_layout_b>;

                        constexpr uint32_t barrier_count
                                = igemm_quantize_functor::barrier_count;
                        constexpr uint32_t slm_size
                                = igemm_quantize_functor::slm_size;
                        xetla_nbarrier_init<barrier_count>();
                        xetla_local_init<slm_size>();

                        igemm_quantize_functor::run(ei, A, B, C, scale, offset,
                                matrix_m, matrix_n, matrix_k);
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
            (gemm_result_validate<data_type_a, data_type_b, data_type_c,
                    data_type_param>(A, B, C, scale, offset, matrix_m, matrix_k,
                    matrix_n, mem_layout_a, mem_layout_b, queue)));

    free(A, context);
    free(B, context);
    free(C, context);
    free(scale, context);
    free(offset, context);
}

TEST(igemm_quantize, cm_esimd) {
    igemm_quantize_run<int8_t, int8_t, uint8_t, float>();
}
