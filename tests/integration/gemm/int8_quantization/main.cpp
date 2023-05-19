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

using namespace cl::sycl;
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
    queue Queue {};
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    data_type_a *A = static_cast<data_type_a *>(
            malloc_shared(size_a * sizeof(data_type_a), Device, Context));
    data_type_b *B = static_cast<data_type_b *>(
            malloc_shared(size_b * sizeof(data_type_b), Device, Context));
    data_type_c *C = static_cast<data_type_c *>(
            malloc_shared(size_c * sizeof(data_type_c), Device, Context));

    data_type_param *offset = static_cast<data_type_param *>(
            malloc_shared(matrix_n * sizeof(data_type_param), Device, Context));

    data_type_param *scale = static_cast<data_type_param *>(
            malloc_shared(matrix_n * sizeof(data_type_param), Device, Context));

    for (unsigned i = 0; i < size_a; ++i) {
        A[i] = data_type_a((i * 3) % 17);
    }
    for (unsigned i = 0; i < size_b; ++i) {
        B[i] = data_type_b((i * 5) % 19);
    }
    for (unsigned i = 0; i < size_c; ++i) {
        C[i] = data_type_c(0);
    }

    for (unsigned i = 0; i < matrix_n; ++i) {
        offset[i] = (random_float() - 0.5f);
        scale[i] = (random_float() - 0.5f);
    }

    // here keep the same dim in CM and esimd, diff the index in kernel code
    cl::sycl::range<3> GroupRange {1, (matrix_m + wg_tile_m - 1) / wg_tile_m,
            (matrix_n + wg_tile_n - 1) / wg_tile_n};
    cl::sycl::range<3> LocalRange {1, (wg_tile_m + sg_tile_m - 1) / sg_tile_m,
            (wg_tile_n + sg_tile_n - 1) / sg_tile_n};
    cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

    try {
        auto e_esimd = Queue.submit([&](handler &cgh) {
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
                    matrix_n, mem_layout_a, mem_layout_b)));

    free(A, Context);
    free(B, Context);
    free(C, Context);
    free(scale, Context);
    free(offset, Context);
}

TEST(igemm_quantize, cm_esimd) {
    igemm_quantize_run<int8_t, int8_t, uint8_t, float>();
}
