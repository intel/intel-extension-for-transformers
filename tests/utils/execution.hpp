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

#pragma once

#include "common.hpp"
#include "profiling.hpp"
#include "xetla.hpp"

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;

template <class Test, typename validate_func, typename KERNEL,
        int SLMSIZE = 128 * 1024, int BARNUM = 32>
void gemm_exec(const std::string &compile_str, size_t batch = 1) {
    test_result result = test_result::complete;

    using gemm_op_t = typename KERNEL::gemm_op_t;

    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_c = typename Test::data_type_c;
    using data_type_acc = typename Test::data_type_acc;

    constexpr size_t matrix_m = Test::mat_m;
    constexpr size_t matrix_n = Test::mat_n;
    constexpr size_t matrix_k = Test::mat_k;

    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;

    size_t size_a = matrix_m * matrix_k;
    size_t size_b = matrix_k * matrix_n;
    size_t size_c = matrix_m * matrix_n;
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on batch: " << batch << ", "
              << device.get_info<info::device::name>() << "\n";

    auto A = alloc_device_and_init<data_type_a>(
            batch * size_a,
            [](data_type_a *data, size_t idx) {
                data[idx] = static_cast<data_type_a>(random_float());
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type_b>(
            batch * size_b,
            [](data_type_b *data, size_t idx) {
                data[idx] = static_cast<data_type_b>(random_float());
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type_c>(
            batch * size_c,
            [](data_type_c *data, size_t idx) {
                data[idx] = static_cast<data_type_c>(0);
            },
            queue, device, context);

    size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);
    auto Acc = alloc_device_and_init<data_type_acc>(
            batch * size_acc,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0);
            },
            queue, device, context);
    auto Cnt = alloc_device_and_init<uint32_t>(
            batch * size_cnt,
            [](uint32_t *data, size_t idx) {
                data[idx] = static_cast<uint32_t>(0);
            },
            queue, device, context);

    try {
        std::vector<kernel_id> kernelId = {get_kernel_id<Test>()};
        auto inputBundle
                = get_kernel_bundle<bundle_state::input>(context, kernelId);
        setenv("SYCL_PROGRAM_COMPILE_OPTIONS", compile_str.c_str(), 1);
        kernel_bundle<bundle_state::executable> exeBundle = build(inputBundle);
        unsetenv("SYCL_PROGRAM_COMPILE_OPTIONS");

        using namespace gpu::xetla::group;
        using namespace gpu::xetla::kernel;
        using namespace gpu::xetla::subgroup;

        typename gemm_op_t::arguments_t arg(matrix_m, matrix_k, matrix_n,
                nullptr,
                Test::layout_a == mem_layout::col_major ? matrix_m : matrix_k,
                nullptr,
                Test::layout_b == mem_layout::col_major ? matrix_k : matrix_n,
                nullptr, matrix_n, nullptr, nullptr);

        cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(arg);

        for (size_t i = 0; i < batch; i++) {
            auto A_ptr = A + i * size_a;
            auto B_ptr = B + i * size_b;
            auto C_ptr = C + i * size_c;
            auto Acc_ptr = Acc + i * size_acc;
            auto Cnt_ptr = Cnt + i * size_cnt;

            arg.matA_base = A_ptr;
            arg.matB_base = B_ptr;
            arg.matC_base = C_ptr;
            arg.acc_base = Acc_ptr;
            arg.cnt_base = Cnt_ptr;

            if (!gemm_op_t::can_implement(arg)) {
                std::cout << "The arguments cannot be supported, skip ... "
                          << std::endl;
                result = test_result::skip;
                break;
            }

            auto e_esimd = queue.submit([&](handler &cgh) {
                cgh.use_kernel_bundle(exeBundle);
                cgh.parallel_for<Test>(
                        nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                            gpu::xetla::xetla_local_init<SLMSIZE>();
                            gpu::xetla::xetla_nbarrier_init<BARNUM>();
                            KERNEL::run(item, A_ptr, B_ptr, C_ptr, matrix_m,
                                    matrix_n, matrix_k, Acc_ptr, Cnt_ptr);
                        });
            });
            e_esimd.wait();
        }
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        result = test_result::fail;
    }

    // validation
    if (result == test_result::complete) {
        validate_func vfunc;
        ASSERT_EQ(0, vfunc(A, B, C, queue));
    }

    free(A, context);
    free(B, context);
    free(C, context);
    free(Acc, context);
    free(Cnt, context);

    if (result == test_result::skip) {
        GTEST_SKIP();
    } else if (result != test_result::complete) {
        FAIL();
    }
}

/// @brief The template function to execute kernel in esimd way for unit test framework
///
/// @tparam data_type data_type The data type of buffer used in kernel and buffer allocation
/// @tparam KERNEL the kernel function struct
/// @param nd_range the range of workitems
/// @param validate_result validation function, taking 3 parameters buffer A, B as input C as output
///
template <typename data_type, class KERNEL, size_t SLMSIZE = 8 * 1024,
        size_t BARNUM = 32, size_t Size = 4096>
void kernel_run(auto nd_range, auto validate_result) {

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

    try {
        auto e_esimd = queue.submit([&](handler &cgh) {
            cgh.parallel_for<>(nd_range, [=](nd_item<1> ndi) KERNEL_MAIN {
                gpu::xetla::xetla_local_init<SLMSIZE>();
                gpu::xetla::xetla_nbarrier_init<BARNUM>();
                KERNEL::run(&ndi, A, B, C);
            });
        });
        e_esimd.wait();
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    auto A_host = alloc_host_and_copy<data_type>(A, Size, queue);
    auto B_host = alloc_host_and_copy<data_type>(B, Size, queue);
    auto C_host = alloc_host_and_copy<data_type>(C, Size, queue);

    ASSERT_EQ(0, validate_result(A_host, B_host, C_host));

    free(A, context);
    free(B, context);
    free(C, context);

    free(A_host);
    free(B_host);
    free(C_host);
}
