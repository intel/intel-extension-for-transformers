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

#include "profiling.hpp"

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;

template <class Test, typename data_type_a, typename data_type_b,
        typename data_type_c, typename data_type_acc,
        template <typename, typename, typename> class initialize_func,
        template <class, typename, typename, typename, typename>
        class validate_func,
        template <class, typename, typename, typename, typename> class KERNEL,
        int SLMSIZE = 128 * 1024, int BARNUM = 32>
void gemm_exec(size_t matrix_m, size_t matrix_n, size_t matrix_k,
        std::string compile_str) {

    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;

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
                data[idx] = static_cast<data_type_a>(random_float());
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type_b>(
            size_b,
            [](data_type_b *data, size_t idx) {
                data[idx] = static_cast<data_type_b>(random_float());
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type_c>(
            size_c,
            [](data_type_c *data, size_t idx) {
                data[idx] = static_cast<data_type_c>(0.0f);
            },
            queue, device, context);

    size_t group_range_m = (matrix_m % wg_tile_m == 0)
            ? matrix_m / wg_tile_m
            : (matrix_m / wg_tile_m) + 1;
    size_t group_range_n = (matrix_n % wg_tile_n == 0)
            ? matrix_n / wg_tile_n
            : (matrix_n / wg_tile_n) + 1;
    size_t subgroup_range_m = (wg_tile_m % sg_tile_m == 0)
            ? wg_tile_m / sg_tile_m
            : (wg_tile_m / sg_tile_m) + 1;
    size_t subgroup_range_n = (wg_tile_n % sg_tile_n == 0)
            ? wg_tile_n / sg_tile_n
            : (wg_tile_n / sg_tile_n) + 1;
    cl::sycl::range<3> group_range {
            Test::l3_kslicing, group_range_m, group_range_n};
    cl::sycl::range<3> local_range {
            Test::slm_kslicing, subgroup_range_m, subgroup_range_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    try {
        std::vector<kernel_id> kernelId = {get_kernel_id<Test>()};
        auto inputBundle
                = get_kernel_bundle<bundle_state::input>(context, kernelId);
        setenv("SYCL_PROGRAM_COMPILE_OPTIONS", compile_str.c_str(), 1);
        kernel_bundle<bundle_state::executable> exeBundle = build(inputBundle);
        unsetenv("SYCL_PROGRAM_COMPILE_OPTIONS");
        auto e_esimd = queue.submit([&](handler &cgh) {
            cgh.use_kernel_bundle(exeBundle);
            cgh.parallel_for<Test>(
                    nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                        gpu::xetla::xetla_exec_item<3> ei(item);
                        gpu::xetla::xetla_local_init<SLMSIZE>();
                        gpu::xetla::xetla_nbarrier_init<BARNUM>();
                        KERNEL<Test, data_type_a, data_type_b, data_type_c,
                                data_type_acc>::run(ei, A, B, C, matrix_m,
                                matrix_n, matrix_k);
                    });
        });
        e_esimd.wait();

    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    // validation
    validate_func<Test, data_type_a, data_type_b, data_type_c, data_type_acc>
            vfunc;
    ASSERT_EQ(0, vfunc(A, B, C, queue, context));

    free(A, context);
    free(B, context);
    free(C, context);
}

/// @brief The template function to execute kernel in esimd way for unit test framework
///
/// @tparam data_type data_type The data type of buffer used in kernel and buffer allocation
/// @tparam KERNEL the kernel function struct
/// @param nd_range the range of workitems
/// @param validate_result validation function, taking 3 parameters buffer A, B as input C as output
///
template <typename data_type, class KERNEL, int SLMSIZE = 128 * 1024,
        int BARNUM = 32, int Size = 4096>
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
            cgh.parallel_for<>(nd_range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                gpu::xetla::xetla_exec_item ei(ndi);
                gpu::xetla::xetla_local_init<SLMSIZE>();
                gpu::xetla::xetla_nbarrier_init<BARNUM>();
                KERNEL::run(&ei, A, B, C);
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
