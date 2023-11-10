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
#include "softmax_fwd_kernel.hpp"
#include "xetla.hpp"

using namespace gpu::xetla;

//Test: accept different test data
//iter: indicate the iterations of the kernel
template <class Test>
void softmax_fwd_run() {
    //Accept incoming parameters
    size_t mat_n = Test::mat_n;
    size_t mat_m = Test::mat_m;
    constexpr size_t sg_n = Test::sg_n;
    constexpr size_t sg_m = Test::sg_m;
    constexpr size_t wg_n = Test::wg_n;
    constexpr size_t wg_m = Test::wg_m;

    using data_type_in = typename Test::data_type_in;
    using data_type_acc = typename Test::data_type_acc;
    using data_type_out = typename Test::data_type_out;

    int size_in = mat_n * mat_m;
    int size_out = mat_n * mat_m;

    //Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<sycl::info::queue::context>();
    auto device = queue.get_info<sycl::info::queue::device>();

    std::cout << "Running on " << device.get_info<sycl::info::device::name>()
              << "\n";

    //Define and initialize the data required for the calculation
    auto buffer_in = alloc_device_and_init<data_type_in>(
            size_in,
            [](data_type_in *data, size_t idx) {
                data[idx] = static_cast<data_type_in>(random_float());
            },
            queue, device, context);
    auto buffer_out = alloc_device_and_init<data_type_out>(
            size_out,
            [](data_type_out *data, size_t idx) {
                data[idx] = static_cast<data_type_out>(0);
            },
            queue, device, context);

    data_type_acc sqrt_dk_inv = 0.125f;

    size_t group_range_m = (mat_m + wg_m - 1) / wg_m;
    size_t group_range_n = (mat_n + wg_n - 1) / wg_n;
    size_t subgroup_range_m = (wg_m + sg_m - 1) / sg_m;
    size_t subgroup_range_n = (wg_n + sg_n - 1) / sg_n;

    std::cout << " group_num_x: " << group_range_n
              << ",  group_num_y: " << group_range_m << "\n";
    std::cout << " group_size_x: " << subgroup_range_n
              << ",  group_size_y: " << subgroup_range_m << std::endl;
    cl::sycl::range<3> group_range {1, group_range_m, group_range_n};
    cl::sycl::range<3> local_range {1, subgroup_range_m, subgroup_range_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    // esimd kernel prepratation and execution
    {
        std::vector<kernel_id> kernelId = {get_kernel_id<Test>()};
        auto inputBundle
                = get_kernel_bundle<bundle_state::input>(context, kernelId);
        setenv("SYCL_PROGRAM_COMPILE_OPTIONS",
                " -vc-codegen -doubleGRF  -Xfinalizer ' "
                "-printregusage -enableBCR  "
                "-DPASTokenReduction '",
                1);
        kernel_bundle<bundle_state::executable> exeBundle = build(inputBundle);
        unsetenv("SYCL_PROGRAM_COMPILE_OPTIONS");
        try {

            auto e_softmax_fwd = queue.submit([&](handler &cgh) {
                cgh.use_kernel_bundle(exeBundle);
                cgh.parallel_for<Test>(
                        nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                            using softmax_fwd_func
                                    = softmax_fwd_test_func<data_type_in,
                                            data_type_out, data_type_acc, wg_n,
                                            wg_m, sg_n, sg_m>;
                            constexpr uint32_t barrier_count
                                    = softmax_fwd_func::barrier_count;
                            constexpr uint32_t slm_size
                                    = softmax_fwd_func::slm_size;

                            xetla_nbarrier_init<barrier_count>();
                            xetla_local_init<slm_size>();

                            softmax_fwd_func::run(item, buffer_in, buffer_out,
                                    mat_m, mat_n, mat_n, sqrt_dk_inv);
                        });
            });
            e_softmax_fwd.wait();

        } catch (cl::sycl::exception const &e) {
            std::cout << "SYCL exception caught: " << e.what() << '\n';
            FAIL();
        }
    }

    // validation
    auto buffer_in_host
            = alloc_host_and_copy<data_type_in>(buffer_in, size_in, queue);
    auto buffer_out_host
            = alloc_host_and_copy<data_type_out>(buffer_out, size_out, queue);
    ASSERT_EQ(0,
            (fwd_reduction_result_validate<data_type_in, data_type_out,
                    data_type_acc>(buffer_in_host, buffer_out_host, mat_m,
                    mat_n, sqrt_dk_inv)));

    free(buffer_in, context);
    free(buffer_out, context);

    free(buffer_in_host);
    free(buffer_out_host);
}

TEST(softmax_fwd_test, esimd) {
    softmax_fwd_run<mat0_96x2048x2048_bf16>();
}
