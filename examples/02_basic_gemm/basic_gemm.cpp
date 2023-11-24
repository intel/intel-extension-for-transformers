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
#include "tests/utils/utils.hpp"
#include "xetla.hpp"

void basic_gemm_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    // GEMM input size
    uint32_t matrix_m = 4096;
    uint32_t matrix_n = 4096;
    uint32_t matrix_k = 4096;

    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
    using data_type_acc = float;

    // Turn on the profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    // Define SYCL queue, context and device
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

    // Define the shape of workgroup and subgroup
    // It's tunable parameters based on different input shape and hardware for
    // better performance
    constexpr uint32_t wg_tile_m = 256;
    constexpr uint32_t wg_tile_n = 256;
    constexpr uint32_t sg_tile_m = 32;
    constexpr uint32_t sg_tile_n = 64;

    // Workload mapping, linear mapping will be used in the code
    // Suppose it is divisible.
    uint32_t group_range_m = matrix_m / wg_tile_m;
    uint32_t group_range_n = matrix_n / wg_tile_n;

    // Each subgroup will be executed in one hardware thread
    // Calculate how many threads in a workgroup
    uint32_t thread_range_m = wg_tile_m / sg_tile_m;
    uint32_t thread_range_n = wg_tile_n / sg_tile_n;

    // leading dimension
    uint32_t lda = matrix_k;
    uint32_t ldb = matrix_n;
    uint32_t ldc = matrix_n;

    // Ndrange and workgroup shape
    cl::sycl::range<3> group_range {1, group_range_m, group_range_n};
    cl::sycl::range<3> local_range {1, thread_range_m, thread_range_n};

    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    constexpr uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof("basic_gemm", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                using namespace gpu::xetla;
                using namespace gpu::xetla::group;
                using namespace gpu::xetla::kernel;
                using namespace gpu::xetla::subgroup;

                // wrap the nd_range to XeTLA range

                // Performance tuning setting based on different shapes
                static constexpr uint32_t periodic_sync_interval = 8;
                static constexpr uint32_t prefetch_distance = 3;
                // should larger than 8
                static constexpr uint32_t k_stride = 32;

                // Step 1: define mirco-kernel's configuration
                using wg_shape = shape<wg_tile_n, wg_tile_m>;
                using sg_shape = shape<sg_tile_n, sg_tile_m>;

                // Mirco-kernel configuration
                using gemm_tune_option
                        = dict_t<elem_t_t<tune_key::SG_TILE_SHAPE, sg_shape>,
                                elem_v_t<tune_key::PREFETCH_DISTANCE,
                                        prefetch_distance>,
                                elem_v_t<tune_key::PERIODIC_SYNC_INTERVAL,
                                        periodic_sync_interval>>;
                using gemm_t = xetla::group::default_gemm_selector_t<
                        data_type_a, // input datatype for A
                        mem_layout::row_major, // memory layout for A
                        8, // leading dimension for A, in unit of element
                        mem_space::
                                global, // memory reading from global mem for A
                        data_type_b, // input datatype for B
                        mem_layout::row_major, // memory layout for B
                        8, // leading dimension for B, in unit of element
                        mem_space::
                                global, // memory reading from global mem for B
                        data_type_acc, // accumulator data type for intermediate resutls
                        wg_shape, // computation tile shape
                        k_stride, // elements in each iteration
                        gpu_arch::Xe, // GPU arch
                        gemm_tune_option>;
                gemm_t gemm;

                // Step 2: epilogue function to overwrite the result
                using epilogue_tune_option
                        = dict_t<elem_t_t<tune_key::SG_TILE_SHAPE, sg_shape>>;
                using epilogue_t = xetla::group::default_epilogue_selector_t<
                        data_type_c, // onput datatype for C
                        mem_layout::row_major, // memory layout for C
                        8, // leading dimension for C, in unit of element
                        mem_space::global, // memory writing to global mem for C
                        wg_shape, // computation tile shape
                        k_stride, // elements in each iteration
                        gpu_arch::Xe, // GPU arch
                        epilogue_tune_option>;

                // Step 3: define the shared local memory usages
                // developers have the responsibility to set
                // shared loacal memory through XeTLA API
                static constexpr uint32_t barrier_count = gemm_t::barrier_count;
                static constexpr uint32_t slm_size = gemm_t::slm_size;
                xetla_nbarrier_init<barrier_count>();
                xetla_local_init<slm_size>();

                // Step 4: ecah workgroup gets it individual index to start computation
                int start_n = item.get_group(2) * wg_tile_n;
                int start_m = item.get_group(1) * wg_tile_m;
                // no slicing in K direction so start from zero for all WG
                int start_k = 0;

                // Each workgroup will compute all data in K based on no k_sliciing
                // The developer can set how much data a subgroup compute by k_stride
                uint32_t wg_tile_k = matrix_k;
                uint32_t inner_loop_count
                        = (wg_tile_k + k_stride - 1) / k_stride;

                // Step 5: define the workgroup start point for each workgroup
                using mem_desc_input_a = gemm_t::mem_desc_a_t;
                using mem_desc_input_b = gemm_t::mem_desc_b_t;
                using mem_desc_output_c = epilogue_t::mem_desc_c_t;
                mem_desc_input_a md_a(
                        {A}, {matrix_k, matrix_m, lda}, {start_k, start_m});
                mem_desc_input_b md_b(
                        {B}, {matrix_n, matrix_k, ldb}, {start_n, start_k});
                mem_desc_output_c md_c(
                        {C}, {matrix_n, matrix_m, ldc}, {start_n, start_m});

                // Step 6: real calculation with accumulator varibales which suppose
                // will be in register.
                gemm_t::matAcc_t matAcc;
                matAcc.init(0);

                gemm_t::arguments_t gemm_args(md_a, md_b, inner_loop_count);

                // the results is in the matAcc rather than real output C
                gemm_t::work_group_t g(item.get_local_linear_id());
                gemm(g, matAcc, gemm_args);

                // Step 7: write the results from matACC to real output C
                epilogue_t epilogue;
                epilogue(g, matAcc, md_c);
            });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
        }
    }

    ASSERT_EQ(0,
            gemm_result_validate(A, B, C, 1, matrix_m, matrix_k, matrix_n,
                    queue, mem_layout::row_major, mem_layout::row_major));

    // performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
}

int main() {
    // This case shows how to use batch-reduce (br) GEMM microkernel to
    // solve a standard GEMM
    basic_gemm_run(10);
    return (0);
}
