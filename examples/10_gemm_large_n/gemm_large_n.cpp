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

void gemm_large_n_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    //GEMM input size
    size_t matrix_m = 4096;
    size_t matrix_n = 51200;
    size_t matrix_k = 4096;

    size_t size_a = matrix_m * matrix_k;
    size_t size_b = matrix_k * matrix_n;
    size_t size_c = matrix_m * matrix_n;

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
    using data_type_acc = float;

    //Turn on the profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    //Define SYCL queue, context and device
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

    //Define the shape of workgroup and subgroup
    //It's tunable parameters based on different input shape and hardware for better performance
    constexpr uint32_t wg_tile_m = 256;
    constexpr uint32_t wg_tile_n = 256;
    constexpr uint32_t sg_tile_m = 32;
    constexpr uint32_t sg_tile_n = 64;

    //There are implicit requirement for sg_tile_k range
    constexpr uint32_t sg_tile_k = 32;

    // This parameter indicates the workgroup number in
    // single Xe-core on vectorizonal direction
    // available settings: 1, 2, 4 ,8, 16, 32, 64
    // default 8
    static constexpr uint32_t wg_num_n = 8;

    // Org the compute shape for sub-matrix
    using tile_shape
            = xetla::group::tile_shape_t<wg_tile_n, // workgroup size in dim0
                    wg_tile_m, //	workgroup size in dim1
                    sg_tile_n, //	subgroup size in dim0
                    sg_tile_m>; //	subgroup size in dim1

    // Mirco-kernel configuration
    using gemm_t = xetla::group::gemm_selector_t<
            data_type_a, // input datatype for A
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for A
            mem_layout::row_major, // memory layout for B
            mem_space::global, // memory reading from global mem for A
            mem_space::global, // memory reading from global mem for B
            8, // buffer alignment for A, in unit of element
            8, // buffer alignment for B, in unit of element
            data_type_acc, // accumulator data type for intermediate resutls
            tile_shape, // computation tile shape
            sg_tile_k, // elements in each iteration
            mma_engine::xmx, // compute engine
            gpu_arch::Xe, 3, 8> // GPU arch
            ::gemm;

    using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_default<gpu_arch::Xe>, tile_shape,
            mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>>;

    using group_swizzle
            = xetla::kernel::group_swizzle_snake<wg_num_n, gpu_arch::Xe>;

    using dispatch_policy
            = xetla::kernel::dispatch_policy_default<group_swizzle>;

    using gemm_op_t = xetla::kernel::gemm_universal_t<dispatch_policy, gemm_t,
            epilogue_t>;

    // set up gemm arguments
    typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A,
            matrix_k, B, matrix_n, C, matrix_n);

    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
    if (!gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        free(A, queue);
        free(B, queue);
        free(C, queue);
        FAIL();
    }

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof("gemm_large_n", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                // allocate slm and nbarrier resource
                slm_barrier_init<gemm_op_t>();
                gemm_op_t gemm_op;
                gemm_op(item, gemm_arg);
            });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
        }
        // sleep 1 second after finishing each gpu event
        sleep(1);
    }

    prof.print_profiling_result(profiling_selector::GPU);

    ASSERT_EQ(0,
            gemm_result_validate(A, B, C, 1, matrix_m, matrix_k, matrix_n,
                    queue, mem_layout::row_major, mem_layout::row_major));

    free(A, context);
    free(B, context);
    free(C, context);
}

int main() {
    // A example code of XeTLA dispatch policy block under which
    // each workgroup in single Xe-core will be dispatched in a
    // rectangular shape. User can control the width and height of
    // the rectangular and by doing so, gemm algorithm can perform
    // more cache friendly.
    // In this case, we default use 8 as the width of workgroup
    // rectangular.
    gemm_large_n_run(10);
    return (0);
}
