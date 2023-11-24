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

#include "batch_gemm.hpp"

void batch_gemm_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    // batch size
    uint32_t batch_size = 256;

    //GEMM input size
    uint32_t matrix_m = 512;
    uint32_t matrix_n = 768;
    uint32_t matrix_k = 64;

    // define slice of each matrices
    uint32_t size_a_slice = matrix_m * matrix_k;
    uint32_t size_b_slice = matrix_k * matrix_n;
    uint32_t size_c_slice = matrix_m * matrix_n;

    // calculate total size of matrices
    uint32_t size_a = batch_size * size_a_slice;
    uint32_t size_b = batch_size * size_b_slice;
    uint32_t size_c = batch_size * size_c_slice;

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

    //There are implicit requirement for wg_tile_k range
    constexpr uint32_t wg_tile_k = 32;
    constexpr uint32_t sync_freq = 8;
    constexpr uint32_t stages = 3;

    // Org the compute shape for sub-matrix
    using wg_shape = shape<wg_tile_n, wg_tile_m>;
    using sg_shape = shape<sg_tile_n, sg_tile_m>;

    // Mirco-kernel configuration
    using tune_option
            = dict_t<elem_v_t<tune_key::PARAM_OPTIMZER_TYPE,
                             tune_key_value::PARAM_OPTIMZER_DECISION_TREE>,
                    elem_t_t<tune_key::SG_TILE_SHAPE, sg_shape>,
                    elem_v_t<tune_key::PREFETCH_DISTANCE, stages>,
                    elem_v_t<tune_key::PERIODIC_SYNC_INTERVAL, sync_freq>>;
    using gemm_t = xetla::group::default_gemm_selector_t<
            data_type_a, // input datatype for A
            mem_layout::row_major, // memory layout for A
            8, // leading dimension for A, in unit of element
            mem_space::global, // memory reading from global mem for A
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for B
            8, // leading dimension for B, in unit of element
            mem_space::global, // memory reading from global mem for B
            data_type_acc, // accumulator data type for intermediate resutls
            wg_shape, // computation tile shape
            wg_tile_k, // elements in each iteration
            gpu_arch::Xe, // GPU arch
            tune_option>;

    using epilogue_t = xetla::group::default_epilogue_selector_t<
            data_type_c, // onput datatype for C
            mem_layout::row_major, // memory layout for C
            8, // leading dimension for C, in unit of element
            mem_space::global, // memory writing to global mem for C
            wg_shape, // computation tile shape
            wg_tile_k, // elements in each iteration
            gpu_arch::Xe, // GPU arch
            tune_option>;

    using batch_gemm_op_t
            = xetla::kernel::batch_gemm_t<gemm_t, epilogue_t, gpu_arch::Xe>;

    // set up gemm_universal arguments
    typename batch_gemm_op_t::arguments_t gemm_arg(batch_size, matrix_m,
            matrix_k, matrix_n, A, matrix_k, B, matrix_n, C, matrix_n);

    cl::sycl::nd_range<3> nd_range = batch_gemm_op_t::get_nd_range(gemm_arg);
    if (!batch_gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        free(A, context);
        free(B, context);
        free(C, context);
        FAIL();
    }

    uint32_t warmup = 10;
    size_t ops = batch_size * 2 * static_cast<size_t>(matrix_m) * matrix_n
            * matrix_k;
    profiling_helper prof("batch_gemm", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                // allocate slm and nbarrier resource
                slm_barrier_init<batch_gemm_op_t>();
                batch_gemm_op_t batch_gemm_op;
                batch_gemm_op(item, gemm_arg);
            });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
        }
    }

    ASSERT_EQ(0,
            gemm_result_validate(A, B, C, batch_size, matrix_m, matrix_k,
                    matrix_n, queue, mem_layout::row_major,
                    mem_layout::row_major));

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
}

int main() {
    // The purpose of this example is to demonstrate how to calculate matrix
    // multiplication of high order.
    //   C[i] = A[i] x B[i] for i in range(0, batch_size)
    // where:
    //   shape(A) = [batch_size x m, k]
    //   shape(B) = [batch_size x k, n]
    //   shape(C) = [batch_size x m, n]

    batch_gemm_run(10);
    return (0);
}
