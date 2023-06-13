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

enum class batch_impl_t { for_loop = 0, nd_range = 1 };

template <batch_impl_t batch_impl = batch_impl_t::for_loop>
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

    //There are implicit requirement for sg_tile_k range
    constexpr uint32_t sg_tile_k = 32;

    // Org the compute shape for sub-matrix
    using tile_shape
            = xetla::group::tile_shape_t<wg_tile_n, // workgroup size in dim0
                    wg_tile_m, //	workgroup size in dim1
                    sg_tile_n, //	subgroup size in dim0
                    sg_tile_m>; //	subgroup size in dim1

    // Mirco-kernel configuration
    using brgemm_config = xetla::group::brgemm_selector_t<
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
            gpu_arch::Xe> // GPU arch
            ::brgemm;

    using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_default<result_overwrite,
                    gpu_arch::Xe>,
            tile_shape,
            mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>>;

    using gemm_op_t = xetla::kernel::gemm_t<
            xetla::kernel::dispatch_policy_default<gpu_arch::Xe>, brgemm_config,
            epilogue_t>;

    //Ndrange and workgroup shape
    cl::sycl::range<3> group_range
            = gemm_op_t::get_group_range(matrix_m, matrix_n);
    cl::sycl::range<3> local_range = gemm_op_t::get_local_range();

    // [Batch] Extend index space, the z dimension corresponds to batch
    // dimension
    try {
        if constexpr (batch_impl == batch_impl_t::nd_range) {
            group_range[0] = batch_size;
        }
    } catch (sycl::exception const &e) {
        sycl::free(A, context);
        sycl::free(B, context);
        sycl::free(C, context);
        std::cout << "invalid parameter: " << e.what() << '\n';
        return;
    }

    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    uint32_t warmup = 10;
    long ops = batch_size * 2 * static_cast<long>(matrix_m) * matrix_n
            * matrix_k;
    profiling_helper prof("batch_gemm", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                xetla_exec_item<3> ei(item);
                slm_barrier_init<gemm_op_t>();
                gemm_op_t gemm_op;
                if constexpr (batch_impl == batch_impl_t::for_loop) {
                    // [Batch] One work-item computes all slices in the
                    // batch dimension
                    for (uint32_t batch = 0; batch < batch_size; batch++) {
                        typename gemm_op_t::arguments_t arg(matrix_m, matrix_k,
                                matrix_n, A + size_a_slice * batch, matrix_k,
                                B + size_b_slice * batch, matrix_n,
                                C + size_c_slice * batch, matrix_n);
                        gemm_op(ei, arg);
                    }
                } else {
                    // [Batch] Get batch index from group_range
                    // One work-item is responsible for one slice only
                    uint32_t batch = ei.get_group(0);
                    typename gemm_op_t::arguments_t arg(matrix_m, matrix_k,
                            matrix_n, A + size_a_slice * batch, matrix_k,
                            B + size_b_slice * batch, matrix_n,
                            C + size_c_slice * batch, matrix_n);
                    gemm_op(ei, arg);
                }
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
                    matrix_n, queue, context, mem_layout::row_major,
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
    //   C = A x B
    // where:
    //   shape(A) = [batch_size, m, k]
    //   shape(B) = [batch_size, k, n]
    //   shape(C) = [batch_size, m, n]

    // This example provides two implementation,
    // - batch_impl_t::for_loop shows the basic usage
    //   of varying base pointer inside the kernel
    // - batch_impl_t::nd_range shows the idiomatic
    //   mapping of task decomposition to index space

    // Note:
    //   - comments related to batch will have the "[Batch]" prefix
    // batch_gemm_run<batch_impl_t::for_loop>(10);
    batch_gemm_run<batch_impl_t::nd_range>(10);
    return (0);
}
