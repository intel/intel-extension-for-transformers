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

void splitk_brgemm_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    // GEMM input size
    uint32_t matrix_m = 256;
    uint32_t matrix_n = 256;
    uint32_t matrix_k = 8192;

    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
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

    // [Split-K] parameter S
    // Adjust this number to 1, 4, or 8 to observe the performance variation
    constexpr uint32_t split_k_S = 8;

    // Define the shape of workgroup and subgroup
    // It's tunable parameters based on different input shape and hardware for
    // better performance
    constexpr uint32_t wg_tile_m = 64;
    constexpr uint32_t wg_tile_n = 64;
    constexpr uint32_t sg_tile_m = 8;
    constexpr uint32_t sg_tile_n = 16;

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
    // [Split-K] expand index space from (i, j) to (o, i, j)
    cl::sycl::range<3> GroupRange {split_k_S, group_range_m, group_range_n};
    cl::sycl::range<3> LocalRange {1, thread_range_m, thread_range_n};

    cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof("splitk_brgemm", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        // [Split-K] reset output matrix C to zero
        queue.memset(C, 0, size_c * sizeof(C[0])).wait();

        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                using namespace gpu::xetla;
                using namespace gpu::xetla::group;
                using namespace gpu::xetla::kernel;
                using namespace gpu::xetla::subgroup;

                // wrap the NDrange to XeTLA range
                xetla_exec_item<3> ei(item);

                // Step 1: basic computation information
                // define A, B and accumulator datatype
                // Using float as accumuator for better accuracy
                using compute_attr = compute_attr_t<data_type_a, data_type_b,
                        data_type_acc>;

                // Performance tuning setting based on different shapes
                static constexpr uint32_t periodic_sync_interval = 8;
                static constexpr uint32_t prefetch_distance = 3;
                // should larger than 8
                static constexpr uint32_t k_iter_num = 32;
                using perf_tuning_knob = perf_tuning_knob_t<k_iter_num,
                        prefetch_distance, periodic_sync_interval>;

                // specific the computation, performance tuning and computation core
                using compute_policy = compute_policy_default_xmx<compute_attr,
                        perf_tuning_knob, gpu_arch::Xe>;

                // Step 2: define the memory layout & location of input/output
                // this setting could be used to optimize the data re-use in shared
                // local memory
                using mem_desc_input_a = mem_desc_t<data_type_a,
                        mem_layout::row_major, mem_space::global>;
                using mem_desc_input_b = mem_desc_t<data_type_b,
                        mem_layout::row_major, mem_space::global>;
                using mem_desc_output_c = mem_desc_t<data_type_c,
                        mem_layout::row_major, mem_space::global>;

                // Step 3: define mirco-kernel's configuration
                using tile_shape = tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n,
                        sg_tile_m>;
                using brgemm_t = brgemm_t<compute_policy, tile_shape,
                        mem_desc_input_a, mem_desc_input_b>;
                brgemm_t brgemm;

                // Step 4: epilogue function to define how to write result back or post
                // fusion
                // [Split-K] When Split-K is used, update_method should be set to
                // result_reduce_sum in order to aggregate partial sum from each sub-task
                // to the final output matrix C
                using epilogue_t = epilogue_t<
                        epilogue_policy_tile_op<chained_tile_op_t<>,
                                result_reduce_sum, gpu_arch::Xe>,
                        tile_shape, mem_desc_output_c>;

                // Step 5: define the shared local memory usages
                // developers have the responsibility to set
                // shared loacal memory through XeTLA API
                static constexpr uint32_t barrier_count
                        = brgemm_t::barrier_count;
                static constexpr uint32_t slm_size = brgemm_t::slm_size;
                xetla_nbarrier_init<barrier_count>();
                xetla_local_init<slm_size>();

                // Step 6: ecah workgroup gets it individual index to start computation
                int start_n = ei.get_group(2) * wg_tile_n;
                int start_m = ei.get_group(1) * wg_tile_m;
                // [Split-K] parameter R
                int split_k_R = matrix_k / split_k_S;
                // [Split-K] start index E
                int split_k_E = ei.get_group(0) * split_k_R;
                // [Split-K] end index D = E + R
                uint32_t split_k_D = split_k_E + split_k_R;
                int start_k = split_k_E;

                // Each workgroup will compute all data in K based on no k_sliciing
                // The developer can set how much data a subgroup compute by k_iter_num
                // [Split-K] block matrix has size split_k_R in dimension k
                uint32_t wg_tile_k = split_k_R;
                uint32_t inner_loop_count = wg_tile_k / k_iter_num;

                // Step 7: define the workgroup start point for each workgroup
                // [Split-K] block matricies partitioned along dimension k
                mem_desc_input_a md_a(
                        {A}, {split_k_D, matrix_m, lda}, {start_k, start_m});
                mem_desc_input_b md_b(
                        {B}, {matrix_n, split_k_D, ldb}, {start_n, start_k});
                mem_desc_output_c md_c(
                        {C}, {matrix_n, matrix_m, ldc}, {start_n, start_m});

                // Step 8: real calculation with accumulator varibales which suppose
                // will be in register.
                brgemm_t::matAcc_t matAcc;
                matAcc.init(0);

                brgemm_t::arguments_t brgemm_args(md_a, md_b, inner_loop_count);

                // the results is in the matAcc rather than real output C
                // [Split-K] matAcc = A<i, :>[:, E:D] x B<:, j>[E:D, :]
                brgemm_t::work_group_t g(ei.get_local_linear_id());
                brgemm(g, matAcc, brgemm_args);

                // Step 9: write the results from matACC to real output C
                // [Split-K] With the epilogue_t parameter update_method=result_reduce_sum,
                // the partial sum matAcc is aggregated to the final output matrix C:
                //   C<i, j> += matAcc
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
                    queue, context, mem_layout::row_major,
                    mem_layout::row_major));

    // performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
}

int main() {
    // In this example, BRGEMM with slicing in K dimension is demonstrated.

    // Split-K version, when compared to the vanilla BRGEMM, splits the overall
    // task into multiple sub-tasks: each sub-task computes a part of the
    // final result and will be aggregated using the reduce sum method.

    // This can be best illustrated using an example.

    // Suppose the final task is to calculate the following:
    //   C = A x B
    // where:
    //   shape(A) = [m, k]
    //   shape(B) = [k, n]
    //   shape(C) = [m, n]

    // In the basic task decomposition, it could be partitioned into
    // multiple block matrices as sub-tasks:
    //   C = [C<i, j>], i=1..I, j=1..J
    // where:
    //   shape(C<i, j>) = [H, W]
    //   i, j: work-item indices
    //   H: #block matrix rows
    //   W: #block matrix columns
    //   I = m / H
    //   J = n / W

    // Each block matrix C<i, j> is a work-item with index (i, j) which will
    // be executed by a GPU thread, essentially a block matrix multiplication:
    //   C<i, j> = A<i, :> x B<:, j>
    // where:
    //   <r, c>: block matrix selector

    // In Split-K, the above task is again partitioned into multiple sub-tasks
    // along dimension k in S chunks.

    // For a work-item with index (o, i, j), the sub-task becomes:
    //   C<i, j> += A<i, o> x B<o, j>
    // The block matrix C<i, j> needs to aggregate its partial inner product.

    // Using the numpy notation to expand indices, the above is equivalent to:
    //   C<i, j> += A<i, :>[:, E:D] x B<:, j>[E:D, :]
    // where:
    //   S          <- #chunks
    //   R = k / S  <- chunk size
    //   E = o * R  <- start index
    //   D = E + R  <- end index

    // In the kernel code, the above variable will be prefixed with "split_k":
    //   S -> split_k_S
    //   E -> split_k_E
    //   D -> split_k_D
    //   R -> split_k_R

    // This alternative method of task decomposition is comparable with
    // partitioning along dimension m and dimension n.

    // It features that the matrices in BRGEMM do not varies the shape
    // of the output matrix when changing the chunk number S.

    // Note:
    //   - currently (2023-04-17) only float is supported as output data type
    //     when update_method=result_reduce_sum
    //   - comments related to split k will have the "[Split-K]" prefix
    splitk_brgemm_run(10);
    return (0);
}
