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
#include <algorithm>
#include "tests/utils/utils.hpp"
#include "xetla.hpp"

#include "gemm_polynomial.hpp"

using namespace cl::sycl;
using namespace gpu::xetla;

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float>
int gemm_polynomial_result_validate(data_type_a *A_device,
        data_type_b *B_device, data_type_c *C_device, int m, int k, int n,
        sycl::queue &queue, mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    auto A = alloc_host_and_copy<data_type_a>(A_device, m * k, queue);
    auto B = alloc_host_and_copy<data_type_b>(B_device, k * n, queue);
    auto C = alloc_host_and_copy<data_type_c>(C_device, m * n, queue);

    bool is_col_major_a = mem_layout_a_ == mem_layout::col_major;
    bool is_col_major_b = mem_layout_b_ == mem_layout::col_major;
    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
    std::vector<data_type_acc> gold_C(m * n, 0);
    get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_b_, A, B, gold_C.data());
    std::vector<data_type_acc> coeff({1.0f, 2.0f, 4.0f});
    std::transform(gold_C.cbegin(), gold_C.cend(), gold_C.begin(),
            [&coeff](data_type_acc x) {
                data_type_acc res = 0.0f;
                for (int i = 0; i < coeff.size(); ++i) {
                    res = x * res;
                    res += static_cast<data_type_acc>(coeff[i]);
                }
                return res;
            });
    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m, n, n);

    bool result = buff_cmp::xetla_buff_cmp(
            data, other, "gemm_polynomial validation");

    free(A);
    free(B);
    free(C);

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

void gemm_polynomial_run(int iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    // GEMM input size
    uint32_t matrix_m = 256;
    uint32_t matrix_n = 1024;
    uint32_t matrix_k = 768;

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

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k
            + 6 * static_cast<long>(matrix_m) * matrix_n;
    profiling_helper prof("gemm_polynomial", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                using namespace gpu::xetla;
                using namespace gpu::xetla::group;
                using namespace gpu::xetla::kernel;
                using namespace gpu::xetla::subgroup;

                // wrap the nd_range to XeTLA range

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
                using gemm_t = gemm_t<compute_policy, tile_shape,
                        mem_desc_input_a, mem_desc_input_b>;
                gemm_t gemm;

                // Step 4: epilogue function to define how to write result back or post
                // fusion
                // [Polynomial] Define tile_op used in epilogue_t: polynomial_op_t
                // evaluate a 2nd-order polynomial, datatype matches accumulator datatype
                using polynomial_op_t = polynomial_op_t<data_type_acc, 3>;
                using tile_op_t = chained_tile_op_t<polynomial_op_t>;
                // [Polynomial] epilogue_t is an elementwise operation that will be applied to the
                // accumulator C_acc in the final stage, in which
                //   C_acc = A x B
                // is already calculated.
                // Mathematically epilogue_t is a map that applies to each element:
                //   epilogue_t: [m, n] -> [m, n], C_acc |-> tile_op_t(C_acc)
                using epilogue_t = epilogue_t<
                        epilogue_policy_tile_op<tile_op_t, gpu_arch::Xe>,
                        tile_shape, mem_desc_output_c>;

                // [Polynomial] define arguments for each epilogue_tile_op in chained_tile_op_t<>
                using epilogue_tile_op_args_t
                        = epilogue_t::tile_op_t::arguments_t;
                // [Polynomial] specify polynomial_op_t coefficients
                // a_2 = 1.0f, a_1 = 2.0f, a_0 = 4.0f
                using coeff_t = polynomial_op_t::coeff_t;
                coeff_t polynomial_coeff({1.0f, 2.0f, 4.0f});
                epilogue_tile_op_args_t tile_op_args(
                        // [Polynomial] polynomial_op_t coefficients
                        polynomial_coeff);

                // [Polynomial] pass arguments of chained_tile_op_t<> to epilogue_args
                using epilogue_args_t = epilogue_t::arguments_t;
                epilogue_args_t epilogue_args(tile_op_args);

                // Step 5: define the shared local memory usages
                // developers have the responsibility to set
                // shared loacal memory through XeTLA API
                static constexpr uint32_t barrier_count = gemm_t::barrier_count;
                static constexpr uint32_t slm_size = gemm_t::slm_size;
                xetla_nbarrier_init<barrier_count>();
                xetla_local_init<slm_size>();

                // Step 6: ecah workgroup gets it individual index to start computation
                int start_n = item.get_group(2) * wg_tile_n;
                int start_m = item.get_group(1) * wg_tile_m;
                // no slicing in K direction so start from zero for all WG
                int start_k = 0;

                // Each workgroup will compute all data in K based on no k_sliciing
                // The developer can set how much data a subgroup compute by k_iter_num
                uint32_t wg_tile_k = matrix_k;
                uint32_t inner_loop_count = wg_tile_k / k_iter_num;

                // Step 7: define the workgroup start point for each workgroup
                mem_desc_input_a md_a(
                        {A}, {matrix_k, matrix_m, lda}, {start_k, start_m});
                mem_desc_input_b md_b(
                        {B}, {matrix_n, matrix_k, ldb}, {start_n, start_k});
                mem_desc_output_c md_c(
                        {C}, {matrix_n, matrix_m, ldc}, {start_n, start_m});

                // Step 8: real calculation with accumulator varibales which suppose
                // will be in register.
                gemm_t::matAcc_t matAcc;
                matAcc.init(0);

                gemm_t::arguments_t gemm_args(md_a, md_b, inner_loop_count);

                // the results is in the matAcc rather than real output C
                gemm_t::work_group_t g(item.get_local_linear_id());
                gemm(g, matAcc, gemm_args);

                // Step 9: write the results from matACC to real output C
                epilogue_t epilogue;
                epilogue(g, matAcc, md_c, epilogue_args);
            });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
        }
    }

    ASSERT_EQ(0,
            gemm_polynomial_result_validate(A, B, C, matrix_m, matrix_k,
                    matrix_n, queue, mem_layout::row_major,
                    mem_layout::row_major));

    // performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
}

int main() {
    // This case shows how to extend existing code base to support custom
    // tile_op into gemm.

    // A new custom elementwise operation will be created by defining a
    // new class of type tile_op_t.
    // By passing this tile_op_t type to epilogue_t, the operation will
    // be applied to the result of gemm, here provides some example:
    // - gemm
    //   C  = A x B
    // - tile_op_t=relu_op_t
    //   C = ReLU(A x B)
    // - tile_op_t=polynomial_op_t
    //   C = PolynomialOp(A x B)  <- the example to be implemented
    // where:
    //   PolynomialOp: x |-> a_n * x^n + a_{n-1} * x^{n-1} + ... + a_0
    // The coefficients (a_n, a_{n-1}, ..., a_0) will be passed as arguments
    // to the custom op polynomial_op_t.

    // Note:
    //   - comments related to this example will be prefixed with "[Polynomial]"
    gemm_polynomial_run(10);
    return (0);
}
