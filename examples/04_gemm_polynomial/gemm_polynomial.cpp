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

    // epilogue function to define how to write result back or post
    // fusion
    // [Polynomial] Define tile_op used in epilogue_t: polynomial_op_t
    // evaluate a 2nd-order polynomial, datatype matches accumulator datatype
    using polynomial_op_t = xetla::subgroup::polynomial_op_t<data_type_acc, 3>;
    using tile_op_t = xetla::subgroup::chained_tile_op_t<polynomial_op_t>;
    // [Polynomial] epilogue_t is an elementwise operation that will be applied to the
    // accumulator C_acc in the final stage, in which
    //   C_acc = A x B
    // is already calculated.
    // Mathematically epilogue_t is a map that applies to each element:
    //   epilogue_t: [m, n] -> [m, n], C_acc |-> tile_op_t(C_acc)
    using epilogue_policy
            = xetla::group::epilogue_policy_tile_op<tile_op_t, gpu_arch::Xe>;

    // Mirco-kernel configuration
    using tune_option = dict_t<
            elem_v_t<tune_key::PARAM_OPTIMZER_TYPE,
                    tune_key_value::PARAM_OPTIMZER_DECISION_TREE>,
            elem_t_t<tune_key::EPILOGUE_POLICY, epilogue_policy>,
            elem_t_t<tune_key::WG_TILE_SHAPE, shape<wg_tile_n, wg_tile_m>>,
            elem_t_t<tune_key::SG_TILE_SHAPE, shape<sg_tile_n, sg_tile_m>>>;
    using default_config_t = gpu::xetla::kernel::default_gemm_config_t<
            data_type_a, // input datatype for A
            mem_layout::row_major, // memory layout for A
            8, // leading dimension alignment for A, in unit of element
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for B
            8, // leading dimension alignment for B, in unit of element
            data_type_c, // output datatype for C
            mem_layout::row_major, // memory layout for C
            8, // leading dimension alignment for C, in unit of element
            data_type_acc, // accumulator data type for intermediate resutls
            gpu_arch::Xe, // GPU arch
            tune_option>;

    using gemm_op_t = typename default_config_t::type;
    using epilogue_t = typename default_config_t::epilogue_t;

    // [Polynomial] define arguments for each epilogue_tile_op in chained_tile_op_t<>
    using epilogue_tile_op_args_t = epilogue_t::tile_op_t::arguments_t;
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

    // set up gemm_universal arguments
    typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A,
            matrix_k, B, matrix_n, C, matrix_n, {epilogue_args});

    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);

    if (!gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        free(A, context);
        free(B, context);
        free(C, context);
        FAIL();
    }

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k
            + 6 * static_cast<long>(matrix_m) * matrix_n;
    profiling_helper prof("gemm_polynomial", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
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
