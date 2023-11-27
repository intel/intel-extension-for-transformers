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

using namespace cl::sycl;
using namespace gpu::xetla;
using namespace gpu;

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_d, typename data_type_acc = float>
int gemm_relu_bias_result_validate(data_type_a *A_device, data_type_b *B_device,
        data_type_c *C_device, data_type_d *D_device, uint32_t m, uint32_t k,
        uint32_t n, sycl::queue &queue,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    auto A = alloc_host_and_copy<data_type_a>(A_device, m * k, queue);
    auto B = alloc_host_and_copy<data_type_b>(B_device, k * n, queue);
    auto C = alloc_host_and_copy<data_type_c>(C_device, m * n, queue);
    auto D = alloc_host_and_copy<data_type_d>(D_device, n, queue);

    bool is_col_major_a = mem_layout_a_ == mem_layout::col_major;
    bool is_col_major_b = mem_layout_b_ == mem_layout::col_major;
    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
    std::vector<data_type_acc> gold_C(m * n, 0);
    get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_b_, A, B, gold_C.data());
    // ReLU
    std::transform(gold_C.cbegin(), gold_C.cend(), gold_C.begin(),
            [](data_type_acc c) { return c > 0.0f ? c : 0.0f; });
    // BiasAdd
    for (uint32_t i = 0; i < gold_C.size(); ++i) {
        uint32_t col = gold_C.size() % n;
        gold_C[i] += D[col];
    }
    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m, n, n);

    bool result = buff_cmp::xetla_buff_cmp(
            data, other, "gemm_relu_bias validation");

    free(A);
    free(B);
    free(C);
    free(D);

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

void gemm_relu_bias_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    //GEMM input size
    uint32_t matrix_m = 4096;
    uint32_t matrix_n = 4096;
    uint32_t matrix_k = 4096;

    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;
    uint32_t size_d = 1 * matrix_n;

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
    using data_type_d = float;
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
    auto D = alloc_device_and_init<data_type_d>(
            size_d,
            [](data_type_d *data, size_t idx) {
                data[idx] = static_cast<data_type_d>(random_float());
            },
            queue, device, context);

    //Define the shape of workgroup
    //It's tunable parameters based on different input shape and hardware for better performance
    constexpr uint32_t wg_tile_m = 256;
    constexpr uint32_t wg_tile_n = 256;

    // [ReLuBias] Chain multiple elementwise op in chained_tile_op_t<>: relu_op_t, bias_add_op_t
    using mem_desc_bias_t = xetla::mem_desc_t<float, mem_layout::row_major,
            mem_space::global>;

    using bias_op_t
            = xetla::subgroup::bias_add_op_t<mem_desc_bias_t, gpu_arch::Xe>;
    using tile_op_t = xetla::subgroup::chained_tile_op_t<
            xetla::subgroup::relu_op_t, // apply elementwise ReLU
            bias_op_t // apply elementwise BiasAdd
            >;
    // [ReLuBias] epilogue_t is an elementwise operation that will be applied to the
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
            elem_t_t<tune_key::WG_TILE_SHAPE, shape<wg_tile_n, wg_tile_m>>>;
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

    // [ReLuBias] define the shape of bias matrix D, which should be identitcal to C
    bias_op_t::shape_t bias_add_shape(matrix_n, 1, matrix_n);
    // [ReLuBias] pass arguments of chained_tile_op_t<> to epilogue_args
    using epilogue_args_t = typename default_config_t::epilogue_t::arguments_t;
    epilogue_args_t epilogue_args({//epilogue_args init list
            // [ReLuBias] 1. relu_op_t
            // ReLU accepts no arguments
            {},
            // [ReLuBias] 2. bias_add_op_t
            // It accepts the base pointer to matrix D, and its dimensions
            {D, bias_add_shape}});
    // [ReLuBias] assign epilogue_args to gemm_op_t::arguments_t
    typename gemm_op_t::arguments_t arg(matrix_m, matrix_k, matrix_n, A,
            matrix_k, B, matrix_n, C, matrix_n, epilogue_args);
    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(arg);
    if (!gemm_op_t::can_implement(arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        FAIL();
    }

    constexpr uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k
            + matrix_m * matrix_n;
    profiling_helper prof("gemm_relu_bias_run", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                // allocate slm and nbarrier resource
                slm_barrier_init<gemm_op_t>();
                gemm_op_t gemm_op;
                gemm_op(item, arg);
            });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
        }
    }

    ASSERT_EQ(0,
            gemm_relu_bias_result_validate(A, B, C, D, matrix_m, matrix_k,
                    matrix_n, queue, mem_layout::row_major,
                    mem_layout::row_major));

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
    free(D, context);
}

int main() {
    // The purpose of this example is to illustrate the epilogue_t API in XeTLA.

    // It allows user to implement multiple Ops inside a kernel call to avoid
    // overheads in invokation, memory transfer, etc.
    // Take the following python code as an example:

    // Original:
    // > import torch as to
    // > x = to.matmul(A, B)
    // > y = to.nn.functional.relu(x)

    // It takes two kernel invokations and the ReLU Op is a elementwise operation
    // that could be fused into MatMul Op, which is basically calling GEMM kernel.

    // Fusion:
    // > import torch as to
    // > y = MatMulReLU(A, B)
    // The MatMulReLU Op corresponds to the second example presented below.

    // It allows the user to apply custom operations in GEMM computation.
    // Here provides some possible configurations using epilogue_t:
    // - GEMM
    //   C  = A x B
    // - tile_op_t=relu_op_t
    //   C = ReLU(A x B)
    // - tile_op_t=[relu_op_t, bias_add_op_t]
    //   C = BiasAdd(ReLU(A x B))
    //     = ReLU(A x B) + D
    //  where:
    //    shape(A) = [m, k]
    //    shape(B) = [k, n]
    //    shape(C) = [m, n]
    //    shape(D) = [1, n]
    // This example will implement the last variant that chains multiple
    // operations, which demonstrates its maximal flexibility.
    // checkout op_functor.hpp for more elementwise ops

    // Note:
    //   - comments related to this example will be prefixed with "[ReLuBias]"
    gemm_relu_bias_run(10);
    return (0);
}
