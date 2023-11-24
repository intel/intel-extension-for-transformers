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

void stream_k_gemm_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    //GEMM input size
    uint32_t matrix_m = 3072;
    uint32_t matrix_n = 3072;
    uint32_t matrix_k = 4096;

    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;

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

    //Define and initialize the data required for the calculation
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

    //StreamK parameters - xecores available for stream_k dispatch
    uint32_t avail_xecores = 32;

    // Org the compute shape for sub-matrix
    using tile_shape
            = xetla::group::tile_shape_t<wg_tile_n, // workgroup size in dim0
                    wg_tile_m, //	workgroup size in dim1
                    sg_tile_n, //	subgroup size in dim0
                    sg_tile_m>; //	subgroup size in dim1

    // Mirco-kernel configuration
    using gemm_config = xetla::group::gemm_selector_t<
            data_type_a, // input datatype for A
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for A
            mem_layout::row_major, // memory layout for B
            mem_space::global, // memory reading from global mem for A
            mem_space::global, // memory reading from global mem for B
            8, // leading dimension for A, in unit of element
            8, // leading dimension for B, in unit of element
            data_type_acc, // accumulator data type for intermediate resutls
            tile_shape, // computation tile shape
            sg_tile_k, // elements in each iteration
            mma_engine::xmx, // compute engine
            gpu_arch::Xe, 3,
            4> // GPU arch, prefetch stages, periodic sync frequency
            ::gemm;

    using dispatch_stream_k
            = gpu::xetla::kernel::dispatch_policy_stream_k<gpu_arch::Xe>;

    using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_default<gpu_arch::Xe>, tile_shape,
            mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>>;

    using gemm_op_t = xetla::kernel::gemm_universal_t<dispatch_stream_k,
            gemm_config, epilogue_t>;

    // setup stream_k workgroup split
    dispatch_stream_k stream_k(matrix_m, matrix_k, matrix_n, wg_tile_m,
            gemm_config::k_stride, wg_tile_n, sg_tile_m, sg_tile_n,
            avail_xecores);

    // allocate temp buffers for global split
    size_t size_acc = gemm_op_t::get_acc_buf_size(stream_k);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(stream_k);

    auto Acc = alloc_device_and_init<data_type_acc>(
            size_acc,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0.0f);
            },
            queue, device, context);
    auto Cnt = alloc_device_and_init<uint32_t>(
            size_cnt,
            [](uint32_t *data, size_t idx) {
                data[idx] = static_cast<uint32_t>(0);
            },
            queue, device, context);

    // set up gemm arguments
    typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A,
            matrix_k, B, matrix_n, C, matrix_n, Acc, matrix_n, Cnt, size_cnt,
            stream_k);

    cl::sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);

    if (!gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        free(A, queue);
        free(B, queue);
        free(C, queue);
        free(Acc, queue);
        free(Cnt, queue);
        FAIL();
    }

    uint32_t warmup = 5;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof("stream_k_universalgemm", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(NDRange, [=](nd_item<3> item) KERNEL_MAIN {
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
    }

    ASSERT_EQ(0,
            gemm_result_validate(A, B, C, 1, matrix_m, matrix_k, matrix_n,
                    queue, mem_layout::row_major, mem_layout::row_major));

    //performance
    if (iter > 0) { prof.print_profiling_result(profiling_selector::GPU); }

    free(A, context);
    free(B, context);
    free(C, context);
    free(Acc, context);
    free(Cnt, context);
}

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

void stream_k_gemm_relu_biasadd_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    //GEMM input size
    uint32_t matrix_m = 3072;
    uint32_t matrix_n = 3072;
    uint32_t matrix_k = 4096;

    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;
    uint32_t size_bias = matrix_n;

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
    using data_type_acc = float;
    using data_type_bias = float;

    //Turn on the profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    //Define SYCL queue, context and device
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    //Define and initialize the data required for the calculation
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

    auto Bias = alloc_device_and_init<data_type_bias>(
            size_bias,
            [](data_type_bias *data, size_t idx) {
                data[idx] = static_cast<data_type_bias>(random_float());
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
    using gemm_config = xetla::group::gemm_selector_t<
            data_type_a, // input datatype for A
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for A
            mem_layout::row_major, // memory layout for B
            mem_space::global, // memory reading from global mem for A
            mem_space::global, // memory reading from global mem for B
            8, // leading dimension for A, in unit of element
            8, // leading dimension for B, in unit of element
            data_type_acc, // accumulator data type for intermediate resutls
            tile_shape, // computation tile shape
            sg_tile_k, // elements in each iteration
            mma_engine::xmx, // compute engine
            gpu_arch::Xe> // GPU arch
            ::gemm;

    // [ReLuBias] Chain multiple elementwise op in chained_tile_op_t<>: relu_op_t, bias_add_op_t
    using bias_op_t
            = xetla::subgroup::bias_add_op_t<data_type_bias, gpu_arch::Xe>;
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
    using mem_desc_output_t
            = mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>;
    using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_tile_op<tile_op_t, gpu_arch::Xe>,
            tile_shape, mem_desc_output_t>;

    using dispatch_stream_k
            = gpu::xetla::kernel::dispatch_policy_stream_k<gpu_arch::Xe>;

    using gemm_op_t = xetla::kernel::gemm_universal_t<dispatch_stream_k,
            gemm_config, epilogue_t>;

    // [ReLuBias] define the shape of bias matrix bias, which should be identitcal to C
    bias_op_t::shape_t bias_add_shape(matrix_n, 1, matrix_n);
    // [ReLuBias] pass arguments of chained_tile_op_t<> to epilogue_args
    using epilogue_args_t = epilogue_t::arguments_t;
    epilogue_args_t epilogue_args({//epilogue_args init list
            // [ReLuBias] 1. relu_op_t
            // ReLU accepts no arguments
            {},
            // [ReLuBias] 2. bias_add_op_t
            // It accepts the base pointer to matrix bias, and its dimensions
            {Bias, bias_add_shape}});

    // setup stream_k workgroup split
    dispatch_stream_k stream_k(matrix_m, matrix_k, matrix_n, wg_tile_m,
            gemm_config::k_stride, wg_tile_n, sg_tile_m, sg_tile_n);
    // allocate temp buffers for splitK
    size_t size_acc = gemm_op_t::get_acc_buf_size(stream_k);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(stream_k);

    auto Acc = alloc_device_and_init<data_type_acc>(
            size_acc,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0.0f);
            },
            queue, device, context);
    auto Cnt = alloc_device_and_init<uint32_t>(
            size_cnt,
            [](uint32_t *data, size_t idx) {
                data[idx] = static_cast<uint32_t>(0);
            },
            queue, device, context);

    // set up gemm arguments
    typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A,
            matrix_k, B, matrix_n, C, matrix_n, Acc, matrix_n, Cnt, size_cnt,
            stream_k, epilogue_args);

    cl::sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);

    if (!gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        FAIL();
    }

    uint32_t warmup = 5;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof(
            "stream_k_universal_gemm_relu_biasadd", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(NDRange, [=](nd_item<3> item) KERNEL_MAIN {
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
    }

    ASSERT_EQ(0,
            gemm_relu_bias_result_validate(A, B, C, Bias, matrix_m, matrix_k,
                    matrix_n, queue, mem_layout::row_major,
                    mem_layout::row_major));

    //performance
    if (iter > 0) { prof.print_profiling_result(profiling_selector::GPU); }

    free(A, context);
    free(B, context);
    free(C, context);
    free(Acc, context);
    free(Cnt, context);
    free(Bias, context);
}

int main() {

    //Example for stream_k parallel decomposition algorithm
    //Implementation loosely based on this paper "Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU" (https://arxiv.org/abs/2301.03598)
    //First example demonstrates stream_k GEMM split , Second example demonstrates stream_k GEMM + post-op fusion
    stream_k_gemm_run(10);
    stream_k_gemm_relu_biasadd_run(10);
    return (0);
}
