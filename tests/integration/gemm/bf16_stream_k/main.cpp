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

#include "utils/utils.hpp"
#include "xetla.hpp"

using namespace gpu::xetla;
//The number of times the kernel is executed
constexpr int ITER = 1;

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_d, typename data_type_acc = float>
int gemm_result_validate(data_type_a *A_device, data_type_b *B_device,
        data_type_c *C_device, data_type_d *D_device, uint32_t m, uint32_t k,
        uint32_t n, sycl::queue &queue,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major,
        bool postop_enable = false) {

    auto A = alloc_host_and_copy<data_type_a>(A_device, m * k, queue);
    auto B = alloc_host_and_copy<data_type_b>(B_device, k * n, queue);
    auto C = alloc_host_and_copy<data_type_c>(C_device, m * n, queue);
    auto D = alloc_host_and_copy<data_type_d>(D_device, n, queue);

    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
    std::vector<data_type_acc> gold_C(m * n, 0);
    get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_b_, A, B, gold_C.data());

    if (postop_enable) {

        // ReLU
        std::transform(gold_C.cbegin(), gold_C.cend(), gold_C.begin(),
                [](data_type_acc c) { return c > 0.0f ? c : 0.0f; });
        // BiasAdd
        for (uint32_t i = 0; i < gold_C.size(); ++i) {
            uint32_t col = gold_C.size() % n;
            gold_C[i] += D[col];
        }
    }

    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m, n, n);

    bool result = buff_cmp::xetla_buff_cmp(data, other, "gemm validation");

    free(A);
    free(B);
    free(C);
    free(D);

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

//Test 0.25 wave on 1 Tile PVC.  Multiple stream_k regions are created. Generalizes to fixed split-K technique
class Test1 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1024;
    static constexpr size_t mat_n = 1024;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t num_xecores = 64;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    static constexpr bool postop_enable = false;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
};

//Test 2 waves on 1 Tile PVC.  StreamK split not used, Generalizes to regular data-parallel GEMM approach.
class Test2 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 2048;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t num_xecores = 64;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    static constexpr bool postop_enable = false;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
};

//Test  1.56 wave on 1 Tile PVC.  One StreamK wave created.
class Test3 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 2560;
    static constexpr size_t mat_n = 2560;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t num_xecores = 64;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    static constexpr bool postop_enable = false;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
};

//Test  2.25 wave on 1 Tile PVC.  One StreamK wave and one data parallel wave created.
class Test4 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 3072;
    static constexpr size_t mat_n = 3072;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr size_t num_xecores = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    static constexpr bool postop_enable = false;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
};

//Test  2.25 wave on 1 Tile PVC.  One StreamK wave and one data parallel wave created. Also uses fused epilogue operations
class Test5 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 3072;
    static constexpr size_t mat_n = 3072;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr size_t num_xecores = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    static constexpr bool postop_enable = true;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
};

//Test 3.515625 wave on 1 Tile PVC.  One StreamK wave and Two Data Parallel waves created. Also tests big_groups logic, where one of the stream_k groups perform a extra iteration
class Test6 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 3840;
    static constexpr size_t mat_n = 3840;
    static constexpr size_t mat_k = 3840;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr size_t num_xecores = 64;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    static constexpr bool postop_enable = false;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
};

template <class Test>
void stream_k_gemm_run(uint32_t iter) {

    using namespace gpu;
    //Accept incoming parameters
    constexpr size_t matrix_m = Test::mat_m;
    constexpr size_t matrix_n = Test::mat_n;
    constexpr size_t matrix_k = Test::mat_k;

    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;

    constexpr mem_layout mem_layout_a = Test::layout_a;
    constexpr mem_layout mem_layout_b = Test::layout_b;
    constexpr bool postop_enable = Test::postop_enable;

    //StreamK parameters - xecores available for stream_k dispatch
    constexpr size_t avail_xecores = Test::num_xecores;

    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;
    uint32_t atomic_flag_space = 1024;
    uint32_t size_bias = matrix_n;

    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_c = typename Test::data_type_c;
    using data_type_acc = float;
    using data_type_bias = float;

    //Turn on the profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    //Define SYCL queue, context and device
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    // Org the compute shape for sub-matrix
    using tile_shape
            = xetla::group::tile_shape_t<wg_tile_n, // workgroup size in dim0
                    wg_tile_m, //	workgroup size in dim1
                    sg_tile_n, //	subgroup size in dim0
                    sg_tile_m>; //	subgroup size in dim1

    static constexpr uint32_t periodic_sync_interval = 4;
    static constexpr uint32_t prefetch_distance = 4;

    // Mirco-kernel configuration
    using gemm_config = typename xetla::group::gemm_selector_t<
            data_type_a, // input datatype for A
            data_type_b, // input datatype for B
            mem_layout_a, // memory layout for A
            mem_layout_b, // memory layout for B
            mem_space::global, // memory reading from global mem for A
            mem_space::global, // memory reading from global mem for B
            8, // leading dimension for A, in unit of element
            8, // leading dimension for B, in unit of element
            data_type_acc, // accumulator data type for intermediate resutls
            tile_shape, // computation tile shape
            sg_tile_k, // elements in each iteration
            mma_engine::xmx, // compute engine
            gpu_arch::Xe, prefetch_distance,
            periodic_sync_interval> // GPU arch, prefetch stages, periodic sync frequency
            ::gemm;

    // Test post-op fusion with stream_k
    //[ReLuBias] epilogue_t is an elementwise operation that will be applied to the
    // accumulator C_acc in the final stage, in which
    //   C_acc = A x B
    // is already calculated.
    // Mathematically epilogue_t is a map that applies to each element:
    //   epilogue_t: [m, n] -> [m, n], C_acc |-> tile_op_t(C_acc)
    // [ReLuBias] Chain multiple elementwise op in chained_tile_op_t<>: relu_op_t, bias_add_op_t
    using mem_desc_bias_t = xetla::mem_desc_t<data_type_bias,
            mem_layout::row_major, mem_space::global>;
    using bias_op_t
            = xetla::subgroup::bias_add_op_t<mem_desc_bias_t, gpu_arch::Xe>;
    using tile_op_t = xetla::subgroup::chained_tile_op_t<
            xetla::subgroup::relu_op_t, // apply elementwise ReLU
            bias_op_t // apply elementwise BiasAdd
            >;

    using epilogue_policy_t = typename std::conditional<postop_enable == 0,
            xetla::group::epilogue_policy_default<gpu_arch::Xe>,
            xetla::group::epilogue_policy_tile_op<tile_op_t,
                    gpu_arch::Xe>>::type;

    using epilogue_t = xetla::group::epilogue_t<epilogue_policy_t, tile_shape,
            mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>>;

    using dispatch_stream_k
            = gpu::xetla::kernel::dispatch_policy_stream_k<gpu_arch::Xe>;

    using gemm_op_t = xetla::kernel::gemm_universal_t<dispatch_stream_k,
            gemm_config, epilogue_t>;

    // setup stream_k workgroup split
    dispatch_stream_k stream_k(matrix_m, matrix_k, matrix_n, wg_tile_m,
            gemm_config::k_stride, wg_tile_n, sg_tile_m, sg_tile_n,
            avail_xecores);

    setenv("SYCL_PROGRAM_COMPILE_OPTIONS",
            " -vc-codegen -doubleGRF -vc-disable-indvars-opt "
            " -Xfinalizer '-printregusage -enableBCR -DPASTokenReduction '",
            1);

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

    auto Bias = alloc_device_and_init<data_type_bias>(
            size_bias,
            [](data_type_bias *data, size_t idx) {
                data[idx] = static_cast<data_type_bias>(random_float());
            },
            queue, device, context);

    using epilogue_args_t = typename epilogue_t::arguments_t;
    uint32_t warmup = 0;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof("stream_k_universal_gemm", ops, "gflops");

    if constexpr (postop_enable) {

        // [ReLuBias] define the shape of matrix bias, which should be identitcal to C
        bias_op_t::shape_t bias_add_shape(matrix_n, 1, matrix_n);
        // [ReLuBias] pass arguments of chained_tile_op_t<> to epilogue_args

        epilogue_args_t epilogue_args({//epilogue_args init list
                // [ReLuBias] 1. relu_op_t
                // ReLU accepts no arguments
                {},
                // [ReLuBias] 2. bias_add_op_t
                // It accepts the base pointer to matrix bias, and its dimensions
                {Bias, bias_add_shape}});

        typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n,
                A, matrix_k, B, matrix_n, C, matrix_n, Acc, matrix_n, Cnt,
                size_cnt, stream_k, epilogue_args);

        cl::sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);

        if (!gemm_op_t::can_implement(gemm_arg)) {
            std::cout << "The arguments cannot be supported, aborting ... "
                      << std::endl;
            FAIL();
        }

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

    }

    else {

        epilogue_args_t epilogue_args {};
        // set up gemm arguments
        typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n,
                A, matrix_k, B, matrix_n, C, matrix_n, Acc, matrix_n, Cnt,
                size_cnt, stream_k, epilogue_args);

        if (!gemm_op_t::can_implement(gemm_arg)) {
            std::cout << "The arguments cannot be supported, aborting ... "
                      << std::endl;
            FAIL();
        }
        cl::sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);

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
    }

    unsetenv("SYCL_PROGRAM_COMPILE_OPTIONS");

    ASSERT_EQ(0,
            gemm_result_validate(A, B, C, Bias, matrix_m, matrix_k, matrix_n,
                    queue, mem_layout_a, mem_layout_b, postop_enable));

    //performance
    if (iter > 0) { prof.print_profiling_result(profiling_selector::GPU); }

    free(A, context);
    free(B, context);
    free(C, context);
    free(Acc, context);
    free(Cnt, context);
    free(Bias, context);
}

template <typename T>
class stream_k_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(stream_k_gemm_test);

TYPED_TEST_P(stream_k_gemm_test, esimd) {
    stream_k_gemm_run<TypeParam>(ITER);
}

REGISTER_TYPED_TEST_SUITE_P(stream_k_gemm_test, esimd);
using tests = ::testing::Types<Test1, Test2, Test3, Test4, Test5, Test6>;

INSTANTIATE_TYPED_TEST_SUITE_P(
        stream_k_gemm_test_suite, stream_k_gemm_test, tests);
