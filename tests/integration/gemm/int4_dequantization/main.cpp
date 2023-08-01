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
        typename data_type_acc = float>
int gemm_result_validate(data_type_a *A, data_type_b *B, data_type_c *C,
        uint32_t m, uint32_t k, uint32_t n,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
    std::vector<data_type_acc> gold_C(m * n, 0);
    get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_b_, A, B, gold_C.data());
    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m, n, n);

    bool result = buff_cmp::xetla_buff_cmp(data, other, "gemm validation");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

class qkv {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 4096 * 3;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 32;
    static constexpr size_t dequant_s = 128;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t slm_kslicing = 2;
    static constexpr size_t l3_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};

class output_proj {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 64;
    static constexpr size_t dequant_s = 128;
    static constexpr size_t num_buffer = 128;
    static constexpr size_t slm_kslicing = 8;
    static constexpr size_t l3_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};

class ffn1 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 16384;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 32;
    static constexpr size_t dequant_s = 128;
    static constexpr size_t num_buffer = 32;
    static constexpr size_t slm_kslicing = 2;
    static constexpr size_t l3_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};

class ffn2 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 16384;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 64;
    static constexpr size_t dequant_s = 128;
    static constexpr size_t num_buffer = 32;
    static constexpr size_t slm_kslicing = 8;
    static constexpr size_t l3_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};

class last {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 50400;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 32;
    static constexpr size_t dequant_s = 128;
    static constexpr size_t num_buffer = 16;
    static constexpr size_t slm_kslicing = 1;
    static constexpr size_t l3_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};

template <class Test>
void dequantize_gemm_run(int iter) {
    using namespace gpu;
    //Accept incoming parameters
    constexpr size_t matrix_m = Test::mat_m;
    constexpr size_t matrix_n = Test::mat_n;
    constexpr size_t matrix_k = Test::mat_k;
    constexpr uint32_t l3_kslicing = Test::l3_kslicing;
    constexpr uint32_t slm_kslicing = Test::slm_kslicing;

    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;
    constexpr size_t dequant_s = Test::dequant_s;
    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_c = typename Test::data_type_c;
    using data_type_zero_pt = int4x2;
    using data_type_scale = fp16;
    using data_type_acc_in = fp16;
    using data_type_acc = float;

    constexpr size_t size_a = matrix_m * matrix_k;
    constexpr size_t size_b = matrix_k * matrix_n / 2;

    constexpr size_t size_scale_m = matrix_k / dequant_s;
    constexpr size_t size_scale_n = matrix_n;
    constexpr size_t size_scale = size_scale_m * size_scale_n;

    constexpr size_t size_zero_pt_m = matrix_k / dequant_s;
    constexpr size_t size_zero_pt_n = matrix_n / 2;
    constexpr size_t size_zero_pt = size_zero_pt_m * size_zero_pt_n;

    constexpr size_t size_c = matrix_m * matrix_n;
    uint32_t lda = matrix_k;
    uint32_t ldb = matrix_n;
    uint32_t ldc = matrix_n;
    uint32_t ld_scale = size_scale_n;
    uint32_t ld_zero_pt = size_zero_pt_n;

    //Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto Queue = queue(properties);
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    //Define and initialize the data required for the calculation
    auto *A_h = static_cast<data_type_a *>(
            malloc_host(size_a * sizeof(data_type_a), Context));
    auto *B_h = static_cast<data_type_b *>(
            malloc_host(size_b * sizeof(data_type_b), Context));
    auto *C_h = static_cast<data_type_c *>(
            malloc_host(size_c * sizeof(data_type_c), Context));
    auto *scale_h = static_cast<data_type_scale *>(
            malloc_host(size_scale * sizeof(data_type_scale), Context));
    auto *zero_pt_h = static_cast<data_type_zero_pt *>(
            malloc_host(size_zero_pt * sizeof(data_type_zero_pt), Context));

    auto *A_d = static_cast<data_type_a *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_a * sizeof(data_type_a), Device, Context));
    auto *B_d = static_cast<data_type_b *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_b * sizeof(data_type_b), Device, Context));
    auto *C_d = static_cast<data_type_c *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_c * sizeof(data_type_c), Device, Context));
    auto *scale_d = static_cast<data_type_scale *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_scale * sizeof(data_type_scale), Device, Context));
    auto *zero_pt_d = static_cast<data_type_zero_pt *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_zero_pt * sizeof(data_type_zero_pt), Device, Context));

    for (unsigned i = 0; i < size_a; ++i) {
        A_h[i] = random_float();
    }
    for (unsigned i = 0; i < size_b; ++i) {
        B_h[i] = uint8_t(rand());
    }
    for (unsigned i = 0; i < size_scale; ++i) {
        scale_h[i] = random_float();
    }
    for (unsigned i = 0; i < size_zero_pt; ++i) {
        zero_pt_h[i] = uint8_t(rand());
    }
    for (unsigned i = 0; i < size_c; ++i) {
        C_h[i] = 0;
    }

    Queue.memcpy((void *)A_d, (void *)A_h, size_a * sizeof(data_type_a)).wait();
    Queue.memcpy((void *)B_d, (void *)B_h, size_b * sizeof(data_type_b)).wait();
    Queue.memcpy((void *)C_d, (void *)C_h, size_c * sizeof(data_type_c)).wait();
    Queue.memcpy((void *)scale_d, (void *)scale_h,
                 size_scale * sizeof(data_type_scale))
            .wait();
    Queue.memcpy((void *)zero_pt_d, (void *)zero_pt_h,
                 size_zero_pt * sizeof(data_type_zero_pt))
            .wait();

    using tile_shape = xetla::group::tile_shape_t<wg_tile_n, wg_tile_m,
            sg_tile_n, sg_tile_m>;
    static constexpr uint32_t periodic_sync_interval = 1;
    static constexpr uint32_t prefetch_distance = 3;

    using mem_desc_a_t = xetla::mem_desc_t<data_type_a, mem_layout::row_major,
            mem_space::global>;
    using mem_desc_b_t = xetla::mem_desc_t<data_type_b, mem_layout::row_major,
            mem_space::global>;
    using mem_desc_c_t = xetla::mem_desc_t<data_type_c, mem_layout::row_major,
            mem_space::global>;

    using compute_attr = xetla::group::compute_attr_t<data_type_acc_in,
            data_type_acc_in, data_type_acc>;
    using perf_tuning_knob = xetla::group::perf_tuning_knob_t<sg_tile_k,
            prefetch_distance, periodic_sync_interval>;
    using compute_policy
            = xetla::group::compute_policy_int4_dequantize_xmx<compute_attr,
                    perf_tuning_knob, data_type_scale, data_type_zero_pt,
                    dequant_s, gpu_arch::Xe>;
    using brgemm_t = xetla::group::brgemm_t<compute_policy, tile_shape,
            mem_desc_a_t, mem_desc_b_t>;

    using update_method = typename std::conditional<(l3_kslicing > 1),
            result_reduce_sum, result_overwrite>::type;
    using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_default<update_method, gpu_arch::Xe>,
            tile_shape, mem_desc_c_t>;
    using gemm_op_t = xetla::kernel::gemm_t<
            gpu::xetla::kernel::dispatch_policy_int4_dequantize_kslicing<
                    l3_kslicing, slm_kslicing, gpu_arch::Xe>,
            brgemm_t, epilogue_t>;

    // set up gemm arguments
    typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A_d,
            matrix_k, B_d, matrix_n, C_d, matrix_n, scale_d, matrix_n,
            zero_pt_d, matrix_n);

    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
    if (!gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        FAIL();
    }

    size_t ops = 2 * matrix_m * matrix_n * matrix_k;
    profiling_helper prof("dequantize_gemm", ops, "gflops");
    try {
        for (int i = 0; i < iter; i++) {
            prof.cpu_start();
            auto e_esimd = Queue.submit([&](handler &cgh) {
                cgh.parallel_for<Test>(
                        nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                            xetla_exec_item<3> ei(item);
                            // allocate slm and nbarrier resource
                            slm_barrier_init<gemm_op_t>();
                            gemm_op_t gemm_op;
                            gemm_op(ei, gemm_arg);
                        });
            });
            e_esimd.wait();
            prof.cpu_end();
            prof.add_gpu_event(e_esimd);
        }
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    std::vector<fp16> dequantize_b(matrix_k * matrix_n, 0);
    for (int i = 0; i < matrix_k / dequant_s; i++) {
        for (int j = 0; j < matrix_n / 2; j++) {
            int start_in = i * dequant_s * matrix_n / 2 + j;
            int start_zero_pt = i * size_zero_pt_n + j;
            int start_out = i * dequant_s * matrix_n + j * 2;
            int start_scale = i * size_scale_n + j * 2;
            for (int ii = 0; ii < dequant_s; ii++) {
                uint8_t data_in = B_h[start_in + ii * matrix_n / 2];
                uint8_t data_zero_pt = zero_pt_h[start_zero_pt];
                int8_t data_0 = int8_t(data_in & 0x0f);
                int8_t data_1 = int8_t(data_in >> 4);
                int8_t zero_pt_0 = int8_t((data_zero_pt & 0x0f) + 1);
                int8_t zero_pt_1 = int8_t((data_zero_pt >> 4) + 1);
                dequantize_b[start_out + ii * matrix_n]
                        = fp16(data_0 - zero_pt_0) * scale_h[start_scale];
                dequantize_b[start_out + ii * matrix_n + 1]
                        = fp16(data_1 - zero_pt_1) * scale_h[start_scale + 1];
            }
        }
    }

    Queue.memcpy((void *)C_h, (void *)C_d, size_c * sizeof(data_type_c)).wait();
    ASSERT_EQ(0,
            gemm_result_validate(A_h, dequantize_b.data(), C_h, matrix_m,
                    matrix_k, matrix_n));

    free(A_h, Context);
    free(B_h, Context);
    free(C_h, Context);
    free(scale_h, Context);
    free(zero_pt_h, Context);
    free(A_d, Context);
    free(B_d, Context);
    free(C_d, Context);
    free(scale_d, Context);
    free(zero_pt_d, Context);
}

template <typename T>
class dequantize_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(dequantize_gemm_test);

TYPED_TEST_P(dequantize_gemm_test, esimd) {
    dequantize_gemm_run<TypeParam>(ITER);
}

REGISTER_TYPED_TEST_SUITE_P(dequantize_gemm_test, esimd);
using tests = ::testing::Types<qkv, output_proj, ffn1, ffn2, last>;

INSTANTIATE_TYPED_TEST_SUITE_P(
        dequantize_gemm_test_suite, dequantize_gemm_test, tests);
