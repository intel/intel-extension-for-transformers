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

#include "softmax.hpp"
#include "tests/utils/utils.hpp"
#include "xetla.hpp"

using namespace gpu::xetla;
using namespace cl::sycl;

#define SIMD 32

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float>
int gemm_softmax_result_validate(data_type_a *A_ptr, data_type_b *B_ptr,
        data_type_c *C_ptr, uint32_t m, uint32_t k, uint32_t n,
        uint32_t batch_num, mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    uint32_t err_cnt = 0;
    bool is_col_major_a = mem_layout_a_ == mem_layout::col_major;
    bool is_col_major_b = mem_layout_b_ == mem_layout::col_major;
    uint32_t size_a = m * k;
    uint32_t size_b = k * n;
    uint32_t size_c = m * n;
    std::vector<data_type_acc> tmp_A(A_ptr, A_ptr + batch_num * size_a);
    std::vector<data_type_acc> tmp_B(B_ptr, B_ptr + batch_num * size_b);
    std::vector<data_type_acc> gold_C(batch_num * size_c, 0);
    for (uint32_t batch_id = 0; batch_id < batch_num; batch_id++) {
        get_gemm_gold(m, n, k, mem_layout_a_, mem_layout_b_,
                tmp_A.data() + batch_id * size_a,
                tmp_B.data() + batch_id * size_b,
                gold_C.data() + batch_id * size_c);
    }

    for (uint32_t batch_id = 0; batch_id < batch_num; ++batch_id) {
        for (uint32_t i = 0; i < m; i++) {
            data_type_acc row_max = 0;
            data_type_acc exp_sum = 0;
            uint32_t sfx_offset = batch_id * size_c + i * n;
            for (uint32_t j = 0; j < n; ++j) {
                row_max = max(row_max, gold_C[sfx_offset + j]);
            }
            for (uint32_t j = 0; j < n; ++j) {
                gold_C[sfx_offset + j] = exp(gold_C[sfx_offset + j] - row_max);
                exp_sum += gold_C[sfx_offset + j];
            }
            for (uint32_t j = 0; j < n; ++j) {
                gold_C[sfx_offset + j] /= exp_sum;
            }
        }
    }

    buff_cmp::buff_vals<data_type_c> data(C_ptr, m * batch_num, n, n);
    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m * batch_num, n, n);
    bool result
            = buff_cmp::xetla_buff_cmp(data, other, "gemm_softmax validation");

    std::cout << ((!result) ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

void gemm_softmax_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    uint32_t batch_num = 256;
    uint32_t matrix_m = 512;
    uint32_t matrix_n = 512;
    uint32_t matrix_k = 64;

    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
    using data_type_sfx = float;

    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    auto Queue = queue(properties);
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    data_type_a *A = static_cast<data_type_a *>(malloc_shared(
            batch_num * size_a * sizeof(data_type_a), Device, Context));
    data_type_b *B = static_cast<data_type_b *>(malloc_shared(
            batch_num * size_b * sizeof(data_type_b), Device, Context));
    data_type_c *C = static_cast<data_type_c *>(malloc_shared(
            batch_num * size_c * sizeof(data_type_c), Device, Context));

    for (unsigned i = 0; i < batch_num * size_a; ++i) {
        A[i] = static_cast<data_type_a>(random_float());
    }
    for (unsigned i = 0; i < batch_num * size_b; ++i) {
        B[i] = static_cast<data_type_b>(random_float());
    }
    for (unsigned i = 0; i < batch_num * size_c; ++i) {
        C[i] = static_cast<data_type_c>(0.f);
    }

    constexpr uint32_t wg_tile_m = 64;
    constexpr uint32_t wg_tile_n = 512;
    constexpr uint32_t sg_tile_m = 32;
    constexpr uint32_t sg_tile_n = 32;
    constexpr uint32_t sg_tile_k = 16;

    // buffer size of softmax row data
    constexpr uint32_t softmax_size = 512;
    // default set Thread num = 32 to maximize EU utilization
    constexpr uint32_t thread_num = 32;
    //"Row data need to be in a same work group!"
    assert(matrix_n == wg_tile_n);

    size_t group_range_m = matrix_m / wg_tile_m;
    size_t group_range_n = matrix_n / wg_tile_n;
    constexpr size_t subgroup_range_m = wg_tile_m / sg_tile_m;
    constexpr size_t subgroup_range_n = wg_tile_n / sg_tile_n;

    static_assert(subgroup_range_m * subgroup_range_n == thread_num,
            "Given thread number should equal to pre-set value 32!");
    std::cout << "group_num_x: " << group_range_n
              << ", group_num_y: " << group_range_m
              << ", group_num_z: " << batch_num << "\n";
    std::cout << "group_size_x: " << subgroup_range_n
              << ", group_size_y: " << subgroup_range_m << std::endl;
    cl::sycl::range<3> GroupRange {batch_num, group_range_m, group_range_n};
    cl::sycl::range<3> LocalRange {1, subgroup_range_m, subgroup_range_n};
    cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof("gemm_softmax", ops, "gflops");
    try {
        for (uint32_t i = 0; i < iter + warmup; i++) {
            if (i >= warmup) { prof.cpu_start(); }
            auto gpu_event = Queue.submit([&](handler &cgh) {
                cgh.parallel_for<class
                        Test>(Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                    using namespace gpu::xetla;
                    using namespace gpu::xetla::group;
                    using namespace gpu::xetla::kernel;
                    using namespace gpu::xetla::subgroup;

                    xetla_exec_item<3> ei(item);
                    uint32_t batch_id = ei.get_group(0);
                    // disable sync in brgemm
                    static constexpr uint32_t periodic_sync_interval = 0;
                    static constexpr uint32_t prefetch_distance = 3;
                    using tile_shape = tile_shape_t<wg_tile_n, wg_tile_m,
                            sg_tile_n, sg_tile_m>;

                    using brgemm_t = typename brgemm_selector_t<data_type_a,
                            data_type_b, mem_layout::row_major,
                            mem_layout::col_major, mem_space::global,
                            mem_space::global, 8, 8, float, tile_shape,
                            sg_tile_k, mma_engine::xmx, gpu_arch::Xe,
                            prefetch_distance, periodic_sync_interval>::brgemm;
                    using epilogue_t = epilogue_t<
                            epilogue_policy_tile_op<chained_tile_op_t<>,
                                    result_overwrite, gpu_arch::Xe>,
                            tile_shape,
                            mem_desc_t<data_type_sfx, mem_layout::row_major,
                                    mem_space::local>>;
                    using gemm_op_t
                            = gemm_t<dispatch_policy_default<gpu_arch::Xe>,
                                    brgemm_t, epilogue_t>;

                    // initialize SLM size
                    constexpr uint32_t slm_size
                            = wg_tile_m * wg_tile_n * sizeof(data_type_sfx);
                    xetla_local_init<slm_size>();

                    // initialize named barrier count
                    // we only need to do thread sync while store gemm results to SLM
                    // one barrier is enough for that
                    xetla_nbarrier_init<1>();
                    xetla_nbarrier_t<thread_num, thread_num> nbarrier;
                    nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);

                    // initialize gemm op: gemm result store to shared local memory

                    typename gemm_op_t::arguments_t arg(matrix_m, matrix_k,
                            matrix_n, A + batch_id * size_a, matrix_k,
                            B + batch_id * size_b, matrix_k, 0, matrix_n);
                    gemm_op_t gemm_op;
                    gemm_op(ei, arg);
                    xetla_fence<memory_kind::shared_local>();
                    nbarrier.arrive_wait();

                    // softmax start
                    using softmax_op_t = xetla_softmax_fwd_t<data_type_sfx,
                            data_type_c, tile_shape, mem_space::local,
                            mem_space::global, SIMD, thread_num, softmax_size>;
                    typename softmax_op_t::arguments_t arg1;
                    softmax_op_t softmax_op;

                    arg1.data_in_base = 0;
                    arg1.data_out_ptr = C + batch_id * size_c;

                    softmax_op(ei, &arg1);
                });
            });
            gpu_event.wait();

            if (i >= warmup) {
                prof.cpu_end();
                prof.add_gpu_event(gpu_event);
            }
        }
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    ASSERT_EQ(0,
            gemm_softmax_result_validate(A, B, C, matrix_m, matrix_k, matrix_n,
                    batch_num, mem_layout::row_major, mem_layout::col_major));

    // performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(A, Context);
    free(B, Context);
    free(C, Context);
}

int main() {
    // This example implements batch-GeMM with softmax activation.
    // Softmax needs entire row data for reduced sum and reduced max,
    // So result of batch-GeMM will be written into SLM.
    // When all thread in a work group finishing their job softmax start.
    // To simlify the calculation of softmax, we make each single thread
    // load entire one row data so that there's no data sharing
    // necessity among threads.

    // Description:
    // This kernel can be descripted as following
    // mathematical expression:
    //   C = softmax(A · B.transpose(-1, -2))
    // where:
    //   A, B is the input data
    //   C is the output data
    //   shape(A) = [256, 512, 64]
    //   shape(B) = [256, 512, 64]
    //   shape(C) = [256, 512, 512]

    // To make each single thread load entire one row data
    // we need to reshape the surface:
    //   [1, 512] will be seen as [16, 32] with row major layout
    // After this all operations will be implemented in register.

    gemm_softmax_run(10);
    return 0;
}
