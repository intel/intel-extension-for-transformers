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

using namespace gpu::xetla;
using namespace cl::sycl;

#define SIMD 32

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float>
int gemm_softmax_result_validate(data_type_a *A_device, data_type_b *B_device,
        data_type_c *C_device, uint32_t m, uint32_t k, uint32_t n,
        uint32_t batch_num, sycl::queue &queue,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    uint32_t err_cnt = 0;
    bool is_col_major_a = mem_layout_a_ == mem_layout::col_major;
    bool is_col_major_b = mem_layout_b_ == mem_layout::col_major;
    uint32_t size_a = m * k;
    uint32_t size_b = k * n;
    uint32_t size_c = m * n;

    auto A_ptr = alloc_host_and_copy<data_type_a>(
            A_device, batch_num * size_a, queue);
    auto B_ptr = alloc_host_and_copy<data_type_b>(
            B_device, batch_num * size_b, queue);
    auto C_ptr = alloc_host_and_copy<data_type_c>(
            C_device, batch_num * size_c, queue);

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
                gold_C[sfx_offset + j]
                        = std::exp(gold_C[sfx_offset + j] - row_max);
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

    free(A_ptr);
    free(B_ptr);
    free(C_ptr);

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

    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto A = alloc_device_and_init<data_type_a>(
            batch_num * size_a,
            [](data_type_a *data, size_t idx) {
                data[idx] = static_cast<data_type_a>(random_float());
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type_b>(
            batch_num * size_b,
            [](data_type_b *data, size_t idx) {
                data[idx] = static_cast<data_type_b>(random_float());
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type_c>(
            batch_num * size_c,
            [](data_type_c *data, size_t idx) {
                data[idx] = static_cast<data_type_c>(0.0f);
            },
            queue, device, context);

    constexpr uint32_t wg_tile_m = 128;
    constexpr uint32_t wg_tile_n = 512;
    constexpr uint32_t sg_tile_m = 32;
    constexpr uint32_t sg_tile_n = 64;

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
    cl::sycl::range<3> group_range {batch_num, group_range_m, group_range_n};
    cl::sycl::range<3> local_range {1, subgroup_range_m, subgroup_range_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    uint32_t warmup = 10;
    long ops
            = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k * batch_num;
    profiling_helper prof("gemm_softmax", ops, "gflops");
    try {
        for (uint32_t i = 0; i < iter + warmup; i++) {
            if (i >= warmup) { prof.cpu_start(); }
            auto gpu_event = queue.submit([&](handler &cgh) {
                cgh.parallel_for<class
                        Test>(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                    using namespace gpu::xetla;
                    using namespace gpu::xetla::group;
                    using namespace gpu::xetla::kernel;
                    using namespace gpu::xetla::subgroup;

                    uint32_t batch_id = item.get_group(0);

                    // Performance tuning setting based on different shapes
                    static constexpr uint32_t periodic_sync_interval = 8;
                    static constexpr uint32_t prefetch_distance = 3;
                    // should larger than 8
                    static constexpr uint32_t k_iter_num = 16;

                    // Step 1: define mirco-kernel's configuration
                    using wg_shape = shape<wg_tile_n, wg_tile_m>;
                    using sg_shape = shape<sg_tile_n, sg_tile_m>;

                    // Mirco-kernel configuration
                    using tune_option = dict_t<
                            elem_v_t<tune_key::PARAM_OPTIMZER_TYPE,
                                    tune_key_value::
                                            PARAM_OPTIMZER_DECISION_TREE>,
                            elem_t_t<tune_key::SG_TILE_SHAPE, sg_shape>,
                            elem_v_t<tune_key::PREFETCH_DISTANCE,
                                    prefetch_distance>,
                            elem_v_t<tune_key::PERIODIC_SYNC_INTERVAL,
                                    periodic_sync_interval>>;
                    using gemm_op_t = xetla::group::default_gemm_selector_t<
                            data_type_a, // input datatype for A
                            mem_layout::row_major, // memory layout for A
                            8, // leading dimension for A, in unit of element
                            mem_space::
                                    global, // memory reading from global mem for A
                            data_type_b, // input datatype for B
                            mem_layout::col_major, // memory layout for B
                            8, // leading dimension for B, in unit of element
                            mem_space::
                                    global, // memory reading from global mem for B
                            data_type_sfx, // accumulator data type for intermediate resutls
                            wg_shape, // computation tile shape
                            k_iter_num, // elements in each iteration
                            gpu_arch::Xe, // GPU arch
                            tune_option>;

                    using gemm_args_t = gemm_op_t::arguments_t;

                    using epilogue_t = xetla::group::default_epilogue_selector_t<
                            data_type_c, // onput datatype for C
                            mem_layout::row_major, // memory layout for C
                            8, // leading dimension for C, in unit of element
                            mem_space::
                                    global, // memory writing to global mem for C
                            wg_shape, // computation tile shape
                            k_iter_num, // elements in each iteration
                            gpu_arch::Xe, // GPU arch
                            tune_option>;

                    // using experimental::group::softmax
                    // define softmax forward op
                    using tile_shape = typename gemm_op_t::tile_shape;
                    using softmax_fwd_t = softmax_t<
                            softmax_policy_fwd<data_type_sfx, gpu_arch::Xe>,
                            tile_shape>;
                    using softmax_fwd_args_t =
                            typename softmax_fwd_t::arguments_t;

                    // initialize shared local memory and named barrier
                    static constexpr uint32_t barrier_count
                            = gemm_op_t::barrier_count
                            + softmax_fwd_t::get_barrier_count::count;
                    static constexpr uint32_t slm_size = gemm_op_t::slm_size
                            + softmax_fwd_t::get_slm_size::size;
                    xetla_nbarrier_init<barrier_count>();
                    xetla_local_init<slm_size>();

                    // matA & matB & matC base address and load width
                    data_type_a *matA_ptr = A + batch_id * size_a;
                    uint32_t matA_ld = matrix_k;
                    data_type_b *matB_ptr = B + batch_id * size_b;
                    uint32_t matB_ld = matrix_k;
                    data_type_c *matC_ptr = C + batch_id * size_c;
                    uint32_t matC_ld = matrix_n;

                    // ecah workgroup gets it individual index to start computation
                    int start_n = item.get_group(2) * wg_tile_n;
                    int start_m = item.get_group(1) * wg_tile_m;
                    int start_k = 0;
                    uint32_t wg_tile_k = matrix_k;
                    uint32_t boundary_n = (start_n + wg_tile_n) > matrix_n
                            ? matrix_n
                            : (start_n + wg_tile_n);
                    uint32_t boundary_m = (start_m + wg_tile_m) > matrix_m
                            ? matrix_m
                            : (start_m + wg_tile_m);
                    uint32_t boundary_k = wg_tile_k;
                    uint32_t inner_loop_count
                            = (wg_tile_k + k_iter_num - 1) / k_iter_num;

                    // initialize the memory description of matA & matB & matC
                    using mem_desc_a_t = typename gemm_op_t::mem_desc_a_t;
                    using mem_desc_b_t = typename gemm_op_t::mem_desc_b_t;
                    using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;
                    mem_desc_a_t mem_desc_a(matA_ptr,
                            {boundary_k, boundary_m, matA_ld},
                            {start_k, start_m});
                    mem_desc_b_t mem_desc_b(matB_ptr,
                            {boundary_n, boundary_k, matB_ld},
                            {start_n, start_k});
                    mem_desc_c_t mem_desc_c(matC_ptr,
                            {boundary_n, boundary_m, matC_ld},
                            {start_n, start_m});

                    // call gemm function and result will be written in matAcc
                    using gemm_args_t = typename gemm_op_t::arguments_t;
                    gemm_args_t gemm_args(
                            mem_desc_a, mem_desc_b, inner_loop_count);
                    gemm_op_t::work_group_t g(item.get_local_linear_id());
                    gemm_op_t::matAcc_t matAcc(0);
                    gemm_op_t gemm;
                    gemm(g, matAcc, gemm_args);

                    // for each matAcc, call softmax op
                    softmax_fwd_t softmax_fwd;
                    softmax_fwd_args_t softmax_fwd_args(1);
                    softmax_fwd(g, matAcc, {}, softmax_fwd_args);

                    // write matAcc value into pointer C
                    epilogue_t epilogue;
                    epilogue(g, matAcc, mem_desc_c);
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
                    batch_num, queue, mem_layout::row_major,
                    mem_layout::col_major));

    // performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
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
