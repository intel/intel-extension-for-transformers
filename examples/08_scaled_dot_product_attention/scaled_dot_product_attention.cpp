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

using namespace gpu::xetla;
using namespace cl::sycl;

#define SIMD 32

#define batch_num 16
#define head_num 16
#define sequence_len 512
#define head_size 64

template <typename dtype_in, typename dtype_out, typename data_type_acc = float>
int sdp_fwd_result_validate(dtype_in *q_device, dtype_in *k_device,
        dtype_in *v_device, dtype_in *mask_device, dtype_out *c_device,
        uint32_t qk_m, uint32_t qk_k, uint32_t qk_n, uint32_t sv_m,
        uint32_t sv_k, uint32_t sv_n, uint32_t batch_cnt, sycl::queue &queue,
        mem_layout mem_layout_qk_a_ = mem_layout::row_major,
        mem_layout mem_layout_qk_b_ = mem_layout::row_major,
        mem_layout mem_layout_sv_a_ = mem_layout::row_major,
        mem_layout mem_layout_sv_b_ = mem_layout::row_major) {
    uint32_t err_cnt = 0;

    uint32_t matrix_size_a = qk_m * qk_k;
    uint32_t matrix_size_b = qk_k * qk_n;
    uint32_t matrix_size_c = qk_m * qk_n;

    auto Q_ptr = alloc_host_and_copy<dtype_in>(
            q_device, batch_cnt * matrix_size_a, queue);
    auto K_ptr = alloc_host_and_copy<dtype_in>(
            k_device, batch_cnt * matrix_size_a, queue);
    auto V_ptr = alloc_host_and_copy<dtype_in>(
            v_device, batch_cnt * matrix_size_a, queue);
    auto mask_ptr = alloc_host_and_copy<dtype_in>(
            mask_device, batch_num * matrix_size_c, queue);
    auto C_ptr = alloc_host_and_copy<dtype_out>(
            c_device, batch_cnt * sv_m * sv_n, queue);

    std::vector<data_type_acc> tmp_Q(Q_ptr, Q_ptr + batch_cnt * matrix_size_a);
    std::vector<data_type_acc> tmp_K(K_ptr, K_ptr + batch_cnt * matrix_size_b);
    std::vector<data_type_acc> tmp_mask(
            mask_ptr, mask_ptr + batch_num * matrix_size_c);
    std::vector<data_type_acc> gold_C(batch_cnt * matrix_size_c, 0);
    for (uint32_t batch_id = 0; batch_id < batch_cnt; batch_id++) {
        get_gemm_gold(qk_m, qk_n, qk_k, mem_layout_qk_a_, mem_layout_qk_b_,
                tmp_Q.data() + batch_id * matrix_size_a,
                tmp_K.data() + batch_id * matrix_size_b,
                gold_C.data() + batch_id * matrix_size_c);
        for (uint32_t i = 0; i < qk_m; i++) {
            for (uint32_t j = 0; j < qk_n; j++) {
                uint32_t res_idx = batch_id * matrix_size_c + i * qk_n + j;
                uint32_t mask_idx
                        = batch_id / head_num * matrix_size_c + i * qk_n + j;
                gold_C[res_idx] *= 0.125;
                gold_C[res_idx] += tmp_mask[mask_idx];
            }
        }
        for (uint32_t i = 0; i < qk_m; i++) {
            data_type_acc row_max = 0;
            data_type_acc exp_sum = 0;
            uint32_t sfx_offset = batch_id * matrix_size_c + i * qk_n;
            for (uint32_t j = 0; j < qk_n; j++) {
                row_max = max(row_max, gold_C[sfx_offset + j]);
            }
            for (uint32_t j = 0; j < qk_n; j++) {
                gold_C[sfx_offset + j]
                        = std::exp(gold_C[sfx_offset + j] - row_max);
                exp_sum += gold_C[sfx_offset + j];
            }
            for (uint32_t j = 0; j < qk_n; j++) {
                gold_C[sfx_offset + j] /= exp_sum;
            }
        }
    }
    matrix_size_a = sv_m * sv_k;
    matrix_size_b = sv_k * sv_n;
    matrix_size_c = sv_m * sv_n;
    std::vector<data_type_acc> tmp_V(V_ptr, V_ptr + batch_cnt * matrix_size_b);
    std::vector<data_type_acc> gold_C1(batch_cnt * matrix_size_c, 0);
    // second gemm on host
    for (uint32_t batch_id = 0; batch_id < batch_cnt; batch_id++) {
        get_gemm_gold(sv_m, sv_n, sv_k, mem_layout_sv_a_, mem_layout_sv_b_,
                gold_C.data() + batch_id * matrix_size_a,
                tmp_V.data() + batch_id * matrix_size_b,
                gold_C1.data() + batch_id * matrix_size_c);
    }
    // permute 0213
    std::vector<data_type_acc> gold_C2(batch_cnt * matrix_size_c, 0);
    for (uint32_t batch_id = 0; batch_id < batch_cnt; ++batch_id) {
        for (uint32_t i = 0; i < sv_m; ++i) {
            for (uint32_t j = 0; j < sv_n; ++j) {
                uint32_t src_id = sequence_len * head_size * batch_id
                        + i * head_size + j;

                uint32_t h = src_id % head_size;
                uint32_t f = src_id / head_size % sequence_len;
                uint32_t n = src_id / (sequence_len * head_size) % head_num;
                uint32_t b = src_id / (sequence_len * head_size * head_num)
                        % batch_num;

                uint32_t dst_id = b * head_num * sequence_len * head_size
                        + f * head_num * head_size + n * head_size + h;
                gold_C2[dst_id] = gold_C1[src_id];
            }
        }
    }
    buff_cmp::buff_vals<dtype_out> data(C_ptr, sv_m * batch_cnt, sv_n, sv_n);
    buff_cmp::buff_vals<dtype_out, data_type_acc> other(
            gold_C2.data(), sv_m * batch_cnt, sv_n, sv_n);
    bool result = buff_cmp::xetla_buff_cmp(data, other, "sdp validation");

    free(Q_ptr);
    free(K_ptr);
    free(V_ptr);
    free(mask_ptr);
    free(C_ptr);

    std::cout << ((!result) ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

void sdp_fwd_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    using dtype_in = bf16;
    using dtype_out = bf16;
    using dtype_sfx = float;

    constexpr uint32_t batch_cnt = batch_num * head_num;
    // arguments for first gemm
    constexpr uint32_t matrix_m_qk = sequence_len;
    constexpr uint32_t matrix_n_qk = sequence_len;
    constexpr uint32_t matrix_k_qk = head_size;

    constexpr uint32_t wg_tile_m_qk = 64;
    constexpr uint32_t wg_tile_n_qk = 512;
    constexpr uint32_t sg_tile_m_qk = 32;
    constexpr uint32_t sg_tile_n_qk = 32;
    constexpr uint32_t wg_tile_k_qk = 32;

    // arguments for second gemm
    constexpr uint32_t matrix_m_sv = sequence_len;
    constexpr uint32_t matrix_n_sv = head_size;
    constexpr uint32_t matrix_k_sv = sequence_len;

    constexpr uint32_t wg_tile_m_sv = 64;
    constexpr uint32_t wg_tile_n_sv = 64;
    constexpr uint32_t sg_tile_m_sv = 8;
    constexpr uint32_t sg_tile_n_sv = 16;
    constexpr uint32_t wg_tile_k_sv = 32;

    // buffer size of softmax row data
    constexpr uint32_t softmax_sz = sequence_len;
    // default set Thread num = 32 to maximize EU utilization
    constexpr uint32_t thread_num = 32;

    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    constexpr uint32_t size_qkv = matrix_m_qk * matrix_k_qk;
    constexpr uint32_t size_mask = matrix_m_qk * matrix_n_qk;
    constexpr uint32_t size_out = matrix_m_sv * matrix_n_sv;

    auto q = alloc_device_and_init<dtype_in>(
            batch_cnt * size_qkv,
            [](dtype_in *data, size_t idx) {
                data[idx] = static_cast<dtype_in>(random_float());
            },
            queue, device, context);
    auto k = alloc_device_and_init<dtype_in>(
            batch_cnt * size_qkv,
            [](dtype_in *data, size_t idx) {
                data[idx] = static_cast<dtype_in>(random_float());
            },
            queue, device, context);
    auto v = alloc_device_and_init<dtype_in>(
            batch_cnt * size_qkv,
            [](dtype_in *data, size_t idx) {
                data[idx] = static_cast<dtype_in>(random_float());
            },
            queue, device, context);
    auto attn_mask = alloc_device_and_init<dtype_in>(
            batch_num * size_mask,
            [](dtype_in *data, size_t idx) {
                data[idx] = static_cast<dtype_in>(random_float());
            },
            queue, device, context);
    auto out = alloc_device_and_init<dtype_out>(
            batch_cnt * size_out,
            [](dtype_out *data, size_t idx) {
                data[idx] = static_cast<dtype_out>(0.0f);
            },
            queue, device, context);

    constexpr uint32_t group_range_m = matrix_m_qk / wg_tile_m_qk;
    constexpr uint32_t group_range_n = matrix_n_qk / wg_tile_n_qk;
    constexpr uint32_t subgroup_range_m = wg_tile_m_qk / sg_tile_m_qk;
    constexpr uint32_t subgroup_range_n = wg_tile_n_qk / sg_tile_n_qk;

    static_assert(subgroup_range_m * subgroup_range_n == thread_num,
            "Given thread number should equal to pre-set value 32!");
    std::cout << "group_num_x: " << group_range_n
              << ", group_num_y: " << group_range_m
              << ", group_num_z: " << batch_cnt << "\n";
    std::cout << "group_size_x: " << subgroup_range_n
              << ", group_size_y: " << subgroup_range_m << std::endl;
    cl::sycl::range<3> group_range {batch_cnt, group_range_m, group_range_n};
    cl::sycl::range<3> local_range {1, subgroup_range_m, subgroup_range_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    constexpr uint32_t warmup = 10;
    long ops = long(4 * batch_num * head_num * sequence_len) * sequence_len
            * head_size;
    profiling_helper prof("sdp", ops, "gflops");
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
                    // disable sync in gemm
                    static constexpr uint32_t periodic_sync_interval = 0;
                    static constexpr uint32_t prefetch_distance = 3;

                    using wg_shape0 = shape<wg_tile_n_qk, wg_tile_m_qk>;
                    using sg_shape0 = shape<sg_tile_n_qk, sg_tile_m_qk>;

                    using post_op0_t = scalar_mul_op_t<float, gpu_arch::Xe>;
                    using post_op1_t = elemwise_reduce_op_t<reduce_op::sum,
                            dtype_in, gpu_arch::Xe>;
                    using post_op_t = chained_tile_op_t<post_op0_t, post_op1_t>;
                    using epilogue_policy0
                            = xetla::group::epilogue_policy_tile_op<post_op_t,
                                    gpu_arch::Xe>;
                    using group_swizzle = group_swizzle_default<gpu_arch::Xe>;

                    using tune_option0 = dict_t<
                            elem_v_t<tune_key::PARAM_OPTIMZER_TYPE,
                                    tune_key_value::
                                            PARAM_OPTIMZER_DECISION_TREE>,
                            elem_t_t<tune_key::EPILOGUE_POLICY,
                                    epilogue_policy0>,
                            elem_t_t<tune_key::SG_TILE_SHAPE, sg_shape0>,
                            elem_v_t<tune_key::PREFETCH_DISTANCE,
                                    prefetch_distance>,
                            elem_v_t<tune_key::PERIODIC_SYNC_INTERVAL,
                                    periodic_sync_interval>>;
                    using gemm0_t = xetla::group::default_gemm_selector_t<
                            dtype_in, // input datatype for A
                            mem_layout::row_major, // memory layout for A
                            8, // leading dimension for A, in unit of element
                            mem_space::
                                    global, // memory reading from global mem for A
                            dtype_in, // input datatype for B
                            mem_layout::row_major, // memory layout for B
                            8, // leading dimension for B, in unit of element
                            mem_space::
                                    global, // memory reading from global mem for B
                            float, // accumulator data type for intermediate resutls
                            wg_shape0, // computation tile shape
                            wg_tile_k_qk, // elements in each iteration
                            gpu_arch::Xe, // GPU arch
                            tune_option0>;
                    using epilogue0_t = xetla::group::default_epilogue_selector_t<
                            dtype_sfx, // onput datatype for C
                            mem_layout::row_major, // memory layout for C
                            8, // leading dimension for C, in unit of element
                            mem_space::
                                    local, // memory writing to local mem for C
                            wg_shape0, // computation tile shape
                            wg_tile_k_qk, // elements in each iteration
                            gpu_arch::Xe, // GPU arch
                            tune_option0>;
                    using gemm_op0_t = gemm_universal_t<
                            dispatch_policy_default<group_swizzle>, gemm0_t,
                            epilogue0_t>;

                    using tile_shape0 = typename gemm0_t::tile_shape;

                    // initialize SLM size
                    constexpr uint32_t slm_size
                            = wg_tile_m_qk * wg_tile_n_qk * sizeof(dtype_sfx);
                    xetla_local_init<slm_size>();

                    // initialize named barrier count
                    // we only need to do thread sync while store gemm results to SLM
                    // one barrier is enough for that
                    xetla_nbarrier_init<1>();
                    xetla_nbarrier_t<thread_num, thread_num, gpu_arch::Xe>
                            nbarrier;
                    nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);

                    // initialize gemm op: gemm result store to shared local memory
                    typename post_op0_t::arguments_t post_op0_arg(0.125);
                    typename post_op1_t::arguments_t post_op1_arg(
                            attn_mask + batch_id / head_num * size_mask
                                    + wg_tile_m_qk * wg_tile_n_qk
                                            * item.get_group(
                                                    1), // attn_mask pre-load ptr batch offset
                            {matrix_n_qk, // attn_mask tdesc width
                                    matrix_m_qk, // attn_mask tdesc height
                                    matrix_n_qk} // attn_mask tdesc pitch
                    );
                    typename gemm_op0_t::arguments_t arg0(matrix_m_qk,
                            matrix_k_qk, matrix_n_qk,
                            q + batch_id * size_qkv, // matA_ptr + batch offset
                            matrix_k_qk, // matA load width
                            k + batch_id * size_qkv, // matB_ptr + batch offset
                            matrix_k_qk, // matB load width
                            0, // matC_base
                            matrix_n_qk, // matC load width
                            {{post_op0_arg, post_op1_arg}});
                    gemm_op0_t gemm_op0;
                    gemm_op0(item, arg0);
                    xetla_fence<memory_kind::shared_local>();
                    nbarrier.arrive_wait();

                    // softmax start: result store to SLM
                    using softmax_op_t = xetla_softmax_fwd_t<dtype_sfx,
                            dtype_in, tile_shape0, mem_space::local,
                            mem_space::local, SIMD, thread_num, softmax_sz>;
                    typename softmax_op_t::arguments_t arg1;
                    softmax_op_t softmax_op;

                    arg1.data_in_base = 0;
                    arg1.data_out_base = 0;

                    softmax_op(item, &arg1);
                    xetla_fence<memory_kind::shared_local>();
                    nbarrier.arrive_wait();

                    // second gemm: use gemm to get matAcc for permute storage
                    using wg_shape1 = shape<wg_tile_n_sv, wg_tile_m_sv>;
                    using sg_shape1 = shape<sg_tile_n_sv, sg_tile_m_sv>;

                    using tune_option1 = dict_t<
                            elem_v_t<tune_key::PARAM_OPTIMZER_TYPE,
                                    tune_key_value::
                                            PARAM_OPTIMZER_DECISION_TREE>,
                            elem_t_t<tune_key::SG_TILE_SHAPE, sg_shape1>,
                            elem_v_t<tune_key::PREFETCH_DISTANCE,
                                    prefetch_distance>,
                            elem_v_t<tune_key::PERIODIC_SYNC_INTERVAL,
                                    periodic_sync_interval>>;
                    // Using gemm_selector to get a specific gemm class
                    using gemm1_t = xetla::group::default_gemm_selector_t<
                            dtype_in, // input datatype for A
                            mem_layout::row_major, // memory layout for A
                            8, // leading dimension for A, in unit of element
                            mem_space::
                                    local, // memory reading from local mem for A
                            dtype_in, // input datatype for B
                            mem_layout::row_major, // memory layout for B
                            8, // leading dimension for B, in unit of element
                            mem_space::
                                    global, // memory reading from global mem for B
                            float, // accumulator data type for intermediate resutls
                            wg_shape1, // computation tile shape
                            wg_tile_k_sv, // elements in each iteration
                            gpu_arch::Xe, // GPU arch
                            tune_option1>;

                    using tile_shape1 = typename gemm1_t::tile_shape;

                    // gemm arguments include matA & matB load information and
                    // cycle number on k-dimension
                    using gemm_args_t = typename gemm1_t::arguments_t;
                    using work_group_t = typename gemm1_t::work_group_t;
                    using mem_desc_a_t = typename gemm1_t::mem_desc_a_t;
                    using mem_desc_b_t = typename gemm1_t::mem_desc_b_t;
                    // Using gemm::matAcc init a matC class for future storage
                    using matAcc_t = typename gemm1_t::matAcc_t;
                    using matC_t = tile_t<dtype_out,
                            tile_desc_t<matAcc_t::tile_size_x,
                                    matAcc_t::tile_size_y,
                                    matAcc_t::block_size_x,
                                    matAcc_t::block_size_y, reg_layout::tiled>>;
                    // Following six variables is a conterpart of gemm::arguments
                    // Reuse this three variables for new gemm
                    uint32_t matrix_m = matrix_m_sv;
                    uint32_t matrix_k = matrix_k_sv;
                    uint32_t matrix_n = matrix_n_sv;
                    // matA & matB base address and load width
                    uint32_t matA_base = 0; // matA_base
                    uint32_t matA_ld = matrix_k_sv; // matA load width
                    dtype_in *matB_ptr = v
                            + batch_id * size_qkv; // matB_ptr + batch offset
                    uint32_t matB_ld = matrix_n_sv; // matB load width

                    int start_n = item.get_group(2) * wg_tile_n_sv;
                    int start_m = item.get_group(1) * wg_tile_m_sv;
                    int start_k = 0;
                    uint32_t wg_tile_k = matrix_k;
                    uint32_t boundary_n = (start_n + wg_tile_n_sv) > matrix_n
                            ? matrix_n
                            : (start_n + wg_tile_n_sv);
                    uint32_t boundary_m = (start_m + wg_tile_m_sv) > matrix_m
                            ? matrix_m
                            : (start_m + wg_tile_m_sv);
                    uint32_t boundary_k = wg_tile_k;

                    work_group_t g;
                    g.init(item.get_local_linear_id());

                    mem_desc_a_t mem_desc_a;
                    mem_desc_b_t mem_desc_b;
                    mem_desc_a.init(matA_base,
                            {wg_tile_k, wg_tile_m_sv, wg_tile_k}, {0, 0});
                    mem_desc_b.init(matB_ptr, {boundary_n, boundary_k, matB_ld},
                            {start_n, start_k});

                    uint32_t inner_loop_count
                            = (wg_tile_k + wg_tile_k_sv - 1) / wg_tile_k_sv;
                    gemm_args_t gemm_args(
                            mem_desc_a, mem_desc_b, inner_loop_count);
                    matAcc_t matAcc;
                    matC_t matC;
                    gemm1_t gemm;

                    matAcc.init(0);
                    gemm(g, matAcc, gemm_args);
                    // permute store
                    subgroup::elemwise_cvt<matC_t, matAcc_t>(matC, matAcc);
                    xetla_tdescriptor transpose_tdecs;
                    // Define a temprary vector as output buffer
                    xetla_vector<dtype_out, sg_tile_n_sv> out_reg;
                    // Calculate new coordination of each element
                    uint32_t b = item.get_group(0) / head_num;
                    uint32_t n = item.get_group(0) % head_num;
                    uint32_t f = start_m + gemm1_t::get_matC_offset_y(g);
                    uint32_t h = start_n + gemm1_t::get_matC_offset_x(g);

                    // transpose 8 * 16 tile and store to global
                    for (uint32_t j = 0; j < sg_tile_m_sv; ++j, ++f) {
                        uint32_t dst_offset
                                = b * head_num * sequence_len * head_size
                                + f * head_num * head_size + n * head_size;
                        out_reg = matC.reg.xetla_select<sg_tile_n_sv, 1>(
                                j * sg_tile_n_sv);
                        xetla_fill_tdesc<dtype_out, sg_tile_n_sv, 1, 1>(
                                transpose_tdecs.xetla_format<uint32_t>(),
                                out + dst_offset, head_size, 1, head_size, h,
                                0);
                        xetla_tstore_global<dtype_out, sg_tile_n_sv,
                                cache_hint::write_back, cache_hint::write_back,
                                gpu_arch::Xe>(transpose_tdecs, out_reg);
                    }
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
            sdp_fwd_result_validate(q, k, v, attn_mask, out, matrix_m_qk,
                    matrix_k_qk, matrix_n_qk, matrix_m_sv, matrix_k_sv,
                    matrix_n_sv, batch_cnt, queue, mem_layout::row_major,
                    mem_layout::col_major, mem_layout::row_major,
                    mem_layout::row_major));

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(q, context);
    free(k, context);
    free(v, context);
    free(attn_mask, context);
    free(out, context);
}

int main() {
    // This example implements scaled-dot-production with batch_size: 16,
    // num_heads: 16, sequence_lenth: 512, head_size: 64. It will be shown how to
    // remap the index space of each work-item used for gemm1, softmax and gemm2.

    // Description:
    // Scaled-dot-production mechanism can be seen as two chained batch MatMul with
    // a softmax in the middle layer. It can be descripted as following
    // mathematical expression:
    //   softmax(Q · (K.transpose(-1, -2)) * (1 / sqr_root(num_heads)) + attn_mask) · V
    // where:
    //   Q, K, V: input data
    //   shape(Q) = [16 x 16, 512, 64]
    //   shape(K) = [16 x 16, 512, 64]
    //   shape(V) = [16 x 16, 512, 64]
    //   shape(attn_mask) = [16, 512, 512]

    // This kernel is designed to execute the following task:
    // 1: S = (Q · (K.transpose(-1, -2))) * (1 / sqr_root(num_heads)) + attn_mask
    // 2: S' = softmax(S)
    // 3: O = S' · V

    sdp_fwd_run(10);
    return 0;
}
