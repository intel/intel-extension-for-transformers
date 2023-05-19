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

#define B 16
#define N 16
#define F 512
#define T 512
#define H 64

template <typename dtype_in, typename dtype_out, typename data_type_acc = float>
int mha_fwd_result_validate(dtype_in *Q_ptr, dtype_in *K_ptr, dtype_in *V_ptr,
        dtype_in *bias_ptr, dtype_out *C_ptr, uint32_t qk_m, uint32_t qk_k,
        uint32_t qk_n, uint32_t sv_m, uint32_t sv_k, uint32_t sv_n,
        uint32_t batch_num, mem_layout mem_layout_qk_a_ = mem_layout::row_major,
        mem_layout mem_layout_qk_b_ = mem_layout::row_major,
        mem_layout mem_layout_sv_a_ = mem_layout::row_major,
        mem_layout mem_layout_sv_b_ = mem_layout::row_major) {
    uint32_t err_cnt = 0;

    uint32_t matrix_size_a = qk_m * qk_k;
    uint32_t matrix_size_b = qk_k * qk_n;
    uint32_t matrix_size_c = qk_m * qk_n;
    std::vector<data_type_acc> tmp_Q(Q_ptr, Q_ptr + batch_num * matrix_size_a);
    std::vector<data_type_acc> tmp_K(K_ptr, K_ptr + batch_num * matrix_size_b);
    std::vector<data_type_acc> tmp_bias(bias_ptr, bias_ptr + B * matrix_size_c);
    std::vector<data_type_acc> gold_C(batch_num * matrix_size_c, 0);
    for (uint32_t batch_id = 0; batch_id < batch_num; batch_id++) {
        get_gemm_gold(qk_m, qk_n, qk_k, mem_layout_qk_a_, mem_layout_qk_b_,
                tmp_Q.data() + batch_id * matrix_size_a,
                tmp_K.data() + batch_id * matrix_size_b,
                gold_C.data() + batch_id * matrix_size_c);
        for (uint32_t i = 0; i < qk_m; i++) {
            for (uint32_t j = 0; j < qk_n; j++) {
                uint32_t res_idx = batch_id * matrix_size_c + i * qk_n + j;
                uint32_t bias_idx = batch_id / N * matrix_size_c + i * qk_n + j;
                gold_C[res_idx] *= 0.125;
                gold_C[res_idx] += tmp_bias[bias_idx];
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
                gold_C[sfx_offset + j] = exp(gold_C[sfx_offset + j] - row_max);
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
    std::vector<data_type_acc> tmp_V(V_ptr, V_ptr + batch_num * matrix_size_b);
    std::vector<data_type_acc> gold_C1(batch_num * matrix_size_c, 0);
    // second gemm on host
    for (uint32_t batch_id = 0; batch_id < batch_num; batch_id++) {
        get_gemm_gold(sv_m, sv_n, sv_k, mem_layout_sv_a_, mem_layout_sv_b_,
                gold_C.data() + batch_id * matrix_size_a,
                tmp_V.data() + batch_id * matrix_size_b,
                gold_C1.data() + batch_id * matrix_size_c);
    }
    // permute 0213
    std::vector<data_type_acc> gold_C2(batch_num * matrix_size_c, 0);
    for (uint32_t batch_id = 0; batch_id < batch_num; ++batch_id) {
        for (uint32_t i = 0; i < sv_m; ++i) {
            for (uint32_t j = 0; j < sv_n; ++j) {
                uint32_t src_id = F * H * batch_id + i * H + j;

                uint32_t h = src_id % H;
                uint32_t f = src_id / H % F;
                uint32_t n = src_id / (F * H) % N;
                uint32_t b = src_id / (F * H * N) % B;

                uint32_t dst_id = b * N * F * H + f * N * H + n * H + h;
                gold_C2[dst_id] = gold_C1[src_id];
            }
        }
    }
    buff_cmp::buff_vals<dtype_out> data(C_ptr, sv_m * batch_num, sv_n, sv_n);
    buff_cmp::buff_vals<dtype_out, data_type_acc> other(
            gold_C1.data(), sv_m * batch_num, sv_n, sv_n);
    bool result = buff_cmp::xetla_buff_cmp(data, other, "mha validation");
    std::cout << ((!result) ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

void mha_fwd_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    using dtype_in = bf16;
    using dtype_out = bf16;
    using dtype_sfx = float;

    constexpr uint32_t batch_num = B * N;
    // arguments for first gemm
    constexpr uint32_t matrix_m_qk = F;
    constexpr uint32_t matrix_n_qk = T;
    constexpr uint32_t matrix_k_qk = H;

    constexpr uint32_t wg_tile_m_qk = 64;
    constexpr uint32_t wg_tile_n_qk = 512;
    constexpr uint32_t sg_tile_m_qk = 32;
    constexpr uint32_t sg_tile_n_qk = 32;
    constexpr uint32_t sg_tile_k_qk = 32;

    // arguments for second gemm
    constexpr uint32_t matrix_m_sv = F;
    constexpr uint32_t matrix_n_sv = H;
    constexpr uint32_t matrix_k_sv = T;

    constexpr uint32_t wg_tile_m_sv = 64;
    constexpr uint32_t wg_tile_n_sv = 64;
    constexpr uint32_t sg_tile_m_sv = 8;
    constexpr uint32_t sg_tile_n_sv = 16;
    constexpr uint32_t sg_tile_k_sv = 32;

    // buffer size of softmax row data
    constexpr uint32_t softmax_sz = F;
    // default set Thread num = 32 to maximize EU utilization
    constexpr uint32_t thread_num = 32;

    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    auto Queue = queue(properties);
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    constexpr uint32_t size_qkv = matrix_m_qk * matrix_k_qk;
    constexpr uint32_t size_bias = matrix_m_qk * matrix_n_qk;
    constexpr uint32_t size_out = matrix_m_sv * matrix_n_sv;

    dtype_in *q = (dtype_in *)static_cast<dtype_in *>(malloc_shared(
            batch_num * size_qkv * sizeof(dtype_in), Device, Context));
    dtype_in *k = (dtype_in *)static_cast<dtype_in *>(malloc_shared(
            batch_num * size_qkv * sizeof(dtype_in), Device, Context));
    dtype_in *v = (dtype_in *)static_cast<dtype_in *>(malloc_shared(
            batch_num * size_qkv * sizeof(dtype_in), Device, Context));
    dtype_in *bias = (dtype_in *)static_cast<dtype_in *>(
            malloc_shared(B * size_bias * sizeof(dtype_in), Device, Context));
    dtype_out *out = (dtype_out *)static_cast<dtype_out *>(malloc_shared(
            batch_num * size_out * sizeof(dtype_out), Device, Context));

    for (uint32_t i = 0; i < batch_num * size_qkv; ++i) {
        q[i] = dtype_in(random_float());
    }
    for (uint32_t i = 0; i < batch_num * size_qkv; ++i) {
        k[i] = dtype_in(random_float());
    }
    for (uint32_t i = 0; i < batch_num * size_qkv; ++i) {
        v[i] = dtype_in(random_float());
    }
    for (uint32_t i = 0; i < B * size_bias; ++i) {
        bias[i] = dtype_in(random_float());
    }
    for (uint32_t i = 0; i < batch_num * size_out; ++i) {
        out[i] = dtype_out(0.f);
    }

    constexpr uint32_t group_range_m = matrix_m_qk / wg_tile_m_qk;
    constexpr uint32_t group_range_n = matrix_n_qk / wg_tile_n_qk;
    constexpr uint32_t subgroup_range_m = wg_tile_m_qk / sg_tile_m_qk;
    constexpr uint32_t subgroup_range_n = wg_tile_n_qk / sg_tile_n_qk;

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

    constexpr uint32_t warmup = 10;
    constexpr long ops = long(4 * B * N * F) * T * H;
    profiling_helper prof("mha", ops, "gflops");
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

                    using tile_shape0 = tile_shape_t<wg_tile_n_qk, wg_tile_m_qk,
                            sg_tile_n_qk, sg_tile_m_qk>;
                    using post_op0_t = scalar_mul_op_t<float, gpu_arch::Xe>;
                    using post_op1_t = elemwise_reduce_op_t<reduce_op::sum,
                            dtype_in, gpu_arch::Xe>;

                    using post_op_t = chained_tile_op_t<post_op0_t, post_op1_t>;

                    using brgemm0_t = typename brgemm_selector_t<dtype_in,
                            dtype_in, mem_layout::row_major,
                            mem_layout::col_major, mem_space::global,
                            mem_space::global, 8, 8, float, tile_shape0,
                            sg_tile_k_qk, mma_engine::xmx, gpu_arch::Xe,
                            prefetch_distance, periodic_sync_interval>::brgemm;
                    using epilogue0_t = epilogue_t<
                            epilogue_policy_tile_op<post_op_t, result_overwrite,
                                    gpu_arch::Xe>,
                            tile_shape0,
                            mem_desc_t<dtype_sfx, mem_layout::row_major,
                                    mem_space::local>>;
                    using gemm_op0_t
                            = gemm_t<dispatch_policy_default<gpu_arch::Xe>,
                                    brgemm0_t, epilogue0_t>;

                    // initialize SLM size
                    constexpr uint32_t slm_size
                            = wg_tile_m_qk * wg_tile_n_qk * sizeof(dtype_sfx);
                    xetla_local_init<slm_size>();

                    // initialize named barrier count
                    // we only need to do thread sync while store gemm results to SLM
                    // one barrier is enough for that
                    xetla_nbarrier_init<1>();
                    xetla_nbarrier_t<thread_num, thread_num> nbarrier;
                    nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);

                    // initialize gemm op: gemm result store to shared local memory
                    typename post_op0_t::arguments_t post_op0_arg(0.125);
                    typename post_op1_t::arguments_t post_op1_arg(
                            bias + batch_id / N * size_bias
                                    + wg_tile_m_qk * wg_tile_n_qk
                                            * ei.get_group(
                                                    1), // bias pre-load ptr batch offset
                            {matrix_n_qk, // bias tdesc width
                                    matrix_m_qk, // bias tdesc height
                                    matrix_n_qk} // bais tdesc pitch
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
                    gemm_op0(ei, arg0);
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

                    softmax_op(ei, &arg1);
                    xetla_fence<memory_kind::shared_local>();
                    nbarrier.arrive_wait();

                    // second gemm: use brgemm to get matAcc for permute storage
                    using tile_shape1 = tile_shape_t<wg_tile_n_sv, wg_tile_m_sv,
                            sg_tile_n_sv, sg_tile_m_sv>;
                    // Using brgemm_selector to get a specific brgemm class
                    using brgemm1_t = typename brgemm_selector_t<dtype_in,
                            dtype_in, mem_layout::row_major,
                            mem_layout::row_major, mem_space::local,
                            mem_space::global, 8, 8, float, tile_shape1,
                            sg_tile_k_sv, mma_engine::xmx, gpu_arch::Xe,
                            prefetch_distance, periodic_sync_interval>::brgemm;
                    // brgemm arguments include matA & matB load information and
                    // cycle number on k-dimension
                    using brgemm_args_t = typename brgemm1_t::arguments_t;
                    using work_group_t = typename brgemm1_t::work_group_t;
                    using mem_desc_a_t = typename brgemm1_t::mem_desc_a_t;
                    using mem_desc_b_t = typename brgemm1_t::mem_desc_b_t;
                    // Using brgemm::matAcc init a matC class for future storage
                    using matAcc_t = typename brgemm1_t::matAcc_t;
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

                    int start_n = ei.get_group(2) * wg_tile_n_sv;
                    int start_m = ei.get_group(1) * wg_tile_m_sv;
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
                    g.init(ei.get_local_linear_id());

                    mem_desc_a_t mem_desc_a;
                    mem_desc_b_t mem_desc_b;
                    mem_desc_a.init(matA_base,
                            {wg_tile_k, wg_tile_m_sv, wg_tile_k}, {0, 0});
                    mem_desc_b.init(matB_ptr, {boundary_n, boundary_k, matB_ld},
                            {start_n, start_k});

                    uint32_t inner_loop_count
                            = (wg_tile_k + sg_tile_k_sv - 1) / sg_tile_k_sv;
                    brgemm_args_t brgemm_args(
                            mem_desc_a, mem_desc_b, inner_loop_count);
                    matAcc_t matAcc;
                    matC_t matC;
                    brgemm1_t brgemm;

                    matAcc.init(0);
                    brgemm(g, matAcc, brgemm_args);
                    // permute store
                    subgroup::elemwise_cvt<matC_t, matAcc_t>(matC, matAcc);
                    xetla_tdescriptor transpose_tdecs;
                    // Define a temprary vector as output buffer
                    xetla_vector<dtype_out, sg_tile_n_sv> out_reg;
                    // Calculate new coordination of each element
                    uint32_t b = ei.get_group(0) / N;
                    uint32_t n = ei.get_group(0) % N;
                    uint32_t f = start_m + brgemm1_t::get_matC_offset_y(g);
                    uint32_t h = start_n + brgemm1_t::get_matC_offset_x(g);

                    // transpose 8 * 16 tile and store to global
                    for (uint32_t j = 0; j < sg_tile_m_sv; ++j, ++f) {
                        uint32_t dst_offset = b * N * F * H + n * F * H + f * H;
                        out_reg = matC.reg.xetla_select<sg_tile_n_sv, 1>(
                                j * sg_tile_n_sv);
                        xetla_fill_tdesc<dtype_out, sg_tile_n_sv, 1, 1>(
                                transpose_tdecs.xetla_format<uint32_t>(),
                                out + dst_offset, H, 1, H, h, 0);
                        xetla_tstore_global<dtype_out, sg_tile_n_sv,
                                cache_hint::write_back, cache_hint::write_back>(
                                transpose_tdecs, out_reg);
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
            mha_fwd_result_validate(q, k, v, bias, out, matrix_m_qk,
                    matrix_k_qk, matrix_n_qk, matrix_m_sv, matrix_k_sv,
                    matrix_n_sv, batch_num, mem_layout::row_major,
                    mem_layout::col_major, mem_layout::row_major,
                    mem_layout::row_major));

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(q, Context);
    free(k, Context);
    free(v, Context);
    free(bias, Context);
    free(out, Context);
}

int main() {
    // This example implements multi-head attention with batch_size: 16,
    // num_heads: 16, sequence_lenth: 512, head_size: 64. It will be shown how to
    // remap the index space of each work-item used for gemm1, softmax and gemm2.

    // Description:
    // Multi-head attention mechanism can be seen as two chained batch MatMul with
    // a softmax in the middle layer. It can be descripted as following
    // mathematical expression:
    //   softmax(Q · (K.transpose(-1, -2)) * (1 / sqr_root(num_heads)) + B) · V
    // where:
    //   Q, K, V: input data
    //   shape(Q) = [16 x 16, 512, 64]
    //   shape(K) = [16 x 16, 512, 64]
    //   shape(V) = [16 x 16, 512, 64]
    //   shape(B) = [16, 512, 512]

    // This kernel is designed to execute the following task:
    // 1: S = (Q · (K.transpose(-1, -2))) * (1 / sqr_root(num_heads)) + B
    // 2: S' = softmax(S)
    // 3: O = S' · V

    mha_fwd_run(10);
    return 0;
}
