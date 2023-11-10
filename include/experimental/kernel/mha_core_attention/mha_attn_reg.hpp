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

#pragma once

#include "common/common.hpp"
#include "group/group.hpp"
#include "subgroup/subgroup.hpp"

namespace gpu::xetla::kernel {

#define list_width 16
#define rand_threshold_const 0x80000000
#define SIGN_BIT_DW 0x80000000
#define SIGN_BIT_W16 0x8000
#define SIGN_BIT_B8 0x80

template <typename dtype_bin_, typename dtype_bot_, typename dtype_sfx_,
        typename dtype_acc_, int HWThreadNum, bool Dopt_RandGenflag = true,
        uint16_t RandSIMD = 16, int Max_SeqLen = 2048>
struct xetla_mha_attn_reg_fwd_t {
    using dtype_bin = dtype_bin_;
    using dtype_bot = dtype_bot_;
    using dtype_sfx = dtype_sfx_;
    using dtype_acc = dtype_acc_;

    static constexpr int ThreadNum = HWThreadNum;
    static constexpr int max_seqlen = Max_SeqLen;
    static constexpr mem_space mem_space_a = mem_space::global;
    static constexpr mem_space mem_space_b = mem_space::global;
    static constexpr mem_space mem_space_c = mem_space::global;
    static constexpr uint16_t Rand_SIMD = RandSIMD;

    static constexpr mem_layout mem_layout_a = mem_layout::row_major;
    static constexpr mem_layout mem_layout_QKT_b = mem_layout::col_major;
    static constexpr mem_layout mem_layout_out_b = mem_layout::row_major;
    static constexpr mem_layout mem_layout_c = mem_layout::row_major;

    static constexpr mem_space gemm_mem_space_a = mem_space_a;
    static constexpr mem_layout gemm_mem_layout_a = mem_layout_a;

    static constexpr mem_space gemm_mem_space_b = mem_space_b;
    static constexpr mem_layout gemm_mem_layout_QKT_b = mem_layout_QKT_b;
    static constexpr mem_layout gemm_mem_layout_out_b = mem_layout_out_b;

    static constexpr uint32_t periodic_sync_interval = 0;
    static constexpr uint32_t prefetch_distance = 3;
    static constexpr uint32_t k_stride
            = 32 / sizeof(dtype_bin); //gemm_t::k_stride;
    using bgm_perf_tuning_knob = group::perf_tuning_knob_t<k_stride,
            prefetch_distance, periodic_sync_interval>;

    using tile_attr_128x128 = group::tile_shape_t<128, 128, 32, 16>;
    using tile_attr_128x256 = group::tile_shape_t<256, 128, 64, 16>;
    using tile_attr_64x384 = group::tile_shape_t<384, 64, 48, 16>;
    using tile_attr_64x512 = group::tile_shape_t<512, 64, 64, 16>;
    using tile_attr_32x1024 = group::tile_shape_t<1024, 32, 64, 16>;
    using tile_attr_16x2048 = group::tile_shape_t<2048, 16, 64, 16>;
    using tile_attr_128x64 = group::tile_shape_t<64, 128, 16, 16>;

    using mem_desc_a_QKT
            = mem_desc_t<dtype_bin, gemm_mem_layout_a, gemm_mem_space_a>;
    using mem_desc_b_QKT
            = mem_desc_t<dtype_bin, gemm_mem_layout_QKT_b, gemm_mem_space_b>;
    using compute_policy_QKT = group::compute_policy_default_xmx<
            group::compute_attr_t<dtype_bin, dtype_bin, dtype_acc>,
            bgm_perf_tuning_knob, gpu_arch::Xe>;

    using mem_desc_a_out
            = mem_desc_t<dtype_sfx, gemm_mem_layout_a, gemm_mem_space_a>;
    using mem_desc_b_out
            = mem_desc_t<dtype_bin, gemm_mem_layout_out_b, gemm_mem_space_b>;
    using compute_policy_out = group::compute_policy_default_xmx<
            group::compute_attr_t<dtype_sfx, dtype_bin, dtype_acc>,
            bgm_perf_tuning_knob, gpu_arch::Xe>;

    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint16_t sfx_type_size = sizeof(dtype_sfx);
    static_assert((sfx_type_size == 1) || (sfx_type_size == 2)
            || (sfx_type_size == 4));

    using work_group_t = work_group_t<ThreadNum>;

    using pre_processing_128x128
            = group::pre_processing_default_t<tile_attr_128x128, gpu_arch::Xe>;
    using pre_processing_128x256
            = group::pre_processing_default_t<tile_attr_128x256, gpu_arch::Xe>;
    using pre_processing_64x384
            = group::pre_processing_default_t<tile_attr_64x384, gpu_arch::Xe>;
    using pre_processing_64x512
            = group::pre_processing_default_t<tile_attr_64x512, gpu_arch::Xe>;
    using pre_processing_32x1024
            = group::pre_processing_default_t<tile_attr_32x1024, gpu_arch::Xe>;
    using pre_processing_16x2048
            = group::pre_processing_default_t<tile_attr_16x2048, gpu_arch::Xe>;
    using pre_processing_128x64
            = group::pre_processing_matA_neg_filter_t<tile_attr_128x64,
                    gpu_arch::Xe>;

    using gemm_op_128x128_t
            = group::gemm_t<compute_policy_QKT, tile_attr_128x128,
                    mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_128x128>;
    using gemm_op_128x256_t
            = group::gemm_t<compute_policy_QKT, tile_attr_128x256,
                    mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_128x256>;
    using gemm_op_64x384_t = group::gemm_t<compute_policy_QKT, tile_attr_64x384,
            mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_64x384>;
    using gemm_op_64x512_t = group::gemm_t<compute_policy_QKT, tile_attr_64x512,
            mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_64x512>;
    using gemm_op_32x1024_t
            = group::gemm_t<compute_policy_QKT, tile_attr_32x1024,
                    mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_32x1024>;
    using gemm_op_16x2048_t
            = group::gemm_t<compute_policy_QKT, tile_attr_16x2048,
                    mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_16x2048>;
    using gemm_op_128x64_t = group::gemm_t<compute_policy_out, tile_attr_128x64,
            mem_desc_a_out, mem_desc_b_out, pre_processing_128x64>;

    using gemm_arguments_128x128 = typename gemm_op_128x128_t::arguments_t;
    using gemm_arguments_128x256 = typename gemm_op_128x256_t::arguments_t;
    using gemm_arguments_64x384 = typename gemm_op_64x384_t::arguments_t;
    using gemm_arguments_64x512 = typename gemm_op_64x512_t::arguments_t;
    using gemm_arguments_32x1024 = typename gemm_op_32x1024_t::arguments_t;
    using gemm_arguments_16x2048 = typename gemm_op_16x2048_t::arguments_t;
    using gemm_arguments_128x64 = typename gemm_op_128x64_t::arguments_t;

    using matAcc_128x128_t = typename gemm_op_128x128_t::matAcc_t;
    using matAcc_128x256_t = typename gemm_op_128x256_t::matAcc_t;
    using matAcc_64x384_t = typename gemm_op_64x384_t::matAcc_t;
    using matAcc_64x512_t = typename gemm_op_64x512_t::matAcc_t;
    using matAcc_32x1024_t = typename gemm_op_32x1024_t::matAcc_t;
    using matAcc_16x2048_t = typename gemm_op_16x2048_t::matAcc_t;
    using matAcc_128x64_t = typename gemm_op_128x64_t::matAcc_t;

    using mat_128x128_tile_desc_t
            = subgroup::tile_desc_t<matAcc_128x128_t::tile_desc::tile_size_x,
                    matAcc_128x128_t::tile_desc::tile_size_y,
                    matAcc_128x128_t::tile_desc::block_size_x,
                    matAcc_128x128_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using mat_128x256_tile_desc_t
            = subgroup::tile_desc_t<matAcc_128x256_t::tile_desc::tile_size_x,
                    matAcc_128x256_t::tile_desc::tile_size_y,
                    matAcc_128x256_t::tile_desc::block_size_x,
                    matAcc_128x256_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using mat_64x384_tile_desc_t
            = subgroup::tile_desc_t<matAcc_64x384_t::tile_desc::tile_size_x,
                    matAcc_64x384_t::tile_desc::tile_size_y,
                    matAcc_64x384_t::tile_desc::block_size_x,
                    matAcc_64x384_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using mat_64x512_tile_desc_t
            = subgroup::tile_desc_t<matAcc_64x512_t::tile_desc::tile_size_x,
                    matAcc_64x512_t::tile_desc::tile_size_y,
                    matAcc_64x512_t::tile_desc::block_size_x,
                    matAcc_64x512_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using mat_32x1024_tile_desc_t
            = subgroup::tile_desc_t<matAcc_32x1024_t::tile_desc::tile_size_x,
                    matAcc_32x1024_t::tile_desc::tile_size_y,
                    matAcc_32x1024_t::tile_desc::block_size_x,
                    matAcc_32x1024_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using mat_16x2048_tile_desc_t
            = subgroup::tile_desc_t<matAcc_16x2048_t::tile_desc::tile_size_x,
                    matAcc_16x2048_t::tile_desc::tile_size_y,
                    matAcc_16x2048_t::tile_desc::block_size_x,
                    matAcc_16x2048_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using mat_128x64_tile_desc_t
            = subgroup::tile_desc_t<matAcc_128x64_t::tile_desc::tile_size_x,
                    matAcc_128x64_t::tile_desc::tile_size_y,
                    matAcc_128x64_t::tile_desc::block_size_x,
                    matAcc_128x64_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using matC_128x128_t = subgroup::tile_t<dtype_sfx, mat_128x128_tile_desc_t>;
    using matC_128x256_t = subgroup::tile_t<dtype_sfx, mat_128x256_tile_desc_t>;
    using matC_64x384_t = subgroup::tile_t<dtype_sfx, mat_64x384_tile_desc_t>;
    using matC_64x512_t = subgroup::tile_t<dtype_sfx, mat_64x512_tile_desc_t>;
    using matC_32x1024_t = subgroup::tile_t<dtype_sfx, mat_32x1024_tile_desc_t>;
    using matC_16x2048_t = subgroup::tile_t<dtype_sfx, mat_16x2048_tile_desc_t>;
    using matC_128x64_t = subgroup::tile_t<dtype_sfx, mat_128x64_tile_desc_t>;

    using matC_128x128_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            mat_128x128_tile_desc_t,
            (global_kslicing > 1) ? msg_type::atomic_add
                                  : subgroup::msg_type_v<
                                          mat_128x128_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matC_128x256_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            mat_128x256_tile_desc_t,
            (global_kslicing > 1) ? msg_type::atomic_add
                                  : subgroup::msg_type_v<
                                          mat_128x256_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matC_64x384_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            mat_64x384_tile_desc_t,
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<mat_64x384_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matC_64x512_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            mat_64x512_tile_desc_t,
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<mat_64x512_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matC_32x1024_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            mat_32x1024_tile_desc_t,
            (global_kslicing > 1) ? msg_type::atomic_add
                                  : subgroup::msg_type_v<
                                          mat_32x1024_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matC_16x2048_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            mat_16x2048_tile_desc_t,
            (global_kslicing > 1) ? msg_type::atomic_add
                                  : subgroup::msg_type_v<
                                          mat_16x2048_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matC_128x64_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            mat_128x64_tile_desc_t,
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<mat_128x64_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;

    using matDpotMk_128x128_t
            = subgroup::tile_t<uint8_t, mat_128x128_tile_desc_t>;
    using matDpotMk_128x256_t
            = subgroup::tile_t<uint8_t, mat_128x256_tile_desc_t>;
    using matDpotMk_64x384_t
            = subgroup::tile_t<uint8_t, mat_64x384_tile_desc_t>;
    using matDpotMk_64x512_t
            = subgroup::tile_t<uint8_t, mat_64x512_tile_desc_t>;
    using matDpotMk_32x1024_t
            = subgroup::tile_t<uint8_t, mat_32x1024_tile_desc_t>;
    using matDpotMk_16x2048_t
            = subgroup::tile_t<uint8_t, mat_16x2048_tile_desc_t>;
    using matDpotMk_128x64_t
            = subgroup::tile_t<uint8_t, mat_128x64_tile_desc_t>;

    using matDpotMk_128x128_payload_t = subgroup::mem_payload_t<
            mem_desc_t<uint8_t, mem_layout_c, mem_space_c>,
            mat_128x128_tile_desc_t,
            subgroup::msg_type_v<mat_128x128_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matDpotMk_128x256_payload_t = subgroup::mem_payload_t<
            mem_desc_t<uint8_t, mem_layout_c, mem_space_c>,
            mat_128x256_tile_desc_t,
            subgroup::msg_type_v<mat_128x256_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matDpotMk_64x384_payload_t = subgroup::mem_payload_t<
            mem_desc_t<uint8_t, mem_layout_c, mem_space_c>,
            mat_64x384_tile_desc_t,
            subgroup::msg_type_v<mat_64x384_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matDpotMk_64x512_payload_t = subgroup::mem_payload_t<
            mem_desc_t<uint8_t, mem_layout_c, mem_space_c>,
            mat_64x512_tile_desc_t,
            subgroup::msg_type_v<mat_64x512_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matDpotMk_32x1024_payload_t = subgroup::mem_payload_t<
            mem_desc_t<uint8_t, mem_layout_c, mem_space_c>,
            mat_32x1024_tile_desc_t,
            subgroup::msg_type_v<mat_32x1024_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matDpotMk_16x2048_payload_t = subgroup::mem_payload_t<
            mem_desc_t<uint8_t, mem_layout_c, mem_space_c>,
            mat_16x2048_tile_desc_t,
            subgroup::msg_type_v<mat_16x2048_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matDpotMk_128x64_payload_t = subgroup::mem_payload_t<
            mem_desc_t<uint8_t, mem_layout_c, mem_space_c>,
            mat_128x64_tile_desc_t,
            subgroup::msg_type_v<mat_128x64_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;

    /// @brief Arguments for xetla_softmax_fwd_t::run.
    /// User should prepare matQ_ptr, matK_ptr, matQKT_ptr, ...
    struct arguments_t {
        // assume base address, surface width, height, pitch, start coordinate was set
        uint32_t *mList_ptr;
        dtype_bin *matQ_ptr;
        dtype_bin *matK_ptr;
        dtype_bin *matV_ptr;
        uint32_t *matMkin_ptr;
        uint32_t *matMkdpot_ptr;
        dtype_sfx *matQKT_ptr;
        dtype_bot *matOut_ptr;
        float *Max_ptr;
        float *Sum_ptr;
        float Pinv;
        float Scaling;
    };

    /// @brief Main execution function for fused mha softmax
    /// The basic process is GEMM -> Softmax -> GEMM.
    /// @param args [in] Includes base descriptors and tid info.
    __XETLA_API static void call(sycl::nd_item<3> &item, arguments_t *args) {

        int tru_seqlen = 0;
        int tru_seqlen_ex = 0;
        int seqlen_entry = 0;

        int groupid = item.get_group(0);
        int hiddensize = 1024;
        int numhead = 16;
        int hdsz = 64;
        int wg_tile_QKT_k = hdsz; //args->matrix_k;
        int wg_tile_out_k;
        int batchid = groupid / numhead;
        int headid = groupid % numhead;

        work_group_t g_thd32_tid;
        int tid_linear = item.get_local_linear_id();
        g_thd32_tid.init(tid_linear);

        uint32_t batch_offset = sizeof(uint32_t) * list_width * batchid;
        xetla_vector<uint32_t, list_width> list_offsets
                = xetla_vector_gen<uint32_t, list_width>(0, 1);
        list_offsets *= sizeof(uint32_t);
        list_offsets += batch_offset;

        xetla_vector<uint32_t, list_width> list_vec
                = xetla_load_global<uint32_t, 1, data_size::default_size,
                        cache_hint::read_invalidate, cache_hint::cached,
                        list_width>(args->mList_ptr, list_offsets);
        tru_seqlen = list_vec[0];
        seqlen_entry = list_vec[1];
        wg_tile_out_k = tru_seqlen;
        tru_seqlen_ex = tru_seqlen; //DW align
        if (sfx_type_size == 2)
            tru_seqlen_ex = (((tru_seqlen + 1) >> 1) << 1);
        else if (sfx_type_size == 1)
            tru_seqlen_ex = (((tru_seqlen + 3) >> 2) << 2);
        //float totalscaling = args->Pinv * args->Scaling;

        xetla_rand_t<Rand_SIMD> Rand_Gen;
        uint32_t rand_threshold = rand_threshold_const;
        if constexpr (Dopt_RandGenflag == true) {
            uint64_t rand_seed = 67280421310721;
            uint64_t rand_subseq
                    = (groupid * ThreadNum + tid_linear) * Rand_SIMD;
            uint64_t rand_offset = list_vec.xetla_format<uint64_t>()[1];
            if (list_vec[4] != 0) rand_threshold = list_vec[4];
            if (rand_offset == 0) {
                xetla_vector<uint32_t, 4> time_stamp = get_time_stamp();
                rand_offset = time_stamp.xetla_format<uint64_t>()[0];
            }
            Rand_Gen.init(rand_seed, rand_subseq, rand_offset);
        }

        //std_leqlen = 256
        int all_vert_loop_num = 2;
        int all_vert_stride = 128;
        int all_vert128_shift = 0;
        int block_16x16_num = 4;
        int tid_x_shift = 0;

        int std_seqlen;
        if (tru_seqlen <= 128) {
            std_seqlen = 128;
            tid_x_shift = 2; //        16x32 128/32 = 4
            all_vert_loop_num = 1;
            block_16x16_num = 2;
        } else if (tru_seqlen <= 256) {
            std_seqlen = 256;
            tid_x_shift = 2; //        16x64 256/64 = 4
        } else if (tru_seqlen <= 384) {
            std_seqlen = 384;
            all_vert_stride = 64;
            all_vert128_shift = 1;
            block_16x16_num = 3;
            tid_x_shift = 3; //        16x48 384/48 = 8
            all_vert_loop_num = (tru_seqlen + all_vert_stride - 1) >> 6;
        } else if (tru_seqlen <= 512) {
            std_seqlen = 512;
            all_vert_stride = 64;
            all_vert128_shift = 1;
            tid_x_shift = 3; //        16x64 512/64 = 8
            all_vert_loop_num = (tru_seqlen + all_vert_stride - 1) >> 6;
        } else if (tru_seqlen <= 1024) {
            std_seqlen = 1024;
            all_vert_stride = 32;
            all_vert128_shift = 2;
            tid_x_shift = 4; //        16x64 1024/64 = 16
            all_vert_loop_num = (tru_seqlen + all_vert_stride - 1) >> 5;
        } else if (tru_seqlen <= 2048) {
            std_seqlen = 2048;
            all_vert_stride = 16;
            all_vert128_shift = 3;
            tid_x_shift = 5; //        16x64 2048/64 = 32
            all_vert_loop_num = (tru_seqlen + all_vert_stride - 1) >> 4;
        }
        all_vert_loop_num = ((all_vert_loop_num + (1 << all_vert128_shift) - 1)
                                    >> all_vert128_shift)
                << all_vert128_shift;
        int tid_x = tid_linear & ((1 << tid_x_shift) - 1);
        int tid_y = tid_linear >> tid_x_shift;

        xetla_nbarrier_t<32, 32, gpu_arch::Xe> first_nbarr;
        xetla_nbarrier_t<32, 32, gpu_arch::Xe> second_nbarr;
        xetla_nbarrier_t<32, 32, gpu_arch::Xe> third_nbarr;
        first_nbarr.init_nbarrier(31, nbarrier_role::producer_consumer);
        second_nbarr.init_nbarrier(30, nbarrier_role::producer_consumer);
        third_nbarr.init_nbarrier(29, nbarrier_role::producer_consumer);

        xetla_vector<int8_t, 4 * 16> attn_mk_4x16;
        int valid_block_16x16_x = (tid_x + 1) * 16 * block_16x16_num;
        {
            int bndy_block_num = 0;
            if (valid_block_16x16_x <= tru_seqlen)
                valid_block_16x16_x = block_16x16_num;
            else {
                bndy_block_num = valid_block_16x16_x;
                valid_block_16x16_x = (tru_seqlen + 15 + 16 * block_16x16_num
                                              - valid_block_16x16_x)
                        >> 4;
                bndy_block_num = bndy_block_num
                        + (valid_block_16x16_x - block_16x16_num) * 16
                        - tru_seqlen;
            }

            xetla_vector<uint32_t, 16> address_attn_mk
                    = xetla_vector_gen<uint32_t, 16>(0, 1);
            int attn_mk_address_offset
                    = (batchid * Max_SeqLen) + (tid_x * 16 * block_16x16_num);
            address_attn_mk *= sizeof(uint32_t);
            address_attn_mk += attn_mk_address_offset;
            attn_mk_4x16.xetla_format<uint32_t>().xetla_select<16, 1>(0)
                    = xetla_load_global<uint32_t, 1, data_size::default_size,
                            cache_hint::read_invalidate, cache_hint::cached,
                            16>(args->matMkin_ptr, address_attn_mk);

            for (int i = 1; i <= bndy_block_num; i++)
                attn_mk_4x16[valid_block_16x16_x * 16 - i] = 1;
        }

        for (int all_vert_loop = 0; all_vert_loop < all_vert_loop_num;
                all_vert_loop++) {

            xetla_vector<float, 4 * 16 * 16> matElem_reg_4x16x16;
            xetla_vector<uint8_t, 4 * 16 * 16> rand_bit;
            bool valid_compute = true;

            if (((all_vert_loop * all_vert_stride + tid_y * 16) >= tru_seqlen)
                    || ((tid_x * 16 * block_16x16_num) >= tru_seqlen))
                valid_compute = false;

            if (valid_compute) {

                switch (std_seqlen) {
                    case 128: {
                        gemm_arguments_128x128 gemm_arg_128x128;
                        matAcc_128x128_t matAcc_128x128;

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_128x128.matA_base_desc.init({args->matQ_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose
                        gemm_arg_128x128.matB_base_desc.init({args->matK_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_128x128.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_128x128.init(0);

                        gemm_op_128x128_t gemm_op_128x128;

                        gemm_op_128x128(
                                g_thd32_tid, matAcc_128x128, gemm_arg_128x128);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<16 * 32, 1>(0)
                                = matAcc_128x128.reg * args->Pinv;
                    } break;
                    case 256: {
                        gemm_arguments_128x256 gemm_arg_128x256;
                        matAcc_128x256_t matAcc_128x256;

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_128x256.matA_base_desc.init({args->matQ_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose
                        gemm_arg_128x256.matB_base_desc.init({args->matK_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_128x256.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_128x256.init(0);

                        gemm_op_128x256_t gemm_op_128x256;

                        gemm_op_128x256(
                                g_thd32_tid, matAcc_128x256, gemm_arg_128x256);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<4 * 16 * 16, 1>(0)
                                = matAcc_128x256.reg * args->Pinv;

                    } break;
                    case 384: {
                        gemm_arguments_64x384 gemm_arg_64x384;
                        matAcc_64x384_t matAcc_64x384;

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_64x384.matA_base_desc.init({args->matQ_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose
                        gemm_arg_64x384.matB_base_desc.init({args->matK_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_64x384.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_64x384.init(0);

                        gemm_op_64x384_t gemm_op_64x384;
                        gemm_op_64x384(
                                g_thd32_tid, matAcc_64x384, gemm_arg_64x384);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<3 * 16 * 16, 1>(0)
                                = matAcc_64x384.reg * args->Pinv;

                    } break;
                    case 512: {
                        gemm_arguments_64x512 gemm_arg_64x512;
                        matAcc_64x512_t matAcc_64x512;

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_64x512.matA_base_desc.init({args->matQ_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose
                        gemm_arg_64x512.matB_base_desc.init({args->matK_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_64x512.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_64x512.init(0);

                        gemm_op_64x512_t gemm_op_64x512;
                        gemm_op_64x512(
                                g_thd32_tid, matAcc_64x512, gemm_arg_64x512);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<4 * 16 * 16, 1>(0)
                                = matAcc_64x512.reg * args->Pinv;

                    } break;
                    case 1024: {
                        gemm_arguments_32x1024 gemm_arg_32x1024;
                        matAcc_32x1024_t matAcc_32x1024;

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_32x1024.matA_base_desc.init({args->matQ_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose
                        gemm_arg_32x1024.matB_base_desc.init({args->matK_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_32x1024.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_32x1024.init(0);
                        gemm_op_32x1024_t gemm_op_32x1024;
                        gemm_op_32x1024(
                                g_thd32_tid, matAcc_32x1024, gemm_arg_32x1024);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<4 * 16 * 16, 1>(0)
                                = matAcc_32x1024.reg * args->Pinv;
                    } break;
                    case 2048: {
                        gemm_arguments_16x2048 gemm_arg_16x2048;
                        matAcc_16x2048_t matAcc_16x2048;

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_16x2048.matA_base_desc.init({args->matQ_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose
                        gemm_arg_16x2048.matB_base_desc.init({args->matK_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_16x2048.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_16x2048.init(0);
                        gemm_op_16x2048_t gemm_op_16x2048;
                        gemm_op_16x2048(
                                g_thd32_tid, matAcc_16x2048, gemm_arg_16x2048);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<4 * 16 * 16, 1>(0)
                                = matAcc_16x2048.reg * args->Pinv;
                    }
                } //switch

                { //softmax
                    xetla_vector<float, 16 * 1> matElem_reg_max_local;
                    xetla_vector<float, 16 * 1> matElem_reg_max_global;
                    xetla_vector<uint8_t, 16 * 4> matElem_reg_attn_mk;

                    xetla_vector<uint32_t, 16> address_fmax
                            = xetla_vector_gen<uint32_t, 16>(0, 1);
                    int address_offset
                            = (batchid * numhead + headid) * Max_SeqLen
                            + all_vert_stride * all_vert_loop + tid_y * 16;
                    address_fmax += address_offset;
                    address_fmax *= sizeof(float);

                    {
                        xetla_vector<float, 16 * 16> matElem_reg_Max;
                        xetla_vector<float, 16 * 8> matElem_reg_Max_8;
                        xetla_vector<float, 16 * 4> matElem_reg_Max_4;
                        xetla_vector<float, 16 * 2> matElem_reg_Max_2;

                        {
#pragma unroll
                            for (int i = 0; i < 16; i++) {
                                matElem_reg_4x16x16.xetla_select<16, 1>(16 * i)
                                        .merge(-1e32,
                                                attn_mk_4x16.xetla_select<16,
                                                        1>(0)
                                                        > 0);
                            }

                            matElem_reg_Max
                                    = matElem_reg_4x16x16
                                              .xetla_select<16 * 16, 1>(0);
                        }

                        if (valid_block_16x16_x > 1) {
#pragma unroll
                            for (int i = 0; i < 16; i++) {
                                matElem_reg_4x16x16
                                        .xetla_select<16, 1>(
                                                16 * i + 16 * 16 * 1)
                                        .merge(-1e32,
                                                attn_mk_4x16.xetla_select<16,
                                                        1>(16)
                                                        > 0);
                            }
                            matElem_reg_Max.merge(
                                    matElem_reg_4x16x16
                                            .xetla_select<16 * 16, 1>(
                                                    16 * 16 * 1),
                                    matElem_reg_4x16x16
                                                    .xetla_select<16 * 16, 1>(
                                                            16 * 16 * 1)
                                            > matElem_reg_Max);

                            if (valid_block_16x16_x > 2) {
#pragma unroll
                                for (int i = 0; i < 16; i++) {
                                    matElem_reg_4x16x16
                                            .xetla_select<16, 1>(
                                                    16 * i + 16 * 16 * 2)
                                            .merge(-1e32,
                                                    attn_mk_4x16.xetla_select<
                                                            16, 1>(16 * 2)
                                                            > 0);
                                }
                                matElem_reg_Max.merge(
                                        matElem_reg_4x16x16
                                                .xetla_select<16 * 16, 1>(
                                                        16 * 16 * 2),
                                        matElem_reg_4x16x16.xetla_select<
                                                16 * 16, 1>(16 * 16 * 2)
                                                > matElem_reg_Max);
                                if (valid_block_16x16_x > 3) {
#pragma unroll
                                    for (int i = 0; i < 16; i++) {
                                        matElem_reg_4x16x16
                                                .xetla_select<16, 1>(
                                                        16 * i + 16 * 16 * 3)
                                                .merge(-1e32,
                                                        attn_mk_4x16.xetla_select<
                                                                16, 1>(16 * 3)
                                                                > 0);
                                    }
                                    matElem_reg_Max.merge(
                                            matElem_reg_4x16x16
                                                    .xetla_select<16 * 16, 1>(
                                                            16 * 16 * 3),
                                            matElem_reg_4x16x16.xetla_select<
                                                    16 * 16, 1>(16 * 16 * 3)
                                                    > matElem_reg_Max);
                                }
                            }
                        }

                        matElem_reg_Max_8.xetla_format<float, 16, 8>()
                                .xetla_select<16, 1, 8, 1>(0, 0)
                                .merge(matElem_reg_Max
                                                .xetla_format<float, 16, 16>()
                                                .xetla_select<16, 1, 8, 1>(
                                                        0, 0),
                                        matElem_reg_Max
                                                .xetla_format<float, 16, 16>()
                                                .xetla_select<16, 1, 8, 1>(
                                                        0, 8),
                                        matElem_reg_Max.xetla_format<float, 16,
                                                               16>()
                                                        .xetla_select<16, 1, 8,
                                                                1>(0, 0)
                                                > matElem_reg_Max
                                                          .xetla_format<float,
                                                                  16, 16>()
                                                          .xetla_select<16, 1,
                                                                  8, 1>(0, 8));
                        matElem_reg_Max_4.xetla_format<float, 16, 4>()
                                .xetla_select<16, 1, 4, 1>(0, 0)
                                .merge(matElem_reg_Max_8
                                                .xetla_format<float, 16, 8>()
                                                .xetla_select<16, 1, 4, 1>(
                                                        0, 0),
                                        matElem_reg_Max_8
                                                .xetla_format<float, 16, 8>()
                                                .xetla_select<16, 1, 4, 1>(
                                                        0, 4),
                                        matElem_reg_Max_8
                                                        .xetla_format<float, 16,
                                                                8>()
                                                        .xetla_select<16, 1, 4,
                                                                1>(0, 0)
                                                > matElem_reg_Max_8
                                                          .xetla_format<float,
                                                                  16, 8>()
                                                          .xetla_select<16, 1,
                                                                  4, 1>(0, 4));
                        matElem_reg_Max_2.xetla_format<float, 16, 2>()
                                .xetla_select<16, 1, 2, 1>(0, 0)
                                .merge(matElem_reg_Max_4
                                                .xetla_format<float, 16, 4>()
                                                .xetla_select<16, 1, 2, 1>(
                                                        0, 0),
                                        matElem_reg_Max_4
                                                .xetla_format<float, 16, 4>()
                                                .xetla_select<16, 1, 2, 1>(
                                                        0, 2),
                                        matElem_reg_Max_4
                                                        .xetla_format<float, 16,
                                                                4>()
                                                        .xetla_select<16, 1, 2,
                                                                1>(0, 0)
                                                > matElem_reg_Max_4
                                                          .xetla_format<float,
                                                                  16, 4>()
                                                          .xetla_select<16, 1,
                                                                  2, 1>(0, 2));
                        matElem_reg_max_local.xetla_format<float, 16, 1>()
                                .xetla_select<16, 1, 1, 1>(0, 0)
                                .merge(matElem_reg_Max_2
                                                .xetla_format<float, 16, 2>()
                                                .xetla_select<16, 1, 1, 1>(
                                                        0, 0),
                                        matElem_reg_Max_2
                                                .xetla_format<float, 16, 2>()
                                                .xetla_select<16, 1, 1, 1>(
                                                        0, 1),
                                        matElem_reg_Max_2
                                                        .xetla_format<float, 16,
                                                                2>()
                                                        .xetla_select<16, 1, 1,
                                                                1>(0, 0)
                                                > matElem_reg_Max_2
                                                          .xetla_format<float,
                                                                  16, 2>()
                                                          .xetla_select<16, 1,
                                                                  1, 1>(0, 1));

                        xetla_mask<16> pred = 1;
                        xetla_tatomic_store_global<float, 16, cache_hint::none,
                                cache_hint::none, atomic_op::fmax>(
                                (uint64_t)args->Max_ptr, address_fmax,
                                matElem_reg_max_local.xetla_select<16, 1>(0),
                                pred);
                    }

                    first_nbarr.arrive();
                    if constexpr (Dopt_RandGenflag == true) {
                        xetla_vector<uint32_t, 4 * RandSIMD> rand_data;
#pragma unroll
                        for (int i = 0; i < ((16 * 16) / (2 * 4 * RandSIMD));
                                i++) {
                            rand_data = Rand_Gen.rand();
                            rand_bit.xetla_select<4 * RandSIMD, 1>(
                                    i * (4 * RandSIMD))
                                    = rand_data > rand_threshold;
                        }
                    }
                    first_nbarr.wait();

                    {
                        matElem_reg_max_global = xetla_load_global<float, 1,
                                data_size::default_size,
                                cache_hint::read_invalidate, cache_hint::cached,
                                16>(args->Max_ptr, address_fmax);

                        auto matElem_reg_max_use = matElem_reg_max_global;

#pragma unroll
                        for (int i = 0; i < 16; i++) {
                            matElem_reg_4x16x16.xetla_select<16 * 16, 1>(0)
                                    .xetla_select<16, 1>(i * 16)
                                    = matElem_reg_4x16x16
                                              .xetla_select<16 * 16, 1>(0)
                                              .xetla_select<16, 1>(i * 16)
                                    - matElem_reg_max_use[i];

                            matElem_reg_4x16x16.xetla_select<16 * 16, 1>(0)
                                    .xetla_select<16, 1>(i * 16)
                                    = xetla_exp<float, 16>(
                                            matElem_reg_4x16x16
                                                    .xetla_select<16 * 16, 1>(0)
                                                    .xetla_select<16, 1>(
                                                            i * 16));
                        }

                        if (valid_block_16x16_x > 1) {
#pragma unroll
                            for (int i = 0; i < 16; i++) {
                                matElem_reg_4x16x16
                                        .xetla_select<16 * 16, 1>(16 * 16 * 1)
                                        .xetla_select<16, 1>(i * 16)
                                        = matElem_reg_4x16x16
                                                  .xetla_select<16 * 16, 1>(
                                                          16 * 16 * 1)
                                                  .xetla_select<16, 1>(i * 16)
                                        - matElem_reg_max_use[i];

                                matElem_reg_4x16x16
                                        .xetla_select<16 * 16, 1>(16 * 16 * 1)
                                        .xetla_select<16, 1>(i * 16)
                                        = xetla_exp<float, 16>(
                                                matElem_reg_4x16x16
                                                        .xetla_select<16 * 16,
                                                                1>(16 * 16 * 1)
                                                        .xetla_select<16, 1>(
                                                                i * 16));
                            }

                            if (valid_block_16x16_x > 2) {
#pragma unroll
                                for (int i = 0; i < 16; i++) {
                                    matElem_reg_4x16x16
                                            .xetla_select<16 * 16, 1>(
                                                    16 * 16 * 2)
                                            .xetla_select<16, 1>(i * 16)
                                            = matElem_reg_4x16x16
                                                      .xetla_select<16 * 16, 1>(
                                                              16 * 16 * 2)
                                                      .xetla_select<16, 1>(
                                                              i * 16)
                                            - matElem_reg_max_use[i];
                                    matElem_reg_4x16x16
                                            .xetla_select<16 * 16, 1>(
                                                    16 * 16 * 2)
                                            .xetla_select<16, 1>(i * 16)
                                            = xetla_exp<float, 16>(
                                                    matElem_reg_4x16x16
                                                            .xetla_select<
                                                                    16 * 16, 1>(
                                                                    16 * 16 * 2)
                                                            .xetla_select<16,
                                                                    1>(i * 16));
                                }

                                if (valid_block_16x16_x > 3) {
#pragma unroll
                                    for (int i = 0; i < 16; i++) {
                                        matElem_reg_4x16x16
                                                .xetla_select<16 * 16, 1>(
                                                        16 * 16 * 3)
                                                .xetla_select<16, 1>(i * 16)
                                                = matElem_reg_4x16x16
                                                          .xetla_select<16 * 16,
                                                                  1>(
                                                                  16 * 16 * 3)
                                                          .xetla_select<16, 1>(
                                                                  i * 16)
                                                - matElem_reg_max_use[i];
                                        matElem_reg_4x16x16
                                                .xetla_select<16 * 16, 1>(
                                                        16 * 16 * 3)
                                                .xetla_select<16, 1>(i * 16)
                                                = xetla_exp<float, 16>(
                                                        matElem_reg_4x16x16
                                                                .xetla_select<
                                                                        16 * 16,
                                                                        1>(16
                                                                        * 16
                                                                        * 3)
                                                                .xetla_select<
                                                                        16, 1>(i
                                                                        * 16));
                                    }
                                }
                            }
                        }
                    }

                    xetla_vector<float, 16 * 1> matElem_reg_Sum_1;

                    {
                        xetla_vector<float, 16 * 16> matElem_reg_Sum;
                        xetla_vector<float, 16 * 8> matElem_reg_Sum_8;
                        xetla_vector<float, 16 * 4> matElem_reg_Sum_4;
                        xetla_vector<float, 16 * 2> matElem_reg_Sum_2;

                        matElem_reg_Sum
                                = matElem_reg_4x16x16.xetla_select<16 * 16, 1>(
                                        0);

                        if (valid_block_16x16_x > 1) {
                            matElem_reg_Sum
                                    += matElem_reg_4x16x16
                                               .xetla_select<16 * 16, 1>(
                                                       16 * 16 * 1);
                            if (valid_block_16x16_x > 2) {
                                matElem_reg_Sum
                                        += matElem_reg_4x16x16
                                                   .xetla_select<16 * 16, 1>(
                                                           16 * 16 * 2);
                                if (valid_block_16x16_x > 3)
                                    matElem_reg_Sum
                                            += matElem_reg_4x16x16.xetla_select<
                                                    16 * 16, 1>(16 * 16 * 3);
                            }
                        }
                        matElem_reg_Sum_8.xetla_format<float, 16, 8>()
                                = matElem_reg_Sum.xetla_format<float, 16, 16>()
                                          .xetla_select<16, 1, 8, 1>(0, 0)
                                + matElem_reg_Sum.xetla_format<float, 16, 16>()
                                          .xetla_select<16, 1, 8, 1>(0, 8);

                        matElem_reg_Sum_4.xetla_format<float, 16, 4>()
                                = matElem_reg_Sum_8.xetla_format<float, 16, 8>()
                                          .xetla_select<16, 1, 4, 1>(0, 0)
                                + matElem_reg_Sum_8.xetla_format<float, 16, 8>()
                                          .xetla_select<16, 1, 4, 1>(0, 4);

                        matElem_reg_Sum_2.xetla_format<float, 16, 2>()
                                = matElem_reg_Sum_4.xetla_format<float, 16, 4>()
                                          .xetla_select<16, 1, 2, 1>(0, 0)
                                + matElem_reg_Sum_4.xetla_format<float, 16, 4>()
                                          .xetla_select<16, 1, 2, 1>(0, 2);

                        matElem_reg_Sum_1.xetla_format<float, 16, 1>()
                                = matElem_reg_Sum_2.xetla_format<float, 16, 2>()
                                          .xetla_select<16, 1, 1, 1>(0, 0)
                                + matElem_reg_Sum_2.xetla_format<float, 16, 2>()
                                          .xetla_select<16, 1, 1, 1>(0, 1);

                        xetla_mask<16> pred = 1;
                        xetla_tatomic_store_global<float, 16, cache_hint::none,
                                cache_hint::none, atomic_op::fadd>(
                                (uint64_t)args->Sum_ptr, address_fmax,
                                matElem_reg_Sum_1.xetla_select<16, 1>(0), pred);
                    }

                    second_nbarr.arrive();
                    if constexpr (Dopt_RandGenflag == true) {
                        xetla_vector<uint32_t, 4 * RandSIMD> rand_data;
#pragma unroll
                        for (int i = ((16 * 16) / (2 * 4 * RandSIMD));
                                i < ((16 * 16) / (4 * RandSIMD)); i++) {
                            rand_data = Rand_Gen.rand();
                            rand_bit.xetla_select<4 * RandSIMD, 1>(
                                    i * (4 * RandSIMD))
                                    = rand_data > rand_threshold;
                        }
                    }
                    second_nbarr.wait();

                    {
                        matElem_reg_Sum_1 = xetla_load_global<float, 1,
                                data_size::default_size,
                                cache_hint::read_invalidate, cache_hint::cached,
                                16>(args->Sum_ptr, address_fmax);

                        matElem_reg_Sum_1
                                = xetla_inv<float, 16>(matElem_reg_Sum_1);
                        matElem_reg_Sum_1 *= args->Scaling;

#pragma unroll
                        for (int i = 0; i < 16; i++) {
                            matElem_reg_4x16x16.xetla_select<16 * 16, 1>(0)
                                    .xetla_select<16, 1>(i * 16)
                                    = matElem_reg_4x16x16
                                              .xetla_select<16 * 16, 1>(0)
                                              .xetla_select<16, 1>(i * 16)
                                    * matElem_reg_Sum_1[i];
                        }

                        if (valid_block_16x16_x > 1) {
#pragma unroll
                            for (int i = 0; i < 16; i++) {
                                matElem_reg_4x16x16
                                        .xetla_select<16 * 16, 1>(16 * 16 * 1)
                                        .xetla_select<16, 1>(i * 16)
                                        = matElem_reg_4x16x16
                                                  .xetla_select<16 * 16, 1>(
                                                          16 * 16 * 1)
                                                  .xetla_select<16, 1>(i * 16)
                                        * matElem_reg_Sum_1[i];
                            }

                            if constexpr (Dopt_RandGenflag == true) {
                                xetla_vector<uint32_t, 4 * RandSIMD> rand_data;
#pragma unroll
                                for (int i = 0;
                                        i < ((16 * 16) / (4 * RandSIMD)); i++) {
                                    rand_data = Rand_Gen.rand();
                                    rand_bit.xetla_select<4 * RandSIMD, 1>(
                                            (i * (4 * RandSIMD))
                                            + (16 * 16 * 1))
                                            = rand_data > rand_threshold;
                                }
                            }

                            if (valid_block_16x16_x > 2) {
#pragma unroll
                                for (int i = 0; i < 16; i++) {
                                    matElem_reg_4x16x16
                                            .xetla_select<16 * 16, 1>(
                                                    16 * 16 * 2)
                                            .xetla_select<16, 1>(i * 16)
                                            = matElem_reg_4x16x16
                                                      .xetla_select<16 * 16, 1>(
                                                              16 * 16 * 2)
                                                      .xetla_select<16, 1>(
                                                              i * 16)
                                            * matElem_reg_Sum_1[i];
                                }

                                if constexpr (Dopt_RandGenflag == true) {
                                    xetla_vector<uint32_t, 4 * RandSIMD>
                                            rand_data;
#pragma unroll
                                    for (int i = 0;
                                            i < ((16 * 16) / (4 * RandSIMD));
                                            i++) {
                                        rand_data = Rand_Gen.rand();
                                        rand_bit.xetla_select<4 * RandSIMD, 1>(
                                                (i * (4 * RandSIMD))
                                                + (16 * 16 * 2))
                                                = rand_data > rand_threshold;
                                    }
                                }

                                if (valid_block_16x16_x > 3) {
#pragma unroll
                                    for (int i = 0; i < 16; i++) {
                                        matElem_reg_4x16x16
                                                .xetla_select<16 * 16, 1>(
                                                        16 * 16 * 3)
                                                .xetla_select<16, 1>(i * 16)
                                                = matElem_reg_4x16x16
                                                          .xetla_select<16 * 16,
                                                                  1>(
                                                                  16 * 16 * 3)
                                                          .xetla_select<16, 1>(
                                                                  i * 16)
                                                * matElem_reg_Sum_1[i];
                                    }

                                    if constexpr (Dopt_RandGenflag == true) {
                                        xetla_vector<uint32_t, 4 * RandSIMD>
                                                rand_data;
#pragma unroll
                                        for (int i = 0; i
                                                < ((16 * 16) / (4 * RandSIMD));
                                                i++) {
                                            rand_data = Rand_Gen.rand();
                                            rand_bit.xetla_select<4 * RandSIMD,
                                                    1>((i * (4 * RandSIMD))
                                                    + (16 * 16 * 3))
                                                    = rand_data
                                                    > rand_threshold;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } //softmax

                //store
                switch (std_seqlen) {
                    case 128: {
                        matC_128x128_t matC_128x128;
                        matC_128x128_payload_t matC_128x128_payload;
                        matDpotMk_128x128_t matDpotMk_128x128;
                        matDpotMk_128x128_payload_t matDpotMk_128x128_payload;

                        int width_c = max_seqlen;
                        int height_c
                                = max_seqlen * (batchid * numhead + headid + 1);
                        int pitch_c = max_seqlen;
                        int start_x_c = gemm_op_128x128_t::get_matC_offset_x(
                                g_thd32_tid);
                        int start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_128x128_t::get_matC_offset_y(
                                        g_thd32_tid);
                        matC_128x128_payload.init(args->matQKT_ptr, width_c,
                                height_c, pitch_c, start_x_c, start_y_c);

                        if constexpr (Dopt_RandGenflag == false) {
                            uint8_t *matMkdpot_byte_ptr
                                    = (uint8_t *)(args->matMkdpot_ptr);
                            matDpotMk_128x128_payload.init(matMkdpot_byte_ptr,
                                    width_c, height_c, pitch_c, start_x_c,
                                    start_y_c);
                            subgroup::tile_load(matDpotMk_128x128,
                                    matDpotMk_128x128_payload);
                        }

                        xetla_vector<float, 16 * 32> matElem_reg_store
                                = matElem_reg_4x16x16.xetla_format<float>()
                                          .xetla_select<16 * 32, 1>(0);
                        matC_128x128.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_store);

                        if constexpr (Dopt_RandGenflag == false) {
                            rand_bit.xetla_select<16 * 16 * 2, 1>(0)
                                    = matDpotMk_128x128.reg;
                        }

                        if constexpr (sfx_type_size == 2) {
                            xetla_vector<uint16_t, 2 * 16 * 16> drop_mk_w = 0;
                            drop_mk_w.xetla_merge(SIGN_BIT_W16,
                                    rand_bit.xetla_select<16 * 16 * 2, 1>(0)
                                            > 0);
                            matC_128x128.reg.xetla_format<uint16_t>()
                                    |= drop_mk_w;
                        }
                        if constexpr (sfx_type_size == 1) {
                            xetla_vector<uint8_t, 2 * 16 * 16> drop_mk_b = 0;
                            drop_mk_b.xetla_merge(SIGN_BIT_B8,
                                    rand_bit.xetla_select<16 * 16 * 2, 1>(0)
                                            > 0);
                            matC_128x128.reg.xetla_format<uint8_t>()
                                    |= drop_mk_b;
                        }

                        subgroup::tile_store(
                                matC_128x128, matC_128x128_payload);
                        xetla_fence<memory_kind::untyped_global>();

                    } break;
                    case 256: {
                        matC_128x256_t matC_128x256;
                        matC_128x256_payload_t matC_128x256_payload;
                        matDpotMk_128x256_t matDpotMk_128x256;
                        matDpotMk_128x256_payload_t matDpotMk_128x256_payload;

                        int width_c = max_seqlen;
                        int height_c
                                = max_seqlen * (batchid * numhead + headid + 1);
                        int pitch_c = max_seqlen;
                        int start_x_c = gemm_op_128x256_t::get_matC_offset_x(
                                g_thd32_tid);
                        int start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_128x256_t::get_matC_offset_y(
                                        g_thd32_tid);

                        matC_128x256_payload.init(args->matQKT_ptr, width_c,
                                height_c, pitch_c, start_x_c, start_y_c);

                        if constexpr (Dopt_RandGenflag == false) {
                            uint8_t *matMkdpot_byte_ptr
                                    = (uint8_t *)(args->matMkdpot_ptr);
                            matDpotMk_128x256_payload.init(matMkdpot_byte_ptr,
                                    width_c, height_c, pitch_c, start_x_c,
                                    start_y_c);
                            subgroup::tile_load(matDpotMk_128x256,
                                    matDpotMk_128x256_payload);
                        }

                        matC_128x256.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_4x16x16);

                        if constexpr (Dopt_RandGenflag == false) {
                            rand_bit = matDpotMk_128x256.reg;
                        }

                        if constexpr (sfx_type_size == 2) {
                            xetla_vector<uint16_t, 4 * 16 * 16> drop_mk_w = 0;
                            drop_mk_w.xetla_merge(SIGN_BIT_W16,
                                    rand_bit.xetla_select<16 * 16 * 4, 1>(0)
                                            > 0);
                            matC_128x256.reg.xetla_format<uint16_t>()
                                    |= drop_mk_w;
                        }
                        if constexpr (sfx_type_size == 1) {
                            xetla_vector<uint8_t, 4 * 16 * 16> drop_mk_b = 0;
                            drop_mk_b.xetla_merge(SIGN_BIT_B8,
                                    rand_bit.xetla_select<16 * 16 * 4, 1>(0)
                                            > 0);
                            matC_128x256.reg.xetla_format<uint8_t>()
                                    |= drop_mk_b;
                        }

                        subgroup::tile_store(
                                matC_128x256, matC_128x256_payload);
                        xetla_fence<memory_kind::untyped_global>();

                    } break;
                    case 384: {
                        matC_64x384_t matC_64x384;
                        matC_64x384_payload_t matC_64x384_payload;
                        matDpotMk_64x384_t matDpotMk_64x384;
                        matDpotMk_64x384_payload_t matDpotMk_64x384_payload;

                        int width_c = max_seqlen;
                        int height_c
                                = max_seqlen * (batchid * numhead + headid + 1);
                        int pitch_c = max_seqlen;
                        int start_x_c = gemm_op_64x384_t::get_matC_offset_x(
                                g_thd32_tid);
                        int start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_64x384_t::get_matC_offset_y(
                                        g_thd32_tid);

                        matC_64x384_payload.init(args->matQKT_ptr, width_c,
                                height_c, pitch_c, start_x_c, start_y_c);

                        if constexpr (Dopt_RandGenflag == false) {
                            uint8_t *matMkdpot_byte_ptr
                                    = (uint8_t *)(args->matMkdpot_ptr);
                            matDpotMk_64x384_payload.init(matMkdpot_byte_ptr,
                                    width_c, height_c, pitch_c, start_x_c,
                                    start_y_c);
                            subgroup::tile_load(
                                    matDpotMk_64x384, matDpotMk_64x384_payload);
                        }

                        xetla_vector<float, 3 * 16 * 16> matElem_reg_store
                                = matElem_reg_4x16x16.xetla_format<float>()
                                          .xetla_select<3 * 16 * 16, 1>(0);
                        matC_64x384.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_store);

                        if constexpr (Dopt_RandGenflag == false) {
                            rand_bit.xetla_select<16 * 16 * 3, 1>(0)
                                    = matDpotMk_64x384.reg;
                        }

                        if constexpr (sfx_type_size == 2) {
                            xetla_vector<uint16_t, 3 * 16 * 16> drop_mk_w = 0;
                            drop_mk_w.xetla_merge(SIGN_BIT_W16,
                                    rand_bit.xetla_select<16 * 16 * 3, 1>(0)
                                            > 0);
                            matC_64x384.reg.xetla_format<uint16_t>()
                                    |= drop_mk_w;
                        }
                        if constexpr (sfx_type_size == 1) {
                            xetla_vector<uint8_t, 3 * 16 * 16> drop_mk_b = 0;
                            drop_mk_b.xetla_merge(SIGN_BIT_B8,
                                    rand_bit.xetla_select<16 * 16 * 3, 1>(0)
                                            > 0);
                            matC_64x384.reg.xetla_format<uint8_t>()
                                    |= drop_mk_b;
                        }

                        subgroup::tile_store(matC_64x384, matC_64x384_payload);
                        xetla_fence<memory_kind::untyped_global>();
                    } break;
                    case 512: {
                        matC_64x512_t matC_64x512;
                        matC_64x512_payload_t matC_64x512_payload;
                        matDpotMk_64x512_t matDpotMk_64x512;
                        matDpotMk_64x512_payload_t matDpotMk_64x512_payload;

                        int width_c = max_seqlen;
                        int height_c
                                = max_seqlen * (batchid * numhead + headid + 1);
                        int pitch_c = max_seqlen;
                        int start_x_c = gemm_op_64x512_t::get_matC_offset_x(
                                g_thd32_tid);
                        int start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_64x512_t::get_matC_offset_y(
                                        g_thd32_tid);
                        matC_64x512_payload.init(args->matQKT_ptr, width_c,
                                height_c, pitch_c, start_x_c, start_y_c);

                        if constexpr (Dopt_RandGenflag == false) {
                            uint8_t *matMkdpot_byte_ptr
                                    = (uint8_t *)(args->matMkdpot_ptr);
                            matDpotMk_64x512_payload.init(matMkdpot_byte_ptr,
                                    width_c, height_c, pitch_c, start_x_c,
                                    start_y_c);
                            subgroup::tile_load(
                                    matDpotMk_64x512, matDpotMk_64x512_payload);
                        }

                        matC_64x512.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_4x16x16);

                        if constexpr (Dopt_RandGenflag == false) {
                            rand_bit = matDpotMk_64x512.reg;
                        }

                        if constexpr (sfx_type_size == 2) {
                            xetla_vector<uint16_t, 4 * 16 * 16> drop_mk_w = 0;
                            drop_mk_w.xetla_merge(SIGN_BIT_W16,
                                    rand_bit.xetla_select<16 * 16 * 4, 1>(0)
                                            > 0);
                            matC_64x512.reg.xetla_format<uint16_t>()
                                    |= drop_mk_w;
                        }
                        if constexpr (sfx_type_size == 1) {
                            xetla_vector<uint8_t, 4 * 16 * 16> drop_mk_b = 0;
                            drop_mk_b.xetla_merge(SIGN_BIT_B8,
                                    rand_bit.xetla_select<16 * 16 * 4, 1>(0)
                                            > 0);
                            matC_64x512.reg.xetla_format<uint8_t>()
                                    |= drop_mk_b;
                        }

                        subgroup::tile_store(matC_64x512, matC_64x512_payload);
                        xetla_fence<memory_kind::untyped_global>();
                    } break;
                    case 1024: {
                        matC_32x1024_t matC_32x1024;
                        matC_32x1024_payload_t matC_32x1024_payload;
                        matDpotMk_32x1024_t matDpotMk_32x1024;
                        matDpotMk_32x1024_payload_t matDpotMk_32x1024_payload;

                        int width_c = max_seqlen;
                        int height_c
                                = max_seqlen * (batchid * numhead + headid + 1);
                        int pitch_c = max_seqlen;
                        int start_x_c = gemm_op_32x1024_t::get_matC_offset_x(
                                g_thd32_tid);
                        int start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_32x1024_t::get_matC_offset_y(
                                        g_thd32_tid);

                        matC_32x1024_payload.init(args->matQKT_ptr, width_c,
                                height_c, pitch_c, start_x_c, start_y_c);

                        if constexpr (Dopt_RandGenflag == false) {
                            uint8_t *matMkdpot_byte_ptr
                                    = (uint8_t *)(args->matMkdpot_ptr);
                            matDpotMk_32x1024_payload.init(matMkdpot_byte_ptr,
                                    width_c, height_c, pitch_c, start_x_c,
                                    start_y_c);
                            subgroup::tile_load(matDpotMk_32x1024,
                                    matDpotMk_32x1024_payload);
                        }

                        matC_32x1024.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_4x16x16);

                        if constexpr (Dopt_RandGenflag == false) {
                            rand_bit = matDpotMk_32x1024.reg;
                        }

                        if constexpr (sfx_type_size == 2) {
                            xetla_vector<uint16_t, 4 * 16 * 16> drop_mk_w = 0;
                            drop_mk_w.xetla_merge(SIGN_BIT_W16,
                                    rand_bit.xetla_select<16 * 16 * 4, 1>(0)
                                            > 0);
                            matC_32x1024.reg.xetla_format<uint16_t>()
                                    |= drop_mk_w;
                        }
                        if constexpr (sfx_type_size == 1) {
                            xetla_vector<uint8_t, 4 * 16 * 16> drop_mk_b = 0;
                            drop_mk_b.xetla_merge(SIGN_BIT_B8,
                                    rand_bit.xetla_select<16 * 16 * 4, 1>(0)
                                            > 0);
                            matC_32x1024.reg.xetla_format<uint8_t>()
                                    |= drop_mk_b;
                        }

                        subgroup::tile_store(
                                matC_32x1024, matC_32x1024_payload);
                        xetla_fence<memory_kind::untyped_global>();
                    } break;
                    case 2048: {
                        matC_16x2048_t matC_16x2048;
                        matC_16x2048_payload_t matC_16x2048_payload;
                        matDpotMk_16x2048_t matDpotMk_16x2048;
                        matDpotMk_16x2048_payload_t matDpotMk_16x2048_payload;

                        int width_c = max_seqlen;
                        int height_c
                                = max_seqlen * (batchid * numhead + headid + 1);
                        int pitch_c = max_seqlen;
                        int start_x_c = gemm_op_16x2048_t::get_matC_offset_x(
                                g_thd32_tid);
                        int start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_16x2048_t::get_matC_offset_y(
                                        g_thd32_tid);

                        matC_16x2048_payload.init(args->matQKT_ptr, width_c,
                                height_c, pitch_c, start_x_c, start_y_c);

                        if constexpr (Dopt_RandGenflag == false) {
                            uint8_t *matMkdpot_byte_ptr
                                    = (uint8_t *)(args->matMkdpot_ptr);
                            matDpotMk_16x2048_payload.init(matMkdpot_byte_ptr,
                                    width_c, height_c, pitch_c, start_x_c,
                                    start_y_c);
                            subgroup::tile_load(matDpotMk_16x2048,
                                    matDpotMk_16x2048_payload);
                        }

                        matC_16x2048.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_4x16x16);

                        if constexpr (Dopt_RandGenflag == false) {
                            rand_bit = matDpotMk_16x2048.reg;
                        }

                        if constexpr (sfx_type_size == 2) {
                            xetla_vector<uint16_t, 4 * 16 * 16> drop_mk_w = 0;
                            drop_mk_w.xetla_merge(SIGN_BIT_W16,
                                    rand_bit.xetla_select<16 * 16 * 4, 1>(0)
                                            > 0);
                            matC_16x2048.reg.xetla_format<uint16_t>()
                                    |= drop_mk_w;
                        }
                        if constexpr (sfx_type_size == 1) {
                            xetla_vector<uint8_t, 4 * 16 * 16> drop_mk_b = 0;
                            drop_mk_b.xetla_merge(SIGN_BIT_B8,
                                    rand_bit.xetla_select<16 * 16 * 4, 1>(0)
                                            > 0);
                            matC_16x2048.reg.xetla_format<uint8_t>()
                                    |= drop_mk_b;
                        }

                        subgroup::tile_store(
                                matC_16x2048, matC_16x2048_payload);
                        xetla_fence<memory_kind::untyped_global>();
                    }
                } //store switch

            } else { //valid_compute
                first_nbarr.arrive();
                first_nbarr.wait();

                second_nbarr.arrive();
                second_nbarr.wait();
            }

            //QKTV
            int all_vert128_loop = all_vert_loop >> all_vert128_shift;
            if (((((all_vert128_loop + 1) << all_vert128_shift) - 1)
                        == all_vert_loop)
                    || (all_vert128_shift == 0)) {

                third_nbarr.arrive();
                third_nbarr.wait();

                gemm_arguments_128x64 gemm_arg_128x64;
                matAcc_128x64_t matAcc_128x64;
                matC_128x64_t matC_128x64;
                matC_128x64_payload_t matC_128x64_payload;

                uint32_t width_a = tru_seqlen_ex;
                uint32_t height_a = (batchid * numhead + headid) * max_seqlen
                        + tru_seqlen;
                uint32_t pitch_a = max_seqlen;
                int start_x_a = 0;
                int start_y_a = (batchid * numhead + headid) * max_seqlen
                        + all_vert128_loop * 128;

                gemm_arg_128x64.matA_base_desc.init({args->matQKT_ptr},
                        {width_a, height_a, pitch_a}, {start_x_a, start_y_a});

                uint32_t width_b = (headid + 1) * hdsz;
                uint32_t height_b = tru_seqlen + seqlen_entry;
                uint32_t pitch_b = hiddensize;
                int start_x_b = headid * hdsz;
                int start_y_b = seqlen_entry;

                gemm_arg_128x64.matB_base_desc.init({args->matV_ptr},
                        {width_b, height_b, pitch_b}, {start_x_b, start_y_b});

                gemm_arg_128x64.inner_loop_count
                        = (wg_tile_out_k + k_stride - 1) / k_stride;

                matAcc_128x64.init(0);

                gemm_op_128x64_t gemm_op_128x64;

                gemm_op_128x64(g_thd32_tid, matAcc_128x64, gemm_arg_128x64);

                int width_c = (headid + 1) * hdsz;
                int height_c = tru_seqlen + seqlen_entry;
                int pitch_c = hiddensize;
                int start_x_c = headid * hdsz
                        + gemm_op_128x64_t::get_matC_offset_x(g_thd32_tid);
                int start_y_c = all_vert128_loop * 128 + seqlen_entry
                        + gemm_op_128x64_t::get_matC_offset_y(g_thd32_tid);

                matC_128x64_payload.init(args->matOut_ptr, width_c, height_c,
                        pitch_c, start_x_c, start_y_c);
                subgroup::elemwise_cvt<matC_128x64_t, matAcc_128x64_t>(
                        matC_128x64, matAcc_128x64);
                subgroup::tile_store(matC_128x64, matC_128x64_payload);
            }

        } //all_vert128_loop
    } //xetla_softmax_fwd_t::call()
}; //struct xetla_softmax_fwd_t

template <typename dtype_bwd_bin_, typename dtype_bwd_bot_,
        typename dtype_bwd_sfx_, typename dtype_bwd_acc_, int HWThreadNum,
        bool Dopt_RandGenflag = true, bool Mkin_flag = false,
        int Max_SeqLen = 512>
struct xetla_mha_attn_reg_bwd_t {
    using dtype_bin = dtype_bwd_bin_;
    using dtype_bot = dtype_bwd_bot_;
    using dtype_sfx = dtype_bwd_sfx_;
    using dtype_acc = dtype_bwd_acc_;

    static constexpr int ThreadNum = HWThreadNum;
    static_assert(ThreadNum == 32);
    static constexpr mem_space mem_space_a = mem_space::global;
    static constexpr mem_space mem_space_b = mem_space::global;
    static constexpr mem_space mem_space_c = mem_space::global;

    static constexpr mem_layout mem_layout_a = mem_layout::row_major;
    static constexpr mem_layout mem_layout_trnp_a = mem_layout::col_major;
    static constexpr mem_layout mem_layout_QKT_b = mem_layout::col_major;
    static constexpr mem_layout mem_layout_out_b = mem_layout::row_major;
    static constexpr mem_layout mem_layout_c = mem_layout::row_major;

    static constexpr mem_space gemm_mem_space_a = mem_space_a;
    static constexpr mem_space gemm_mem_space_trnp_a = mem_space_a;
    static constexpr mem_layout gemm_mem_layout_a = mem_layout_a;
    static constexpr mem_layout gemm_mem_layout_trnp_a = mem_layout_trnp_a;

    static constexpr mem_space gemm_mem_space_b = mem_space_b;
    static constexpr mem_layout gemm_mem_layout_QKT_b = mem_layout_QKT_b;
    static constexpr mem_layout gemm_mem_layout_out_b = mem_layout_out_b;

    static constexpr uint32_t periodic_sync_interval = 0;
    static constexpr uint32_t prefetch_distance = 3;

    static constexpr uint32_t k_stride
            = 32 / sizeof(dtype_bin); //gemm_t::k_stride;
    using bgm_perf_tuning_knob = group::perf_tuning_knob_t<k_stride,
            prefetch_distance, periodic_sync_interval>;

    using tile_attr_128x128 = group::tile_shape_t<128, 128, 32, 16>;
    using tile_attr_128x256 = group::tile_shape_t<256, 128, 64, 16>;
    using tile_attr_64x384 = group::tile_shape_t<384, 64, 48, 16>;
    using tile_attr_64x512 = group::tile_shape_t<512, 64, 64, 16>;
    using tile_attr_32x1024 = group::tile_shape_t<1024, 32, 64, 16>;
    using tile_attr_16x2048 = group::tile_shape_t<2048, 16, 64, 16>;
    using tile_attr_256x64 = group::tile_shape_t<64, 256, 16, 32>;
    using tile_attr_128x64 = group::tile_shape_t<64, 128, 16, 16>;

    using mem_desc_a_QKT
            = mem_desc_t<dtype_bin, gemm_mem_layout_a, gemm_mem_space_a>;
    using mem_desc_b_QKT
            = mem_desc_t<dtype_bin, gemm_mem_layout_QKT_b, gemm_mem_space_b>;
    using compute_policy_QKT = group::compute_policy_default_xmx<
            group::compute_attr_t<dtype_bin, dtype_bin, dtype_acc>,
            bgm_perf_tuning_knob, gpu_arch::Xe>;

    using mem_desc_a_out
            = mem_desc_t<dtype_sfx, gemm_mem_layout_a, gemm_mem_space_a>;
    using mem_desc_b_out
            = mem_desc_t<dtype_bin, gemm_mem_layout_out_b, gemm_mem_space_b>;
    using compute_policy_out = group::compute_policy_default_xmx<
            group::compute_attr_t<dtype_sfx, dtype_bin, dtype_acc>,
            bgm_perf_tuning_knob, gpu_arch::Xe>;

    using mem_desc_a_out_b_trnp_a = mem_desc_t<dtype_sfx,
            gemm_mem_layout_trnp_a, gemm_mem_space_trnp_a>;
    using mem_desc_b_out_b_trnp_a
            = mem_desc_t<dtype_bin, gemm_mem_layout_out_b, gemm_mem_space_b>;
    using compute_policy_out_b_trnp_a = group::compute_policy_default_xmx<
            group::compute_attr_t<dtype_sfx, dtype_bin, dtype_acc>,
            bgm_perf_tuning_knob, gpu_arch::Xe>;

    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint16_t sfx_type_size = sizeof(dtype_sfx);
    static_assert((sfx_type_size == 1) || (sfx_type_size == 2)
            || (sfx_type_size == 4));

    using work_group_t = work_group_t<ThreadNum>;

    using pre_processing_128x128
            = group::pre_processing_default_t<tile_attr_128x128, gpu_arch::Xe>;
    using pre_processing_128x256
            = group::pre_processing_default_t<tile_attr_128x256, gpu_arch::Xe>;
    using pre_processing_64x384
            = group::pre_processing_default_t<tile_attr_64x384, gpu_arch::Xe>;
    using pre_processing_64x512
            = group::pre_processing_default_t<tile_attr_64x512, gpu_arch::Xe>;
    using pre_processing_32x1024
            = group::pre_processing_default_t<tile_attr_32x1024, gpu_arch::Xe>;
    using pre_processing_16x2048
            = group::pre_processing_default_t<tile_attr_16x2048, gpu_arch::Xe>;
    using pre_processing_128x64
            = group::pre_processing_default_t<tile_attr_128x64, gpu_arch::Xe>;
    using pre_processing_256x64
            = group::pre_processing_default_t<tile_attr_256x64, gpu_arch::Xe>;
    using pre_processing_128x64_af
            = group::pre_processing_matA_neg_filter_t<tile_attr_128x64,
                    gpu_arch::Xe>;
    using pre_processing_256x64_af
            = group::pre_processing_matA_neg_filter_t<tile_attr_256x64,
                    gpu_arch::Xe>;

    using gemm_op_128x128_t
            = group::gemm_t<compute_policy_QKT, tile_attr_128x128,
                    mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_128x128>;
    using gemm_op_128x256_t
            = group::gemm_t<compute_policy_QKT, tile_attr_128x256,
                    mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_128x256>;
    using gemm_op_64x384_t = group::gemm_t<compute_policy_QKT, tile_attr_64x384,
            mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_64x384>;
    using gemm_op_64x512_t = group::gemm_t<compute_policy_QKT, tile_attr_64x512,
            mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_64x512>;
    using gemm_op_32x1024_t
            = group::gemm_t<compute_policy_QKT, tile_attr_32x1024,
                    mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_32x1024>;
    using gemm_op_16x2048_t
            = group::gemm_t<compute_policy_QKT, tile_attr_16x2048,
                    mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_16x2048>;

    using gemm_op_128x64_t = group::gemm_t<compute_policy_out, tile_attr_128x64,
            mem_desc_a_out, mem_desc_b_out, pre_processing_128x64>;
    using gemm_op_128x64_trnp_a_t = group::gemm_t<compute_policy_out_b_trnp_a,
            tile_attr_128x64, mem_desc_a_out_b_trnp_a, mem_desc_b_out_b_trnp_a,
            pre_processing_128x64>;
    using gemm_op_256x64_trnp_a_t = group::gemm_t<compute_policy_out_b_trnp_a,
            tile_attr_256x64, mem_desc_a_out_b_trnp_a, mem_desc_b_out_b_trnp_a,
            pre_processing_256x64>;
    using gemm_op_128x64_trnp_af_t = group::gemm_t<compute_policy_out_b_trnp_a,
            tile_attr_128x64, mem_desc_a_out_b_trnp_a, mem_desc_b_out_b_trnp_a,
            pre_processing_128x64_af>;
    using gemm_op_256x64_trnp_af_t = group::gemm_t<compute_policy_out_b_trnp_a,
            tile_attr_256x64, mem_desc_a_out_b_trnp_a, mem_desc_b_out_b_trnp_a,
            pre_processing_256x64_af>;

    using gemm_arguments_128x128 = typename gemm_op_128x128_t::arguments_t;
    using gemm_arguments_128x256 = typename gemm_op_128x256_t::arguments_t;
    using gemm_arguments_64x384 = typename gemm_op_64x384_t::arguments_t;
    using gemm_arguments_64x512 = typename gemm_op_64x512_t::arguments_t;
    using gemm_arguments_32x1024 = typename gemm_op_32x1024_t::arguments_t;
    using gemm_arguments_16x2048 = typename gemm_op_16x2048_t::arguments_t;

    using gemm_arguments_128x64 = typename gemm_op_128x64_t::arguments_t;
    using gemm_arguments_128x64_trnp_a =
            typename gemm_op_128x64_trnp_a_t::arguments_t;
    using gemm_arguments_256x64_trnp_a =
            typename gemm_op_256x64_trnp_a_t::arguments_t;
    using gemm_arguments_128x64_trnp_af =
            typename gemm_op_128x64_trnp_af_t::arguments_t;
    using gemm_arguments_256x64_trnp_af =
            typename gemm_op_256x64_trnp_af_t::arguments_t;

    using matAcc_128x128_t = typename gemm_op_128x128_t::matAcc_t;
    using matAcc_128x256_t = typename gemm_op_128x256_t::matAcc_t;
    using matAcc_64x384_t = typename gemm_op_64x384_t::matAcc_t;
    using matAcc_64x512_t = typename gemm_op_64x512_t::matAcc_t;
    using matAcc_32x1024_t = typename gemm_op_32x1024_t::matAcc_t;
    using matAcc_16x2048_t = typename gemm_op_16x2048_t::matAcc_t;

    using matAcc_128x64_t = typename gemm_op_128x64_t::matAcc_t;
    using matAcc_128x64_trnp_a_t = typename gemm_op_128x64_trnp_a_t::matAcc_t;
    using matAcc_256x64_trnp_a_t = typename gemm_op_256x64_trnp_a_t::matAcc_t;
    using matAcc_128x64_trnp_af_t = typename gemm_op_128x64_trnp_af_t::matAcc_t;
    using matAcc_256x64_trnp_af_t = typename gemm_op_256x64_trnp_af_t::matAcc_t;

    using matC_128x128_tile_desc_t
            = subgroup::tile_desc_t<matAcc_128x128_t::tile_desc::tile_size_x,
                    matAcc_128x128_t::tile_desc::tile_size_y,
                    matAcc_128x128_t::tile_desc::block_size_x,
                    matAcc_128x128_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using matC_128x256_tile_desc_t
            = subgroup::tile_desc_t<matAcc_128x256_t::tile_desc::tile_size_x,
                    matAcc_128x256_t::tile_desc::tile_size_y,
                    matAcc_128x256_t::tile_desc::block_size_x,
                    matAcc_128x256_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using matC_64x384_tile_desc_t
            = subgroup::tile_desc_t<matAcc_64x384_t::tile_desc::tile_size_x,
                    matAcc_64x384_t::tile_desc::tile_size_y,
                    matAcc_64x384_t::tile_desc::block_size_x,
                    matAcc_64x384_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using matC_64x512_tile_desc_t
            = subgroup::tile_desc_t<matAcc_64x512_t::tile_desc::tile_size_x,
                    matAcc_64x512_t::tile_desc::tile_size_y,
                    matAcc_64x512_t::tile_desc::block_size_x,
                    matAcc_64x512_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using matC_32x1024_tile_desc_t
            = subgroup::tile_desc_t<matAcc_32x1024_t::tile_desc::tile_size_x,
                    matAcc_32x1024_t::tile_desc::tile_size_y,
                    matAcc_32x1024_t::tile_desc::block_size_x,
                    matAcc_32x1024_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using matC_16x2048_tile_desc_t
            = subgroup::tile_desc_t<matAcc_16x2048_t::tile_desc::tile_size_x,
                    matAcc_16x2048_t::tile_desc::tile_size_y,
                    matAcc_16x2048_t::tile_desc::block_size_x,
                    matAcc_16x2048_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using matC_128x128_t
            = subgroup::tile_t<dtype_sfx, matC_128x128_tile_desc_t>;
    using matC_128x256_t
            = subgroup::tile_t<dtype_sfx, matC_128x256_tile_desc_t>;
    using matC_64x384_t = subgroup::tile_t<dtype_sfx, matC_64x384_tile_desc_t>;
    using matC_64x512_t = subgroup::tile_t<dtype_sfx, matC_64x512_tile_desc_t>;
    using matC_32x1024_t
            = subgroup::tile_t<dtype_sfx, matC_32x1024_tile_desc_t>;
    using matC_16x2048_t
            = subgroup::tile_t<dtype_sfx, matC_16x2048_tile_desc_t>;

    using matC_128x128_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_128x128_tile_desc_t,
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<matC_128x128_tile_desc_t,
                            mem_space_c>,
            gpu_arch::Xe>;
    using matC_128x256_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_128x256_tile_desc_t,
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<matC_128x256_tile_desc_t,
                            mem_space_c>,
            gpu_arch::Xe>;
    using matC_64x384_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_64x384_tile_desc_t,
            (global_kslicing > 1) ? msg_type::atomic_add
                                  : subgroup::msg_type_v<
                                          matC_64x384_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matC_64x512_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_64x512_tile_desc_t,
            (global_kslicing > 1) ? msg_type::atomic_add
                                  : subgroup::msg_type_v<
                                          matC_64x512_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matC_32x1024_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_32x1024_tile_desc_t,
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<matC_32x1024_tile_desc_t,
                            mem_space_c>,
            gpu_arch::Xe>;
    using matC_16x2048_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_16x2048_tile_desc_t,
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<matC_16x2048_tile_desc_t,
                            mem_space_c>,
            gpu_arch::Xe>;

    using matC_128x64_tile_desc_t
            = subgroup::tile_desc_t<matAcc_128x64_t::tile_desc::tile_size_x,
                    matAcc_128x64_t::tile_desc::tile_size_y,
                    matAcc_128x64_t::tile_desc::block_size_x,
                    matAcc_128x64_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using matC_128x64_trnp_a_tile_desc_t = subgroup::tile_desc_t<
            matAcc_128x64_trnp_a_t::tile_desc::tile_size_x,
            matAcc_128x64_trnp_a_t::tile_desc::tile_size_y,
            matAcc_128x64_trnp_a_t::tile_desc::block_size_x,
            matAcc_128x64_trnp_a_t::tile_desc::block_size_y, reg_layout::tiled>;
    using matC_256x64_trnp_a_tile_desc_t = subgroup::tile_desc_t<
            matAcc_256x64_trnp_a_t::tile_desc::tile_size_x,
            matAcc_256x64_trnp_a_t::tile_desc::tile_size_y,
            matAcc_256x64_trnp_a_t::tile_desc::block_size_x,
            matAcc_256x64_trnp_a_t::tile_desc::block_size_y, reg_layout::tiled>;
    using matC_128x64_trnp_af_tile_desc_t = subgroup::tile_desc_t<
            matAcc_128x64_trnp_af_t::tile_desc::tile_size_x,
            matAcc_128x64_trnp_af_t::tile_desc::tile_size_y,
            matAcc_128x64_trnp_af_t::tile_desc::block_size_x,
            matAcc_128x64_trnp_af_t::tile_desc::block_size_y,
            reg_layout::tiled>;
    using matC_256x64_trnp_af_tile_desc_t = subgroup::tile_desc_t<
            matAcc_256x64_trnp_af_t::tile_desc::tile_size_x,
            matAcc_256x64_trnp_af_t::tile_desc::tile_size_y,
            matAcc_256x64_trnp_af_t::tile_desc::block_size_x,
            matAcc_256x64_trnp_af_t::tile_desc::block_size_y,
            reg_layout::tiled>;
    using matC_128x64_t = subgroup::tile_t<dtype_bot, matC_128x64_tile_desc_t>;
    using matC_128x64_trnp_a_t
            = subgroup::tile_t<dtype_bot, matC_128x64_trnp_a_tile_desc_t>;
    using matC_256x64_trnp_a_t
            = subgroup::tile_t<dtype_bot, matC_256x64_trnp_a_tile_desc_t>;
    using matC_128x64_trnp_af_t
            = subgroup::tile_t<dtype_bot, matC_128x64_trnp_af_tile_desc_t>;
    using matC_256x64_trnp_af_t
            = subgroup::tile_t<dtype_bot, matC_256x64_trnp_af_tile_desc_t>;

    using matC_128x64_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_bot, mem_layout_c, mem_space_c>,
            matC_128x64_tile_desc_t,
            (global_kslicing > 1) ? msg_type::atomic_add
                                  : subgroup::msg_type_v<
                                          matC_128x64_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matC_128x64_trnp_a_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_bot, mem_layout_c, mem_space_c>,
            matC_128x64_trnp_a_tile_desc_t,
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<matC_128x64_trnp_a_tile_desc_t,
                            mem_space_c>,
            gpu_arch::Xe>;
    using matC_256x64_trnp_a_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_bot, mem_layout_c, mem_space_c>,
            matC_256x64_trnp_a_tile_desc_t,
            subgroup::msg_type_v<matC_256x64_trnp_a_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matC_128x64_trnp_af_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_bot, mem_layout_c, mem_space_c>,
            matC_128x64_trnp_af_tile_desc_t,
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<matC_128x64_trnp_af_tile_desc_t,
                            mem_space_c>,
            gpu_arch::Xe>;
    using matC_256x64_trnp_af_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_bot, mem_layout_c, mem_space_c>,
            matC_256x64_trnp_af_tile_desc_t,
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<matC_256x64_trnp_af_tile_desc_t,
                            mem_space_c>,
            gpu_arch::Xe>;

    using matW_128x128_t
            = subgroup::tile_t<dtype_sfx, matC_128x128_tile_desc_t>;
    using matW_128x256_t
            = subgroup::tile_t<dtype_sfx, matC_128x256_tile_desc_t>;
    using matW_64x384_t = subgroup::tile_t<dtype_sfx, matC_64x384_tile_desc_t>;
    using matW_64x512_t = subgroup::tile_t<dtype_sfx, matC_64x512_tile_desc_t>;
    using matW_32x1024_t
            = subgroup::tile_t<dtype_sfx, matC_32x1024_tile_desc_t>;
    using matW_16x2048_t
            = subgroup::tile_t<dtype_sfx, matC_16x2048_tile_desc_t>;

    using matW_128x128_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_128x128_tile_desc_t,
            subgroup::msg_type_v<matC_128x128_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matW_128x256_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_128x256_tile_desc_t,
            subgroup::msg_type_v<matC_128x256_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matW_64x384_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_64x384_tile_desc_t,
            subgroup::msg_type_v<matC_64x384_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matW_64x512_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_64x512_tile_desc_t,
            subgroup::msg_type_v<matC_64x512_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matW_32x1024_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_32x1024_tile_desc_t,
            subgroup::msg_type_v<matC_32x1024_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;
    using matW_16x2048_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_16x2048_tile_desc_t,
            subgroup::msg_type_v<matC_16x2048_tile_desc_t, mem_space_c>,
            gpu_arch::Xe>;

#if 0
    //512 = 16x32 or 8x64
    using matElem_ld_t = subgroup::tile_t<dtype_sfx, matElem_tile_desc>;
    using matElem_st_t = subgroup::tile_t<dtype_sfx, matElem_tile_desc>;
    using matElem_ld_payload_t = subgroup::mem_payload_t<mem_desc_t<dtype_sfx, mem_space::global, mem_layout::row_major>,
           matElem_tile_desc,
           subgroup::msg_type_v<matElem_tile_desc, mem_space::global>>;
    using matElem_st_payload_t = subgroup::mem_payload_t<mem_desc_t<dtype_sfx, mem_space::global, mem_layout::row_major>, 
           matElem_tile_desc,
           msg_type::block_2d>;
#endif

    /// @brief Arguments for xetla_softmax_bwd_t::run.
    /// User should prepare matQ_ptr, matK_ptr, matV_ptr, ...
    struct arguments_t {
        // assume base address, surface width, height, pitch, start coordinate was set
        uint32_t *mList_ptr;
        dtype_bin *matQ_ptr;
        dtype_bin *matK_ptr;
        dtype_bin *matV_ptr;
        uint32_t *matMkin_ptr;
        uint32_t *matMkdpot_ptr;
        dtype_bin *matdO_ptr;
        dtype_sfx *matW_ptr;
        dtype_sfx *matdW_ptr;
        dtype_bot *matdV_ptr;
        dtype_bot *matdQ_ptr;
        dtype_bot *matdK_ptr;
        float *matSum_ptr;
        float Pinv;
        float Scaling;
    };

    /// @brief Main execution function for fused mha softmax
    /// The basic process is GEMM -> Softmax -> GEMM.
    /// @param args [in] Includes base descriptors and tid info.
    __XETLA_API static void call(sycl::nd_item<3> &item, arguments_t *args) {

        int tru_seqlen = 0;
        int tru_seqlen_ex = 0;
        int seqlen_entry = 0;
        int hiddensize = 1024;
        int numhead = 16;
        int hdsz = 64;
        int max_seqlen = Max_SeqLen;
        int wg_tile_QKT_k = hdsz; //args->matrix_k;
        int wg_tile_out_k;

        int groupid = item.get_group(0);
        int batchid = groupid / numhead;
        int headid = groupid % numhead;

        work_group_t g_thd32_tid;
        int tid_linear = item.get_local_linear_id();
        g_thd32_tid.init(tid_linear);

        //float totalscaling = args->Pinv * args->Scaling;

        uint32_t batch_offset = sizeof(uint32_t) * list_width * batchid;
        xetla_vector<uint32_t, list_width> list_offsets
                = xetla_vector_gen<uint32_t, list_width>(0, 1);
        list_offsets *= sizeof(uint32_t);
        list_offsets += batch_offset;
        xetla_vector<uint32_t, list_width> list_vec
                = xetla_load_global<uint32_t, 1, data_size::default_size,
                        cache_hint::read_invalidate, cache_hint::cached,
                        list_width>(args->mList_ptr, list_offsets);
        tru_seqlen = list_vec[0];
        seqlen_entry = list_vec[1];
        wg_tile_out_k = tru_seqlen;
        tru_seqlen_ex = tru_seqlen; //4: dw aligned
        if (sfx_type_size == 2)
            tru_seqlen_ex = ((tru_seqlen + 1) >> 1) << 1;
        else if (sfx_type_size == 1)
            tru_seqlen_ex = ((tru_seqlen + 3) >> 2) << 2;

        //reset for all std_seqlen
        int all_vert_loop_num = 0;
        int transp128_loop_num = 0;
        int transp256_loop_num = 0;
        int offset_blk_128x128 = 0;
        int all_vert_stride = 0;
        int all_vert128_shift = 0;
        int block_16x16_num = 0;
        int tid_x_shift = 0;

        int std_seqlen;
        if (tru_seqlen <= 128) {
            std_seqlen = 128;
            all_vert_loop_num = 1;
            transp128_loop_num = 1;
            tid_x_shift = 2; //        16x32 128/32 = 4
            all_vert_loop_num = 1;
            all_vert_stride = 128;
            block_16x16_num = 2;
        } else if (tru_seqlen <= 256) {
            std_seqlen = 256;
            transp256_loop_num = 1;
            all_vert_loop_num = 2;
            all_vert_stride = 128;
            all_vert128_shift = 0;
            block_16x16_num = 4;
            tid_x_shift = 2; //        16x64 256/64 = 4
        } else if (tru_seqlen <= 384) {
            std_seqlen = 384;
            transp128_loop_num = 1;
            transp256_loop_num = 1;
            offset_blk_128x128 = 256;
            all_vert_stride = 64;
            all_vert128_shift = 1;
            block_16x16_num = 3;
            tid_x_shift = 3; //        16x48 384/48 = 8
            all_vert_loop_num = (tru_seqlen + all_vert_stride - 1) >> 6;
        } else if (tru_seqlen <= 512) {
            std_seqlen = 512;
            transp256_loop_num = 2;
            all_vert_stride = 64;
            all_vert128_shift = 1;
            block_16x16_num = 4;
            tid_x_shift = 3; //        16x64 512/64 = 8
            all_vert_loop_num = (tru_seqlen + all_vert_stride - 1) >> 6;
        } else if (tru_seqlen <= 1024) {
            std_seqlen = 1024;
            transp256_loop_num = 4;
            all_vert_stride = 32;
            all_vert128_shift = 2;
            block_16x16_num = 4;
            tid_x_shift = 4; //        16x64 1024/64 = 16
            all_vert_loop_num = (tru_seqlen + all_vert_stride - 1) >> 5;
        } else if (tru_seqlen <= 2048) {
            std_seqlen = 2048;
            transp256_loop_num = 8;
            all_vert_stride = 16;
            all_vert128_shift = 3;
            block_16x16_num = 4;
            tid_x_shift = 5; //        16x64 2048/64 = 32
            all_vert_loop_num = (tru_seqlen + all_vert_stride - 1) >> 4;
        }
        all_vert_loop_num = ((all_vert_loop_num + (1 << all_vert128_shift) - 1)
                                    >> all_vert128_shift)
                << all_vert128_shift;
        int tid_x = tid_linear & ((1 << tid_x_shift) - 1);
        int tid_y = tid_linear >> tid_x_shift;

        static_assert(ThreadNum == 32, "All Thread Sync");
        xetla_nbarrier_t<ThreadNum, ThreadNum, gpu_arch::Xe> first_nbarr;
        xetla_nbarrier_t<ThreadNum, ThreadNum, gpu_arch::Xe> second_nbarr;

        int max_2d_nbar_id = ThreadNum >> 1;
        first_nbarr.init_nbarrier(
                max_2d_nbar_id, nbarrier_role::producer_consumer);
        second_nbarr.init_nbarrier(
                max_2d_nbar_id + 1, nbarrier_role::producer_consumer);

        xetla_nbarrier_t<ThreadNum, ThreadNum, gpu_arch::Xe> all_nbarr;
        all_nbarr.init_nbarrier(
                ThreadNum - 1, nbarrier_role::producer_consumer);

        for (int transp128_loop = 0; transp128_loop < transp128_loop_num;
                transp128_loop++) {
            gemm_arguments_128x64_trnp_af gemm_arg_128x64;
            matAcc_128x64_trnp_af_t matAcc_128x64;
            matC_128x64_trnp_af_t matC_128x64;
            matC_128x64_trnp_af_payload_t matC_128x64_payload;

            uint32_t width_a = tru_seqlen_ex;
            uint32_t height_a
                    = (batchid * numhead + headid) * max_seqlen + tru_seqlen;
            uint32_t pitch_a = max_seqlen;
            int start_x_a = transp128_loop * 128 + offset_blk_128x128;
            int start_y_a = (batchid * numhead + headid) * max_seqlen;

            gemm_arg_128x64.matA_base_desc.init({args->matW_ptr},
                    {height_a, width_a, pitch_a}, {start_y_a, start_x_a});

            uint32_t width_b = (headid + 1) * hdsz;
            uint32_t height_b = tru_seqlen + seqlen_entry;
            uint32_t pitch_b = hiddensize;
            int start_x_b = headid * hdsz;
            int start_y_b = seqlen_entry;

            gemm_arg_128x64.matB_base_desc.init({args->matdO_ptr},
                    {width_b, height_b, pitch_b}, {start_x_b, start_y_b});

            gemm_arg_128x64.inner_loop_count
                    = (wg_tile_out_k + k_stride - 1) / k_stride;

            matAcc_128x64.init(0);
            gemm_op_128x64_trnp_af_t gemm_op_128x64_trnp_af;
            gemm_op_128x64_trnp_af(g_thd32_tid, matAcc_128x64, gemm_arg_128x64);

            int width_c = (headid + 1) * hdsz;
            int height_c = tru_seqlen + seqlen_entry;
            int pitch_c = hiddensize;
            int start_x_c = headid * hdsz
                    + gemm_op_128x64_trnp_af_t::get_matC_offset_x(g_thd32_tid);
            int start_y_c = transp128_loop * 128 + seqlen_entry
                    + offset_blk_128x128
                    + gemm_op_128x64_trnp_af_t::get_matC_offset_y(g_thd32_tid);

            matC_128x64_payload.init(args->matdV_ptr, width_c, height_c,
                    pitch_c, start_x_c, start_y_c);
            subgroup::elemwise_cvt<matC_128x64_trnp_af_t,
                    matAcc_128x64_trnp_af_t>(matC_128x64, matAcc_128x64);
            subgroup::tile_store(matC_128x64, matC_128x64_payload);

            //add global sync if nbarr used inside gemm
            all_nbarr.arrive();
            all_nbarr.wait();
        }

        for (int transp256_loop = 0; transp256_loop < transp256_loop_num;
                transp256_loop++) {
            gemm_arguments_256x64_trnp_af gemm_arg_256x64;
            matAcc_256x64_trnp_af_t matAcc_256x64;
            matC_256x64_trnp_af_t matC_256x64;
            matC_256x64_trnp_af_payload_t matC_256x64_payload;

            uint32_t width_a = tru_seqlen_ex;
            uint32_t height_a
                    = (batchid * numhead + headid) * max_seqlen + tru_seqlen;
            uint32_t pitch_a = max_seqlen;
            int start_x_a = transp256_loop * 256;
            int start_y_a = (batchid * numhead + headid) * max_seqlen;

            gemm_arg_256x64.matA_base_desc.init({args->matW_ptr},
                    {height_a, width_a, pitch_a}, {start_y_a, start_x_a});

            uint32_t width_b = (headid + 1) * hdsz;
            uint32_t height_b = tru_seqlen + seqlen_entry;
            uint32_t pitch_b = hiddensize;
            int start_x_b = headid * hdsz;
            int start_y_b = seqlen_entry;

            gemm_arg_256x64.matB_base_desc.init({args->matdO_ptr},
                    {width_b, height_b, pitch_b}, {start_x_b, start_y_b});

            gemm_arg_256x64.inner_loop_count
                    = (wg_tile_out_k + k_stride - 1) / k_stride;

            matAcc_256x64.init(0);
            gemm_op_256x64_trnp_af_t gemm_op_256x64_trnp_af;
            gemm_op_256x64_trnp_af(g_thd32_tid, matAcc_256x64, gemm_arg_256x64);

            int width_c = (headid + 1) * hdsz;
            int height_c = tru_seqlen + seqlen_entry;
            int pitch_c = hiddensize;
            int start_x_c = headid * hdsz
                    + gemm_op_256x64_trnp_af_t::get_matC_offset_x(g_thd32_tid);
            int start_y_c = transp256_loop * 256 + seqlen_entry
                    + gemm_op_256x64_trnp_af_t::get_matC_offset_y(g_thd32_tid);

            matC_256x64_payload.init(args->matdV_ptr, width_c, height_c,
                    pitch_c, start_x_c, start_y_c);
            subgroup::elemwise_cvt<matC_256x64_trnp_af_t,
                    matAcc_256x64_trnp_af_t>(matC_256x64, matAcc_256x64);
            subgroup::tile_store(matC_256x64, matC_256x64_payload);

            //add global sync if nbarr used inside gemm
            all_nbarr.arrive();
            all_nbarr.wait();
        }

        int valid_block_16x16_x = (tid_x + 1) * 16 * block_16x16_num;
        {
            int bndy_block_num = 0;
            if (valid_block_16x16_x <= tru_seqlen)
                valid_block_16x16_x = block_16x16_num;
            else {
                bndy_block_num = valid_block_16x16_x;
                valid_block_16x16_x = (tru_seqlen + 15 + 16 * block_16x16_num
                                              - valid_block_16x16_x)
                        >> 4;
                bndy_block_num = bndy_block_num
                        + (valid_block_16x16_x - block_16x16_num) * 16
                        - tru_seqlen;
            }
        }

        for (int all_vert_loop = 0; all_vert_loop < all_vert_loop_num;
                all_vert_loop++) {
            xetla_vector<float, 4 * 16 * 16> matElem_reg_4x16x16;
            xetla_vector<float, 4 * 16 * 16> matW_reg_4x16x16;
            xetla_vector<uint8_t, 4 * 16 * 16> Sign_reg_4x16x16 = 0;
            bool valid_compute = true;

            int ld_st_width_c = max_seqlen;
            int ld_st_height_c = max_seqlen * (batchid * numhead + headid + 1);
            int ld_st_pitch_c = max_seqlen;
            int ld_st_start_x_c = 0;
            int ld_st_start_y_c = 0;

            if (((all_vert_loop * all_vert_stride + tid_y * 16) >= tru_seqlen)
                    || ((tid_x * 16 * block_16x16_num) >= tru_seqlen))
                valid_compute = false;

            if (valid_compute) {

                switch (std_seqlen) {
                    case 128: {
                        gemm_arguments_128x128 gemm_arg_128x128;
                        matAcc_128x128_t matAcc_128x128;

                        matW_128x128_t matW_128x128;
                        matW_128x128_payload_t matW_128x128_payload;

                        ld_st_start_x_c = gemm_op_128x128_t::get_matC_offset_x(
                                g_thd32_tid);
                        ld_st_start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_128x128_t::get_matC_offset_y(
                                        g_thd32_tid);

                        matW_128x128_payload.init(args->matW_ptr, ld_st_width_c,
                                ld_st_height_c, ld_st_pitch_c, ld_st_start_x_c,
                                ld_st_start_y_c);
                        subgroup::tile_load(matW_128x128, matW_128x128_payload);

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_128x128.matA_base_desc.init({args->matdO_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose, be swapped in init
                        gemm_arg_128x128.matB_base_desc.init({args->matV_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_128x128.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_128x128.init(0);
                        gemm_op_128x128_t gemm_op_128x128;
                        gemm_op_128x128(
                                g_thd32_tid, matAcc_128x128, gemm_arg_128x128);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<16 * 32, 1>(0)
                                = matAcc_128x128.reg;

                        if constexpr (sfx_type_size == 1) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 2, 1>(0)
                                    .merge(1,
                                            matW_128x128.reg.xetla_format<
                                                    int8_t>()
                                                    < 0);
                            matW_128x128.reg.xetla_format<uint8_t>() &= 0x7F;
                        }
                        if constexpr (sfx_type_size == 2) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 2, 1>(0)
                                    .merge(1,
                                            matW_128x128.reg.xetla_format<
                                                    int16_t>()
                                                    < 0);
                            matW_128x128.reg.xetla_format<uint16_t>() &= 0x7FFF;
                        }

                        xetla_vector<float, 2 * 16 * 16> matElem_reg_conv;
                        matElem_reg_conv
                                = xetla_cvt<float, dtype_sfx>(matW_128x128.reg);
                        matW_reg_4x16x16.xetla_select<16 * 16 * 2, 1>(0)
                                = matElem_reg_conv;
                    } break;

                    case 256: {
                        gemm_arguments_128x256 gemm_arg_128x256;
                        matAcc_128x256_t matAcc_128x256;

                        matW_128x256_t matW_128x256;
                        matW_128x256_payload_t matW_128x256_payload;

                        ld_st_start_x_c = gemm_op_128x256_t::get_matC_offset_x(
                                g_thd32_tid);
                        ld_st_start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_128x256_t::get_matC_offset_y(
                                        g_thd32_tid);

                        matW_128x256_payload.init(args->matW_ptr, ld_st_width_c,
                                ld_st_height_c, ld_st_pitch_c, ld_st_start_x_c,
                                ld_st_start_y_c);
                        subgroup::tile_load(matW_128x256, matW_128x256_payload);

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_128x256.matA_base_desc.init({args->matdO_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose, be swapped in init
                        gemm_arg_128x256.matB_base_desc.init({args->matV_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_128x256.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_128x256.init(0);
                        gemm_op_128x256_t gemm_op_128x256;
                        gemm_op_128x256(
                                g_thd32_tid, matAcc_128x256, gemm_arg_128x256);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<16 * 16 * 4, 1>(0)
                                = matAcc_128x256.reg;

                        if constexpr (sfx_type_size == 1) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                    .merge(1,
                                            matW_128x256.reg.xetla_format<
                                                    int8_t>()
                                                    < 0);
                            matW_128x256.reg.xetla_format<uint8_t>() &= 0x7F;
                        }
                        if constexpr (sfx_type_size == 2) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                    .merge(1,
                                            matW_128x256.reg.xetla_format<
                                                    int16_t>()
                                                    < 0);
                            matW_128x256.reg.xetla_format<uint16_t>() &= 0x7FFF;
                        }

                        matW_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                = xetla_cvt<float, dtype_sfx>(matW_128x256.reg);
                    } break;

                    case 384: {
                        gemm_arguments_64x384 gemm_arg_64x384;
                        matAcc_64x384_t matAcc_64x384;

                        matW_64x384_t matW_64x384;
                        matW_64x384_payload_t matW_64x384_payload;

                        ld_st_start_x_c = gemm_op_64x384_t::get_matC_offset_x(
                                g_thd32_tid);
                        ld_st_start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_64x384_t::get_matC_offset_y(
                                        g_thd32_tid);
                        matW_64x384_payload.init(args->matW_ptr, ld_st_width_c,
                                ld_st_height_c, ld_st_pitch_c, ld_st_start_x_c,
                                ld_st_start_y_c);
                        subgroup::tile_load(matW_64x384, matW_64x384_payload);

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_64x384.matA_base_desc.init({args->matdO_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose, be swapped in init
                        gemm_arg_64x384.matB_base_desc.init({args->matV_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_64x384.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_64x384.init(0);
                        gemm_op_64x384_t gemm_op_64x384;
                        gemm_op_64x384(
                                g_thd32_tid, matAcc_64x384, gemm_arg_64x384);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<16 * 16 * 3, 1>(0)
                                = matAcc_64x384.reg;

                        if constexpr (sfx_type_size == 1) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 3, 1>(0)
                                    .merge(1,
                                            matW_64x384.reg.xetla_format<
                                                    int8_t>()
                                                    < 0);
                            matW_64x384.reg.xetla_format<uint8_t>() &= 0x7F;
                        }
                        if constexpr (sfx_type_size == 2) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 3, 1>(0)
                                    .merge(1,
                                            matW_64x384.reg.xetla_format<
                                                    int16_t>()
                                                    < 0);
                            matW_64x384.reg.xetla_format<uint16_t>() &= 0x7FFF;
                        }

                        xetla_vector<float, 3 * 16 * 16> matElem_reg_conv;
                        matElem_reg_conv
                                = xetla_cvt<float, dtype_sfx>(matW_64x384.reg);
                        matW_reg_4x16x16.xetla_select<16 * 16 * 3, 1>(0)
                                = matElem_reg_conv;
                    } break;

                    case 512: {
                        gemm_arguments_64x512 gemm_arg_64x512;
                        matAcc_64x512_t matAcc_64x512;

                        matW_64x512_t matW_64x512;
                        matW_64x512_payload_t matW_64x512_payload;

                        ld_st_start_x_c = gemm_op_64x512_t::get_matC_offset_x(
                                g_thd32_tid);
                        ld_st_start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_64x512_t::get_matC_offset_y(
                                        g_thd32_tid);
                        matW_64x512_payload.init(args->matW_ptr, ld_st_width_c,
                                ld_st_height_c, ld_st_pitch_c, ld_st_start_x_c,
                                ld_st_start_y_c);
                        subgroup::tile_load(matW_64x512, matW_64x512_payload);

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_64x512.matA_base_desc.init({args->matdO_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose, be swapped in init
                        gemm_arg_64x512.matB_base_desc.init({args->matV_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_64x512.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_64x512.init(0);
                        gemm_op_64x512_t gemm_op_64x512;
                        gemm_op_64x512(
                                g_thd32_tid, matAcc_64x512, gemm_arg_64x512);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<16 * 16 * 4, 1>(0)
                                = matAcc_64x512.reg;

                        if constexpr (sfx_type_size == 1) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                    .merge(1,
                                            matW_64x512.reg.xetla_format<
                                                    int8_t>()
                                                    < 0);
                            matW_64x512.reg.xetla_format<uint8_t>() &= 0x7F;
                        }
                        if constexpr (sfx_type_size == 2) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                    .merge(1,
                                            matW_64x512.reg.xetla_format<
                                                    int16_t>()
                                                    < 0);
                            matW_64x512.reg.xetla_format<uint16_t>() &= 0x7FFF;
                        }

                        matW_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                = xetla_cvt<float, dtype_sfx>(matW_64x512.reg);
                    } break;

                    case 1024: {
                        gemm_arguments_32x1024 gemm_arg_32x1024;
                        matAcc_32x1024_t matAcc_32x1024;

                        matW_32x1024_t matW_32x1024;
                        matW_32x1024_payload_t matW_32x1024_payload;

                        ld_st_start_x_c = gemm_op_32x1024_t::get_matC_offset_x(
                                g_thd32_tid);
                        ld_st_start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_32x1024_t::get_matC_offset_y(
                                        g_thd32_tid);
                        matW_32x1024_payload.init(args->matW_ptr, ld_st_width_c,
                                ld_st_height_c, ld_st_pitch_c, ld_st_start_x_c,
                                ld_st_start_y_c);
                        subgroup::tile_load(matW_32x1024, matW_32x1024_payload);

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_32x1024.matA_base_desc.init({args->matdO_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose, be swapped in init
                        gemm_arg_32x1024.matB_base_desc.init({args->matV_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_32x1024.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_32x1024.init(0);
                        gemm_op_32x1024_t gemm_op_32x1024;
                        gemm_op_32x1024(
                                g_thd32_tid, matAcc_32x1024, gemm_arg_32x1024);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<16 * 16 * 4, 1>(0)
                                = matAcc_32x1024.reg;

                        if constexpr (sfx_type_size == 1) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                    .merge(1,
                                            matW_32x1024.reg.xetla_format<
                                                    int8_t>()
                                                    < 0);
                            matW_32x1024.reg.xetla_format<uint8_t>() &= 0x7F;
                        }
                        if constexpr (sfx_type_size == 2) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                    .merge(1,
                                            matW_32x1024.reg.xetla_format<
                                                    int16_t>()
                                                    < 0);
                            matW_32x1024.reg.xetla_format<uint16_t>() &= 0x7FFF;
                        }

                        matW_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                = xetla_cvt<float, dtype_sfx>(matW_32x1024.reg);
                    } break;

                    case 2048: {
                        gemm_arguments_16x2048 gemm_arg_16x2048;
                        matAcc_16x2048_t matAcc_16x2048;

                        matW_16x2048_t matW_16x2048;
                        matW_16x2048_payload_t matW_16x2048_payload;

                        ld_st_start_x_c = gemm_op_16x2048_t::get_matC_offset_x(
                                g_thd32_tid);
                        ld_st_start_y_c
                                = (batchid * numhead + headid) * max_seqlen
                                + all_vert_loop * all_vert_stride
                                + gemm_op_16x2048_t::get_matC_offset_y(
                                        g_thd32_tid);
                        matW_16x2048_payload.init(args->matW_ptr, ld_st_width_c,
                                ld_st_height_c, ld_st_pitch_c, ld_st_start_x_c,
                                ld_st_start_y_c);
                        subgroup::tile_load(matW_16x2048, matW_16x2048_payload);

                        uint32_t width_a = (headid + 1) * hdsz;
                        uint32_t height_a = tru_seqlen + seqlen_entry;
                        uint32_t pitch_a = hiddensize;
                        int start_x_a = headid * hdsz;
                        int start_y_a = all_vert_loop * all_vert_stride
                                + seqlen_entry;

                        gemm_arg_16x2048.matA_base_desc.init({args->matdO_ptr},
                                {width_a, height_a, pitch_a},
                                {start_x_a, start_y_a});

                        uint32_t width_b = (headid + 1) * hdsz;
                        uint32_t height_b = tru_seqlen + seqlen_entry;
                        uint32_t pitch_b = hiddensize;
                        int start_x_b = headid * hdsz;
                        int start_y_b = seqlen_entry;

                        //B transpose, be swapped in init
                        gemm_arg_16x2048.matB_base_desc.init({args->matV_ptr},
                                {height_b, width_b, pitch_b},
                                {start_y_b, start_x_b});

                        gemm_arg_16x2048.inner_loop_count
                                = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                        matAcc_16x2048.init(0);
                        gemm_op_16x2048_t gemm_op_16x2048;
                        gemm_op_16x2048(
                                g_thd32_tid, matAcc_16x2048, gemm_arg_16x2048);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<16 * 16 * 4, 1>(0)
                                = matAcc_16x2048.reg;

                        if constexpr (sfx_type_size == 1) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                    .merge(1,
                                            matW_16x2048.reg.xetla_format<
                                                    int8_t>()
                                                    < 0);
                            matW_16x2048.reg.xetla_format<uint8_t>() &= 0x7F;
                        }
                        if constexpr (sfx_type_size == 2) {
                            Sign_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                    .merge(1,
                                            matW_16x2048.reg.xetla_format<
                                                    int16_t>()
                                                    < 0);
                            matW_16x2048.reg.xetla_format<uint16_t>() &= 0x7FFF;
                        }

                        matW_reg_4x16x16.xetla_select<16 * 16 * 4, 1>(0)
                                = xetla_cvt<float, dtype_sfx>(matW_16x2048.reg);
                    } break;
                } //switch

                //softmax
                {

                    xetla_vector<float, 16 * 1> matElem_reg_Sum_1;
                    xetla_vector<float, 16 * 16> matElem_reg_Sum;
                    xetla_vector<float, 16 * 8> matElem_reg_Sum_8;
                    xetla_vector<float, 16 * 4> matElem_reg_Sum_4;
                    xetla_vector<float, 16 * 2> matElem_reg_Sum_2;

                    matElem_reg_4x16x16.xetla_format<float>()
                            .xetla_select<16 * 16 * 2, 1>(0)
                            *= matW_reg_4x16x16.xetla_select<16 * 16 * 2, 1>(0);

                    matElem_reg_4x16x16.xetla_format<float>()
                            .xetla_select<16 * 16 * 2, 1>(0)
                            .merge(0.0,
                                    Sign_reg_4x16x16.xetla_select<16 * 16 * 2,
                                            1>(0)
                                            > 0);

                    matElem_reg_Sum = matElem_reg_4x16x16.xetla_format<float>()
                                              .xetla_select<16 * 16, 1>(0)
                            + matElem_reg_4x16x16.xetla_format<float>()
                                      .xetla_select<16 * 16, 1>(16 * 16);

                    if (valid_block_16x16_x > 2) {

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<16 * 16, 1>(16 * 16 * 2)
                                *= matW_reg_4x16x16.xetla_select<16 * 16, 1>(
                                        16 * 16 * 2);

                        matElem_reg_4x16x16.xetla_format<float>()
                                .xetla_select<16 * 16, 1>(16 * 16 * 2)
                                .merge(0.0,
                                        Sign_reg_4x16x16.xetla_select<16 * 16,
                                                1>(16 * 16 * 2)
                                                > 0);

                        matElem_reg_Sum = matElem_reg_Sum
                                + matElem_reg_4x16x16.xetla_format<float>()
                                          .xetla_select<16 * 16, 1>(
                                                  16 * 16 * 2);

                        if (valid_block_16x16_x > 3) {
                            matElem_reg_4x16x16.xetla_format<float>()
                                    .xetla_select<16 * 16, 1>(16 * 16 * 3)
                                    *= matW_reg_4x16x16
                                               .xetla_select<16 * 16, 1>(
                                                       16 * 16 * 3);

                            matElem_reg_4x16x16.xetla_format<float>()
                                    .xetla_select<16 * 16, 1>(16 * 16 * 3)
                                    .merge(0.0,
                                            Sign_reg_4x16x16.xetla_select<
                                                    16 * 16, 1>(16 * 16 * 3)
                                                    > 0);

                            matElem_reg_Sum = matElem_reg_Sum
                                    + matElem_reg_4x16x16.xetla_format<float>()
                                              .xetla_select<16 * 16, 1>(
                                                      16 * 16 * 3);
                        }
                    }

                    matElem_reg_Sum_8.xetla_format<float, 16, 8>()
                            = matElem_reg_Sum.xetla_format<float, 16, 16>()
                                      .xetla_select<16, 1, 8, 1>(0, 0)
                            + matElem_reg_Sum.xetla_format<float, 16, 16>()
                                      .xetla_select<16, 1, 8, 1>(0, 8);

                    matElem_reg_Sum_4.xetla_format<float, 16, 4>()
                            = matElem_reg_Sum_8.xetla_format<float, 16, 8>()
                                      .xetla_select<16, 1, 4, 1>(0, 0)
                            + matElem_reg_Sum_8.xetla_format<float, 16, 8>()
                                      .xetla_select<16, 1, 4, 1>(0, 4);

                    matElem_reg_Sum_2.xetla_format<float, 16, 2>()
                            = matElem_reg_Sum_4.xetla_format<float, 16, 4>()
                                      .xetla_select<16, 1, 2, 1>(0, 0)
                            + matElem_reg_Sum_4.xetla_format<float, 16, 4>()
                                      .xetla_select<16, 1, 2, 1>(0, 2);

                    matElem_reg_Sum_1.xetla_format<float, 16, 1>()
                            = matElem_reg_Sum_2.xetla_format<float, 16, 2>()
                                      .xetla_select<16, 1, 1, 1>(0, 0)
                            + matElem_reg_Sum_2.xetla_format<float, 16, 2>()
                                      .xetla_select<16, 1, 1, 1>(0, 1);

                    xetla_vector<uint32_t, 16> address_fsum
                            = xetla_vector_gen<uint32_t, 16>(0, 1);
                    int address_offset
                            = (batchid * numhead + headid) * Max_SeqLen
                            + all_vert_stride * all_vert_loop + tid_y * 16;
                    address_fsum += address_offset;
                    address_fsum *= sizeof(float);

                    xetla_mask<16> pred = 1;
                    xetla_tatomic_store_global<float, 16, cache_hint::none,
                            cache_hint::none, atomic_op::fadd>(
                            (uint64_t)args->matSum_ptr, address_fsum,
                            matElem_reg_Sum_1.xetla_select<16, 1>(0), pred);

                    first_nbarr.arrive();
                    first_nbarr.wait();

                    matElem_reg_Sum_1 = xetla_load_global<float, 1,
                            data_size::default_size,
                            cache_hint::read_invalidate, cache_hint::cached,
                            16>(args->matSum_ptr, address_fsum);

                    matElem_reg_Sum_1 *= args->Scaling;

#pragma unroll
                    for (int i = 0; i < 16; i++) {
                        matW_reg_4x16x16.xetla_select<16 * 16, 1>(16 * 16 * 0)
                                .xetla_select<16, 1>(i * 16)
                                = matW_reg_4x16x16
                                          .xetla_select<16 * 16, 1>(16 * 16 * 0)
                                          .xetla_select<16, 1>(i * 16)
                                * matElem_reg_Sum_1[i];

                        matElem_reg_4x16x16
                                .xetla_select<16 * 16, 1>(16 * 16 * 0)
                                .xetla_select<16, 1>(i * 16)
                                = matElem_reg_4x16x16
                                          .xetla_select<16 * 16, 1>(16 * 16 * 0)
                                          .xetla_select<16, 1>(i * 16)
                                - matW_reg_4x16x16
                                          .xetla_select<16 * 16, 1>(16 * 16 * 0)
                                          .xetla_select<16, 1>(i * 16);

                        matElem_reg_4x16x16
                                .xetla_select<16 * 16, 1>(16 * 16 * 0)
                                .xetla_select<16, 1>(i * 16)
                                *= args->Pinv;
                    }

                    if (valid_block_16x16_x > 1) {
#pragma unroll
                        for (int i = 0; i < 16; i++) {
                            matW_reg_4x16x16
                                    .xetla_select<16 * 16, 1>(16 * 16 * 1)
                                    .xetla_select<16, 1>(i * 16)
                                    = matW_reg_4x16x16
                                              .xetla_select<16 * 16, 1>(
                                                      16 * 16 * 1)
                                              .xetla_select<16, 1>(i * 16)
                                    * matElem_reg_Sum_1[i];

                            matElem_reg_4x16x16
                                    .xetla_select<16 * 16, 1>(16 * 16 * 1)
                                    .xetla_select<16, 1>(i * 16)
                                    = matElem_reg_4x16x16
                                              .xetla_select<16 * 16, 1>(
                                                      16 * 16 * 1)
                                              .xetla_select<16, 1>(i * 16)
                                    - matW_reg_4x16x16
                                              .xetla_select<16 * 16, 1>(
                                                      16 * 16 * 1)
                                              .xetla_select<16, 1>(i * 16);

                            matElem_reg_4x16x16
                                    .xetla_select<16 * 16, 1>(16 * 16 * 1)
                                    .xetla_select<16, 1>(i * 16)
                                    *= args->Pinv;
                        }

                        if (valid_block_16x16_x > 2) {
#pragma unroll
                            for (int i = 0; i < 16; i++) {
                                matW_reg_4x16x16
                                        .xetla_select<16 * 16, 1>(16 * 16 * 2)
                                        .xetla_select<16, 1>(i * 16)
                                        = matW_reg_4x16x16
                                                  .xetla_select<16 * 16, 1>(
                                                          16 * 16 * 2)
                                                  .xetla_select<16, 1>(i * 16)
                                        * matElem_reg_Sum_1[i];

                                matElem_reg_4x16x16
                                        .xetla_select<16 * 16, 1>(16 * 16 * 2)
                                        .xetla_select<16, 1>(i * 16)
                                        = matElem_reg_4x16x16
                                                  .xetla_select<16 * 16, 1>(
                                                          16 * 16 * 2)
                                                  .xetla_select<16, 1>(i * 16)
                                        - matW_reg_4x16x16
                                                  .xetla_select<16 * 16, 1>(
                                                          16 * 16 * 2)
                                                  .xetla_select<16, 1>(i * 16);

                                matElem_reg_4x16x16
                                        .xetla_select<16 * 16, 1>(16 * 16 * 2)
                                        .xetla_select<16, 1>(i * 16)
                                        *= args->Pinv;
                            }

                            if (valid_block_16x16_x > 3) {
#pragma unroll
                                for (int i = 0; i < 16; i++) {
                                    matW_reg_4x16x16
                                            .xetla_select<16 * 16, 1>(
                                                    16 * 16 * 3)
                                            .xetla_select<16, 1>(i * 16)
                                            = matW_reg_4x16x16
                                                      .xetla_select<16 * 16, 1>(
                                                              16 * 16 * 3)
                                                      .xetla_select<16, 1>(
                                                              i * 16)
                                            * matElem_reg_Sum_1[i];

                                    matElem_reg_4x16x16
                                            .xetla_select<16 * 16, 1>(
                                                    16 * 16 * 3)
                                            .xetla_select<16, 1>(i * 16)
                                            = matElem_reg_4x16x16
                                                      .xetla_select<16 * 16, 1>(
                                                              16 * 16 * 3)
                                                      .xetla_select<16, 1>(
                                                              i * 16)
                                            - matW_reg_4x16x16
                                                      .xetla_select<16 * 16, 1>(
                                                              16 * 16 * 3)
                                                      .xetla_select<16, 1>(
                                                              i * 16);

                                    matElem_reg_4x16x16
                                            .xetla_select<16 * 16, 1>(
                                                    16 * 16 * 3)
                                            .xetla_select<16, 1>(i * 16)
                                            *= args->Pinv;
                                }
                            }
                        }
                    }
                }

                //store
                switch (std_seqlen) {
                    case 128: {
                        matC_128x128_t matC_128x128;
                        matC_128x128_payload_t matC_128x128_payload;

                        matC_128x128_payload.init(args->matdW_ptr,
                                ld_st_width_c, ld_st_height_c, ld_st_pitch_c,
                                ld_st_start_x_c, ld_st_start_y_c);

                        xetla_vector<float, 16 * 32> matElem_reg_store
                                = matElem_reg_4x16x16.xetla_format<float>()
                                          .xetla_select<16 * 32, 1>(0);
                        matC_128x128.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_store);

                        subgroup::tile_store(
                                matC_128x128, matC_128x128_payload);
                        xetla_fence<memory_kind::untyped_global>();
                    } break;

                    case 256: {
                        matC_128x256_t matC_128x256;
                        matC_128x256_payload_t matC_128x256_payload;

                        matC_128x256_payload.init(args->matdW_ptr,
                                ld_st_width_c, ld_st_height_c, ld_st_pitch_c,
                                ld_st_start_x_c, ld_st_start_y_c);

                        matC_128x256.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_4x16x16);

                        subgroup::tile_store(
                                matC_128x256, matC_128x256_payload);
                        xetla_fence<memory_kind::untyped_global>();
                    } break;

                    case 384: {
                        matC_64x384_t matC_64x384;
                        matC_64x384_payload_t matC_64x384_payload;

                        matC_64x384_payload.init(args->matdW_ptr, ld_st_width_c,
                                ld_st_height_c, ld_st_pitch_c, ld_st_start_x_c,
                                ld_st_start_y_c);

                        xetla_vector<float, 16 * 16 * 3> matElem_reg_store
                                = matElem_reg_4x16x16.xetla_format<float>()
                                          .xetla_select<16 * 16 * 3, 1>(0);
                        matC_64x384.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_store);

                        subgroup::tile_store(matC_64x384, matC_64x384_payload);
                        xetla_fence<memory_kind::untyped_global>();
                    } break;

                    case 512: {
                        matC_64x512_t matC_64x512;
                        matC_64x512_payload_t matC_64x512_payload;

                        matC_64x512_payload.init(args->matdW_ptr, ld_st_width_c,
                                ld_st_height_c, ld_st_pitch_c, ld_st_start_x_c,
                                ld_st_start_y_c);

                        matC_64x512.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_4x16x16);

                        subgroup::tile_store(matC_64x512, matC_64x512_payload);
                        xetla_fence<memory_kind::untyped_global>();
                    } break;

                    case 1024: {
                        matC_32x1024_t matC_32x1024;
                        matC_32x1024_payload_t matC_32x1024_payload;

                        matC_32x1024_payload.init(args->matdW_ptr,
                                ld_st_width_c, ld_st_height_c, ld_st_pitch_c,
                                ld_st_start_x_c, ld_st_start_y_c);

                        matC_32x1024.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_4x16x16);

                        subgroup::tile_store(
                                matC_32x1024, matC_32x1024_payload);
                        xetla_fence<memory_kind::untyped_global>();
                    } break;

                    case 2048: {
                        matC_16x2048_t matC_16x2048;
                        matC_16x2048_payload_t matC_16x2048_payload;

                        matC_16x2048_payload.init(args->matdW_ptr,
                                ld_st_width_c, ld_st_height_c, ld_st_pitch_c,
                                ld_st_start_x_c, ld_st_start_y_c);

                        matC_16x2048.reg = xetla_cvt<dtype_sfx, float>(
                                matElem_reg_4x16x16);

                        subgroup::tile_store(
                                matC_16x2048, matC_16x2048_payload);
                        xetla_fence<memory_kind::untyped_global>();
                    } break;
                } //switch

            } //valid coputing
            else {
                first_nbarr.arrive();
                first_nbarr.wait();
            }

            second_nbarr.arrive();
            second_nbarr.wait();

            int all_vert128_loop = all_vert_loop >> all_vert128_shift;
            if (((((all_vert128_loop + 1) << all_vert128_shift) - 1)
                        == all_vert_loop)
                    || (all_vert128_shift == 0)) { //dQ
                gemm_arguments_128x64 gemm_arg_128x64;
                matAcc_128x64_t matAcc_128x64;
                matC_128x64_t matC_128x64;
                matC_128x64_payload_t matC_128x64_payload;

                uint32_t width_a = tru_seqlen_ex;
                uint32_t height_a = (batchid * numhead + headid) * max_seqlen
                        + tru_seqlen;
                uint32_t pitch_a = max_seqlen;
                int start_x_a = 0;
                int start_y_a = (batchid * numhead + headid) * max_seqlen
                        + all_vert128_loop * 128;

                gemm_arg_128x64.matA_base_desc.init({args->matdW_ptr},
                        {width_a, height_a, pitch_a}, {start_x_a, start_y_a});

                uint32_t width_b = (headid + 1) * hdsz;
                uint32_t height_b = tru_seqlen + seqlen_entry;
                uint32_t pitch_b = hiddensize;
                int start_x_b = headid * hdsz;
                int start_y_b = seqlen_entry;

                gemm_arg_128x64.matB_base_desc.init({args->matK_ptr},
                        {width_b, height_b, pitch_b}, {start_x_b, start_y_b});

                gemm_arg_128x64.inner_loop_count
                        = (wg_tile_out_k + k_stride - 1) / k_stride;

                matAcc_128x64.init(0);
                gemm_op_128x64_t gemm_op_128x64;
                gemm_op_128x64(g_thd32_tid, matAcc_128x64, gemm_arg_128x64);

                int ld_st_width_c = (headid + 1) * hdsz;
                int height_c = tru_seqlen + seqlen_entry;
                int pitch_c = hiddensize;
                int start_x_c = headid * hdsz
                        + gemm_op_128x64_t::get_matC_offset_x(g_thd32_tid);
                int start_y_c = all_vert128_loop * 128 + seqlen_entry
                        + gemm_op_128x64_t::get_matC_offset_y(g_thd32_tid);

                matC_128x64_payload.init(args->matdQ_ptr, ld_st_width_c,
                        height_c, pitch_c, start_x_c, start_y_c);
                subgroup::elemwise_cvt<matC_128x64_t, matAcc_128x64_t>(
                        matC_128x64, matAcc_128x64);
                subgroup::tile_store(matC_128x64, matC_128x64_payload);
            }
        } //all_vert128_loop

        for (int transp256_loop = 0; transp256_loop < transp256_loop_num;
                transp256_loop++) {
            gemm_arguments_256x64_trnp_a gemm_arg_256x64;
            matAcc_256x64_trnp_a_t matAcc_256x64;
            matC_256x64_trnp_a_t matC_256x64;
            matC_256x64_trnp_a_payload_t matC_256x64_payload;

            uint32_t width_a = tru_seqlen_ex;
            uint32_t height_a
                    = (batchid * numhead + headid) * max_seqlen + tru_seqlen;
            uint32_t pitch_a = max_seqlen;
            int start_x_a = transp256_loop * 256;
            int start_y_a = (batchid * numhead + headid) * max_seqlen;

            gemm_arg_256x64.matA_base_desc.init({args->matdW_ptr},
                    {height_a, width_a, pitch_a}, {start_y_a, start_x_a});

            uint32_t width_b = (headid + 1) * hdsz;
            uint32_t height_b = tru_seqlen + seqlen_entry;
            uint32_t pitch_b = hiddensize;
            int start_x_b = headid * hdsz;
            int start_y_b = seqlen_entry;

            gemm_arg_256x64.matB_base_desc.init({args->matQ_ptr},
                    {width_b, height_b, pitch_b}, {start_x_b, start_y_b});

            gemm_arg_256x64.inner_loop_count
                    = (wg_tile_out_k + k_stride - 1) / k_stride;

            matAcc_256x64.init(0);
            gemm_op_256x64_trnp_a_t gemm_op_256x64_trnp_a;
            gemm_op_256x64_trnp_a(g_thd32_tid, matAcc_256x64, gemm_arg_256x64);

            int width_c = (headid + 1) * hdsz;
            int height_c = tru_seqlen + seqlen_entry;
            int pitch_c = hiddensize;
            int start_x_c = headid * hdsz
                    + gemm_op_256x64_trnp_a_t::get_matC_offset_x(g_thd32_tid);
            int start_y_c = transp256_loop * 256 + seqlen_entry
                    + gemm_op_256x64_trnp_a_t::get_matC_offset_y(g_thd32_tid);

            matC_256x64_payload.init(args->matdK_ptr, width_c, height_c,
                    pitch_c, start_x_c, start_y_c);
            subgroup::elemwise_cvt<matC_256x64_trnp_a_t,
                    matAcc_256x64_trnp_a_t>(matC_256x64, matAcc_256x64);
            subgroup::tile_store(matC_256x64, matC_256x64_payload);

            all_nbarr.arrive();
            all_nbarr.wait();
        }

        for (int transp128_loop = 0; transp128_loop < transp128_loop_num;
                transp128_loop++) {
            gemm_arguments_128x64_trnp_a gemm_arg_128x64;
            matAcc_128x64_trnp_a_t matAcc_128x64;
            matC_128x64_trnp_a_t matC_128x64;
            matC_128x64_trnp_a_payload_t matC_128x64_payload;

            uint32_t width_a = tru_seqlen_ex;
            uint32_t height_a
                    = (batchid * numhead + headid) * max_seqlen + tru_seqlen;
            uint32_t pitch_a = max_seqlen;
            int start_x_a = transp128_loop * 128 + offset_blk_128x128;
            int start_y_a = (batchid * numhead + headid) * max_seqlen;

            gemm_arg_128x64.matA_base_desc.init({args->matdW_ptr},
                    {height_a, width_a, pitch_a}, {start_y_a, start_x_a});

            uint32_t width_b = (headid + 1) * hdsz;
            uint32_t height_b = tru_seqlen + seqlen_entry;
            uint32_t pitch_b = hiddensize;
            int start_x_b = headid * hdsz;
            int start_y_b = seqlen_entry;

            gemm_arg_128x64.matB_base_desc.init({args->matQ_ptr},
                    {width_b, height_b, pitch_b}, {start_x_b, start_y_b});

            gemm_arg_128x64.inner_loop_count
                    = (wg_tile_out_k + k_stride - 1) / k_stride;

            matAcc_128x64.init(0);
            gemm_op_128x64_trnp_a_t gemm_op_128x64_trnp_a;
            gemm_op_128x64_trnp_a(g_thd32_tid, matAcc_128x64, gemm_arg_128x64);

            int width_c = (headid + 1) * hdsz;
            int height_c = tru_seqlen + seqlen_entry;
            int pitch_c = hiddensize;
            int start_x_c = headid * hdsz
                    + gemm_op_128x64_trnp_a_t::get_matC_offset_x(g_thd32_tid);
            int start_y_c = transp128_loop * 128 + seqlen_entry
                    + offset_blk_128x128
                    + gemm_op_128x64_trnp_a_t::get_matC_offset_y(g_thd32_tid);

            matC_128x64_payload.init(args->matdK_ptr, width_c, height_c,
                    pitch_c, start_x_c, start_y_c);
            subgroup::elemwise_cvt<matC_128x64_trnp_a_t,
                    matAcc_128x64_trnp_a_t>(matC_128x64, matAcc_128x64);
            subgroup::tile_store(matC_128x64, matC_128x64_payload);

            all_nbarr.arrive();
            all_nbarr.wait();
        } //transp128_loop

    } //xetla_softmax_bwd_t::call
}; //struct xetla_softmax_bwd_t

} // namespace gpu::xetla::kernel
