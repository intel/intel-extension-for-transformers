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

/// @brief
///
/// @tparam dtype_bin_
/// @tparam dtype_bot_
/// @tparam dtype_sfx_
/// @tparam dtype_acc_
/// @tparam HWThreadNum
/// @tparam Dopt_RandGenflag
/// @tparam RandSIMD
/// @tparam Max_SeqLen
template <typename dtype_bin_, typename dtype_bot_, typename dtype_sfx_,
        typename dtype_acc_, int HWThreadNum, bool Dopt_RandGenflag = true,
        uint16_t RandSIMD = 16, int Max_SeqLen = 512>
struct xetla_mha_core_attn_fwd_t {
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
    using tile_attr_128x256 = group::tile_shape_t<256, 128, 32, 32>;
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
    using pre_processing_128x64
            = group::pre_processing_matA_neg_filter_t<tile_attr_128x64,
                    gpu_arch::Xe>;

    using gemm_op_128x128_t
            = group::gemm_t<compute_policy_QKT, tile_attr_128x128,
                    mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_128x128>;
    using gemm_op_128x256_t
            = group::gemm_t<compute_policy_QKT, tile_attr_128x256,
                    mem_desc_a_QKT, mem_desc_b_QKT, pre_processing_128x256>;
    using gemm_op_128x64_t = group::gemm_t<compute_policy_out, tile_attr_128x64,
            mem_desc_a_out, mem_desc_b_out, pre_processing_128x64>;

    using gemm_arguments_128x128 = typename gemm_op_128x128_t::arguments_t;
    using gemm_arguments_128x256 = typename gemm_op_128x256_t::arguments_t;
    using gemm_arguments_128x64 = typename gemm_op_128x64_t::arguments_t;

    using matAcc_128x128_t = typename gemm_op_128x128_t::matAcc_t;
    using matAcc_128x256_t = typename gemm_op_128x256_t::matAcc_t;
    using matAcc_128x64_t = typename gemm_op_128x64_t::matAcc_t;

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
    using matC_128x64_tile_desc_t
            = subgroup::tile_desc_t<matAcc_128x64_t::tile_desc::tile_size_x,
                    matAcc_128x64_t::tile_desc::tile_size_y,
                    matAcc_128x64_t::tile_desc::block_size_x,
                    matAcc_128x64_t::tile_desc::block_size_y,
                    reg_layout::tiled>;
    using matC_128x128_t
            = subgroup::tile_t<dtype_sfx, matC_128x128_tile_desc_t>;
    using matC_128x256_t
            = subgroup::tile_t<dtype_sfx, matC_128x256_tile_desc_t>;
    using matC_128x64_t = subgroup::tile_t<dtype_bot, matC_128x64_tile_desc_t>;
    using matC_128x128_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_128x128_tile_desc_t,
            (global_kslicing > 1) ? msg_type::atomic_add : msg_type::block_2d,
            gpu_arch::Xe>;
    using matC_128x256_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout_c, mem_space_c>,
            matC_128x256_tile_desc_t,
            (global_kslicing > 1) ? msg_type::atomic_add : msg_type::block_2d,
            gpu_arch::Xe>;
    using matC_128x64_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_bot, mem_layout_c, mem_space_c>,
            matC_128x64_tile_desc_t,
            (global_kslicing > 1) ? msg_type::atomic_add : msg_type::block_2d,
            gpu_arch::Xe>;

    //512 = 16x32 or 8x64
    using matElem_tile_desc_t
            = gpu::xetla::subgroup::tile_desc_t<64 / sfx_type_size,
                    8 * sfx_type_size, 64 / sfx_type_size, 8 * sfx_type_size,
                    reg_layout::tiled>;
    using matElem_ld_t
            = gpu::xetla::subgroup::tile_t<dtype_sfx, matElem_tile_desc_t>;
    using matElem_ld_payload_t = gpu::xetla::subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout::row_major, mem_space::global>,
            matElem_tile_desc_t,
            subgroup::msg_type_v<matElem_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;
    using matElem_st_t
            = gpu::xetla::subgroup::tile_t<dtype_sfx, matElem_tile_desc_t>;
    using matElem_st_payload_t = gpu::xetla::subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout::row_major, mem_space::global>,
            matElem_tile_desc_t,
            subgroup::msg_type_v<matElem_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;
    using matElem_reg_t = gpu::xetla::subgroup::tile_t<float,
            gpu::xetla::subgroup::tile_desc_t<32, 16, 32, 16,
                    reg_layout::tiled>>;

    /// @brief Arguments for xetla_softmax_fwd_t::run.
    /// User should prepare matQ_ptr, matK_ptr, matQKT_ptr, ...
    ///
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
        float Pinv;
        float Scaling;
    };

    /// @brief Main execution function for fused mha softmax.
    /// The basic process is GEMM -> Softmax -> GEMM.
    ///
    /// @param ei
    /// @param args Includes base descriptors and tid info.
    /// @return
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
        int blk_128x128_one = 0;
        int blk_128x256_loop_num = 1;
        int offset_blk_128x128 = 0;

        int std_seqlen;
        if (tru_seqlen <= 128) {
            std_seqlen = 128;
            all_vert_loop_num = 1;
            blk_128x128_one = 1;
            blk_128x256_loop_num = 0;
        } else if (tru_seqlen <= 256)
            std_seqlen = 256;
        else if (tru_seqlen <= 384) {
            std_seqlen = 384;
            all_vert_loop_num = 3;
            blk_128x128_one = 1;
            blk_128x256_loop_num = 1;
            offset_blk_128x128 = 256;
        } else {
            std_seqlen = 512;
            all_vert_loop_num = 4;
            blk_128x128_one = 0;
            blk_128x256_loop_num = 2;
        }

        xetla_nbarrier_t<32, 32, gpu_arch::Xe> first_nbarr;
        xetla_nbarrier_t<32, 32, gpu_arch::Xe> second_nbarr;
        first_nbarr.init_nbarrier(0, nbarrier_role::producer_consumer);
        second_nbarr.init_nbarrier(1, nbarrier_role::producer_consumer);

        for (int all_vert128_loop = 0; all_vert128_loop < all_vert_loop_num;
                all_vert128_loop++) {
            for (int hor_256_loop = 0; hor_256_loop < blk_128x256_loop_num;
                    hor_256_loop++) {
                gemm_arguments_128x256 gemm_arg_128x256;
                matAcc_128x256_t matAcc_128x256;
                matC_128x256_t matC_128x256;
                matC_128x256_payload_t matC_128x256_payload;

                uint32_t width_a = (headid + 1) * hdsz;
                uint32_t height_a = tru_seqlen + seqlen_entry;
                uint32_t pitch_a = hiddensize;
                int start_x_a = headid * hdsz;
                int start_y_a = all_vert128_loop * 128 + seqlen_entry;

                gemm_arg_128x256.matA_base_desc.init({args->matQ_ptr},
                        {width_a, height_a, pitch_a}, {start_x_a, start_y_a});

                uint32_t width_b = (headid + 1) * hdsz;
                uint32_t height_b = tru_seqlen + seqlen_entry;
                uint32_t pitch_b = hiddensize;
                int start_x_b = headid * hdsz;
                int start_y_b = hor_256_loop * 256 + seqlen_entry;

                //B transpose
                gemm_arg_128x256.matB_base_desc.init({args->matK_ptr},
                        {height_b, width_b, pitch_b}, {start_y_b, start_x_b});

                gemm_arg_128x256.inner_loop_count
                        = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                matAcc_128x256.init(0);
                gemm_op_128x256_t gemm_op_128x256;

                gemm_op_128x256(g_thd32_tid, matAcc_128x256, gemm_arg_128x256);

                uint32_t width_c = max_seqlen;
                uint32_t height_c
                        = max_seqlen * (batchid * numhead + headid + 1);
                uint32_t pitch_c = max_seqlen;
                int start_x_c
                        = gemm_op_128x256_t::get_matC_offset_x(g_thd32_tid)
                        + hor_256_loop * 256;
                int start_y_c = (batchid * numhead + headid) * max_seqlen
                        + all_vert128_loop * 128
                        + gemm_op_128x256_t::get_matC_offset_y(g_thd32_tid);

                matC_128x256_payload.init(args->matQKT_ptr, width_c, height_c,
                        pitch_c, start_x_c, start_y_c);
                subgroup::elemwise_cvt<matC_128x256_t, matAcc_128x256_t>(
                        matC_128x256, matAcc_128x256);
                subgroup::tile_store(matC_128x256, matC_128x256_payload);
                xetla_fence<memory_kind::untyped_global>();
            }

            for (int blk_128x128_loop = 0; blk_128x128_loop < blk_128x128_one;
                    blk_128x128_loop++) {
                gemm_arguments_128x128 gemm_arg_128x128;
                matAcc_128x128_t matAcc_128x128;
                matC_128x128_t matC_128x128;
                matC_128x128_payload_t matC_128x128_payload;

                uint32_t width_a = (headid + 1) * hdsz;
                uint32_t height_a = tru_seqlen + seqlen_entry;
                uint32_t pitch_a = hiddensize;
                int start_x_a = headid * hdsz;
                int start_y_a = all_vert128_loop * 128 + seqlen_entry;

                gemm_arg_128x128.matA_base_desc.init({args->matQ_ptr},
                        {width_a, height_a, pitch_a}, {start_x_a, start_y_a});

                uint32_t width_b = (headid + 1) * hdsz;
                uint32_t height_b = tru_seqlen + seqlen_entry;
                uint32_t pitch_b = hiddensize;
                int start_x_b = headid * hdsz;
                int start_y_b = offset_blk_128x128 + seqlen_entry;

                //B transpose
                gemm_arg_128x128.matB_base_desc.init({args->matK_ptr},
                        {height_b, width_b, pitch_b}, {start_y_b, start_x_b});

                gemm_arg_128x128.inner_loop_count
                        = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                matAcc_128x128.init(0);
                gemm_op_128x128_t gemm_op_128x128;

                gemm_op_128x128(g_thd32_tid, matAcc_128x128, gemm_arg_128x128);

                uint32_t width_c = max_seqlen;
                uint32_t height_c
                        = max_seqlen * (batchid * numhead + headid + 1);
                uint32_t pitch_c = max_seqlen;
                int start_x_c = offset_blk_128x128
                        + gemm_op_128x128_t::get_matC_offset_x(g_thd32_tid);
                int start_y_c = (batchid * numhead + headid) * max_seqlen
                        + all_vert128_loop * 128
                        + gemm_op_128x128_t::get_matC_offset_y(g_thd32_tid);

                matC_128x128_payload.init(args->matQKT_ptr, width_c, height_c,
                        pitch_c, start_x_c, start_y_c);
                subgroup::elemwise_cvt<matC_128x128_t, matAcc_128x128_t>(
                        matC_128x128, matAcc_128x128);
                subgroup::tile_store(matC_128x128, matC_128x128_payload);
                xetla_fence<memory_kind::untyped_global>();
            }

            //fwd softmax
            {
                int elem_Ln512_loop_num = 4;
                int height_8x64_512 = 8 * sfx_type_size;
                int width_8x16_512 = 64 / sfx_type_size;
                int height_elem_offset
                        = (max_seqlen * (batchid * numhead + headid)
                                  + (all_vert128_loop * 128) + (tid_linear * 4))
                        * height_8x64_512;
                int width_elem = width_8x16_512;
                int height_elem;
                int pitch_elem = width_elem;
                int start_x_elem = 0;
                int start_y_elem;
                int bndy_mk_lp_start = (tru_seqlen + 31) >> 5; //32
                int bndy_mk_lp_shift
                        = 32 - (bndy_mk_lp_start << 5) + tru_seqlen;

                xetla_vector<uint32_t, 16> mkin_vec16;
                uint32_t mk_attn_all
                        = sizeof(uint32_t) * (max_seqlen / 32) * (batchid);
                xetla_vector<uint32_t, 16> mk_attn_offsets
                        = xetla_vector_gen<uint32_t, 16>(0, 1);
                mk_attn_offsets *= sizeof(uint32_t);
                mk_attn_offsets += mk_attn_all;
                mkin_vec16 = xetla_load_global<uint32_t, 1,
                        data_size::default_size, cache_hint::read_invalidate,
                        cache_hint::cached, 16>(
                        args->matMkin_ptr, mk_attn_offsets);

                uint32_t mk_offset_all = sizeof(uint32_t) * (max_seqlen / 32)
                        * ((batchid * numhead + headid) * max_seqlen
                                + (all_vert128_loop * 128) + tid_linear * 4);
                xetla_vector<uint32_t, 16> mk_offsets
                        = xetla_vector_gen<uint32_t, 16>(0, 1);
                mk_offsets *= sizeof(uint32_t);
                mk_offsets += mk_offset_all;

                first_nbarr.arrive();
                first_nbarr.wait();

                for (int elem_Ln512_loop = 0;
                        elem_Ln512_loop < elem_Ln512_loop_num;
                        elem_Ln512_loop++) {
                    matElem_ld_t matQKT_rd;
                    matElem_ld_payload_t matQKT_rd_payload;
                    matElem_st_t matQKT_st;
                    matElem_st_payload_t matQKT_st_payload;
                    matElem_reg_t matQKT_reg16x32;

                    xetla_vector<uint32_t, 16> mkdpot_vec16;

                    start_y_elem = height_elem_offset
                            + elem_Ln512_loop * height_8x64_512;
                    height_elem = start_y_elem
                            + ((std_seqlen * sfx_type_size) / 64);

                    matQKT_rd_payload.init(args->matQKT_ptr, width_elem,
                            height_elem, pitch_elem, start_x_elem,
                            start_y_elem);
                    matQKT_st_payload.init(args->matQKT_ptr, width_elem,
                            height_elem, pitch_elem, start_x_elem,
                            start_y_elem);

                    subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                            matQKT_rd, matQKT_rd_payload);

                    if constexpr (Dopt_RandGenflag == false) {
                        mkdpot_vec16 = xetla_load_global<uint32_t, 1,
                                data_size::default_size,
                                cache_hint::read_invalidate, cache_hint::cached,
                                16>(args->matMkdpot_ptr, mk_offsets);
                    }

                    mk_offsets += sizeof(uint32_t) * (max_seqlen / 32);

                    for (int j = bndy_mk_lp_start; j < 16; j++)
                        mkin_vec16[j] = 0xFFFFFFFF;
                    if (bndy_mk_lp_shift < 32) {
                        uint32_t tmp = 0xFFFFFFFF;
                        tmp >>= bndy_mk_lp_shift;
                        tmp <<= bndy_mk_lp_shift;
                        mkin_vec16[bndy_mk_lp_start - 1] |= tmp;
                        //mkin_vec16[bndy_mk_lp_start - 1] <<= bndy_mk_lp_shift;
                        //mkin_vec16[bndy_mk_lp_start - 1] >>= bndy_mk_lp_shift;
                    }

                    matQKT_reg16x32.reg
                            = xetla_cvt<float, dtype_sfx>(matQKT_rd.reg);
                    matQKT_reg16x32.reg = matQKT_reg16x32.reg * args->Pinv;

#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        uint32_t mkdata_i = mkin_vec16[j];
                        xetla_mask_int<32> mkdata
                                = xetla_mask_int_gen<32>(mkdata_i);
                        matQKT_reg16x32.reg.xetla_format<float>()
                                .xetla_select<32, 1>(j * 32)
                                .xetla_merge(-1e32,
                                        matQKT_reg16x32.reg
                                                .xetla_format<float>()
                                                .xetla_select<32, 1>(j * 32),
                                        mkdata);
                    }

                    xetla_vector<float, 16> QKT_reg16_f;
                    QKT_reg16_f = -1e32;
#pragma unroll
                    for (int j = 0; j < 32; j++) {
                        xetla_mask<16> filter_max = (QKT_reg16_f
                                > matQKT_reg16x32.reg.xetla_format<float>()
                                          .xetla_select<16, 1>(j * 16));
                        QKT_reg16_f.xetla_merge(QKT_reg16_f,
                                matQKT_reg16x32.reg.xetla_format<float>()
                                        .xetla_select<16, 1>(j * 16),
                                filter_max);
                    }

                    xetla_mask<8> filter_max8
                            = (QKT_reg16_f.xetla_select<8, 1>(0)
                                    > QKT_reg16_f.xetla_select<8, 1>(8));
                    QKT_reg16_f.xetla_select<8, 1>(0).xetla_merge(
                            QKT_reg16_f.select<8, 1>(0),
                            QKT_reg16_f.select<8, 1>(8), filter_max8);
                    xetla_mask<4> filter_max4
                            = (QKT_reg16_f.xetla_select<4, 1>(0)
                                    > QKT_reg16_f.xetla_select<4, 1>(4));
                    QKT_reg16_f.xetla_select<4, 1>(0).xetla_merge(
                            QKT_reg16_f.select<4, 1>(0),
                            QKT_reg16_f.select<4, 1>(4), filter_max4);
                    xetla_mask<2> filter_max2
                            = (QKT_reg16_f.xetla_select<2, 1>(0)
                                    > QKT_reg16_f.xetla_select<2, 1>(2));
                    QKT_reg16_f.xetla_select<2, 1>(0).xetla_merge(
                            QKT_reg16_f.select<2, 1>(0),
                            QKT_reg16_f.select<2, 1>(2), filter_max2);
                    xetla_mask<1> filter_max1
                            = (QKT_reg16_f.xetla_select<1, 1>(0)
                                    > QKT_reg16_f.xetla_select<1, 1>(1));
                    QKT_reg16_f.xetla_select<1, 1>(0).xetla_merge(
                            QKT_reg16_f.xetla_select<1, 1>(0),
                            QKT_reg16_f.xetla_select<1, 1>(1), filter_max1);

                    {
                        float tmp_max = QKT_reg16_f[0];
                        matQKT_reg16x32.reg = matQKT_reg16x32.reg - tmp_max;
                    }

#pragma unroll
                    for (int j = 0; j < 16; j++)
                        matQKT_reg16x32.reg.xetla_format<float>()
                                .xetla_select<32, 1>(j * 32)
                                = xetla_exp<float, 32>(
                                        matQKT_reg16x32.reg
                                                .xetla_format<float>()
                                                .xetla_select<32, 1>(j * 32));

                    QKT_reg16_f = matQKT_reg16x32.reg.xetla_format<float>()
                                          .xetla_select<16, 1>(0)
                            + matQKT_reg16x32.reg.xetla_format<float>()
                                      .xetla_select<16, 1>(16);
#pragma unroll
                    for (int j = 2; j < 32; j++)
                        QKT_reg16_f = QKT_reg16_f
                                + matQKT_reg16x32.reg.xetla_format<float>()
                                          .xetla_select<16, 1>(j * 16);

                    QKT_reg16_f.xetla_select<8, 1>(0)
                            += QKT_reg16_f.xetla_select<8, 1>(8);
                    QKT_reg16_f.xetla_select<4, 1>(0)
                            += QKT_reg16_f.xetla_select<4, 1>(4);
                    QKT_reg16_f.xetla_select<2, 1>(0)
                            += QKT_reg16_f.xetla_select<2, 1>(2);
                    QKT_reg16_f.xetla_select<1, 1>(0)
                            += QKT_reg16_f.xetla_select<1, 1>(1);

                    QKT_reg16_f.xetla_select<1, 1>(0) = xetla_inv<float, 1>(
                            QKT_reg16_f.xetla_select<1, 1>(0));
                    {
                        float tmp = QKT_reg16_f[0];
                        QKT_reg16_f = tmp;
                    }

#pragma unroll
                    for (int j = 0; j < 32; j++)
                        matQKT_reg16x32.reg.xetla_format<float>()
                                .xetla_select<16, 1>(j * 16)
                                *= QKT_reg16_f;

                    xetla_mask<(Max_SeqLen >> 2)> rand_bit;
                    xetla_vector<uint32_t, 4 * RandSIMD> rand_data;

                    matQKT_reg16x32.reg = matQKT_reg16x32.reg * args->Scaling;

                    using matElem_reg_w_t = subgroup::tile_t<uint16_t,
                            subgroup::tile_desc_t<32, 1, 32, 1,
                                    reg_layout::tiled>>;
                    using matElem_reg_b_t = subgroup::tile_t<uint8_t,
                            subgroup::tile_desc_t<32, 1, 32, 1,
                                    reg_layout::tiled>>;
                    matElem_reg_w_t drop_mk_w;
                    matElem_reg_b_t drop_mk_b;

                    if constexpr (Dopt_RandGenflag == true) {
                        matQKT_st.reg = xetla_cvt<dtype_sfx, float>(
                                matQKT_reg16x32.reg);

                        using matElem_reg_w_t
                                = gpu::xetla::subgroup::tile_t<uint16_t,
                                        gpu::xetla::subgroup::tile_desc_t<32, 1,
                                                32, 1, reg_layout::tiled>>;
                        using matElem_reg_b_t
                                = gpu::xetla::subgroup::tile_t<uint8_t,
                                        gpu::xetla::subgroup::tile_desc_t<32, 1,
                                                32, 1, reg_layout::tiled>>;
                        matElem_reg_w_t drop_mk_w;
                        matElem_reg_b_t drop_mk_b;

#pragma unroll
                        for (int i = 0; i < (Max_SeqLen / (4 * 4 * RandSIMD));
                                i++) {
                            rand_data = Rand_Gen.rand();
                            rand_bit.xetla_select<4 * RandSIMD, 1>(
                                    i * (4 * RandSIMD))
                                    = rand_data > rand_threshold;
                        }
#pragma unroll
                        for (int j = 0; j < 4; j++) {

                            if constexpr (sfx_type_size == 2) {
                                drop_mk_w.reg.xetla_select<32, 1>(0)
                                        .xetla_merge(SIGN_BIT_W16, 0,
                                                rand_bit.xetla_select<32, 1>(
                                                        j * 32));
                                matQKT_st.reg.xetla_format<uint16_t>()
                                        .xetla_select<32, 1>(j * 32)
                                        |= drop_mk_w.reg.xetla_select<32, 1>(0);
                            }
                            if constexpr (sfx_type_size == 1) {
                                drop_mk_b.reg.xetla_select<32, 1>(0)
                                        .xetla_merge(SIGN_BIT_B8, 0,
                                                rand_bit.xetla_select<32, 1>(
                                                        j * 32));
                                matQKT_st.reg.xetla_format<uint8_t>()
                                        .xetla_select<32, 1>(j * 32)
                                        |= drop_mk_b.reg.xetla_select<32, 1>(0);
                            }
                        }

                        if (std_seqlen > 128) {
#pragma unroll
                            for (int i = 0;
                                    i < (Max_SeqLen / (4 * 4 * RandSIMD));
                                    i++) {
                                rand_data = Rand_Gen.rand();
                                rand_bit.xetla_select<4 * RandSIMD, 1>(
                                        i * (4 * RandSIMD))
                                        = rand_data > rand_threshold;
                            }
#pragma unroll
                            for (int j = 4; j < 8; j++) {
                                if constexpr (sfx_type_size == 2) {
                                    drop_mk_w.reg.xetla_select<32, 1>(0)
                                            .xetla_merge(SIGN_BIT_W16, 0,
                                                    rand_bit.xetla_select<32,
                                                            1>((j - 4) * 32));
                                    matQKT_st.reg.xetla_format<uint16_t>()
                                            .xetla_select<32, 1>(j * 32)
                                            |= drop_mk_w.reg
                                                       .xetla_select<32, 1>(0);
                                }
                                if constexpr (sfx_type_size == 1) {
                                    drop_mk_b.reg.xetla_select<32, 1>(0)
                                            .xetla_merge(SIGN_BIT_B8, 0,
                                                    rand_bit.xetla_select<32,
                                                            1>((j - 4) * 32));
                                    matQKT_st.reg.xetla_format<uint8_t>()
                                            .xetla_select<32, 1>(j * 32)
                                            |= drop_mk_b.reg
                                                       .xetla_select<32, 1>(0);
                                }
                            }

                            if (std_seqlen > 256) {
#pragma unroll
                                for (int i = 0;
                                        i < (Max_SeqLen / (4 * 4 * RandSIMD));
                                        i++) {
                                    rand_data = Rand_Gen.rand();
                                    rand_bit.xetla_select<4 * RandSIMD, 1>(
                                            i * (4 * RandSIMD))
                                            = rand_data > rand_threshold;
                                }
#pragma unroll
                                for (int j = 8; j < 12; j++) {
                                    if constexpr (sfx_type_size == 2) {
                                        drop_mk_w.reg.xetla_select<32, 1>(0)
                                                .xetla_merge(SIGN_BIT_W16, 0,
                                                        rand_bit.xetla_select<
                                                                32, 1>(
                                                                (j - 8) * 32));
                                        matQKT_st.reg.xetla_format<uint16_t>()
                                                .xetla_select<32, 1>(j * 32)
                                                |= drop_mk_w.reg
                                                           .xetla_select<32, 1>(
                                                                   0);
                                    }
                                    if constexpr (sfx_type_size == 1) {
                                        drop_mk_b.reg.xetla_select<32, 1>(0)
                                                .xetla_merge(SIGN_BIT_B8, 0,
                                                        rand_bit.xetla_select<
                                                                32, 1>(
                                                                (j - 8) * 32));
                                        matQKT_st.reg.xetla_format<uint8_t>()
                                                .xetla_select<32, 1>(j * 32)
                                                |= drop_mk_b.reg
                                                           .xetla_select<32, 1>(
                                                                   0);
                                    }
                                }
                                if (std_seqlen > 384) {
#pragma unroll
                                    for (int i = 0; i
                                            < (Max_SeqLen / (4 * 4 * RandSIMD));
                                            i++) {
                                        rand_data = Rand_Gen.rand();
                                        rand_bit.xetla_select<4 * RandSIMD, 1>(
                                                i * (4 * RandSIMD))
                                                = rand_data > rand_threshold;
                                    }
#pragma unroll
                                    for (int j = 12; j < 16; j++) {
                                        if constexpr (sfx_type_size == 2) {
                                            drop_mk_w.reg.xetla_select<32, 1>(0)
                                                    .xetla_merge(SIGN_BIT_W16,
                                                            0,
                                                            rand_bit.xetla_select<
                                                                    32, 1>(
                                                                    (j - 12)
                                                                    * 32));
                                            matQKT_st.reg
                                                    .xetla_format<uint16_t>()
                                                    .xetla_select<32, 1>(j * 32)
                                                    |= drop_mk_w.reg
                                                               .xetla_select<32,
                                                                       1>(0);
                                        }
                                        if constexpr (sfx_type_size == 1) {
                                            drop_mk_b.reg.xetla_select<32, 1>(0)
                                                    .xetla_merge(SIGN_BIT_B8, 0,
                                                            rand_bit.xetla_select<
                                                                    32, 1>(
                                                                    (j - 12)
                                                                    * 32));
                                            matQKT_st.reg
                                                    .xetla_format<uint8_t>()
                                                    .xetla_select<32, 1>(j * 32)
                                                    |= drop_mk_b.reg
                                                               .xetla_select<32,
                                                                       1>(0);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        matQKT_st.reg = xetla_cvt<dtype_sfx, float>(
                                matQKT_reg16x32.reg);
#pragma unroll
                        for (int j = 0; j < 16; j++) {
                            uint32_t mkdata_i = mkdpot_vec16[j];
                            xetla_mask_int<32> mkdata
                                    = xetla_mask_int_gen<32>(mkdata_i);
                            if constexpr (sfx_type_size == 2) {
                                drop_mk_w.reg.xetla_select<32, 1>(0)
                                        .xetla_merge(SIGN_BIT_W16, 0, mkdata);
                                matQKT_st.reg.xetla_format<uint16_t>()
                                        .xetla_select<32, 1>(j * 32)
                                        |= drop_mk_w.reg.xetla_select<32, 1>(0);
                            }
                            if constexpr (sfx_type_size == 1) {
                                drop_mk_b.reg.xetla_select<32, 1>(0)
                                        .xetla_merge(SIGN_BIT_B8, 0, mkdata);
                                matQKT_st.reg.xetla_format<uint8_t>()
                                        .xetla_select<32, 1>(j * 32)
                                        |= drop_mk_b.reg.xetla_select<32, 1>(0);
                            }
                        }

                        matQKT_st.reg = xetla_cvt<dtype_sfx, float>(
                                matQKT_reg16x32.reg);
                    }

                    subgroup::tile_store(matQKT_st, matQKT_st_payload);
                    xetla_fence<memory_kind::untyped_global>();
                }

                second_nbarr.arrive();
                second_nbarr.wait();
            }

            //QKtV
            {
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

                uint32_t width_c = (headid + 1) * hdsz;
                uint32_t height_c = tru_seqlen + seqlen_entry;
                uint32_t pitch_c = hiddensize;
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

/// @brief
///
/// @tparam dtype_bwd_bin_
/// @tparam dtype_bwd_bot_
/// @tparam dtype_bwd_sfx_
/// @tparam dtype_bwd_acc_
/// @tparam HWThreadNum
/// @tparam Dopt_RandGenflag
/// @tparam Mkin_flag
template <typename dtype_bwd_bin_, typename dtype_bwd_bot_,
        typename dtype_bwd_sfx_, typename dtype_bwd_acc_, int HWThreadNum,
        bool Dopt_RandGenflag = true, bool Mkin_flag = false>
struct xetla_mha_core_attn_bwd_t {
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
    using tile_attr_128x256 = group::tile_shape_t<256, 128, 32, 32>;
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
    using matC_128x128_t
            = subgroup::tile_t<dtype_sfx, matC_128x128_tile_desc_t>;
    using matC_128x256_t
            = subgroup::tile_t<dtype_sfx, matC_128x256_tile_desc_t>;
    using matC_128x64_t = subgroup::tile_t<dtype_bot, matC_128x64_tile_desc_t>;
    using matC_128x64_trnp_a_t
            = subgroup::tile_t<dtype_bot, matC_128x64_trnp_a_tile_desc_t>;
    using matC_256x64_trnp_a_t
            = subgroup::tile_t<dtype_bot, matC_256x64_trnp_a_tile_desc_t>;
    using matC_128x64_trnp_af_t
            = subgroup::tile_t<dtype_bot, matC_128x64_trnp_af_tile_desc_t>;
    using matC_256x64_trnp_af_t
            = subgroup::tile_t<dtype_bot, matC_256x64_trnp_af_tile_desc_t>;

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
            (global_kslicing > 1)
                    ? msg_type::atomic_add
                    : subgroup::msg_type_v<matC_256x64_trnp_a_tile_desc_t,
                            mem_space_c>,
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

    //512 = 16x32 or 8x64
    using matElem_tile_desc_t
            = gpu::xetla::subgroup::tile_desc_t<64 / sfx_type_size,
                    8 * sfx_type_size, 64 / sfx_type_size, 8 * sfx_type_size,
                    reg_layout::tiled>;
    using matElem_ld_t
            = gpu::xetla::subgroup::tile_t<dtype_sfx, matElem_tile_desc_t>;
    using matElem_st_t
            = gpu::xetla::subgroup::tile_t<dtype_sfx, matElem_tile_desc_t>;
    using matElem_ld_payload_t = gpu::xetla::subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout::row_major, mem_space::global>,
            matElem_tile_desc_t,
            subgroup::msg_type_v<matElem_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;
    using matElem_st_payload_t = gpu::xetla::subgroup::mem_payload_t<
            mem_desc_t<dtype_sfx, mem_layout::row_major, mem_space::global>,
            matElem_tile_desc_t, msg_type::block_2d, gpu_arch::Xe>;
    using matElem_reg_t = gpu::xetla::subgroup::tile_t<float,
            gpu::xetla::subgroup::tile_desc_t<32, 16, 32, 16,
                    reg_layout::tiled>>;
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
        float Pinv;
        float Scaling;
    };

    /// @brief Main execution function for fused mha softmax.
    /// The basic process is GEMM -> Softmax -> GEMM.
    ///
    /// @param ei
    /// @param args Includes base descriptors and tid info.
    /// @return
    __XETLA_API static void call(sycl::nd_item<3> &item, arguments_t *args) {

        int tru_seqlen = 0;
        int tru_seqlen_ex = 0;
        int seqlen_entry = 0;
        int hiddensize = 1024;
        int numhead = 16;
        int hdsz = 64;
        int max_seqlen = 512;
        int wg_tile_QKT_k = hdsz; //args->matrix_k;
        int wg_tile_out_k;

        int groupid = item.get_group(0);
        int batchid = groupid / numhead;
        int headid = groupid % numhead;

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
        int blk_128x128_one = 0;
        int blk_128x256_loop_num = 0;
        int offset_blk_128x128 = 0;
        int std_seqlen;
        if (tru_seqlen <= 128) {
            std_seqlen = 128;
            all_vert_loop_num = 1;
            transp128_loop_num = 1;
            blk_128x128_one = 1;
        } else if (tru_seqlen <= 256) {
            std_seqlen = 256;
            all_vert_loop_num = 2;
            transp256_loop_num = 1;
            blk_128x256_loop_num = 1;
        } else if (tru_seqlen <= 384) {
            std_seqlen = 384;
            all_vert_loop_num = 3;
            transp128_loop_num = 1;
            transp256_loop_num = 1;
            blk_128x128_one = 1;
            blk_128x256_loop_num = 1;
            offset_blk_128x128 = 256;
        } else {
            std_seqlen = 512;
            all_vert_loop_num = 4;
            transp256_loop_num = 2;
            blk_128x256_loop_num = 2;
        }

        work_group_t g_thd32_tid;
        int tid_linear = item.get_local_linear_id();
        g_thd32_tid.init(tid_linear);

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

        for (int all_vert128_loop = 0; all_vert128_loop < all_vert_loop_num;
                all_vert128_loop++) {
            //dW
            for (int hor_256_loop = 0; hor_256_loop < blk_128x256_loop_num;
                    hor_256_loop++) {
                gemm_arguments_128x256 gemm_arg_128x256;
                matAcc_128x256_t matAcc_128x256;
                matC_128x256_t matC_128x256;
                matC_128x256_payload_t matC_128x256_payload;

                uint32_t width_a = (headid + 1) * hdsz;
                uint32_t height_a = tru_seqlen + seqlen_entry;
                uint32_t pitch_a = hiddensize;
                int start_x_a = headid * hdsz;
                int start_y_a = all_vert128_loop * 128 + seqlen_entry;

                gemm_arg_128x256.matA_base_desc.init({args->matdO_ptr},
                        {width_a, height_a, pitch_a}, {start_x_a, start_y_a});

                uint32_t width_b = (headid + 1) * hdsz;
                uint32_t height_b = tru_seqlen + seqlen_entry;
                uint32_t pitch_b = hiddensize;
                int start_x_b = headid * hdsz;
                int start_y_b = hor_256_loop * 256 + seqlen_entry;

                //B transpose, be swapped in init
                gemm_arg_128x256.matB_base_desc.init({args->matV_ptr},
                        {height_b, width_b, pitch_b}, {start_y_b, start_x_b});

                gemm_arg_128x256.inner_loop_count
                        = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                matAcc_128x256.init(0);

                gemm_op_128x256_t gemm_op_128x256;
                gemm_op_128x256(g_thd32_tid, matAcc_128x256, gemm_arg_128x256);

                int width_c = max_seqlen;
                int height_c = max_seqlen * (batchid * numhead + headid + 1);
                int pitch_c = max_seqlen;
                int start_x_c
                        = gemm_op_128x256_t::get_matC_offset_x(g_thd32_tid)
                        + hor_256_loop * 256;
                int start_y_c = (batchid * numhead + headid) * max_seqlen
                        + all_vert128_loop * 128
                        + gemm_op_128x256_t::get_matC_offset_y(g_thd32_tid);

                matC_128x256_payload.init(args->matdW_ptr, width_c, height_c,
                        pitch_c, start_x_c, start_y_c);
                subgroup::elemwise_cvt<matC_128x256_t, matAcc_128x256_t>(
                        matC_128x256, matAcc_128x256);
                subgroup::tile_store(matC_128x256, matC_128x256_payload);
                xetla_fence<memory_kind::untyped_global>();
            }

            for (int blk_128x128_loop = 0; blk_128x128_loop < blk_128x128_one;
                    blk_128x128_loop++) {
                gemm_arguments_128x128 gemm_arg_128x128;
                matAcc_128x128_t matAcc_128x128;
                matC_128x128_t matC_128x128;
                matC_128x128_payload_t matC_128x128_payload;

                uint32_t width_a = (headid + 1) * hdsz;
                uint32_t height_a = tru_seqlen + seqlen_entry;
                uint32_t pitch_a = hiddensize;
                int start_x_a = headid * hdsz;
                int start_y_a = all_vert128_loop * 128 + seqlen_entry;

                gemm_arg_128x128.matA_base_desc.init({args->matdO_ptr},
                        {width_a, height_a, pitch_a}, {start_x_a, start_y_a});

                uint32_t width_b = (headid + 1) * hdsz;
                uint32_t height_b = tru_seqlen + seqlen_entry;
                uint32_t pitch_b = hiddensize;
                int start_x_b = headid * hdsz;
                int start_y_b = offset_blk_128x128 + seqlen_entry;

                //B transpose, be swapped in init
                gemm_arg_128x128.matB_base_desc.init({args->matV_ptr},
                        {height_b, width_b, pitch_b}, {start_y_b, start_x_b});

                gemm_arg_128x128.inner_loop_count
                        = (wg_tile_QKT_k + k_stride - 1) / k_stride;

                matAcc_128x128.init(0);

                gemm_op_128x128_t gemm_op_128x128;
                gemm_op_128x128(g_thd32_tid, matAcc_128x128, gemm_arg_128x128);

                int width_c = max_seqlen;
                int height_c = max_seqlen * (batchid * numhead + headid + 1);
                int pitch_c = max_seqlen;
                int start_x_c = offset_blk_128x128
                        + gemm_op_128x128_t::get_matC_offset_x(g_thd32_tid);
                int start_y_c = (batchid * numhead + headid) * max_seqlen
                        + all_vert128_loop * 128
                        + gemm_op_128x128_t::get_matC_offset_y(g_thd32_tid);

                matC_128x128_payload.init(args->matdW_ptr, width_c, height_c,
                        pitch_c, start_x_c, start_y_c);
                subgroup::elemwise_cvt<matC_128x128_t, matAcc_128x128_t>(
                        matC_128x128, matAcc_128x128);
                subgroup::tile_store(matC_128x128, matC_128x128_payload);
                xetla_fence<memory_kind::untyped_global>();
            }

            int elem_Ln512_loop_num = 4;
            int height_8x64_512 = 8 * sfx_type_size;
            int width_8x16_512 = 64 / sfx_type_size;
            int height_elem_offset
                    = (max_seqlen * (batchid * numhead + headid)
                              + (all_vert128_loop * 128) + (tid_linear * 4))
                    * height_8x64_512;
            int width_elem = width_8x16_512;
            int height_elem;
            int pitch_elem = width_elem;
            int start_x_elem = 0;
            int start_y_elem;

            xetla_vector<uint32_t, 16> mkin_vec16;
            if constexpr (Mkin_flag == true) {
                uint32_t mk_attn_all
                        = sizeof(uint32_t) * (max_seqlen / 32) * (batchid);
                xetla_vector<uint32_t, 16> mk_attn_offsets
                        = xetla_vector_gen<uint32_t, 16>(0, 1);
                mk_attn_offsets *= sizeof(uint32_t);
                mk_attn_offsets += mk_attn_all;
                mkin_vec16 = xetla_load_global<uint32_t, 1,
                        data_size::default_size, cache_hint::read_invalidate,
                        cache_hint::cached, 16>(
                        args->matMkin_ptr, mk_attn_offsets);
            }

            uint32_t mk_offset_all;
            xetla_vector<uint32_t, 16> mk_offsets
                    = xetla_vector_gen<uint32_t, 16>(0, 1);
            if constexpr (Dopt_RandGenflag == false) {
                mk_offset_all = sizeof(uint32_t) * (max_seqlen / 32)
                        * ((batchid * numhead + headid) * max_seqlen
                                + (all_vert128_loop * 128) + tid_linear * 4);
                mk_offsets *= sizeof(uint32_t);
                mk_offsets += mk_offset_all;
            }

            first_nbarr.arrive();
            first_nbarr.wait();

            for (int elem_Ln512_loop = 0; elem_Ln512_loop < elem_Ln512_loop_num;
                    elem_Ln512_loop++) {
                matElem_ld_t matdW_rd;
                matElem_ld_payload_t matdW_rd_payload;
                matElem_ld_t matW_rd;
                matElem_ld_payload_t matW_rd_payload;
                matElem_st_t matdW_st;
                matElem_st_payload_t matdW_st_payload;
                matElem_reg_t matdW_reg16x32;
                matElem_reg_t matW_reg16x32;
                xetla_vector<uint32_t, 16> mkdpot_vec16;

                start_y_elem = height_elem_offset
                        + elem_Ln512_loop * height_8x64_512;
                height_elem
                        = start_y_elem + ((std_seqlen * sfx_type_size) / 64);

                matdW_rd_payload.init(args->matdW_ptr, width_elem, height_elem,
                        pitch_elem, start_x_elem, start_y_elem);
                matW_rd_payload.init(args->matW_ptr, width_elem, height_elem,
                        pitch_elem, start_x_elem, start_y_elem);
                matdW_st_payload.init(args->matdW_ptr, width_elem, height_elem,
                        pitch_elem, start_x_elem, start_y_elem);

                subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                        matdW_rd, matdW_rd_payload);
                subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                        matW_rd, matW_rd_payload);

                if constexpr (Dopt_RandGenflag == false) {
                    mkdpot_vec16 = xetla_load_global<uint32_t, 1,
                            data_size::default_size,
                            cache_hint::read_invalidate, cache_hint::cached,
                            16>(args->matMkdpot_ptr, mk_offsets);
                    mk_offsets += sizeof(uint32_t) * (max_seqlen / 32);
                }

                matdW_reg16x32.reg = xetla_cvt<float, dtype_sfx>(matdW_rd.reg);
                matW_reg16x32.reg = xetla_cvt<float, dtype_sfx>(matW_rd.reg);

                if constexpr (Dopt_RandGenflag == false) {

#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        uint32_t mkdata_i = mkdpot_vec16[j];
                        xetla_mask_int<32> mkdata
                                = xetla_mask_int_gen<32>(mkdata_i);
                        matdW_reg16x32.reg.xetla_format<float>()
                                .xetla_select<32, 1>(j * 32)
                                .xetla_merge(0.0,
                                        matdW_reg16x32.reg.xetla_format<float>()
                                                .xetla_select<32, 1>(j * 32),
                                        mkdata);
                    }
                    matdW_reg16x32.reg = matW_reg16x32.reg * matdW_reg16x32.reg;
                } else {
#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        xetla_mask<32> mask;
                        if constexpr (sfx_type_size == 2) {
                            mask = matW_rd.reg.xetla_format<int16_t>()
                                            .xetla_select<32, 1>(j * 32)
                                    < 0;
                            matW_rd.reg.xetla_format<uint16_t>()
                                    .xetla_select<32, 1>(j * 32)
                                    &= 0x7FFF;
                        }
                        if constexpr (sfx_type_size == 1) {
                            mask = matW_rd.reg.xetla_format<int8_t>()
                                            .xetla_select<32, 1>(j * 32)
                                    < 0;
                            matW_rd.reg.xetla_format<uint8_t>()
                                    .xetla_select<32, 1>(j * 32)
                                    &= 0x7F;
                        }
                        matW_reg16x32.reg.xetla_format<float>()
                                .xetla_select<32, 1>(j * 32)
                                .xetla_merge(0.0, mask);
                    }

                    matdW_reg16x32.reg = matW_reg16x32.reg * matdW_reg16x32.reg;

                    matW_reg16x32.reg
                            = xetla_cvt<float, dtype_sfx>(matW_rd.reg);
                    matW_reg16x32.reg *= args->Scaling;
                }

                xetla_vector<float, 16> mdw_sum
                        = matdW_reg16x32.reg.xetla_select<16, 1>(0);
#pragma unroll
                for (int j = 1; j < 32; j++)
                    mdw_sum = mdw_sum
                            + matdW_reg16x32.reg.xetla_select<16, 1>(j * 16);

                mdw_sum.xetla_select<8, 1>(0) = mdw_sum.xetla_select<8, 1>(0)
                        + mdw_sum.xetla_select<8, 1>(8);
                mdw_sum.xetla_select<4, 1>(0) = mdw_sum.xetla_select<4, 1>(0)
                        + mdw_sum.xetla_select<4, 1>(4);
                mdw_sum.xetla_select<2, 1>(0) = mdw_sum.xetla_select<2, 1>(0)
                        + mdw_sum.xetla_select<2, 1>(2);
                mdw_sum.xetla_select<1, 1>(0) = mdw_sum.xetla_select<1, 1>(0)
                        + mdw_sum.xetla_select<1, 1>(1);
                {
                    float sumtmp = mdw_sum[0];
                    matW_reg16x32.reg = matW_reg16x32.reg * sumtmp;
                }

                matdW_reg16x32.reg -= matW_reg16x32.reg;

                matdW_reg16x32.reg = matdW_reg16x32.reg * args->Pinv;

                if constexpr (Mkin_flag == true) {
#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        uint32_t mkdata_i = mkin_vec16[j];
                        xetla_mask_int<32> mkdata
                                = xetla_mask_int_gen<32>(mkdata_i);
                        matdW_reg16x32.reg.xetla_format<float>()
                                .xetla_select<32, 1>(j * 32)
                                .xetla_merge(0.0,
                                        matdW_reg16x32.reg.xetla_format<float>()
                                                .xetla_select<32, 1>(j * 32),
                                        mkdata);
                    }
                }

                matdW_st.reg = xetla_cvt<dtype_sfx, float>(matdW_reg16x32.reg);

                subgroup::tile_store(matdW_st, matdW_st_payload);
                xetla_fence<memory_kind::untyped_global>();
            }

            second_nbarr.arrive();
            second_nbarr.wait();

            { //dQ
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

                int width_c = (headid + 1) * hdsz;
                int height_c = tru_seqlen + seqlen_entry;
                int pitch_c = hiddensize;
                int start_x_c = headid * hdsz
                        + gemm_op_128x64_t::get_matC_offset_x(g_thd32_tid);
                int start_y_c = all_vert128_loop * 128 + seqlen_entry
                        + gemm_op_128x64_t::get_matC_offset_y(g_thd32_tid);

                matC_128x64_payload.init(args->matdQ_ptr, width_c, height_c,
                        pitch_c, start_x_c, start_y_c);
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
