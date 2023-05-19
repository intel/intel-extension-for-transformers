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

/// @file
/// C++ API

#pragma once

#include "subgroup/tile/api.hpp"

namespace gpu::xetla::subgroup {

/// @brief Is the tile mma operation functor, specialized for Xe and fpu engine.
template <typename matA_t_, typename matB_t_, typename matAcc_src_t_,
        typename matAcc_dst_t_>
struct tile_mma_t<matA_t_, matB_t_, matAcc_src_t_, matAcc_dst_t_,
        mma_engine::fpu, gpu_arch::Xe> {
    using matA_t = matA_t_;
    using matB_t = matB_t_;
    using matSrc_t = matAcc_src_t_;
    using matDst_t = matAcc_dst_t_;
    using dtype_a = typename matA_t::dtype;
    using dtype_b = typename matB_t::dtype;
    using dtype_src = typename matSrc_t::dtype;
    using dtype_dst = typename matDst_t::dtype;

    using arch_attr = arch_attr_t<gpu_arch::Xe>;
    using register_attr = typename arch_attr::register_attr;

    static_assert(matA_t::tile_desc::reg_transpose,
            "For FMAOp GEMM, the register layout of matA should be col-major");
    static_assert(!matB_t::tile_desc::reg_transpose,
            "For FMAOp GEMM, the register layout of matB should be row-major");

    static constexpr uint32_t a_tile_w = matA_t::tile_desc::tile_size_y;
    static constexpr uint32_t a_tile_h = matA_t::tile_desc::tile_size_x;
    static constexpr uint32_t a_tile_elems = matA_t::tile_desc::tile_elems;
    static constexpr uint32_t a_block_w = matA_t::tile_desc::block_size_y;
    static constexpr uint32_t a_block_h = matA_t::tile_desc::block_size_x;
    static constexpr uint32_t a_block_elems = matA_t::tile_desc::block_elems;

    static constexpr uint32_t b_tile_w = matB_t::tile_desc::tile_size_x;
    static constexpr uint32_t b_tile_h = matB_t::tile_desc::tile_size_y;
    static constexpr uint32_t b_tile_elems = matB_t::tile_desc::tile_elems;
    static constexpr uint32_t b_block_w = matB_t::tile_desc::block_size_x;
    static constexpr uint32_t b_block_h = matB_t::tile_desc::block_size_y;
    static constexpr uint32_t b_block_elems = matB_t::tile_desc::block_elems;

    static constexpr uint32_t tile_size_m = matDst_t::tile_desc::tile_size_y;
    static constexpr uint32_t tile_size_k = a_tile_h;
    static constexpr uint32_t tile_size_n = matDst_t::tile_desc::tile_size_x;
    static constexpr uint32_t tile_elems = tile_size_m * tile_size_n;
    static constexpr uint32_t block_size_n = matDst_t::tile_desc::block_size_x;
    static constexpr uint32_t block_size_m = matDst_t::tile_desc::block_size_y;
    static constexpr uint32_t block_elems = block_size_m * block_size_n;
    // in case of padding
    static_assert(tile_size_m <= matA_t::tile_desc::tile_size_y,
            "matAcc tile m should match with matA tile m");
    static_assert(
            a_tile_h == b_tile_h, "matA tile k should match with matB tile k");
    static_assert(tile_size_n == matB_t::tile_desc::tile_size_x,
            "matAcc tile n should match with matB tile n");
    static_assert(block_size_m == a_block_w,
            "matAcc block m should match with matA block m");
    static_assert(block_size_n == b_block_w,
            "matAcc block n should match with matB block n");
    static_assert((a_tile_h == a_block_h) && (b_tile_h == a_block_h),
            "For SGEMM, no split along k dim for every inner loop");

    static constexpr int32_t num_block_n = matDst_t::tile_desc::num_block_x;
    static constexpr int32_t num_block_m = matDst_t::tile_desc::num_block_y;
    static constexpr int32_t num_block = num_block_m * num_block_n;

    static constexpr int32_t num_acc = register_attr::acc_reg_in_bytes
            / (block_size_n * sizeof(dtype_dst));
    static_assert(block_size_m % num_acc == 0,
            "block_size_m should be a multiple of num_acc");

    __XETLA_API static void mma(
            matA_t &a, matB_t &b, matSrc_t &src, matDst_t &dst) {

#pragma unroll
        for (int i = 0; i < tile_size_m / block_size_m; i++) {
            xetla_vector<dtype_a, a_block_elems> a_subtile
                    = a.reg.xetla_select<a_block_elems, 1>(i * a_block_elems);
#pragma unroll
            for (int j = 0; j < num_block_n; j++) {
                auto b_subtile = b.reg.xetla_select<b_block_elems, 1>(
                        j * b_block_elems);
                auto src_subtile = src.reg.xetla_select<block_elems, 1>(
                        (i * num_block_n + j) * block_elems);
                auto dst_subtile = dst.reg.xetla_select<block_elems, 1>(
                        (i * num_block_n + j) * block_elems);
                auto dst_2d = dst_subtile.xetla_format<dtype_dst, block_size_m,
                        block_size_n>();
                auto src_2d = src_subtile.xetla_format<dtype_src, block_size_m,
                        block_size_n>();
                auto b_2d = b_subtile.xetla_format<dtype_b, b_block_h,
                        b_block_w>();
#pragma unroll
                for (int sub_m = 0; sub_m < block_size_m; sub_m += num_acc) {
#pragma unroll
                    for (int i_acc = 0; i_acc < num_acc; i_acc++) {
                        xetla_vector<dtype_dst, block_size_n> acc_tmp;
                        acc_tmp = a_subtile[i_acc + sub_m] * b_2d.row(0)
                                + src_2d.row(i_acc + sub_m);
#pragma unroll
                        for (int k = 1; k < tile_size_k - 1; k++) {
                            int a_offset = k * a_block_w + i_acc + sub_m;
                            acc_tmp += a_subtile[a_offset] * b_2d.row(k);
                        }
                        {
                            int a_offset = (tile_size_k - 1) * a_block_w + i_acc
                                    + sub_m;
                            dst_2d.row(i_acc + sub_m) = acc_tmp
                                    + a_subtile[a_offset]
                                            * b_2d.row(tile_size_k - 1);
                        }
                    }

                    SW_BARRIER();
                }
            }
        }
        // process the tail
        if constexpr ((tile_size_m % block_size_m) != 0) {
            constexpr uint32_t tail_start_m
                    = block_size_m * (tile_size_m / block_size_m);
            constexpr uint32_t a_subtile_w = a_tile_w - tail_start_m;
            constexpr uint32_t a_subtile_elems = a_block_h * a_subtile_w;
            constexpr uint32_t acc_subtile_m = tile_size_m - tail_start_m;
            constexpr uint32_t acc_subtile_elems = acc_subtile_m * block_size_n;
            xetla_vector<dtype_a, a_subtile_elems> a_subtile
                    = a.reg.xetla_select<a_subtile_elems, 1>(
                            a_block_h * tail_start_m);
#pragma unroll
            for (int j = 0; j < num_block_n; j++) {
                auto b_subtile = b.reg.xetla_select<b_block_elems, 1>(
                        j * b_block_elems);
                auto src_subtile = src.reg.xetla_select<acc_subtile_elems, 1>(
                        (tail_start_m * tile_size_n) + j * acc_subtile_elems);
                auto dst_subtile = dst.reg.xetla_select<acc_subtile_elems, 1>(
                        (tail_start_m * tile_size_n) + j * acc_subtile_elems);
                auto dst_2d = dst_subtile.xetla_format<dtype_dst, acc_subtile_m,
                        block_size_n>();
                auto src_2d = src_subtile.xetla_format<dtype_src, acc_subtile_m,
                        block_size_n>();
                auto b_2d = b_subtile.xetla_format<dtype_b, b_block_h,
                        b_block_w>();
#pragma unroll
                for (int sub_m = 0; sub_m <= (int)acc_subtile_m - num_acc;
                        sub_m += num_acc) {
#pragma unroll
                    for (int i_acc = 0; i_acc < num_acc; i_acc++) {
                        xetla_vector<dtype_dst, block_size_n> acc_tmp;
                        acc_tmp = a_subtile[i_acc + sub_m] * b_2d.row(0)
                                + src_2d.row(i_acc + sub_m);
#pragma unroll
                        for (int k = 1; k < tile_size_k - 1; k++) {
                            int a_offset = k * a_subtile_w + i_acc + sub_m;
                            acc_tmp += a_subtile[a_offset] * b_2d.row(k);
                        }
                        {
                            int a_offset = (tile_size_k - 1) * a_subtile_w
                                    + i_acc + sub_m;
                            dst_2d.row(i_acc + sub_m) = acc_tmp
                                    + a_subtile[a_offset]
                                            * b_2d.row(tile_size_k - 1);
                        }
                    }
                    SW_BARRIER();
                }
            }
            SW_BARRIER();
            //process the tail, change the processing order
            if constexpr ((tile_size_m % num_acc) != 0) {
                constexpr uint32_t remain_num_acc = tile_size_m % num_acc;
                constexpr uint32_t remain_start_m
                        = tile_size_m / num_acc * num_acc;
                constexpr uint32_t tail_remain_offset
                        = remain_start_m - tail_start_m;
#pragma unroll
                for (int j = 0; j < num_block_n; j++) {

                    auto b_subtile = b.reg.xetla_select<b_block_elems, 1>(
                            j * b_block_elems);
                    auto src_subtile
                            = src.reg.xetla_select<acc_subtile_elems, 1>(
                                    (tail_start_m * tile_size_n)
                                    + j * acc_subtile_elems);
                    auto dst_subtile
                            = dst.reg.xetla_select<acc_subtile_elems, 1>(
                                    (tail_start_m * tile_size_n)
                                    + j * acc_subtile_elems);
                    auto dst_2d = dst_subtile.xetla_format<dtype_dst,
                            acc_subtile_m, block_size_n>();
                    auto src_2d = src_subtile.xetla_format<dtype_src,
                            acc_subtile_m, block_size_n>();
                    auto b_2d = b_subtile.xetla_format<dtype_b, b_block_h,
                            b_block_w>();
#pragma unroll
                    for (int i_acc = 0; i_acc < remain_num_acc; i_acc++) {
                        uint32_t m_index = i_acc + tail_remain_offset;
                        xetla_vector<dtype_dst, block_size_n> acc_tmp;
                        acc_tmp = a_subtile[m_index] * b_2d.row(0)
                                + src_2d.row(m_index);
#pragma unroll
                        for (int k = 1; k < tile_size_k - 1; k++) {
                            int a_offset = k * a_subtile_w + m_index;
                            acc_tmp += a_subtile[a_offset] * b_2d.row(k);
                        }
                        {
                            constexpr int kk = tile_size_k - 1;
                            int a_offset = kk * a_subtile_w + m_index;
                            dst_2d.row(m_index) = acc_tmp
                                    + a_subtile[a_offset] * b_2d.row(kk);
                        }
                    }
                }
            }
            SW_BARRIER();
        }
    }
};

} // namespace gpu::xetla::subgroup
