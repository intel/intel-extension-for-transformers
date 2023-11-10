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

#include "group/epilogue/api.hpp"
#include "group/epilogue/common.hpp"
#include "group/epilogue/epilogue_policy.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_epilogue
/// @{

/// @brief Is the epilogue functor specialized for stream_k
template <typename tile_shape_, typename epilogue_t_, typename mem_desc_d_t_,
        typename mem_desc_atomic_sync_t_>
struct epilogue_stream_k_t {

    static constexpr gpu_arch arch_tag = gpu_arch::Xe;
    using epilogue_t = epilogue_t_;
    using mem_desc_d_t = mem_desc_d_t_;
    using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;
    using mem_desc_atomic_sync_t = mem_desc_atomic_sync_t_;
    using tile_shape = tile_shape_;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    using work_group_t = typename tile_shape::work_group_t;
    static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
    static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;

    //Barrier required to synchronize all threads in workgroup for atomic sync across xecores
    static constexpr uint32_t barrier_count = 1;
    static constexpr uint32_t slm_size
            = mem_desc_c_t::is_local ? wg_tile_m * wg_tile_n : 0;
    static constexpr uint32_t N_SG = wg_size_x * wg_size_y;

    xetla_nbarrier_t<N_SG, N_SG, arch_tag> nbarrier;

    using dtype_d = typename mem_desc_d_t::dtype;
    using dtype_flag = typename mem_desc_atomic_sync_t::dtype;

    //Use special residual op for finishing SK groups to read from scratchspace buffer and reduce in GRF; They also store zeros in scratchspace buffer
    using residual_op_t
            = subgroup::elemwise_reduce_op_stream_k_t<reduce_op::sum, dtype_d>;
    using residual_op_args_t = typename residual_op_t::arguments_t;

    static constexpr mem_layout mem_layout_d = mem_desc_d_t::layout;
    static constexpr mem_space mem_space_d = mem_desc_d_t::space;
    static constexpr msg_type msg_type_d_block2d = msg_type::block_2d;
    static constexpr msg_type msg_type_d_atomic = msg_type::atomic_add;

    /// @brief Updates tile base descriptor based on the tid.
    __XETLA_API static void update_sg_tile_tdesc(
            work_group_t &g, mem_desc_d_t &mem_desc_d) {
        int32_t sg_idx = g.get_id() % wg_size_x;
        int32_t sg_idy = g.get_id() / wg_size_x;
        int32_t tile_offset_n = sg_idx * sg_tile_n;
        int32_t tile_offset_m = sg_idy * sg_tile_m;
        mem_desc_d.update_coord(tile_offset_n, tile_offset_m);
    }

    /// @brief Epilogue for stream_k.
    ///Differentiate between Non-finishing SK groups vs finishing SK groups vs DP groups
    ///Initial SK groups perform atomic writes to scratchspace
    ///Final SK groups wait for their peers to finish , reads partial data from scratchspace and reduce in GRF
    ///DP groups and finishing SK groups perform regular epilogue operations.
    /// @tparam matAcc_t Is the type of the input tile.
    /// @param g Is the workgroup of the current tile.
    /// @param matAcc Is the input tile.
    /// @param mem_desc_c Is the memory description of matC, including base, shape and coordinate.
    /// @param dp_group indicates whether current group is data-parallel or stream_k
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g, matAcc_t &matAcc,
            mem_desc_c_t mem_desc_c, mem_desc_d_t mem_desc_d,
            mem_desc_atomic_sync_t mem_desc_atomic_sync, int group_idx,
            int first_group_idx, bool tile_finished, bool tile_started,
            epilogue_args_t epilogue_args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {

        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
        static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
        static constexpr uint32_t block_elems = matAcc_t::block_elems;

        using matD_tile_desc_t = subgroup::tile_desc_t<tile_size_x, tile_size_y,
                block_size_x, block_size_y, reg_layout::tiled>;

        using matD_t = subgroup::tile_t<dtype_d, matD_tile_desc_t>;
        using matD_atomic_payload_t = subgroup::mem_payload_t<mem_desc_d_t,
                matD_tile_desc_t, msg_type_d_atomic, arch_tag>;

        uint32_t nbarrier_id = nbarrier_base;
        nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);

        update_sg_tile_tdesc(g, mem_desc_d);

        //Addressing for atomic signal
        xetla_mask<16> pred(0);
        pred[0] = 1;
        xetla_vector<uint32_t, 16> flag_offsets
                = xetla_vector_gen<uint32_t, 16>(0, 1);
        flag_offsets
                += first_group_idx; // first_group_idx indicates the first peer of the sliced tile
        flag_offsets = flag_offsets * sizeof(dtype_flag);
        int32_t sg_id = g.get_id();
        dtype_flag *flag_pointer = mem_desc_atomic_sync.base.base;

        //SK group , Sliced Tile - SK group handles starting slice or middle slice
        if (!tile_finished) {

            //Perform atomic writes and signal to atomic counter
            matD_atomic_payload_t matD_atomic_payload(mem_desc_d);
            //Atomic store with OOB check
            subgroup::tile_store(matAcc, matD_atomic_payload);

            //Fence to guarantee write completion
            xetla_fence<memory_kind::untyped_global, fence_op::evict,
                    fence_scope::tile>();

            //Group sync to make sure fence is sent
            nbarrier.arrive();
            nbarrier.wait();

            //Signal to other peers
            if (sg_id == 0) {
                xetla_vector<dtype_flag, 16> signal_val(1);
                xetla_tatomic_store_global<dtype_flag, 16, cache_hint::uncached,
                        cache_hint::write_back, atomic_op::iadd>(
                        (uint64_t)flag_pointer, flag_offsets, signal_val, pred);
            }

        } else {

            //last SK group of corresponding sliced tile
            if (!tile_started) {

                //Number of previous peers that have contributed to this sliced tile
                uint32_t num_peers = group_idx - first_group_idx;

                //Group sync
                nbarrier.arrive();
                nbarrier.wait();

                if (sg_id == 0) {

                    xetla_vector<dtype_flag, 16> ret_val(0);
                    xetla_vector<dtype_flag, 16> old_val = num_peers;
                    xetla_vector<dtype_flag, 16> zero_val(0);

                    //Use atomic cmpxchg to test if previous peers have finished writing
                    //Exchange with value zero to clear the flag
                    while (ret_val[0] != num_peers) {

                        ret_val = xetla_atomic_global<atomic_op::cmpxchg,
                                dtype_flag, 16, data_size::default_size,
                                cache_hint::uncached, cache_hint::write_back>(
                                flag_pointer, flag_offsets, old_val, zero_val,
                                pred);
                    }
                }
                //Group sync
                nbarrier.arrive();
                nbarrier.wait();

                //Invoke stream_k residual op
                residual_op_t residual_op;
                residual_op_args_t residual_args(
                        mem_desc_d.base, mem_desc_d.shape);

                residual_op(matAcc, mem_desc_d.coord, residual_args);
            }

            //Finishing SK groups and DP Groups perform normal epilogue operations - post_op fusion + output conversion and write to output buffer
            epilogue_t epilogue;
            epilogue(g, matAcc, mem_desc_c, epilogue_args, slm_base,
                    nbarrier_base);
        }
    }
};

/// @} xetla_epilogue

} // namespace gpu::xetla::group
