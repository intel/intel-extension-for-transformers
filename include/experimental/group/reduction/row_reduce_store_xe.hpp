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

#include "experimental/group/reduction/reduction_api.hpp"

namespace gpu::xetla::group {

template <typename dtype_acc, typename dtype_out, uint32_t row_size,
        uint32_t wg_size_x, uint32_t wg_size_y, uint32_t max_simd_len>
struct group_row_reduce_store_t<dtype_acc, dtype_out, row_size, wg_size_x,
        wg_size_y, max_simd_len, gpu_arch::Xe> {
    static constexpr uint32_t block_size_x
            = gpu::xetla::subgroup::detail::gcd<row_size, max_simd_len>::value;
    static_assert(block_size_x >= 8,
            "if block_size_x is less than 8, the efficiency will be low. "
            "Please choose another tile_size_x");
    static constexpr uint32_t num_block_x = row_size / block_size_x;
    static_assert((num_block_x < wg_size_y) || (num_block_x % wg_size_y == 0),
            "num_block_x should be less than wg_size_y or num_block_x should "
            "be a multiple of wg_size_y");
    static constexpr uint32_t num_block_per_thd
            = (num_block_x < wg_size_y) ? 1 : num_block_x / wg_size_y;
    static constexpr uint32_t cooperative_thd_num
            = (num_block_x < wg_size_y) ? num_block_x : wg_size_y;
    static constexpr uint32_t local_tile_size_x
            = num_block_per_thd * block_size_x;
    using local_st_tile_desc_t = subgroup::tile_desc_t<row_size, 1,
            block_size_x, 1, reg_layout::tiled>;
    using local_st_t = subgroup::tile_t<dtype_acc, local_st_tile_desc_t>;
    using local_st_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_acc, mem_layout::row_major, mem_space::local>,
            local_st_tile_desc_t,
            subgroup::msg_type_v<local_st_tile_desc_t, mem_space::local>,
            gpu_arch::Xe>;
    using local_ld_tile_desc_t = subgroup::tile_desc_t<local_tile_size_x,
            wg_size_y, block_size_x, wg_size_y, reg_layout::tiled>;
    using local_ld_t = subgroup::tile_t<dtype_acc, local_ld_tile_desc_t>;
    using local_ld_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_acc, mem_layout::row_major, mem_space::local>,
            local_ld_tile_desc_t,
            subgroup::msg_type_v<local_ld_tile_desc_t, mem_space::local>,
            gpu_arch::Xe>;

    //If the local tile size is small, we still can use 2D block store
    using global_st_tile_desc_t = subgroup::tile_desc_t<local_tile_size_x, 1,
            block_size_x, 1, reg_layout::tiled>;
    using global_st_t = subgroup::tile_t<dtype_out, global_st_tile_desc_t>;
    using global_st_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>,
            global_st_tile_desc_t,
            (local_tile_size_x * sizeof(dtype_out) > 64) ? msg_type::block_1d
                                                         : msg_type::block_2d,
            gpu_arch::Xe>;
    xetla_nbarrier_t<wg_size_y, wg_size_y, gpu_arch::Xe> nbarrier;
    local_st_t local_st;
    local_st_payload_t local_st_payload;
    local_ld_t local_ld;
    local_ld_payload_t local_ld_payload;
    uint32_t sg_idx;
    uint32_t sg_idy;
    inline void init(uint32_t sg_idx_ = 0, uint32_t sg_idy_ = 0,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        sg_idx = sg_idx_;
        sg_idy = sg_idy_;
        nbarrier.init_nbarrier(
                sg_idx + nbarrier_base, nbarrier_role::producer_consumer);
        local_st_payload.init(slm_base, row_size * wg_size_x, wg_size_y,
                row_size * wg_size_x, row_size * sg_idx, sg_idy);
        local_ld_payload.init(slm_base, row_size * wg_size_x, wg_size_y,
                row_size * wg_size_x,
                row_size * sg_idx + local_tile_size_x * sg_idy, 0);
    }

    inline KERNEL_FUNC void operator()(dtype_out *ptr, uint32_t st_width,
            uint32_t st_height, uint32_t st_pitch, int start_n_base,
            int start_m_base, xetla_vector<dtype_acc, row_size> buffer) {
        local_st.reg = buffer;
        subgroup::tile_store(local_st, local_st_payload);
        xetla_fence<memory_kind::shared_local>();
        nbarrier.arrive();
        nbarrier.wait();
        if (sg_idy < cooperative_thd_num) {
            subgroup::tile_load(local_ld, local_ld_payload);
            global_st_t global_st;
            global_st_payload_t global_st_payload(ptr, st_width, st_height,
                    st_pitch, start_n_base + local_tile_size_x * sg_idy,
                    start_m_base);
            global_st.reg = subgroup::tile_reduce<reduce_op::sum, dtype_out,
                    dtype_acc, 0>(local_ld);
            subgroup::tile_store<cache_hint::uncached>(
                    global_st, global_st_payload);
        }
        nbarrier.arrive();
        nbarrier.wait();
    }
};

template <typename dtype_acc, typename dtype_out, uint32_t row_size,
        uint32_t wg_size_x, uint32_t max_simd_len>
struct group_row_reduce_store_t<dtype_acc, dtype_out, row_size, wg_size_x, 1,
        max_simd_len, gpu_arch::Xe> {
    static constexpr uint32_t block_size_x
            = gpu::xetla::subgroup::detail::gcd<row_size, max_simd_len>::value;

    using global_st_tile_desc_t = subgroup::tile_desc_t<row_size, 1,
            block_size_x, 1, reg_layout::tiled>;
    using global_st_t = subgroup::tile_t<dtype_out, global_st_tile_desc_t>;
    using global_st_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>,
            global_st_tile_desc_t,
            (row_size * sizeof(dtype_out) > 64) ? msg_type::block_1d
                                                : msg_type::block_2d,
            gpu_arch::Xe>;
    inline void init(uint32_t sg_idx_ = 0, uint32_t sg_idy_ = 0,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {}

    inline KERNEL_FUNC void operator()(dtype_out *ptr, uint32_t st_width,
            uint32_t st_height, uint32_t st_pitch, int start_n_base,
            int start_m_base, xetla_vector<dtype_acc, row_size> buffer) {
        global_st_t global_st;
        global_st_payload_t global_st_payload;
        global_st.reg = xetla_cvt<dtype_out, dtype_acc, row_size>(buffer);
        global_st_payload.init(
                ptr, st_width, st_height, st_pitch, start_n_base, start_m_base);
        subgroup::tile_store<cache_hint::uncached>(
                global_st, global_st_payload);
    }
};

} // namespace gpu::xetla::group
