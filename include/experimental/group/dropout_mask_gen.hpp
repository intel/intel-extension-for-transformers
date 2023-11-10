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

#include "common/common.hpp"
#include "subgroup/tile/tile.hpp"

namespace gpu::xetla::group {

/// @brief
///
/// @tparam dtype_mask_
/// @tparam wg_tile_n_
/// @tparam wg_tile_m_
/// @tparam sg_tile_n_
/// @tparam sg_tile_m_
/// @tparam random_simd_
/// @tparam arch_
template <typename dtype_mask_, uint32_t wg_tile_n_, uint32_t wg_tile_m_,
        uint32_t sg_tile_n_, uint32_t sg_tile_m_, uint32_t random_simd_ = 16,
        gpu_arch arch_ = gpu_arch::Xe>
struct mask_gen_t {
    using dtype_mask = dtype_mask_;
    static constexpr uint32_t wg_tile_n = wg_tile_n_;
    static constexpr uint32_t wg_tile_m = wg_tile_m_;
    static constexpr uint32_t sg_tile_n = sg_tile_n_;
    static constexpr uint32_t sg_tile_m = sg_tile_m_;
    static constexpr uint32_t random_simd = random_simd_;

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;

    /// @brief
    ///
    struct arguments_t {
        dtype_mask *mask_ptr;
        uint32_t matrix_m;
        uint32_t matrix_n;
        uint32_t mask_ld;
        uint64_t rand_seed = 67280421310721;
        uint64_t *rand_offset_ptr;
        float dropout_prob;
    };

    using load_store_attr = typename arch_attr_t<
            arch_>::template load_store_attr<msg_type::block_2d>;
    static constexpr uint32_t max_store_width_in_bytes
            = load_store_attr::max_store_width_in_bytes;
    static constexpr uint32_t max_store_width_in_elem
            = max_store_width_in_bytes / sizeof(dtype_mask);
    static constexpr uint32_t max_store_height_in_elem
            = load_store_attr::max_store_height_in_elem;
    static constexpr uint32_t tile_size_x = sg_tile_n;
    static constexpr uint32_t tile_size_y = sg_tile_m;
    // block_size_x should be power of 2 and tile_size_x should be divided by block_size_x
    static constexpr uint32_t block_size_x
            = max_store_width_in_elem > tile_size_x
            ? tile_size_x
            : gpu::xetla::subgroup::detail::gcd<tile_size_x,
                    max_store_width_in_elem>::value;
    static_assert(block_size_x >= 8,
            "if block_size_x less than 8, the efficiency will be low. Please "
            "choose another tile_size_x");
    static constexpr uint32_t block_size_y
            = max_store_height_in_elem > tile_size_y ? tile_size_y
                                                     : max_store_height_in_elem;

    using mask_out_tile_desc_t = subgroup::tile_desc_t<tile_size_x, tile_size_y,
            block_size_x, block_size_y, reg_layout::tiled>;
    using mask_out_tile_t = subgroup::tile_t<dtype_mask, mask_out_tile_desc_t>;
    using mask_out_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>,
            mask_out_tile_desc_t,
            (sg_tile_m == 1) ? msg_type::block_1d : msg_type::block_2d,
            gpu_arch::Xe>;
    static constexpr uint32_t tile_size = tile_size_x * tile_size_y;

    /// @brief
    /// @param args
    /// @param wg_idx
    /// @param wg_idy
    /// @param sg_idx
    /// @param sg_idy
    /// @param linear_idx
    /// @return
    __XETLA_API KERNEL_FUNC void operator()(arguments_t *args, uint32_t wg_idx,
            uint32_t wg_idy, uint32_t sg_idx, uint32_t sg_idy,
            uint32_t linear_idx) {
        xetla_vector<uint64_t, 1> rand_offset_ptr_v
                = xetla_load_global<uint64_t, 1, data_size::default_size,
                        cache_hint::cached, cache_hint::cached>(
                        args->rand_offset_ptr, 0);
        uint32_t threshold = uint32_t(args->dropout_prob * float(4294967296));
        mask_out_tile_t mask_out;
        int start_m = wg_idy * wg_tile_m + sg_idy * sg_tile_m;
        int start_n = wg_idx * wg_tile_n + sg_idx * sg_tile_n;
        mask_out_payload_t mask_out_payload(args->mask_ptr, args->matrix_n,
                args->matrix_m, args->mask_ld, start_n, start_m);
        static constexpr uint32_t random_len = 4 * random_simd;
        xetla_rand_t<random_simd> rand_gen;
        rand_gen.init(args->rand_seed, linear_idx, rand_offset_ptr_v[0]);

        xetla_vector<dtype_mask, tile_size> mask;
#pragma unroll
        for (int i = 0; i < tile_size / random_len; i++) {
            auto mask_sub = mask.xetla_select<random_len, 1>(i * random_len);
            xetla_vector<uint32_t, random_len> rand_val = rand_gen.rand();
            xetla_mask<random_len> mask_flag = rand_val < threshold;
            mask_sub.xetla_merge(1, 0, mask_flag);
        }
        if constexpr (tile_size % random_len != 0) {
            constexpr uint32_t remain_len = tile_size % random_len;
            constexpr uint32_t remain_start
                    = tile_size / random_len * random_len;
            auto mask_sub = mask.xetla_select<remain_len, 1>(remain_start);
            // drop, still generate random_len
            xetla_vector<uint32_t, random_len> rand_val = rand_gen.rand();
            xetla_mask<random_len> mask_flag = rand_val < threshold;
            mask_sub.xetla_merge(
                    1, 0, mask_flag.xetla_select<remain_len, 1>(0));
        }
        mask_out.reg = mask;
        subgroup::tile_store<cache_hint::uncached>(mask_out, mask_out_payload);
    }
};
} // namespace gpu::xetla::group
