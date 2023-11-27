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
#include "subgroup/tile/impl/op_function.hpp"
#include "subgroup/tile/impl/payload_xe.hpp"

namespace gpu::xetla::subgroup {
namespace detail {
template <typename payload_t>
struct check_prefetch_type {
    static constexpr bool is_global_block_2d_xe
            = ((payload_t::memory_space == mem_space::global)
                    && (payload_t::tile_desc::tile_size_y != 1)
                    && (payload_t::arch_tag <= gpu_arch::Xe)
                    && (payload_t::message_type == msg_type::block_2d));

    static constexpr bool is_global_block_1d_xe
            = ((payload_t::memory_space == mem_space::global)
                    && (payload_t::tile_desc::tile_size_y == 1)
                    && (payload_t::arch_tag <= gpu_arch::Xe));

    static constexpr bool is_global_unaligned_2d_xe
            = ((payload_t::memory_space == mem_space::global)
                    && (payload_t::tile_desc::tile_size_y != 1)
                    && (payload_t::arch_tag -= gpu_arch::Xe)
                    && (payload_t::message_type == msg_type::unaligned_2d));

    static constexpr bool is_local_xe
            = ((payload_t::memory_space == mem_space::local)
                    && (payload_t::arch_tag == gpu_arch::Xe));
};

} // namespace detail

/// @brief Is prefetch data func, which data located in global memory is prefetched to
/// cache, where has higher bandwidth. e.g. In gemm, prefetch next iteration data for mma
/// consumption. This func is specicalized for block 2d scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation.
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the information for prefetches.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_prefetch_type<payload_t>::is_global_block_2d_xe
        && payload_t::arch_tag == gpu_arch::Xe>
tile_prefetch(payload_t &payload) {
    using dtype = typename payload_t::dtype;
    static constexpr uint32_t num_tdesc = payload_t::num_tdesc;
    auto tdesc_2d
            = payload.tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();

#pragma unroll
    for (int i = 0; i < num_tdesc; i++) {
        xetla_tprefetch_global<dtype, L1, L2, payload_t::arch_tag>(
                tdesc_2d.row(i));
    }
}

/// @brief Is prefetch data func, which data located in global memory is prefetched to
/// cache, where has higher bandwidth. e.g. In gemm, prefetch next iteration data for mma
/// consumption. This func is specicalized for block 1d scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the information for prefetches.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_prefetch_type<payload_t>::is_global_block_2d_xe
        && payload_t::arch_tag == gpu_arch::Dg2>
tile_prefetch(payload_t &payload) {
    using dtype = typename payload_t::dtype;
    using tile_desc = typename payload_t::tile_desc;
    using prefetch_dtype = typename payload_t::prefetch_dtype;
    constexpr uint32_t num_channel = payload_t::num_channel;
    constexpr uint32_t load_elems = num_channel * payload_t::simd_exec_size;
    constexpr uint32_t scale_factor = payload_t::scale_factor;
#pragma unroll
    for (int i = 0; i < tile_desc::tile_size_y / tile_desc::block_size_y; i++) {
        uint32_t offset_y = i * tile_desc::block_size_y;
#pragma unroll
        for (int j = 0; j < tile_desc::num_block_x; j++) {
            uint32_t offset_x = j * tile_desc::block_size_x;
#pragma unroll
            for (int sub_block_y = 0; sub_block_y < tile_desc::block_size_y;
                    sub_block_y += num_channel) {
                uint32_t address_offset = payload_t::trans
                        ? offset_x * payload.pitch_in_bytes
                                + (offset_y + sub_block_y) * sizeof(dtype)
                        : offset_x * sizeof(dtype)
                                + (offset_y + sub_block_y)
                                        * payload.pitch_in_bytes;
                xetla_mask<num_channel> pred_y
                        = payload.base_y + offset_y + sub_block_y + num_channel
                                > payload.height_in_elems
                        ? (xetla_vector_gen<uint32_t, num_channel>(0, 1)
                                < (payload.height_in_elems % num_channel))
                        : 1;

                xetla_prefetch_global<prefetch_dtype, payload_t::simd_exec_size,
                        data_size::default_size, L1, L2,
                        payload_t::num_channel>(payload.base_ptr,
                        payload.channel_offset + payload.base_offset
                                + address_offset,
                        pred_y);
            }
        }
    }
}

/// @brief Is prefetch data func, which data located in global memory is prefetched to
/// cache, where has higher bandwidth. e.g. In gemm, prefetch next iteration data for mma
/// consumption. This func is specicalized for block 1d scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the information for prefetches.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_prefetch_type<payload_t>::is_global_block_1d_xe>
tile_prefetch(payload_t &payload) {
    using dtype = typename payload_t::dtype;
    using tile_desc = typename payload_t::tile_desc;
    using prefetch_dtype = typename payload_t::prefetch_dtype;
    constexpr uint32_t prefetch_len
            = tile_desc::tile_size_x / payload_t::scale_factor;
    // TODO (read from arch register info)
    constexpr uint32_t reg_in_bytes
            = payload_t::arch_tag == gpu_arch::Xe ? 64 : 32;
    if constexpr (prefetch_len >= reg_in_bytes) {
#pragma unroll
        for (int j = 0; j < prefetch_len / reg_in_bytes; j++) {
            uint32_t offset_x = j * reg_in_bytes * payload_t::scale_factor;
            uint32_t address_offset = offset_x * sizeof(dtype);
            xetla_prefetch_global<prefetch_dtype, reg_in_bytes,
                    data_size::default_size, L1, L2>(
                    payload.base_ptr, payload.base_offset + address_offset);
        }
    }
    constexpr uint32_t tail_len = prefetch_len % reg_in_bytes;
    uint32_t tail_offset = prefetch_len / reg_in_bytes * reg_in_bytes
            * payload_t::scale_factor;
    detail::process_1d_tail<tail_len, reg_in_bytes / 2, L1, L2, payload_t>(
            payload, tail_offset);
}

/// @brief Is prefetch data func.
/// Current shared local memory prefetch is not supported yet. Only used to keep the consistency with global prefetch.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info.
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the information for prefetches.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_prefetch_type<payload_t>::is_local_xe>
tile_prefetch(payload_t &payload) {}

} // namespace gpu::xetla::subgroup
