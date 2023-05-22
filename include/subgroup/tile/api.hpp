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

#include "subgroup/tile/common.hpp"

namespace gpu::xetla::subgroup {

/// @brief Is the xetla tile mma operation definition API.
/// @tparam matA_t Is the tile type for matA.
/// @tparam matB_t Is the tile type for matB.
/// @tparam matAcc_src_t Is the tile type for src data of matC.
/// @tparam matAcc_dst_t Is the tile type for dst data of matC.
/// @tparam engine Is the compute engine, fpu or xmx.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename matA_t, typename matB_t, typename matAcc_src_t,
        typename matAcc_dst_t, mma_engine engine, gpu_arch arch_tag>
struct tile_mma_t {};

/// @brief Is to illustrate the memory information
/// @tparam  dtype Is the data type
/// @tparam  tile_desc Is the tile descriptor
/// @tparam  message_type Is the form will be used to load/store
/// @tparam  memory_space Is the data location
/// @tparam  memory_layout Is the memory layout
template <typename dtype, typename tile_desc, msg_type message_type,
        mem_layout memory_layout, mem_space memory_space,
        gpu_arch arch_tag = gpu_arch::Xe>
struct mem_payload_t {};

/// @brief Is to illustrate the memory information to prefetch data to cache.
/// @tparam dtype_ Is the data type.
/// @tparam tile_desc_ Is the tile descriptor.
/// @tparam mem_layout_ Is the memory data layout.
/// @tparam mem_space_ Is the memory space.
/// @tparam cooperative_num_ Is the thread nums to prefetch data.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_, typename tile_desc_, mem_layout mem_layout_,
        mem_space mem_space_, uint32_t cooperative_num_,
        gpu_arch arch_tag = gpu_arch::Xe>
struct prefetch_payload_t {};

/// @brief Is to illustrate the tile information about a sub matrix.
/// @tparam tile_size_x_ Is the horizon tile size.
/// @tparam tile_size_y_ Is the vertical tile size.
/// @tparam block_size_x_ Is the horizon block size.
/// @tparam block_size_y_ Is the vertical block size.
/// @tparam reg_layout_ Is the register layout i.e. tiled, vnni_tiled and so on.
/// @tparam gpu_arch_ Is the hardware architecture tag.
template <uint32_t tile_size_x_, uint32_t tile_size_y_, uint32_t block_size_x_,
        uint32_t block_size_y_, reg_layout reg_layout_ = reg_layout::tiled>
struct tile_desc_t {
    static constexpr uint32_t tile_size_x = tile_size_x_;
    static constexpr uint32_t tile_size_y = tile_size_y_;

    static constexpr uint32_t block_size_x = block_size_x_;
    static constexpr uint32_t block_size_y = block_size_y_;
    static constexpr uint32_t remained_size_y = tile_size_y % block_size_y;

    static constexpr reg_layout register_layout = reg_layout_;
    static constexpr bool reg_transpose
            = reg_layout_ == reg_layout::transpose_tiled;

    static_assert(
            tile_size_x >= block_size_x, "tile_size_x should >= block_size_x ");
    static_assert(
            tile_size_y >= block_size_y, "tile_size_y should >= block_size_y ");
    static_assert(
            (tile_size_y == 1) || ((block_size_x & (block_size_x - 1)) == 0),
            "if tile_size_y > 1, block_size_x should be power of 2 ");
    static_assert(tile_size_x % (block_size_x) == 0,
            "Tile_size_x should be a multiple of block_size_x ");

    static constexpr uint32_t num_block_x = tile_size_x / block_size_x;
    static constexpr uint32_t num_block_y = tile_size_y / block_size_y;
    static constexpr uint32_t num_block
            = num_block_x * (num_block_y + (remained_size_y > 0 ? 1 : 0));

    static constexpr uint32_t block_elems = block_size_x * block_size_y;
    static constexpr uint32_t tile_elems = tile_size_x * tile_size_y;
};

/// @brief Is a struct contains some register file.
/// @tparam dtype_ Is the data type.
/// @tparam tile_desc_ Is tile_desc_t struct.
template <typename dtype_, typename tile_desc_>
struct tile_t : public tile_desc_ {
    using dtype = dtype_;
    using tile_desc = tile_desc_;
    xetla_vector<dtype, tile_desc::tile_elems> reg;

    // Cannot init value by constructor
    inline tile_t(native_type_t<dtype> val) {
        static_assert(!is_internal_type<dtype>::value,
                "compiler currently does NOT support using plained BF16 data "
                "to initialize a BF16 data");
        this->reg = val;
    }

    inline tile_t() = default;
    __XETLA_API void init(native_type_t<dtype> val) { this->reg = val; }
};

} // namespace gpu::xetla::subgroup
