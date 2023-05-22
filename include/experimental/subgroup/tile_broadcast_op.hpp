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

#include "subgroup/subgroup.hpp"

namespace gpu::xetla::subgroup {

struct tile_minus {
    template <typename dtype, int vec_len>
    static xetla_vector<dtype, vec_len> inline func(
            xetla_vector<dtype, vec_len> vec_data, dtype data) {
        return vec_data - data;
    }
};

struct tile_div {
    template <typename dtype, int vec_len>
    static xetla_vector<dtype, vec_len> inline func(
            xetla_vector<dtype, vec_len> vec_data, dtype data) {
        return vec_data / data;
    }
};

template <typename op, typename matAcc_t>
inline void tile_broadcast_op(matAcc_t &matAcc,
        xetla_vector<typename matAcc_t::dtype, matAcc_t::tile_size_y> data) {
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;
    static constexpr int32_t num_block_y = matAcc_t::num_block_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    using dtype = typename matAcc_t::dtype;
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto acc_reg = (matAcc.reg)
                                   .xetla_select<block_elems, 1>(
                                           (i * num_block_x + j) * block_elems);
            auto acc_reg_2d
                    = acc_reg.xetla_format<dtype, block_size_y, block_size_x>();
#pragma unroll
            for (int row_i = 0; row_i < block_size_y; row_i++) {
                acc_reg_2d.row(row_i) = op::template func<dtype, block_size_x>(
                        acc_reg_2d.row(row_i), data[block_size_y * i + row_i]);
            }
        }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr uint32_t tail_start_y
                = tile_size_y / block_size_y * block_size_y;
        constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
        constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto acc_reg = (matAcc.reg)
                                   .xetla_select<tail_block_elems, 1>(
                                           tail_start_y * tile_size_x
                                           + j * tail_block_elems);
            auto acc_reg_2d
                    = acc_reg.xetla_format<dtype, tail_size_y, block_size_x>();
#pragma unroll
            for (int row_i = 0; row_i < tail_size_y; row_i++) {
                acc_reg_2d.row(row_i) = op::template func<dtype, block_size_x>(
                        acc_reg_2d.row(row_i), data[tail_start_y + row_i]);
            }
        }
    }
}

} // namespace gpu::xetla::subgroup
