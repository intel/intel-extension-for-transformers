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
#include "subgroup/tile/common.hpp"
namespace gpu::xetla::subgroup {

/// @brief Is the element wise data conversion, the src and dst tile should have the same layout.
///
/// @tparam T_dst Is the destination tile data type.
/// @tparam T_src Is the source tile data type.
/// @param dst Is the reference of the destination tile object.
/// @param src Is the reference of the destination tile object.
/// @return No return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API
        typename std::enable_if_t<(T_src::register_layout != reg_layout::linear)
                && (T_dst::register_layout != reg_layout::linear)
                && (T_src::block_size_y == T_dst::block_size_y)
                && (T_src::block_size_x == T_dst::block_size_x)
                && (T_src::tile_size_y == T_dst::tile_size_y)
                && (T_src::tile_size_x == T_dst::tile_size_x)>
        elemwise_cvt(T_dst &dst, T_src &src) {
    dst.reg = xetla_cvt<typename T_dst::dtype, typename T_src::dtype>(src.reg);
}

/// @brief element wise data conversion with scaling, the src and dst tile should have the same layout.
/// @tparam T_dst is the destination tile data type.
/// @tparam T_src is the source tile data type.
/// @param dst is the reference of the destination tile object.
/// @param src is the reference of the destination tile object.
/// @param scale is the scaling value to be applied before the assignment.
/// @return no return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API
        typename std::enable_if_t<(T_src::register_layout != reg_layout::linear)
                && (T_dst::register_layout != reg_layout::linear)
                && (T_src::block_size_y == T_dst::block_size_y)
                && (T_src::block_size_x == T_dst::block_size_x)
                && (T_src::tile_size_y == T_dst::tile_size_y)
                && (T_src::tile_size_x == T_dst::tile_size_x)>
        elemwise_cvt(T_dst &dst, T_src &src, float scale) {
    dst.reg = xetla_cvt<typename T_dst::dtype, typename T_src::dtype>(
            src.reg, scale);
}

/// @brief Is the element wise op for relu.
///
/// @tparam T Is the tile data type.
/// @tparam post_op Is the post processing op kind, should be relu.
/// @param mat_Acc Is the reference of the tile object.
/// @return No return, update the data in-place.
template <typename T, post_kind post_op>
__XETLA_API typename std::enable_if_t<post_op == post_kind::relu> elemwise_op(
        T &mat_Acc) {
    xetla_mask<T::tile_elems> mask = mat_Acc.reg <= 0;
    mat_Acc.reg.xetla_merge(0, mask);
}

/// @brief Is the element wise op for gelu.
///
/// @tparam T Is the tile data type.
/// @tparam post_op Is the post processing op kind, should be gelu.
/// @param mat_Acc Is the reference of the tile object.
/// @return No return, update the data in-place.
template <typename T, post_kind post_op>
__XETLA_API typename std::enable_if_t<post_op == post_kind::gelu> elemwise_op(
        T &mat_Acc) {
    using dtype = typename T::dtype;
    constexpr dtype C0 = 0.044715f;
    constexpr dtype sqrt_two_over_pi = 0.79788458347320556640625f;
    // total flag register
    constexpr int elems = 8 * 16;
    constexpr int rounds = T::tile_elems / elems;
#pragma unroll
    for (int i = 0; i < rounds; ++i) {
        auto sub_vec = mat_Acc.reg.xetla_select<elems, 1>(elems * i);
        xetla_vector<dtype, elems> sub_vec_x
                = (sqrt_two_over_pi * sub_vec * (1.f + C0 * sub_vec * sub_vec));
        xetla_vector<dtype, elems> tanh_value
                = xetla_tanh<dtype, elems>(sub_vec_x);
        sub_vec = 0.5f * sub_vec * (1.f + tanh_value);
    }
    constexpr int remained_elems = T::tile_elems % elems;
    if constexpr (remained_elems != 0) {
        auto sub_vec = mat_Acc.reg.xetla_select<remained_elems, 1>(
                elems * (T::tile_elems / elems));
        xetla_vector<dtype, remained_elems> sub_vec_x
                = (sqrt_two_over_pi * sub_vec * (1.f + C0 * sub_vec * sub_vec));
        xetla_vector<dtype, remained_elems> tanh_value
                = xetla_tanh<dtype, remained_elems>(sub_vec_x);
        sub_vec = 0.5f * sub_vec * (1.f + tanh_value);
    }
}

/// @brief Is the element wise op for tanh.
///
/// @tparam T Is the tile data type.
/// @tparam post_op Is the post processing op kind, should be tanh.
/// @param mat_Acc Is the reference of the tile object.
/// @return No return, update the data in-place.
template <typename T, post_kind post_op>
__XETLA_API typename std::enable_if_t<post_op == post_kind::tanh> elemwise_op(
        T &mat_Acc) {
    constexpr int elems = T::block_elems;
    constexpr int rounds = T::tile_elems / elems;
    using dtype = typename T::dtype;
#pragma unroll
    for (int i = 0; i < rounds; ++i) {
        auto sub_vec = mat_Acc.reg.xetla_select<elems, 1>(elems * i);
        sub_vec = xetla_tanh<dtype, elems>(sub_vec);
    }
    constexpr int remained_elems = T::tile_elems % elems;
    if constexpr (remained_elems != 0) {
        auto sub_vec = mat_Acc.reg.xetla_select<remained_elems, 1>(
                elems * (T::tile_elems / elems));
        sub_vec = xetla_tanh<dtype, remained_elems>(sub_vec);
    }
}

/// @brief Is the element wise op for sigmoid.
///
/// @tparam T Is the tile data type.
/// @tparam post_op Is the post processing op kind, should be sigmoid.
/// @param mat_Acc Is the reference of the tile object..
/// @return No return, update the data in-place.
template <typename T, post_kind post_op>
__XETLA_API static typename std::enable_if_t<post_op == post_kind::sigmoid>
elemwise_op(T &mat_Acc) {
    constexpr int elems = T::block_elems;
    constexpr int rounds = T::tile_elems / elems;
    constexpr float one = 1.0f;
#pragma unroll
    for (int i = 0; i < rounds; ++i) {
        auto sub_vec = mat_Acc.reg.xetla_select<elems, 1>(elems * i);
        xetla_mask<elems> mask = sub_vec >= 10;
        xetla_vector<typename T::dtype, elems> temp_vec
                = xetla_exp<typename T::dtype, elems>(sub_vec);
        xetla_vector<typename T::dtype, elems> sigmoid_value
                = temp_vec / (temp_vec + one);
        sigmoid_value.xetla_merge(1, mask);
        sub_vec = sigmoid_value;
    }
    constexpr int remained_elems = T::tile_elems % elems;
    if constexpr (remained_elems != 0) {
        auto sub_vec = mat_Acc.reg.xetla_select<remained_elems, 1>(
                elems * (T::tile_elems / elems));
        xetla_mask<remained_elems> mask = sub_vec >= 250;
        xetla_vector<typename T::dtype, remained_elems> temp_vec
                = xetla_exp<typename T::dtype, remained_elems>(sub_vec);
        xetla_vector<typename T::dtype, remained_elems> sigmoid_value
                = temp_vec / (temp_vec + one);
        sigmoid_value.xetla_merge(1, mask);
        sub_vec = sigmoid_value;
    }
}

/// @brief Converts tiled layout to vnni_tiled layout format.
///
/// @tparam T Is the tile data type.
/// @param mat_Acc Is the reference of the tile object.
/// @return No return, update the data in-place.
template <typename T>
__XETLA_API
        typename std::enable_if_t<T::register_layout == reg_layout::vnni_tiled>
        vnni_convert(T &mat_Acc) {
    constexpr uint32_t tile_size_y = T::tile_size_y;
    constexpr uint32_t tile_size_x = T::tile_size_x;
    constexpr uint32_t tile_elems = tile_size_y * tile_size_x;
    constexpr uint32_t block_size_y = T::block_size_y;
    constexpr uint32_t block_size_x = T::block_size_x;
    constexpr uint32_t block_elems = block_size_y * block_size_x;
    constexpr int32_t num_block_x = tile_size_x / block_size_x;
    using dtype = typename T::dtype;
    constexpr int32_t vnni_stride = sizeof(uint32_t) / sizeof(dtype);
    constexpr int32_t move_cols = block_size_x * vnni_stride;
    constexpr int32_t move_rows = block_size_y / vnni_stride;
    xetla_vector<dtype, tile_elems> rdst;
    static_assert(block_size_y % vnni_stride == 0, "vnni alignement check");
    if constexpr (tile_size_x == 1) { return; }
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<block_elems, 1>(
                                       (i * num_block_x + j) * block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>, block_size_y,
                    block_size_x>();
            auto reg_dst = rdst.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    move_rows, move_cols>();
#pragma unroll
            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d
                        .xetla_select<move_rows, 1, block_size_x, vnni_stride>(
                                0, vnni_i)
                        = reg_2d.xetla_select<move_rows, vnni_stride,
                                block_size_x, 1>(vnni_i, 0);
            }
        }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr int i = tile_size_y / block_size_y;
        constexpr uint32_t remain_elems_start = i * block_size_y * tile_size_x;
        constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
        constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;
        static_assert(
                remain_size_y % vnni_stride == 0, "vnni alignement check");
        constexpr int32_t remain_move_cols = block_size_x * vnni_stride;
        constexpr int32_t remain_move_rows = remain_size_y / vnni_stride;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<remain_block_elems, 1>(
                                       remain_elems_start
                                       + j * remain_block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>, remain_size_y,
                    block_size_x>();
            auto reg_dst = rdst.xetla_select<remain_block_elems, 1>(
                    remain_elems_start + j * remain_block_elems);
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    remain_move_rows, remain_move_cols>();
            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d.xetla_select<remain_move_rows, 1, block_size_x,
                        vnni_stride>(0, vnni_i)
                        = reg_2d.xetla_select<remain_move_rows, vnni_stride,
                                block_size_x, 1>(vnni_i, 0);
            }
        }
    }
    mat_Acc.reg = rdst;
}

/// @brief Converts vnni_tiled layout format to tiled layout.
///
/// @tparam T Is the tile data type.
/// @param mat_Acc Is the reference of the tile object.
/// @return No return, update the data in-place.
template <typename T>
__XETLA_API typename std::enable_if_t<T::register_layout == reg_layout::tiled>
vnni_reverse(T &mat_Acc) {
    constexpr uint32_t tile_size_y = T::tile_size_y;
    constexpr uint32_t tile_size_x = T::tile_size_x;
    constexpr uint32_t tile_elems = tile_size_y * tile_size_x;
    constexpr uint32_t block_size_y = T::block_size_y;
    constexpr uint32_t block_size_x = T::block_size_x;
    constexpr uint32_t block_elems = block_size_y * block_size_x;
    constexpr int32_t num_block_x = tile_size_x / block_size_x;
    using dtype = typename T::dtype;
    constexpr int32_t vnni_stride = sizeof(uint32_t) / sizeof(dtype);
    constexpr int32_t move_cols = block_size_x * vnni_stride;
    constexpr int32_t move_rows = block_size_y / vnni_stride;
    xetla_vector<dtype, tile_elems> rdst;
    static_assert(block_size_y % vnni_stride == 0, "vnni alignement check");
    if constexpr (tile_size_x == 1) { return; }
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<block_elems, 1>(
                                       (i * num_block_x + j) * block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>, move_rows,
                    move_cols>();
            auto reg_dst = rdst.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    block_size_y, block_size_x>();
#pragma unroll
            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d
                        .xetla_select<move_rows, vnni_stride, block_size_x, 1>(
                                vnni_i, 0)
                        = reg_2d.xetla_select<move_rows, 1, block_size_x,
                                vnni_stride>(0, vnni_i);
            }
        }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr int i = tile_size_y / block_size_y;
        constexpr uint32_t remain_elems_start = i * block_size_y * tile_size_x;
        constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
        constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;
        static_assert(
                remain_size_y % vnni_stride == 0, "vnni alignement check");
        constexpr int32_t remain_move_cols = block_size_x * vnni_stride;
        constexpr int32_t remain_move_rows = remain_size_y / vnni_stride;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<remain_block_elems, 1>(
                                       remain_elems_start
                                       + j * remain_block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>,
                    remain_move_rows, remain_move_cols>();
            auto reg_dst = rdst.xetla_select<remain_block_elems, 1>(
                    remain_elems_start + j * remain_block_elems);
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    remain_size_y, block_size_x>();
            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d.xetla_select<remain_move_rows, vnni_stride,
                        block_size_x, 1>(vnni_i, 0)
                        = reg_2d.xetla_select<remain_move_rows, 1, block_size_x,
                                vnni_stride>(0, vnni_i);
            }
        }
    }
    mat_Acc.reg = rdst;
}

/// @brief Converts vnni_tiled layout format to transpose_tiled layout.
///
/// @tparam T Is the tile data type.
/// @param mat_Acc Is the reference of the tile object.
/// @return No return, update the data in-place.
template <typename T>
__XETLA_API typename std::enable_if_t<T::register_layout
        == reg_layout::transpose_tiled>
vnni_reverse(T &mat_Acc) {
    constexpr uint32_t tile_size_y = T::tile_size_y;
    constexpr uint32_t tile_size_x = T::tile_size_x;
    constexpr uint32_t tile_elems = tile_size_y * tile_size_x;
    constexpr uint32_t block_size_y = T::block_size_y;
    constexpr uint32_t block_size_x = T::block_size_x;
    constexpr uint32_t block_elems = block_size_y * block_size_x;
    constexpr int32_t num_block_x = tile_size_x / block_size_x;
    using dtype = typename T::dtype;
    constexpr int32_t vnni_stride = sizeof(uint32_t) / sizeof(dtype);
    constexpr int32_t move_cols = block_size_y * vnni_stride;
    constexpr int32_t move_rows = block_size_x / vnni_stride;
    xetla_vector<dtype, tile_elems> rdst;
    static_assert(block_size_x % vnni_stride == 0, "vnni alignement check");
    if constexpr (tile_size_y == 1) { return; }
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<block_elems, 1>(
                                       (i * num_block_x + j) * block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>, move_rows,
                    move_cols>();
            auto reg_dst = rdst.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            //transpose
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    block_size_x, block_size_y>();
            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d
                        .xetla_select<move_rows, vnni_stride, block_size_y, 1>(
                                vnni_i, 0)
                        = reg_2d.xetla_select<move_rows, 1, block_size_y,
                                vnni_stride>(0, vnni_i);
            }
        }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr int i = tile_size_y / block_size_y;
        constexpr uint32_t remain_elems_start = i * block_size_y * tile_size_x;
        constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
        constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;
        constexpr int32_t remain_move_cols = remain_size_y * vnni_stride;
        constexpr int32_t remain_move_rows = block_size_x / vnni_stride;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<remain_block_elems, 1>(
                                       remain_elems_start
                                       + j * remain_block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>,
                    remain_move_rows, remain_move_cols>();
            auto reg_dst = rdst.xetla_select<remain_block_elems, 1>(
                    remain_elems_start + j * remain_block_elems);
            //transpose
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    block_size_x, remain_size_y>();

            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d.xetla_select<remain_move_rows, vnni_stride,
                        remain_size_y, 1>(vnni_i, 0)
                        = reg_2d.xetla_select<remain_move_rows, 1,
                                remain_size_y, vnni_stride>(0, vnni_i);
            }
        }
    }
    mat_Acc.reg = rdst;
}

/// @brief Changes vnni layout.
///
/// @tparam T_dst Is the destination tile data type.
/// @tparam T_src Is the source tile data type.
/// @param dst Is the reference of the destination tile object.
/// @param src Is the reference of the destination tile object.
/// @return No return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API
        typename std::enable_if_t<(T_src::block_size_y == T_dst::block_size_y)
                && (T_src::block_size_x == T_dst::block_size_x)
                && (T_src::tile_size_y == T_dst::tile_size_y)
                && (T_src::tile_size_x == T_dst::tile_size_x)>
        vnni_transform(T_dst &dst, T_src &src) {
    constexpr uint32_t tile_size_y = T_dst::tile_size_y;
    constexpr uint32_t tile_size_x = T_dst::tile_size_x;
    constexpr uint32_t tile_elems = tile_size_y * tile_size_x;
    constexpr uint32_t block_size_y = T_dst::block_size_y;
    constexpr uint32_t block_size_x = T_dst::block_size_x;
    constexpr uint32_t block_elems = block_size_y * block_size_x;
    constexpr int32_t num_block_x = tile_size_x / block_size_x;
    using dtype_dst = typename T_dst::dtype;
    using dtype_src = typename T_src::dtype;
    constexpr uint32_t vnni_row_src = sizeof(uint32_t) / sizeof(dtype_src);
    constexpr uint32_t vnni_row_dst = sizeof(uint32_t) / sizeof(dtype_dst);
    constexpr int32_t vnni_row
            = vnni_row_src > vnni_row_dst ? vnni_row_src : vnni_row_dst;
    static_assert(block_size_y % vnni_row == 0);
    static_assert(tile_size_y % vnni_row == 0);
    constexpr int32_t move_elems = vnni_row * block_size_x;
    xetla_vector<dtype_dst, tile_elems> reg_src
            = xetla_cvt<dtype_dst, dtype_src, tile_elems>(src.reg);
    if constexpr (sizeof(dtype_src) == sizeof(dtype_dst)) {
        dst.reg = reg_src;
        return;
    }
    xetla_vector<dtype_dst, tile_elems> reg_dst;
    constexpr uint32_t scale_factor
            = detail::gcd<vnni_row_src, vnni_row_dst>::value;
    using move_dtype = get_uint_type_t<sizeof(dtype_dst) * scale_factor>;
    constexpr uint32_t select_stride = vnni_row / scale_factor;
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_src_blk = reg_src.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            auto reg_dst_blk = reg_dst.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            for (int row_i = 0; row_i < block_size_y; row_i += vnni_row) {
                auto reg_src_move = reg_src_blk
                                            .xetla_select<move_elems, 1>(
                                                    row_i * block_size_x)
                                            .xetla_format<move_dtype>();
                auto reg_dst_move = reg_dst_blk
                                            .xetla_select<move_elems, 1>(
                                                    row_i * block_size_x)
                                            .xetla_format<move_dtype>();
#pragma unroll
                for (int move_i = 0; move_i < select_stride; move_i++) {
                    if constexpr (sizeof(dtype_dst) > sizeof(dtype_src)) {
                        reg_dst_move.xetla_select<block_size_x, 1>(
                                move_i * block_size_x)
                                = reg_src_move.xetla_select<block_size_x,
                                        select_stride>(move_i);
                    } else {
                        reg_dst_move.xetla_select<block_size_x, select_stride>(
                                move_i)
                                = reg_src_move.xetla_select<block_size_x, 1>(
                                        move_i * block_size_x);
                    }
                }
            }
        }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr int i = tile_size_y / block_size_y;
        constexpr uint32_t remain_elems_start = i * block_size_y * tile_size_x;
        constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
        constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_src_blk = reg_src.xetla_select<remain_block_elems, 1>(
                    remain_elems_start + j * remain_block_elems);
            auto reg_dst_blk = reg_dst.xetla_select<remain_block_elems, 1>(
                    remain_elems_start + j * remain_block_elems);
            // for mma, here we can guarantee that the remaining is a multiple of
            // vnni_row
            for (int row_i = 0; row_i < remain_size_y; row_i += vnni_row) {
                auto reg_src_move = reg_src_blk
                                            .xetla_select<move_elems, 1>(
                                                    row_i * block_size_x)
                                            .xetla_format<move_dtype>();
                auto reg_dst_move = reg_dst_blk
                                            .xetla_select<move_elems, 1>(
                                                    row_i * block_size_x)
                                            .xetla_format<move_dtype>();
#pragma unroll
                for (int move_i = 0; move_i < select_stride; move_i++) {
                    if constexpr (sizeof(dtype_dst) > sizeof(dtype_src)) {
                        reg_dst_move.xetla_select<block_size_x, 1>(
                                move_i * block_size_x)
                                = reg_src_move.xetla_select<block_size_x,
                                        select_stride>(move_i);
                    } else {
                        reg_dst_move.xetla_select<block_size_x, select_stride>(
                                move_i)
                                = reg_src_move.xetla_select<block_size_x, 1>(
                                        move_i * block_size_x);
                    }
                }
            }
        }
    }
    dst.reg = reg_dst;
}

/// @brief Broadcasts 1d src tile to the entire 2d tile, as well as do the data conversion.
///
/// @tparam T_dst Is the destination tile data type.
/// @tparam T_src Is the source tile data type, interpreted as 1D data.
/// @param dst Is the reference of the destination tile object.
/// @param src Is the reference of the destination tile object.
/// @return No return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API
        typename std::enable_if_t<(T_dst::register_layout == reg_layout::tiled)
                && (T_src::register_layout == reg_layout::tiled)
                && (T_src::tile_size_x == T_dst::tile_size_x)
                && (T_src::tile_size_y == 1)>
        row_broadcast(T_dst &dst, T_src &src) {
    static constexpr uint32_t dst_tile_size_y = T_dst::tile_size_y;
    static constexpr uint32_t dst_tile_size_x = T_dst::tile_size_x;
    static constexpr uint32_t dst_tile_elems = T_dst::tile_elems;
    static constexpr uint32_t dst_block_size_y = T_dst::block_size_y;
    static constexpr uint32_t dst_block_size_x = T_dst::block_size_x;
    static constexpr uint32_t dst_block_elems = T_dst::block_elems;
    static constexpr int32_t dst_num_block_y = T_dst::num_block_y;
    static constexpr int32_t dst_num_block_x = T_dst::num_block_x;
    using dst_dtype = typename T_dst::dtype;
    using src_dtype = typename T_src::dtype;

#pragma unroll
    for (int i = 0; i < dst_tile_size_y / dst_block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < dst_num_block_x; j++) {
            auto dst_reg = (dst.reg)
                                   .xetla_select<dst_block_elems, 1>(
                                           (i * dst_num_block_x + j)
                                           * dst_block_elems)
                                   .xetla_format<dst_dtype, dst_block_size_y,
                                           dst_block_size_x>();
#pragma unroll
            for (int row_i = 0; row_i < dst_block_size_y; row_i++) {
                auto src_reg = src.reg.xetla_select<dst_block_size_x, 1>(
                        j * dst_block_size_x);
                dst_reg.row(row_i)
                        = xetla_cvt<dst_dtype, src_dtype, dst_block_size_x>(
                                src_reg);
            }
        }
    }

    // process the tail
    if constexpr ((dst_tile_size_y % dst_block_size_y) != 0) {
        constexpr uint32_t tail_start_y
                = dst_tile_size_y / dst_block_size_y * dst_block_size_y;
        constexpr int32_t dst_tail_size_y = dst_tile_size_y % dst_block_size_y;
        constexpr int32_t dst_tail_block_elems
                = dst_tail_size_y * dst_block_size_x;
#pragma unroll
        for (int j = 0; j < dst_num_block_x; j++) {
            auto dst_reg = (dst.reg)
                                   .xetla_select<dst_tail_block_elems, 1>(
                                           tail_start_y * dst_tile_size_x
                                           + j * dst_tail_block_elems)
                                   .xetla_format<dst_dtype, dst_tail_size_y,
                                           dst_block_size_x>();
#pragma unroll
            for (int row_i = 0; row_i < dst_tail_size_y; row_i++) {
                auto src_reg = src.reg.xetla_select<dst_block_size_x, 1>(
                        j * dst_block_size_x);
                dst_reg.row(row_i)
                        = xetla_cvt<dst_dtype, src_dtype, dst_block_size_x>(
                                src_reg);
            }
        }
    }
}

} // namespace gpu::xetla::subgroup
