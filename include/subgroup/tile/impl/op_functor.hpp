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
#include "subgroup/tile/impl/load_xe.hpp"
#include "subgroup/tile/impl/payload_xe.hpp"
#include "subgroup/tile/impl/prefetch_xe.hpp"
#include "subgroup/tile/impl/store_xe.hpp"
namespace gpu::xetla::subgroup {

/// @brief Is none op functor, for placeholder purpose.
/// Used in epilogue::tile_op or chained_tile_op.
struct none_op_t {
    struct arguments_t {};
    template <typename matAcc_t, typename coord_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {}
};

/// @brief Is the element-wise relu op functor.
/// Get the relu input from matAcc, update the the relu output in place,
/// Used in epilogue::tile_op or chained_tile_op.
struct relu_op_t {
    struct arguments_t {};
    template <typename matAcc_t, typename coord_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        xetla_mask<matAcc_t::tile_elems> mask = matAcc.reg <= 0;
        matAcc.reg.xetla_merge(0, mask);
    }
};

/// @brief Is the element-wise tanh op functor.
/// Get the tanh input from matAcc, update the the tanh output in place,
/// Used in epilogue::tile_op or chained_tile_op.
struct tanh_op_t {
    struct arguments_t {};
    template <typename matAcc_t, typename coord_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        constexpr int elems = matAcc_t::tile_desc::block_elems;
        constexpr int rounds = matAcc_t::tile_desc::tile_elems / elems;
        using dtype = typename matAcc_t::dtype;
#pragma unroll
        for (int i = 0; i < rounds; ++i) {
            auto sub_vec = matAcc.reg.xetla_select<elems, 1>(elems * i);
            sub_vec = xetla_tanh<dtype, elems>(sub_vec);
        }
        constexpr int remained_elems = matAcc_t::tile_desc::tile_elems % elems;
        if constexpr (remained_elems != 0) {
            auto sub_vec = matAcc.reg.xetla_select<remained_elems, 1>(
                    elems * (matAcc_t::tile_elems / elems));
            sub_vec = xetla_tanh<dtype, remained_elems>(sub_vec);
        }
    }
};

/// @brief Is the element-wise sigmoid op functor.
/// Get the sigmoid input from matAcc, update the the sigmoid output in place,
/// Used in epilogue::tile_op or chained_tile_op.
struct sigmoid_op_t {
    struct arguments_t {};
    template <typename matAcc_t, typename coord_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        constexpr int elems = matAcc_t::tile_desc::block_elems;
        constexpr int rounds = matAcc_t::tile_desc::tile_elems / elems;
        constexpr float one = 1.0f;
#pragma unroll
        for (int i = 0; i < rounds; ++i) {
            auto sub_vec = matAcc.reg.xetla_select<elems, 1>(elems * i);
            xetla_mask<elems> mask = sub_vec >= 10;
            xetla_vector<typename matAcc_t::dtype, elems> temp_vec
                    = xetla_exp<typename matAcc_t::dtype, elems>(sub_vec);
            xetla_vector<typename matAcc_t::dtype, elems> sigmoid_value
                    = temp_vec / (temp_vec + one);
            sigmoid_value.xetla_merge(1, mask);
            sub_vec = sigmoid_value;
        }
        constexpr int remained_elems = matAcc_t::tile_desc::tile_elems % elems;
        if constexpr (remained_elems != 0) {
            auto sub_vec = matAcc.reg.xetla_select<remained_elems, 1>(
                    elems * (matAcc_t::tile_elems / elems));
            xetla_mask<remained_elems> mask = sub_vec >= 250;
            xetla_vector<typename matAcc_t::dtype, remained_elems> temp_vec
                    = xetla_exp<typename matAcc_t::dtype, remained_elems>(
                            sub_vec);
            xetla_vector<typename matAcc_t::dtype, remained_elems> sigmoid_value
                    = temp_vec / (temp_vec + one);
            sigmoid_value.xetla_merge(1, mask);
            sub_vec = sigmoid_value;
        }
    }
};

/// @brief Is the element-wise gelu inference forward op functor.
/// Get the gelu input from matAcc, update the the gelu output in place,
/// Used in epilogue::tile_op or chained_tile_op.
struct gelu_fwd_op_t {
    struct arguments_t {};
    template <typename matAcc_t, typename coord_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        using dtype = typename matAcc_t::dtype;
        constexpr dtype C0 = 0.044715f;
        constexpr dtype sqrt_two_over_pi = 0.79788458347320556640625f;
        // total flag register
        constexpr int elems = 8 * 16;
        constexpr int rounds = matAcc_t::tile_elems / elems;
#pragma unroll
        for (int i = 0; i < rounds; ++i) {
            auto sub_vec = matAcc.reg.xetla_select<elems, 1>(elems * i);
            xetla_vector<dtype, elems> sub_vec_x = (sqrt_two_over_pi * sub_vec
                    * (1.f + C0 * sub_vec * sub_vec));
            xetla_vector<dtype, elems> tanh_value
                    = xetla_tanh<dtype, elems>(sub_vec_x);
            sub_vec = 0.5f * sub_vec * (1.f + tanh_value);
        }
        constexpr int remained_elems = matAcc_t::tile_elems % elems;
        if constexpr (remained_elems != 0) {
            auto sub_vec = matAcc.reg.xetla_select<remained_elems, 1>(
                    elems * (matAcc_t::tile_elems / elems));
            xetla_vector<dtype, remained_elems> sub_vec_x = (sqrt_two_over_pi
                    * sub_vec * (1.f + C0 * sub_vec * sub_vec));
            xetla_vector<dtype, remained_elems> tanh_value
                    = xetla_tanh<dtype, remained_elems>(sub_vec_x);
            sub_vec = 0.5f * sub_vec * (1.f + tanh_value);
        }
    }
};

/// @brief Is the element-wise gelu training forward op functor.
/// Get the gelu input from matAcc, update the the gelu output in place,
/// and dump the intermediate buffer_w to memory for backward purpose.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_out Is the data type of the intermediate buffer_w.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_out, gpu_arch arch_tag = gpu_arch::Xe>
struct gelu_fwd_w_op_t {};
/// @brief Is the element-wise gelu training forward op functor, specialized for Xe architecture.
template <typename dtype_out_>
struct gelu_fwd_w_op_t<dtype_out_, gpu_arch::Xe> {
    using dtype_out = dtype_out_;
    using mem_desc_w_t
            = mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>;
    using shape_t = typename mem_desc_w_t::shape_t;
    using coord_t = typename mem_desc_w_t::coord_t;
    using base_t = typename mem_desc_w_t::base_t;

    struct arguments_t {
        shape_t shape;
        base_t base;
        inline arguments_t() = default;
        inline arguments_t(base_t base_, shape_t shape_)
            : base(base_), shape(shape_) {}
    };
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        using dtype_acc = typename matAcc_t::dtype;
        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;

        mem_desc_w_t mem_desc_w(args.base, args.shape, coord);
        using bwd_w_tile_desc_t = tile_desc_t<block_size_x, block_size_y,
                block_size_x, block_size_y, reg_layout::tiled>;
        using bwd_w_tile_t = tile_t<dtype_out, bwd_w_tile_desc_t>;
        using bwd_w_payload_t = mem_payload_t<dtype_out, bwd_w_tile_desc_t,
                msg_type::block_2d, mem_desc_w_t::layout, mem_desc_w_t::space,
                gpu_arch::Xe>;
        bwd_w_tile_t bwd_w;
        bwd_w_payload_t bwd_w_payload(mem_desc_w);
        // start compute
        constexpr dtype_acc c0 = 0.044715f;
        constexpr dtype_acc d0 = 0.134145f;
        constexpr dtype_acc sqrt_two_over_pi = 0.79788458347320556640625f;
        constexpr uint32_t block_elems = matAcc_t::block_elems;
        constexpr uint32_t num_block_x = matAcc_t::num_block_x;
#pragma unroll
        for (int i = 0; i < tile_size_y / block_size_y; ++i) {
#pragma unroll
            for (int j = 0; j < num_block_x; ++j) {
                auto x = matAcc.reg.xetla_select<block_elems, 1>(
                        block_elems * (i * num_block_x + j));
                xetla_vector<dtype_acc, block_elems> z
                        = xetla_tanh<dtype_acc, block_elems>(
                                sqrt_two_over_pi * (x + c0 * x * x * x));
                xetla_vector<dtype_acc, block_elems> w = (0.5f * (1.f + z)
                        + 0.5f * x * (1.f - z * z)
                                * (sqrt_two_over_pi * (1.f + d0 * x * x)));
                x = 0.5f * x * (1.f + z);
                bwd_w.reg = xetla_cvt<dtype_out, dtype_acc, block_elems>(w);
                tile_store<cache_hint::uncached>(bwd_w, bwd_w_payload);
                bwd_w_payload.template update_tdesc<tdesc_update_dir::x_dir>(
                        block_size_x);
            }
            bwd_w_payload.template update_tdesc<tdesc_update_dir::x_dir>(
                    -1 * tile_size_x);
            bwd_w_payload.template update_tdesc<tdesc_update_dir::y_dir>(
                    block_size_y);
        }
        if constexpr (tile_size_y % block_size_y != 0) {
            constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
            constexpr uint32_t remain_y_start
                    = tile_size_y / block_size_y * block_size_y;
            constexpr uint32_t remain_elems_start
                    = remain_y_start * tile_size_x;
            constexpr uint32_t remain_block_elems
                    = remain_size_y * block_size_x;
            //     using remain_bwd_w_tile_t = xetla_tile_store_t<dtype_out,
            //             block_size_x, remain_size_y, block_size_x, remain_size_y,
            //             mem_desc_w_t::layout, mem_desc_w_t::space, store_op::normal,
            //             gpu_arch::Xe, reg_layout::tiled>;
            using remain_bwd_w_tile_desc_t
                    = tile_desc_t<block_size_x, remain_size_y, block_size_x,
                            remain_size_y, reg_layout::tiled>;
            using remain_bwd_w_tile_t
                    = tile_t<dtype_out, remain_bwd_w_tile_desc_t>;
            using remain_bwd_w_payload_t = mem_payload_t<dtype_out,
                    remain_bwd_w_tile_desc_t, msg_type::block_2d,
                    mem_desc_w_t::layout, mem_desc_w_t::space, gpu_arch::Xe>;

            mem_desc_w.update_coord_y(remain_y_start);
            remain_bwd_w_payload_t remain_bwd_w_payload(mem_desc_w);
            remain_bwd_w_tile_t remain_bwd_w;
#pragma unroll
            for (int j = 0; j < num_block_x; ++j) {
                auto x = matAcc.reg.xetla_select<remain_block_elems, 1>(
                        remain_elems_start + remain_block_elems * j);
                xetla_vector<dtype_acc, remain_block_elems> z
                        = xetla_tanh<dtype_acc, remain_block_elems>(
                                sqrt_two_over_pi * (x + c0 * x * x * x));
                xetla_vector<dtype_acc, remain_block_elems> w = (0.5f
                                * (1.f + z)
                        + 0.5f * x * (1.f - z * z)
                                * (sqrt_two_over_pi * (1.f + d0 * x * x)));
                x = 0.5f * x * (1.f + z);
                remain_bwd_w.reg
                        = xetla_cvt<dtype_out, dtype_acc, remain_block_elems>(
                                w);
                tile_store<cache_hint::uncached>(
                        remain_bwd_w, remain_bwd_w_payload);
                remain_bwd_w_payload
                        .template update_tdesc<tdesc_update_dir::x_dir>(
                                block_size_x);
            }
        }
    }
};

/// @brief Is the element-wise gelu backward op functor.
/// Load the gelu forward input buffer from memory and get the gradient data from matAcc, update the output in place.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_in Is the data type of the gelu forward input buffer.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_in, gpu_arch arch_tag = gpu_arch::Xe>
struct gelu_bwd_op_t {};
/// @brief Is the element-wise gelu backward op functor, specialized for Xe architecture.
template <typename dtype_in_>
struct gelu_bwd_op_t<dtype_in_, gpu_arch::Xe> {
    using dtype_in = dtype_in_;
    using mem_desc_x_t
            = mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
    using shape_t = typename mem_desc_x_t::shape_t;
    using coord_t = typename mem_desc_x_t::coord_t;
    using base_t = typename mem_desc_x_t::base_t;
    struct arguments_t {
        shape_t shape;
        base_t base;
        inline arguments_t() = default;
        inline arguments_t(base_t base_, shape_t shape_)
            : base(base_), shape(shape_) {}
    };
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        using dtype_acc = typename matAcc_t::dtype;
        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;

        using bwd_x_tile_desc_t = tile_desc_t<tile_size_x, tile_size_y,
                block_size_x, block_size_y, reg_layout::tiled>;
        using bwd_x_tile_t = tile_t<dtype_in, bwd_x_tile_desc_t>;
        using bwd_x_payload_t = mem_payload_t<dtype_in, bwd_x_tile_desc_t,
                msg_type::block_2d, mem_desc_x_t::layout, mem_desc_x_t::space,
                gpu_arch::Xe>;
        using bwd_x_acc_tile_t = tile_t<dtype_acc, bwd_x_tile_desc_t>;
        bwd_x_tile_t bwd_x;
        bwd_x_acc_tile_t bwd_x_acc;
        // init tdesc
        mem_desc_x_t mem_desc_x(args.base, args.shape, coord);
        bwd_x_payload_t bwd_x_payload(mem_desc_x);
        tile_load<cache_hint::cached, cache_hint::cached>(bwd_x, bwd_x_payload);
        elemwise_cvt(bwd_x_acc, bwd_x);
        // start compute
        constexpr dtype_acc c0 = 0.044715f;
        constexpr dtype_acc d0 = 0.134145f;
        constexpr dtype_acc sqrt_two_over_pi = 0.79788458347320556640625f;
        constexpr uint32_t block_elems = matAcc_t::block_elems;
        constexpr uint32_t num_block_x = matAcc_t::num_block_x;
#pragma unroll
        for (int i = 0; i < tile_size_y / block_size_y; ++i) {
#pragma unroll
            for (int j = 0; j < num_block_x; ++j) {
                auto x = bwd_x_acc.reg.xetla_select<block_elems, 1>(
                        block_elems * (i * num_block_x + j));
                auto dy = matAcc.reg.xetla_select<block_elems, 1>(
                        block_elems * (i * num_block_x + j));
                xetla_vector<dtype_acc, block_elems> z
                        = xetla_tanh<dtype_acc, block_elems>(
                                sqrt_two_over_pi * (x + c0 * x * x * x));
                xetla_vector<dtype_acc, block_elems> w = (0.5f * (1.f + z)
                        + 0.5f * x * (1.f - z * z)
                                * (sqrt_two_over_pi * (1.f + d0 * x * x)));
                dy = w * dy;
            }
        }
        if constexpr (tile_size_y % block_size_y != 0) {
            constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
            constexpr uint32_t remain_y_start
                    = tile_size_y / block_size_y * block_size_y;
            constexpr uint32_t remain_elems_start
                    = remain_y_start * tile_size_x;
            constexpr uint32_t remain_block_elems
                    = remain_size_y * block_size_x;
#pragma unroll
            for (int j = 0; j < num_block_x; ++j) {
                auto x = bwd_x_acc.reg.xetla_select<remain_block_elems, 1>(
                        remain_elems_start + remain_block_elems * j);
                auto dy = matAcc.reg.xetla_select<remain_block_elems, 1>(
                        remain_elems_start + remain_block_elems * j);
                xetla_vector<dtype_acc, remain_block_elems> z
                        = xetla_tanh<dtype_acc, remain_block_elems>(
                                sqrt_two_over_pi * (x + c0 * x * x * x));
                xetla_vector<dtype_acc, remain_block_elems> w = (0.5f
                                * (1.f + z)
                        + 0.5f * x * (1.f - z * z)
                                * (sqrt_two_over_pi * (1.f + d0 * x * x)));
                dy = w * dy;
            }
        }
    }
};

/// @brief Is the bias_add op functor.
/// Load the 1d bias data from memory and get the input from matAcc, update the output in place.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_bias Is the data type of bias buffer.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_bias, gpu_arch arch_tag = gpu_arch::Xe>
struct bias_add_op_t {};
/// @brief Is the bias_add op functor, specialized for Xe architecture.
template <typename dtype_bias_>
struct bias_add_op_t<dtype_bias_, gpu_arch::Xe> {
    using dtype_bias = dtype_bias_;
    using mem_desc_bias_t
            = mem_desc_t<dtype_bias, mem_layout::row_major, mem_space::global>;
    using shape_t = typename mem_desc_bias_t::shape_t;
    using coord_t = typename mem_desc_bias_t::coord_t;
    using base_t = typename mem_desc_bias_t::base_t;

    struct arguments_t {
        shape_t shape;
        base_t base;
        inline arguments_t() = default;
        inline arguments_t(base_t base_, shape_t shape_)
            : base(base_), shape(shape_) {}
    };
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        using dtype_acc = typename matAcc_t::dtype;
        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
        static constexpr int32_t num_block_x = matAcc_t::num_block_x;
        static constexpr int32_t num_block_y = matAcc_t::num_block_y;
        static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
        static constexpr uint32_t block_elems = matAcc_t::block_elems;

        using bias_tile_desc_t = tile_desc_t<tile_size_x, 1, block_size_x, 1,
                reg_layout::tiled>;
        using bias_t = tile_t<dtype_bias, bias_tile_desc_t>;
        using bias_payload_t = mem_payload_t<dtype_bias, bias_tile_desc_t,
                msg_type_v<bias_tile_desc_t, mem_desc_bias_t::space>,
                mem_desc_bias_t::layout, mem_desc_bias_t::space, gpu_arch::Xe>;
        coord_t bias_coord(coord.x, 0);
        mem_desc_bias_t mem_desc_bias(args.base, args.shape, bias_coord);
        bias_t bias;
        bias_payload_t bias_payload(mem_desc_bias);
        tile_load<cache_hint::cached, cache_hint::cached>(bias, bias_payload);

#pragma unroll
        for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
            for (int j = 0; j < num_block_x; j++) {
                auto dst_reg
                        = matAcc.reg
                                  .xetla_select<block_elems, 1>(
                                          (i * num_block_x + j) * block_elems)
                                  .xetla_format<dtype_acc, block_size_y,
                                          block_size_x>();
#pragma unroll
                for (int row_i = 0; row_i < block_size_y; row_i++) {
                    auto src_reg = bias.reg.xetla_select<block_size_x, 1>(
                            j * block_size_x);
                    dst_reg.row(row_i)
                            = xetla_cvt<dtype_acc, dtype_bias, block_size_x>(
                                      src_reg)
                            + dst_reg.row(row_i);
                }
            }
        }
        // process the tail
        if constexpr ((tile_size_y % block_size_y) != 0) {
            constexpr uint32_t tail_start_y
                    = tile_size_y / block_size_y * block_size_y;
            constexpr int32_t tail_size_y = tile_size_y % block_size_y;
            constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
            for (int j = 0; j < num_block_x; j++) {
                auto dst_reg = matAcc.reg
                                       .xetla_select<tail_block_elems, 1>(
                                               tail_start_y * tile_size_x
                                               + j * tail_block_elems)
                                       .xetla_format<dtype_acc, tail_size_y,
                                               block_size_x>();
#pragma unroll
                for (int row_i = 0; row_i < tail_size_y; row_i++) {
                    auto src_reg = bias.reg.xetla_select<block_size_x, 1>(
                            j * block_size_x);
                    dst_reg.row(row_i)
                            = xetla_cvt<dtype_acc, dtype_bias, block_size_x>(
                                      src_reg)
                            + dst_reg.row(row_i);
                }
            }
        }
    }
};

/// @brief Is the element-wise reduce op functor.
/// Load one buffer from memory and get another from matAcc,
/// element-wise reduce and update the output in place.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam reduce_kind Is the reduce type, can be sum, prod, min and max.
/// @tparam dtype_in Is the memory side buffer data type.
/// @tparam arch_tag Is the hardware architecture tag.
template <reduce_op reduce_kind, typename dtype_in,
        gpu_arch arch_tag = gpu_arch::Xe>
struct elemwise_reduce_op_t {};
/// @brief Is the element-wise reduce op functor, specialized for Xe architecture.
template <reduce_op reduce_kind_, typename dtype_in_>
struct elemwise_reduce_op_t<reduce_kind_, dtype_in_, gpu_arch::Xe> {
    using dtype_in = dtype_in_;
    using mem_desc_in_t
            = mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
    using shape_t = typename mem_desc_in_t::shape_t;
    using coord_t = typename mem_desc_in_t::coord_t;
    using base_t = typename mem_desc_in_t::base_t;
    static constexpr reduce_op reduce_kind = reduce_kind_;

    struct arguments_t {
        shape_t shape;
        base_t base;
        inline arguments_t() = default;
        inline arguments_t(base_t base_, shape_t shape_)
            : base(base_), shape(shape_) {}
    };
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        using dtype_acc = typename matAcc_t::dtype;
        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
        static constexpr int32_t num_block_x = matAcc_t::num_block_x;
        static constexpr int32_t num_block_y = matAcc_t::num_block_y;
        static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
        static constexpr uint32_t block_elems = matAcc_t::block_elems;

        using mat_in_tile_desc_t = tile_desc_t<tile_size_x, tile_size_y,
                block_size_x, block_size_y, reg_layout::tiled>;
        using mat_in_tile_t = tile_t<dtype_in, mat_in_tile_desc_t>;
        using mat_in_payload_t = mem_payload_t<dtype_in, mat_in_tile_desc_t,
                msg_type_v<mat_in_tile_desc_t, mem_desc_in_t::space>,
                mem_desc_in_t::layout, mem_desc_in_t::space, gpu_arch::Xe>;
        using mat_in_tile_acc_t = tile_t<dtype_acc, mat_in_tile_desc_t>;
        mem_desc_in_t mem_desc_in(args.base, args.shape, coord);
        mat_in_tile_t mat_in;
        mat_in_payload_t mat_in_payload(mem_desc_in);
        tile_load<cache_hint::cached, cache_hint::cached>(
                mat_in, mat_in_payload);
        mat_in_tile_acc_t mat_in_acc;
        elemwise_cvt(mat_in_acc, mat_in);

#pragma unroll
        for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
            for (int j = 0; j < num_block_x; j++) {
                auto dst_reg = matAcc.reg.xetla_select<block_elems, 1>(
                        (i * num_block_x + j) * block_elems);
                auto src_reg = mat_in_acc.reg.xetla_select<block_elems, 1>(
                        (i * num_block_x + j) * block_elems);
                dst_reg = reduce_helper<reduce_kind, dtype_acc, block_elems>(
                        src_reg, dst_reg);
            }
        }
        // process the tail
        if constexpr ((tile_size_y % block_size_y) != 0) {
            constexpr uint32_t tail_start_y
                    = tile_size_y / block_size_y * block_size_y;
            constexpr int32_t tail_size_y = tile_size_y % block_size_y;
            constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
            for (int j = 0; j < num_block_x; j++) {
                auto dst_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
                        tail_start_y * tile_size_x + j * tail_block_elems);
                auto src_reg = mat_in_acc.reg.xetla_select<tail_block_elems, 1>(
                        tail_start_y * tile_size_x + j * tail_block_elems);
                dst_reg = reduce_helper<reduce_kind, dtype_acc,
                        tail_block_elems>(src_reg, dst_reg);
            }
        }
    }
};

/// @brief Is the dropout op functor.
/// Load the mask from memory and get input from matAcc,
/// do the scaling and zero out, update the output in place.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_mask Is the mask data type.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_mask = uint8_t, gpu_arch arch_tag = gpu_arch::Xe>
struct dropout_op_t {};
/// @brief Is the dropout op functor, specialized for Xe architecture.
template <typename dtype_mask_>
struct dropout_op_t<dtype_mask_, gpu_arch::Xe> {
    using dtype_mask = dtype_mask_;
    using mem_desc_mask_t
            = mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>;
    using shape_t = typename mem_desc_mask_t::shape_t;
    using coord_t = typename mem_desc_mask_t::coord_t;
    using base_t = typename mem_desc_mask_t::base_t;
    static constexpr uint32_t num_flag = 4;
    static constexpr uint32_t unroll_size = num_flag * 16;
    struct arguments_t {
        shape_t shape;
        base_t base;
        float prob;
        float scale;
        inline arguments_t() = default;
        inline arguments_t(
                base_t base_, shape_t shape_, float prob_, float scale_)
            : base(base_), shape(shape_), prob(prob_), scale(scale_) {}
    };

    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        using dtype_acc = typename matAcc_t::dtype;
        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
        static constexpr int32_t num_block_x = matAcc_t::num_block_x;
        static constexpr int32_t num_block_y = matAcc_t::num_block_y;
        static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
        static constexpr uint32_t block_elems = matAcc_t::block_elems;
        if (args.prob == 0) { return; }
        using mask_in_tile_desc_t = tile_desc_t<tile_size_x, tile_size_y,
                block_size_x, block_size_y, reg_layout::tiled>;
        using mask_in_tile_t = tile_t<dtype_mask, mask_in_tile_desc_t>;
        using mask_in_payload_t = mem_payload_t<dtype_mask, mask_in_tile_desc_t,
                msg_type_v<mask_in_tile_desc_t, mem_desc_mask_t::space>,
                mem_desc_mask_t::layout, mem_desc_mask_t::space, gpu_arch::Xe>;
        mem_desc_mask_t mem_desc_mask(args.base, args.shape, coord);
        mask_in_tile_t mask_in;
        mask_in_payload_t mask_in_payload(mem_desc_mask);
        tile_load<cache_hint::cached, cache_hint::cached>(
                mask_in, mask_in_payload);
        matAcc.reg = matAcc.reg * args.scale;
#pragma unroll
        for (int i = 0; i < tile_elems / unroll_size; i++) {
            xetla_mask<unroll_size> mask_flag
                    = mask_in.reg.xetla_select<unroll_size, 1>(i * unroll_size)
                    > 0;
            matAcc.reg.xetla_select<unroll_size, 1>(i * unroll_size)
                    .xetla_merge(0, mask_flag);
        }
        if constexpr (tile_elems % unroll_size != 0) {
            constexpr uint32_t remain_len = tile_elems % unroll_size;
            constexpr uint32_t remain_start
                    = tile_elems / unroll_size * unroll_size;
            xetla_mask<remain_len> mask_flag
                    = mask_in.reg.xetla_select<remain_len, 1>(remain_start) > 0;
            matAcc.reg.xetla_select<remain_len, 1>(remain_start)
                    .xetla_merge(0, mask_flag);
        }
    }
};

/// @brief Is the random number generator and dropout op functor.
/// Generate the mask data and get input from matAcc, do the scaling and zero out,
/// update the output in place, dump the mask buffer to memory.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_mask Is the mask data type.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_mask = uint8_t, gpu_arch arch_tag = gpu_arch::Xe>
struct rng_dropout_op_t {};
/// @brief Is the random number generator and dropout op functor, specialized for Xe architecture.
template <typename dtype_mask_>
struct rng_dropout_op_t<dtype_mask_, gpu_arch::Xe> {
    using dtype_mask = dtype_mask_;
    using mem_desc_mask_t
            = mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>;
    using shape_t = typename mem_desc_mask_t::shape_t;
    using coord_t = typename mem_desc_mask_t::coord_t;
    using base_t = typename mem_desc_mask_t::base_t;
    static constexpr uint32_t random_simd = 16;
    static constexpr uint32_t random_len = 4 * random_simd;
    xetla_rand_t<random_simd> rand_gen;

    struct arguments_t {
        shape_t mask_shape;
        base_t mask_base;
        uint64_t *rand_offset_ptr;
        float prob;
        uint64_t rand_seed;

        inline arguments_t() = default;
        inline arguments_t(base_t mask_base_, shape_t mask_shape_, float prob_,
                uint64_t *rand_offset_ptr_,
                uint64_t rand_seed_ = 67280421310721)
            : mask_base(mask_base_)
            , mask_shape(mask_shape_)
            , prob(prob_)
            , rand_offset_ptr(rand_offset_ptr_)
            , rand_seed(rand_seed_) {}
    };

    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        using dtype_acc = typename matAcc_t::dtype;
        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
        static constexpr int32_t num_block_x = matAcc_t::num_block_x;
        static constexpr int32_t num_block_y = matAcc_t::num_block_y;
        static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
        static constexpr uint32_t block_elems = matAcc_t::block_elems;

        using mask_out_tile_desc_t = tile_desc_t<tile_size_x, tile_size_y,
                block_size_x, block_size_y, reg_layout::tiled>;
        using mask_out_tile_t = tile_t<dtype_mask, mask_out_tile_desc_t>;
        using mask_out_payload_t = mem_payload_t<dtype_mask,
                mask_out_tile_desc_t,
                msg_type_v<mask_out_tile_desc_t, mem_desc_mask_t::space>,
                mem_desc_mask_t::layout, mem_desc_mask_t::space, gpu_arch::Xe>;
        if (args.prob == 0) { return; }
        //calculate the scale internally
        float scale = 1.f / (1.f - args.prob);
        uint32_t threshold = uint32_t(args.prob * float(4294967296));
        xetla_vector<uint64_t, 1> rand_offset_v
                = xetla_load_global<uint64_t, 1, data_size::default_size,
                        cache_hint::cached, cache_hint::cached>(
                        args.rand_offset_ptr, 0);
        uint64_t rand_offset = rand_offset_v[0];
        uint64_t rand_subseq = uint64_t(coord.y) << 32 | uint64_t(coord.x);
        rand_gen.init(args.rand_seed, rand_subseq, rand_offset);

        xetla_vector<dtype_mask, tile_elems> mask;
        matAcc.reg = matAcc.reg * scale;
#pragma unroll
        for (int i = 0; i < tile_elems / random_len; i++) {
            auto out_sub
                    = matAcc.reg.xetla_select<random_len, 1>(i * random_len);
            auto mask_sub = mask.xetla_select<random_len, 1>(i * random_len);
            xetla_vector<uint32_t, random_len> rand_val = rand_gen.rand();
            xetla_mask<random_len> mask_flag = rand_val < threshold;
            out_sub.xetla_merge(0, mask_flag);
            mask_sub.xetla_merge(1, 0, mask_flag);
        }
        if constexpr (tile_elems % random_len != 0) {
            constexpr uint32_t remain_len = tile_elems % random_len;
            constexpr uint32_t remain_start
                    = tile_elems / random_len * random_len;
            auto out_sub = matAcc.reg.xetla_select<remain_len, 1>(remain_start);
            auto mask_sub = mask.xetla_select<remain_len, 1>(remain_start);
            // dropout, still generate random_len
            xetla_vector<uint32_t, random_len> rand_val = rand_gen.rand();
            xetla_mask<random_len> mask_flag = rand_val < threshold;
            out_sub.xetla_merge(0, mask_flag.xetla_select<remain_len, 1>(0));
            mask_sub.xetla_merge(
                    1, 0, mask_flag.xetla_select<remain_len, 1>(0));
        }
        mem_desc_mask_t mem_desc_mask(args.mask_base, args.mask_shape, coord);
        mask_out_tile_t mask_out;
        mask_out_payload_t mask_out_payload(mem_desc_mask);
        mask_out.reg = mask;
        tile_store<cache_hint::streaming>(mask_out, mask_out_payload);
    }
};

/// @brief Is the quantization op functor.
/// Load 1d scale and offset from memory and get input from matAcc, do the quantization,
/// update the output to matC.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_offset_scale Is the offset and scale data type.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_offset_scale, gpu_arch arch_tag = gpu_arch::Xe>
struct quant_op_t {};
/// @brief Is the quantization op functor, specialized for Xe architecture.
template <typename dtype_offset_scale_>
struct quant_op_t<dtype_offset_scale_, gpu_arch::Xe> {
    using dtype_offset_scale = dtype_offset_scale_;
    using mem_desc_scale_t = mem_desc_t<dtype_offset_scale,
            mem_layout::row_major, mem_space::global>;
    using mem_desc_offset_t = mem_desc_t<dtype_offset_scale,
            mem_layout::row_major, mem_space::global>;
    using shape_t = typename mem_desc_offset_t::shape_t;
    using coord_t = typename mem_desc_offset_t::coord_t;
    using base_t = typename mem_desc_offset_t::base_t;

    struct arguments_t {
        shape_t shape_offset_scale;
        base_t base_scale, base_offset;
        inline arguments_t() = default;
        inline arguments_t(base_t base_scale_, base_t base_offset_,
                shape_t shape_offset_scale_)
            : base_scale(base_scale_)
            , base_offset(base_offset_)
            , shape_offset_scale(shape_offset_scale_) {}
    };
    template <typename matC_t, typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matC_t &matC, matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args) {
        using dtype_acc = typename matAcc_t::dtype;
        using dtype_out = typename matC_t::dtype;
        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
        static constexpr int32_t num_block_x = matAcc_t::num_block_x;
        static constexpr int32_t num_block_y = matAcc_t::num_block_y;
        static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
        static constexpr uint32_t block_elems = matAcc_t::block_elems;

        using scale_tile_desc_t = tile_desc_t<tile_size_x, 1, block_size_x, 1,
                reg_layout::tiled>;
        using scale_tile_t = tile_t<dtype_offset_scale, scale_tile_desc_t>;
        using scale_payload_t
                = mem_payload_t<dtype_offset_scale, scale_tile_desc_t,
                        msg_type_v<scale_tile_desc_t, mem_desc_scale_t::space>,
                        mem_desc_scale_t::layout, mem_desc_scale_t::space,
                        gpu_arch::Xe>;
        coord_t scale_coord(coord.x, 0);
        mem_desc_scale_t mem_desc_scale(
                args.base_scale, args.shape_offset_scale, scale_coord);
        scale_tile_t scale;
        scale_payload_t scale_payload(mem_desc_scale);
        tile_load<cache_hint::cached, cache_hint::cached>(scale, scale_payload);

        using offset_tile_desc_t = tile_desc_t<tile_size_x, 1, block_size_x, 1,
                reg_layout::tiled>;
        using offset_tile_t = tile_t<dtype_offset_scale, offset_tile_desc_t>;
        using offset_payload_t = mem_payload_t<dtype_offset_scale,
                offset_tile_desc_t,
                msg_type_v<offset_tile_desc_t, mem_desc_offset_t::space>,
                mem_desc_offset_t::layout, mem_desc_offset_t::space,
                gpu_arch::Xe>;
        coord_t offset_coord(coord.x, 0);
        mem_desc_offset_t mem_desc_offset(
                args.base_offset, args.shape_offset_scale, offset_coord);
        offset_tile_t offset;
        offset_payload_t offset_payload(mem_desc_offset);
        tile_load<cache_hint::cached, cache_hint::cached>(
                offset, offset_payload);
        xettp_saturation_on_tag saturate_tag;

#pragma unroll
        for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
            for (int j = 0; j < num_block_x; j++) {
                auto dst_reg
                        = matC.reg.xetla_select<block_elems, 1>(
                                          (i * num_block_x + j) * block_elems)
                                  .xetla_format<dtype_out, block_size_y,
                                          block_size_x>();
                auto src_reg
                        = matAcc.reg
                                  .xetla_select<block_elems, 1>(
                                          (i * num_block_x + j) * block_elems)
                                  .xetla_format<dtype_acc, block_size_y,
                                          block_size_x>();

#pragma unroll
                for (int row_i = 0; row_i < block_size_y; row_i++) {
                    auto offset_reg = offset.reg.xetla_select<block_size_x, 1>(
                            j * block_size_x);
                    auto scale_reg = scale.reg.xetla_select<block_size_x, 1>(
                            j * block_size_x);
                    dst_reg.row(row_i) = xetla_add<dtype_out,
                            dtype_offset_scale, block_size_x>(
                            (scale_reg
                                    * xetla_cvt<dtype_offset_scale, dtype_acc,
                                            block_size_x>(src_reg.row(row_i))),
                            offset_reg, saturate_tag);
                }
            }
        }
        // process the tail
        if constexpr ((tile_size_y % block_size_y) != 0) {
            constexpr uint32_t tail_start_y
                    = tile_size_y / block_size_y * block_size_y;
            constexpr int32_t tail_size_y = tile_size_y % block_size_y;
            constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
            for (int j = 0; j < num_block_x; j++) {
                auto dst_reg = matC.reg.xetla_select<tail_block_elems, 1>(
                                               tail_start_y * tile_size_x
                                               + j * tail_block_elems)
                                       .xetla_format<dtype_out, tail_size_y,
                                               block_size_x>();
                auto src_reg = matAcc.reg
                                       .xetla_select<tail_block_elems, 1>(
                                               tail_start_y * tile_size_x
                                               + j * tail_block_elems)
                                       .xetla_format<dtype_acc, tail_size_y,
                                               block_size_x>();
#pragma unroll
                for (int row_i = 0; row_i < tail_size_y; row_i++) {
                    auto offset_reg = offset.reg.xetla_select<block_size_x, 1>(
                            j * block_size_x);
                    auto scale_reg = scale.reg.xetla_select<block_size_x, 1>(
                            j * block_size_x);
                    dst_reg.row(row_i) = xetla_add<dtype_out,
                            dtype_offset_scale, block_size_x>(
                            (scale_reg
                                    * xetla_cvt<dtype_offset_scale, dtype_acc,
                                            block_size_x>(src_reg)),
                            offset_reg, saturate_tag);
                }
            }
        }
    }
};

/// @brief Is the scalar_multiply op functor.
/// Get the input from matAcc, multiply with a scalar, update the output in place.
/// Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_in Is the data type of multiplier buffer.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename dtype_in, gpu_arch arch_tag = gpu_arch::Xe>
struct scalar_mul_op_t {};
/// @brief Is the scalar_multiply op functor, specialized for Xe architecture.
template <typename dtype_in_>
struct scalar_mul_op_t<dtype_in_, gpu_arch::Xe> {
    using dtype_in = dtype_in_;
    using mem_desc_in_t
            = mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
    using coord_t = typename mem_desc_in_t::coord_t;

    struct arguments_t {
        dtype_in multiplier;
        inline arguments_t() = default;
        inline arguments_t(dtype_in multiplier_) : multiplier(multiplier_) {}
    };

    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc, coord_t coord,
            arguments_t args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        using dtype_acc = typename matAcc_t::dtype;
        static_assert(std::is_same<dtype_in, dtype_acc>::value,
                "Given multiplier must have same type as matAcc!");
        matAcc.reg *= args.multiplier;
    }
};

} // namespace gpu::xetla::subgroup