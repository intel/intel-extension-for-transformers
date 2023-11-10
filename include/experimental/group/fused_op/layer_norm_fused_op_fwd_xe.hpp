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

#include "experimental/group/fused_op/layer_norm_fused_op_api.hpp"

namespace gpu::xetla::group {

/// @brief
///
/// @tparam dtype_in
/// @tparam dtype_out
/// @tparam dtype_acc
template <typename dtype_in, typename dtype_out, typename dtype_acc>
struct ln_fwd_fused_op_arguments_t {
    dtype_in *bias_ptr;
    dtype_in *res_add_ptr;
    dtype_out *bias_dropout_res_ptr;
    uint8_t *mask_ptr;
    uint32_t matrix_m;
    uint32_t matrix_n;
    uint32_t mat_ld;
    uint32_t mask_ld;
    uint64_t rand_seed = 67280421310721;
    uint64_t *rand_offset_ptr;
    float dropout_prob;
    // dropout_scale =  1 / (1-dropout_prob)
    float dropout_scale;
};

/// @brief
///
/// @tparam ln_fused_op_kind_
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
template <ln_fwd_fused_kind ln_fused_op_kind_, typename dtype_in_,
        typename dtype_out_, typename dtype_acc_, typename layer_norm_attr_>
struct ln_fwd_fused_op_t<ln_fused_op_kind_, dtype_in_, dtype_out_, dtype_acc_,
        layer_norm_attr_, gpu_arch::Xe> {
    static constexpr ln_fwd_fused_kind fused_op_kind = ln_fused_op_kind_;
    using dtype_acc = dtype_acc_;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using arguments_t
            = ln_fwd_fused_op_arguments_t<dtype_in, dtype_out, dtype_acc>;
    static constexpr uint32_t wg_tile_m = layer_norm_attr_::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr_::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr_::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr_::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr_::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr_::wg_num_n;
    static constexpr uint32_t chunk_size = layer_norm_attr_::chunk_size;
    static constexpr uint32_t n_chunks = sg_tile_n / chunk_size;

    /// @brief
    ///
    /// @param args
    /// @param wg_idx
    /// @param wg_idy
    /// @param sg_idx
    /// @param sg_idy
    /// @return
    __XETLA_API void init(arguments_t *args, uint32_t wg_idx, uint32_t wg_idy,
            uint32_t sg_idx, uint32_t sg_idy, uint32_t start_m) {}

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, chunk_size> pre_op(
            xetla_vector<dtype_acc, chunk_size> input) {
        return input;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, chunk_size> post_op(
            xetla_vector<dtype_acc, chunk_size> input) {
        return input;
    }
};
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
template <typename dtype_in_, typename dtype_out_, typename dtype_acc_,
        typename layer_norm_attr_>
struct ln_fwd_fused_op_t<ln_fwd_fused_kind::bias_dropout_resAdd_ln, dtype_in_,
        dtype_out_, dtype_acc_, layer_norm_attr_, gpu_arch::Xe> {
    static constexpr ln_fwd_fused_kind fused_op_kind
            = ln_fwd_fused_kind::bias_dropout_resAdd_ln;
    using dtype_acc = dtype_acc_;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_mask = uint8_t;
    using arguments_t
            = ln_fwd_fused_op_arguments_t<dtype_in, dtype_out, dtype_acc>;
    static constexpr uint32_t wg_tile_m = layer_norm_attr_::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr_::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr_::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr_::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr_::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr_::wg_num_n;
    static constexpr uint32_t chunk_size = layer_norm_attr_::chunk_size;
    static constexpr uint32_t n_chunks = sg_tile_n / chunk_size;

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;

    static_assert((sg_tile_n % (sizeof(uint32_t) / sizeof(dtype_mask)) == 0),
            "sg_tile_n need to be DW aligned");

    using ln_fwd_tile_desc_t = subgroup::tile_desc_t<chunk_size, 1, chunk_size,
            1, reg_layout::tiled>;
    using bias_in_t = subgroup::tile_t<dtype_in, ln_fwd_tile_desc_t>;
    using bias_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    using res_in_t = subgroup::tile_t<dtype_in, ln_fwd_tile_desc_t>;
    using res_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    using mask_in_t = subgroup::tile_t<dtype_mask, ln_fwd_tile_desc_t>;
    using mask_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    using bias_dropout_res_out_t
            = subgroup::tile_t<dtype_out, ln_fwd_tile_desc_t>;
    using bias_dropout_res_out_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    bias_in_t bias_in;
    bias_in_payload_t bias_in_payload;
    bias_dropout_res_out_t bias_dropout_res_out;
    bias_dropout_res_out_payload_t bias_dropout_res_out_payload;
    res_in_t res_in;
    res_in_payload_t res_in_payload;
    mask_in_t mask_in;
    mask_in_payload_t mask_in_payload;
    uint32_t mat_ld;
    uint32_t mask_ld;
    uint32_t matrix_n;
    uint32_t matrix_m;
    float dropout_scale;
    float dropout_prob;

    /// @brief
    ///
    /// @param args
    /// @param wg_idx
    /// @param wg_idy
    /// @param sg_idx
    /// @param sg_idy
    /// @return
    __XETLA_API void init(arguments_t *args, uint32_t wg_idx, uint32_t wg_idy,
            uint32_t sg_idx, uint32_t sg_idy, uint32_t start_m) {
        int start_n = wg_idx * wg_tile_n + sg_idx * sg_tile_n;
        mat_ld = args->mat_ld;
        mask_ld = args->mask_ld;
        matrix_n = args->matrix_n;
        matrix_m = args->matrix_m;
        dropout_scale = args->dropout_scale;
        dropout_prob = args->dropout_prob;
        bias_in_payload.init(args->bias_ptr, matrix_n, 1, mat_ld, start_n, 0);
        res_in_payload.init(args->res_add_ptr, matrix_n, matrix_m, mat_ld,
                start_n, start_m);
        bias_dropout_res_out_payload.init(args->bias_dropout_res_ptr, matrix_n,
                matrix_m, mat_ld, start_n, start_m);
        mask_in_payload.init(
                args->mask_ptr, matrix_n, matrix_m, mask_ld, start_n, start_m);
        if constexpr (n_chunks == 1) {
            subgroup::tile_load(bias_in, bias_in_payload);
        }
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, chunk_size> pre_op(
            xetla_vector<dtype_acc, chunk_size> input) {
        subgroup::tile_load(res_in, res_in_payload);
        if constexpr (n_chunks == 1) {
            res_in_payload.update_tdesc(wg_num_m * wg_tile_m * mat_ld);
        } else {
            res_in_payload.update_tdesc(chunk_size);
        }
        if constexpr (n_chunks != 1) {
            subgroup::tile_load(bias_in, bias_in_payload);
            bias_in_payload.update_tdesc(chunk_size);
        }
        xetla_vector<dtype_acc, chunk_size> bias_input
                = xetla_cvt<dtype_acc, dtype_in>(bias_in.reg);
        // bias_add
        xetla_vector<dtype_acc, chunk_size> output
                = reduce_helper<reduce_op::sum, dtype_acc, chunk_size>(
                        input, bias_input);
        if (dropout_prob != 0) {
            // dropout
            subgroup::tile_load(mask_in, mask_in_payload);
            SW_BARRIER();
            if constexpr (n_chunks == 1) {
                mask_in_payload.update_tdesc(wg_num_m * wg_tile_m * mask_ld);
            } else {
                mask_in_payload.update_tdesc(chunk_size);
            }
            output = drop_out<dtype_acc, chunk_size>(
                    output, mask_in.reg, dropout_scale);
        }
        // res_add, generate mixed mode
        xetla_vector<dtype_acc, chunk_size> res_input
                = xetla_cvt<dtype_acc, dtype_in>(res_in.reg);
        output = reduce_helper<reduce_op::sum, dtype_acc, chunk_size>(
                output, res_input);
        bias_dropout_res_out.reg = xetla_cvt<dtype_out, dtype_acc>(output);
        subgroup::tile_store<cache_hint::uncached>(
                bias_dropout_res_out, bias_dropout_res_out_payload);
        if constexpr (n_chunks == 1) {
            bias_dropout_res_out_payload.update_tdesc(
                    wg_num_m * wg_tile_m * mat_ld);
        } else {
            bias_dropout_res_out_payload.update_tdesc(chunk_size);
        }
        return output;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, chunk_size> post_op(
            xetla_vector<dtype_acc, chunk_size> input) {
        return input;
    }
};

/// @brief
///
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
template <typename dtype_in_, typename dtype_out_, typename dtype_acc_,
        typename layer_norm_attr_>
struct ln_fwd_fused_op_t<ln_fwd_fused_kind::ln_dropout, dtype_in_, dtype_out_,
        dtype_acc_, layer_norm_attr_, gpu_arch::Xe> {
    static constexpr ln_fwd_fused_kind fused_op_kind
            = ln_fwd_fused_kind::ln_dropout;
    using dtype_acc = dtype_acc_;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_mask = uint8_t;
    using arguments_t
            = ln_fwd_fused_op_arguments_t<dtype_in, dtype_out, dtype_acc>;
    static constexpr uint32_t wg_tile_m = layer_norm_attr_::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr_::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr_::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr_::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr_::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr_::wg_num_n;
    static constexpr uint32_t chunk_size = layer_norm_attr_::chunk_size;
    static constexpr uint32_t n_chunks = sg_tile_n / chunk_size;

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;

    static_assert((sg_tile_n % (sizeof(uint32_t) / sizeof(dtype_mask)) == 0),
            "sg_tile_n need to be DW aligned");

    using ln_fwd_tile_desc_t = subgroup::tile_desc_t<chunk_size, 1, chunk_size,
            1, reg_layout::tiled>;
    using mask_in_t = subgroup::tile_t<dtype_mask, ln_fwd_tile_desc_t>;
    using mask_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    mask_in_t mask_in;
    mask_in_payload_t mask_in_payload;
    uint32_t mask_ld;
    uint32_t matrix_m;
    uint32_t matrix_n;
    float dropout_scale;

    /// @brief
    ///
    /// @param args
    /// @param wg_idx
    /// @param wg_idy
    /// @param sg_idx
    /// @param sg_idy
    /// @return
    __XETLA_API void init(arguments_t *args, uint32_t wg_idx, uint32_t wg_idy,
            uint32_t sg_idx, uint32_t sg_idy, uint32_t start_m) {
        int start_n = wg_idx * wg_tile_n + sg_idx * sg_tile_n;
        dropout_scale = args->dropout_scale;
        mask_ld = args->mask_ld;
        matrix_m = args->matrix_m;
        matrix_n = args->matrix_n;
        mask_in_payload.init(
                args->mask_ptr, matrix_n, matrix_m, mask_ld, start_n, start_m);
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, chunk_size> pre_op(
            xetla_vector<dtype_acc, chunk_size> input) {
        return input;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, chunk_size> post_op(
            xetla_vector<dtype_acc, chunk_size> input) {
        // dropout
        subgroup::tile_load(mask_in, mask_in_payload);
        SW_BARRIER();
        if constexpr (n_chunks == 1) {
            mask_in_payload.update_tdesc(wg_num_m * wg_tile_m * mask_ld);
        } else {
            mask_in_payload.update_tdesc(chunk_size);
        }
        xetla_vector<dtype_acc, chunk_size> output
                = drop_out<dtype_acc, chunk_size>(
                        input, mask_in.reg, dropout_scale);
        return output;
    }
};

/// @brief
///
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
template <typename dtype_in_, typename dtype_out_, typename dtype_acc_,
        typename layer_norm_attr_>
struct ln_fwd_fused_op_t<ln_fwd_fused_kind::bias_rng_dropout_resAdd_ln,
        dtype_in_, dtype_out_, dtype_acc_, layer_norm_attr_, gpu_arch::Xe> {
    static constexpr ln_fwd_fused_kind fused_op_kind
            = ln_fwd_fused_kind::bias_rng_dropout_resAdd_ln;
    using dtype_acc = dtype_acc_;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_mask = uint8_t;
    using arguments_t
            = ln_fwd_fused_op_arguments_t<dtype_in, dtype_out, dtype_acc>;
    static constexpr uint32_t wg_tile_m = layer_norm_attr_::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr_::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr_::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr_::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr_::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr_::wg_num_n;
    static constexpr uint32_t chunk_size = layer_norm_attr_::chunk_size;
    static constexpr uint32_t n_chunks = sg_tile_n / chunk_size;

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;

    using ln_fwd_tile_desc_t = subgroup::tile_desc_t<chunk_size, 1, chunk_size,
            1, reg_layout::tiled>;
    using bias_in_t = subgroup::tile_t<dtype_in, ln_fwd_tile_desc_t>;
    using bias_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    using res_in_t = subgroup::tile_t<dtype_in, ln_fwd_tile_desc_t>;
    using res_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    using mask_out_t = subgroup::tile_t<dtype_mask, ln_fwd_tile_desc_t>;
    using mask_out_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    using bias_dropout_res_out_t
            = subgroup::tile_t<dtype_out, ln_fwd_tile_desc_t>;
    using bias_dropout_res_out_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;

    bias_in_t bias_in;
    bias_in_payload_t bias_in_payload;
    bias_dropout_res_out_t bias_dropout_res_out;
    bias_dropout_res_out_payload_t bias_dropout_res_out_payload;
    res_in_t res_in;
    res_in_payload_t res_in_payload;
    mask_out_t mask_out;
    mask_out_payload_t mask_out_payload;
    uint32_t mat_ld;
    uint32_t mask_ld;
    uint32_t matrix_n;
    uint32_t matrix_m;
    float dropout_prob;
    dropout_fwd_t<chunk_size> dropout_fwd;

    /// @brief
    ///
    /// @param args
    /// @param wg_idx
    /// @param wg_idy
    /// @param sg_idx
    /// @param sg_idy
    /// @return
    __XETLA_API void init(arguments_t *args, uint32_t wg_idx, uint32_t wg_idy,
            uint32_t sg_idx, uint32_t sg_idy, uint32_t start_m) {
        int start_n = wg_idx * wg_tile_n + sg_idx * sg_tile_n;
        xetla_vector<uint64_t, 1> rand_offset_ptr_v
                = xetla_load_global<uint64_t, 1, data_size::default_size,
                        cache_hint::cached, cache_hint::cached>(
                        args->rand_offset_ptr, 0);
        mat_ld = args->mat_ld;
        mask_ld = args->mask_ld;
        matrix_n = args->matrix_n;
        matrix_m = args->matrix_m;
        uint32_t threshold = uint32_t(args->dropout_prob * float(4294967296));
        dropout_prob = args->dropout_prob;
        bias_in_payload.init(args->bias_ptr, matrix_n, 1, mat_ld, start_n, 0);
        res_in_payload.init(args->res_add_ptr, matrix_n, matrix_m, mat_ld,
                start_n, start_m);
        bias_dropout_res_out_payload.init(args->bias_dropout_res_ptr, matrix_n,
                matrix_m, mat_ld, start_n, start_m);
        mask_out_payload.init(
                args->mask_ptr, matrix_n, matrix_m, mask_ld, start_n, start_m);
        int linear_idx = (wg_idy * wg_size_y + sg_idy) * (wg_size_x * wg_num_n)
                + wg_idx * wg_size_x + sg_idx;
        dropout_fwd.init(args->rand_seed, linear_idx, rand_offset_ptr_v[0],
                threshold, args->dropout_scale);
        if constexpr (n_chunks == 1) {
            subgroup::tile_load(bias_in, bias_in_payload);
        }
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, chunk_size> pre_op(
            xetla_vector<dtype_acc, chunk_size> input) {
        subgroup::tile_load(res_in, res_in_payload);
        if constexpr (n_chunks == 1) {
            res_in_payload.update_tdesc(wg_num_m * wg_tile_m * mat_ld);
        } else {
            res_in_payload.update_tdesc(chunk_size);
        }
        if constexpr (n_chunks != 1) {
            subgroup::tile_load(bias_in, bias_in_payload);
            bias_in_payload.update_tdesc(chunk_size);
        }
        xetla_vector<dtype_acc, chunk_size> bias_input
                = xetla_cvt<dtype_acc, dtype_in>(bias_in.reg);
        // bias_add
        xetla_vector<dtype_acc, chunk_size> output
                = reduce_helper<reduce_op::sum, dtype_acc, chunk_size>(
                        input, bias_input);
        if (dropout_prob != 0) {
            // dropout
            output = dropout_fwd.template process<dtype_acc>(output);
            mask_out.reg = dropout_fwd.get_mask();
            subgroup::tile_store<cache_hint::uncached>(
                    mask_out, mask_out_payload);
            if constexpr (n_chunks == 1) {
                mask_out_payload.update_tdesc(wg_num_m * wg_tile_m * mask_ld);
            } else {
                mask_out_payload.update_tdesc(chunk_size);
            }
        }
        // res_add, generate mixed mode
        xetla_vector<dtype_acc, chunk_size> res_input
                = xetla_cvt<dtype_acc, dtype_in>(res_in.reg);
        output = reduce_helper<reduce_op::sum, dtype_acc, chunk_size>(
                output, res_input);
        bias_dropout_res_out.reg = xetla_cvt<dtype_out, dtype_acc>(output);
        subgroup::tile_store<cache_hint::uncached>(
                bias_dropout_res_out, bias_dropout_res_out_payload);
        if constexpr (n_chunks == 1) {
            bias_dropout_res_out_payload.update_tdesc(
                    wg_num_m * wg_tile_m * mat_ld);
        } else {
            bias_dropout_res_out_payload.update_tdesc(chunk_size);
        }
        return output;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, chunk_size> post_op(
            xetla_vector<dtype_acc, chunk_size> input) {
        return input;
    }
};

/// @brief
///
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
template <typename dtype_in_, typename dtype_out_, typename dtype_acc_,
        typename layer_norm_attr_>
struct ln_fwd_fused_op_t<ln_fwd_fused_kind::ln_rng_dropout, dtype_in_,
        dtype_out_, dtype_acc_, layer_norm_attr_, gpu_arch::Xe> {
    static constexpr ln_fwd_fused_kind fused_op_kind
            = ln_fwd_fused_kind::ln_rng_dropout;
    using dtype_acc = dtype_acc_;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_mask = uint8_t;
    using arguments_t
            = ln_fwd_fused_op_arguments_t<dtype_in, dtype_out, dtype_acc>;
    static constexpr uint32_t wg_tile_m = layer_norm_attr_::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr_::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr_::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr_::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr_::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr_::wg_num_n;
    static constexpr uint32_t chunk_size = layer_norm_attr_::chunk_size;
    static constexpr uint32_t n_chunks = sg_tile_n / chunk_size;
    static_assert(sg_tile_n % chunk_size == 0,
            "Current impl does not support tailing mechanism");

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;

    using mask_out_tile_desc_t = subgroup::tile_desc_t<chunk_size, 1,
            chunk_size, 1, reg_layout::tiled>;
    using mask_out_t = subgroup::tile_t<dtype_mask, mask_out_tile_desc_t>;
    using mask_out_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>,
            mask_out_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    mask_out_t mask_out;
    mask_out_payload_t mask_out_payload;
    dropout_fwd_t<chunk_size> dropout_fwd;
    uint32_t mask_ld;
    uint32_t matrix_m;
    uint32_t matrix_n;

    /// @brief
    ///
    /// @param args
    /// @param wg_idx
    /// @param wg_idy
    /// @param sg_idx
    /// @param sg_idy
    /// @return
    __XETLA_API void init(arguments_t *args, uint32_t wg_idx, uint32_t wg_idy,
            uint32_t sg_idx, uint32_t sg_idy, uint32_t start_m) {
        int start_n = wg_idx * wg_tile_n + sg_idx * sg_tile_n;
        xetla_vector<uint64_t, 1> rand_offset_ptr_v
                = xetla_load_global<uint64_t, 1, data_size::default_size,
                        cache_hint::cached, cache_hint::cached>(
                        args->rand_offset_ptr, 0);
        mask_ld = args->mask_ld;
        matrix_m = args->matrix_m;
        matrix_n = args->matrix_n;
        uint32_t threshold = uint32_t(args->dropout_prob * float(4294967296));
        mask_out_payload.init(
                args->mask_ptr, matrix_n, matrix_m, mask_ld, start_n, start_m);
        int linear_idx = (wg_idy * wg_size_y + sg_idy) * (wg_size_x * wg_num_n)
                + wg_idx * wg_size_x + sg_idx;
        dropout_fwd.init(args->rand_seed, linear_idx, rand_offset_ptr_v[0],
                threshold, args->dropout_scale);
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, chunk_size> pre_op(
            xetla_vector<dtype_acc, chunk_size> input) {
        return input;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, chunk_size> post_op(
            xetla_vector<dtype_acc, chunk_size> input) {
        xetla_vector<dtype_acc, chunk_size> output
                = dropout_fwd.template process<dtype_acc>(input);
        mask_out.reg = dropout_fwd.get_mask();
        subgroup::tile_store<cache_hint::uncached>(mask_out, mask_out_payload);
        if constexpr (n_chunks == 1) {
            mask_out_payload.update_tdesc(wg_num_m * wg_tile_m * mask_ld);
        } else {
            mask_out_payload.update_tdesc(chunk_size);
        }
        return output;
    }
};

} // namespace gpu::xetla::group
