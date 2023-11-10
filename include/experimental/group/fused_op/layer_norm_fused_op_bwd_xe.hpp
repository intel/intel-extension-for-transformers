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
struct ln_bwd_fused_op_arguments_t {
    dtype_acc *dbias_acc_ptr;
    dtype_out *dx_resAdd_ptr;
    dtype_in *gradAdd_ptr;
    uint8_t *mask_ptr;
    uint32_t matrix_m;
    uint32_t matrix_n;
    uint32_t mat_ld;
    uint32_t mask_ld;
    // dropout_scale_inv =  (1-dropout_prob)
    float dropout_prob;
    float dropout_scale_inv;
};

/// @brief
///
/// @tparam ln_fused_op_kind_
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
template <ln_bwd_fused_kind ln_fused_op_kind_, typename dtype_in_,
        typename dtype_out_, typename dtype_acc_, typename layer_norm_attr_>
struct ln_bwd_fused_op_t<ln_fused_op_kind_, dtype_in_, dtype_out_, dtype_acc_,
        layer_norm_attr_, gpu_arch::Xe> {
    static constexpr ln_bwd_fused_kind fused_op_kind = ln_fused_op_kind_;
    using dtype_acc = dtype_acc_;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using arguments_t
            = ln_bwd_fused_op_arguments_t<dtype_in, dtype_out, dtype_acc>;
    static constexpr uint32_t wg_tile_m = layer_norm_attr_::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr_::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr_::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr_::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr_::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr_::wg_num_n;

    /// @brief
    ///
    /// @param args
    /// @param wg_idx
    /// @param wg_idy
    /// @param sg_idx
    /// @param sg_idy
    /// @return
    __XETLA_API void init(arguments_t *args, uint32_t wg_idx, uint32_t wg_idy,
            uint32_t sg_idx, uint32_t sg_idy) {}

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, sg_tile_n> pre_op(
            xetla_vector<dtype_acc, sg_tile_n> input) {
        return input;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, sg_tile_n> post_op(
            xetla_vector<dtype_acc, sg_tile_n> input) {
        return input;
    }

    /// @brief
    ///
    /// @tparam reduce_t
    /// @param ln_group_row_reduce
    /// @return
    template <typename reduce_t>
    __XETLA_API void final_op(reduce_t &ln_group_row_reduce) {}
};

/// @brief
///
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
template <typename dtype_in_, typename dtype_out_, typename dtype_acc_,
        typename layer_norm_attr_>
struct ln_bwd_fused_op_t<ln_bwd_fused_kind::bias_dropout_resAdd_ln, dtype_in_,
        dtype_out_, dtype_acc_, layer_norm_attr_, gpu_arch::Xe> {
    static constexpr ln_bwd_fused_kind fused_op_kind
            = ln_bwd_fused_kind::bias_dropout_resAdd_ln;
    using dtype_acc = dtype_acc_;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_mask = uint8_t;
    using arguments_t
            = ln_bwd_fused_op_arguments_t<dtype_in, dtype_out, dtype_acc>;
    static constexpr uint32_t wg_tile_m = layer_norm_attr_::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr_::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr_::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr_::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr_::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr_::wg_num_n;

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;

    static_assert((sg_tile_n % (sizeof(uint32_t) / sizeof(dtype_mask)) == 0),
            "sg_tile_n need to be DW aligned");
    using ln_bwd_tile_desc_t = subgroup::tile_desc_t<sg_tile_n, 1, sg_tile_n, 1,
            reg_layout::tiled>;
    using dx_resAdd_out_t = subgroup::tile_t<dtype_out, ln_bwd_tile_desc_t>;
    using dx_resAdd_out_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>,
            ln_bwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    using mask_in_t = subgroup::tile_t<dtype_mask, ln_bwd_tile_desc_t>;
    using mask_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>,
            ln_bwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    dx_resAdd_out_t dx_resAdd_out;
    dx_resAdd_out_payload_t dx_resAdd_out_payload;
    mask_in_t mask_in;
    mask_in_payload_t mask_in_payload;
    uint32_t mat_ld;
    uint32_t mask_ld;
    uint32_t matrix_n;
    uint32_t matrix_m;
    int32_t dbias_n;
    int32_t dbias_m;
    dtype_acc *dbias_acc_ptr;
    xetla_vector<dtype_acc, sg_tile_n> dbias;
    float dropout_prob;
    float dropout_scale_inv;

    /// @brief
    ///
    /// @param args
    /// @param wg_idx
    /// @param wg_idy
    /// @param sg_idx
    /// @param sg_idy
    /// @return
    __XETLA_API void init(arguments_t *args, uint32_t wg_idx, uint32_t wg_idy,
            uint32_t sg_idx, uint32_t sg_idy) {
        int start_n = wg_idx * wg_tile_n + sg_idx * sg_tile_n;
        int start_m = wg_idy * wg_tile_m + sg_idy * sg_tile_m;
        dbias = 0;
        mat_ld = args->mat_ld;
        mask_ld = args->mask_ld;
        matrix_n = args->matrix_n;
        matrix_m = args->matrix_m;
        dx_resAdd_out_payload.init(args->dx_resAdd_ptr, matrix_n, matrix_m,
                mat_ld, start_n, start_m);
        mask_in_payload.init(
                args->mask_ptr, matrix_n, matrix_m, mask_ld, start_n, start_m);
        dbias_acc_ptr = args->dbias_acc_ptr;
        dropout_scale_inv = args->dropout_scale_inv;
        dropout_prob = args->dropout_prob;
        dbias_n = start_n;
        dbias_m = wg_idy;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, sg_tile_n> pre_op(
            xetla_vector<dtype_acc, sg_tile_n> input) {
        return input;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, sg_tile_n> post_op(
            xetla_vector<dtype_acc, sg_tile_n> input) {
        xetla_vector<dtype_acc, sg_tile_n> output = input;
        dx_resAdd_out.reg = xetla_cvt<dtype_out, dtype_acc>(input);
        subgroup::tile_store<cache_hint::uncached>(
                dx_resAdd_out, dx_resAdd_out_payload);
        dx_resAdd_out_payload.update_tdesc(wg_num_m * wg_tile_m * mat_ld);
        if (dropout_prob != 0) {
            subgroup::tile_load(mask_in, mask_in_payload);
            SW_BARRIER();
            mask_in_payload.update_tdesc(wg_num_m * wg_tile_m * mask_ld);
            output = drop_out<dtype_acc, sg_tile_n>(
                    output, mask_in.reg, dropout_scale_inv);
        }
        dbias += output;
        return output;
    }

    /// @brief
    ///
    /// @tparam reduce_t
    /// @param ln_group_row_reduce
    /// @return
    template <typename reduce_t>
    __XETLA_API void final_op(reduce_t &ln_group_row_reduce) {
        ln_group_row_reduce(dbias_acc_ptr, matrix_n, wg_num_m, matrix_n,
                dbias_n, dbias_m, dbias);
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
struct ln_bwd_fused_op_t<ln_bwd_fused_kind::ln_dropout_gradAdd, dtype_in_,
        dtype_out_, dtype_acc_, layer_norm_attr_, gpu_arch::Xe> {
    static constexpr ln_bwd_fused_kind fused_op_kind
            = ln_bwd_fused_kind::ln_dropout_gradAdd;
    using dtype_acc = dtype_acc_;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_mask = uint8_t;
    using arguments_t
            = ln_bwd_fused_op_arguments_t<dtype_in, dtype_out, dtype_acc>;
    static constexpr uint32_t wg_tile_m = layer_norm_attr_::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr_::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr_::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr_::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr_::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr_::wg_num_n;

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;

    static_assert((sg_tile_n % (sizeof(uint32_t) / sizeof(dtype_mask)) == 0),
            "sg_tile_n need to be DW aligned");
    using ln_bwd_tile_desc_t = subgroup::tile_desc_t<sg_tile_n, 1, sg_tile_n, 1,
            reg_layout::tiled>;
    using grad_in_t = subgroup::tile_t<dtype_out, ln_bwd_tile_desc_t>;
    using grad_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>,
            ln_bwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;
    using mask_in_t = subgroup::tile_t<dtype_mask, ln_bwd_tile_desc_t>;
    using mask_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>,
            ln_bwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;

    grad_in_t grad_in;
    grad_in_payload_t grad_in_payload;
    mask_in_t mask_in;
    mask_in_payload_t mask_in_payload;

    uint32_t mat_ld;
    uint32_t mask_ld;
    uint32_t matrix_n;
    uint32_t matrix_m;
    float dropout_prob;
    float dropout_scale_inv;

    /// @brief
    ///
    /// @param args
    /// @param wg_idx
    /// @param wg_idy
    /// @param sg_idx
    /// @param sg_idy
    /// @return
    __XETLA_API void init(arguments_t *args, uint32_t wg_idx, uint32_t wg_idy,
            uint32_t sg_idx, uint32_t sg_idy) {
        int start_n = wg_idx * wg_tile_n + sg_idx * sg_tile_n;
        int start_m = wg_idy * wg_tile_m + sg_idy * sg_tile_m;
        mat_ld = args->mat_ld;
        mask_ld = args->mask_ld;
        matrix_n = args->matrix_n;
        matrix_m = args->matrix_m;
        grad_in_payload.init(args->gradAdd_ptr, matrix_n, matrix_m, mat_ld,
                start_n, start_m);
        mask_in_payload.init(
                args->mask_ptr, matrix_n, matrix_m, mask_ld, start_n, start_m);
        dropout_scale_inv = args->dropout_scale_inv;
        dropout_prob = args->dropout_prob;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, sg_tile_n> pre_op(
            xetla_vector<dtype_acc, sg_tile_n> input) {
        subgroup::tile_load(grad_in, grad_in_payload);
        grad_in_payload.update_tdesc(wg_num_m * wg_tile_m * mat_ld);
        xetla_vector<dtype_acc, sg_tile_n> grad_input
                = xetla_cvt<dtype_acc, dtype_in>(grad_in.reg);
        // grad_add
        xetla_vector<dtype_acc, sg_tile_n> output
                = reduce_helper<reduce_op::sum, dtype_acc, sg_tile_n>(
                        input, grad_input);
        if (dropout_prob != 0) {
            // dropout
            subgroup::tile_load(mask_in, mask_in_payload);
            SW_BARRIER();
            mask_in_payload.update_tdesc(wg_num_m * wg_tile_m * mask_ld);
            output = drop_out<dtype_acc, sg_tile_n>(
                    output, mask_in.reg, dropout_scale_inv);
        }
        return output;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, sg_tile_n> post_op(
            xetla_vector<dtype_acc, sg_tile_n> input) {
        return input;
    }

    /// @brief
    ///
    /// @tparam reduce_t
    /// @param ln_group_row_reduce
    /// @return
    template <typename reduce_t>
    __XETLA_API void final_op(reduce_t &ln_group_row_reduce) {}
};

/// @brief
///
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
template <typename dtype_in_, typename dtype_out_, typename dtype_acc_,
        typename layer_norm_attr_>
struct ln_bwd_fused_op_t<ln_bwd_fused_kind::ln_dropout, dtype_in_, dtype_out_,
        dtype_acc_, layer_norm_attr_, gpu_arch::Xe> {
    static constexpr ln_bwd_fused_kind fused_op_kind
            = ln_bwd_fused_kind::ln_dropout;
    using dtype_acc = dtype_acc_;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_mask = uint8_t;
    using arguments_t
            = ln_bwd_fused_op_arguments_t<dtype_in, dtype_out, dtype_acc>;
    static constexpr uint32_t wg_tile_m = layer_norm_attr_::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr_::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr_::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr_::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr_::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr_::wg_num_n;

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
    using ln_bwd_tile_desc_t = subgroup::tile_desc_t<sg_tile_n, 1, sg_tile_n, 1,
            reg_layout::tiled>;
    using mask_in_t = subgroup::tile_t<dtype_mask, ln_bwd_tile_desc_t>;
    using mask_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>,
            ln_bwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;

    mask_in_t mask_in;
    mask_in_payload_t mask_in_payload;
    uint32_t matrix_n;
    uint32_t matrix_m;
    uint32_t mask_ld;
    float dropout_scale_inv;

    /// @brief
    ///
    /// @param args
    /// @param wg_idx
    /// @param wg_idy
    /// @param sg_idx
    /// @param sg_idy
    /// @return
    __XETLA_API void init(arguments_t *args, uint32_t wg_idx, uint32_t wg_idy,
            uint32_t sg_idx, uint32_t sg_idy) {
        int start_m = wg_idy * wg_tile_m + sg_idy * sg_tile_m;
        int start_n = wg_idx * wg_tile_n + sg_idx * sg_tile_n;
        mask_ld = args->mask_ld;
        matrix_m = args->matrix_m;
        matrix_n = args->matrix_n;
        mask_in_payload.init(
                args->mask_ptr, matrix_n, matrix_m, mask_ld, start_n, start_m);
        dropout_scale_inv = args->dropout_scale_inv;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, sg_tile_n> pre_op(
            xetla_vector<dtype_acc, sg_tile_n> input) {
        xetla_vector<dtype_acc, sg_tile_n> output;
        // dropout
        subgroup::tile_load(mask_in, mask_in_payload);
        SW_BARRIER();
        mask_in_payload.update_tdesc(wg_num_m * wg_tile_m * mask_ld);
        output = drop_out<dtype_acc, sg_tile_n>(
                input, mask_in.reg, dropout_scale_inv);
        return output;
    }

    /// @brief
    ///
    /// @param input
    /// @return
    __XETLA_API xetla_vector<dtype_acc, sg_tile_n> post_op(
            xetla_vector<dtype_acc, sg_tile_n> input) {
        return input;
    }

    /// @brief
    ///
    /// @tparam reduce_t
    /// @param ln_group_row_reduce
    /// @return
    template <typename reduce_t>
    __XETLA_API void final_op(reduce_t &ln_group_row_reduce) {}
};

} // namespace gpu::xetla::group
