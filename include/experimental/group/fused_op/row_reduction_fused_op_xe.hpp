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

#include "experimental/group/fused_op/row_reduction_fused_op_api.hpp"

namespace gpu::xetla::group {

/// @brief
///
/// @tparam dtype_in
/// @tparam dtype_out
/// @tparam dtype_acc
template <typename dtype_in, typename dtype_out, typename dtype_acc>
struct xetla_row_reduction_fused_op_arguments_t {
    dtype_in *gelu_bwd_w_ptr;
    dtype_out *gelu_bwd_x_ptr;
    dtype_out *dropout_bwd_ptr;
    uint8_t *mask_ptr;
    float dropout_prob;
    float dropout_scale_inv;
    uint32_t matrix_m;
    uint32_t matrix_n;
    uint32_t mat_in_ld;
    uint32_t mat_out_ld;
};

/// @brief
///
/// @tparam fused_op_kind_
/// @tparam dtype_in_
/// @tparam dtype_out_
/// @tparam dtype_acc_
/// @tparam reduction_attr_
template <reduction_fused_kind fused_op_kind_, typename dtype_in_,
        typename dtype_out_, typename dtype_acc_, typename reduction_attr_>
struct row_reduction_fused_op_t<fused_op_kind_, dtype_in_, dtype_out_,
        dtype_acc_, reduction_attr_, gpu_arch::Xe> {
    static constexpr reduction_fused_kind fused_op_kind = fused_op_kind_;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_acc = dtype_acc_;
    using arguments_t = xetla_row_reduction_fused_op_arguments_t<dtype_in,
            dtype_out, dtype_acc>;
    __XETLA_API row_reduction_fused_op_t(
            arguments_t *args, int start_n = 0, int start_m = 0) {}
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc) {}
    __XETLA_API void update_tdesc(int offset_n = 0, int offset_m = 0) {}
};

template <typename dtype_in_, typename dtype_out_, typename dtype_acc_,
        typename reduction_attr_>
struct row_reduction_fused_op_t<reduction_fused_kind::bias_gelu_w_bwd,
        dtype_in_, dtype_out_, dtype_acc_, reduction_attr_, gpu_arch::Xe> {
    static constexpr reduction_fused_kind fused_op_kind
            = reduction_fused_kind::bias_gelu_w_bwd;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_acc = dtype_acc_;
    using arguments_t = xetla_row_reduction_fused_op_arguments_t<dtype_in,
            dtype_out, dtype_acc>;
    mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>
            w_load_base_desc;
    mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>
            x_store_base_desc;
    __XETLA_API row_reduction_fused_op_t(
            arguments_t *args, int start_n = 0, int start_m = 0) {
        w_load_base_desc.init({args->gelu_bwd_w_ptr},
                {args->matrix_n, args->matrix_m, args->mat_in_ld},
                {start_n, start_m});
        x_store_base_desc.init({args->gelu_bwd_x_ptr},
                {args->matrix_n, args->matrix_m, args->mat_out_ld},
                {start_n, start_m});
    }

    /// @brief
    ///
    /// @tparam matAcc_t
    /// @param matAcc
    /// @return
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc) {
        static_assert(std::is_same<remove_const_t<dtype_acc>,
                              typename matAcc_t::dtype>::value,
                "dtype_acc should match with matAcc");
        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
        static constexpr uint32_t num_elems = matAcc_t::tile_elems;
        using dgelu_tile_desc_t = subgroup::tile_desc_t<tile_size_x,
                tile_size_y, block_size_x, block_size_y, reg_layout::tiled>;
        using dgelu_w_in_t = subgroup::tile_t<dtype_in, dgelu_tile_desc_t>;
        using dgelu_w_in_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>,
                dgelu_tile_desc_t,
                subgroup::msg_type_v<dgelu_tile_desc_t, mem_space::global>,
                gpu_arch::Xe>;
        using dgelu_x_out_t = subgroup::tile_t<dtype_out, dgelu_tile_desc_t>;
        using dgelu_x_out_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>,
                dgelu_tile_desc_t, msg_type::block_2d, gpu_arch::Xe>;
        dgelu_w_in_t dgelu_w_in;
        dgelu_w_in_payload_t dgelu_w_in_payload(w_load_base_desc);
        subgroup::tile_load(dgelu_w_in, dgelu_w_in_payload);
        xetla_vector<dtype_acc, num_elems> w
                = xetla_cvt<dtype_acc, dtype_in, num_elems>(dgelu_w_in.reg);
        matAcc.reg = matAcc.reg * w;
        dgelu_x_out_t dgelu_x_out;
        dgelu_x_out_payload_t dgelu_x_out_payload(x_store_base_desc);
        subgroup::elemwise_cvt(dgelu_x_out, matAcc);
        subgroup::tile_store<cache_hint::uncached>(
                dgelu_x_out, dgelu_x_out_payload);
    }

    /// @brief
    ///
    /// @param offset_n
    /// @param offset_m
    /// @return
    __XETLA_API void update_tdesc(int offset_n = 0, int offset_m = 0) {
        w_load_base_desc.update_coord(offset_n, offset_m);
        x_store_base_desc.update_coord(offset_n, offset_m);
    }
};

template <typename dtype_in_, typename dtype_out_, typename dtype_acc_,
        typename reduction_attr_>
struct row_reduction_fused_op_t<reduction_fused_kind::bias_dropout_bwd,
        dtype_in_, dtype_out_, dtype_acc_, reduction_attr_, gpu_arch::Xe> {
    static constexpr reduction_fused_kind fused_op_kind
            = reduction_fused_kind::bias_dropout_bwd;
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_acc = dtype_acc_;
    using dtype_mask = uint8_t;
    using arguments_t = xetla_row_reduction_fused_op_arguments_t<dtype_in,
            dtype_out, dtype_acc>;
    mem_desc_t<dtype_mask, mem_layout::row_major, mem_space::global>
            mask_load_base_desc;
    mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>
            dropout_bwd_store_base_desc;
    float dropout_prob;
    float dropout_scale_inv;

    __XETLA_API row_reduction_fused_op_t(
            arguments_t *args, int start_n = 0, int start_m = 0) {

        mask_load_base_desc.init({args->mask_ptr},
                {args->matrix_n, args->matrix_m, args->mat_in_ld},
                {start_n, start_m});
        dropout_bwd_store_base_desc.init({args->dropout_bwd_ptr},
                {args->matrix_n, args->matrix_m, args->mat_out_ld},
                {start_n, start_m});
        dropout_scale_inv = args->dropout_scale_inv;
        dropout_prob = args->dropout_prob;
    }

    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc) {
        static_assert(std::is_same<remove_const_t<dtype_acc>,
                              typename matAcc_t::dtype>::value,
                "dtype_acc should match with matAcc");
        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
        static constexpr uint32_t num_elems = matAcc_t::tile_elems;
        using reduction_tile_desc_t = subgroup::tile_desc_t<tile_size_x,
                tile_size_y, block_size_x, block_size_y, reg_layout::tiled>;
        using mask_in_t = subgroup::tile_t<dtype_mask, reduction_tile_desc_t>;
        using mask_in_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_mask, mem_layout::row_major,
                        mem_space::global>,
                reduction_tile_desc_t,
                subgroup::msg_type_v<reduction_tile_desc_t, mem_space::global>,
                gpu_arch::Xe>;
        using dropout_bwd_out_t
                = subgroup::tile_t<dtype_out, reduction_tile_desc_t>;
        using dropout_bwd_out_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>,
                reduction_tile_desc_t,
                subgroup::msg_type_v<reduction_tile_desc_t, mem_space::global>,
                gpu_arch::Xe>;
        if (dropout_prob != 0) {
            mask_in_t mask_in;
            mask_in_payload_t mask_in_payload(mask_load_base_desc);
            subgroup::tile_load(mask_in, mask_in_payload);
            SW_BARRIER();
            matAcc.reg = drop_out<dtype_acc, tile_size_x * tile_size_y>(
                    matAcc.reg, mask_in.reg, dropout_scale_inv);
        }
        dropout_bwd_out_t dropout_bwd_out;
        dropout_bwd_out_payload_t dropout_bwd_out_payload(
                dropout_bwd_store_base_desc);
        subgroup::elemwise_cvt(dropout_bwd_out, matAcc);
        subgroup::tile_store<cache_hint::uncached>(
                dropout_bwd_out, dropout_bwd_out_payload);
    }

    __XETLA_API void update_tdesc(int offset_n = 0, int offset_m = 0) {
        mask_load_base_desc.update_coord(offset_n, offset_m);
        dropout_bwd_store_base_desc.update_coord(offset_n, offset_m);
    }
};

} // namespace gpu::xetla::group
