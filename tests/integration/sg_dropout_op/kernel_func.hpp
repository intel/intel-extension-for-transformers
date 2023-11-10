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

#pragma once

#include "xetla.hpp"

using namespace gpu::xetla;

enum class dropout_op : uint8_t { normal = 0, mask_gen = 1 };

template <typename dtype_in, typename dtype_out, typename dtype_acc, int wg_n,
        int wg_m, int sg_n, int sg_m, dropout_op dropout_kind>
struct dropout_func_t {
    using mem_desc_in_t
            = mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
    using mem_desc_out_t
            = mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>;

    using tile_shape = group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
    using work_group_t = typename tile_shape::work_group_t;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;

    using in_block_size = subgroup::get_load_block_size_auto<dtype_in, sg_n,
            sg_m, gpu_arch::Xe, mem_layout::row_major, reg_layout::tiled>;
    static constexpr uint32_t tile_size_x = sg_n;
    static constexpr uint32_t tile_size_y = sg_m;
    static constexpr uint32_t block_size_x
            = subgroup::detail::gcd<in_block_size::block_size_x, 16>::value;
    static constexpr uint32_t block_size_y = in_block_size::block_size_y;

    using tile_desc_t = subgroup::tile_desc_t<tile_size_x, tile_size_y,
            block_size_x, block_size_y, reg_layout::tiled>;
    using matAcc_t = subgroup::tile_t<dtype_acc, tile_desc_t>;
    using mat_in_t = subgroup::tile_t<dtype_in, tile_desc_t>;
    using mat_in_payload_t = subgroup::mem_payload_t<mem_desc_in_t, tile_desc_t,
            subgroup::msg_type_v<tile_desc_t, mem_space::global>, gpu_arch::Xe>;

    using tile_op_t =
            typename std::conditional<dropout_kind == dropout_op::normal,
                    subgroup::dropout_op_t<uint8_t, gpu_arch::Xe>,
                    subgroup::rng_dropout_op_t<uint8_t, gpu_arch::Xe>>::type;

    using epilogue_t = group::epilogue_t<
            group::epilogue_policy_tile_op<tile_op_t, gpu_arch::Xe>, tile_shape,
            mem_desc_out_t>;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    static inline void run(sycl::nd_item<3> &item, dtype_in *mat_in_ptr,
            uint8_t *mask_ptr, dtype_out *mat_out_ptr,
            uint64_t *rand_offset_ptr, uint32_t mat_m, uint32_t mat_n,
            uint32_t mat_ld, float dropout_prob, float dropout_scale) {
        work_group_t g(item.get_local_linear_id());
        int start_n = item.get_group(2) * wg_n;
        int start_m = item.get_group(1) * wg_m;
        uint32_t boundary_n
                = (start_n + wg_n) > mat_n ? mat_n : (start_n + wg_n);
        uint32_t boundary_m
                = (start_m + wg_m) > mat_m ? mat_m : (start_m + wg_m);
        mem_desc_in_t mem_desc_in({mat_in_ptr},
                {boundary_n, boundary_m, mat_ld}, {start_n, start_m});
        int32_t sg_idx = g.get_id() % wg_size_x;
        int32_t sg_idy = g.get_id() / wg_size_x;
        int32_t tile_offset_n = sg_idx * sg_n;
        int32_t tile_offset_m = sg_idy * sg_m;
        mem_desc_in.update_coord(tile_offset_n, tile_offset_m);

        mem_desc_out_t mem_desc_out({mat_out_ptr},
                {boundary_n, boundary_m, mat_ld}, {start_n, start_m});
        mat_in_t mat_in;
        mat_in_payload_t mat_in_payload;
        mat_in_payload.init(mem_desc_in);
        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                mat_in, mat_in_payload);
        matAcc_t matAcc;
        subgroup::elemwise_cvt(matAcc, mat_in);
        epilogue_t epilogue;
        if constexpr (dropout_kind == dropout_op::normal) {
            epilogue_args_t epilogue_args {
                    {{mask_ptr}, {boundary_n, boundary_m, mat_ld}, dropout_prob,
                            dropout_scale}};
            epilogue(g, matAcc, mem_desc_out, epilogue_args);
        } else {
            epilogue_args_t epilogue_args {
                    {{mask_ptr}, {boundary_n, boundary_m, mat_ld}, dropout_prob,
                            rand_offset_ptr}};
            epilogue(g, matAcc, mem_desc_out, epilogue_args);
        }
    }
};
