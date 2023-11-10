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

#include "group/reduction/reduction.hpp"
#include "group/softmax/api.hpp"
#include "group/softmax/common.hpp"
#include "group/softmax/softmax_policy.hpp"

namespace gpu::xetla::group {

template <typename dtype_in_, typename dtype_acc_, typename tile_shape_>
class softmax_t<softmax_policy_bwd<dtype_in_, dtype_acc_, gpu_arch::Xe>,
        tile_shape_> {

public:
    using tile_shape = tile_shape_;
    using dtype_in = dtype_in_;
    using dtype_acc = dtype_acc_;
    static constexpr gpu_arch arch_tag = gpu_arch::Xe;

private:
    using mem_desc_in_t
            = mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
    using shape_t = typename mem_desc_in_t::shape_t;
    using coord_t = typename mem_desc_in_t::coord_t;
    using base_t = typename mem_desc_in_t::base_t;
    using work_group_t = typename tile_shape::work_group_t;
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;

    using wg_reduce_sum_t = group_reduce_t<dtype_acc, 1, sg_tile_m,
            reduce_op::sum, wg_size_x, true, gpu_arch::Xe>;

public:
    struct arguments_t {
        shape_t shape;
        base_t base;
        dtype_acc sqrt_dk_inv;
        inline arguments_t() = default;
        inline arguments_t(base_t base_, shape_t shape_, dtype_acc sqrt_dk_inv_)
            : base(base_), shape(shape_), sqrt_dk_inv(sqrt_dk_inv_) {}
    };
    struct get_barrier_count {
        static constexpr uint32_t count = (wg_size_x > 1) ? wg_size_y : 0;
    };

    struct get_slm_size {
        static constexpr uint32_t size = (wg_size_x > 1)
                ? wg_size_y * wg_size_x * sg_tile_m * sizeof(dtype_acc)
                : 0;
    };

    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g, matAcc_t &matAcc,
            coord_t coord, const arguments_t &args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        static_assert(std::is_same<typename matAcc_t::dtype, dtype_acc>::value,
                "matAcc dtype should match with dtype_acc");

        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
        static constexpr int32_t num_block_x = matAcc_t::num_block_x;
        static constexpr int32_t num_block_y = matAcc_t::num_block_y;
        static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
        static constexpr uint32_t block_elems = matAcc_t::block_elems;
        static_assert((sg_tile_m == tile_size_y) && (sg_tile_n == tile_size_x),
                "tile size should match");
        using mat_in_tile_desc_t = subgroup::tile_desc_t<tile_size_x,
                tile_size_y, block_size_x, block_size_y, reg_layout::tiled>;
        using mat_in_t = subgroup::tile_t<dtype_in, mat_in_tile_desc_t>;
        using mat_in_payload_t = subgroup::mem_payload_t<mem_desc_in_t,
                mat_in_tile_desc_t,
                subgroup::msg_type_v<mat_in_tile_desc_t, mem_desc_in_t::space>,
                gpu_arch::Xe>;

        int32_t sg_idx = g.get_id() % wg_size_x;
        int32_t sg_idy = g.get_id() / wg_size_x;
        int32_t tile_offset_n = sg_idx * sg_tile_n;
        int32_t tile_offset_m = sg_idy * sg_tile_m;
        coord.x += tile_offset_n;
        coord.y += tile_offset_m;
        uint32_t nbarrier_id = nbarrier_base + sg_idy;
        uint32_t slm_base_addr
                = slm_base + sg_idy * wg_size_x * sg_tile_m * sizeof(dtype_acc);

        mem_desc_in_t mem_desc_in(args.base, args.shape, coord);
        mat_in_t mat_in;
        mat_in_payload_t mat_in_payload(mem_desc_in);
        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                mat_in, mat_in_payload);
        matAcc_t mat_in_acc;
        subgroup::elemwise_cvt(mat_in_acc, mat_in);
        matAcc.reg = matAcc.reg * mat_in_acc.reg;
        xetla_vector<dtype_acc, sg_tile_m> local_sum
                = subgroup::tile_reduce<reduce_op::sum, dtype_acc, dtype_acc,
                        1>(matAcc);
        wg_reduce_sum_t wg_reduce_sum(sg_idx, nbarrier_id, slm_base_addr);
        xetla_vector<dtype_acc, sg_tile_m> group_sum = wg_reduce_sum(local_sum);
        subgroup::tile_broadcast_op<subgroup::tile_minus, matAcc_t>(
                matAcc, group_sum);
        matAcc.reg = matAcc.reg * mat_in_acc.reg * args.sqrt_dk_inv;
    }
};

} // namespace gpu::xetla::group
