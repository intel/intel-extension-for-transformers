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

template <typename dtype_acc_, typename tile_shape_>
class softmax_t<softmax_policy_fwd<dtype_acc_, gpu_arch::Xe>, tile_shape_> {

public:
    using tile_shape = tile_shape_;
    using dtype_acc = dtype_acc_;
    static constexpr gpu_arch arch_tag = gpu_arch::Xe;
    struct arguments_t {
        dtype_acc sqrt_dk_inv;
        inline arguments_t() = default;
        inline arguments_t(dtype_acc sqrt_dk_inv_)
            : sqrt_dk_inv(sqrt_dk_inv_) {}
    };

private:
    using coord_t = mem_coord_t<2>;
    using work_group_t = typename tile_shape::work_group_t;
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;

    using wg_reduce_max_t = group_reduce_t<dtype_acc, 1, sg_tile_m,
            reduce_op::max, wg_size_x, true, gpu_arch::Xe>;
    using wg_reduce_sum_t = group_reduce_t<dtype_acc, 1, sg_tile_m,
            reduce_op::sum, wg_size_x, true, gpu_arch::Xe>;

public:
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
                "matAcc dtype_acc should match with dtype_acc");
        int32_t sg_idx = g.get_id() % wg_size_x;
        int32_t sg_idy = g.get_id() / wg_size_x;
        uint32_t nbarrier_id = nbarrier_base + sg_idy;
        uint32_t slm_base_addr
                = slm_base + sg_idy * wg_size_x * sg_tile_m * sizeof(dtype_acc);
        xetla_nbarrier_t<wg_size_x, wg_size_x, arch_tag> nbarrier;
        nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);
        xetla_vector<dtype_acc, sg_tile_m> local_max
                = subgroup::tile_reduce<reduce_op::max, dtype_acc, dtype_acc,
                        1>(matAcc);
        wg_reduce_max_t wg_reduce_max(sg_idx, nbarrier_id, slm_base_addr);
        xetla_vector<dtype_acc, sg_tile_m> group_max = wg_reduce_max(local_max);
        if constexpr (wg_size_x > 1) { nbarrier.arrive(); }
        subgroup::tile_broadcast_op<subgroup::tile_minus, matAcc_t>(
                matAcc, group_max);
        matAcc.reg = matAcc.reg * args.sqrt_dk_inv;
        matAcc.reg = xetla_exp<dtype_acc>(matAcc.reg);
        xetla_vector<dtype_acc, sg_tile_m> local_sum
                = subgroup::tile_reduce<reduce_op::sum, dtype_acc, dtype_acc,
                        1>(matAcc);
        wg_reduce_sum_t wg_reduce_sum(sg_idx, nbarrier_id, slm_base_addr);
        if constexpr (wg_size_x > 1) { nbarrier.wait(); }
        xetla_vector<dtype_acc, sg_tile_m> group_sum = wg_reduce_sum(local_sum);
        subgroup::tile_broadcast_op<subgroup::tile_div, matAcc_t>(
                matAcc, group_sum);
    }
};

} // namespace gpu::xetla::group
