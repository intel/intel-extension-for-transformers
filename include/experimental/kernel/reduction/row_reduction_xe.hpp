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

#include "experimental/group/fused_op/row_reduction_fused_op_xe.hpp"
#include "experimental/group/reduction/row_reduce_store_xe.hpp"
#include "experimental/kernel/reduction/api.hpp"
#include "experimental/kernel/reduction/common.hpp"
#include "experimental/kernel/reduction/config.hpp"

namespace gpu::xetla::kernel {

/// @brief Is the row_reduction functor for Xe
/// The idea is threads in group will cooperatively process matrix_m x wg_tile_n.
/// Each time, each thread will load sg_tile_m x sg_tile_n data into register and do the reduction.
/// A dynamic counter will be set in SLM to solve load imbalance problem.
///
/// @tparam dtype_in_ Is the data type of input.
/// @tparam dtype_out_ Is the data type of output.
/// @tparam dtype_acc_ Is the accumulation data type.
/// @tparam reduction_attr_ Is the tile size for each group to do the reduction.
/// @tparam fused_op_t_ Is the fused op functor.
template <typename dtype_in_, typename dtype_out_, typename dtype_acc_,
        typename reduction_attr_, typename fused_op_t_>
struct xetla_row_reduction_t<dtype_in_, dtype_out_, dtype_acc_, reduction_attr_,
        gpu_arch::Xe, fused_op_t_> {
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_acc = dtype_acc_;
    using reduction_attr = reduction_attr_;
    using fused_op_t = fused_op_t_;
    using fused_op_arguments_t = typename fused_op_t::arguments_t;

    static constexpr uint32_t wg_tile_m = reduction_attr::wg_tile_m;
    static constexpr uint32_t wg_tile_n = reduction_attr::wg_tile_n;
    static constexpr uint32_t sg_tile_m = reduction_attr::sg_tile_m;
    static constexpr uint32_t sg_tile_n = reduction_attr::sg_tile_n;
    static constexpr bool is_dynamic_job = reduction_attr::is_dynamic_job;
    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
    using work_group_t = work_group_t<wg_size_x * wg_size_y>;
    static constexpr bool use_dynamic_job = is_dynamic_job && (wg_size_y > 1);
    using load_store_attr = typename arch_attr_t<
            gpu_arch::Xe>::template load_store_attr<msg_type::block_2d>;
    static constexpr uint32_t max_load_height_in_elem
            = load_store_attr::max_load_height_in_elem;
    static constexpr uint32_t max_load_width_in_bytes
            = load_store_attr::max_load_width_in_bytes;
    static constexpr uint32_t max_store_width_in_bytes
            = load_store_attr::max_store_width_in_bytes;
    static constexpr uint32_t max_load_width_in_elem
            = max_load_width_in_bytes / sizeof(dtype_in);
    static constexpr uint32_t max_store_width_in_elem
            = max_store_width_in_bytes / sizeof(dtype_out);

    static constexpr uint32_t tile_size_x = sg_tile_n;
    static constexpr uint32_t tile_size_y = sg_tile_m;

    static constexpr uint32_t max_simd_len = max_store_width_in_elem;

    /// block_size_x should be power of 2 and tile_size_x should be divided by block_size_x
    static constexpr uint32_t block_size_x
            = max_load_width_in_elem > tile_size_x
            ? tile_size_x
            : gpu::xetla::subgroup::detail::gcd<tile_size_x,
                    max_load_width_in_elem>::value;
    static_assert(block_size_x >= 8,
            "if block_size_x less than 8, the efficiency will be low. Please "
            "choose another tile_size_x");
    static constexpr uint32_t block_size_y
            = max_load_height_in_elem > tile_size_y ? tile_size_y
                                                    : max_load_height_in_elem;

    static constexpr uint32_t SIMD = 16;

    using global_ld_tile_desc_t = subgroup::tile_desc_t<tile_size_x,
            tile_size_y, block_size_x, block_size_y, reg_layout::tiled>;
    using global_ld_t = subgroup::tile_t<dtype_in, global_ld_tile_desc_t>;
    using global_ld_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>,
            global_ld_tile_desc_t,
            subgroup::msg_type_v<global_ld_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;
    using mat_buffer_t = subgroup::tile_t<dtype_acc,
            subgroup::tile_desc_t<tile_size_x, 1, block_size_x, 1,
                    reg_layout::tiled>>;
    using matAcc_t = subgroup::tile_t<dtype_acc, global_ld_tile_desc_t>;
    using row_reduce_store_t = group::group_row_reduce_store_t<dtype_acc,
            dtype_out, sg_tile_n, wg_size_x, wg_size_y, max_simd_len>;

    /// @brief
    ///
    struct arguments_t {
        dtype_in *mat_in_ptr;
        dtype_out *mat_out_ptr;
        uint32_t matrix_m;
        uint32_t matrix_n;
        uint32_t mat_in_ld;
    };

    /// @brief
    ///
    struct get_barrier_count {
        static constexpr uint32_t count = (wg_size_y > 1) ? wg_size_x : 0;
    };

    static constexpr uint32_t counter_size
            = use_dynamic_job ? SIMD * sizeof(int) * wg_size_x : 0;
    static constexpr uint32_t row_buffer_size = (wg_size_y > 1)
            ? tile_size_x * wg_size_x * wg_size_y * sizeof(dtype_acc)
            : 0;

    /// @brief
    ///
    struct get_slm_size {
        static constexpr uint32_t size = row_buffer_size + counter_size;
    };

    /// @brief Main execution function for row reduction.
    /// The basic process is 1) load data -> reduction 2) data sharing via SLM -> reduction 3) write out.
    ///
    /// @param ei
    /// @param args Includes base pointer and matrix size.
    /// @param fused_op_args
    /// @param slm_base
    /// @param nbarrier_base
    /// @return
    __XETLA_API static void call(sycl::nd_item<3> &item, arguments_t *args,
            fused_op_arguments_t *fused_op_args = nullptr,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        work_group_t g;
        g.init(item.get_local_linear_id());
        int sg_idx = g.get_id() % wg_size_x;
        int sg_idy = g.get_id() / wg_size_x;

        int global_start_x_in
                = item.get_group(2) * wg_tile_n + sg_idx * sg_tile_n;
        int global_start_y_in = sg_idy * sg_tile_m;
        xetla_nbarrier_t<wg_size_y, wg_size_y, gpu_arch::Xe> nbarrier;
        nbarrier.init_nbarrier(
                nbarrier_base + sg_idx, nbarrier_role::producer_consumer);
        if constexpr (use_dynamic_job) {
            /////for slm dynamic job queue
            xetla_vector<uint32_t, SIMD> offsets(
                    slm_base + row_buffer_size + sg_idx * SIMD * sizeof(int));
            xetla_mask<SIMD> pred(0);
            pred[0] = 1;
            if (sg_idy == 0) {
                xetla_vector<int, SIMD> init(wg_size_y);
                xetla_store_local<int, 1, data_size::default_size, SIMD>(
                        offsets, init, pred);
                xetla_fence<memory_kind::shared_local>();
            }
            nbarrier.arrive();
        }

        global_ld_t mat_global_ld;
        fused_op_t fused_op(
                fused_op_args, global_start_x_in, global_start_y_in);
        global_ld_payload_t mat_global_ld_payload(args->mat_in_ptr,
                args->matrix_n, args->matrix_m, args->mat_in_ld,
                global_start_x_in, global_start_y_in);
        mat_buffer_t mat_buffer(0);
        if constexpr (use_dynamic_job) {
            nbarrier.wait();
            int job_id = sg_idy;
            xetla_vector<uint32_t, SIMD> offsets(
                    slm_base + row_buffer_size + sg_idx * SIMD * sizeof(int));
            xetla_mask<SIMD> pred(0);
            pred[0] = 1;
            while (job_id * tile_size_y < args->matrix_m) {
                xetla_vector<int, SIMD> next_job
                        = xetla_atomic_local<atomic_op::iinc, int, SIMD>(
                                offsets, pred);
                subgroup::tile_load(mat_global_ld, mat_global_ld_payload);
                matAcc_t matAcc;
                subgroup::elemwise_cvt<matAcc_t, global_ld_t>(
                        matAcc, mat_global_ld);
                fused_op(matAcc);
                mat_buffer.reg += subgroup::tile_reduce<reduce_op::sum,
                        dtype_acc, dtype_acc, 0>(matAcc);
                mat_global_ld_payload
                        .template update_tdesc<tdesc_update_dir::y_dir>(
                                (next_job[0] - job_id) * tile_size_y);
                fused_op.update_tdesc(0, (next_job[0] - job_id) * tile_size_y);
                job_id = next_job[0];
            }
        } else {
            for (int job_id = sg_idy; job_id * tile_size_y < args->matrix_m;
                    job_id += wg_size_y) {
                subgroup::tile_load(mat_global_ld, mat_global_ld_payload);
                matAcc_t matAcc;
                subgroup::elemwise_cvt<matAcc_t, global_ld_t>(
                        matAcc, mat_global_ld);
                fused_op(matAcc);
                mat_buffer.reg += subgroup::tile_reduce<reduce_op::sum,
                        dtype_acc, dtype_acc, 0>(matAcc);
                fused_op.update_tdesc(0, wg_size_y * tile_size_y);
                mat_global_ld_payload
                        .template update_tdesc<tdesc_update_dir::y_dir>(
                                wg_size_y * tile_size_y);
            }
        }

        row_reduce_store_t row_reduce_store;
        uint32_t slm_row_reduce_base = slm_base;
        uint32_t nbarrier_row_reduce_base = nbarrier_base;
        row_reduce_store.init(
                sg_idx, sg_idy, slm_row_reduce_base, nbarrier_row_reduce_base);
        row_reduce_store(args->mat_out_ptr, args->matrix_n, 1, args->matrix_n,
                global_start_x_in, 0, mat_buffer.reg);
    }
};

} // namespace gpu::xetla::kernel
