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

#include "experimental/kernel/data_transformer/api.hpp"
#include "experimental/kernel/data_transformer/common.hpp"
#include "experimental/kernel/data_transformer/config.hpp"
#include "group/reduction/reduction_xe.hpp"

namespace gpu::xetla::kernel {

/// @brief Is the data_transformer functor for Xe
/// Each time, each thread will load sg_tile_m x sg_tile_n data into register and do the data convert.
///
/// @tparam dtype_in_ Is the data type of input.
/// @tparam dtype_out_ Is the data type of output.
/// @tparam dtype_compute_ Is the data type of calculation.
/// @tparam data_transformer_attr_ Is the tile size for each group to do the data convert.
/// @tparam mem_layout_in_ Indicates the input data col major or row major.
/// @tparam need_fp8_op Indicates whether fp8-related operations are required.
template <typename dtype_in_, typename dtype_out_, typename dtype_compute_,
        typename data_transformer_attr_, mem_layout mem_layout_in_,
        int need_fp8_op>
struct xetla_data_transformer<dtype_in_, dtype_out_, dtype_compute_,
        data_transformer_attr_, mem_layout_in_, need_fp8_op, gpu_arch::Xe> {
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using dtype_compute = dtype_compute_;
    using data_transformer_attr = data_transformer_attr_;

    static constexpr mem_layout mem_layout_in = mem_layout_in_;

    static constexpr bool is_col_major_in
            = mem_layout_in == mem_layout::col_major;

    static constexpr uint32_t wg_tile_m = data_transformer_attr::wg_tile_m;
    static constexpr uint32_t wg_tile_n = data_transformer_attr::wg_tile_n;
    static constexpr uint32_t sg_tile_m = data_transformer_attr::sg_tile_m;
    static constexpr uint32_t sg_tile_n = data_transformer_attr::sg_tile_n;

    static constexpr uint32_t tile_size_x = sg_tile_n;
    static constexpr uint32_t tile_size_y = sg_tile_m;

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;

    using load_store_attr = typename arch_attr_t<
            gpu_arch::Xe>::template load_store_attr<msg_type::block_2d>;
    static constexpr uint32_t max_load_height_in_elem
            = load_store_attr::max_load_height_in_elem;
    static constexpr uint32_t max_load_width_in_bytes
            = load_store_attr::max_load_width_in_bytes;
    static constexpr uint32_t max_store_width_in_bytes
            = load_store_attr::max_store_width_in_bytes;
    static constexpr uint32_t max_trans_block_width
            = load_store_attr::max_trans_load_width_in_bytes / sizeof(dtype_in);
    static constexpr uint32_t max_load_width_in_elem
            = max_load_width_in_bytes / sizeof(dtype_in);
    static constexpr uint32_t max_store_width_in_elem
            = max_store_width_in_bytes / sizeof(dtype_out);

    /// block_size_x should be power of 2 and tile_size_x should be divided by block_size_x
    static constexpr uint32_t load_size_x
            = gpu::xetla::subgroup::detail::gcd<tile_size_x,
                    max_load_width_in_elem>::value;
    static_assert(load_size_x >= 8,
            "if block_size_x less than 8, the efficiency will be low. Please "
            "choose another tile_size_x");
    static constexpr uint32_t st_size_x = max_store_width_in_elem > tile_size_x
            ? tile_size_x
            : gpu::xetla::subgroup::detail::gcd<tile_size_x,
                    max_store_width_in_elem>::value;
    static_assert(st_size_x >= 8,
            "if st_block_size_x less than 8, the efficiency will be "
            "low. ");
    static constexpr uint32_t block_size_x
            = gpu::xetla::subgroup::detail::gcd<load_size_x, st_size_x>::value;

    static constexpr uint32_t block_size_y_limit
            = is_col_major_in ? max_trans_block_width : max_load_height_in_elem;

    static constexpr uint32_t block_size_y = block_size_y_limit > tile_size_y
            ? tile_size_y
            : block_size_y_limit;

    static constexpr reg_layout in_reg_layout = reg_layout::tiled;

    using global_ld_tile_desc_t = subgroup::tile_desc_t<tile_size_x,
            tile_size_y, block_size_x, block_size_y, in_reg_layout>;
    using global_ld_t = subgroup::tile_t<dtype_in, global_ld_tile_desc_t>;
    using global_ld_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_in, mem_layout_in, mem_space::global>,
            global_ld_tile_desc_t,
            subgroup::msg_type_v<global_ld_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;

    using global_st_tile_desc_t = subgroup::tile_desc_t<tile_size_x,
            tile_size_y, block_size_x, block_size_y, reg_layout::tiled>;
    using global_st_t = subgroup::tile_t<dtype_out, global_st_tile_desc_t>;
    using global_st_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_out, mem_layout::row_major, mem_space::global>,
            global_st_tile_desc_t, msg_type::block_2d, gpu_arch::Xe>;
    using global_compute_tile_desc = subgroup::tile_desc_t<tile_size_x,
            tile_size_y, block_size_x, block_size_y, reg_layout::tiled>;
    using global_compute_t
            = subgroup::tile_t<dtype_compute, global_compute_tile_desc>;

    using wg_reduce_t
            = group::group_reduce_t<dtype_compute, tile_size_x * tile_size_y, 1,
                    reduce_op::max, wg_size_x * wg_size_y, true, gpu_arch::Xe>;

    /// @brief Arguments for gemm::run.
    /// User should prepare mat_in_ptr, mat_out_ptr, matrix_m, matrix_n, matrix_in_ld,
    /// matrix_out_ld, wg_ld_start_x, wg_ld_start_y, wg_st_start_x, wg_st_start_y.
    ///
    struct arguments_t {
        dtype_in *mat_in_ptr;
        dtype_out *mat_out_ptr;
        uint32_t matrix_m;
        uint32_t matrix_n;
        uint32_t matrix_in_ld;
        uint32_t matrix_out_ld;
        dtype_compute *amax_ptr;
        dtype_compute *scale;
        uint32_t wg_ld_start_x;
        uint32_t wg_ld_start_y;
        uint32_t wg_st_start_x;
        uint32_t wg_st_start_y;
    };

    /// @brief
    ///
    struct get_barrier_count {
        static constexpr uint32_t count
                = (wg_size_x * wg_size_y > 1) ? wg_size_x * wg_size_y : 0;
    };

    /// @brief
    ///
    struct get_slm_size {
        static constexpr uint32_t size = (wg_size_x * wg_size_y > 1)
                ? wg_size_x * wg_size_y * sizeof(dtype_compute)
                : 0;
    };

    /// @brief Main execution function for data_transformer.
    /// The basic process is load data -> data_transformer -> write out.
    /// @param item Is the sycl::nd_item.
    /// @param args Includes base pointer and matrix size.
    /// @return
    __XETLA_API static void call(sycl::nd_item<3> &item, arguments_t *args) {
        int tid_x = item.get_local_id(2);
        int tid_y = item.get_local_id(1);
        uint32_t sg_id = item.get_local_linear_id();

        global_ld_t mat_global_ld;
        global_ld_payload_t global_ld_payload;
        global_st_t mat_global_st;
        global_st_payload_t global_st_payload;
        global_compute_t mat_global_compute;

        //input and output starting point
        int global_ld_start_x;
        int global_ld_start_y;

        if constexpr (mem_layout_in == mem_layout::row_major) {
            global_ld_start_x = args->wg_ld_start_x + tid_x * sg_tile_n;
            global_ld_start_y = args->wg_ld_start_y + tid_y * sg_tile_m;
        } else {
            global_ld_start_x = args->wg_ld_start_x + tid_y * sg_tile_m;
            global_ld_start_y = args->wg_ld_start_y + tid_x * sg_tile_n;
        }

        int global_st_start_x = args->wg_st_start_x + tid_x * sg_tile_n;
        int global_st_start_y = args->wg_st_start_y + tid_y * sg_tile_m;

        if constexpr (mem_layout_in == mem_layout::row_major) {
            global_ld_payload.init(args->mat_in_ptr, args->matrix_n,
                    args->matrix_m, args->matrix_in_ld, global_ld_start_x,
                    global_ld_start_y);
        } else {
            global_ld_payload.init(args->mat_in_ptr, args->matrix_m,
                    args->matrix_n, args->matrix_in_ld, global_ld_start_x,
                    global_ld_start_y);
        }

        global_st_payload.init(args->mat_out_ptr, args->matrix_n,
                args->matrix_m, args->matrix_out_ld, global_st_start_x,
                global_st_start_y);

        subgroup::tile_load(mat_global_ld, global_ld_payload);

        if constexpr (need_fp8_op) {
            subgroup::elemwise_cvt(mat_global_compute, mat_global_ld);

            static constexpr uint32_t simd = 16;
            uint64_t offset = 0;

            xetla_vector<dtype_compute, 1> local_scale
                    = xetla_load_global<dtype_compute, 1,
                            data_size::default_size, cache_hint::cached,
                            cache_hint::cached>(args->scale, offset);

            mat_global_compute.reg
                    = mat_global_compute.reg * (dtype_compute)(local_scale[0]);

            subgroup::elemwise_cvt(mat_global_st, mat_global_compute);

            wg_reduce_t wg_reduce;
            wg_reduce.init(sg_id, 0, 0);

            mat_global_compute.reg = xetla_abs<dtype_compute,
                    global_compute_t::tile_desc::tile_elems>(
                    mat_global_compute.reg);

            xetla_vector<dtype_compute, 1> local_wg_max
                    = wg_reduce(mat_global_compute.reg);

            xetla_mask<simd> pred(0);
            pred[0] = 1;

            xetla_vector<dtype_compute, simd> local_max(local_wg_max[0]);
            xetla_vector<uint32_t, simd> offsets
                    = xetla_vector_gen<uint32_t, simd>(0, 1);

            xetla_tatomic_store_global<dtype_compute, simd,
                    cache_hint::uncached, cache_hint::write_back,
                    atomic_op::fmax>((uint64_t)args->amax_ptr,
                    offsets * sizeof(dtype_compute), local_max, pred);
        } else {
            subgroup::elemwise_cvt(mat_global_st, mat_global_ld);
        }

        subgroup::tile_store<cache_hint::uncached>(
                mat_global_st, global_st_payload);
    }
};

} // namespace gpu::xetla::kernel
