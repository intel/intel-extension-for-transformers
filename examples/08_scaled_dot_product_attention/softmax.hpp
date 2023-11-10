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
using namespace gpu::xetla::group;
using namespace gpu::xetla::subgroup;

template <typename dtype_in_, typename dtype_out_, typename tile_shape_,
        mem_space mem_space_in_, mem_space mem_space_out_, uint32_t SIMD_,
        uint32_t thread_num_, uint32_t softmax_size_>
struct xetla_softmax_fwd_t {
    using dtype_in = dtype_in_;
    using dtype_out = dtype_out_;
    using tile_shape = tile_shape_;

    static constexpr mem_space mem_space_in = mem_space_in_;
    static constexpr mem_space mem_space_out = mem_space_out_;

    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    static constexpr uint32_t wg_tile_m = sg_tile_m * wg_size_y;
    static constexpr uint32_t wg_tile_n = sg_tile_n * wg_size_x;

    static constexpr uint32_t SIMD = SIMD_;
    static constexpr uint32_t thread_num = thread_num_;
    static constexpr uint32_t softmax_size = softmax_size_;
    static constexpr uint32_t block_height = softmax_size / SIMD;

    // each tile load one row from SLM
    // change data surface to imp 2D-block load
    // SIMD is the tile width
    // 2 * blockHeight is the tile height
    // SIMD * 2 * blockHeight equals to the elements number of one row
    using softmax_tile_desc_t = subgroup::tile_desc_t<SIMD, block_height, SIMD,
            block_height, reg_layout::tiled>;
    using softmax_load_t = subgroup::tile_t<dtype_in, softmax_tile_desc_t>;
    using softmax_load_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_in, mem_layout::row_major, mem_space_in>,
            softmax_tile_desc_t,
            subgroup::msg_type_v<softmax_tile_desc_t, mem_space_in>,
            gpu_arch::Xe>;

    // this tile will store the softmax result to global memory
    using softmax_store_t = subgroup::tile_t<dtype_out, softmax_tile_desc_t>;
    using softmax_store_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_out, mem_layout::row_major, mem_space_out>,
            softmax_tile_desc_t,
            subgroup::msg_type_v<softmax_tile_desc_t, mem_space_out>,
            gpu_arch::Xe>;

    struct arguments_t {
        // available while original data is from SLM
        uint32_t data_in_base;
        // available while processed data is to SLM
        uint32_t data_out_base;
        // available while original data is from global memory
        dtype_in *data_in_ptr;
        // available while processed data is to global memory
        dtype_out *data_out_ptr;
    };

    __XETLA_API KERNEL_FUNC void operator()(
            sycl::nd_item<3> &item, arguments_t *args) {

        softmax_load_t softmax_load;
        softmax_load_payload_t softmax_load_payload;
        softmax_store_t softmax_store;
        softmax_store_payload_t softmax_store_payload;

        uint32_t local_offset_y = block_height * item.get_local_linear_id();

        // read original data from SLM:
        // reshape 1 * 512 to 32 * 16
        // each thread load two rows:
        // thread#i will load row#i and row#(i + 32)

        softmax_load_payload.init(args->data_in_base, SIMD,
                block_height * wg_tile_m, SIMD, 0, local_offset_y);
        softmax_store_payload.init(args->data_out_base, SIMD,
                block_height * wg_tile_m, SIMD, 0, local_offset_y);

        xetla_vector<float, softmax_size> row_data_32;

        uint32_t inner_loop_count = (wg_tile_m % thread_num == 0)
                ? wg_tile_m / thread_num
                : (wg_tile_m / thread_num) + 1;

        for (int row = 0; row < inner_loop_count; ++row) {
            subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                    softmax_load, softmax_load_payload);
            softmax_load_payload.template update_tdesc<tdesc_update_dir::y_dir>(
                    block_height * thread_num);

            row_data_32 = softmax_load.reg.xetla_select<softmax_size, 1>(0);

            // get max
            float xmax = hmax<float, float, softmax_size>(row_data_32);

            // get exp_sum
            row_data_32 -= xmax;
            row_data_32 = exp(row_data_32);
            float exp_sum = sum<float, float, softmax_size>(row_data_32);

            // get softmax elementwise result
            row_data_32 /= exp_sum;

            softmax_store.reg.xetla_select<softmax_size, 1>(0)
                    = xetla_cvt<dtype_out, float, softmax_size>(row_data_32);

            tile_store(softmax_store, softmax_store_payload);
            softmax_store_payload
                    .template update_tdesc<tdesc_update_dir::y_dir>(
                            block_height * thread_num);
        }
    } // void run()
}; // struct xetla_softmax_fwd_t
