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

template <typename dtype, uint32_t tile_size_x, uint32_t tile_size_y,
        uint32_t block_size_x, uint32_t block_size_y>
struct conversion_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        using linear_desc = subgroup::tile_desc_t<tile_size_x, tile_size_y,
                tile_size_x, tile_size_y, reg_layout::linear>;
        using tiled_desc = subgroup::tile_desc_t<tile_size_x, tile_size_y,
                block_size_x, block_size_y, reg_layout::tiled>;

        using linear_tile_t = subgroup::tile_t<dtype, linear_desc>;
        using tiled_tile_t = subgroup::tile_t<dtype, tiled_desc>;

        using payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::global>,
                tiled_desc, msg_type::block_2d, gpu_arch::Xe>;

        tiled_tile_t data_tile;
        linear_tile_t linear_data_tile;

        payload_t data_payload(a, tile_size_x, tile_size_y, tile_size_x, 0, 0);
        payload_t result_payload(
                c, tile_size_x, tile_size_y, tile_size_x, 0, 0);

        auto mask = xetla_vector_gen<dtype, tile_size_x>(0, 1);

        tile_load(data_tile, data_payload);
        layout_convert(linear_data_tile, data_tile); // tiled to linear

        auto linear_data_tile_2d
                = linear_data_tile.reg.xetla_format<native_type_t<dtype>,
                        tile_size_y, tile_size_x>();
#pragma unroll
        for (int i = 0; i < tile_size_y; ++i) {
            linear_data_tile_2d.xetla_select<1, 1, tile_size_x, 1>(i, 0)
                    += mask;
        }

        layout_convert(data_tile, linear_data_tile); // linear to tiled
        tile_store(data_tile, result_payload);
    }
};