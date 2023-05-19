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

#include "experimental/kernel/layer_norm/common.hpp"

namespace gpu::xetla::kernel {

/// @brief Sets up attribute of the layer norm.
///
/// @tparam wg_tile_n_ Is the num of cols processed by one workgroup. Should equal to matrix_n in the current design.
/// @tparam wg_tile_m_ Is the num of rows processed by one workgroup in each inner loop. Mainly used for row reduction in the BWD path
/// @tparam sg_tile_n_ Is the num of cols processed by one subgroup. wg_tile_n % sg_tile_n == 0.
/// @tparam sg_tile_m_ Is the num of rows processed by one subgroup in each inner loop. Mainly used for row reduction in the BWD path
/// @tparam wg_num_m_ Is the num of total workgroups launched in y direction, will be used in static persistent thread mode.
/// @tparam wg_num_n_ Is the num of total workgroups launched in x direction. Currently, it should be 1.
template <uint32_t wg_tile_n_, uint32_t wg_tile_m_, uint32_t sg_tile_n_,
        uint32_t sg_tile_m_ = 1, uint32_t wg_num_m_ = 1, uint32_t wg_num_n_ = 1>
struct layer_norm_attr_t {
    static constexpr uint32_t wg_tile_m = wg_tile_m_;
    static constexpr uint32_t wg_tile_n = wg_tile_n_;
    static constexpr uint32_t sg_tile_m = sg_tile_m_;
    static constexpr uint32_t sg_tile_n = sg_tile_n_;
    static constexpr uint32_t wg_num_m = wg_num_m_;
    static constexpr uint32_t wg_num_n = wg_num_n_;

    static_assert(sg_tile_m == 1,
            "Currently, we don't see the value to set sg_tile_m > 1. Maybe it "
            "can be used to save the L1 BW when load gamma/beta");
    static_assert(wg_num_n == 1,
            "Current design doesn't support cross workgroup sync. So, wg_num_n "
            "should be 1, i.e. one entire row should be processed inside the "
            "workgroup.");
    static_assert(wg_tile_n % sg_tile_n == 0,
            "Current design we don't enable the boundary check");
};

} // namespace gpu::xetla::kernel
