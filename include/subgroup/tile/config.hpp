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

#include "subgroup/tile/common.hpp"

namespace gpu::xetla::subgroup {

/// @brief
///
/// @tparam update_dir_
/// @tparam omode_
/// @tparam cyclic_size_
template <tdesc_update_dir update_dir_,
        offset_mode omode_ = offset_mode::const_offset,
        uint32_t cyclic_size_ = 1>
struct mem_update_config_t {
    static constexpr tdesc_update_dir update_dir = update_dir_;
    static constexpr offset_mode omode = omode_;
    static constexpr uint32_t cyclic_size = cyclic_size_;
    //    static_assert((omode != offset_mode::const_offset) || cyclic_size == 1,
    //            "for const_offset, the cyclic size should be 1");
};

} // namespace gpu::xetla::subgroup
