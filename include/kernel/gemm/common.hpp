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

#include "common/common.hpp"
#include "group/group.hpp"
#include "subgroup/subgroup.hpp"

namespace gpu::xetla::kernel {
namespace detail {

template <typename T>
inline bool check_2d_block_restriction(T *base, uint32_t mat_ld) {
    bool implementable = true;
    constexpr int pitch_alignment_bytes = 8;
    constexpr int base_alignment_bytes = 64;
    constexpr int min_pitch_bytes = 64;

    implementable &= (mat_ld * sizeof(T) % pitch_alignment_bytes) == 0;
    implementable &= mat_ld * sizeof(T) >= min_pitch_bytes;
    implementable &= (uint64_t(base) % base_alignment_bytes) == 0;

    return implementable;
}

template <typename T>
inline bool check_dw_align(T *base, uint32_t mat_ld) {
    bool implementable = true;
    constexpr int pitch_alignment_bytes = 4;
    constexpr int base_alignment_bytes = 4;
    constexpr int min_pitch_bytes = 4;

    implementable &= (mat_ld * sizeof(T) % pitch_alignment_bytes) == 0;
    implementable &= mat_ld * sizeof(T) >= min_pitch_bytes;
    implementable &= (uint64_t(base) % base_alignment_bytes) == 0;

    return implementable;
}
} // namespace detail

} // namespace gpu::xetla::kernel
