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

#include "common/core/common.hpp"

namespace gpu::xetla::limitation {

namespace slm {
static inline void check_alignment(const char *tag, auto offset) {
    if (offset % 4) {
        DEVICE_PRINTF("%s: Base-address of SLM must be 4B aligned but is %d\n",
                tag, offset);
    }
}

static inline void check_alignment(auto offset) {
    check_alignment("Unknown", offset);
}
} // namespace slm

namespace brgemm {
namespace default_fpu {
template <typename dtype_a, typename dtype_b, typename dtype_mma_a,
        typename dtype_mma_b, typename dtype_mma_acc>
struct check_dtype_default_fpu_xe {
    static_assert(std::is_same<remove_const_t<dtype_a>,
                          remove_const_t<dtype_mma_a>>::value,
            "in current fpu path, dtype_mma_a should be the same as "
            "dtype_a");
    static_assert(std::is_same<remove_const_t<dtype_b>,
                          remove_const_t<dtype_mma_b>>::value,
            "in current fpu path, dtype_mma_a should be the same as "
            "dtype_a");

    static_assert(std::is_same<remove_const_t<dtype_mma_a>, float>::value,
            "current only support sgemm");
    static_assert(std::is_same<remove_const_t<dtype_mma_b>, float>::value,
            "current only support sgemm");
    static_assert(std::is_same<remove_const_t<dtype_mma_acc>, float>::value,
            "current only support sgemm");
};

template <mem_layout mem_layout_a, mem_layout mem_layout_b,
        mem_space mem_space_a, mem_space mem_space_b>
struct check_memory_default_fpu_xe {
    static constexpr bool is_col_major_a
            = mem_layout_a == mem_layout::col_major;
    static constexpr bool is_col_major_b
            = mem_layout_b == mem_layout::col_major;
    static constexpr bool is_local_a = mem_space_a == mem_space::local;
    static constexpr bool is_local_b = mem_space_b == mem_space::local;
    static_assert(
            !is_local_a, "current don't support matA load from local memory");
    static_assert(
            !is_local_b, "current don't support matB load from local memory");
};

template <typename arch_attr, typename dtype_mma, int tile_size_x_a,
        int tile_size_y_a, int block_size_x_a, int block_size_y_a,
        int tile_size_x_b, int tile_size_y_b, int block_size_x_b,
        int block_size_y_b>
struct check_tile_size_default_fpu_xe {
    using register_attr = typename arch_attr::register_attr;
    static constexpr uint32_t reg_in_bytes = register_attr::reg_in_bytes;
    static constexpr uint32_t simd_len = reg_in_bytes / sizeof(dtype_mma);

    static_assert((block_size_x_b % simd_len == 0),
            "block_size_x_b should be a multiple of simd_len");
};
} // namespace default_fpu

namespace default_xmx {
template <typename dtype_a, typename dtype_b, typename dtype_mma_a,
        typename dtype_mma_b>
struct check_dtype_default_xmx_xe {
    static_assert(std::is_same<remove_const_t<dtype_mma_a>,
                          remove_const_t<dtype_mma_b>>::value,
            "dtype_mma_a should be the same as dtype_mma_b in xe arch ");
    static_assert((sizeof(dtype_mma_a) == sizeof(dtype_a))
                    || (sizeof(dtype_mma_a) == 2 * sizeof(dtype_a))
                    || (2 * sizeof(dtype_mma_a) == sizeof(dtype_a)),
            "Current we cannot support fp32 <->fp8, since it will meet a lot "
            "of HW limitations. ");
    static_assert((sizeof(dtype_mma_b) == sizeof(dtype_b))
                    || (sizeof(dtype_mma_b) == 2 * sizeof(dtype_b))
                    || (2 * sizeof(dtype_mma_b) == sizeof(dtype_b)),
            "Current we cannot support fp32 <->fp8, since it will meet a lot "
            "of HW limitations. ");
};

template <mem_layout mem_layout_a, mem_layout mem_layout_b,
        mem_space mem_space_a, mem_space mem_space_b>
struct check_memory_default_xmx_xe {
    static constexpr bool is_col_major_a
            = mem_layout_a == mem_layout::col_major;
    static constexpr bool is_col_major_b
            = mem_layout_b == mem_layout::col_major;
    static constexpr bool is_local_a = mem_space_a == mem_space::local;
    static constexpr bool is_local_b = mem_space_b == mem_space::local;
    static_assert(
            !is_local_b, "current don't support matB load from local memory");
    static_assert(!is_local_a || !is_col_major_a,
            "if matA load from local memory, then matA should be row-major");
};

template <typename arch_attr, typename dtype_mma, int tile_size_x_a,
        int tile_size_y_a, int block_size_x_a, int block_size_y_a,
        int tile_size_x_b, int tile_size_y_b, int block_size_x_b,
        int block_size_y_b>
struct check_tile_size_default_xmx_xe {
    using mma_attr = typename arch_attr::mma_attr;
    static constexpr int32_t mma_m = mma_attr::mma_m_in_elem;
    static constexpr int32_t mma_n = mma_attr::mma_n_in_elem;
    static constexpr int32_t mma_k
            = mma_attr::mma_k_in_bytes / sizeof(dtype_mma);

    static_assert(tile_size_x_a % mma_k == 0,
            "tile_size_x_a should be a multiple of mma_k");
    static_assert(
            block_size_x_a == mma_k, "block_size_x_a should be equal to mma_k");
    static_assert(tile_size_y_a % mma_m == 0,
            "tile_size_y_a should be a multiple of mma_m");
    static_assert(block_size_y_a % mma_m == 0,
            "block_size_y_a should be a multiple of mma_m");

    static_assert(tile_size_x_b % mma_n == 0,
            "tile_size_x_b should be a multiple of mma_n");
    static_assert(
            block_size_x_b == mma_n, "block_size_x_b should be equal to mma_n");
    static_assert(tile_size_y_b % mma_k == 0,
            "tile_size_y_b should be a multiple of mma_k");
    static_assert(block_size_y_b % mma_k == 0,
            "block_size_y_b should be a multiple of mma_k");
};
} // namespace default_xmx
} // namespace brgemm

} // namespace gpu::xetla::limitation
