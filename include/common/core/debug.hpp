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

#include "common/core/common.hpp"
#include <CL/sycl.hpp>
#include <ext/intel/esimd.hpp>

namespace gpu::xetla {

// debug context
// =========================================================
#if defined DEBUG && defined LOG_PRINT
namespace debug_ctx {

static constexpr size_t reg_start = 128 * 64; // start from GRF128 and down

namespace nd_item {
using element_type = uint16_t;
static constexpr size_t element_num = 8;
static constexpr size_t max_dims = 3;
static constexpr size_t dims_pos = 0;
static constexpr size_t dims_global_start = 1;
static constexpr size_t dims_local_start = 1 + max_dims;

static constexpr size_t nd_item_offset
        = reg_start - element_num * sizeof(element_type);
static inline ESIMD_PRIVATE ESIMD_REGISTER(nd_item_offset)
        __ESIMD_NS::simd<element_type, element_num> saved_nd_item;

template <size_t dims>
static inline void set(sycl::nd_item<dims> item) {
    static_assert(dims <= max_dims);

    saved_nd_item[dims_pos] = dims;

#pragma unroll
    for (auto i = 0; i < dims; i++) {
        saved_nd_item[dims_global_start + i] = item.get_group(i);
    }

#pragma unroll
    for (auto i = 0; i < dims; i++) {
        saved_nd_item[dims_local_start + i] = item.get_local_id(i);
    }
}

static inline uint16_t get_dims() {
    return saved_nd_item[dims_pos];
}

static inline int16_t get_group_id(size_t dim) {
    return saved_nd_item[dims_global_start + dim];
}

static inline int16_t get_local_id(size_t dim) {
    return saved_nd_item[dims_local_start + dim];
}
}; // namespace nd_item
}; // namespace debug_ctx
#endif

// EOT message
// =========================================================
#if 0 // has bug in current driver, will open this in next driver
static constexpr size_t exit_offset = reg_start - 8 * sizeof(int);
ESIMD_PRIVATE ESIMD_REGISTER(exit_offset) __ESIMD_NS::simd<int, 8> reg_exit;
ESIMD_INLINE void xetla_thread_exit() {
    constexpr uint32_t exDesc = 0x0;
    constexpr uint32_t desc = 0x02000010;
    constexpr uint8_t execSize = 0x83;
    constexpr uint8_t sfid = 0x3;
    constexpr uint8_t numSrc0 = 0x1;
    constexpr uint8_t numSrc1 = 0x0;
    constexpr uint8_t isEOT = 0x1;
    return sycl::ext::intel::experimental::esimd::raw_send(
            reg_exit, exDesc, desc, execSize, sfid, numSrc0, isEOT);
}
#endif

// 1. define XETLA_PRINTF
// =========================================================
#ifdef LOG_PRINT
// log on
#define STR_APPEND(a, b, c) a b c
#ifdef __SYCL_DEVICE_ONLY__
// kernel printf
#ifdef DEBUG
#define XETLA_PRINTF(s, ...) \
    do { \
        const __attribute__((opencl_constant)) char f[] = STR_APPEND( \
                "[XeTLA] [KERNEL] [group(%d, %d, %d), local(%d, " \
                "%d, %d)] : ", \
                s, "\n"); \
        sycl::ext::oneapi::experimental::printf(f, \
                debug_ctx::nd_item::get_group_id(0), \
                debug_ctx::nd_item::get_group_id(1), \
                debug_ctx::nd_item::get_group_id(2), \
                debug_ctx::nd_item::get_local_id(0), \
                debug_ctx::nd_item::get_local_id(1), \
                debug_ctx::nd_item::get_local_id(2), ##__VA_ARGS__); \
    } while (0)
#else
#define XETLA_PRINTF(s, ...) \
    do { \
        const __attribute__((opencl_constant)) char f[] \
                = STR_APPEND("[XeTLA] [KERNEL] : ", s, "\n"); \
        sycl::ext::oneapi::experimental::printf(f, ##__VA_ARGS__); \
    } while (0)
#endif
#else
// host printf
#define XETLA_PRINTF(s, ...) \
    do { \
        const char *f = STR_APPEND("[XeTLA] [HOST] : ", s, "\n"); \
        printf(f, ##__VA_ARGS__); \
    } while (0)
#endif

#else
// log off
#define XETLA_PRINTF(s, ...) \
    do { \
    } while (0)
#endif

// 2. define XETLA_ASSERT
// =========================================================
#ifdef __SYCL_DEVICE_ONLY__
// kernel assert
#define XETLA_ASSERT(c, s, ...) \
    do { \
    } while (0)
#else
// host asset
#ifdef DEBUG
// host assert in debug version
#define XETLA_ASSERT(c, s, ...) \
    do { \
        if (!(c)) { XETLA_PRINTF(s, ##__VA_ARGS__); } \
    } while (0)
#else
// host assert in release version
#define XETLA_ASSERT(c, s, ...) \
    do { \
    } while (0)
#endif
#endif

// 3. define DEBUG_INVOKE
// =========================================================
#ifdef DEBUG
enum class dbg_level : uint8_t {
    kernel = 0,
    workgroup = 1,
    subgroup = 2,
    core = 3
};
#define DEBUG_INVOKE(level, ...) \
    do { \
        if constexpr (DEBUG >= static_cast<uint8_t>(level)) { \
            if (!(__VA_ARGS__)) { XETLA_PRINTF("L%d: " #__VA_ARGS__, level); } \
        } \
    } while (0)
#else
#define DEBUG_INVOKE(level, ...) \
    do { \
    } while (0)
#endif

} // namespace gpu::xetla
