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
#include "common/utils/tensor_descriptor.hpp"

#define IN_RANGE(x, l, r) ((x) >= (l) && (x) <= (r))

namespace gpu::xetla {

namespace core {
template <gpu_arch arch, typename T>
struct general_1d {};
template <gpu_arch arch, typename T>
class block_2d {};

template <typename T>
struct general_1d<gpu_arch::Xe, T> {
    template <uint8_t NElts>
    static inline bool check_restriction(uint64_t offset, uint64_t p = 0) {
        static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "block 1d only support D32/D64");
        static_assert(NElts == 1 || NElts == 2 || NElts == 3 || NElts == 4
                        || NElts == 8 || NElts == 16 || NElts == 32
                        || NElts == 64,
                "block 1d not support the NElts");

        bool ret = ((p + offset) % sizeof(T) == 0);
        XETLA_ASSERT(
                ret, "block 1d require data size aligned but is %u", offset);
        return ret;
    }

    template <uint8_t NElts, int N, typename Toffset = uint32_t>
    static inline bool check_restriction(
            xetla_vector<Toffset, N> offsets, uint64_t p = 0) {
        static_assert(
                N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32,
                "scatter not supported N");
        if constexpr (NElts != 1) {
            static_assert(NElts == 2 || NElts == 3 || NElts == 4 || NElts == 8,
                    "scatter not supported NElts");
            for (auto i = 0; i < N; i++) {
                bool ret = ((p + offsets[i]) % sizeof(T) == 0);
                XETLA_ASSERT(ret, "scatter require data size aligned but is %u",
                        offsets[i]);
                if (!ret) { return false; }
            }
        }

        return true;
    }
};

template <typename T>
class block_2d<gpu_arch::Xe, T> {
public:
    template <bool transpose, bool vnni_transform>
    static inline bool check_load(xetla_tdescriptor tdesc) {
        if (!check_common(tdesc)) { return false; }

        bool ret = false;
        uint32_t block_width
                = gpu::xetla::detail::xetla_get_block_width_x(tdesc) + 1;
        uint32_t block_height
                = gpu::xetla::detail::xetla_get_block_width_y(tdesc) + 1;
        uint8_t array_len
                = gpu::xetla::detail::xetla_get_block_array_len(tdesc) + 1;
        int32_t block_start_x
                = gpu::xetla::detail::xetla_get_tensor_offset_x(tdesc);

        ret = ((block_width * block_height * element_size)
                <= (32 * bytes_per_grf));
        XETLA_ASSERT(ret,
                "2D Block Loads upto 32 GRFs are can be read but is %u:%u",
                block_width, block_height);
        if (!ret) { return false; }

        if constexpr (transpose || vnni_transform) {
            if constexpr (transpose) {
                ret = (array_len == 1);
                XETLA_ASSERT(ret,
                        "Transposed load do not allow an array length more "
                        "than 1 but is %d",
                        array_len);
                if (!ret) { return false; }

                ret = (element_size == 4 || element_size == 8);
                XETLA_ASSERT(ret,
                        "For 2D Block Load with Transpose, allowed data "
                        "sizes "
                        "are d32 and d64 only but is %u",
                        element_size);
                if (!ret) { return false; }

                auto block_height_in_bytes = block_height * element_size;
                ret = (block_height_in_bytes == 4 || block_height_in_bytes == 8
                        || block_height_in_bytes == 16
                        || block_height_in_bytes == 32
                        || block_height_in_bytes == 64
                        || block_height_in_bytes == 128);
                XETLA_ASSERT(ret,
                        "For 2D load with Transpose, the pre-operation "
                        "block "
                        "Height is padded up to the next power-of-two "
                        "value "
                        "(minimum 4 bytes)"
                        "but is %u:%u",
                        block_height, element_size);
                if (!ret) { return false; }

                if (element_size == 4) {
                    ret = (IN_RANGE(block_height, 1, 32)
                            && IN_RANGE(block_width, 1, 8));
                    XETLA_ASSERT(ret,
                            "block height must be in 1 ~ 32, width in 1 ~ "
                            "8 "
                            "but is %u:%u",
                            block_height, block_width);
                    if (!ret) { return false; }
                } else {
                    ret = (block_start_x >= 0);
                    XETLA_ASSERT(ret,
                            "When element is d64, block X_offset must be "
                            "non-negative but is %d",
                            block_start_x);
                    if (!ret) { return false; }

                    ret = ((block_height == 8)
                            && (block_width == 1 || block_width == 2
                                    || block_width == 4));
                    XETLA_ASSERT(ret,
                            "block height must be 8, width is 1/2/4 but is "
                            "%u:%u",
                            block_height, block_width);
                    if (!ret) { return false; }
                }
            }

            if constexpr (vnni_transform) {
                ret = (element_size == 1 || element_size == 2);
                XETLA_ASSERT(ret,
                        "For 2D Block Load with VNNI Transform, allowed "
                        "data "
                        "sizes are d8 and d16 only. but is %u",
                        element_size);
                if (!ret) { return false; }

                if constexpr (element_size == 1) {
                    ret = IN_RANGE(block_height, 4, 32)
                            && IN_RANGE(block_width, 4, 16)
                            && (array_len == 1 || array_len == 2
                                    || array_len == 4)
                            && (block_width * array_len) <= 64
                            && (block_height * element_size) % 4 == 0;
                    XETLA_ASSERT(ret,
                            "For 2D Block Load with VNNI Transform, height "
                            "in "
                            "4 ~ 32, width in 4 ~ 16, array len 1/2/4, and "
                            "width X array_length <= 64 but is %u:%u:%u",
                            block_height, block_width, array_len);
                    if (!ret) { return false; }
                } else {
                    ret = IN_RANGE(block_height, 2, 32)
                            && IN_RANGE(block_width, 2, 16)
                            && (array_len == 1 || array_len == 2
                                    || array_len == 4)
                            && (block_width * array_len) <= 32
                            && (block_height * element_size) % 4 == 0;
                    XETLA_ASSERT(ret,
                            "For 2D Block Load with VNNI Transform, height "
                            "in "
                            "2 ~ 32, width in 2 ~ 16, array len 1/2/4, and "
                            "width X array_length <= 32 but is %u:%u:%u",
                            block_height, block_width, array_len);
                    if (!ret) { return false; }
                }
            }
        } else {
            ret = ((block_width * array_len * element_size) <= 64);
            XETLA_ASSERT(ret,
                    "2D block load operations block_width times array_size "
                    "must "
                    "not exceed 64 bytes but is %u:%u",
                    block_width, array_len);
            if (!ret) { return false; }
        }

        return true;
    }

    static inline bool check_store(xetla_tdescriptor tdesc) {
        if (!check_common(tdesc)) { return false; }

        uint32_t block_width
                = gpu::xetla::detail::xetla_get_block_width_x(tdesc) + 1;
        uint32_t block_height
                = gpu::xetla::detail::xetla_get_block_width_y(tdesc) + 1;
        uint8_t array_len
                = gpu::xetla::detail::xetla_get_block_array_len(tdesc) + 1;

        bool ret = false;

        ret = (element_size == 1 || element_size == 2 || element_size == 4
                || element_size == 8);
        XETLA_ASSERT(ret,
                "2D block store only support element size 1/2/4/8 but is "
                "%u",
                element_size);
        if (!ret) { return false; }

        ret = IN_RANGE(block_height, 1, 8);
        XETLA_ASSERT(ret, "2D block height only support 1 ~ 8 but is %u",
                block_height);
        if (!ret) { return false; }

        ret = (array_len == 1);
        XETLA_ASSERT(
                ret, "2D block array len only support 1 but is %u", array_len);
        if (!ret) { return false; }

        ret = ((block_width * element_size) >= 4
                && (block_width * element_size) <= 64
                && (block_width * block_height * element_size) <= 512);
        XETLA_ASSERT(ret,
                "2D block Store, block width * element size in 4 ~ 64 and "
                "Total GRF data should not exceed 512 bytes but is "
                "%u:%u:%u",
                block_width, block_height, element_size);
        if (!ret) { return false; }

        return true;
    }

    static inline bool check_surface(
            uint64_t base, uint32_t width, uint32_t height, uint32_t pitch) {
        if (check_base_address(base) && check_surface_width(width)
                && check_surface_height(height)
                && check_surface_pitch(pitch, width)) {
            return true;
        }

        return false;
    }

private:
    static constexpr auto element_size = sizeof(T);
    static constexpr uint32_t max_24bit = 16 * 1024 * 1024; // 2 ^ 24
    static constexpr auto bytes_per_grf
            = register_attr_t<grf_mode::double_grf, gpu_arch::Xe>::reg_in_bytes;

    static inline bool check_base_address(uint64_t base) {
        bool ret = ((base % 64) == 0);
        XETLA_ASSERT(
                ret, "Base address must be CL (64B) aligned but is %p", base);
        if (!ret) { return false; }

        return ret;
    }

    static inline bool check_surface_width(uint32_t width_in_bytes) {
        bool ret = (width_in_bytes <= max_24bit);
        XETLA_ASSERT(ret,
                "Only 24 bits are supported for surface width(in bytes) "
                "but is %u",
                width_in_bytes);
        if (!ret) { return false; }

        ret = (width_in_bytes >= 64);
        XETLA_ASSERT(ret,
                "Surface width(in bytes) must be equal or greater than 64B "
                "but is %u",
                width_in_bytes);
        if (!ret) { return false; }

        ret = ((width_in_bytes % (element_size > 4 ? element_size : 4)) == 0);
        XETLA_ASSERT(ret,
                "Surface width(in bytes) must be aligned to MAX(DW, "
                "element_size) "
                "but is "
                "%u",
                width_in_bytes);
        if (!ret) { return false; }

        return ret;
    }

    static inline bool check_surface_height(uint32_t height_in_elements) {
        bool ret = (height_in_elements < max_24bit);
        XETLA_ASSERT(ret,
                "Only 24 bits are supported for surface height(in "
                "elements) but is %u",
                height_in_elements);
        if (!ret) { return false; }

        return ret;
    }

    static inline bool check_surface_pitch(
            uint32_t pitch_in_bytes, uint32_t width_in_bytes) {
        bool ret = (pitch_in_bytes >= width_in_bytes);
        XETLA_ASSERT(ret,
                "Pitch(in bytes) must be greater or equal to Width(in "
                "bytes) but is %u:%u",
                pitch_in_bytes, width_in_bytes);
        if (!ret) { return false; }

        ret = (pitch_in_bytes <= max_24bit);
        XETLA_ASSERT(ret,
                "Only 24 bits are supported for surface pitch(in bytes) "
                "but is %u",
                pitch_in_bytes);
        if (!ret) { return false; }

        ret = (pitch_in_bytes >= 64);
        XETLA_ASSERT(ret,
                "Surface pitch(in bytes) must be equal or greater than 64B "
                "but is %u",
                pitch_in_bytes);
        if (!ret) { return false; }

        ret = ((pitch_in_bytes % 8) == 0);
        XETLA_ASSERT(ret,
                "Surface pitch(in bytes) must be a multiple of QW (8 "
                "bytes) but is "
                "%u",
                pitch_in_bytes);
        if (!ret) { return false; }

        return ret;
    }

    static inline bool check_block_start_x(int32_t x_in_elements) {
        bool ret = true;

        if constexpr (element_size == 1 || element_size == 2) {
            ret = ((element_size == 1 && (x_in_elements % 4) == 0)
                    || (element_size == 2 && (x_in_elements % 2) == 0));
            XETLA_ASSERT(ret,
                    "For element d8, Block StartX must be a multiple of "
                    "4. For element d16, Block Start X must be a "
                    "multiple of 2. but is "
                    "%d:%u",
                    x_in_elements, element_size);
            if (!ret) { return false; }
        }

        return ret;
    }

    static inline bool check_block_width(uint32_t width_in_elements) {
        bool ret = IN_RANGE(width_in_elements, 1, 64);
        XETLA_ASSERT(ret,
                "Block width(in elements) must be between 1-64 but is %u",
                width_in_elements);
        if (!ret) { return false; }

        auto width_in_bytes = width_in_elements * element_size;

        ret = (width_in_bytes % 4 == 0);
        XETLA_ASSERT(ret,
                "Block width in bytes must "
                "be a "
                "multiple of 4 bytes but is %u",
                width_in_bytes);
        if (!ret) { return false; }

        ret = (width_in_bytes == 4 || width_in_bytes == 8
                || width_in_bytes == 16 || width_in_bytes == 32
                || width_in_bytes == 64 || width_in_bytes == 128
                || width_in_bytes == 256 || width_in_bytes == 512);
        XETLA_ASSERT(ret,
                "2D block load/store , the block width in bytes is padded "
                "up to 2^X "
                "in the GRF but is %u:%u",
                width_in_elements, element_size);
        if (!ret) { return false; }

        return ret;
    }

    static inline bool check_block_height(uint32_t height_in_elements) {
        bool ret = IN_RANGE(height_in_elements, 1, 32);
        XETLA_ASSERT(ret,
                "Block height(in elements) must be between 1-32 but is %u",
                height_in_elements);
        if (!ret) { return false; }
        return ret;
    }

    static inline bool check_array_len(uint32_t len) {
        bool ret = IN_RANGE(len, 1, 4);
        XETLA_ASSERT(ret, "Array Length must be in 1-4 but is %u", len);
        if (!ret) { return false; }
        return ret;
    }

    static inline bool check_block(int32_t x, int32_t y, uint32_t width,
            uint32_t height, uint8_t array_len) {
        if (check_block_start_x(x) && check_block_width(width)
                && check_block_height(height) && check_array_len(array_len)) {
            return true;
        }
        return false;
    }

    static inline bool check_common(xetla_tdescriptor tdesc) {
        uint64_t base
                = gpu::xetla::detail::xetla_get_tensor_base_address(tdesc);
        uint32_t surface_width
                = gpu::xetla::detail::xetla_get_tensor_width_x(tdesc) + 1;
        uint32_t surface_height
                = gpu::xetla::detail::xetla_get_tensor_width_y(tdesc) + 1;
        uint32_t surface_pitch
                = gpu::xetla::detail::xetla_get_tensor_pitch_x(tdesc) + 1;
        int32_t block_start_x
                = gpu::xetla::detail::xetla_get_tensor_offset_x(tdesc);
        int32_t block_start_y
                = gpu::xetla::detail::xetla_get_tensor_offset_y(tdesc);

        uint32_t block_width
                = gpu::xetla::detail::xetla_get_block_width_x(tdesc) + 1;
        uint32_t block_height
                = gpu::xetla::detail::xetla_get_block_width_y(tdesc) + 1;
        uint8_t array_len
                = gpu::xetla::detail::xetla_get_block_array_len(tdesc) + 1;

        if (check_surface(base, surface_width, surface_height, surface_pitch)
                && check_block(block_start_x, block_start_y, block_width,
                        block_height, array_len)) {
            return true;
        }

        return false;
    }
};
} // namespace core

namespace subgroup {

template <gpu_arch arch, typename dtype, typename mem_dtype>
struct check_load {
    template <bool mem_transform, size_t block_size_x>
    struct global_2d {
        using load_store_attr = typename arch_attr_t<
                arch>::template load_store_attr<msg_type::block_2d>;
        static constexpr int32_t max_vnni_block_width
                = load_store_attr::max_vnni_load_width_in_elems;
        static_assert(!mem_transform || block_size_x <= max_vnni_block_width,
                "For VNNI transform, the maximum block width is 16 width.");
        static constexpr int32_t max_block_width
                = load_store_attr::max_load_width_in_bytes / sizeof(dtype);
        static_assert((max_block_width % block_size_x) == 0,
                "max_block_width should be a multiply of block size x.");
    };

    struct global_1d {
        static_assert(sizeof(mem_dtype) == 4 || sizeof(mem_dtype) == 8,
                "tile 1d only support D32/D64");
    };

    template <bool mem_transform, size_t block_size_x>
    struct unaligned_2d {
        using load_store_attr = typename arch_attr_t<
                arch>::template load_store_attr<msg_type::block_2d>;
        static constexpr int32_t max_vnni_block_width
                = load_store_attr::max_vnni_load_width_in_elems;
        static_assert(!mem_transform || block_size_x <= max_vnni_block_width,
                "For VNNI transform, the maximum block width is 16 width.");
        static constexpr int32_t max_block_width
                = load_store_attr::max_load_width_in_bytes / sizeof(dtype);
        static_assert((max_block_width % block_size_x) == 0,
                "max_block_width should be a multiply of block size x.");
    };

    template <mem_layout memory_layout, size_t block_size_x, size_t tile_bytes,
            size_t min_bytes, size_t block_bytes, size_t num_channel_x,
            size_t num_channel>
    struct local_scatter {
        static_assert(memory_layout == mem_layout::row_major,
                "only support row major in local load, you can use local "
                "store "
                "to "
                "do the transpose");
        static_assert(
                sizeof(mem_dtype) >= 4, "load size should at least DW aligned");
        static_assert((block_size_x * sizeof(dtype) % sizeof(mem_dtype)) == 0,
                "bytes per row should be a multiply of sizeof load_dtype");
        static_assert(((tile_bytes % min_bytes) == 0
                              && (block_bytes % min_bytes) == 0),
                "currently, we are not able to handle the corner case");
        static_assert((num_channel_x > 0) && (num_channel_x <= num_channel),
                "The number of simd channel x should be greater than 0 and "
                "less "
                "than num_channel");
    };

    struct local_1d {
        static_assert(sizeof(mem_dtype) == 4 || sizeof(mem_dtype) == 8,
                "tile 1d only support D32/D64");
    };
};

template <gpu_arch arch, typename dtype, typename mem_dtype = uint32_t>
struct check_store {
    template <size_t block_size_x>
    struct global_2d {
        using load_store_attr = typename arch_attr_t<
                arch>::template load_store_attr<msg_type::block_2d>;

        static constexpr int32_t max_block_width
                = load_store_attr::max_load_width_in_bytes / sizeof(dtype);
        static_assert((max_block_width % block_size_x) == 0,
                "max_block_width should be a multiply of block size x.");
    };

    struct global_1d {
        static_assert(sizeof(mem_dtype) == 4 || sizeof(mem_dtype) == 8,
                "tile 1d only support D32/D64");
    };

    template <size_t block_size_x>
    struct unaligned_2d {
        using load_store_attr = typename arch_attr_t<
                arch>::template load_store_attr<msg_type::block_2d>;

        static constexpr int32_t max_block_width
                = load_store_attr::max_load_width_in_bytes / sizeof(dtype);
        static_assert((max_block_width % block_size_x) == 0,
                "max_block_width should be a multiply of block size x.");
    };

    template <size_t tile_bytes, size_t min_store_bytes, size_t block_bytes,
            size_t num_channel_x, size_t num_channel>
    struct global_atomic {
        static_assert(std::is_same<remove_const_t<dtype>, uint32_t>::value
                        || std::is_same<remove_const_t<dtype>, uint64_t>::value
                        || std::is_same<remove_const_t<dtype>, int>::value
                        || std::is_same<remove_const_t<dtype>, float>::value
                        || std::is_same<remove_const_t<dtype>, double>::value,
                "for global atomic add, we only support fp32 and fp64, "
                "uin32_t, uint64_t, int");
        static_assert(((tile_bytes % min_store_bytes) == 0
                              && (block_bytes % min_store_bytes) == 0),
                "currently, we are not able to handle the corner case");
        static_assert((num_channel_x > 0) && (num_channel_x <= num_channel),
                "The number of simd channel x should be greater than 0 and "
                "less "
                "than num_channel");
        static_assert(sizeof(dtype) >= 4, "Only support DW and QW atomic add");
    };

    template <size_t tile_bytes, size_t min_bytes, size_t block_bytes,
            size_t num_channel_x, size_t num_channel>
    struct local_scatter {
        static_assert(((tile_bytes % min_bytes) == 0
                              && (block_bytes % min_bytes) == 0),
                "currently, we are not able to handle the corner case");
        static_assert((num_channel_x > 0) && (num_channel_x <= num_channel),
                "The number of simd channel x should be greater than 0 and "
                "less "
                "than num_channel");
    };

    template <size_t tile_bytes, size_t min_store_bytes, size_t block_bytes,
            size_t num_channel_x, size_t num_channel>
    struct local_scatter_vnni_col {
        static_assert(((tile_bytes % min_store_bytes) == 0
                              && (block_bytes % min_store_bytes) == 0),
                "currently, we are not able to handle the corner case");

        static_assert((num_channel_x > 0) && (num_channel_x <= num_channel),
                "The number of simd channel x should be greater than 0 and "
                "less "
                "than num_channel");
    };

    struct local_1d {
        static_assert(sizeof(mem_dtype) == 4 || sizeof(mem_dtype) == 8,
                "tile 1d only support D32/D64");
    };
};
} // namespace subgroup

namespace group {
template <gpu_arch arch = gpu_arch::Xe, class enable = void>
struct gemm {};

template <gpu_arch arch>
struct gemm<arch, std::enable_if_t<(arch <= gpu_arch::Xe)>> {
    struct default_fpu {
        template <typename dtype_a, typename dtype_b, typename dtype_mma_a,
                typename dtype_mma_b, typename dtype_mma_acc>
        struct check_dtype_default {
            static_assert(
                    std::is_same<remove_const_t<dtype_mma_a>, float>::value,
                    "current only support sgemm");
            static_assert(
                    std::is_same<remove_const_t<dtype_mma_b>, float>::value,
                    "current only support sgemm");
            static_assert(
                    std::is_same<remove_const_t<dtype_mma_acc>, float>::value,
                    "current only support sgemm");
        };

        template <mem_layout mem_layout_a, mem_layout mem_layout_b,
                mem_space mem_space_a, mem_space mem_space_b>
        struct check_memory_default {
            static constexpr bool is_col_major_a
                    = mem_layout_a == mem_layout::col_major;
            static constexpr bool is_col_major_b
                    = mem_layout_b == mem_layout::col_major;
            static constexpr bool is_local_a = mem_space_a == mem_space::local;
            static constexpr bool is_local_b = mem_space_b == mem_space::local;
            static_assert(!is_local_a,
                    "current don't support matA load from local memory");
            static_assert(!is_local_b,
                    "current don't support matB load from local memory");
        };

        template <typename dtype_mma, int tile_size_x_a, int tile_size_y_a,
                int block_size_x_a, int block_size_y_a, int tile_size_x_b,
                int tile_size_y_b, int block_size_x_b, int block_size_y_b>
        struct check_tile_size_default {
            static constexpr uint32_t reg_in_bytes
                    = register_attr_t<grf_mode::double_grf,
                            gpu_arch::Xe>::reg_in_bytes;
            static constexpr uint32_t simd_len
                    = reg_in_bytes / sizeof(dtype_mma);

            static_assert((block_size_x_b % simd_len == 0),
                    "block_size_x_b should be a multiple of simd_len");
            static_assert((tile_size_x_a % block_size_x_a) == 0);
            static_assert((tile_size_y_b % block_size_y_b) == 0);
            static_assert(block_size_x_a == block_size_y_b);
        };
    };

    struct default_xmx {
        template <typename dtype_a, typename dtype_b, typename dtype_mma_a,
                typename dtype_mma_b>
        struct check_dtype_default {
            static_assert(std::is_same<remove_const_t<dtype_mma_a>,
                                  remove_const_t<dtype_mma_b>>::value,
                    "dtype_mma_a should be the same as dtype_mma_b in xe "
                    "arch ");
            static_assert((sizeof(dtype_mma_a) == sizeof(dtype_a))
                            || (sizeof(dtype_mma_a) == 2 * sizeof(dtype_a))
                            || (2 * sizeof(dtype_mma_a) == sizeof(dtype_a)),
                    "Current we cannot support fp32 <->fp8, since it will "
                    "meet a "
                    "lot "
                    "of HW limitations. ");
            static_assert((sizeof(dtype_mma_b) == sizeof(dtype_b))
                            || (sizeof(dtype_mma_b) == 2 * sizeof(dtype_b))
                            || (2 * sizeof(dtype_mma_b) == sizeof(dtype_b)),
                    "Current we cannot support fp32 <->fp8, since it will "
                    "meet a "
                    "lot "
                    "of HW limitations. ");
        };

        template <mem_layout mem_layout_a, mem_layout mem_layout_b,
                mem_space mem_space_a, mem_space mem_space_b>
        struct check_memory_default {
            static constexpr bool is_col_major_a
                    = mem_layout_a == mem_layout::col_major;
            static constexpr bool is_col_major_b
                    = mem_layout_b == mem_layout::col_major;
            static constexpr bool is_local_a = mem_space_a == mem_space::local;
            static constexpr bool is_local_b = mem_space_b == mem_space::local;
            static_assert(!is_local_a || !is_col_major_a,
                    "if matA load from local memory, then matA should be "
                    "row-major");
            static_assert(!is_local_b || !is_col_major_b,
                    "if matB load from local memory, then matB should be "
                    "row-major");
        };

        template <typename dtype_mma, int tile_size_x_a, int tile_size_y_a,
                int block_size_x_a, int block_size_y_a, int tile_size_x_b,
                int tile_size_y_b, int block_size_x_b, int block_size_y_b>
        struct check_tile_size_default {
            using mma_attr = mma_attr_t<gpu_arch::Xe>;
            static constexpr int32_t mma_m = mma_attr::mma_m_in_elem;
            static constexpr int32_t mma_n = mma_attr::mma_n_in_elem;
            static constexpr int32_t mma_k
                    = mma_attr::mma_k_in_bytes / sizeof(dtype_mma);

            static_assert(tile_size_x_a % mma_k == 0,
                    "tile_size_x_a should be a multiple of mma_k");
            static_assert(block_size_x_a == mma_k,
                    "block_size_x_a should be equal to mma_k");
            static_assert(tile_size_y_a % mma_m == 0,
                    "tile_size_y_a should be a multiple of mma_m");
            static_assert(block_size_y_a % mma_m == 0,
                    "block_size_y_a should be a multiple of mma_m");

            static_assert(tile_size_x_b % mma_n == 0,
                    "tile_size_x_b should be a multiple of mma_n");
            static_assert(block_size_x_b == mma_n,
                    "block_size_x_b should be equal to mma_n");
            static_assert(tile_size_y_b % mma_k == 0,
                    "tile_size_y_b should be a multiple of mma_k");
            static_assert(block_size_y_b % mma_k == 0,
                    "block_size_y_b should be a multiple of mma_k");
        };
    };
};
} // namespace group

namespace kernel {
template <gpu_arch arch, typename T>
class general_1d {};
template <gpu_arch arch, typename T>
class block_2d {};

template <typename T>
class general_1d<gpu_arch::Xe, T> {
public:
    static inline bool check_alignment(T *base, uint32_t pitch) {
        auto pitch_in_bytes = pitch * element_size;
        bool ret = ((pitch_in_bytes % pitch_alignment_bytes) == 0);
        XETLA_ASSERT(ret, "Pitch in byte must be a multiple of 4 but is %u:%u",
                pitch, element_size);
        if (!ret) { return false; }

        ret = (pitch_in_bytes >= min_pitch_bytes);
        XETLA_ASSERT(ret,
                "Pitch in byte must be equal or greater than 4B but is "
                "%u:%u",
                pitch, element_size);
        if (!ret) { return false; }

        ret = ((((uint64_t)base) % base_alignment_bytes) == 0);
        XETLA_ASSERT(ret, "Base address must be 4B aligned but is %p", base);
        return ret;
    }

private:
    static constexpr size_t element_size = sizeof(T);
    static constexpr size_t pitch_alignment_bytes = 4;
    static constexpr size_t min_pitch_bytes = 4;
    static constexpr size_t base_alignment_bytes = 4;
};

template <typename T>
class block_2d<gpu_arch::Xe, T> {
public:
    static inline bool check_tensor(
            uint64_t base, uint32_t width, uint32_t height, uint32_t pitch) {
        return core::block_2d<gpu_arch::Xe, T>::check_surface(
                base, width * element_size, height, pitch * element_size);
    }

private:
    static constexpr size_t element_size = sizeof(T);
};
} // namespace kernel

} // namespace gpu::xetla
