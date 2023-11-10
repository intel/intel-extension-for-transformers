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

#include "common/utils/common.hpp"
#include "common/utils/raw_send_load_store.hpp"

namespace gpu::xetla {

template <int dim = 2>
struct mem_coord_t {};
template <>
struct mem_coord_t<2> {
    int x;
    int y;
    inline mem_coord_t(int x_, int y_) : x(x_), y(y_) {}
    inline mem_coord_t() = default;
    inline mem_coord_t(const mem_coord_t<2> &coord) {
        this->x = coord.x;
        this->y = coord.y;
    }
    inline mem_coord_t<2> &operator=(const mem_coord_t<2> &coord) {
        // Be aware of the risks:
        // self_assign: No protection against the object assigning to itself.
        // if (this == &coord){
        //     return *this;
        // }

        this->x = coord.x;
        this->y = coord.y;
        return *this;
    }
    inline void init(int x_, int y_) {
        this->x = x_;
        this->y = y_;
    }
};

template <int dim = 2>
struct mem_shape_t {};
template <>
struct mem_shape_t<2> {
    uint32_t x;
    uint32_t y;
    uint32_t stride;
    inline mem_shape_t() = default;
    inline mem_shape_t(
            uint32_t shape_x_, uint32_t shape_y_, uint32_t row_stride_)
        : x(shape_x_), y(shape_y_), stride(row_stride_) {}
    inline mem_shape_t(const mem_shape_t<2> &shape) {
        this->x = shape.x;
        this->y = shape.y;
        this->stride = shape.stride;
    }
    inline mem_shape_t<2> &operator=(const mem_shape_t<2> &shape) {
        // Be aware of the risks:
        // self_assign: No protection against the object assigning to itself.
        // if (this == &shape){
        //     return *this;
        // }
        this->x = shape.x;
        this->y = shape.y;
        this->stride = shape.stride;
        return *this;
    }
    inline void init(
            uint32_t shape_x_, uint32_t shape_y_, uint32_t row_stride_) {
        this->x = shape_x_;
        this->y = shape_y_;
        this->stride = row_stride_;
    }
};

template <typename dtype_, mem_space space_>
struct mem_base_t {};
template <typename dtype_>
struct mem_base_t<dtype_, mem_space::global> {
    using dtype = dtype_;
    dtype *base;
    inline mem_base_t() = default;
    inline mem_base_t(dtype *base_) : base(base_) {}
    inline mem_base_t(const mem_base_t<dtype, mem_space::global> &mem_base)
        : base(mem_base.base) {}
    inline mem_base_t<dtype, mem_space::global> &operator=(
            const mem_base_t<dtype, mem_space::global> &mem_base) {
        // Be aware of the risks:
        // self_assign: No protection against the object assigning to itself.
        // if (this == &mem_base){
        //     return *this;
        // }
        this->base = mem_base.base;
        return *this;
    }
    inline void init(dtype *base_) { base = base_; }
    inline void update(int offset) { base = base + offset; }
};
template <typename dtype_>
struct mem_base_t<dtype_, mem_space::local> {
    using dtype = dtype_;
    uint32_t base;
    inline mem_base_t() = default;
    inline mem_base_t(uint32_t base_) { init(base_); }
    inline mem_base_t(const mem_base_t<dtype, mem_space::local> &mem_base) {
        init(mem_base.base);
    }
    inline mem_base_t<dtype, mem_space::local> &operator=(
            const mem_base_t<dtype, mem_space::local> &mem_base) {
        // Be aware of the risks:
        // self_assign: No protection against the object assigning to itself.
        // if (this == &mem_base){
        //     return *this;
        // }
        init(mem_base.base);
        return *this;
    }
    inline void init(uint32_t base_) { base = base_; }
    inline void update(int offset) { init(base + offset * sizeof(dtype)); }
};

template <typename dtype_, mem_layout layout_, mem_space space_,
        uint32_t alignment_ = 8, int dim_ = 2>
struct mem_desc_t {};

template <typename dtype_, mem_layout layout_, mem_space space_,
        uint32_t alignment_>
struct mem_desc_t<dtype_, layout_, space_, alignment_, 2> {
    using dtype = dtype_;
    static constexpr mem_layout layout = layout_;
    static constexpr mem_space space = space_;
    static constexpr int dim = 2;
    static constexpr uint32_t alignment = alignment_;
    static constexpr uint32_t alignment_in_bytes = alignment_ * sizeof(dtype);

    static constexpr bool is_col_major = layout == mem_layout::col_major;
    static constexpr bool is_local = space == mem_space::local;
    using shape_t = mem_shape_t<dim>;
    using coord_t = mem_coord_t<dim>;
    using base_t = mem_base_t<dtype, space>;

    using this_type_t = mem_desc_t<dtype, layout_, space_, alignment, 2>;

    inline mem_desc_t() = default;
    inline mem_desc_t(base_t base_, shape_t shape_, coord_t coord_)
        : base(base_), shape(shape_), coord(coord_) {}
    // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
    // Please check if you need to add self-define destructor
    // inline ~mem_desc_t(){}
    inline mem_desc_t(const this_type_t &mem_desc)
        : base(mem_desc.base), shape(mem_desc.shape), coord(mem_desc.coord) {}

    inline this_type_t &operator=(const this_type_t &mem_desc) {
        this->base = mem_desc.base;
        this->shape = mem_desc.shape;
        this->coord = mem_desc.coord;
        return *this;
    }
    inline void init(base_t base_, shape_t shape_, coord_t coord_) {
        base = base_;
        shape = shape_;
        coord = coord_;
    }
    inline void update_coord(int offset_x, int offset_y) {
        coord.x += offset_x;
        coord.y += offset_y;
    }
    inline void update_coord_x(int offset_x) { coord.x += offset_x; }
    inline void update_coord_y(int offset_y) { coord.y += offset_y; }
    inline xetla_tdescriptor get_tdesc() {
        uint32_t width = is_col_major ? shape.y : shape.x;
        uint32_t height = is_col_major ? shape.x : shape.y;
        uint32_t pitch = shape.stride;
        int coord_x = is_col_major ? coord.y : coord.x;
        int coord_y = is_col_major ? coord.x : coord.y;
        return xetla_get_tdesc<dtype>(
                base.base, width, height, pitch, coord_x, coord_y);
    }

    shape_t shape;
    coord_t coord;
    base_t base;
};

} // namespace gpu::xetla
