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
using namespace gpu::xetla::subgroup;

template <typename dtype, int dst_swidth, int dst_sheight, int dst_spitch,
        int twidth, int theight, int bwidth, int bheight,
        bool transform = false, bool transpose = false,
        int src_spitch = dst_spitch>
struct tile_load_store_func {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {

        constexpr int ele_per_dw
                = transform ? sizeof(uint32_t) / sizeof(dtype) : 1;

        static constexpr reg_layout a_reg_layout
                = transform ? reg_layout::vnni_tiled : reg_layout::tiled;
        static constexpr mem_layout a_mem_layout
                = transpose ? mem_layout::col_major : mem_layout::row_major;

        using tile_desc_a
                = tile_desc_t<twidth, theight, bwidth, bheight, a_reg_layout>;
        using tile_desc_c = tile_desc_t<twidth * ele_per_dw,
                theight / ele_per_dw, bwidth * ele_per_dw, bheight / ele_per_dw,
                reg_layout::tiled>;

        using matA_t = tile_t<dtype, tile_desc_a>;
        using matC_t = tile_t<dtype, tile_desc_c>;

        using payload_load_t
                = mem_payload_t<dtype, tile_desc_a, msg_type::block_2d,
                        a_mem_layout, mem_space::global, gpu_arch::Xe>;

        using payload_store_t
                = mem_payload_t<dtype, tile_desc_c, msg_type::block_2d,
                        mem_layout::row_major, mem_space::global, gpu_arch::Xe>;

        matA_t matA;
        matC_t matC;

        mem_desc_t<dtype, a_mem_layout, mem_space::global> mem_desc_a(
                {a}, {dst_swidth, dst_sheight, src_spitch}, {0, 0});
        mem_desc_t<dtype, mem_layout::row_major, mem_space::global> mem_desc_c(
                {c},
                {dst_swidth * ele_per_dw, dst_sheight / ele_per_dw,
                        dst_spitch * ele_per_dw},
                {0, 0});

        payload_load_t payload_load(mem_desc_a);
        payload_store_t payload_store(mem_desc_c);

        tile_load(matA, payload_load);
        matC.reg = matA.reg;
        tile_store(matC, payload_store);
    }
};

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight, bool transform = false,
        bool transpose = false>
struct tile_load_store_atomic_func {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {

        mem_desc_t<dtype, mem_layout::row_major, mem_space::global> mem_desc_c(
                {c}, {swidth, sheight, spitch}, {0, 0});

        using tile_desc = tile_desc_t<twidth, theight, bwidth, bheight,
                reg_layout::tiled>;

        using matA_t = tile_t<dtype, tile_desc>;
        using matBias_t = tile_t<dtype, tile_desc>;

        using payload_block_2d_t
                = mem_payload_t<dtype, tile_desc, msg_type::block_2d,
                        mem_layout::row_major, mem_space::global, gpu_arch::Xe>;
        using payload_atomic_t
                = mem_payload_t<dtype, tile_desc, msg_type::atomic_add,
                        mem_layout::row_major, mem_space::global, gpu_arch::Xe>;

        matA_t matA;
        matBias_t matBias;
        payload_block_2d_t payload_load(a, swidth, sheight, spitch, 0, 0);
        payload_block_2d_t payload_store(mem_desc_c);
        payload_atomic_t payload_store_add(mem_desc_c);

        tile_load(matA, payload_load);
        matBias.reg = matA.reg;
        matA.reg = 0;
        tile_store(matA, payload_store);
        SW_BARRIER();
        tile_store(matBias, payload_store_add);
    }
};

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight, bool transform = false,
        bool transpose = false>
struct tile_load_broadcast_store_func {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {

        using tile_desc_a
                = tile_desc_t<twidth, 1, bwidth, 1, reg_layout::tiled>;
        using tile_desc_c = tile_desc_t<twidth, theight, bwidth, bheight,
                reg_layout::tiled>;

        using matA_t = tile_t<dtype, tile_desc_a>;
        using matC_t = tile_t<dtype, tile_desc_c>;

        using payload_load_t
                = mem_payload_t<dtype, tile_desc_a, msg_type::block_1d,
                        mem_layout::row_major, mem_space::global, gpu_arch::Xe>;

        using payload_store_t
                = mem_payload_t<dtype, tile_desc_c, msg_type::block_2d,
                        mem_layout::row_major, mem_space::global, gpu_arch::Xe>;

        payload_load_t payload_load(a, swidth, sheight, spitch, 0, 0);
        payload_store_t payload_store(c, swidth, sheight, spitch, 0, 0);

        matA_t matA;
        matC_t matC;

        tile_load(matA, payload_load);

        subgroup::row_broadcast(matC, matA);
        tile_store(matC, payload_store);
    }
};

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight, bool transform = false,
        bool transpose = false>
struct tile_load_store_1d_func {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {

        using tile_desc = tile_desc_t<twidth, theight, bwidth, bheight,
                reg_layout::tiled>;
        using matA_t = tile_t<dtype, tile_desc>;

        using payload_t = mem_payload_t<dtype, tile_desc, msg_type::block_1d,
                mem_layout::row_major, mem_space::global, gpu_arch::Xe>;

        matA_t matA;
        payload_t payload_load(a, swidth, sheight, spitch, 0, 0);
        payload_t payload_store(c, swidth, sheight, spitch, 0, 0);

        tile_load(matA, payload_load);
        tile_store(matA, payload_store);
    }
};

template <typename dtype, int dst_swidth, int dst_sheight, int dst_spitch,
        int twidth, int theight, int bwidth, int bheight, int offset_x,
        int offset_y, int src_spitch = dst_spitch>
struct tile_padding_load_store_func {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {

        mem_desc_t<dtype, mem_layout::row_major, mem_space::global> mem_desc_c(
                {c}, {dst_swidth, dst_sheight, dst_spitch}, {0, 0});

        using tile_desc = tile_desc_t<twidth, theight, bwidth, bheight,
                reg_layout::tiled>;
        using matA_t = tile_t<dtype, tile_desc>;

        using payload_t = mem_payload_t<dtype, tile_desc, msg_type::block_2d,
                mem_layout::row_major, mem_space::global, gpu_arch::Xe>;

        matA_t matA;

        payload_t payload_load(
                a, dst_swidth, dst_sheight, src_spitch, offset_x, offset_y);
        payload_t payload_store(mem_desc_c);

        tile_load(matA, payload_load);
        tile_store(matA, payload_store);
    }
};
