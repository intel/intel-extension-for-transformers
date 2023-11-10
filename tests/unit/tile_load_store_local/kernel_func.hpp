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

#include "subgroup/subgroup.hpp"
#include "xetla.hpp"

using namespace gpu::xetla;
using namespace gpu::xetla::subgroup;

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight, int doffset = 0,
        bool transform = false, bool transpose = false>
struct tile_load_store_local_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        using tile_desc = tile_desc_t<twidth, theight, bwidth, bheight>;
        using mat_t = tile_t<dtype, tile_desc>;
        using payload_global_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::global>,
                tile_desc, msg_type::block_2d, gpu_arch::Xe>;
        using payload_local_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
                tile_desc, msg_type::scatter, gpu_arch::Xe>;

        payload_global_t global_ld_payload(a, swidth, sheight, spitch, 0, 0);
        payload_global_t global_st_payload(c, swidth, sheight, spitch, 0, 0);
        payload_local_t slm_ld_payload(
                (uint32_t)0, twidth, theight, twidth, 0, 0);
        payload_local_t slm_st_payload(
                (uint32_t)0, twidth, theight, twidth, doffset, 0);
        slm_ld_payload.update_tdesc(doffset);
        mat_t mat;
        tile_load(mat, global_ld_payload);
        tile_store(mat, slm_st_payload);
        mat.reg = 0;
        tile_load(mat, slm_ld_payload);
        tile_store(mat, global_st_payload);
    }
};

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight, int doffset = 0,
        bool transform = false, bool transpose = false>
struct tile_load_store_vnni_local_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        using tile_desc = tile_desc_t<twidth, theight, bwidth, bheight>;
        using vnni_tile_desc = tile_desc_t<twidth, theight, bwidth, bheight,
                reg_layout::vnni_tiled>;
        using mat_t = tile_t<dtype, tile_desc>;
        using vnni_mat_t = tile_t<dtype, vnni_tile_desc>;
        using payload_global_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::global>,
                tile_desc, msg_type::block_2d, gpu_arch::Xe>;
        using vnni_payload_global_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::global>,
                vnni_tile_desc, msg_type::block_2d, gpu_arch::Xe>;
        using payload_local_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
                tile_desc, msg_type::scatter, gpu_arch::Xe>;
        using vnni_payload_local_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
                vnni_tile_desc, msg_type::scatter, gpu_arch::Xe>;

        payload_global_t global_ld_payload(a, swidth, sheight, spitch, 0, 0);
        vnni_payload_global_t vnni_global_ld_payload(
                a, swidth, sheight, spitch, 0, 0);
        payload_global_t global_st_payload(c, swidth, sheight, spitch, 0, 0);
        vnni_payload_global_t vnni_global_st_payload(
                c, swidth, sheight, spitch, 0, 0);

        payload_local_t slm_ld_payload(
                (uint32_t)0, twidth, theight, twidth, 0, 0);
        vnni_payload_local_t vnni_slm_ld_payload(
                (uint32_t)0, twidth, theight, twidth, 0, 0);
        payload_local_t slm_st_payload(
                (uint32_t)0, twidth, theight, twidth, doffset, 0);
        vnni_payload_local_t vnni_slm_st_payload(
                (uint32_t)0, twidth, theight, twidth, doffset, 0);

        slm_ld_payload.update_tdesc(doffset);
        vnni_slm_ld_payload.update_tdesc(doffset);

        mat_t mat;
        vnni_mat_t vnni_mat;

        tile_load(mat, global_ld_payload);
        tile_store(mat, slm_st_payload);

        // store vnni format to c
        tile_load(vnni_mat, vnni_slm_ld_payload);
        tile_store(vnni_mat, vnni_global_st_payload);
        vnni_mat.reg = 0;
        // convert the format of a
        tile_load(vnni_mat, vnni_global_ld_payload);
        tile_store(vnni_mat, vnni_global_ld_payload);
    }
};

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight, bool transform = false,
        bool transpose = false>
struct tile_transpose_store_local_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {

        /// in order to use the same input and validation function, we should
        /// 1) transpose the data in global mem
        /// 2) load the col major data and transpose store to slm
        using tile_desc = tile_desc_t<twidth, theight, bwidth, bheight,
                sizeof(dtype) == 4 ? reg_layout::tiled
                                   : reg_layout::vnni_tiled>;
        using mat_t = tile_t<dtype, tile_desc>;
        using payload_matA_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::col_major, mem_space::global>,
                tile_desc, msg_type::block_2d, gpu_arch::Xe>;
        using payload_global_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::global>,
                tile_desc, msg_type::block_2d, gpu_arch::Xe>;
        using tile_config_store = tile_desc_t<twidth, theight, bwidth, bheight,
                reg_layout::vnni_tiled_col_major>;
        using mat_slm_t = tile_t<dtype, tile_config_store>;
        using slm_store_payload_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
                tile_config_store, msg_type::scatter, gpu_arch::Xe>;
        using tile_config_load = tile_desc_t<twidth, theight, bwidth, bheight>;
        using slm_load_payload_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
                tile_config_load, msg_type::scatter, gpu_arch::Xe>;

        mat_t mat;
        payload_matA_t payload_matA(a, swidth, sheight, spitch, 0, 0);
        tile_load(mat, payload_matA);
        mat_slm_t mat_slm;
        slm_store_payload_t local_store_A(
                (uint32_t)0, twidth, theight, twidth, 0, 0);
        slm_load_payload_t local_load_A(
                (uint32_t)0, twidth, theight, twidth, 0, 0);
        mat_slm.reg = mat.reg;
        tile_store(mat_slm, local_store_A);
        tile_load(mat_slm, local_load_A);
        mat.reg = 0;
        mat.reg = mat_slm.reg;
        payload_global_t payload_matC(c, swidth, sheight, spitch, 0, 0);
        tile_store(mat, payload_matC);
    }
};

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight, int doffset = 0,
        bool transform = false, bool transpose = false>
struct tile_load_store_1d_local_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        using tile_desc = tile_desc_t<twidth, theight, bwidth, bheight>;
        using mat_t = tile_t<dtype, tile_desc>;
        using payload_global = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::global>,
                tile_desc,
                theight == 1 ? msg_type::block_1d : msg_type::block_2d,
                gpu_arch::Xe>;
        using payload_local = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
                tile_desc, msg_type::block_1d, gpu_arch::Xe>;
        using payload_local_ld = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
                tile_desc, msg_type::scatter, gpu_arch::Xe>;

        mat_t mat;
        payload_local local_block((uint32_t)0, twidth, theight, twidth, 0, 0);
        payload_local_ld local_load((uint32_t)0, twidth, theight, twidth, 0, 0);
        payload_global global_load(a, swidth, sheight, spitch, 0, 0);
        payload_global global_store(c, swidth, sheight, spitch, 0, 0);

        local_block.update_tdesc(doffset);
        local_load.update_tdesc(doffset);

        tile_load(mat, global_load);
        tile_store(mat, local_block);
        mat.reg = 0;
        tile_load(mat, local_load);
        tile_store(mat, global_store);
    }
};
