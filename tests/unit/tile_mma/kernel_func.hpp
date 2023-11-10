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

template <typename dtypeA, typename dtypeB, typename dtypeC, typename dtypeAcc,
        uint32_t m, uint32_t n, uint32_t k>
struct tile_mma_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtypeA *a, dtypeB *b, dtypeC *c) {

        int width_a = k;
        int height_a = m;
        int pitch_a = k;
        int start_x_a = 0;
        int start_y_a = 0;

        int width_b = n;
        int height_b = k;
        int pitch_b = n;
        int start_x_b = 0;
        int start_y_b = 0;

        using matA_tile_desc_t
                = tile_desc_t<k, m, 32 / sizeof(dtypeA), 8, reg_layout::tiled>;
        using matB_tile_desc_t = tile_desc_t<n, k, 16, 32 / sizeof(dtypeB),
                reg_layout::vnni_tiled>;
        using matC_tile_desc_t = tile_desc_t<n, m, 16, 8, reg_layout::tiled>;
        using matA_t = tile_t<dtypeA, matA_tile_desc_t>;
        using matB_t = tile_t<dtypeB, matB_tile_desc_t>;
        using matC_t = tile_t<dtypeC, matC_tile_desc_t>;
        using matA_payload_t = mem_payload_t<
                mem_desc_t<dtypeA, mem_layout::row_major, mem_space::global>,
                matA_tile_desc_t,
                msg_type_v<matA_tile_desc_t, mem_space::global>, gpu_arch::Xe>;
        using matB_payload_t = mem_payload_t<
                mem_desc_t<dtypeB, mem_layout::row_major, mem_space::global>,
                matB_tile_desc_t,
                msg_type_v<matB_tile_desc_t, mem_space::global>, gpu_arch::Xe>;
        using matC_payload_t = mem_payload_t<
                mem_desc_t<dtypeC, mem_layout::row_major, mem_space::global>,
                matC_tile_desc_t, msg_type::block_2d, gpu_arch::Xe>;
        using matAcc_t
                = tile_t<dtypeAcc, tile_desc_t<n, m, 16, 8, reg_layout::tiled>>;

        using tile_mma = tile_mma_t<matAcc_t, matAcc_t, matB_t, matA_t,
                mma_engine::xmx, gpu_arch::Xe>;

        matA_t matA;
        matB_t matB;
        matA_payload_t matA_payload;
        matB_payload_t matB_payload;
        matAcc_t matAcc;
        matC_t matC;
        matC_payload_t matC_payload;
        matA_payload.init(a, width_a, height_a, pitch_a, start_x_a, start_y_a);
        matB_payload.init(b, width_b, height_b, pitch_b, start_x_b, start_y_b);
        matAcc.init(0);

        constexpr tdesc_update_dir matA_update_dir = tdesc_update_dir::x_dir;
        constexpr tdesc_update_dir matB_update_dir = tdesc_update_dir::y_dir;

        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                matA, matA_payload);
        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                matB, matB_payload);
        SW_BARRIER();
        tile_mma::mma(matAcc, matAcc, matB, matA);
        SW_BARRIER();
        matC.reg = xetla_cvt<dtypeC, dtypeAcc, matAcc_t::tile_desc::tile_elems>(
                matAcc.reg);
        matC_payload.init(c, n, m, n, 0, 0);
        tile_store(matC, matC_payload);
    }
};
