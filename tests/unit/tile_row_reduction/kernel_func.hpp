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
        int theight, int bwidth, int bheight, bool transform = false,
        bool transpose = false>
struct tile_row_reduction_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        using matA_tile_desc_t = tile_desc_t<twidth, theight, bwidth, bheight,
                reg_layout::tiled>;
        using matC_tile_desc_t
                = tile_desc_t<twidth, 1, bwidth, 1, reg_layout::tiled>;
        using matA_t = tile_t<dtype, matA_tile_desc_t>;
        using matC_t = tile_t<dtype, matC_tile_desc_t>;
        using matA_payload_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::global>,
                matA_tile_desc_t,
                msg_type_v<matA_tile_desc_t, mem_space::global>, gpu_arch::Xe>;
        using matC_payload_t = mem_payload_t<
                mem_desc_t<dtype, mem_layout::row_major, mem_space::global>,
                matC_tile_desc_t,
                msg_type_v<matC_tile_desc_t, mem_space::global>, gpu_arch::Xe>;
        matA_t matA;
        matC_t matC;
        matA_payload_t matA_payload(a, swidth, sheight, spitch, 0, 0);
        matC_payload_t matC_payload(c, swidth, sheight, spitch, 0, 0);
        matC.init(0);
        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                matA, matA_payload);
        matC.reg = subgroup::tile_reduce<reduce_op::sum, dtype, dtype, 0>(matA);
        tile_store(matC, matC_payload);
    }
};
