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

template <uint32_t element_size>
constexpr uint32_t get_execSize_code() {
    static_assert(element_size == 1 || element_size == 2 || element_size == 4
                    || element_size == 8 || element_size == 16
                    || element_size == 32,
            "element_size not supported!");
    switch (element_size) {
        case 1: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
        case 16: return 4;
        case 32: return 5;
    }
}

template <typename dtype, int SIMD>
struct raw_send_with_2_source_and_1_destination_func {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {
        uint64_t offset = sizeof(int) * SIMD * ei->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(dtype);
        xetla_vector<uint64_t, SIMD> dsec = offsets + (uint64_t)a;
        offsets += offset;

        xetla_vector<dtype, SIMD> A_load_vec = xetla_load_global(a, offsets);
        xetla_vector<dtype, SIMD> Dst = A_load_vec;

        constexpr uint32_t size = SIMD * sizeof(dtype) / 64;
        constexpr uint32_t addr_size = SIMD * sizeof(uint64_t) / 64;
        uint32_t msg_desc = 12;
        msg_desc |= 0x3 << 7;
        msg_desc |= 0x2 << 9;
        msg_desc |= 0x0 << 12;
        msg_desc |= 0x0 << 15;
        msg_desc |= 0x2 << 17;
        msg_desc |= size << 20;
        msg_desc |= addr_size << 25;

        uint32_t exDesc = 0;
        constexpr uint32_t sfid = 0xF;
        constexpr uint32_t numSrc1 = size;
        constexpr uint32_t numSrc0 = addr_size;
        constexpr uint32_t numDst = size;
        constexpr uint32_t execSize = get_execSize_code<SIMD>();

        xetla_raw_send<dtype, SIMD, uint64_t, SIMD, dtype, SIMD, execSize, sfid,
                numSrc0, numSrc1, numDst>(
                Dst.xetla_format<dtype>(), dsec, A_load_vec, exDesc, msg_desc);

        xetla_store_global(c, offsets, Dst);
    }
};
