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

#include <common/common.hpp>

using namespace gpu::xetla;

template <typename dtype, int SIMD, int BLOCK_SIZE,
        gpu_arch arch_tag = gpu_arch::Xe>
KERNEL_FUNC inline void vector_add_func(
        sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {

    xetla_tdescriptor a_src_tdesc;
    xetla_tdescriptor b_src_tdesc;
    xetla_fill_tdesc<dtype, SIMD, SIMD, 1>(a_src_tdesc.xetla_format<uint32_t>(),
            a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 0, 0);
    xetla_fill_tdesc<dtype, SIMD, SIMD, 1>(b_src_tdesc.xetla_format<uint32_t>(),
            b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 0, 0);

    xetla_tprefetch_global<dtype, cache_hint::cached, cache_hint::cached,
            arch_tag>(a_src_tdesc);
    xetla_tprefetch_global<dtype, cache_hint::cached, cache_hint::cached,
            arch_tag>(b_src_tdesc);
    xetla_vector<dtype, SIMD *SIMD> A_load_vec
            = xetla_tload_global<dtype, SIMD * SIMD, cache_hint::cached,
                    cache_hint::cached, false, false, arch_tag>(a_src_tdesc);
    xetla_vector<dtype, SIMD *SIMD> B_load_vec
            = xetla_tload_global<dtype, SIMD * SIMD, cache_hint::cached,
                    cache_hint::cached, false, false, arch_tag>(b_src_tdesc);

    xetla_vector<float, SIMD *SIMD> add_a
            = xetla_cvt<float, dtype, SIMD * SIMD>(A_load_vec);
    xetla_vector<float, SIMD *SIMD> add_b
            = xetla_cvt<float, dtype, SIMD * SIMD>(B_load_vec);

    xetla_vector<float, SIMD *SIMD> result_c = add_a + add_b;
    xetla_vector<dtype, SIMD *SIMD> out
            = xetla_cvt<dtype, float, SIMD * SIMD>(result_c);

    //2D block Store total GRF data should not exceed 512 bytes per message.
    constexpr int max_bytes_per_store = 512;
    constexpr int store_height = max_bytes_per_store / SIMD / sizeof(dtype);
    xetla_tdescriptor dst_tdesc;
    xetla_fill_tdesc<dtype, SIMD, store_height, 1>(
            dst_tdesc.xetla_format<uint32_t>(), c, BLOCK_SIZE, BLOCK_SIZE,
            BLOCK_SIZE, 0, 0);
#pragma unroll
    for (int unroll_i = 0; unroll_i < SIMD; unroll_i += store_height) {
        xetla_tstore_global<dtype, SIMD * store_height, cache_hint::write_back,
                cache_hint::write_back, arch_tag>(dst_tdesc,
                out.xetla_select<SIMD * store_height, 1>(SIMD * unroll_i));
        xetla_update_tdesc_offsety(
                dst_tdesc.xetla_format<uint32_t>(), store_height);
    }
}
