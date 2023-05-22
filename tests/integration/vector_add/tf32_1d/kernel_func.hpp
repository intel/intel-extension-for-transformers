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

#include "common/common.hpp"

using namespace gpu::xetla;

template <typename dtype, int SIMD>
KERNEL_FUNC inline void vector_add_func(
        xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {
    uint64_t offset = sizeof(dtype) * SIMD * ei->get_group(0);
    xetla_vector<uint32_t, SIMD> offsets
            = xetla_vector_gen<uint32_t, SIMD>(0, 1);
    offsets *= sizeof(dtype);
    offsets += offset;
    /// use scattered prefetch for a
    xetla_prefetch_global<dtype, 1, data_size::default_size,
            cache_hint::streaming, cache_hint::cached, SIMD>(a, offsets);
    /// use block prefetch for b
    xetla_prefetch_global<dtype, SIMD, data_size::default_size,
            cache_hint::cached, cache_hint::cached>(b, offset);
    SW_BARRIER();
    /// use scattered load for a
    xetla_vector<dtype, SIMD> ivector1
            = xetla_load_global<dtype, 1, data_size::default_size,
                    cache_hint::read_invalidate, cache_hint::cached, SIMD>(
                    a, offsets);
    /// use block load for b
    xetla_vector<dtype, SIMD> ivector2
            = xetla_load_global<dtype, SIMD, data_size::default_size,
                    cache_hint::uncached, cache_hint::uncached>(b, offset);
    //// tf32 convert to fp32 -> vadd -> fp32 convert to tf32 -> write out /////
    xetla_vector<float, SIMD> add1 = xetla_cvt<float, dtype, SIMD>(ivector1);
    xetla_vector<float, SIMD> add2 = xetla_cvt<float, dtype, SIMD>(ivector2);
    xetla_vector<float, SIMD> out = add1 + add2;
    xetla_vector<dtype, SIMD> ovector = xetla_cvt<dtype, float, SIMD>(out);

    /// use scattered write for c
    xetla_store_global<dtype, 1, data_size::default_size, cache_hint::streaming,
            cache_hint::write_back, SIMD>(c, offsets, ovector);
}
