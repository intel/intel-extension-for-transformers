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

template <typename dtype, int SIMD>
struct local_load_store_block {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {

        uint64_t offset = 0;
        xetla_vector<dtype, SIMD> A_load_vec
                = xetla_load_global<dtype, SIMD>(a, offset);

        xetla_store_local<dtype, SIMD>(offset, A_load_vec);

        xetla_vector<dtype, SIMD> slm_vec
                = xetla_load_local<dtype, SIMD>(offset);

        xetla_store_global<dtype, SIMD>(b, offset, slm_vec);
    }
};

template <typename dtype, int SIMD>
struct local_load_store_scatter {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {
        uint64_t offset = 0;
        xetla_vector<dtype, SIMD> A_load_vec
                = xetla_load_global<dtype, SIMD>(a, offset);

        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets = offsets * sizeof(dtype);

        xetla_store_local<dtype, 1, data_size::default_size, SIMD>(
                offsets, A_load_vec);
        xetla_vector<dtype, SIMD> slm_vec
                = xetla_load_local<dtype, 1, data_size::default_size, SIMD>(
                        offsets);

        xetla_store_global<dtype, SIMD>(b, offset, slm_vec);
    }
};

template <typename dtype, int SIMD>
struct local_load_scatter_mask {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {

        uint64_t offset = 0;
        xetla_vector<dtype, SIMD> A_load_vec
                = xetla_load_global<dtype, SIMD>(a, offset);

        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets = offsets * sizeof(dtype);

        xetla_mask<SIMD> pred(0);
        pred.xetla_select<4, 1>(0) = 1;

        xetla_store_local<dtype, 1, data_size::default_size, SIMD>(
                offsets, A_load_vec);
        xetla_vector<dtype, SIMD> slm_vec(0);
        xetla_vector<dtype, SIMD> tmp
                = xetla_load_local<dtype, 1, data_size::default_size, SIMD>(
                        offsets, pred);
        slm_vec.xetla_merge(tmp, pred);

        xetla_store_global<dtype, SIMD>(b, offset, slm_vec);
    }
};

template <typename dtype, int SIMD>
struct local_store_scatter_mask {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {

        uint64_t offset = 0;
        xetla_vector<dtype, SIMD> A_load_vec
                = xetla_load_global<dtype, SIMD>(a, offset);

        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets = offsets * sizeof(dtype);

        // set default value in local memory
        xetla_vector<dtype, SIMD> default_B(SIMD);
        // xetla_store_local<dtype, 1, data_size::default_size, SIMD>(0, default_B);
        xetla_store_local<dtype, SIMD>(0, default_B);

        xetla_mask<SIMD> pred(0);
        pred.xetla_select<4, 1>(0) = 1;

        xetla_store_local<dtype, 1, data_size::default_size, SIMD>(
                offsets, A_load_vec, pred);
        xetla_vector<dtype, SIMD> slm_vec
                = xetla_load_local<dtype, 1, data_size::default_size, SIMD>(
                        offsets);

        xetla_store_global<dtype, SIMD>(b, offset, slm_vec);
    }
};

template <typename dtype, int SIMD>
struct local_load_store_scatter_nelt2 {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {

        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets = offsets * sizeof(dtype);

        xetla_vector<dtype, SIMD * 2> A_load_vec
                = xetla_load_global<dtype, 2, data_size::default_size,
                        cache_hint::none, cache_hint::none, SIMD>(a, offsets);

        xetla_store_local<dtype, 2, data_size::default_size, SIMD>(
                offsets, A_load_vec);
        xetla_vector<dtype, SIMD * 2> slm_vec
                = xetla_load_local<dtype, 2, data_size::default_size, SIMD>(
                        offsets);

        xetla_store_global<dtype, 2, data_size::default_size, cache_hint::none,
                cache_hint::none, SIMD>(b, offsets, slm_vec);
    }
};
