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
struct rand_func {
    static_assert((std::is_same<remove_const_t<dtype>, uint32_t>::value),
            "only test uint32_t");
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {
        xetla_vector<dtype, SIMD> src0 = xetla_load_global<dtype, SIMD>(a, 0);

        xetla_rand_t<SIMD> rand_gen;
        rand_gen.init(src0[3], 0, src0[5]);
        xetla_vector<uint32_t, 4 *SIMD> ret0 = rand_gen.rand();
        xetla_vector<uint32_t, 4 *SIMD> ret1 = rand_gen.rand();

        xetla_store_global<dtype, 4 * SIMD>(b, 0, ret0);
        xetla_store_global<dtype, 4 * SIMD>(c, 0, ret1);
    }
};
