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
struct imul_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        xetla_vector<dtype, SIMD> src0 = xetla_load_global<dtype, SIMD>(a, 0);
        xetla_vector<dtype, SIMD> lo;
        xetla_vector<dtype, SIMD> hi = xetla_imul<dtype, dtype, dtype, SIMD>(
                lo.xetla_format<dtype>(), src0, -1);
        xetla_store_global<dtype, SIMD>(c, 0, lo);
        xetla_store_global<dtype, SIMD>(b, 0, hi);
    }
};
