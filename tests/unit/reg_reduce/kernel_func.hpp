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

template <typename dtype, int SIMD, reduce_op BinaryOperation>
struct reduce_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        xetla_vector<dtype, SIMD> src0 = xetla_load_global<dtype, SIMD>(a, 0);
        src0 += 5;

        xetla_vector<dtype, 1> dst
                = xetla_reduce<dtype, dtype, SIMD, BinaryOperation>(src0);
        xetla_store_global<dtype, 1>(c, 0, dst);
    }
};