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

int add_update_carry_result_validate(
        uint32_t *A, uint32_t *B, uint32_t *C, unsigned Size);

template <typename dtype, int SIMD>
struct add_update_carry_func {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {
        xetla_vector<dtype, SIMD> src0 = xetla_load_global<dtype, SIMD>(a, 0);
        xetla_vector<dtype, SIMD> carry;
        xetla_vector<dtype, SIMD> dst = xetla_add_c<dtype, SIMD>(
                src0, -1, carry.xetla_format<dtype>());
        xetla_store_global<dtype, SIMD>(c, 0, dst);
        xetla_store_global<dtype, SIMD>(b, 0, carry);
    }
};
