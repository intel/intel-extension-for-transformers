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

template <int SIMD>
struct xetla_abs_vector_version_with_different_input_and_output_types {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        uint64_t offset = sizeof(int) * SIMD * ei->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;

        xetla_vector<int, SIMD> A_load_vec = xetla_load_global(a, offsets);
        A_load_vec -= 8;
        xetla_vector<int, SIMD> result = xetla_vector<int, SIMD>(
                xetla_abs<float, int, SIMD>(A_load_vec));
        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct xetla_abs_vector_version_with_same_input_and_output_types {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        uint64_t offset = sizeof(int) * SIMD * ei->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;

        xetla_vector<int, SIMD> A_load_vec = xetla_load_global(a, offsets);
        A_load_vec -= 8;
        xetla_vector<int, SIMD> result = xetla_abs<int, SIMD>(A_load_vec);
        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct xetla_abs_scalar_version_with_different_input_and_output_types {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        c[0] = static_cast<int>(xetla_abs<float, int>(a[0] - 8));
    }
};

template <int SIMD>
struct xetla_abs_scalar_version_with_same_input_and_output_types {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        c[0] = xetla_abs<int>(a[0] - 8);
    }
};

template <int SIMD>
struct xetla_max_with_vector_Src0_and_vector_Src1 {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        uint64_t offset = sizeof(int) * SIMD * ei->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;

        xetla_vector<int, SIMD> A_load_vec = xetla_load_global(a, offsets);
        xetla_vector<int, SIMD> tmp(8);
        xetla_vector<int, SIMD> result = xetla_max<int, SIMD>(A_load_vec, tmp);
        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct xetla_max_with_vector_Src0_and_scalar_Src1 {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        uint64_t offset = sizeof(int) * SIMD * ei->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;

        xetla_vector<int, SIMD> A_load_vec = xetla_load_global(a, offsets);
        xetla_vector<int, SIMD> result = xetla_max<int, SIMD>(A_load_vec, 8);
        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct xetla_max_with_scalar_Src0_and_vector_Src1 {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        uint64_t offset = sizeof(int) * SIMD * ei->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;

        xetla_vector<int, SIMD> A_load_vec = xetla_load_global(a, offsets);
        xetla_vector<int, SIMD> result = xetla_max<int, SIMD>(8, A_load_vec);
        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct xetla_max_with_scalar_Src0_and_scalar_Src1 {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        c[0] = xetla_max<int>(8, a[0]);
    }
};

template <int SIMD>
struct xetla_min_with_vector_Src0_and_vector_Src1 {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        uint64_t offset = sizeof(int) * SIMD * ei->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;

        xetla_vector<int, SIMD> A_load_vec = xetla_load_global(a, offsets);
        xetla_vector<int, SIMD> tmp(8);
        xetla_vector<int, SIMD> result = xetla_min<int, SIMD>(A_load_vec, tmp);
        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct xetla_min_with_vector_Src0_and_scalar_Src1 {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        uint64_t offset = sizeof(int) * SIMD * ei->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;

        xetla_vector<int, SIMD> A_load_vec = xetla_load_global(a, offsets);
        xetla_vector<int, SIMD> result = xetla_min<int, SIMD>(A_load_vec, 8);
        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct xetla_min_with_scalar_Src0_and_vector_Src1 {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        uint64_t offset = sizeof(int) * SIMD * ei->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;

        xetla_vector<int, SIMD> A_load_vec = xetla_load_global(a, offsets);
        xetla_vector<int, SIMD> result = xetla_min<int, SIMD>(8, A_load_vec);
        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct xetla_min_with_scalar_Src0_and_scalar_Src1 {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, int *a, int *b, int *c) {
        c[0] = xetla_min<int>(8, a[0]);
    }
};
