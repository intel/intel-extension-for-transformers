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
struct exp_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {
        //hardcode for float to avoid load restriction
        xetla_vector<float, SIMD> src0 = xetla_load_global<float, SIMD>(a, 0);
        xetla_vector<dtype, SIMD> src = xetla_cvt<dtype, float, SIMD>(src0) + 5;
        xetla_vector<dtype, SIMD> dst0 = xetla_exp<dtype, SIMD>(src);
        xetla_vector<dtype, 1> dst1;
        dst1[0] = xetla_exp<dtype>(src[0]);
        xetla_store_global<float, SIMD>(
                c, 0, xetla_cvt<float, dtype, SIMD>(dst0));
        xetla_store_global<float, 1>(b, 0, xetla_cvt<float, dtype, 1>(dst1));
    }
};

template <typename dtype, int SIMD>
struct exp2_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {
        //hardcode for float to avoid load restriction
        xetla_vector<float, SIMD> src0 = xetla_load_global<float, SIMD>(a, 0);
        xetla_vector<dtype, SIMD> src = xetla_cvt<dtype, float, SIMD>(src0) + 5;
        xetla_vector<dtype, SIMD> dst0 = xetla_exp2<dtype, SIMD>(src);
        xetla_vector<dtype, 1> dst1;
        dst1[0] = xetla_exp2<dtype>(src[0]);
        xetla_store_global<float, SIMD>(
                c, 0, xetla_cvt<float, dtype, SIMD>(dst0));
        xetla_store_global<float, 1>(b, 0, xetla_cvt<float, dtype, 1>(dst1));
    }
};

template <typename dtype, int SIMD>
struct inv_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {
        //hardcode for float to avoid load restriction
        xetla_vector<float, SIMD> src0 = xetla_load_global<float, SIMD>(a, 0);
        xetla_vector<dtype, SIMD> src = xetla_cvt<dtype, float, SIMD>(src0) + 5;
        xetla_vector<dtype, SIMD> dst0 = xetla_inv<dtype, SIMD>(src);
        xetla_vector<dtype, 1> dst1;
        dst1[0] = xetla_inv<dtype>(src[0]);
        xetla_store_global<float, SIMD>(
                c, 0, xetla_cvt<float, dtype, SIMD>(dst0));
        xetla_store_global<float, 1>(b, 0, xetla_cvt<float, dtype, 1>(dst1));
    }
};

template <typename dtype, int SIMD>
struct sqrt_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {
        //hardcode for float to avoid load restriction
        xetla_vector<float, SIMD> src0 = xetla_load_global<float, SIMD>(a, 0);
        xetla_vector<dtype, SIMD> src = xetla_cvt<dtype, float, SIMD>(src0) + 5;
        xetla_vector<dtype, SIMD> dst0 = xetla_sqrt<dtype, SIMD>(src);
        xetla_vector<dtype, 1> dst1;
        dst1[0] = xetla_sqrt<dtype>(src[0]);
        xetla_store_global<float, SIMD>(
                c, 0, xetla_cvt<float, dtype, SIMD>(dst0));
        xetla_store_global<float, 1>(b, 0, xetla_cvt<float, dtype, 1>(dst1));
    }
};

template <typename dtype, int SIMD>
struct sqrt_ieee_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {
        //hardcode for float to avoid load restriction
        xetla_vector<float, SIMD> src0 = xetla_load_global<float, SIMD>(a, 0);
        xetla_vector<dtype, SIMD> src = xetla_cvt<dtype, float, SIMD>(src0) + 5;
        xetla_vector<dtype, SIMD> dst0 = xetla_sqrt_ieee<dtype, SIMD>(src);
        xetla_vector<dtype, 1> dst1;
        dst1[0] = xetla_sqrt_ieee<dtype>(src[0]);
        xetla_store_global<float, SIMD>(
                c, 0, xetla_cvt<float, dtype, SIMD>(dst0));
        xetla_store_global<float, 1>(b, 0, xetla_cvt<float, dtype, 1>(dst1));
    }
};

template <typename dtype, int SIMD>
struct rsqrt_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {
        //hardcode for float to avoid load restriction
        xetla_vector<float, SIMD> src0 = xetla_load_global<float, SIMD>(a, 0);
        xetla_vector<dtype, SIMD> src = xetla_cvt<dtype, float, SIMD>(src0) + 5;
        xetla_vector<dtype, SIMD> dst0 = xetla_rsqrt<dtype, SIMD>(src);
        xetla_vector<dtype, 1> dst1;
        dst1[0] = xetla_rsqrt<dtype>(src[0]);
        xetla_store_global<float, SIMD>(
                c, 0, xetla_cvt<float, dtype, SIMD>(dst0));
        xetla_store_global<float, 1>(b, 0, xetla_cvt<float, dtype, 1>(dst1));
    }
};

template <typename dtype, int SIMD>
struct tanh_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {
        //hardcode for float to avoid load restriction
        xetla_vector<float, SIMD> src0 = xetla_load_global<float, SIMD>(a, 0);
        xetla_vector<dtype, SIMD> src = xetla_cvt<dtype, float, SIMD>(src0) - 5;
        xetla_vector<dtype, SIMD> dst0 = xetla_tanh<dtype, SIMD>(src);
        xetla_vector<dtype, 1> dst1;
        dst1[0] = xetla_tanh<dtype>(src[0]);
        xetla_store_global<float, SIMD>(
                c, 0, xetla_cvt<float, dtype, SIMD>(dst0));
        xetla_store_global<float, 1>(b, 0, xetla_cvt<float, dtype, 1>(dst1));
    }
};

template <typename dtype, int SIMD>
struct tanh_func_long_vector {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {
        xetla_vector<dtype, SIMD> src0;
        src0.copy_from(a);

        xetla_vector<dtype, SIMD> src = xetla_cvt<dtype, float, SIMD>(src0) - 5;
        xetla_vector<dtype, 1> dst1;
        dst1[0] = xetla_tanh<dtype>(src[0]);
        dst1.copy_to(b);

        xetla_vector<dtype, SIMD> dst0 = xetla_tanh<dtype, SIMD>(src);
        dst0.copy_to(c);
    }
};
