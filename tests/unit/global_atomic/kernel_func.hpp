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
struct global_atomic_iinc_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, int *a, int *b, int *c) {

        uint64_t offset = sizeof(int) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;
        xetla_mask<SIMD> pred(1);

        xetla_atomic_global<atomic_op::iinc, int, SIMD, data_size::default_size,
                cache_hint::uncached, cache_hint::write_back>(a, offsets, pred);
    }
};

template <int SIMD>
struct global_atomic_iinc_mask {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, int *a, int *b, int *c) {

        uint64_t offset = sizeof(int) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;
        xetla_mask<SIMD> pred(0);
        pred.xetla_select<4, 1>(0) = 1;

        xetla_atomic_global<atomic_op::iinc, int, SIMD, data_size::default_size,
                cache_hint::uncached, cache_hint::write_back>(a, offsets, pred);
    }
};

template <int SIMD>
struct global_atomic_iinc_return {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, int *a, int *b, int *c) {

        uint64_t offset = sizeof(int) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<int, SIMD> result = xetla_atomic_global<atomic_op::iinc,
                int, SIMD, data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, pred);

        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct global_atomic_idec_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, int *a, int *b, int *c) {

        uint64_t offset = sizeof(int) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;
        xetla_mask<SIMD> pred(1);

        xetla_atomic_global<atomic_op::idec, int, SIMD, data_size::default_size,
                cache_hint::uncached, cache_hint::write_back>(a, offsets, pred);
    }
};

template <int SIMD>
struct global_atomic_iadd_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, int *a, int *b, int *c) {

        uint64_t offset = sizeof(int) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<int, SIMD> adder(3);
        xetla_atomic_global<atomic_op::iadd, int, SIMD, data_size::default_size,
                cache_hint::uncached, cache_hint::write_back>(
                a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_iadd_mask {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, int *a, int *b, int *c) {

        uint64_t offset = sizeof(int) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;
        xetla_mask<SIMD> pred(0);
        pred.xetla_select<4, 1>(0) = 1;
        xetla_vector<int, SIMD> adder(3);
        xetla_atomic_global<atomic_op::iadd, int, SIMD, data_size::default_size,
                cache_hint::uncached, cache_hint::write_back>(
                a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_iadd_return {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, int *a, int *b, int *c) {

        uint64_t offset = sizeof(int) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<int, SIMD> adder(3);
        xetla_vector<int, SIMD> result = xetla_atomic_global<atomic_op::iadd,
                int, SIMD, data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, adder, pred);

        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct global_atomic_isub_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, int *a, int *b, int *c) {

        uint64_t offset = sizeof(int) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<int, SIMD> adder(3);
        xetla_atomic_global<atomic_op::isub, int, SIMD, data_size::default_size,
                cache_hint::uncached, cache_hint::write_back>(
                a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_smin_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, int *a, int *b, int *c) {

        uint64_t offset = sizeof(int) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<int, SIMD> adder(3);
        xetla_atomic_global<atomic_op::smin, int, SIMD, data_size::default_size,
                cache_hint::uncached, cache_hint::write_back>(
                a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_smax_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, int *a, int *b, int *c) {

        uint64_t offset = sizeof(int) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(int);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<int, SIMD> adder(3);
        xetla_atomic_global<atomic_op::smax, int, SIMD, data_size::default_size,
                cache_hint::uncached, cache_hint::write_back>(
                a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_fadd_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {

        uint64_t offset = sizeof(float) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(float);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<float, SIMD> adder(3);
        xetla_atomic_global<atomic_op::fadd, float, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_fsub_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {

        uint64_t offset = sizeof(float) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(float);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<float, SIMD> adder(3);
        xetla_atomic_global<atomic_op::fsub, float, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_fmin_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {

        uint64_t offset = sizeof(float) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(float);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<float, SIMD> adder(3);
        xetla_atomic_global<atomic_op::fmin, float, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_fmax_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {

        uint64_t offset = sizeof(float) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(float);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<float, SIMD> adder(3);
        xetla_atomic_global<atomic_op::fmax, float, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_umin_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, uint32_t *a, uint32_t *b, uint32_t *c) {

        uint64_t offset = sizeof(uint32_t) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(uint32_t);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<uint32_t, SIMD> adder(3);
        xetla_atomic_global<atomic_op::umin, uint32_t, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_umax_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, uint32_t *a, uint32_t *b, uint32_t *c) {

        uint64_t offset = sizeof(uint32_t) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(uint32_t);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<uint32_t, SIMD> adder(3);
        xetla_atomic_global<atomic_op::umax, uint32_t, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_bit_and_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, uint32_t *a, uint32_t *b, uint32_t *c) {

        uint64_t offset = sizeof(uint32_t) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(uint32_t);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<uint32_t, SIMD> adder(3);
        xetla_atomic_global<atomic_op::bit_and, uint32_t, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_bit_or_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, uint32_t *a, uint32_t *b, uint32_t *c) {

        uint64_t offset = sizeof(uint32_t) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(uint32_t);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<uint32_t, SIMD> adder(3);
        xetla_atomic_global<atomic_op::bit_or, uint32_t, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_bit_xor_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, uint32_t *a, uint32_t *b, uint32_t *c) {

        uint64_t offset = sizeof(uint32_t) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(uint32_t);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<uint32_t, SIMD> adder(3);
        xetla_atomic_global<atomic_op::bit_xor, uint32_t, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, adder, pred);
    }
};

template <int SIMD>
struct global_atomic_load {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, uint32_t *a, uint32_t *b, uint32_t *c) {

        uint64_t offset = sizeof(uint32_t) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(uint32_t);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<uint32_t, SIMD> result
                = xetla_atomic_global<atomic_op::load, uint32_t, SIMD,
                        data_size::default_size, cache_hint::uncached,
                        cache_hint::write_back>(a, offsets, pred);

        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct global_atomic_store {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, uint32_t *a, uint32_t *b, uint32_t *c) {

        uint64_t offset = sizeof(uint32_t) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(uint32_t);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<uint32_t, SIMD> data(3);
        xetla_atomic_global<atomic_op::store, uint32_t, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, data, pred);
    }
};

template <int SIMD>
struct global_atomic_cmpxchg_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, uint32_t *a, uint32_t *b, uint32_t *c) {

        uint64_t offset = sizeof(uint32_t) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(uint32_t);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<uint32_t, SIMD> old_data = xetla_load_global(a, offsets);
        xetla_vector<uint32_t, SIMD> new_data(3);
        xetla_atomic_global<atomic_op::cmpxchg, uint32_t, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, old_data, new_data, pred);
    }
};

template <int SIMD>
struct global_atomic_cmpxchg_mask {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, uint32_t *a, uint32_t *b, uint32_t *c) {

        uint64_t offset = sizeof(uint32_t) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(uint32_t);
        offsets += offset;
        xetla_mask<SIMD> pred(0);
        pred.xetla_select<4, 1>(0) = 1;
        xetla_vector<uint32_t, SIMD> old_data = xetla_load_global(a, offsets);
        xetla_vector<uint32_t, SIMD> new_data(3);
        xetla_atomic_global<atomic_op::cmpxchg, uint32_t, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, old_data, new_data, pred);
    }
};

template <int SIMD>
struct global_atomic_cmpxchg_return {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, uint32_t *a, uint32_t *b, uint32_t *c) {

        uint64_t offset = sizeof(uint32_t) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(uint32_t);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<uint32_t, SIMD> old_data = xetla_load_global(a, offsets);
        xetla_vector<uint32_t, SIMD> new_data(3);
        xetla_vector<uint32_t, SIMD> result
                = xetla_atomic_global<atomic_op::cmpxchg, uint32_t, SIMD,
                        data_size::default_size, cache_hint::uncached,
                        cache_hint::write_back>(
                        a, offsets, old_data, new_data, pred);
        xetla_store_global(c, offsets, result);
    }
};

template <int SIMD>
struct global_atomic_fcmpxchg_base {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, float *a, float *b, float *c) {

        uint64_t offset = sizeof(float) * SIMD * item->get_group(0);
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(float);
        offsets += offset;
        xetla_mask<SIMD> pred(1);
        xetla_vector<float, SIMD> old_data = xetla_load_global(a, offsets);
        xetla_vector<float, SIMD> new_data(3);
        xetla_atomic_global<atomic_op::fcmpxchg, float, SIMD,
                data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>(a, offsets, old_data, new_data, pred);
    }
};
