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

#include "common.hpp"
#include "kernel_func.hpp"
#include "utils/utils.hpp"

using namespace std::placeholders;

/// @brief global atomic iinc base
/// Tested case:
/// - atomic iinc op.
/// - xetla_atomic_global API with [0 src] [no return] [all channel enabled].
TEST(global_atomic_iinc_base, esimd) {
    kernel_run<int, global_atomic_iinc_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<int>, _1, _2, _3, 16,
                    atomic_op::iinc));
}

/// @brief global atomic iinc mask
/// Tested case:
/// - atomic iinc op.
/// - xetla_atomic_global API with [0 src] [no return] [4 channel enabled].
TEST(global_atomic_iinc_mask, esimd) {
    kernel_run<int, global_atomic_iinc_mask<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_with_mask_result_validate<int>, _1, _2, _3,
                    16, atomic_op::iinc, 0xF));
}

/// @brief global atomic iinc return
/// Tested case:
/// - atomic iinc op.
/// - xetla_atomic_global API with [0 src] [with return] [all channel enabled].
TEST(global_atomic_iinc_return, esimd) {
    kernel_run<int, global_atomic_iinc_return<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_with_ret_result_validate<int>, _1, _2, _3,
                    16, atomic_op::iinc));
}

/// @brief global atomic idec base
/// Tested case:
/// - atomic idec op.
/// - xetla_atomic_global API with [0 src] [no return] [all channel enabled].
TEST(global_atomic_idec_base, esimd) {
    kernel_run<int, global_atomic_idec_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<int>, _1, _2, _3, 16,
                    atomic_op::idec));
}

/// @brief global atomic iadd base
/// Tested case:
/// - atomic iadd op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_iadd_base, esimd) {
    kernel_run<int, global_atomic_iadd_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<int>, _1, _2, _3, 16,
                    atomic_op::iadd));
}

/// @brief global atomic iadd mask
/// Tested case:
/// - atomic iadd op.
/// - xetla_atomic_global API with [1 src] [no return] [4 channel enabled].
TEST(global_atomic_iadd_mask, esimd) {
    kernel_run<int, global_atomic_iadd_mask<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_with_mask_result_validate<int>, _1, _2, _3,
                    16, atomic_op::iadd, 0xF));
}

/// @brief global atomic iadd return
/// Tested case:
/// - atomic iadd op.
/// - xetla_atomic_global API with [1 src] [with return] [all channel enabled].
TEST(global_atomic_iadd_return, esimd) {
    kernel_run<int, global_atomic_iadd_return<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_with_ret_result_validate<int>, _1, _2, _3,
                    16, atomic_op::iadd));
}

/// @brief global atomic isub base
/// Tested case:
/// - atomic isub op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_isub_base, esimd) {
    kernel_run<int, global_atomic_isub_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<int>, _1, _2, _3, 16,
                    atomic_op::isub));
}

/// @brief global atomic smin base
/// Tested case:
/// - atomic smin op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_smin_base, esimd) {
    kernel_run<int, global_atomic_smin_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<int>, _1, _2, _3, 16,
                    atomic_op::smin));
}

/// @brief global atomic smax base
/// Tested case:
/// - atomic smax op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_smax_base, esimd) {
    kernel_run<int, global_atomic_smax_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<int>, _1, _2, _3, 16,
                    atomic_op::smax));
}

/// @brief global atomic fadd base
/// Tested case:
/// - atomic fadd op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_fadd_base, esimd) {
    kernel_run<float, global_atomic_fadd_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<float>, _1, _2, _3, 16,
                    atomic_op::fadd));
}

/// @brief global atomic fsub base
/// Tested case:
/// - atomic fsub op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_fsub_base, esimd) {
    kernel_run<float, global_atomic_fsub_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<float>, _1, _2, _3, 16,
                    atomic_op::fsub));
}

/// @brief global atomic fmin base
/// Tested case:
/// - atomic fmin op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_fmin_base, esimd) {
    kernel_run<float, global_atomic_fmin_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<float>, _1, _2, _3, 16,
                    atomic_op::fmin));
}

/// @brief global atomic fmax base
/// Tested case:
/// - atomic fmax op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_fmax_base, esimd) {
    kernel_run<float, global_atomic_fmax_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<float>, _1, _2, _3, 16,
                    atomic_op::fmax));
}
/// @brief global atomic umin base
/// Tested case:
/// - atomic umin op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_umin_base, esimd) {
    kernel_run<uint32_t, global_atomic_umin_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<uint32_t>, _1, _2, _3, 16,
                    atomic_op::umin));
}

/// @brief global atomic umax base
/// Tested case:
/// - atomic umax op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_umax_base, esimd) {
    kernel_run<uint32_t, global_atomic_umax_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<uint32_t>, _1, _2, _3, 16,
                    atomic_op::umax));
}

/// @brief global atomic bit_and base
/// Tested case:
/// - atomic bit_and op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_bit_and_base, esimd) {
    kernel_run<uint32_t, global_atomic_bit_and_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_bit_op_result_validate, _1, _2, _3, 16,
                    atomic_op::bit_and));
}

/// @brief global atomic bit_or base
/// Tested case:
/// - atomic bit_or op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_bit_or_base, esimd) {
    kernel_run<uint32_t, global_atomic_bit_or_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_bit_op_result_validate, _1, _2, _3, 16,
                    atomic_op::bit_or));
}

/// @brief global atomic bit_xor base
/// Tested case:
/// - atomic bit_xor op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_bit_xor_base, esimd) {
    kernel_run<uint32_t, global_atomic_bit_xor_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_bit_op_result_validate, _1, _2, _3, 16,
                    atomic_op::bit_xor));
}

/// @brief global atomic load
/// Tested case:
/// - atomic load op.
/// - xetla_atomic_global API with [0 src] [with return] [all channel enabled].
TEST(global_atomic_load, esimd) {
    kernel_run<uint32_t, global_atomic_load<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<uint32_t>, _1, _2, _3, 16,
                    atomic_op::load));
}
/// @brief global atomic store
/// Tested case:
/// - atomic store op.
/// - xetla_atomic_global API with [1 src] [no return] [all channel enabled].
TEST(global_atomic_store, esimd) {
    kernel_run<uint32_t, global_atomic_store<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<uint32_t>, _1, _2, _3, 16,
                    atomic_op::store));
}
/// @brief global atomic cmpxchg base
/// Tested case:
/// - atomic cmpxchg op.
/// - xetla_atomic_global API with [2 src] [no return] [all channel enabled].
TEST(global_atomic_cmpxchg_base, esimd) {
    kernel_run<uint32_t, global_atomic_cmpxchg_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<uint32_t>, _1, _2, _3, 16,
                    atomic_op::cmpxchg));
}

/// @brief global atomic cmpxchg mask
/// Tested case:
/// - atomic cmpxchg op.
/// - xetla_atomic_global API with [2 src] [no return] [4 channel enabled].
TEST(global_atomic_cmpxchg_mask, esimd) {
    kernel_run<uint32_t, global_atomic_cmpxchg_mask<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_with_mask_result_validate<uint32_t>, _1, _2,
                    _3, 16, atomic_op::cmpxchg, 0xF));
}

/// @brief global atomic cmpxchg return
/// Tested case:
/// - atomic cmpxchg op.
/// - xetla_atomic_global API with [2 src] [with return] [all channel enabled].
TEST(global_atomic_cmpxchg_return, esimd) {
    kernel_run<uint32_t, global_atomic_cmpxchg_return<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_with_ret_result_validate<uint32_t>, _1, _2,
                    _3, 16, atomic_op::cmpxchg));
}
/// @brief global atomic fcmpxchg base
/// Tested case:
/// - atomic fcmpxchg op.
/// - xetla_atomic_global API with [2 src] [no return] [all channel enabled].
TEST(global_atomic_fcmpxchg_base, esimd) {
    kernel_run<float, global_atomic_fcmpxchg_base<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(global_atomic_result_validate<float>, _1, _2, _3, 16,
                    atomic_op::fcmpxchg));
}
