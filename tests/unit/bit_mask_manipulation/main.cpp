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

/// @brief xetla_shl with vector input
/// Tested case:
/// - shl op.
/// - xetla_shl API with [2 src] [with return] [all channel enabled].
TEST(shl_with_vector_input, esimd) {
    kernel_run<uint64_t, shl_with_vector_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::shl_vector));
}

/// @brief xetla_shl with scalar input
/// Tested case:
/// - shl op.
/// - xetla_shl API with [2 src] [with return] [all channel enabled].
TEST(shl_with_scalar_input, esimd) {
    kernel_run<uint64_t, shl_with_scalar_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::shl_scalar));
}

/// @brief xetla_shr with vector input
/// Tested case:
/// - shr op.
/// - xetla_shr API with [2 src] [with return] [all channel enabled].
TEST(shr_with_vector_input, esimd) {
    kernel_run<uint64_t, shr_with_vector_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::shr_vector));
}

/// @brief xetla_shr with scalar input
/// Tested case:
/// - shr op.
/// - xetla_shr API with [2 src] [with return] [all channel enabled].
TEST(shr_with_scalar_input, esimd) {
    kernel_run<uint64_t, shr_with_scalar_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::shr_scalar));
}

/// @brief xetla_rol with two vector input
/// Tested case:
/// - rol op.
/// - xetla_rol API with [2 src] [with return] [all channel enabled].
TEST(rol_with_2_vector_input, esimd) {
    kernel_run<uint64_t, rol_with_2_vector_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::rol_vector));
}

/// @brief xetla_rol with a vector and a scalar input
/// Tested case:
/// - rol op.
/// - xetla_rol API with [2 src] [with return] [all channel enabled].
TEST(rol_with_a_vector_and_a_scalar_input, esimd) {
    kernel_run<uint64_t, rol_with_a_vector_and_a_scalar_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::rol_vector));
}

/// @brief xetla_rol with a vector and a scalar input
/// Tested case:
/// - rol op.
/// - xetla_rol API with [2 src] [with return] [all channel enabled].
TEST(rol_with_2_scalar_input, esimd) {
    kernel_run<uint64_t, rol_with_2_scalar_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::rol_scalar));
}

/// @brief xetla_ror with two vector input
/// Tested case:
/// - ror op.
/// - xetla_ror API with [2 src] [with return] [all channel enabled].
TEST(ror_with_2_vector_input, esimd) {
    kernel_run<uint64_t, ror_with_2_vector_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::ror_vector));
}

/// @brief xetla_ror with a vector and a scalar input
/// Tested case:
/// - ror op.
/// - xetla_ror API with [2 src] [with return] [all channel enabled].
TEST(ror_with_a_vector_and_a_scalar_input, esimd) {
    kernel_run<uint64_t, ror_with_a_vector_and_a_scalar_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::ror_vector));
}

/// @brief xetla_ror with a vector and a scalar input
/// Tested case:
/// - ror op.
/// - xetla_ror API with [2 src] [with return] [all channel enabled].
TEST(ror_with_2_scalar_input, esimd) {
    kernel_run<uint64_t, ror_with_2_scalar_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::ror_scalar));
}

/// @brief xetla_lsr with vector input
/// Tested case:
/// - lsr op.
/// - xetla_lsr API with [2 src] [with return] [all channel enabled].
TEST(lsr_with_vector_input, esimd) {
    kernel_run<uint64_t, lsr_with_vector_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::lsr_vector));
}

/// @brief xetla_lsr with scalar input
/// Tested case:
/// - lsr op.
/// - xetla_lsr API with [2 src] [with return] [all channel enabled].
TEST(lsr_with_scalar_input, esimd) {
    kernel_run<uint64_t, lsr_with_scalar_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::lsr_scalar));
}

/// @brief xetla_asr with vector input
/// Tested case:
/// - asr op.
/// - xetla_asr API with [2 src] [with return] [all channel enabled].
TEST(asr_with_vector_input, esimd) {
    kernel_run<uint64_t, asr_with_vector_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::asr_vector));
}

/// @brief xetla_asr with scalar input
/// Tested case:
/// - asr op.
/// - xetla_asr API with [2 src] [with return] [all channel enabled].
TEST(asr_with_scalar_input, esimd) {
    kernel_run<uint64_t, asr_with_scalar_input<uint64_t, 16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(bit_shift_result_validate<uint64_t>, _1, _2, _3, 16,
                    bit_shift_op::asr_scalar));
}
