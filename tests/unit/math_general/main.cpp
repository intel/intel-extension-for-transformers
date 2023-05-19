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

/// @brief xetla_abs vector version with different input and output types
/// Tested case:
/// - xetla_abs op.
/// - xetla_abs API with [1 src] [1 return] [all channel enabled].

TEST(test_abs_vector_version_with_different_input_and_output_types, esimd) {
    kernel_run<int,
            xetla_abs_vector_version_with_different_input_and_output_types<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::abs_vector));
}

/// @brief xetla_abs vector version with same input and output types
/// Tested case:
/// - xetla_abs op.
/// - xetla_abs API with [1 src] [1 return] [all channel enabled].

TEST(test_abs_vector_version_with_same_input_and_output_types, esimd) {
    kernel_run<int,
            xetla_abs_vector_version_with_same_input_and_output_types<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::abs_vector));
}

/// @brief xetla_abs scalar version with different input and output types
/// Tested case:
/// - xetla_abs op.
/// - xetla_abs API with [1 src] [1 return] [all channel enabled].

TEST(test_abs_scalar_version_with_different_input_and_output_types, esimd) {
    kernel_run<int,
            xetla_abs_scalar_version_with_different_input_and_output_types<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::abs_scalar));
}

/// @brief xetla_abs scalar version with same input and output types
/// Tested case:
/// - xetla_abs op.
/// - xetla_abs API with [1 src] [1 return] [all channel enabled].

TEST(test_abs_scalar_version_with_same_input_and_output_types, esimd) {
    kernel_run<int,
            xetla_abs_scalar_version_with_same_input_and_output_types<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::abs_scalar));
}

/// @brief xetla_max with vector Src0 and vector Src1
/// Tested case:
/// - xetla_max op.
/// - xetla_max API with [2 src] [1 return] [all channel enabled].

TEST(test_max_with_vector_Src0_and_vector_Src1, esimd) {
    kernel_run<int, xetla_max_with_vector_Src0_and_vector_Src1<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::max_vector));
}

/// @brief xetla_max with vector Src0 and scalar Src1
/// Tested case:
/// - xetla_max op.
/// - xetla_max API with [2 src] [1 return] [all channel enabled].

TEST(test_max_with_vector_Src0_and_scalar_Src1, esimd) {
    kernel_run<int, xetla_max_with_vector_Src0_and_scalar_Src1<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::max_vector));
}

/// @brief xetla_max with scalar Src0 and vector Src1
/// Tested case:
/// - xetla_max op.
/// - xetla_max API with [2 src] [1 return] [all channel enabled].

TEST(test_max_with_scalar_Src0_and_vector_Src1, esimd) {
    kernel_run<int, xetla_max_with_scalar_Src0_and_vector_Src1<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::max_vector));
}

/// @brief xetla_max with scalar Src0 and scalar Src1
/// Tested case:
/// - xetla_max op.
/// - xetla_max API with [2 src] [1 return] [all channel enabled].

TEST(test_max_with_scalar_Src0_and_scalar_Src1, esimd) {
    kernel_run<int, xetla_max_with_scalar_Src0_and_scalar_Src1<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::max_scalar));
}

/// @brief xetla_min with vector Src0 and vector Src1
/// Tested case:
/// - xetla_min op.
/// - xetla_min API with [2 src] [1 return] [all channel enabled].

TEST(test_min_with_vector_Src0_and_vector_Src1, esimd) {
    kernel_run<int, xetla_min_with_vector_Src0_and_vector_Src1<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::min_vector));
}

/// @brief xetla_min with vector Src0 and scalar Src1
/// Tested case:
/// - xetla_min op.
/// - xetla_min API with [2 src] [1 return] [all channel enabled].

TEST(test_min_with_vector_Src0_and_scalar_Src1, esimd) {
    kernel_run<int, xetla_min_with_vector_Src0_and_scalar_Src1<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::min_vector));
}

/// @brief xetla_min with scalar Src0 and vector Src1
/// Tested case:
/// - xetla_min op.
/// - xetla_min API with [2 src] [1 return] [all channel enabled].

TEST(test_min_with_scalar_Src0_and_vector_Src1, esimd) {
    kernel_run<int, xetla_min_with_scalar_Src0_and_vector_Src1<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::min_vector));
}

/// @brief xetla_min with scalar Src0 and scalar Src1
/// Tested case:
/// - xetla_min op.
/// - xetla_min API with [2 src] [1 return] [all channel enabled].

TEST(test_min_with_scalar_Src0_and_scalar_Src1, esimd) {
    kernel_run<int, xetla_min_with_scalar_Src0_and_scalar_Src1<16>>(
            cl::sycl::nd_range<1>({1}, {1}),
            std::bind(math_result_validate<int>, _1, _2, _3, 16,
                    math_op::min_scalar));
}
