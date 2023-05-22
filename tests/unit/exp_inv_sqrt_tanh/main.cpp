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

#include <cmath>
#include "common.hpp"
#include "kernel_func.hpp"
#include "utils/utils.hpp"
using namespace std::placeholders;

TEST(test_exp_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, exp_op<float>>, _1, _2, _3, 16);
    kernel_run<float, exp_func<float, 16>>(Range, result_validate);
}

TEST(test_exp_fp16, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate
            = std::bind(kernel_validation<fp16, exp_op<fp16>>, _1, _2, _3, 16);
    kernel_run<float, exp_func<fp16, 16>>(Range, result_validate);
}

////for exp2

TEST(test_exp2_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, exp2_op<float>>, _1, _2, _3, 16);
    kernel_run<float, exp2_func<float, 16>>(Range, result_validate);
}

TEST(test_exp2_fp16, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate
            = std::bind(kernel_validation<fp16, exp2_op<fp16>>, _1, _2, _3, 16);
    kernel_run<float, exp2_func<fp16, 16>>(Range, result_validate);
}

////for inv

TEST(test_inv_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, inv_op<float>>, _1, _2, _3, 16);
    kernel_run<float, inv_func<float, 16>>(Range, result_validate);
}

TEST(test_inv_fp16, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate
            = std::bind(kernel_validation<fp16, inv_op<fp16>>, _1, _2, _3, 16);
    kernel_run<float, inv_func<fp16, 16>>(Range, result_validate);
}

////for sqrt

TEST(test_sqrt_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, sqrt_op<float>>, _1, _2, _3, 16);
    kernel_run<float, sqrt_func<float, 16>>(Range, result_validate);
}

TEST(test_sqrt_fp16, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate
            = std::bind(kernel_validation<fp16, sqrt_op<fp16>>, _1, _2, _3, 16);
    kernel_run<float, sqrt_func<fp16, 16>>(Range, result_validate);
}

////for sqrt_ieee

TEST(test_sqrt_ieee_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, sqrt_ieee_op<float>>, _1, _2, _3, 16);
    kernel_run<float, sqrt_ieee_func<float, 16>>(Range, result_validate);
}

////for rsqrt

TEST(test_rsqrt_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, rsqrt_op<float>>, _1, _2, _3, 16);
    kernel_run<float, rsqrt_func<float, 16>>(Range, result_validate);
}

TEST(test_rsqrt_fp16, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<fp16, rsqrt_op<fp16>>, _1, _2, _3, 16);
    kernel_run<float, rsqrt_func<fp16, 16>>(Range, result_validate);
}

////for tanh

TEST(test_tanh_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, tanh_op<float>>, _1, _2, _3, 16);
    kernel_run<float, tanh_func<float, 16>>(Range, result_validate);
}

////for tanh

TEST(test_tanh_fp32_long_vector, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    constexpr int elem_size = 128;
    auto result_validate = std::bind(
            kernel_validation<float, tanh_op<float>>, _1, _2, _3, elem_size);
    kernel_run<float, tanh_func_long_vector<float, elem_size>>(
            Range, result_validate);
}
