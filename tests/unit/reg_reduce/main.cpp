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

TEST(test_reduce_add_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, reduce_sum<float>>, _1, _2, _3, 64);
    kernel_run<float, reduce_func<float, 64, reduce_op::sum>>(
            Range, result_validate);
}

TEST(test_reduce_mul_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, reduce_prod<float>>, _1, _2, _3, 64);
    kernel_run<float, reduce_func<float, 64, reduce_op::prod>>(
            Range, result_validate);
}

TEST(test_reduce_min_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, reduce_min<float>>, _1, _2, _3, 64);
    kernel_run<float, reduce_func<float, 64, reduce_op::min>>(
            Range, result_validate);
}

TEST(test_reduce_max_fp32, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            kernel_validation<float, reduce_max<float>>, _1, _2, _3, 16);
    kernel_run<float, reduce_func<float, 16, reduce_op::max>>(
            Range, result_validate);
}
