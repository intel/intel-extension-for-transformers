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

TEST(test_linear_layout_1, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(kernel_validation<float>, _1, _2, _3, 32, 32);
    kernel_run<float, conversion_func<float, 32, 32, 16, 16>>(
            nd_range, result_validate);
}

TEST(test_linear_layout_2, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(kernel_validation<float>, _1, _2, _3, 32, 30);
    kernel_run<float, conversion_func<float, 32, 30, 16, 16>>(
            nd_range, result_validate);
}

TEST(test_linear_layout_3, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(kernel_validation<float>, _1, _2, _3, 64, 64);
    kernel_run<float, conversion_func<float, 64, 64, 16, 16>>(
            nd_range, result_validate);
}

TEST(test_linear_layout_4, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(kernel_validation<float>, _1, _2, _3, 64, 60);
    kernel_run<float, conversion_func<float, 64, 60, 16, 16>>(
            nd_range, result_validate);
}