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

TEST(tile_load_store_vnni_local_func, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_load_store_result_validate<bf16>, _1, _2, _3, 128, 32, 32);
    kernel_run<bf16,
            tile_load_store_vnni_local_func<bf16, 128, 64, 128, 32, 32, 16,
                    16>>(nd_range, result_validate);
}

TEST(tile_load_store_local, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_load_store_result_validate<int>, _1, _2, _3, 128, 32, 32);
    kernel_run<int,
            tile_load_store_local_func<int, 128, 64, 128, 32, 32, 16, 16>>(
            nd_range, result_validate);
}

TEST(tile_load_store_local_with_update, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_load_store_result_validate<int>, _1, _2, _3, 128, 32, 32);
    kernel_run<int,
            tile_load_store_local_func<int, 128, 64, 128, 32, 32, 16, 16, 5>>(
            nd_range, result_validate);
}

TEST(tile_transpose_store_local, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_load_store_result_validate<int>, _1, _2, _3, 64, 32, 32);
    kernel_run<int,
            tile_transpose_store_local_func<int, 64, 64, 64, 32, 32, 16, 8>>(
            nd_range, result_validate);
}

TEST(tile_transpose_store_local_bf16, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_load_store_result_validate<bf16>, _1, _2, _3, 64, 32, 32);
    kernel_run<bf16,
            tile_transpose_store_local_func<bf16, 64, 64, 64, 32, 32, 16, 16>>(
            nd_range, result_validate);
}

TEST(tile_load_store_1d_local, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_load_store_result_validate<int>, _1, _2, _3, 128, 32, 32);
    kernel_run<int,
            tile_load_store_1d_local_func<int, 128, 64, 128, 32, 32, 16, 16>>(
            nd_range, result_validate);
}

TEST(tile_load_store_1row_local, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_load_store_result_validate<int>, _1, _2, _3, 128, 32, 1);
    kernel_run<int,
            tile_load_store_1d_local_func<int, 128, 64, 128, 32, 1, 16, 1>>(
            nd_range, result_validate);
}

TEST(tile_load_store_1d_local_with_update, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_load_store_result_validate<int>, _1, _2, _3, 128, 32, 32);
    kernel_run<int,
            tile_load_store_1d_local_func<int, 128, 64, 128, 32, 32, 16, 16,
                    5>>(nd_range, result_validate);
}

TEST(tile_load_store_1row_local_with_update, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_load_store_result_validate<int>, _1, _2, _3, 128, 32, 1);
    kernel_run<int,
            tile_load_store_1d_local_func<int, 128, 64, 128, 32, 1, 16, 1, 5>>(
            nd_range, result_validate);
}
