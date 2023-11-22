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

/* following 9 cases all about padding load/store:
    0  0  0 .... 0  0  0
    0  _____________   0
    .  |            |  .
    .  |   data     |  .
    .  |            |  .
    0  |____________|  0
    0  0  0 .... 0  0  0

*/

// top-left
// 0  0  0
// 0  x  x
// 0  x  x
// TEST(tile_padding_load_store_1, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_padding_load_store_result_validate<int>, _1, _2,
//                     _3, 16, 16, -1, -1);
//     kernel_run<int,
//             tile_padding_load_store_func<int, 16, 16, 16, 16, 16, 16, 16, -1,
//                     -1>>(nd_range, result_validate);
// }

// // top-mid
// // 0  0  0
// // x  x  x
// // x  x  x
// TEST(tile_padding_load_store_2, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_padding_load_store_result_validate<int>, _1, _2,
//                     _3, 16, 16, 0, -1);
//     kernel_run<int,
//             tile_padding_load_store_func<int, 16, 16, 16, 16, 16, 16, 16, 0,
//                     -1>>(nd_range, result_validate);
// }

// // top-right
// // 0  0  0
// // x  x  0
// // x  x  0
// TEST(tile_padding_load_store_3, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_padding_load_store_result_validate<int>, _1, _2,
//                     _3, 16, 16, 1, -1);
//     kernel_run<int,
//             tile_padding_load_store_func<int, 16, 16, 16, 16, 16, 16, 16, 1,
//                     -1>>(nd_range, result_validate);
// }

// // mid-left
// // 0  x  x
// // 0  x  x
// // 0  x  x
// TEST(tile_padding_load_store_4, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_padding_load_store_result_validate<int>, _1, _2,
//                     _3, 16, 16, -1, 0);
//     kernel_run<int,
//             tile_padding_load_store_func<int, 16, 16, 16, 16, 16, 16, 16, -1,
//                     0>>(nd_range, result_validate);
// }

// // mid-mid
// // x  x  x
// // x  x  x
// // x  x  x
// TEST(tile_padding_load_store_5, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_padding_load_store_result_validate<int>, _1, _2,
//                     _3, 16, 16, 0, 0);
//     kernel_run<int,
//             tile_padding_load_store_func<int, 16, 16, 16, 16, 16, 16, 16, 0,
//                     0>>(nd_range, result_validate);
// }

// // mid-right
// // x  x  0
// // x  x  0
// // x  x  0
// TEST(tile_padding_load_store_6, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_padding_load_store_result_validate<int>, _1, _2,
//                     _3, 16, 16, 1, 0);
//     kernel_run<int,
//             tile_padding_load_store_func<int, 16, 16, 16, 16, 16, 16, 16, 1,
//                     0>>(nd_range, result_validate);
// }

// // bottom-left
// // 0  x  x
// // 0  x  x
// // 0  0  0
// TEST(tile_padding_load_store_7, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_padding_load_store_result_validate<int>, _1, _2,
//                     _3, 16, 16, -1, 1);
//     kernel_run<int,
//             tile_padding_load_store_func<int, 16, 16, 16, 16, 16, 16, 16, -1,
//                     1>>(nd_range, result_validate);
// }

// // bottom-mid
// // x  x  x
// // x  x  x
// // 0  0  0
// TEST(tile_padding_load_store_8, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_padding_load_store_result_validate<int>, _1, _2,
//                     _3, 16, 16, 0, 1);
//     kernel_run<int,
//             tile_padding_load_store_func<int, 16, 16, 16, 16, 16, 16, 16, 0,
//                     1>>(nd_range, result_validate);
// }

// // bottom-right
// // x  x  0
// // x  x  0
// // 0  0  0
// TEST(tile_padding_load_store_9, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_padding_load_store_result_validate<int>, _1, _2,
//                     _3, 16, 16, 1, 1);
//     kernel_run<int,
//             tile_padding_load_store_func<int, 16, 16, 16, 16, 16, 16, 16, 1,
//                     1>>(nd_range, result_validate);
// }

// TEST(tile_load_store, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate = std::bind(tile_load_store_result_validate<int>, _1,
//             _2, _3, 128, 64, 32, 32, 0);
//     kernel_run<int, tile_load_store_func<int, 128, 64, 128, 32, 32, 16, 16>>(
//             nd_range, result_validate);
// }

// TEST(tile_load_transpose_store_1, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_load_store_result_validate<int, false, true>, _1,
//                     _2, _3, 128, 64, 32, 32, 0);
//     kernel_run<int,
//             tile_load_store_func<int, 128, 64, 128, 32, 32, 8, 8, false, true,
//                     64>>(nd_range, result_validate);
// }

// TEST(tile_load_transpose_store_2, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_load_store_result_validate<int, false, true>, _1,
//                     _2, _3, 128, 64, 32, 32, 0);
//     kernel_run<int,
//             tile_load_store_func<int, 128, 64, 128, 32, 32, 8, 16, false, true,
//                     64>>(nd_range, result_validate);
// }

// TEST(tile_load_transform_store, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_load_store_result_validate<bf16, true>, _1, _2, _3,
//                     128, 64, 32, 32, 0);
//     kernel_run<bf16,
//             tile_load_store_func<bf16, 128, 64, 128, 32, 32, 16, 16, true,
//                     false>>(nd_range, result_validate);
// }

// TEST(tile_load_store_atomic, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate = std::bind(tile_load_store_result_validate<float>, _1,
//             _2, _3, 128, 64, 32, 32, 0);
//     kernel_run<float,
//             tile_load_store_atomic_func<float, 128, 64, 128, 32, 32, 16, 16>>(
//             nd_range, result_validate);
// }

// TEST(tile_load_store_atomic_oob, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate = std::bind(tile_load_store_result_validate<float>, _1,
//             _2, _3, 30, 31, 32, 32, 0);
//     kernel_run<float,
//             tile_load_store_atomic_func<float, 30, 31, 30, 32, 32, 16, 16>>(
//             nd_range, result_validate);
// }

// TEST(tile_load_store_atomic_disable_oob_check, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate = std::bind(tile_load_store_result_validate<float>, _1,
//             _2, _3, 128, 64, 32, 32, 0);
//     kernel_run<float,
//             tile_load_store_atomic_func<float, 128, 64, 128, 32, 32, 16, 16,
//                     false, false>>(nd_range, result_validate);
// }

// TEST(tile_load_store_atomic_boundary, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate = std::bind(tile_load_store_result_validate<float>, _1,
//             _2, _3, 128, 33554440, 32, 32, 33554432);
//     kernel_run<float,
//             tile_load_store_atomic_func<float, 128, 33554440, 128, 32, 32, 16,
//                     16, true>,
//             128 * 1024, 32, 4294968320U>(nd_range, result_validate);
// }

// TEST(tile_load_broadcast_store, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_load_broadcase_store_result_validate<int>, _1, _2,
//                     _3, 128, 32, 32);
//     kernel_run<int,
//             tile_load_broadcast_store_func<int, 128, 64, 128, 32, 32, 16, 16>>(
//             nd_range, result_validate);
// }

// TEST(tile_load_store_1d, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate = std::bind(tile_load_store_result_validate<int>, _1,
//             _2, _3, 128, 64, 127, 1, 0);
//     kernel_run<int, tile_load_store_1d_func<int, 128, 64, 128, 127, 1, 127, 1>>(
//             nd_range, result_validate);
// }

// TEST(tile_load_store_1d_boundary, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate = std::bind(tile_load_store_result_validate<int>, _1,
//             _2, _3, 128, 33554440, 128, 1, 33554432);
//     kernel_run<int,
//             tile_load_store_1d_func<int, 128, 33554440, 128, 128, 1, 128, 1,
//                     true>,
//             128 * 1024, 32, 4294968320U>(nd_range, result_validate);
// }

TEST(tile_load_store_unaligned_2d, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    using date_type = fp16;
    auto result_validate = std::bind(tile_load_store_result_validate<date_type>,
            _1, _2, _3, 127, 63, 32, 32, 0);
    kernel_run<date_type,
            tile_load_store_unaligned_2d_func<date_type, 16, 16, 16, 16, 16, 16,
                    16>>(nd_range, result_validate);
}

// TEST(tile_load_store_oob_1, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_load_store_result_validate<int, false, true>, _1,
//                     _2, _3, 64, 64, 32 - 1, 32 - 1, 0);
//     kernel_run<int,
//             tile_load_store_oob_func<int, 64, 64, 64, 32, 32, 16, 16, -1, -1>>(
//             nd_range, result_validate);
// }

// TEST(tile_load_store_oob_2, esimd) {
//     cl::sycl::nd_range<1> nd_range({1}, {1});
//     auto result_validate
//             = std::bind(tile_load_store_result_validate<bf16, false, true>, _1,
//                     _2, _3, 64, 64, 32 - 2, 32 - 2, 0);
//     kernel_run<bf16,
//             tile_load_store_oob_func<bf16, 64, 64, 64, 32, 32, 16, 16, -2, -2>>(
//             nd_range, result_validate);
// }
