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

///------------------------------------------------------------------
/// Test local load/ store block datatype + datasize test
/// Tested case:
/// - different datatype for xetla_load_local + xetla_store_local block load/store.
/// - xetla_load_local API with [dst] [1 src] [different datatype]  [NElts1] [SIMD16] [data_size::default_size].
/// - xetla_store_local API with [no return] [2 src] [different datatype] [NElts1] [SIMD16] [data_size::default_size].
///------------------------------------------------------------------
template <typename T>
class load_store_block_datatype_test : public ::testing::Test {};

TYPED_TEST_SUITE_P(load_store_block_datatype_test);

TYPED_TEST_P(load_store_block_datatype_test, esimd) {
    using datatype = TypeParam;

    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            local_load_store_result_validate<datatype>, _1, _2, _3, 16);
    kernel_run<datatype, local_load_store_block<datatype, 16>>(
            nd_range, result_validate);
}

REGISTER_TYPED_TEST_SUITE_P(load_store_block_datatype_test, esimd);

// Test double, float, int
// TODO test bf16
using datatypes = ::testing::Types<int, double, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(
        local_load_store_block, load_store_block_datatype_test, datatypes);
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test local load/ store scatter datatype + datasize test
/// Tested case:
/// - different datatype for xetla_load_local + xetla_store_local block load/store.
/// - xetla_load_local API with [dst] [1 src] [different datatype]  [NElts1] [SIMD16] [data_size::default_size].
/// - xetla_store_local API with [no return] [2 src] [different datatype] [NElts1] [SIMD16] [data_size::default_size].
///------------------------------------------------------------------
template <typename T>
class load_store_scatter_datatype_test : public ::testing::Test {};

TYPED_TEST_SUITE_P(load_store_scatter_datatype_test);

TYPED_TEST_P(load_store_scatter_datatype_test, esimd) {
    using datatype = TypeParam;

    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            local_load_store_result_validate<datatype>, _1, _2, _3, 16);
    kernel_run<datatype, local_load_store_scatter<datatype, 16>>(
            nd_range, result_validate);
}

REGISTER_TYPED_TEST_SUITE_P(load_store_scatter_datatype_test, esimd);

// Test double, float, int
// TODO test bf16
using datatypes = ::testing::Types<int, double, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(
        local_load_store_scatter, load_store_scatter_datatype_test, datatypes);
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test local load scatter with mask
/// Tested case:
/// - different datatype for xetla_load_local + xetla_store_local scatter load/store.
/// - xetla_load_local API with [dst] [1 src] [different datatype]  [NElts1] [SIMD16] [4 channel enabled] [data_size::default_size].
/// - xetla_store_local API with [no return] [2 src] [different datatype] [NElts1] [SIMD16] [all channel enabled] [data_size::default_size].
///------------------------------------------------------------------

TEST(local_load_scatter_mask, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(mask_result_validate<int>, _1, _2, _3, 16, 0xF, 0);
    kernel_run<int, local_load_scatter_mask<int, 16>>(
            nd_range, result_validate);
}
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test local store scatter with mask
/// Tested case:
/// - different datatype for xetla_load_local + xetla_store_local scatter load/store.
/// - xetla_load_local API with [dst] [1 src] [different datatype]  [NElts1] [SIMD16] [all channel enabled] [data_size::default_size].
/// - xetla_store_local API with [no return] [2 src] [different datatype] [NElts1] [SIMD16] [4 channel enabled] [data_size::default_size].
///------------------------------------------------------------------

TEST(local_store_scatter_mask, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(mask_result_validate<int>, _1, _2, _3, 16, 0xF, 16);
    kernel_run<int, local_store_scatter_mask<int, 16>>(
            nd_range, result_validate);
}
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test local store scatter NElts 2
/// Tested case:
/// - different datatype for xetla_load_local + xetla_store_local scatter load/store.
/// - xetla_load_local API with [dst] [1 src] [different datatype]  [NElts2] [SIMD16] [all channel enabled] [data_size::default_size].
/// - xetla_store_local API with [no return] [2 src] [different datatype] [NElts2] [SIMD16] [4 channel enabled] [data_size::default_size].
///------------------------------------------------------------------

TEST(local_store_scatter_nelts2, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(local_load_store_result_validate<int>, _1, _2, _3, 32);
    kernel_run<int, local_load_store_scatter_nelt2<int, 16>>(
            nd_range, result_validate);
}
///------------------------------------------------------------------
