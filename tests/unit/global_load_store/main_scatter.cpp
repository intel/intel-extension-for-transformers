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

#include "kernel_func.hpp"
#include "scatter_common.hpp"
#include "utils/utils.hpp"

using namespace std::placeholders;

///------------------------------------------------------------------
/// Test global load datatype + datasize test with mask
/// Tested case:
/// - different datatype for xetla_load_global + xetla_store_global block scatter.
/// - xetla_load_global API with [dst] [2 src] [4 channel enabled] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
/// - xetla_store_global API with [no return] [3 src] [all channel enabled] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
///------------------------------------------------------------------
template <typename T>
class load_scatter_datatype_test : public ::testing::Test {};

TYPED_TEST_SUITE_P(load_scatter_datatype_test);

TYPED_TEST_P(load_scatter_datatype_test, esimd) {
    using datatype = TypeParam;

    cl::sycl::nd_range<1> nd_range({1}, {1});
    // For load mask kernel, the masked value would be 0 in buffer A and write to bufferB
    auto result_validate
            = std::bind(mask_result_validate<datatype>, _1, _2, _3, 16, 0xF, 0);
    kernel_run<datatype, global_load_scatter_mask<datatype, 16>>(
            nd_range, result_validate);
}

REGISTER_TYPED_TEST_SUITE_P(load_scatter_datatype_test, esimd);

// Test double, float
// TODO test bf16
using datatypes = ::testing::Types<double, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(
        global_load_store_scatter, load_scatter_datatype_test, datatypes);

///------------------------------------------------------------------
/// Test global prefetch datatype + datasize test with mask
/// Tested case:
/// - different datatype for xetla_prefetch block scatter.
/// - xetla_prefetch_global API with [dst] [2 src] [4 channel enabled] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
/// - xetla_load_global API with [dst] [2 src] [4 channel enabled] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
/// - xetla_store_global API with [no return] [3 src] [all channel enabled] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
///------------------------------------------------------------------
template <typename T>
class prefetch_scatter_datatype_test : public ::testing::Test {};

TYPED_TEST_SUITE_P(prefetch_scatter_datatype_test);

TYPED_TEST_P(prefetch_scatter_datatype_test, esimd) {
    using datatype = TypeParam;

    cl::sycl::nd_range<1> nd_range({1}, {1});
    // For load mask kernel, the masked value would be 0 in buffer A and write to bufferB
    auto result_validate
            = std::bind(mask_result_validate<datatype>, _1, _2, _3, 16, 0xF, 0);
    kernel_run<datatype, global_prefetch_scatter_mask<datatype, 16>>(
            nd_range, result_validate);
}

REGISTER_TYPED_TEST_SUITE_P(prefetch_scatter_datatype_test, esimd);

// Test double, float
// TODO test bf16
using datatypes = ::testing::Types<double, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(
        global_load_store_scatter, prefetch_scatter_datatype_test, datatypes);
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test global store datatype + datasize test with mask
/// Tested case:
/// - different datatype for xetla_load_global + xetla_store_global block scatter.
/// - xetla_load_global API with [dst] [2 src] [all channel enabled] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
/// - xetla_store_global API with [no return] [3 src] [4 channel enabled] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
///------------------------------------------------------------------
template <typename T>
class store_scatter_datatype_test : public ::testing::Test {};

TYPED_TEST_SUITE_P(store_scatter_datatype_test);

TYPED_TEST_P(store_scatter_datatype_test, esimd) {
    using datatype = TypeParam;

    cl::sycl::nd_range<1> nd_range({1}, {1});
    // For store mask kernel, we write buffer B as value of SIMD in advance, so masked channel value should be 16
    auto result_validate = std::bind(
            mask_result_validate<datatype>, _1, _2, _3, 16, 0xF, 16);
    kernel_run<datatype, global_store_scatter_mask<datatype, 16>>(
            nd_range, result_validate);
}

REGISTER_TYPED_TEST_SUITE_P(store_scatter_datatype_test, esimd);

// Test double, float
// TODO test bf16
using datatypes = ::testing::Types<double, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(
        global_load_store_scatter, store_scatter_datatype_test, datatypes);
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test global scatter load store with 2 as NElts,
/// Tested case:
/// - default block load store.
/// - xetla_load_global API with [dst] [2 src] [datatype int] [SIMD8] [NElts2].
/// - xetla_store_global API with [no return] [3 src] [datatype int] [SIMD8] [NElts2].
///------------------------------------------------------------------

TEST(load_store_scatter_nelts2, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(load_store_result_validate<int>, _1, _2, _3, 32);
    kernel_run<int, global_load_store_scatter_nelt2<int, 16>>(
            nd_range, result_validate);
}
///------------------------------------------------------------------
