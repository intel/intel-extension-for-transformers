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

#include "block_common.hpp"
#include "kernel_func.hpp"
#include "utils/utils.hpp"

using namespace std::placeholders;

///------------------------------------------------------------------
/// Test global block load store default
/// Tested case:
/// - default block load store.
/// - xetla_load_global API with [dst] [2 src] [datatype int] [SIMD16].
/// - xetla_store_global API with [no return] [3 src] [datatype int] [SIMD16].
///------------------------------------------------------------------

TEST(load_store_block_default, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(load_store_result_validate<int>, _1, _2, _3, 16);
    kernel_run<int, global_load_store_block_default<int, 16>>(
            nd_range, result_validate);
}
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test global block load store default, using xetla_vector_ref instead of xetla_vector
/// Tested case:
/// - default block load store.
/// - xetla_load_global API with [dst] [2 src] [datatype int] [SIMD16].
/// - xetla_store_global API with [no return] [3 src] [datatype int] [SIMD16].
///------------------------------------------------------------------

TEST(load_store_block_default_ref, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(load_store_result_validate<int>, _1, _2, _3, 16);
    kernel_run<int, global_load_store_block_default_ref<int, 16>>(
            nd_range, result_validate);
}
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test global load/ store datatype + datasize test
/// Tested case:
/// - different datatype for xetla_load_global + xetla_store_global block load/store.
/// - xetla_load_global API with [dst] [2 src] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
/// - xetla_store_global API with [no return] [3 src] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
///------------------------------------------------------------------
template <typename T>
class load_store_block_datatype_test : public ::testing::Test {};

TYPED_TEST_SUITE_P(load_store_block_datatype_test);

TYPED_TEST_P(load_store_block_datatype_test, esimd) {
    using datatype = TypeParam;

    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(load_store_result_validate<datatype>, _1, _2, _3, 16);
    kernel_run<datatype, global_load_store_block_default<datatype, 16>>(
            nd_range, result_validate);
}

REGISTER_TYPED_TEST_SUITE_P(load_store_block_datatype_test, esimd);

// Test double, float
// TODO test bf16
using datatypes = ::testing::Types<double, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(
        global_load_store_block, load_store_block_datatype_test, datatypes);
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test global block load cache hint
/// Tested case:
/// - different cache hint for xetla_load_global block store.
/// - xetla_load_global API with [dst] [2 src] [datatype int] [SIMD16] [data_size::default_size] [different L1 L2 cache hint].
///------------------------------------------------------------------
template <typename T>
class load_block_cache_test : public ::testing::Test {};

TYPED_TEST_SUITE_P(load_block_cache_test);

TYPED_TEST_P(load_block_cache_test, esimd) {
    constexpr cache_hint L1H = std::tuple_element_t<0, TypeParam>::value;
    constexpr cache_hint L2H = std::tuple_element_t<1, TypeParam>::value;

    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(load_store_result_validate<int>, _1, _2, _3, 16);
    kernel_run<int, global_load_block_cache<int, 16, L1H, L2H>>(
            nd_range, result_validate);
}

REGISTER_TYPED_TEST_SUITE_P(load_block_cache_test, esimd);

/// @brief Valid Combination of L1 L2 cache for LOAD
/// Bspec link: https://gfxspecs.intel.com/Predator/Home/Index/53560
///
using valid_load_cache_hints
        = ::testing::Types<std::tuple<None, None>, std::tuple<UC, UC>,
                std::tuple<UC, CA>, std::tuple<CA, UC>, std::tuple<CA, CA>,
                std::tuple<ST, UC>, std::tuple<ST, CA>, std::tuple<IAR, CA>>;
INSTANTIATE_TYPED_TEST_SUITE_P(
        global_load_store_block, load_block_cache_test, valid_load_cache_hints);
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test global block store cache hint
/// Tested case:
/// - different cache hint for xetla_store_global block store.
/// - xetla_store_global API with [no return] [3 src] [datatype int] [SIMD16] [data_size::default_size] [different L1 L2 cache hint].
///------------------------------------------------------------------
template <typename T>
class store_block_cache_test : public ::testing::Test {};

TYPED_TEST_SUITE_P(store_block_cache_test);

TYPED_TEST_P(store_block_cache_test, esimd) {
    constexpr cache_hint L1H = std::tuple_element_t<0, TypeParam>::value;
    constexpr cache_hint L2H = std::tuple_element_t<1, TypeParam>::value;

    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(load_store_result_validate<int>, _1, _2, _3, 16);
    kernel_run<int, global_store_block_cache<int, 16, L1H, L2H>>(
            nd_range, result_validate);
}

REGISTER_TYPED_TEST_SUITE_P(store_block_cache_test, esimd);

/// @brief Valid Combination of L1 L2 cache for STORE
/// Bspec link: https://gfxspecs.intel.com/Predator/Home/Index/53561
///
using valid_store_cache_hints
        = ::testing::Types<std::tuple<None, None>, std::tuple<UC, UC>,
                std::tuple<UC, WB>, std::tuple<WT, UC>, std::tuple<WT, WB>,
                std::tuple<ST, UC>, std::tuple<ST, WB>, std::tuple<WB, WB>>;
INSTANTIATE_TYPED_TEST_SUITE_P(global_load_store_block, store_block_cache_test,
        valid_store_cache_hints);
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test global prefetch cache hint
/// Tested case:
/// - different cache hint for xetla_prefetch_global.
/// - xetla_prefetch_global API with [dst] [2 src] [datatype int] [SIMD16] [data_size::default_size] [different L1 L2 cache hint].
///------------------------------------------------------------------
template <typename T>
class prefetch_block_cache_test : public ::testing::Test {};

TYPED_TEST_SUITE_P(prefetch_block_cache_test);

TYPED_TEST_P(prefetch_block_cache_test, esimd) {
    constexpr cache_hint L1H = std::tuple_element_t<0, TypeParam>::value;
    constexpr cache_hint L2H = std::tuple_element_t<1, TypeParam>::value;

    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(load_store_result_validate<int>, _1, _2, _3, 16);
    kernel_run<int, global_prefetch_block<int, 16, L1H, L2H>>(
            nd_range, result_validate);
}

REGISTER_TYPED_TEST_SUITE_P(prefetch_block_cache_test, esimd);

/// @brief Valid Combination of L1 L2 cache for LOAD (PREFETCH)
/// Bspec link: https://gfxspecs.intel.com/Predator/Home/Index/53560
///
using valid_prefetch_cache_hints
        = ::testing::Types<std::tuple<UC, CA>, std::tuple<CA, UC>,
                std::tuple<CA, CA>, std::tuple<ST, UC>, std::tuple<ST, CA>>;
INSTANTIATE_TYPED_TEST_SUITE_P(global_load_store_block,
        prefetch_block_cache_test, valid_prefetch_cache_hints);
///------------------------------------------------------------------

///------------------------------------------------------------------
/// Test global prefetch datatype + datasize test
/// Tested case:
/// - different datatype for xetla_prefetch_global xetla_load_global + xetla_store_global block load/store.
/// - xetla_prefetch_global API with [dst] [2 src] [different datatype] [SIMD16] [data_size::default_size] [L1 L2 cache hint CA,CA].
/// - xetla_load_global API with [dst] [2 src] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
/// - xetla_store_global API with [no return] [3 src] [different datatype] [SIMD16] [data_size::default_size] [default L1 L2 cache hint].
///------------------------------------------------------------------
template <typename T>
class prefetch_block_datatype_test : public ::testing::Test {};

TYPED_TEST_SUITE_P(prefetch_block_datatype_test);

TYPED_TEST_P(prefetch_block_datatype_test, esimd) {
    using datatype = TypeParam;

    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(load_store_result_validate<datatype>, _1, _2, _3, 16);
    kernel_run<datatype,
            global_prefetch_block<datatype, 16, cache_hint::cached,
                    cache_hint::cached>>(nd_range, result_validate);
}

REGISTER_TYPED_TEST_SUITE_P(prefetch_block_datatype_test, esimd);

// Test double, float
// TODO test bf16
using datatypes = ::testing::Types<double, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(
        global_load_store_block, prefetch_block_datatype_test, datatypes);
///------------------------------------------------------------------
