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

TEST(test_named_barrier, esimd) {
    cl::sycl::nd_range<1> Range({16}, {16});
    auto result_validate
            = std::bind(named_barrier_result_validate, _1, _2, _3, 16);
    kernel_run<int, named_barrier_func<int, 16>>(Range, result_validate);
}

///------------------------------------------------------------------
/// Test named test_named_barrier_producer_consumer_1
/// Tested case:
/// - 2 producer and 2 consumer threads
/// - only one named barrier used
/// - tidX=2,3 are producers, reads original data , multiplies by 2 and writes to SLM
/// - tidX=0,1 are consumers, reads multiplied data from SLM and writes to output buffer
///------------------------------------------------------------------

TEST(test_named_barrier_producer_consumer_1, esimd) {
    cl::sycl::nd_range<1> Range({4}, {4});
    auto result_validate
            = std::bind(named_barrier_split_validate, _1, _2, _3, 32, 2);
    kernel_run<int, named_barrier_producer_consumer_1_func<int, 16>>(
            Range, result_validate);
}

///------------------------------------------------------------------
/// Test named test_named_barrier_producer_consumer_2
/// Tested case:
/// - 32 threads in workgroup
/// - 16 producer threads, 16 consumer threads
/// - 16 named barriers used, only 1 producer , 1 consumer per named barrier
/// - tidX=16..31 are producers, reads original data , multiplies by 2 and writes to SLM
/// - tidX=0..15 are consumers, reads multiplied data from SLM and writes to output buffer
///------------------------------------------------------------------

TEST(test_named_barrier_producer_consumer_2, esimd) {
    cl::sycl::nd_range<1> Range({32}, {32});
    auto result_validate
            = std::bind(named_barrier_split_validate, _1, _2, _3, 256, 2);
    kernel_run<int, named_barrier_producer_consumer_2_func<int, 16>>(
            Range, result_validate);
}

///------------------------------------------------------------------
/// Test named test_named_barrier_producer_consumer_3
/// Tested case:
/// - 16 threads in workgroup
/// - 8 producer threads, 8 consumer threads
/// - 16 named barriers used, only 1 producer , 1 consumer per named barrier, each named barrier is used multiple times
/// - tidX=0..7  reads original data , multiplies by 2 writes to SLM, [signal], [wait for th#tidX+8], then multiplies by 2 again, writes to SLM, [signal]
/// - tidX=8..15 [wait for th#tidX+8] reads multiplied data from SLM , [signal], [wait for th#tidX+8] do another read, then add the two vector, and writes to output buffer
///------------------------------------------------------------------

TEST(test_named_barrier_producer_consumer_3, esimd) {
    cl::sycl::nd_range<1> Range({16}, {16});
    auto result_validate
            = std::bind(named_barrier_split_validate, _1, _2, _3, 128, 6);
    kernel_run<int, named_barrier_producer_consumer_3_func<int, 16>>(
            Range, result_validate);
}
