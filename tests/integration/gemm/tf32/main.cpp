/******************************************************************************
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

#include "../tests/utils/utils.hpp"
#include "kernel_func.hpp"
#include "test.hpp"
#include <gtest/gtest.h>

template <class Test>
class result_validate {
public:
    using dtype_a = Test::data_type_a;
    using dtype_b = Test::data_type_b;
    using dtype_c = Test::data_type_c;
    using dtype_acc = float;
    int operator()(dtype_a *A, dtype_b *B, dtype_c *C, sycl::queue &queue) {
        return gemm_result_validate<dtype_a, dtype_b, dtype_c, dtype_acc>(A, B,
                C, 1, Test::mat_m, Test::mat_k, Test::mat_n, queue,
                Test::layout_a, Test::layout_b);
    }
};

template <class Test>
using tf32_gemm_func = tf32_gemm_test_func<typename Test::data_type_a,
        typename Test::data_type_b, typename Test::data_type_c, float,
        Test::wg_m, Test::wg_n, Test::sg_m, Test::sg_n, Test::sg_k,
        Test::layout_a, Test::layout_b, Test::global_kslicing,
        Test::local_kslicing>;

using namespace cl::sycl;

std::string esimd_compile_string
        = " -vc-codegen -doubleGRF "
          " -vc-disable-indvars-opt "
          " -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' ";

template <typename T>
class gemm_tf32 : public ::testing::Test {};
TYPED_TEST_SUITE_P(gemm_tf32);
TYPED_TEST_P(gemm_tf32, esimd) {
    gemm_exec<TypeParam, result_validate<TypeParam>, tf32_gemm_func<TypeParam>>(
            esimd_compile_string);
}
REGISTER_TYPED_TEST_SUITE_P(gemm_tf32, esimd);
INSTANTIATE_TYPED_TEST_SUITE_P(gemm_tf32_suite, gemm_tf32, tests);
