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
#include <gtest/gtest.h>

std::string esimd_compile_string
        = " -vc-codegen -doubleGRF  -Xfinalizer ' -printregusage -noLocalSplit "
          "-enableBCR -nolocalra  ' ";

template <typename T>
class fp32_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(fp32_gemm_test);

TYPED_TEST_P(fp32_gemm_test, esimd) {
    gemm_exec<TypeParam, result_validate<TypeParam>, fp32_gemm_func<TypeParam>>(
            esimd_compile_string, TypeParam::batch_size);
}

REGISTER_TYPED_TEST_SUITE_P(fp32_gemm_test, esimd);
using tests = ::testing::Types<Test1, Test2, Test3, Test4, Test5, Test6, Test7,
        Test8, Test9, Test10, Test11>;
INSTANTIATE_TYPED_TEST_SUITE_P(fp32_gemm_test_suite, fp32_gemm_test, tests);
