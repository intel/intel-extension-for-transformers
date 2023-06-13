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

template <class Test, typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc>
class result_validate {
public:
    int operator()(dtype_a *A, dtype_b *B, dtype_c *C, sycl::queue &queue,
            sycl::context &context) {
        return gemm_result_validate<dtype_a, dtype_b, dtype_c, dtype_acc>(A, B,
                C, 1, Test::mat_m, Test::mat_k, Test::mat_n, queue, context,
                Test::layout_a, Test::layout_b);
    }
};

template <class Test, typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc>
using gemm_func = gemm_test_func<dtype_a, dtype_b, dtype_c, dtype_acc,
        Test::wg_m, Test::wg_n, Test::sg_m, Test::sg_n, Test::sg_k,
        Test::layout_a, Test::layout_b, Test::l3_kslicing, Test::slm_kslicing>;

using namespace cl::sycl;

std::string esimd_compile_string
        = " -vc-codegen -doubleGRF "
          " -vc-disable-indvars-opt "
          " -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' ";

template <typename T>
class gemm_tf32 : public ::testing::Test {};
TYPED_TEST_SUITE_P(gemm_tf32);
TYPED_TEST_P(gemm_tf32, esimd) {
    gemm_exec<TypeParam, typename TypeParam::data_type_a,
            typename TypeParam::data_type_b, typename TypeParam::data_type_c,
            float, result_validate, gemm_func>(TypeParam::mat_m,
            TypeParam::mat_n, TypeParam::mat_k, esimd_compile_string);
}
REGISTER_TYPED_TEST_SUITE_P(gemm_tf32, esimd);
INSTANTIATE_TYPED_TEST_SUITE_P(gemm_tf32_suite, gemm_tf32, tests);
