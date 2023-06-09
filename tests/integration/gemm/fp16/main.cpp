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
#include "utils/utils.hpp"
#include <gtest/gtest.h>

class TestBase {
public:
    static std::string name(size_t mat_m, size_t mat_n, size_t mat_k,
            size_t wg_m, size_t wg_n, size_t sg_m, size_t sg_n, size_t sg_k,
            mem_layout layout_a, mem_layout layout_b) {
        std::string mem_layout_a_str
                = layout_a == mem_layout::col_major ? "col_major" : "row_major";
        std::string mem_layout_b_str
                = layout_b == mem_layout::col_major ? "col_major" : "row_major";
        std::string name = std::string("fp16_gemm_") + std::to_string(mat_m)
                + "x" + std::to_string(mat_n) + "x" + std::to_string(mat_k)
                + "_" + std::to_string(wg_m) + "x" + std::to_string(wg_n) + "_"
                + std::to_string(sg_m) + "x" + std::to_string(sg_n) + "_"
                + mem_layout_a_str + "_" + mem_layout_b_str;
        return name;
    }
};

class Test0 : public TestBase {
public:
    static constexpr size_t mat_m = 256;
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_k = 256;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = fp16;
    using data_type_acc = float;
};

class Test1 : public TestBase {
public:
    static constexpr size_t mat_m = 256;
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_k = 256;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::col_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = fp16;
    using data_type_acc = float;
};
class Test2 : public TestBase {
public:
    static constexpr size_t mat_m = 256;
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_k = 256;
    static constexpr size_t wg_m = 16;
    static constexpr size_t wg_n = 32;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};
class Test3 : public TestBase {
public:
    static constexpr size_t mat_m = 192;
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_k = 256;
    static constexpr size_t wg_m = 192;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 24;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::col_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};
class Test4 : public TestBase {
public:
    static constexpr size_t mat_m = 256;
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_k = 256;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::col_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};
class Test5 : public TestBase {
public:
    static constexpr size_t mat_m = 256;
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_k = 256;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::col_major;
    static constexpr mem_layout layout_b = mem_layout::col_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};
class Test6 : public TestBase {
public:
    static constexpr size_t mat_m = 96;
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_k = 256;
    static constexpr size_t wg_m = 96;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 24;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};
class Test7 : public TestBase {
public:
    static constexpr size_t mat_m = 80;
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_k = 256;
    static constexpr size_t wg_m = 96;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Test8 : public TestBase {
public:
    static constexpr size_t mat_m = 256;
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_k = 256;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 2;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Test9 : public TestBase {
public:
    static constexpr size_t mat_m = 256;
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_k = 256;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 4;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Test10 : public TestBase {
public:
    static constexpr size_t mat_m = 4;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 1024;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Test11 : public TestBase {
public:
    static constexpr size_t mat_m = 4;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 16384;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Test12 : public TestBase {
public:
    static constexpr size_t mat_m = 4;
    static constexpr size_t mat_n = 16384;
    static constexpr size_t mat_k = 1024;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Test13 : public TestBase {
public:
    static constexpr size_t mat_m = 128;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 128;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Test14 : public TestBase {
public:
    static constexpr size_t mat_m = 4;
    static constexpr size_t mat_n = 50400;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Test15 : public TestBase {
public:
    static constexpr size_t mat_m = 128;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 16384;
    static constexpr size_t wg_m = 128;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Test16 : public TestBase {
public:
    static constexpr size_t mat_m = 128;
    static constexpr size_t mat_n = 50400;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 128;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = float;
    using data_type_acc = float;
};

template <typename dtype_a, typename dtype_b, typename dtype_c>
class input_buffer_init {
public:
    void operator()(dtype_a *A, dtype_b *B, dtype_c *C, size_t size_a,
            size_t size_b, size_t size_c) {
        for (unsigned i = 0; i < size_a; ++i) {
            A[i] = (i * 3) % 17;
        }
        for (unsigned i = 0; i < size_b; ++i) {
            B[i] = (i * 5) % 19;
        }
        for (unsigned i = 0; i < size_c; ++i) {
            C[i] = 0;
        }
    }
};

template <class Test, typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc>
class result_validate {

public:
    int operator()(dtype_a *A, dtype_b *B, dtype_c *C, sycl::queue queue,
            sycl::context context) {
        return gemm_result_validate<dtype_a, dtype_b, dtype_c, dtype_acc>(A, B,
                C, 1, Test::mat_m, Test::mat_k, Test::mat_n, queue, context,
                Test::layout_a, Test::layout_b);
    }
};

template <class Test, typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc>
using fp16_gemm_func = fp16_gemm_test_func<dtype_a, dtype_b, dtype_c, dtype_acc,
        Test::wg_m, Test::wg_n, Test::sg_m, Test::sg_n, Test::sg_k,
        Test::layout_a, Test::layout_b, Test::l3_kslicing, Test::slm_kslicing>;

using namespace cl::sycl;

std::string esimd_compile_string
        = " -vc-codegen -doubleGRF "
          " -vc-disable-indvars-opt "
          " -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' ";

template <typename T>
class fp16_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(fp16_gemm_test);
TYPED_TEST_P(fp16_gemm_test, esimd) {
    gemm_exec<TypeParam, typename TypeParam::data_type_a,
            typename TypeParam::data_type_b, typename TypeParam::data_type_c,
            typename TypeParam::data_type_acc, input_buffer_init,
            result_validate, fp16_gemm_func>(TypeParam::mat_m, TypeParam::mat_n,
            TypeParam::mat_k, esimd_compile_string);
}
REGISTER_TYPED_TEST_SUITE_P(fp16_gemm_test, esimd);
using tests = ::testing::Types<Test0, Test1, Test2, Test3, Test4, Test5, Test6,
        Test7, Test8, Test9, Test10, Test11, Test12, Test13, Test14, Test15,
        Test16>;
INSTANTIATE_TYPED_TEST_SUITE_P(fp16_gemm_test_suite, fp16_gemm_test, tests);
