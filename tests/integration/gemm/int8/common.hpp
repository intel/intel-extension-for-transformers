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

#pragma once

#include "kernel_func.hpp"
#include "utils/buff_compare.hpp"
#include "utils/common.hpp"
#include "xetla.hpp"
#include <gtest/gtest.h>

using namespace gpu;
using namespace gpu::xetla;

class TestBase {
public:
    static std::string name(size_t mat_m, size_t mat_n, size_t mat_k,
            size_t wg_m, size_t wg_n, size_t sg_m, size_t sg_n, size_t sg_k,
            mem_layout layout_a, mem_layout layout_b) {
        std::string mem_layout_a_str
                = layout_a == mem_layout::col_major ? "col_major" : "row_major";
        std::string mem_layout_b_str
                = layout_b == mem_layout::col_major ? "col_major" : "row_major";
        std::string name = std::string("int8_gemm_") + std::to_string(mat_m)
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
    static constexpr size_t sg_k = 64;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = int8_t;
    using data_type_acc = int;
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
    static constexpr size_t sg_k = 64;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::col_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = int8_t;
    using data_type_acc = int;
};

class Test2 : public TestBase {
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
    static constexpr mem_layout layout_a = mem_layout::col_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = int8_t;
    using data_type_acc = int;
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
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = int;
    using data_type_acc = int;
};

class Test4 : public TestBase {
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
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = int;
    using data_type_acc = int;
};
class Test5 : public TestBase {
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
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = int;
    using data_type_acc = int;
};

class Test6 : public TestBase {
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
    using data_type_a = int8_t;
    using data_type_b = int8_t;
    using data_type_c = int;
    using data_type_acc = int;
};

template <class Test, typename data_type_a, typename data_type_b,
        typename data_type_c, typename data_type_acc>
using int8_gemm_func = int8gemm_test_func<data_type_a, data_type_b, data_type_c,
        data_type_acc, Test::wg_m, Test::wg_n, Test::sg_m, Test::sg_n,
        Test::sg_k, Test::layout_a, Test::layout_b, Test::l3_kslicing,
        Test::slm_kslicing>;

template <class Test, typename data_type_a, typename data_type_b,
        typename data_type_c, typename data_type_acc>
class result_validate {

public:
    int operator()(data_type_a *A_device, data_type_b *B_device,
            data_type_c *C_device, sycl::queue &queue, sycl::context &context) {
        auto A = alloc_host_and_copy<data_type_a>(
                A_device, Test::mat_m * Test::mat_k, queue);
        auto B = alloc_host_and_copy<data_type_b>(
                B_device, Test::mat_k * Test::mat_n, queue);
        auto C = alloc_host_and_copy<data_type_c>(
                C_device, Test::mat_m * Test::mat_n, queue);

        buff_cmp::buff_vals<data_type_c> data(
                C, Test::mat_m, Test::mat_n, Test::mat_n);
        std::vector<data_type_acc> acc_buffer(Test::mat_m * Test::mat_n, 0);

        {
            bool is_col_major_a = Test::layout_a == mem_layout::col_major;
            bool is_col_major_b = Test::layout_b == mem_layout::col_major;
            for (int i = 0; i < Test::mat_m; i++) {
                for (int j = 0; j < Test::mat_n; j++) {
                    for (int k = 0; k < Test::mat_k; k++) {
                        data_type_acc a_temp = is_col_major_a
                                ? A[i + k * Test::mat_m]
                                : A[i * Test::mat_k + k];
                        data_type_acc b_temp = is_col_major_b
                                ? B[k + j * Test::mat_k]
                                : B[k * Test::mat_n + j];
                        acc_buffer[i * Test::mat_n + j]
                                = acc_buffer[i * Test::mat_n + j]
                                + a_temp * b_temp;
                    }
                }
            }
        }

        buff_cmp::buff_vals<data_type_c, data_type_acc> other(
                acc_buffer.data(), Test::mat_m, Test::mat_n, Test::mat_n);
        bool result = buff_cmp::xetla_buff_cmp(data, other,
                Test::name(Test::mat_m, Test::mat_n, Test::mat_k, Test::wg_m,
                        Test::wg_n, Test::sg_m, Test::sg_n, Test::sg_k,
                        Test::layout_a, Test::layout_b));

        free(A);
        free(B);
        free(C);

        return result ? 0 : 1;
    }
};
