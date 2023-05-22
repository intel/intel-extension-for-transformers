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

#include <iostream>
#include <random>
#include <stdlib.h>
#include <string>
#include "buff_compare.hpp"
#include "common/common.hpp"
#include "gemm_gen.hpp"
#include <CL/sycl.hpp>

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;

#define get_str_tmp(x) #x
#define get_str(x) get_str_tmp(x)

#define random_float() (generate_random<double>())

template <typename data_type>
inline auto getTypeName() {
    fprintf(stderr, "FAIL: Not implemented specialization\n");
    exit(-1);
}

template <>
inline auto getTypeName<int>() {
    return "int";
}
template <>
inline auto getTypeName<float>() {
    return "float";
}
template <>
inline auto getTypeName<uint32_t>() {
    return "uint32_t";
}
template <>
inline auto getTypeName<uint64_t>() {
    return "uint64_t";
}
template <>
inline auto getTypeName<double>() {
    return "double";
}
template <>
inline auto getTypeName<int8_t>() {
    return "int8_t";
}
template <>
inline auto getTypeName<uint8_t>() {
    return "uint8_t";
}

template <>
inline auto getTypeName<gpu::xetla::bf16>() {
    return "bf16";
}

template <>
inline auto getTypeName<gpu::xetla::fp16>() {
    return "fp16";
}

template <>
inline auto getTypeName<gpu::xetla::tf32>() {
    return "tf32";
}

template <typename result_type>
inline result_type generate_random(result_type a = 0.0, result_type b = 1.0) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<result_type> distribution(a, b);

    return distribution(engine);
}

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float>
int gemm_result_validate(data_type_a *A, data_type_b *B, data_type_c *C,
        uint32_t batch_size, uint32_t m, uint32_t k, uint32_t n,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    bool is_col_major_a = mem_layout_a_ == mem_layout::col_major;
    bool is_col_major_b = mem_layout_b_ == mem_layout::col_major;
    // define slice of each matrices
    uint32_t size_a_slice = m * k;
    uint32_t size_b_slice = k * n;
    uint32_t size_c_slice = m * n;
    buff_cmp::buff_vals<data_type_c> data(C, batch_size * m, n, n);
    std::vector<data_type_acc> gold_C(batch_size * m * n, 0);
    for (uint32_t batch = 0; batch < batch_size; batch++) {
        get_gemm_gold<data_type_a, data_type_b, data_type_acc>(m, n, k,
                mem_layout_a_, mem_layout_b_, A + size_a_slice * batch,
                B + size_b_slice * batch, gold_C.data() + size_c_slice * batch);
    }
    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), batch_size * m, n, n);

    bool result = buff_cmp::xetla_buff_cmp(data, other, "gemm validation");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}