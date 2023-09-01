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

#define DEVICE_MEM_ALIGNMENT (64)

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

enum class test_result { complete = 0, skip = 1, fail = 2 };

template <typename result_type>
inline result_type generate_random(result_type a = 0.0, result_type b = 1.0) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<result_type> distribution(a, b);

    return distribution(engine);
}

template <typename data_type>
inline data_type *alloc_device_and_init(size_t size,
        std::function<void(data_type *data, size_t elements)> init_func,
        sycl::queue &queue, sycl::device &device, sycl::context &context) {
    auto host_ptr = static_cast<data_type *>(malloc(size * sizeof(data_type)));

    for (size_t i = 0; i < size; ++i) {
        init_func(host_ptr, i);
    }

    auto device_ptr = static_cast<data_type *>(aligned_alloc_device(
            DEVICE_MEM_ALIGNMENT, size * sizeof(data_type), device, context));

    queue.memcpy((void *)device_ptr, (void *)host_ptr, size * sizeof(data_type))
            .wait();

    free(host_ptr);

    return device_ptr;
}

template <typename data_type>
inline data_type *alloc_host_and_copy(
        data_type *device_ptr, size_t size, sycl::queue &queue) {
    auto host_ptr = static_cast<data_type *>(malloc(size * sizeof(data_type)));

    queue.memcpy(host_ptr, device_ptr, size * sizeof(data_type)).wait();
    return host_ptr;
}

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float>
int gemm_result_validate(data_type_a *A_device, data_type_b *B_device,
        data_type_c *C_device, uint32_t batch_size, uint32_t m, uint32_t k,
        uint32_t n, sycl::queue &queue,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    bool is_col_major_a = mem_layout_a_ == mem_layout::col_major;
    bool is_col_major_b = mem_layout_b_ == mem_layout::col_major;
    // define slice of each matrices
    size_t size_a_slice = m * k;
    size_t size_b_slice = k * n;
    size_t size_c_slice = m * n;

    auto A = alloc_host_and_copy<data_type_a>(
            A_device, batch_size * size_a_slice, queue);
    auto B = alloc_host_and_copy<data_type_b>(
            B_device, batch_size * size_b_slice, queue);
    auto C = alloc_host_and_copy<data_type_c>(
            C_device, batch_size * size_c_slice, queue);

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

    free(A);
    free(B);
    free(C);

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}