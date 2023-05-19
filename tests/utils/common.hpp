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
#include "common/common.hpp"
#include <CL/sycl.hpp>

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
