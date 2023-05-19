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

/// @file
/// C++ API

#pragma once

#include "common/core/common.hpp"

namespace gpu::xetla {
namespace detail {

///
///@brief convert normal data type to dpas argument type
///
///
template <typename dtype>
constexpr gpu::xetla::argument_type mma_argument_type() {
    return gpu::xetla::argument_type::U1;
}

template <>
constexpr gpu::xetla::argument_type mma_argument_type<tf32>() {
    return gpu::xetla::argument_type::TF32;
}

template <>
constexpr gpu::xetla::argument_type mma_argument_type<float>() {
    return gpu::xetla::argument_type::TF32;
}

template <>
constexpr gpu::xetla::argument_type mma_argument_type<int8_t>() {
    return gpu::xetla::argument_type::S8;
}

template <>
constexpr gpu::xetla::argument_type mma_argument_type<uint8_t>() {
    return gpu::xetla::argument_type::U8;
}

template <>
constexpr gpu::xetla::argument_type mma_argument_type<bf16>() {
    return gpu::xetla::argument_type::BF16;
}

template <>
constexpr gpu::xetla::argument_type mma_argument_type<fp16>() {
    return gpu::xetla::argument_type::FP16;
}

/// @brief lookup table for dpas argument type
///
///
template <gpu::xetla::argument_type arg_type>
constexpr __ESIMD_NS::xmx::dpas_argument_type get_argument_type() {
    static_assert(arg_type == gpu::xetla::argument_type::U1
                    || arg_type == gpu::xetla::argument_type::S1
                    || arg_type == gpu::xetla::argument_type::U2
                    || arg_type == gpu::xetla::argument_type::S2
                    || arg_type == gpu::xetla::argument_type::U4
                    || arg_type == gpu::xetla::argument_type::S4
                    || arg_type == gpu::xetla::argument_type::U8
                    || arg_type == gpu::xetla::argument_type::S8
                    || arg_type == gpu::xetla::argument_type::FP16
                    || arg_type == gpu::xetla::argument_type::BF16
                    || arg_type == gpu::xetla::argument_type::TF32,
            "Unsupported argument type");
    switch (arg_type) {
        case gpu::xetla::argument_type::U1:
            return __ESIMD_NS::xmx::dpas_argument_type::u1;
        case gpu::xetla::argument_type::S1:
            return __ESIMD_NS::xmx::dpas_argument_type::s1;
        case gpu::xetla::argument_type::U2:
            return __ESIMD_NS::xmx::dpas_argument_type::u2;
        case gpu::xetla::argument_type::S2:
            return __ESIMD_NS::xmx::dpas_argument_type::s2;
        case gpu::xetla::argument_type::U4:
            return __ESIMD_NS::xmx::dpas_argument_type::u4;
        case gpu::xetla::argument_type::S4:
            return __ESIMD_NS::xmx::dpas_argument_type::s4;
        case gpu::xetla::argument_type::U8:
            return __ESIMD_NS::xmx::dpas_argument_type::u8;
        case gpu::xetla::argument_type::S8:
            return __ESIMD_NS::xmx::dpas_argument_type::s8;
        case gpu::xetla::argument_type::BF16:
            return __ESIMD_NS::xmx::dpas_argument_type::bf16;
        case gpu::xetla::argument_type::FP16:
            return __ESIMD_NS::xmx::dpas_argument_type::fp16;
        case gpu::xetla::argument_type::TF32:
            return __ESIMD_NS::xmx::dpas_argument_type::tf32;
        default:;
    }
}

} // namespace detail

/// @addtogroup xetla_core_math
/// @{

///
///@brief description of xetla mma
/// perform matrix multiply add operation
///@tparam src1_precision is the data precision of src1
///@tparam src2_precision is the data precision of src2
///@tparam systolic_depth is the depth of mma (i.e k dimension size in dword)
///@tparam repeat_count is the row (m) of mma (mxkxn)
///@tparam T is the data type of src0 and dst
///@tparam T1 is the data type of src1
///@tparam T2 is the data type of src2
///@tparam N is the total number of elements in src0 and dst
///@tparam N1 is the total number of elements in src1
///@tparam N2 is the total number of elements in src2
///@tparam Sat is saturation flag
///@param src0 [in] is src0
///@param src1 [in] is src1
///@param src2 [in] is src2
///@param sat  [in] is saturation flag
///@return is dst, a xetla_vector of  type T and element size of N
///

/// todo: we need to remove argument_type and get the currect precision based on data type
template <argument_type src1_precision, argument_type src2_precision,
        int systolic_depth, int repeat_count, typename T, typename T1,
        typename T2, int N, int N1, int N2,
        typename Sat = xettp_saturation_off_tag>
__XETLA_API xetla_vector<T, N> xetla_mma(xetla_vector<T, N> src0,
        xetla_vector<T1, N1> src1, xetla_vector<T2, N2> src2, Sat sat = {}) {
    return __ESIMD_NS::xmx::dpas<systolic_depth, repeat_count, T, T, T1, T2,
            detail::get_argument_type<src1_precision>(),
            detail::get_argument_type<src2_precision>()>(src0, src1, src2);
}

/// @} xetla_core_math

} // namespace gpu::xetla
