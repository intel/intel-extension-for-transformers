//  Copyright (c) 2021 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#ifndef ENGINE_SPARSELIB_INCLUDE_DATA_TYPE_DATA_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_DATA_TYPE_DATA_TYPES_HPP_

#include <unordered_map>
#include "param_types.hpp"
#include "bf16.hpp"
#include "fp16.hpp"
#include "f8.hpp"

typedef int64_t dim_t;
namespace jd {
static std::unordered_map<data_type, const int> type_size = {{data_type::fp32, sizeof(float)},
                                                             {data_type::s32, sizeof(int32_t)},
                                                             {data_type::fp16, sizeof(float16_t)},
                                                             {data_type::bf16, sizeof(bfloat16_t)},
                                                             {data_type::u8, sizeof(uint8_t)},
                                                             {data_type::s8, sizeof(int8_t)},
                                                             {data_type::f8_e4m3, sizeof(float8_e4m3_t)},
                                                             {data_type::f8_e5m2, sizeof(float8_e5m2_t)}};
}  // namespace jd

#endif  // ENGINE_SPARSELIB_INCLUDE_DATA_TYPE_DATA_TYPES_HPP_
