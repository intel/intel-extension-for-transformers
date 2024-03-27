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
#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_DATA_PACK_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_DATA_PACK_HPP_

#include <functional>
#include "param_types.hpp"
#include "data_type/data_types.hpp"
#include "common.h"

namespace jd {
template <typename dst_t, typename src_t>
SPARSE_API_ void pack(
    dst_t* output, src_t* input, dim_t N, dim_t K,
    std::function<dst_t(src_t)> cast_func = [](src_t x) -> dst_t { return static_cast<dst_t>(x); });
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_DATA_PACK_HPP_
