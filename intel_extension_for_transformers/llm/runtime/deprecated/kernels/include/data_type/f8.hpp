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

#ifndef ENGINE_SPARSELIB_INCLUDE_DATA_TYPE_F8_HPP_
#define ENGINE_SPARSELIB_INCLUDE_DATA_TYPE_F8_HPP_

#include <cmath>
#include "param_types.hpp"
#include "common.h"

namespace jd {
struct SPARSE_API_ float8_e4m3_t {
  uint8_t data;
  float8_e4m3_t();
  explicit float8_e4m3_t(int32_t val);
  explicit float8_e4m3_t(float val);
  float8_e4m3_t& operator=(float val);
  operator float() const;
};

struct SPARSE_API_ float8_e5m2_t {
  uint8_t data;

  float8_e5m2_t();
  explicit float8_e5m2_t(int32_t val);
  explicit float8_e5m2_t(float val);
  float8_e5m2_t& operator=(float val);
  operator float() const;
};

static_assert(sizeof(float8_e4m3_t) == 1, "float8_e4m3_t must be 1 bytes");
static_assert(sizeof(float8_e5m2_t) == 1, "float8_e5m2_t must be 1 bytes");
}  // namespace jd

#endif  // ENGINE_SPARSELIB_INCLUDE_DATA_TYPE_F8_HPP_
