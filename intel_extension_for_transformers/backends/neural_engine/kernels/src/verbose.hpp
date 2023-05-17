//  Copyright (c) 2022 Intel Corporation
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
#ifndef ENGINE_SPARSELIB_SRC_VERBOSE_HPP_
#define ENGINE_SPARSELIB_SRC_VERBOSE_HPP_

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mutex>  // NOLINT
#include <string>
#include <cstring>
#include <cassert>
#include <vector>

#include "param_types.hpp"
#include "data_type/data_types.hpp"

namespace jd {
int get_verbose();

bool get_verbose_timestamp();

double get_msec();

template <typename T>
struct setting_t {
 private:
  T value_;
  bool initialized_;

 public:
  constexpr setting_t() : value_(), initialized_(false) {}
  constexpr explicit setting_t(const T init) : value_(init), initialized_(false) {}
  bool initialized() { return initialized_; }
  T get() { return value_; }
  void set(T new_value) {
    value_ = new_value;
    initialized_ = true;
  }
  setting_t(const setting_t&) = delete;
  setting_t& operator=(const setting_t&) = delete;
};

//  A container for primitive desc verbose string.
class kd_info_t {
 public:
  kd_info_t() = default;
  kd_info_t(const kd_info_t& rhs) : str_(rhs.str_), is_initialized_(rhs.is_initialized_) {}
  kd_info_t& operator=(const kd_info_t& rhs) {
    is_initialized_ = rhs.is_initialized_;
    str_ = rhs.str_;
    return *this;
  }

  const char* c_str() const { return str_.c_str(); }
  bool is_initialized() const { return is_initialized_; }

  void init(kernel_kind kind, std::vector<dim_t> shape);

 private:
  std::string str_;

  bool is_initialized_ = false;

  // Alas, `std::once_flag` cannot be manually set and/or copied (in terms of
  // its state). Hence, when `kd_info_t` is copied the `initialization_flag_`
  // is always reset. To avoid re-initialization we use an extra
  // `is_initialized_` flag, that should be checked before calling `init()`.
  std::once_flag initialization_flag_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_VERBOSE_HPP_
