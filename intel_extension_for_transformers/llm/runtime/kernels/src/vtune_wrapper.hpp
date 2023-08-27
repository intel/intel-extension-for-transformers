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

#ifndef ENGINE_SPARSELIB_SRC_VTUNE_WRAPPER_HPP_
#define ENGINE_SPARSELIB_SRC_VTUNE_WRAPPER_HPP_

#include <stdlib.h>
#include <string>
#include <cstring>

#ifdef SPARSE_LIB_USE_VTUNE
#include <ittnotify.h>
#endif

/**
 * The vtune_wrapper_t is a wrapper for vtune itt task
 */
class vtune_wrapper_t {
#ifdef SPARSE_LIB_USE_VTUNE
 private:
  __itt_domain* domain_;
  __itt_string_handle* handle_main_;

 public:
  vtune_wrapper_t();
  explicit vtune_wrapper_t(std::string domain_name);
  ~vtune_wrapper_t() {}

 public:
  void profiling_begin(std::string task_name);
  void profiling_end();
#endif
};

bool get_vtune();

#endif  // ENGINE_SPARSELIB_SRC_VTUNE_WRAPPER_HPP_
