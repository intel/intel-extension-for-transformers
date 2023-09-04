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

#include "vtune_wrapper.hpp"

#ifdef SPARSE_LIB_USE_VTUNE
vtune_wrapper_t::vtune_wrapper_t() {
  domain_ = __itt_domain_create("SparseLib.VTune");  //  default name
}

vtune_wrapper_t::vtune_wrapper_t(std::string domain_name) { domain_ = __itt_domain_create(domain_name.c_str()); }

void vtune_wrapper_t::profiling_begin(std::string task_name) {
  handle_main_ = __itt_string_handle_create(task_name.c_str());
  __itt_task_begin(domain_, __itt_null, __itt_null, handle_main_);
}

void vtune_wrapper_t::profiling_end() { __itt_task_end(domain_); }

bool get_vtune() {
  bool run_time_use_vtune = false;
  const char* val = std::getenv("SPARSE_LIB_VTUNE");
  if (val != nullptr) {
    if (strcmp(val, "1") == 0) {
      run_time_use_vtune = true;
    }
  }
  return run_time_use_vtune;
}
#endif
