//  Copyright (c) 2023 Intel Corporation
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
#include "jblas_common.hpp"
#include "jblas/jit_blas_weight_compression.h"
using namespace jblas;
using namespace ne_jblas;

#ifdef _OPENMP
static parallel::OMPThreading DefaultThreading(4);
#else
static parallel::StdThreading DefaultThreading(4);
#endif  // _OPNEMP

void jblas_init() {
  GetCPUDevice();
  if (_cd->AMX_BF16() || _cd->AMX_INT8()) {
    utils::request_perm_xtile_data();
  }
  _cd->print();
}

void jblas_timer(bool _init) {
  static utils::timer<utils::microseconds> tr;
  if (_init)
    tr.start();
  else
    printf("time :%f us\n", tr.stop());
}

int jblas_set_threads(int _nth) {
  DefaultThreading.set_threads(_nth);
  return DefaultThreading.num_threads();
}

jblas::parallel::IThreading* get_threading() { return &DefaultThreading; }
