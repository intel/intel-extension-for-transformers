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
#pragma once
#include <torch/serialize/input-archive.h>
#include <chrono>
#include <string>
#include "jblas/jit_blas_device.h"
#include "jblas/jit_blas_utils.h"
#include "jblas/jit_blas_parallel.h"
namespace dispatcher_utils {

inline bool check_amx() { return jblas::device::CpuDevice::getInstance()->AMX_BF16(); }
inline bool check_avx512_vnni() { return jblas::device::CpuDevice::getInstance()->AVX512_VNNI(); }
inline bool check_avx_vnni() { return jblas::device::CpuDevice::getInstance()->AVX_VNNI(); };
inline bool check_avx512f() { return jblas::device::CpuDevice::getInstance()->AVX512F(); }
inline bool check_avx2() { return jblas::device::CpuDevice::getInstance()->AVX2(); }

class env_initer {
 public:
  env_initer() {
    if (check_amx()) jblas::utils::request_perm_xtile_data();
    verbose = std::getenv("QBITS_VERBOSE") != nullptr;
    FLAGS_caffe2_log_level = 0;
  }
  bool verbose;
};
static env_initer initer;

enum QBITS_DT {
  QBITS_FP32,
  QBITS_BF16,
  QBITS_FP16,
};

using namespace std;
using namespace std::chrono;
class Timer {
 public:
  void start() { m_start = high_resolution_clock::now(); }
  void stop() { m_end = high_resolution_clock::now(); }
  double get_elapsed_time() const { return duration_cast<nanoseconds>(m_end - m_start).count() / 1e6; }

 private:
  high_resolution_clock::time_point m_start;
  high_resolution_clock::time_point m_end;
};
static Timer timer;
// static jblas::parallel::OMPThreading DefaultThreading(jblas::device::CpuDevice::getInstance()->getThreads());
static jblas::parallel::OMPThreading DefaultThreading(1);
string get_torch_dt_name(torch::Tensor* tensor);

}  // namespace dispatcher_utils
