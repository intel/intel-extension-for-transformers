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

#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_BENCHMARK_UTILS_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_BENCHMARK_UTILS_HPP_

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <utility>
#include <functional>

#include "interface.hpp"

/*
 * @brief Internal Control Variables
 */
extern int benchmark_iter;
extern bool benchmark_refresh;

/*
 * @brief Read environment vars and set internal control variables
 */
void read_benchmark_env();

namespace jd {

enum class bench_status : uint8_t {
  success,
  fail,
  wrong_input,
  unimplemented,
};

struct bench_res_t {
  bench_status stat;
  bool correct;
  double ms;
  double gflops;
};

enum class bench_mode : uint8_t {
  acc,
  perf,
};

struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> rt_data;
};

/*
 * @brief Run benchmark of kernel. Currently this mainly contains 3 parts:
 *            1. Run kernel for multiple iterations to get its execution time.
 *            2. Parse primitive and use execution time to calculate GFLOPS.
 *            3. Refresh some parts of runtime data for kernel before each execution.
 *
 *        To enable benchmarkOrExecute for a new kernel xxxx, you need to
 *        add a case for it in calc_flop and get_refresh_data_idx in benchmark_utils.cpp
 */
bench_res_t benchmarkOrExecute(kernel_proxy* kp, const std::vector<const void*>& rt_data, bench_mode mode);

/*
 * @brief Get execution time of kernel.
 */
double exec_time(kernel_proxy* kp, const std::vector<const void*>& rt_data);

/*
 * @brief Calculate FLOP.
 */
double calc_flop(const kernel_kind, const std::vector<tensor_desc>& ts_descs);

/*
 * @brief Get indices of data that needs refreshing which indicate their positions in tensor vector.
 */
std::vector<int> get_refresh_data_idx(const kernel_kind ker_kind);

/*
 * @brief Allocate new memory for some parts of runtime data for kernel.
 */
bool alloc_new_mem(const std::vector<tensor_desc>& ts_descs, std::vector<const void*>& rt_data,  // NOLINT
                   std::vector<void*>& new_data, const std::vector<int>& idx);                   // NOLINT

/*
 * @brief Free new memory for some parts of runtime data for kernel.
 */
void free_new_mem(std::vector<void*>& new_data);  // NOLINT

/*
 * @brief Refresh some parts of runtime data for kernel.
 */
void refresh_data(const std::vector<tensor_desc>& ts_descs, std::vector<void*>& new_data,  // NOLINT
                  const std::vector<int>& idx, const std::vector<float>& ranges = {-10.0, 10.0});

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_BENCHMARK_UTILS_HPP_
