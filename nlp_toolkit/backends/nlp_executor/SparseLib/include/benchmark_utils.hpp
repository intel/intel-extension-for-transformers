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

#ifndef ENGINE_SPARSELIB_INCLUDE_BENCHMARK_UTILS_HPP_
#define ENGINE_SPARSELIB_INCLUDE_BENCHMARK_UTILS_HPP_

#include <vector>
#include <string>
#include <cstring>

#include "interface.hpp"

namespace jd{

/*
 * @brief Run benchmark of kernel. Currently this mainly contains 3 parts:
 *            1. Run kernel for multiple iterations to get its execution time.
 *            2. Parse primitive and use execution time to calculate GFLOPS.
 *            3. Refresh some parts of runtime data for kernel before each execution.
 *        
 *        To enable benchmark for a new kernel xxxx, you just need 2 steps:
 *            1. Implement calc_flop_xxxx and get_refresh_data_idx_xxxx for it.
 *            2. Simply add a case for it in calc_flop and get_refresh_data_idx in benchmark_utils.cpp  
 */
void benchmarkOrExecute(kernel_proxy* kp, const std::vector<const void*>& rt_data);

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
bool alloc_new_mem(const std::vector<tensor_desc>& ts_descs, std::vector<const void*>& rt_data, std::vector<void*>& new_data, const std::vector<int>& idx);

/*
 * @brief Free new memory for some parts of runtime data for kernel.
 */
void free_new_mem(std::vector<void*>& new_data);

/*
 * @brief Refresh some parts of runtime data for kernel.
 */
void refresh_data(const std::vector<tensor_desc>& ts_descs, std::vector<void*>& new_data, const std::vector<int>& idx, const std::vector<float>& ranges = {-10.0, 10.0});

// Since different kernels use different info to calculate FLOP,
// please implement calc_flop_xxxx for each kernel.

double calc_flop_sparse_matmul(const std::vector<tensor_desc>& ts_descs);

double calc_flop_postop(const std::vector<tensor_desc>& ts_descs);

// Since different kernels may need to refresh different parts of runtime data,
// please implement get_refresh_data_idx_xxxx for each kernel.

std::vector<int> get_refresh_data_idx_sparse_matmul();

std::vector<int> get_refresh_data_idx_postop();

} // namespace jd

#endif  // ENGINE_SPARSELIB_INCLUDE_BENCHMARK_UTILS_HPP_

