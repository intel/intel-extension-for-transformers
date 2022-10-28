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

#include <glog/logging.h>

#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
  std::vector<void*> rt_data;
};

// Kernel developers will implement utility functions by themselves
class kernel_bench {
 protected:
  std::vector<tensor_desc> ts_descs;
  std::vector<float> ranges = {10.0, 10.0};  // Usually size = 2, range of tensors' values
  std::pair<op_args_t, op_args_t> args;
  std::shared_ptr<kernel_proxy> kp;

 public:
  kernel_bench() {}
  virtual ~kernel_bench() {}
  // Use command line input to set config data
  virtual bench_res_t set_config(int argc, char** argv) = 0;
  // Calculate flop
  virtual double calc_flop() const = 0;
  // Determine which part of rt_data needs to be refreshed
  // in every iteration before executing kernel
  virtual std::vector<int> get_refresh_data_idx() const = 0;
  // Calculate reference, only used when testing acc, just like that in gtest file
  virtual void get_true_data() = 0;
  // Test acc, just like that in gtest file
  virtual bool check_result() = 0;
  // Set args for kernel, just like that in gtest file
  virtual void gen_case() = 0;
  // Set kp, use kp->execute() to run kernel
  virtual void set_kernel_proxy() = 0;
  friend class bench_op;
};

class bench_op {
 private:
  std::shared_ptr<kernel_bench> kb;

 public:
  bench_op() {}
  explicit bench_op(const std::shared_ptr<kernel_bench>& kb) : kb(kb) {}
  ~bench_op() {}
  // Main procedure for benchmark
  bench_res_t run_bench(bench_mode mode);
  /*
   * @brief Run benchmark of kernel. Currently this mainly contains 3 parts:
   *            1. Run kernel for multiple iterations to get its execution time.
   *            2. Parse primitive and use execution time to calculate GFLOPS.
   *            3. Refresh some parts of runtime data for kernel before each execution.
   *
   *        To enable benchmarkOrExecute for a new kernel xxxx, you need to
   *        add a case for it in calc_flop and get_refresh_data_idx in benchmark_utils.cpp
   */
  bench_res_t benchmarkOrExecute(bench_mode mode);

  /*
   * @brief Get execution time of kernel.
   */
  double exec_time(std::shared_ptr<kernel_proxy> kp, const std::vector<void*>& rt_data);
  /*
   * @brief Refresh some parts of runtime data for kernel.
   */
  void refresh_data(std::vector<void*>* new_data_pointer, const std::vector<int>& idx);
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
  bool alloc_new_mem(const std::vector<tensor_desc>& ts_descs, std::vector<void*>* rt_data_pointer,
                     std::vector<void*>* new_data_pointer, const std::vector<int>& idx);

  /*
   * @brief Free new memory for some parts of runtime data for kernel.
   */
  void free_new_mem(std::vector<void*>* new_data_pointer);
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_BENCHMARK_UTILS_HPP_
