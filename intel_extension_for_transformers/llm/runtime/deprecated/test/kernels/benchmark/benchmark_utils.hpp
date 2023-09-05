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
#include <cstring>
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
namespace bench {
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
  jd::operator_desc op_desc;
  std::vector<const void*> rt_data;
};

struct bench_data_t {
  jd::operator_desc op_desc;
  mutable jd::exec_context_t ctx_kern;  // mutable to set_workspace
  mutable jd::exec_context_t ctx_ref;   // mutable to set_workspace
};

// Kernel developers will implement utility functions by themselves
class kernel_bench {
 protected:
  std::vector<jd::tensor_desc> ts_descs;
  std::vector<float> ranges = {-10.0, 10.0};  // Usually size = 2, range of tensors' values
  std::pair<op_args_t, op_args_t> args;
  bench_data_t bench_data;  // preferring bench_data; args are deprecated
  std::shared_ptr<jd::kernel_proxy> kp;

 public:
  kernel_bench() : bench_data{{}, jd::exec_context_t(nullptr), jd::exec_context_t(nullptr)} {}
  virtual ~kernel_bench() {}
  // Use command line input to set config data
  virtual bench_res_t set_config(int argc, char** argv) = 0;
  // Calculate flop
  virtual double calc_flop() const = 0;
  // Determine which part of rt_data needs to be refreshed
  // in every iteration before executing kernel
  virtual std::vector<int> get_refresh_data_idx() const = 0;
  // Determine which part of input of context needs to be refreshed in every iteration before executing kernel
  virtual std::vector<int> get_refresh_src_data_idx() const { return {}; }
  // Corresponding index in ts_desc for get_refresh_src_data_idx
  virtual std::vector<int> get_refresh_src_desc_idx() const { return {}; }
  // Determine which part of output of context needs to be refreshed in every iteration before executing kernel
  virtual std::vector<int> get_refresh_dst_data_idx() const { return {}; }
  // Corresponding index in ts_desc for get_refresh_dst_data_idx
  virtual std::vector<int> get_refresh_dst_desc_idx() const { return {}; }
  // Calculate reference, only used when testing acc, just like that in gtest file
  virtual void get_true_data() = 0;
  // Test acc, just like that in gtest file
  virtual bool check_result() = 0;
  // Set args for kernel, just like that in gtest file
  virtual void gen_case() = 0;
  // Set kp, use kp->execute() to run kernel
  virtual void set_kernel_proxy() = 0;
  // The index of workspace pointer in rt_data; Use negative values for kernels which do not need workspace
  virtual int get_workspace_idx() const { return -1; }
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
   * @brief Calculate FLOP.
   */
  double calc_flop(const jd::kernel_kind, const std::vector<jd::tensor_desc>& ts_descs);
  /*
   * @brief Get indices of data that needs refreshing which indicate their positions in tensor vector.
   */
  std::vector<int> get_refresh_data_idx(const jd::kernel_kind ker_kind);
};
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_BENCHMARK_UTILS_HPP_
