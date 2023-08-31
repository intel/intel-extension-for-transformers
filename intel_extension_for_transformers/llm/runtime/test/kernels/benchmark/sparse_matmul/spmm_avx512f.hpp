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
#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_AVX512F_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_AVX512F_HPP_

#include <vector>
#include <string>
#include <unordered_map>

#include "interface.hpp"
#include "benchmark_utils.hpp"
#include "sparse_matmul.hpp"
#include "common_utils.hpp"

#define SPMM_AVX512F_ARG_NUM 4

namespace bench {
class spmm_avx512f_bench : public sparse_matmul_bench {
 private:
  int64_t M, K, N;
  float sparse_ratio;
  std::vector<jd::postop_alg> postop_algs;
 public:
  spmm_avx512f_bench() {}
  virtual ~spmm_avx512f_bench() {}
  bench_res_t set_config(int argc, char** argv) override;
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
};
}  // namespace bench

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_AVX512F_HPP_
