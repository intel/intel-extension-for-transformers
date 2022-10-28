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

#include <omp.h>
#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <exception>
#include <functional>
#include <utility>
#include "interface.hpp"
#include "benchmark_utils.hpp"
#include "sparse_matmul/sparse_matmul.hpp"
#define SPMM_AVX512F_ARG_NUM 4

namespace jd {

class spmm_avx512f_bench : public sparse_matmul_bench {
 private:
  int64_t M, K, N;
  float sparse_ratio;
  std::vector<postop_alg> postop_algs;

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

template <typename T>
void prepare_blocked_sparse_data_spmm_avx512f(T* data, const std::vector<dim_t>& a_shape,
                                              const std::vector<dim_t>& block_shape, float sparsity,
                                              unsigned int* seed);

std::pair<void*, void*> make_data_obj_spmm_avx512f(const std::vector<dim_t>& a_shape, const data_type& a_dt,
                                                   bool is_clear = false,
                                                   float sparsity = 0.f,  // 0 for dense
                                                   format_type a_ft = format_type::uncoded,
                                                   const std::vector<float>& ranges = {-10, 10});

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_AVX512F_HPP_
