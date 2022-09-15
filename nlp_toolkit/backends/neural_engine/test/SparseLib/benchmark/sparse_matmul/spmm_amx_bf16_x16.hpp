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

#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_AMX_BF16_X16_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_AMX_BF16_X16_HPP_

#ifdef SPARSE_LIB_USE_AMX

#include <omp.h>

#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <vector>

#include "interface.hpp"
#include "benchmark_utils.hpp"
#include "sparse_matmul/sparse_matmul.hpp"
#define SPMM_AMX_BF16_X16_ARG_NUM 7

namespace jd {
class spmm_amx_bf16_x16_bench : public sparse_matmul_bench {
 private:
  int64_t M, K, N;
  float sparse_ratio;
  int64_t micro_bs, micro_oc;
  bool bf16_out;

 public:
  spmm_amx_bf16_x16_bench() {}
  virtual ~spmm_amx_bf16_x16_bench() {}

  bench_res_t set_config(int argc, char** argv) override;
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
};

template <typename T>
void prepare_sparse_data_spmm_amx_bf16_x16(T* weight, dim_t N, dim_t K, dim_t n_blksize, dim_t k_blksize, float ratio);

std::pair<const void*, const void*> make_data_obj_spmm_amx_bf16_x16(const data_type& tensor_dt, dim_t rows, dim_t cols,
                                                                    dim_t index, float ratio = 0.9,
                                                                    const std::vector<float>& ranges = {-1, 1});

}  // namespace jd

#endif  // SPARSE_LIB_USE_AMX

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_AMX_BF16_X16_HPP_
