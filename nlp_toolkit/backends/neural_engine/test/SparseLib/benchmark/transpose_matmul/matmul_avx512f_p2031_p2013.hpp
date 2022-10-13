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

#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_MATMUL_AVX512F_P2031_P2013_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_MATMUL_AVX512F_P2031_P2013_HPP_

#include <omp.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "benchmark_utils.hpp"
#include "interface.hpp"
#include "transpose_matmul/transpose_matmul.hpp"

#define MATMUL_AVX512F_P2031_P2013_ARG_NUM 6

namespace jd {
class matmul_avx512f_p2031_p2013_bench : public transpose_matmul_bench {
 private:
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t bs0;
  int64_t bs1;
  std::unordered_map<std::string, std::string> op_attrs = {};
  bool has_binary_add = true;

 public:
  matmul_avx512f_p2031_p2013_bench() {}
  virtual ~matmul_avx512f_p2031_p2013_bench() {}

  bench_res_t set_config(int argc, char** argv) override;
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
};
template <typename T>
void prepare_sparse_data_matmul_avx512f_p2031_p2013(T* vector_data, std::vector<int64_t> a_shape, float sparse_ratio);

std::pair<const void*, const void*> make_data_obj_matmul_avx512f_p2031_p2013(
    const std::vector<int64_t>& a_shape, const data_type& a_dt, bool is_clear = false, float sparse_ratio = 0.7,
    const std::vector<float>& ranges = {-10, 10});

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_MATMUL_AVX512F_P2031_P2013_HPP_
