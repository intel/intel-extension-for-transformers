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
#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_VNNI_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_VNNI_HPP_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "benchmark_utils.hpp"
#include "interface.hpp"
#include "sparse_matmul.hpp"
#define SPMM_VNNI_ARG_NUM 8

namespace bench {
class spmm_vnni_bench : public sparse_matmul_bench {
 private:
  int64_t M, K, N, micro_bs = -1;
  float sparse_ratio;
  jd::data_type dt_dst = jd::data_type::s8;
  std::unordered_map<std::string, std::string> op_attrs = {};
  std::vector<jd::postop_alg> postop_algs = {};
  bool calc_mean_var = false;

 public:
  spmm_vnni_bench() {}
  virtual ~spmm_vnni_bench() {}
  bench_res_t set_config(int argc, char** argv) override;
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
};
template <typename T>
void prepare_sparse_data_spmm_vnni(T* vector_data, std::vector<int64_t> a_shape, float sparse_ratio);
std::pair<const void*, const void*> make_data_obj_spmm_vnni(const std::vector<int64_t>& a_shape,
                                                            const jd::data_type& a_dt, bool is_clear = false,
                                                            float sparse_ratio = 0.7,
                                                            const std::vector<float>& ranges = {-10, 10});
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_VNNI_HPP_
