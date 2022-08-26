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

#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <exception>
#include <utility>
#include <functional>
#include "interface.hpp"
#include "benchmark_utils.hpp"

#define SPMM_VNNI_ARG_NUM 9

namespace jd {

void get_true_data_spmm_vnni(const operator_desc& op_desc, const std::vector<const void*>& rt_data);

bool check_result_spmm_vnni(const std::pair<op_args_t, op_args_t>& args);

template <typename T>
void prepare_sparse_data_spmm_vnni(T* vector_data, std::vector<int64_t> a_shape, float sparse_ratio);

std::pair<const void*, const void*> make_data_obj_spmm_vnni(const std::vector<int64_t>& a_shape, const data_type& a_dt,
                                                            bool is_clear = false, float sparse_ratio = 0.7,
                                                            const std::vector<float>& ranges = {-10, 10});

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, float sparsity, data_type dt_dst = data_type::s8,
                                         const std::string& mkn_blocks = "1,1,1", const std::string& tile_shape = "4,4",
                                         std::string post_op = "");

bench_res_t run_bench_spmm_vnni(bench_mode mode, int argc, char** argv);

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_VNNI_HPP_
