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

#define SPMM_AMX_BF16_X16_ARG_NUM 8

namespace jd {

void get_true_data_spmm_amx_bf16_x16(const operator_desc& op_desc, const std::vector<const void*>& rt_data);

bool check_result_spmm_amx_bf16_x16(const std::pair<op_args_t, op_args_t>& args);

template <typename T>
void prepare_sparse_data_spmm_amx_bf16_x16(T* weight, dim_t N, dim_t K, dim_t n_blksize, dim_t k_blksize, float ratio);

std::pair<const void*, const void*> make_data_obj_spmm_amx_bf16_x16(const data_type& tensor_dt, dim_t rows, dim_t cols,
                                                                    dim_t index, float ratio = 0.9,
                                                                    const std::vector<float>& ranges = {-1, 1});

std::pair<op_args_t, op_args_t> gen_case_spmm_amx_bf16_x16(dim_t M, dim_t K, dim_t N, dim_t micro_bs = 64,
                                                           dim_t micro_oc = -1, float ratio = 0.9,
                                                           bool bf16_out = true);

bench_res_t run_bench_spmm_amx_bf16_x16(bench_mode mode, int argc, char** argv);

}  // namespace jd

#endif  // SPARSE_LIB_USE_AMX

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_AMX_BF16_X16_HPP_
