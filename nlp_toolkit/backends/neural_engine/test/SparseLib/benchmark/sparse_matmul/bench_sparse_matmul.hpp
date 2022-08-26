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

#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_BENCH_SPARSE_MATMUL_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_BENCH_SPARSE_MATMUL_HPP_

#include "benchmark_utils.hpp"
#include "sparse_matmul/spmm_vnni.hpp"
#include "sparse_matmul/spmm_amx_bf16_x16.hpp"
#include "sparse_matmul/spmm_avx512f.hpp"

namespace jd {

bench_res_t test_sparse_matmul(bench_mode mode, int argc, char** argv);

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_BENCH_SPARSE_MATMUL_HPP_
