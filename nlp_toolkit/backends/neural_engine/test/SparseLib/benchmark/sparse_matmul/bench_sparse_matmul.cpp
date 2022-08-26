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

#include "sparse_matmul/bench_sparse_matmul.hpp"

namespace jd {

bench_res_t test_sparse_matmul(bench_mode mode, int argc, char** argv) {
  bench_res_t res;

  if (!strcmp(argv[0], "vnni")) {
    printf("spmm_vnni\n");
    if (argc < SPMM_VNNI_ARG_NUM) {
      std::cerr << "Not enough arguments passed" << std::endl;
      res.stat = bench_status::wrong_input;
      return res;
    }
    res = run_bench_spmm_vnni(mode, --argc, ++argv);
  } else if (!strcmp(argv[0], "amx_bf16_x16")) {
#ifdef SPARSE_LIB_USE_AMX
    printf("spmm_amx_bf16_x16\n");
    if (argc < SPMM_AMX_BF16_X16_ARG_NUM) {
      std::cerr << "Not enough arguments passed" << std::endl;
      res.stat = bench_status::wrong_input;
      return res;
    }
    res = run_bench_spmm_amx_bf16_x16(mode, --argc, ++argv);
#else
    std::cerr << "SPARSE_LIB_USE_AMX is off" << std::endl;
    res.stat = bench_status::unimplemented;
#endif  // SPARSE_LIB_USE_AMX
  } else if (!strcmp(argv[0], "avx512f")) {
    printf("spmm_avx512f\n");
    if (argc < SPMM_AVX512F_ARG_NUM) {
      std::cerr << "Not enough arguments passed" << std::endl;
      res.stat = bench_status::wrong_input;
      return res;
    }
    res = run_bench_spmm_avx512f(mode, --argc, ++argv);
  } else {
    std::cerr << "unknown kernel specification" << std::endl;
    res.stat = bench_status::wrong_input;
  }

  return res;
}

}  // namespace jd
