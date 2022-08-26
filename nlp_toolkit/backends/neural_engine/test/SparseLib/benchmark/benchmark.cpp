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

#include "benchmark_utils.hpp"
#include "sparse_matmul/bench_sparse_matmul.hpp"
#include "eltwiseop/bench_eltwiseop.hpp"
#include "layernorm_ba/bench_layernorm_ba.hpp"

int main(int argc, char** argv) {
  jd::bench_mode mode = jd::bench_mode::acc;
  jd::bench_res_t res;

  if (argc < 5) {
    std::cerr << "Not enough arguments passed" << std::endl;
    return 1;
  }

  --argc;
  ++argv;

  if (!strcmp(argv[0], "acc")) {
    mode = jd::bench_mode::acc;
  } else if (!strcmp(argv[0], "perf")) {
    mode = jd::bench_mode::perf;
  } else {
    std::cerr << "unknown mode" << std::endl;
    return 1;
  }

  --argc;
  ++argv;

  if (!strcmp(argv[0], "sparse_matmul")) {
    res = jd::test_sparse_matmul(mode, --argc, ++argv);
  } else if (!strcmp(argv[0], "eltwiseop")) {
    res = jd::test_eltwiseop(mode, --argc, ++argv);
  } else if (!strcmp(argv[0], "layernorm_ba")) {
    res = jd::test_layernorm_ba(mode, --argc, ++argv);
  } else {
    std::cerr << "unknown kernel type" << std::endl;
    return 1;
  }

  if (res.stat != jd::bench_status::success) {
    std::cerr << "benchmark failed" << std::endl;
    return 1;
  }
  if (mode == jd::bench_mode::acc) {
    if (res.correct) {
      printf("result correct\n");
    } else {
      printf("result incorrect\n");
    }
  } else if (mode == jd::bench_mode::perf) {
    printf("kernel execution time: %lfms,  GFLOPS:%lf\n", res.ms, res.gflops);
  }

  return 0;
}
