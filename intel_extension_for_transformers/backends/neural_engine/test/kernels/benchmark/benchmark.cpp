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
#include "attention/attention.hpp"
#include "eltwiseop/eltwiseop.hpp"
#include "layernorm_ba/layernorm_ba.hpp"
#include "softmax/softmax.hpp"
#include "sparse_matmul/sparse_matmul.hpp"
#include "transpose_matmul/transpose_matmul.hpp"
#include "transpose_mha/transpose_mha.hpp"

int main(int argc, char** argv) {
  jd::bench_mode mode;
  std::shared_ptr<jd::kernel_bench> kb;

  if (argc < 5) {
    LOG(ERROR) << "Not enough arguments passed";
    return 1;
  }

  --argc;
  ++argv;
  // Get mode from command line input
  if (!strcmp(argv[0], "acc")) {
    mode = jd::bench_mode::acc;
  } else if (!strcmp(argv[0], "perf")) {
    mode = jd::bench_mode::perf;
  } else {
    LOG(ERROR) << "unknown mode";
    return 1;
  }

  --argc;
  ++argv;
  // Determine kernel kind
  if (!strcmp(argv[0], "sparse_matmul")) {
    kb = std::make_shared<jd::sparse_matmul_bench>();
  } else if (!strcmp(argv[0], "layernorm_ba")) {
    kb = std::make_shared<jd::layernorm_ba_bench>();
  } else if (!strcmp(argv[0], "eltwiseop")) {
    kb = std::make_shared<jd::eltwiseop_bench>();
  } else if (!strcmp(argv[0], "transpose_matmul")) {
    kb = std::make_shared<jd::transpose_matmul_bench>();
  } else if (!strcmp(argv[0], "softmax")) {
    kb = std::make_shared<jd::softmax_bench>();
  } else if (!strcmp(argv[0], "attention")) {
    kb = std::make_shared<jd::attention_bench>();
  } else if (!strcmp(argv[0], "transpose_mha")) {
    kb = std::make_shared<jd::transpose_mha_bench>();
  } else {
    LOG(ERROR) << "unknown kernel type";
    return 1;
  }
  // Use command line input to set config parameters

  jd::bench_res_t res = kb->set_config(--argc, ++argv);
  // Run benchmark
  jd::bench_op bench(kb);

  try {
    if (res.stat == jd::bench_status::success) res = bench.run_bench(mode);
  } catch (const std::exception& e) {
    LOG(ERROR) << "kernel exception occurred";
    res.stat = jd::bench_status::fail;
  }
  // Print result
  if (res.stat != jd::bench_status::success) {
    LOG(INFO) << "benchmark failed\n";
    return 1;
  }
  if (mode == jd::bench_mode::acc) {
    if (res.correct) {
      LOG(INFO) << "result correct\n";
    } else {
      LOG(INFO) << "result incorrect\n";
    }
  } else if (mode == jd::bench_mode::perf) {
    LOG(INFO) << "kernel execution time:" << res.ms << "ms,  GFLOPS:" << res.gflops;
  }

  return 0;
}
