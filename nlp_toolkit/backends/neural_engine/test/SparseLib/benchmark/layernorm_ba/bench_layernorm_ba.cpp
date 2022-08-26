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

#include "layernorm_ba/bench_layernorm_ba.hpp"

namespace jd {

bench_res_t test_layernorm_ba(bench_mode mode, int argc, char** argv) {
  bench_res_t res;

  if (argc < LAYERNORM_BA_ARG_NUM) {
    std::cerr << "Not enough arguments passed" << std::endl;
    res.stat = bench_status::wrong_input;
    return res;
  }
  printf("layernorm_ba\n");
  res = run_bench_layernorm_ba(mode, argc, argv);

  return res;
}

}  // namespace jd
