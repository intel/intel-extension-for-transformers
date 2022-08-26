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

#include "eltwiseop/bench_eltwiseop.hpp"

namespace jd {

bench_res_t test_eltwiseop(bench_mode mode, int argc, char** argv) {
  bench_res_t res;

  if (argc < ELTWISEOP_ARG_NUM) {
    std::cerr << "Not enough arguments passed" << std::endl;
    res.stat = bench_status::wrong_input;
    return res;
  }
  printf("%s\n", argv[2]);
  res = run_bench_eltwiseop(mode, argc, argv);

  return res;
}

}  // namespace jd
