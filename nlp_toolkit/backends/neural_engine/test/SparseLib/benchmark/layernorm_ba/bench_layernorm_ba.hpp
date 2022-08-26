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

#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_BENCH_LAYERNORM_BA_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_BENCH_LAYERNORM_BA_HPP_

#include "benchmark_utils.hpp"
#include "layernorm_ba/layernorm_ba.hpp"

namespace jd {

bench_res_t test_layernorm_ba(bench_mode, int argc, char** argv);

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_BENCH_LAYERNORM_BA_HPP_
