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

#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_ELTWISEOP_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_ELTWISEOP_HPP_

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <unordered_map>
#include <utility>
#include "interface.hpp"
#include "benchmark_utils.hpp"

#define ELTWISEOP_ARG_NUM 4

namespace jd {

void get_true_data_eltwiseop(const operator_desc& op_desc, const std::vector<const void*>& rf_data);

bool check_result_eltwiseop(const std::pair<op_args_t, op_args_t>& args);

std::pair<op_args_t, op_args_t> gen_case(const std::vector<tensor_desc>& ts_descs, const std::vector<float>& ranges,
                                         const std::unordered_map<std::string, std::string> op_attrs,
                                         const std::vector<postop_attr>& postop_attr);

bench_res_t run_bench_eltwiseop(bench_mode mode, int argc, char** argv);

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_ELTWISEOP_HPP_
