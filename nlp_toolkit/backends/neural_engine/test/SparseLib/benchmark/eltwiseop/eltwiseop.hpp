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

#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "interface.hpp"

#define ELTWISEOP_ARG_NUM 4

namespace jd {

class eltwiseop_bench : public kernel_bench {
 private:
  int64_t M;
  int64_t N;
  data_type dt;
  std::vector<postop_attr> postop_attrs;
  std::unordered_map<std::string, std::string> op_attrs;

 public:
  eltwiseop_bench() {}
  virtual ~eltwiseop_bench() {
    for (auto op_args : {args.first, args.second})
      for (auto rt_data : op_args.rt_data)
        if (rt_data != nullptr) {
          free(const_cast<void*>(rt_data));
        }
  }

  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override {
    return std::accumulate(ts_descs[0].shape().begin(), ts_descs[0].shape().end(), 1, std::multiplies<size_t>());
  }
  std::vector<int> get_refresh_data_idx() const override { return std::vector<int>{0, 1}; }
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
  void set_kernel_proxy() override {
    eltwiseop_desc eltwiseop_desc(args.first.op_desc);
    kp = std::make_shared<eltwiseop>(eltwiseop_desc);
  }
  template <typename T>
  void cast_to_float_array(const void* src, float* dst, int size);
  template <typename T>
  void cast_from_float_array(float* src, void* dst, int size);
};
}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_ELTWISEOP_HPP_
