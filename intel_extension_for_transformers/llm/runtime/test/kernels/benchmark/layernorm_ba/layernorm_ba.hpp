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
#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_LAYERNORM_BA_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_LAYERNORM_BA_HPP_

#include <functional>
#include <memory>
#include <string>
#include <set>
#include <unordered_map>
#include <vector>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "interface.hpp"

#define LAYERNORM_BA_ARG_NUM 4
namespace bench {
class layernorm_ba_bench : public kernel_bench {
 private:
  int64_t M;
  int64_t N;
  jd::data_type in_dt;
  jd::data_type out_dt;
  std::vector<jd::postop_attr> postop_attrs;
  std::unordered_map<std::string, std::string> op_attrs;

 public:
  layernorm_ba_bench() {}
  virtual ~layernorm_ba_bench() {
    std::set<const void*> s;  // some of rt_data (alpha & beta) is shared between p and q
    for (auto op_args : {args.first, args.second})
      for (auto rt_data : op_args.rt_data)
        if (rt_data != nullptr && s.find(rt_data) == s.end()) {
          s.insert(rt_data);
          aligned_allocator_t<char>::deallocate(const_cast<void*>(rt_data));
        }
  }
  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override {
    // flop taken into consideration
    // compute mean: elem_num
    // compute variance: 3 * elem_num
    // compute final result: 2 * elem_num
    return 6 * std::accumulate(ts_descs[0].shape().begin(), ts_descs[0].shape().end(), 1, std::multiplies<size_t>());
  }
  std::vector<int> get_refresh_data_idx() const override { return std::vector<int>{0}; }
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
  void set_kernel_proxy() override {
    jd::layernorm_ba_desc layernorm_ba_desc(args.first.op_desc);
    kp = std::make_shared<jd::layernorm_ba>(layernorm_ba_desc);
  }
};
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_LAYERNORM_BA_HPP_
