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
#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_DYNAMIC_QUANT_MATMUL_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_DYNAMIC_QUANT_MATMUL_HPP_

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "interface.hpp"
#include "kernels/exposed_enum.hpp"

#define DYNAMIC_QUANT_MATMUL_ARG_NUM 5
namespace bench {
class dynamic_quant_matmul_bench : public kernel_bench {
  using io = jd::exposed_enum::dynamic_quant_matmul::io;

 protected:
  int b, m, n, k;
  std::string add_bias, large_wei_threshold;

 private:
  std::unordered_map<std::string, std::string> op_attrs;

 public:
  dynamic_quant_matmul_bench() {}
  virtual ~dynamic_quant_matmul_bench() {
    const auto& p_rt_data = args.first.rt_data;
    const auto& q_rt_data = args.second.rt_data;
    for (auto&& i : p_rt_data) free(const_cast<void*>(i));
    free(const_cast<void*>(q_rt_data[io::DST]));
    free(const_cast<void*>(q_rt_data[io::SCALE_DST]));
    free(const_cast<void*>(q_rt_data[io::WORKSPACE]));
  }
  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override {
    double FLOPs = 0.0f;
    FLOPs += 2. * b * m * n * k;
    return FLOPs;
  }
  std::vector<int> get_refresh_data_idx() const override {
    return std::vector<int>{io::ACTIVATION, io::WEIGHT, io::DST, io::SCALE_A, io::SCALE_W, io::SCALE_DST, io::BIAS};
  }
  int get_workspace_idx() const override { return io::WORKSPACE; }
  // Just like that in gtest file
  void get_true_data() override {}
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
  void set_kernel_proxy() override {
    jd::dynamic_quant_matmul_desc desc(args.first.op_desc);
    kp = std::make_shared<jd::dynamic_quant_matmul>(desc);
  }
};
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_DYNAMIC_QUANT_MATMUL_HPP_
