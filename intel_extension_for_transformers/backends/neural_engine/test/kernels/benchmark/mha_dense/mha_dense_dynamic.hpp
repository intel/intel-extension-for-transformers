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
#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_MHA_DENSE_DYNAMIC_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_MHA_DENSE_DYNAMIC_HPP_

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "interface.hpp"
#include "mha_dense.hpp"

namespace bench {
class mha_dense_dynamic_bench : public mha_dense_bench {
 private:
  int64_t batch_size;
  int64_t head_num;
  int64_t sl_M;
  int64_t head_size;
  int64_t sl_N;
  std::unordered_map<std::string, std::string> op_attrs;

 public:
  static constexpr int MIN_ARG_NUM = 5;
  mha_dense_dynamic_bench() {}
  virtual ~mha_dense_dynamic_bench() {}  // leave memory deallocation to its parent class
  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override {
    double FLOPs = 0.0f;
    // K x Q
    FLOPs += 2. * batch_size * head_num * sl_M * head_size * sl_N;
    // Softmax
    FLOPs += 6. * batch_size * sl_M * head_num * sl_N;
    // V x A
    FLOPs += 2. * batch_size * head_num * sl_M * sl_N * head_size;
    return FLOPs;
  }
  std::vector<int> get_refresh_data_idx() const override {
    return std::vector<int>{io::SRC_Q, io::SRC_K, io::MASK, io::SRC_V, io::DST,
                            // TODO(Yi): seems that refreshing QKV scale needs to be very careful as bad values of these
                            // scale will affect performance dramatically
                            /*io::Q_SCALE, io::K_SCALE, io::V_SCALE,*/
                            io::DST_SCALE};
  }
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
  void set_kernel_proxy() override {
    jd::mha_dense_desc desc(args.first.op_desc);
    kp = std::make_shared<jd::mha_dense>(desc);
  }
};
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_MHA_DENSE_DYNAMIC_HPP_
