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
#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_MHA_DENSE_STATIC_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_MHA_DENSE_STATIC_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "benchmark_utils.hpp"
#include "interface.hpp"
#include "mha_dense.hpp"

namespace bench {
class mha_dense_static_bench : public mha_dense_bench {
 private:
  int64_t head_num;
  int64_t head_size;
  int64_t batch_size;
  int64_t sl_m;
  int64_t sl_n;
  int32_t mask = -1;  // valid seqlen
  jd::data_type dt_dst, dt_src;
  jd::format_type ft_kv = jd::format_type::undef;
  bool stable_softmax = false;
  int badd_dim;  // #dimention of the binary_add src tensor; Non-positive number for disabling binary_add
  std::unordered_map<std::string, std::string> op_attrs;

 public:
  static constexpr int MIN_ARG_NUM = 4;
  mha_dense_static_bench() : mha_dense_bench() {}
  virtual ~mha_dense_static_bench() {}  // leave memory deallocation to its parent class
  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const final;
  std::vector<int> get_refresh_data_idx() const override { return {io::SRC_Q, io::SRC_K, io::SRC_V, io::DST}; }
  std::vector<int> get_refresh_src_data_idx() const override { return {io_src::SRC_Q, io_src::SRC_K, io_src::SRC_V}; }
  std::vector<int> get_refresh_src_desc_idx() const override { return {io::SRC_Q, io::SRC_K, io::SRC_V}; }
  std::vector<int> get_refresh_dst_data_idx() const override { return {io_dst::DST}; }
  std::vector<int> get_refresh_dst_desc_idx() const override { return {io::DST}; }
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
  void set_kernel_proxy() override {
    jd::mha_dense_desc mha_dense_desc(bench_data.op_desc);
    kp = std::make_shared<jd::mha_dense>(mha_dense_desc);
  }
};
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_MHA_DENSE_STATIC_HPP_
