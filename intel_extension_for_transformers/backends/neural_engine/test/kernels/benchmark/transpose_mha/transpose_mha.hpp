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
#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_TRANSPOSE_MHA_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_TRANSPOSE_MHA_HPP_
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <limits>
#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "interface.hpp"
#include "kernels/transpose_mha_types.hpp"
#define TRANSPOSE_MHA_ARG_NUM 4
namespace bench {
class transpose_mha_bench : public kernel_bench {
 protected:
  using io = jd::ssd::transpose_mha_io::io;

 private:
  int64_t head_num;
  int64_t head_size;
  int64_t batch_size;
  int64_t seq_len;
  std::string impl = "";
  std::unordered_map<std::string, std::string> op_attrs;

 public:
  transpose_mha_bench() {}
  virtual ~transpose_mha_bench() {
    const auto& rt_data = args.first.rt_data;
    aligned_allocator_t<int8_t>::deallocate(const_cast<void*>(rt_data[io::SRC_K]));
    aligned_allocator_t<int8_t>::deallocate(const_cast<void*>(rt_data[io::SRC_Q]));
    aligned_allocator_t<float>::deallocate(const_cast<void*>(rt_data[io::MASK]));
    aligned_allocator_t<int8_t>::deallocate(const_cast<void*>(rt_data[io::SRC_V]));
    aligned_allocator_t<uint8_t>::deallocate(const_cast<void*>(rt_data[io::DST]));
    aligned_allocator_t<uint8_t>::deallocate(const_cast<void*>(rt_data[io::TMP2M]));
    for (std::underlying_type<io>::type i = io::SL_PAD; i <= io::transpose_mha_io_MAX; i++) {
      delete reinterpret_cast<const int*>(rt_data[i]);
    }
  }
  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override {
    const auto& spmm_shape = ts_descs[io::SRC_K].shape();
    const auto batch_size = spmm_shape[0];
    const auto head_num = spmm_shape[1];
    const auto head_size = spmm_shape[2];
    const auto seq_len = spmm_shape[3];
    double FLOPs = 0.0f;
    // K x Q
    FLOPs += 2. * batch_size * head_num * seq_len * head_size * seq_len;
    // Softmax
    FLOPs += 6. * batch_size * seq_len * head_num * head_size;
    // V x A
    FLOPs += 2. * batch_size * head_num * head_size * seq_len * seq_len;
    return FLOPs;
  }
  std::vector<int> get_refresh_data_idx() const override {
    return std::vector<int>{io::SRC_K, io::SRC_Q, io::SRC_V, io::DST};
  }
  // Just like that in gtest file
  void get_true_data() override {}
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
  void set_kernel_proxy() override {
    jd::transpose_mha_desc desc(args.first.op_desc);
    kp = std::make_shared<jd::transpose_mha>(desc);
  }
};
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_TRANSPOSE_MHA_HPP_
