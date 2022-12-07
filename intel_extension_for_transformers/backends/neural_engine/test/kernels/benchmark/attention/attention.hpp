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

#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_ATTENTION_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_ATTENTION_HPP_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <set>
#include <vector>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "interface.hpp"
#include "kernels/spmm_types.hpp"

#define ATTENTION_ARG_NUM 4

namespace jd {

class attention_bench : public kernel_bench {
 private:
  int64_t head_num;
  int64_t head_size;
  int64_t batch_size;
  int64_t seq_len;
  float sparsity;
  data_type dt_dst;
  std::unordered_map<std::string, std::string> op_attrs;

 public:
  attention_bench() {}
  virtual ~attention_bench() {}

  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override {
    double FLOPs = 0.0f;
    // Three sparse VNNI kernel
    int oc = ts_descs[ssd::V_WEIGHT].shape()[0];
    int ic = ts_descs[ssd::V_WEIGHT].shape()[1];
    const int other_dim = std::accumulate(ts_descs[ssd::MERGE_SRC].shape().begin(),
                                          ts_descs[ssd::MERGE_SRC].shape().end(), 1, std::multiplies<size_t>()) /
                          ic;
    double vnni_FLOPs = static_cast<double>(oc) * other_dim * ic * 2;
    FLOPs += 3 * vnni_FLOPs;
    // quant & dequant
    double tmp = batch_size * seq_len * head_num * head_size;
    FLOPs += 2 * tmp;
    // Softmax
    FLOPs += 6 * tmp;
    // transpose matmul
    // const dim_t M = shapes[ssd::SRC0][2];  // aka src0_perm_shape[2]
    // const dim_t K = shapes[ssd::SRC0][3];  // aka src0_perm_shape[3]
    // const dim_t N = shapes[ssd::SRC1][1];  // aka src1_perm_shape[3]
    // const dim_t bs0 = shapes[ssd::SRC0][0];
    // const dim_t bs1 = shapes[ssd::SRC0][1];
    FLOPs += static_cast<double>(batch_size) * seq_len * head_size * head_num * head_size * 2;
    // const dim_t M = shapes[ssd::SRC0][3];  // aka src0_perm_shape[2]
    // const dim_t K = shapes[ssd::SRC0][1];  // aka src0_perm_shape[3]
    // const dim_t N = shapes[ssd::SRC1][3];  // aka src1_perm_shape[3]
    // const dim_t bs0 = shapes[ssd::DST0][0];
    // const dim_t bs1 = shapes[ssd::DST0][1];
    FLOPs += static_cast<double>(seq_len) * 64 * head_num * ts_descs[ssd::MERGE_DST].shape()[0] *
             ts_descs[ssd::MERGE_DST].shape()[1] * 2;
    return FLOPs;
  }
  std::vector<int> get_refresh_data_idx() const override { return std::vector<int>{ssd::MERGE_SRC, ssd::MERGE_DST}; }
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
  void set_kernel_proxy() override {
    // ts_descs = attention_ptr->ts_descs;
    attention_desc attention_desc(args.first.op_desc);
    kp = std::make_shared<attention>(attention_desc);
  }
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_LAYERNORM_BA_HPP_
