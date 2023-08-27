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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "interface.hpp"
#include "kernels/attention_types.hpp"
#define ATTENTION_ARG_NUM 4
namespace bench {
class attention_bench : public kernel_bench {
 private:
  int64_t head_num;
  int64_t head_size;
  int64_t batch_size;
  int64_t seq_len;
  float sparsity;
  jd::data_type dt_dst;
  std::unordered_map<std::string, std::string> op_attrs;

 public:
  attention_bench() {}
  virtual ~attention_bench() {
    auto attrs = args.first.op_desc.attrs();
    for (std::string ptr_field : {"q_weight_ptr", "k_weight_ptr", "v_weight_ptr", "q_bias_ptr", "k_bias_ptr",
                                  "v_bias_ptr", "q_scales_ptr", "k_scales_ptr", "v_scales_ptr"}) {
      aligned_allocator_t<uint8_t, 64>::deallocate(reinterpret_cast<void*>(str_to_num<intptr_t>(attrs[ptr_field])));
    }
    for (auto rt_data : {args.first.rt_data, args.second.rt_data}) {
      for (auto idx : {jd::attention_io::MERGE_SRC, jd::attention_io::MERGE_DST, jd::attention_io::Q_K_SRC2}) {
        if (rt_data[idx] != nullptr) {
          aligned_allocator_t<uint8_t, 64>::deallocate(const_cast<void*>(rt_data[idx]));
        }
      }
      delete reinterpret_cast<const float*>(rt_data[jd::attention_io::QK_V_OUTPUT_SCALES]);
      delete reinterpret_cast<const float*>(rt_data[jd::attention_io::QK_V_OUTPUT_ZERO_POINT]);
    }
  }
  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override {
    double FLOPs = 0.0f;
    // Three sparse VNNI kernel
    int oc = ts_descs[jd::attention_io::V_WEIGHT].shape()[0];
    int ic = ts_descs[jd::attention_io::V_WEIGHT].shape()[1];
    const int other_dim =
        std::accumulate(ts_descs[jd::attention_io::MERGE_SRC].shape().begin(),
                        ts_descs[jd::attention_io::MERGE_SRC].shape().end(), 1, std::multiplies<size_t>()) /
        ic;
    double vnni_FLOPs = static_cast<double>(oc) * other_dim * ic * 2;
    FLOPs += 3 * vnni_FLOPs;
    // Softmax
    FLOPs += 6 * batch_size * seq_len * head_num * head_size;
    // transpose matmul
    // const dim_t M = shapes[jd::ssd::SRC0][2];  // aka src0_perm_shape[2]
    // const dim_t K = shapes[jd::ssd::SRC0][3];  // aka src0_perm_shape[3]
    // const dim_t N = shapes[jd::ssd::SRC1][1];  // aka src1_perm_shape[3]
    // const dim_t bs0 = shapes[jd::ssd::SRC0][0];
    // const dim_t bs1 = shapes[jd::ssd::SRC0][1];
    FLOPs += static_cast<double>(batch_size) * seq_len * head_size * head_num * head_size * 2;
    // const dim_t M = shapes[jd::ssd::SRC0][3];  // aka src0_perm_shape[2]
    // const dim_t K = shapes[jd::ssd::SRC0][1];  // aka src0_perm_shape[3]
    // const dim_t N = shapes[jd::ssd::SRC1][3];  // aka src1_perm_shape[3]
    // const dim_t bs0 = shapes[jd::ssd::DST0][0];
    // const dim_t bs1 = shapes[jd::ssd::DST0][1];
    FLOPs += static_cast<double>(seq_len) * seq_len * head_num * ts_descs[jd::attention_io::MERGE_DST].shape()[0] *
             ts_descs[jd::attention_io::MERGE_DST].shape()[1] * 2;
    return FLOPs;
  }
  std::vector<int> get_refresh_data_idx() const override {
    return std::vector<int>{jd::attention_io::MERGE_SRC, jd::attention_io::MERGE_DST};
  }
  // Just like that in gtest file
  void get_true_data() override {}
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
  void set_kernel_proxy() override {
    // ts_descs = attention_ptr->ts_descs;
    jd::attention_desc attention_desc(args.first.op_desc);
    kp = std::make_shared<jd::attention>(attention_desc);
  }
};
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_ATTENTION_HPP_
