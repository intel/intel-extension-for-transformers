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

#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_MHA_DENSE_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_MHA_DENSE_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "benchmark_utils.hpp"
#include "interface.hpp"
#include "kernels/mha_dense_types.hpp"

#define MHA_DENSE_ARG_NUM 4

namespace jd {

class mha_dense_bench : public kernel_bench {
 private:
  int64_t head_num;
  int64_t head_size;
  int64_t batch_size;
  int64_t sl_m;
  int64_t sl_n;
  int32_t mask = -1;  // valid seqlen
  data_type dt_dst;
  int badd_dim;  // #dimention of the binary_add src tensor; Non-positive number for disabling binary_add
  std::unordered_map<std::string, std::string> op_attrs;

 public:
  mha_dense_bench() : kernel_bench() {}
  virtual ~mha_dense_bench() {
    for (auto data : {args.first.rt_data, args.second.rt_data})
      for (auto p : data)
        if (p != nullptr) aligned_allocator_t<char>::deallocate(const_cast<void*>(p));
  }

  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override;
  std::vector<int> get_refresh_data_idx() const override {
    return {mha_dense_io::SRC_Q, mha_dense_io::SRC_K, mha_dense_io::SRC_V, mha_dense_io::DST};
  }
  int get_workspace_idx() const override { return mha_dense_io::WORKSPACE; }
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
  void set_kernel_proxy() override {
    mha_dense_desc mha_dense_desc(args.first.op_desc);
    kp = std::make_shared<mha_dense>(mha_dense_desc);
  }
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_MHA_DENSE_HPP_
