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

#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_TRANSPOSE_MATMUL_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_TRANSPOSE_MATMUL_HPP_

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "benchmark_utils.hpp"
#include "kernels/matmul_types.hpp"
namespace jd {

class transpose_matmul_bench : public kernel_bench {
 private:
  std::shared_ptr<transpose_matmul_bench> smb;

 public:
  transpose_matmul_bench() {}
  virtual ~transpose_matmul_bench() {
    if (smb == nullptr) {  // for a finally derived class
      for (auto rt_data : {args.first.rt_data, args.second.rt_data}) {
        for (size_t idx = 0; idx < rt_data.size(); idx++) {
          if (rt_data[idx] != nullptr) {
            if (idx <= ssd::SRC2) {
              aligned_allocator_t<uint8_t, 64>::deallocate(const_cast<void*>(rt_data[idx]));
            } else {
              delete reinterpret_cast<const float*>(rt_data[idx]);
            }
          }
        }
      }
    }
  }

  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override { return smb->calc_flop(); };
  std::vector<int> get_refresh_data_idx() const override { return smb->get_refresh_data_idx(); }
  // Just like that in gtest file
  void get_true_data() final;
  // Just like that in gtest file
  bool check_result() override { return smb->check_result(); }
  // Just like that in gtest file
  void gen_case() override { smb->gen_case(); }
  void set_kernel_proxy() override {
    args = smb->args;
    ts_descs = smb->ts_descs;
    transpose_matmul_desc spmm_desc(args.first.op_desc);
    kp = std::make_shared<transpose_matmul>(spmm_desc);
  };
};
}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_TRANSPOSE_MATMUL_HPP_
