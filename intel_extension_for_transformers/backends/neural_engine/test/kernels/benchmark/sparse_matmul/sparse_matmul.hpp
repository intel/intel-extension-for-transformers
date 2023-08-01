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
#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_SPARSE_MATMUL_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_SPARSE_MATMUL_HPP_

#include <memory>
#include <vector>
#include <utility>

#include "common_utils.hpp"
#include "benchmark_utils.hpp"
#include "kernels/spmm_types.hpp"
#include "kernels/sparse_data.hpp"

namespace bench {
class sparse_matmul_bench : public kernel_bench {
 private:
  std::shared_ptr<sparse_matmul_bench> smb;

 public:
  sparse_matmul_bench() {}
  virtual ~sparse_matmul_bench() {
    if (smb == nullptr) {  // for a finally derived class
      auto attrs = args.first.op_desc.attrs();
      const uint64_t& sparse_addr = str_to_num<uint64_t>(attrs["sparse_ptr"]);
      if (sparse_addr != 0) {
        auto sparse_data_ptr = reinterpret_cast<jd::bsc_data_t<float>*>(sparse_addr);
        delete sparse_data_ptr;
      }
      for (auto op_args : {args.first, args.second})
        for (auto rt_data : op_args.rt_data)
          if (rt_data != nullptr) {
            aligned_allocator_t<uint8_t, 64>::deallocate(const_cast<void*>(rt_data));
          }
    }
  }
  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override;
  std::vector<int> get_refresh_data_idx() const override { return std::vector<int>{jd::ssd::SRC, jd::ssd::DST}; }
  // Just like that in gtest file
  void get_true_data() override { smb->get_true_data(); }
  // Just like that in gtest file
  bool check_result() override { return smb->check_result(); }
  // Just like that in gtest file
  void gen_case() override { smb->gen_case(); }
  void set_kernel_proxy() override {
    args = smb->args;
    bench_data = smb->bench_data;
    ts_descs = smb->ts_descs;
    jd::sparse_matmul_desc spmm_desc(args.first.op_desc);
    kp = std::make_shared<jd::sparse_matmul>(spmm_desc);
  };
};
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_SPARSE_MATMUL_HPP_
