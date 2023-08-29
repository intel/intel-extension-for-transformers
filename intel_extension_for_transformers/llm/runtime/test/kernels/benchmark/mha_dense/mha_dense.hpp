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

#include <vector>
#include <memory>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "kernels/exposed_enum.hpp"

namespace bench {
class mha_dense_bench : public kernel_bench {
  std::shared_ptr<mha_dense_bench> smb;

 protected:
  using io = jd::exposed_enum::mha_dense::io;
  using io_src = jd::exposed_enum::mha_dense_src::src;
  using io_dst = jd::exposed_enum::mha_dense_dst::dst;
  using io_shape = jd::exposed_enum::mha_dense_shape::shape;

 public:
  mha_dense_bench() {}
  virtual ~mha_dense_bench() {
    if (smb == nullptr) {  // for a finally derived class
      const auto using_bench_data = bench_data.op_desc.engine_kind() != jd::engine_kind::undef;
      if (using_bench_data) {
        for (auto& ctx : {bench_data.ctx_kern, bench_data.ctx_ref})
          for (auto& mems : {ctx.inputs(), ctx.outputs()})
            for (auto& mem : mems) {
              void* h;
              mem->get_handle(&h);
              if (h != nullptr) aligned_allocator_t<uint8_t, 64>::deallocate(h);
              delete mem;
            }
      } else {
        for (auto op_args : {args.first, args.second})
          for (auto rt_data : op_args.rt_data)
            if (rt_data != nullptr) aligned_allocator_t<uint8_t, 64>::deallocate(const_cast<void*>(rt_data));
      }
    }
  }
  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override { return smb->calc_flop(); }
  std::vector<int> get_refresh_data_idx() const override { return smb->get_refresh_data_idx(); }
  int get_workspace_idx() const final { return io::WORKSPACE; }
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
    const auto using_bench_data = bench_data.op_desc.engine_kind() != jd::engine_kind::undef;
    jd::mha_dense_desc mha_dense_desc(using_bench_data ? bench_data.op_desc : args.first.op_desc);
    kp = std::make_shared<jd::mha_dense>(mha_dense_desc);
  };
};
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_MHA_DENSE_HPP_
