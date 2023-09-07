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
#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_AMX_BF16_X16_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_SPMM_AMX_BF16_X16_HPP_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "interface.hpp"
#include "benchmark_utils.hpp"
#include "sparse_matmul.hpp"
#include "kernels/sparse_data.hpp"
#include "common_utils.hpp"

#define SPMM_AMX_BF16_X16_ARG_NUM 7

namespace bench {
class spmm_amx_bf16_x16_bench : public sparse_matmul_bench {
 private:
  int64_t M, K, N;
  float sparse_ratio;
  int64_t micro_bs, micro_oc;
  bool bf16_out;

 public:
  spmm_amx_bf16_x16_bench() {}
  virtual ~spmm_amx_bf16_x16_bench() {
    auto mutable_attrs_ptr = const_cast<std::unordered_map<std::string, std::string>*>(&args.first.op_desc.attrs());
    const auto sparse_addr = str_to_num<uint64_t>((*mutable_attrs_ptr)["sparse_ptr"]);
    const auto all_bsr_data = reinterpret_cast<std::vector<jd::bsr_data_t<jd::bfloat16_t>*>*>(sparse_addr);
    for (auto sparse_data : *all_bsr_data) {
      delete sparse_data;
    }
    delete all_bsr_data;
    (*mutable_attrs_ptr)["sparse_ptr"] = "";  // clear it so that parent class won't double free
  }
  bench_res_t set_config(int argc, char** argv) override;
  // Just like that in gtest file
  void get_true_data() override;
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
};
template <typename T>
void prepare_sparse_data_spmm_amx_bf16_x16(T* weight, dim_t N, dim_t K, dim_t n_blksize, dim_t k_blksize, float ratio);
std::pair<const void*, const void*> make_data_obj_spmm_amx_bf16_x16(const jd::data_type& tensor_dt, dim_t rows,
                                                                    dim_t cols, dim_t index, float ratio = 0.9,
                                                                    const std::vector<float>& ranges = {-.5, .5});
}  // namespace bench
#endif  // SPARSE_LIB_USE_AMX
