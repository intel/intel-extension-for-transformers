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
#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_MATMUL_VNNI_NOPERM_P2031_P1302_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_MATMUL_VNNI_NOPERM_P2031_P1302_HPP_
#include <string>
#include <unordered_map>
#include <vector>

#include "benchmark_utils.hpp"
#include "interface.hpp"
#include "transpose_matmul.hpp"

#define MATMUL_VNNI_NOPERM_P2031_P1302_ARG_NUM 5
namespace bench {
class matmul_vnni_noperm_p2031_p1302_bench : public transpose_matmul_bench {
 private:
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t bs0;
  int64_t bs1;
  std::unordered_map<std::string, std::string> op_attrs = {};
 public:
  matmul_vnni_noperm_p2031_p1302_bench() {}
  virtual ~matmul_vnni_noperm_p2031_p1302_bench() {}
  bench_res_t set_config(int argc, char** argv) override;
  double calc_flop() const override { return static_cast<double>(M) * N * K * bs0 * bs1 * 2; };
  // Just like that in gtest file
  bool check_result() override;
  // Just like that in gtest file
  void gen_case() override;
  std::vector<int> get_refresh_data_idx() const override { return std::vector<int>{io::SRC0, io::SRC1, io::DST0}; }
};
}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_MATMUL_VNNI_NOPERM_P2031_P1302_HPP_
