//  Copyright (c) 2021 Intel Corporation
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

#include <map>
#include <string>

#include "amx_utils.hpp"
#include "cpu_isa.hpp"
#include "gtest/gtest.h"
#include "jit_domain/jit_trans_BA16b4a.hpp"
#include "unit_test_utils.hpp"

namespace jd {
using dt = data_type;

using TBA16b4a_param_t = jit_trans_BA16b4a::params;
inline static std::string TBA16b4aTestParam2str(testing::TestParamInfo<TBA16b4a_param_t> tpi) {
  return std::to_string(tpi.param.M) + "_" + std::to_string(tpi.param.N) + "_ld" + std::to_string(tpi.param.ld_src);
}
class MhaDenseTBA16b4aTest : public testing::TestWithParam<TBA16b4a_param_t> {
 protected:
  MhaDenseTBA16b4aTest() {}
  virtual ~MhaDenseTBA16b4aTest() {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(MhaDenseTBA16b4aTest, ) {
  TBA16b4a_param_t t = testing::TestWithParam<TBA16b4a_param_t>::GetParam();
  const auto M = t.M;
  const auto N = t.N;
  const auto ld_src = t.ld_src;
  size_t dst_size = pad_to(std::max(4, M), 4) * pad_to(N, 16);
  int8_t* src = new int8_t[std::max(4, M) * ld_src];
  for (auto i = 0; i < M * ld_src; ++i) src[i] = i % UINT8_MAX;  // let it overflow to negative number

  // get true data
  int8_t* dst_ref = new int8_t[dst_size];
  std::fill(dst_ref, dst_ref + dst_size, 3);
  for (auto j = 0; j < N; j += 16)
    for (auto i = 0; i < std::max(4, M); i += 4)
      for (auto ii = 0; ii < 4; ++ii)
        for (auto jj = 0; jj < 16; ++jj)
          dst_ref[j * M + i * 16 + jj * 4 + ii] = (i + ii < M && j + jj < N) ? (src[(i + ii) * ld_src + j + jj]) : 0;

  // run kernel
  int8_t* dst = aligned_allocator_t<int8_t>::allocate(dst_size);
  std::fill(dst, dst + dst_size, 3);
  auto jit_ker = new jit_trans_BA16b4a({M, N, ld_src});
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());
  jit_trans_BA16b4a::rt_data_t rt_data{src, dst, jit_trans_BA16b4a::dst_stride(M)};
  (*jit_ker)(&rt_data);

  ASSERT_TRUE(compare_data<int8_t>(dst, dst_size, dst_ref, dst_size, 0));

  delete[] src;
  delete[] dst_ref;
  aligned_allocator_t<uint8_t>::deallocate(dst);
  delete jit_ker;
}
INSTANTIATE_TEST_SUITE_P(SparseLib, MhaDenseTBA16b4aTest,
                         ::testing::ValuesIn(std::vector<TBA16b4a_param_t>{
                             // M N ld_src
                             {4, 64, 64},
                             {384, 64, 64 * 16},  // headsize 64 * 16 heads
                             {384, 32, 32 * 16},  // headszie 32
                             {0, 64, 64 * 16},    // zero 4x64
                         }),
                         TBA16b4aTestParam2str);

}  // namespace jd
