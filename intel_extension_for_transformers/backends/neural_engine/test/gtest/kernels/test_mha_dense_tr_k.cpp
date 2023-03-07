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
#include "jit_domain/jit_trans_AB16a4b.hpp"
#include "unit_test_utils.hpp"

namespace jd {
using dt = data_type;

using TAB16a4b_param_t = jit_trans_AB16a4b::param_t;
inline static std::string TAB16a4bTestParam2str(testing::TestParamInfo<TAB16a4b_param_t> tpi) {
  return std::to_string(tpi.param.M) + "_" + std::to_string(tpi.param.N) + "_ld" + std::to_string(tpi.param.ld_src) +
         "_padn" + std::to_string(tpi.param.pad_n);
}
class MhaDenseTAB16a4bTest : public testing::TestWithParam<TAB16a4b_param_t> {
 protected:
  MhaDenseTAB16a4bTest() {}
  virtual ~MhaDenseTAB16a4bTest() {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(MhaDenseTAB16a4bTest, ) {
  TAB16a4b_param_t t = testing::TestWithParam<TAB16a4b_param_t>::GetParam();
  const auto M = t.M;
  const auto N = t.N;
  const auto ld_src = t.ld_src;
  const auto pad_n = t.pad_n;
  size_t dst_size = pad_to(M, 16) * pad_n;
  int8_t* src = new int8_t[M * ld_src];
  for (auto i = 0; i < M * ld_src; ++i) src[i] = i % UINT8_MAX;

  // get true data
  int8_t* dst_ref = new int8_t[dst_size];
  memset(dst_ref, 3, dst_size);
  for (auto i = 0; i < M; i += 16)
    for (auto j = 0; j < pad_n; j += 4)
      for (auto ii = 0; ii < 16; ++ii)
        for (auto jj = 0; jj < 4; ++jj)
          dst_ref[j * 16 + i * pad_n + ii * 4 + jj] =
              (i + ii < M && j + jj < N) ? (src[(i + ii) * ld_src + j + jj]) : 0;

  // run kernel
  int8_t* dst = aligned_allocator_t<int8_t>::allocate(dst_size);
  memset(dst, 3, dst_size);
  auto jit_ker = new jit_trans_AB16a4b(jit_trans_AB16a4b::param_t{M, N, ld_src, pad_n});
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());
  jit_trans_AB16a4b::rt_data_t rt_data{src, dst};
  (*jit_ker)(&rt_data);

  ASSERT_TRUE(compare_data<int8_t>(dst, dst_size, dst_ref, dst_size, 0));

  delete[] src;
  delete[] dst_ref;
  aligned_allocator_t<int8_t>::deallocate(dst);
  delete jit_ker;
}
INSTANTIATE_TEST_SUITE_P(SparseLib, MhaDenseTAB16a4bTest,
                         ::testing::ValuesIn(std::vector<TAB16a4b_param_t>{
                             // M, N, ld_src, pad_n
                             {16, 64, 64, 64},            // basic
                             {16, 64, 64 * 16, 64},       // 16 as head number
                             {384, 64, 64, 64},           // single headsize64
                             {16, 16, 16, 16},            // smallest
                             {32, 16, 32, 32},            // pad 16 to 32
                             {384, 32, 32 * 16 * 3, 32},  // 16 as head number; 3 as merged qkv
                             {384, 32, 32 * 16 * 3, 64},  // pad to 64
                         }),
                         TAB16a4bTestParam2str);

}  // namespace jd
