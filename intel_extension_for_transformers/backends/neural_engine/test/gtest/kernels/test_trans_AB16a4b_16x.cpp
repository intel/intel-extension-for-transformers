//  Copyright (c) 2023 Intel Corporation
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

#include <omp.h>

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "interface.hpp"
#include "jit_domain/jit_trans_AB16a4b_16x.hpp"
#include "unit_test_utils.hpp"

namespace jd {
struct test_param_t {
  int pad_n;
  bool cvt_s8u8;
  int M;
  int N;
  int ld_src;
};

inline static std::string TAB16a4bTestParam2str(testing::TestParamInfo<test_param_t> tpi) {
  auto&& str = std::to_string(tpi.param.M) + "_" + std::to_string(tpi.param.N) + "_ld" +
               std::to_string(tpi.param.ld_src) + "_padn" + std::to_string(tpi.param.pad_n);
  if (tpi.param.cvt_s8u8) str += "_cvts8u8";
  return str;
}
class TrAB16a4b_16x_Test : public testing::TestWithParam<test_param_t> {
 protected:
  TrAB16a4b_16x_Test() {}
  virtual ~TrAB16a4b_16x_Test() {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(TrAB16a4b_16x_Test, ) {
  test_param_t t = testing::TestWithParam<test_param_t>::GetParam();
  const auto M = t.M;
  const auto N = t.N;
  const auto ld_src = t.ld_src;
  SPARSE_LOG_IF(FATAL, N % 64 > t.pad_n) << "Invalid pad_n";
  const auto paded_n = (N / 64 * 64) + pad_to(N % 64, t.pad_n);
  size_t dst_size = pad_to(M, 16) * paded_n;
  std::unique_ptr<int8_t[]> src(new int8_t[M * ld_src]);
  for (auto i = 0; i < M * ld_src; ++i) src[i] = i % UINT8_MAX;

  // get true data
  const int8_t u8offset = t.cvt_s8u8 ? 128 : 0;
  std::unique_ptr<int8_t[]> dst_ref(new int8_t[dst_size]);
  std::fill_n(dst_ref.get(), 3, dst_size);
  for (auto i = 0; i < M; i += 16)
    for (auto j = 0; j < paded_n; j += 4)
      for (auto ii = 0; ii < 16; ++ii)
        for (auto jj = 0; jj < 4; ++jj)
          dst_ref[j * 16 + i * paded_n + ii * 4 + jj] =
              (i + ii < M && j + jj < N) ? (src[(i + ii) * ld_src + j + jj] + u8offset) : 0;

  // run kernel
  std::shared_ptr<int8_t> dst(aligned_allocator_t<int8_t>::allocate(dst_size),
                              [](auto ptr) { aligned_allocator_t<int8_t>::deallocate(ptr); });
  std::fill_n(dst.get(), 3, dst_size);
  std::unique_ptr<jit_trans_AB16a4b_16x> jit_ker(new jit_trans_AB16a4b_16x({t.pad_n, t.cvt_s8u8, 1}));
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());
  for (int i = 0; i < M; i += 16) {
    jit_trans_AB16a4b_16x::rt_data_t rt_data{
        /*.src = */ src.get() + i * ld_src,
        /*.dst = */ dst.get() + i * paded_n,
        /*.ld_src = */ ld_src,
        /*.M = */ std::min(M - i, 16),
        /*.N = */ N,
    };
    (*jit_ker)(&rt_data);
  }

  ASSERT_TRUE(t.cvt_s8u8 ? compare_data<uint8_t>(dst.get(), dst_size, dst_ref.get(), dst_size, 0)
                         : compare_data<int8_t>(dst.get(), dst_size, dst_ref.get(), dst_size, 0));
}

INSTANTIATE_TEST_SUITE_P(Kernels, TrAB16a4b_16x_Test,
                         ::testing::ValuesIn(std::vector<test_param_t>{
                             // pad_n, cvts8u8, M, N, ld_src
                             {64, false, 16, 64, 64},        // basic
                             {64, true, 16, 64, 64},         // cvts8u8
                             {64, false, 16, 64, 64 * 16},   // 16 as head number
                             {64, false, 16, 40, 40 * 8},    // headsize 40
                             {64, false, 77, 40, 40 * 8},    // M=77
                             {64, false, 77, 80, 80 * 8},    // M=77 hs=80
                             {64, false, 77, 160, 160 * 8},  // M=77 hs=160
                         }),
                         TAB16a4bTestParam2str);
}  // namespace jd
