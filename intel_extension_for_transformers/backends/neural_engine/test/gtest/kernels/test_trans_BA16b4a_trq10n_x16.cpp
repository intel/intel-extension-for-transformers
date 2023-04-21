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

#include <exception>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>

#include "gtest/gtest.h"
#include "interface.hpp"
#include "jit_domain/jit_trans_BA16b4a_trq10n_x16.hpp"
#include "unit_test_utils.hpp"
namespace jd {
struct test_param_t {
  int M;
  int N;
  int ld_src;
};

inline static std::string TBA16b4aTestParam2str(testing::TestParamInfo<test_param_t> tpi) {
  return std::to_string(tpi.param.M) + "_" + std::to_string(tpi.param.N) + "_ld" + std::to_string(tpi.param.ld_src);
}
class TrBA16b4a_x16_Test : public testing::TestWithParam<test_param_t> {
 protected:
  TrBA16b4a_x16_Test() {}
  virtual ~TrBA16b4a_x16_Test() {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(TrBA16b4a_x16_Test, ) {
  test_param_t t = testing::TestWithParam<test_param_t>::GetParam();
  const auto M = t.M;
  const auto N = t.N;
  const auto ld_src = t.ld_src;
  const auto pad_m = pad_to(M, 64);
  const auto pad_n = pad_to(N, 16);
  size_t dst_size = pad_to(M, 64) * pad_n;

  // prepare data
  const auto src = std::unique_ptr<int8_t[]>(new int8_t[M * ld_src]);
  for (auto i = 0; i < M * ld_src; ++i) src[i] = (i * 3 - 1) % UINT8_MAX;
  const auto src_scale = std::unique_ptr<float[]>(new float[M]);
  init_vector(src_scale.get(), M, 1, 1);

  // get true data
  const auto dst_ref = std::unique_ptr<int8_t[]>(new int8_t[dst_size]);
  const auto dst_scale_ref = std::unique_ptr<float[]>(new float[pad_n]);
  std::fill_n(dst_ref.get(), dst_size, 0);
  std::fill_n(dst_scale_ref.get(), pad_n, 0);
  const auto src_f32 = std::unique_ptr<float[]>(new float[M * N]);
  const auto src_f32_absmax = std::unique_ptr<float[]>(new float[pad_n]);
  std::fill_n(src_f32_absmax.get(), pad_n, 0.f);
  for (auto i = 0; i < M; ++i)
    for (auto j = 0; j < N; ++j) {
      const float value = src[i * ld_src + j] * src_scale[i];
      src_f32[i * N + j] = value;
      src_f32_absmax[j] = std::max(src_f32_absmax[j], std::abs(value));
    }
  for (auto j = 0; j < N; ++j) src_f32_absmax[j] = src_f32_absmax[j] / INT8_MAX;
  auto& src_f32_trscale_ref = src_f32_absmax;
  for (auto j = 0; j < pad_n; j += 16)
    for (auto i = 0; i < pad_m; i += 4)
      for (auto ii = 0; ii < 4; ++ii)
        for (auto jj = 0; jj < 16; ++jj)
          dst_ref[j * pad_m + i * 16 + jj * 4 + ii] =
              (i + ii < M && j + jj < N) ? (std::roundf(src_f32[(i + ii) * N + j + jj] / src_f32_trscale_ref[j + jj]))
                                         : 0;

  // run kernel
  int8_t* dst = aligned_allocator_t<int8_t>::allocate(dst_size);
  float* dst_scale = aligned_allocator_t<float>::allocate(pad_n);
  std::fill_n(dst, dst_size, 3);  // fill dst with some random data
  auto jit_ker = std::unique_ptr<jit_trans_BA16b4a_trq10n_x16>(new jit_trans_BA16b4a_trq10n_x16());
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());

  for (int j = 0; j < N; j += 16) {
    jit_trans_BA16b4a_trq10n_x16::rt_data_t rt_data{
        /*.src = */ src.get() + j,
        /*.dst = */ dst + j * pad_m,
        /*.src_scale = */ src_scale.get(),
        /*.dst_scale = */ dst_scale + j,
        /*.ld_src = */ ld_src,
        /*.M = */ M,
        /*.N = */ std::min(N - j, 16),
    };
    (*jit_ker)(&rt_data);
  }
  const auto pad4to64 = pad_m - pad_to(M, 4);
  if (pad4to64) {
    const auto dst_paded = dst + pad_to(M, 4) * 16;
    for (int j = 0; j < N; j += 16) std::fill_n(dst_paded + pad_m * j, pad4to64 * 16, 0);
  }

  ASSERT_TRUE(compare_data<int8_t>(dst, dst_size, dst_ref.get(), dst_size, 4e-3));
  ASSERT_TRUE(compare_data<float>(dst_scale, N, src_f32_trscale_ref.get(), N, 1e-3));

  aligned_allocator_t<int8_t>::deallocate(dst);
  aligned_allocator_t<float>::deallocate(dst_scale);
}

INSTANTIATE_TEST_SUITE_P(Kernels, TrBA16b4a_x16_Test,
                         ::testing::ValuesIn(std::vector<test_param_t>{
                             // M, N, ld_src
                             {4, 64, 64},
                             {4, 13, 16},
                             {4, 16, 16},
                             {4, 19, 32},
                             {4, 32, 64 * 16},
                             {4, 35, 64 * 16},
                             {4, 53, 64 * 16},
                             {4, 64, 64 * 16},    // headsize 64 * 16 heads
                             {16, 64, 64 * 16},   // seqlen=16
                             {384, 32, 32 * 16},  // headszie 32
                         }),
                         TBA16b4aTestParam2str);
}  // namespace jd
