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
#include "jit_domain/jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b.hpp"
#include "unit_test_utils.hpp"
#include "singleton.hpp"

namespace jd {
using dt = data_type;

using MMQK_param_t = jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::param_t;
inline static std::string MMQKTestParam2str(testing::TestParamInfo<MMQK_param_t> tpi) {
  return std::to_string(tpi.param.M) + "_" + std::to_string(tpi.param.K) + "_" + std::to_string(tpi.param.N) + "_ld" +
         std::to_string(tpi.param.ld_src0);
}
class MhaDenseMMQKTest : public testing::TestWithParam<MMQK_param_t> {
 protected:
  MhaDenseMMQKTest() {}
  virtual ~MhaDenseMMQKTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

static const tile_param_t tile_param_full(16, 16, 64, false, 4);
TEST_P(MhaDenseMMQKTest, ) {
  if (!isa_available(avx512_core_amx)) {
    SPARSE_LOG(ERROR) << "AMX unavailable!";
    return;
  }

  MMQK_param_t t = testing::TestWithParam<MMQK_param_t>::GetParam();
  const auto M = t.M;
  const auto K = t.K;
  const auto N = t.N;
  const auto ld_src0 = t.ld_src0;
  if (M % 16 != 0 || N % 16 != 0 || K % 4 != 0) {
    SPARSE_LOG(ERROR) << "Unexpected test arg";
    return;
  }

  const auto dst_size = M * N;
  int8_t* src0 = new int8_t[M * ld_src0];
  int8_t* src1 = new int8_t[K * N];
  for (auto i = 0; i < M * ld_src0; ++i) src0[i] = i % (UINT8_MAX - 1);
  for (auto i = 0; i < K * N; ++i) src0[i] = i % (UINT8_MAX - 1);

  // get true data
  int32_t* dst_ref = new int32_t[dst_size];
  std::fill(dst_ref, dst_ref + dst_size, 3);
  for (auto i = 0; i < M; i += 16)
    for (auto j = 0; j < N; j += 16)
      for (auto ii = 0; ii < 16; ++ii)
        for (auto jj = 0; jj < 16; ++jj) {
          int32_t value = 0;
          for (int k = 0; k < K; k += 4)
            for (int kk = 0; kk < 4; ++kk) {
              const int32_t l = src0[(i + ii) * ld_src0 + (k + kk)];
              const int32_t r = src1[j * K + jj * 4 + k * 16 + kk];
              value += l * r;
            }
          dst_ref[i * N + j * 16 + ii * 16 + jj] = value;
        }

  // prepare amx
  auto amx_config_ = Singleton<amx_tile_config_t>::GetInstance();
  if (amx_config_ == nullptr) {
    SPARSE_LOG(ERROR) << "Unable to config AMX! Skipping...";
    return;
  }
  amx_config_->amx_tile_configure(0, t.pre_amx_cfg != nullptr ? *t.pre_amx_cfg : tile_param_full);

  // run kernel
  int32_t* dst = aligned_allocator_t<int32_t>::allocate(dst_size);
  std::fill(dst, dst + dst_size, 3);
  auto jit_ker = new jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b({M, K, N, ld_src0, t.pre_amx_cfg});
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());
  jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t rt_data{src0, src1, dst};
  (*jit_ker)(&rt_data);

  // release amx
  amx_config_->amx_tile_release(0);

  ASSERT_TRUE(compare_data<int8_t>(dst, dst_size, dst_ref, dst_size, 0));

  delete[] src0;
  delete[] src1;
  delete[] dst_ref;
  aligned_allocator_t<int32_t>::deallocate(dst);
  delete jit_ker;
}
INSTANTIATE_TEST_SUITE_P(SparseLib, MhaDenseMMQKTest,
                         ::testing::ValuesIn(std::vector<MMQK_param_t>{
                             // M K N ld_src0 pre_amx_cfg
                             {16, 64, 16, 64, nullptr},
                             {32, 64, 32, 64, nullptr},
                             {384, 64, 384, 64, &tile_param_full},
                             {384, 64, 384, 64 * 16, &tile_param_full},
                             {16, 32, 16, 32, &tile_param_full},         // headsize 32
                             {16, 16, 16, 32, &tile_param_full},         // headsize 16
                             {384, 32, 384, 32 * 16, &tile_param_full},  // headsize 32; 16 heads
                             {32, 256, 32, 256, nullptr},                // headsize 256
                         }),
                         MMQKTestParam2str);

}  // namespace jd
