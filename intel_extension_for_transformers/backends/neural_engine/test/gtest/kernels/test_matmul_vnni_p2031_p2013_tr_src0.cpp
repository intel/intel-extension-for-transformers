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

#include <omp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <exception>
#include "interface.hpp"
#include "gtest/gtest.h"
#include "unit_test_utils.hpp"
#include "kernels/matmul_types.hpp"
#include "kernels/matmul_ref.hpp"
#include "jit_domain/jit_seq_cpy_2x8x8.hpp"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;

// test seq_cpy_2x8x8
struct seqcpy_param_t {
  dim_t M;
  dim_t N;
  uint8_t offset;
};

inline static std::string to_string(testing::TestParamInfo<seqcpy_param_t> tpi) {
  return std::to_string(tpi.param.M) + "_" + std::to_string(tpi.param.N) + "_off" + std::to_string(tpi.param.offset);
}

class MMVNNIP2031P2013SEQCPYATest : public testing::TestWithParam<seqcpy_param_t> {
 protected:
  MMVNNIP2031P2013SEQCPYATest() {}
  virtual ~MMVNNIP2031P2013SEQCPYATest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MMVNNIP2031P2013SEQCPYATest, ) {
  seqcpy_param_t t = testing::TestWithParam<seqcpy_param_t>::GetParam();
  const int M = t.M, N = t.N;
  const int ld_src = N;
  if (M % 8 != 0) {
    SPARSE_LOG(WARNING) << "M must be a multiple of 8";
    return;
  }
  size_t dst_size = M * (ceil_div(N, 8) * 8);
  int8_t* src = new int8_t[M * N];
  for (auto i = 0; i < M * N; ++i) src[i] = i % UINT8_MAX - INT8_MIN;
  uint8_t* dst = reinterpret_cast<uint8_t*>(aligned_alloc(64, dst_size));
  memset(dst, 3, dst_size);

  // get true data
  uint8_t* dst_ref = new uint8_t[dst_size];
  memset(dst_ref, 3, dst_size);
  for (auto j = 0; j < N; j += 8)
    for (auto i = 0; i < M; i += 4)
      for (auto ii = 0; ii < 4; ++ii)
        for (auto jj = 0; jj < 8; ++jj)
          dst_ref[j * M + i * 8 + jj * 4 + ii] =
              (i + ii >= M || j + jj >= N) ? 0 : (src[(i + ii) * ld_src + j + jj] + t.offset);

  // run kernel
  auto jit_ker = new jit_seq_cpy_2x8x8(jit_seq_cpy_2x8x8::param_t{t.offset});
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());

  const int ld_dst = jit_seq_cpy_2x8x8::dst_step(M);
  for (dim_t i = 0; i < M; i += 8) {
    jit_seq_cpy_2x8x8::rt_data_t rt_data;
    rt_data.src = src + i * N;
    rt_data.dst = dst + i * 8;
    rt_data.N = N;
    rt_data.ld_src = ld_src;
    rt_data.ld_dst = ld_dst;
    (*jit_ker)(&rt_data);
  }

  ASSERT_TRUE(compare_data<uint8_t>(dst, dst_size, dst_ref, dst_size, 0));

  delete[] src;
  delete[] dst_ref;
  free(dst);
  delete jit_ker;
}

INSTANTIATE_TEST_SUITE_P(SparseLib, MMVNNIP2031P2013SEQCPYATest,
                         ::testing::ValuesIn(std::vector<seqcpy_param_t>{
                             {64, 64, 0},
                             {64, 1, 0},
                             {64, 15, 0},
                             {64, 63, 0},
                             {64, 65, 0},
                             {64, 64, 128},
                             {64, 1, 128},
                             {64, 15, 128},
                             {64, 63, 128},
                             {64, 65, 131},
                         }),
                         to_string);
}  // namespace jd
