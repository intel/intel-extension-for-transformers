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
#include "jit_domain/jit_seq_cpy_48x4.hpp"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;

// test seq_cpy_48x4

struct seqcpy_param_t {
  dim_t M;
  dim_t N;
  bool sum;
  int32_t sum_pad_val;
};

inline static std::string to_string(testing::TestParamInfo<seqcpy_param_t> tpi) {
  std::string repr = std::to_string(tpi.param.M) + "_" + std::to_string(tpi.param.N);
  if (tpi.param.sum) repr += "_sum";
  if (tpi.param.sum && tpi.param.sum_pad_val != 0) repr += "_pad" + std::to_string(tpi.param.sum_pad_val);
  return repr;
}

class MMVNNIP2031P2013SEQCPYBTest : public testing::TestWithParam<seqcpy_param_t> {
 protected:
  MMVNNIP2031P2013SEQCPYBTest() {}
  virtual ~MMVNNIP2031P2013SEQCPYBTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MMVNNIP2031P2013SEQCPYBTest, ) {
  seqcpy_param_t t = testing::TestWithParam<seqcpy_param_t>::GetParam();
  const int M = t.M, N = t.N;
  const int ld_src = N;
  const int pad_n = pad_to(N, 48);
  const int max_sum_size = pad_n * 16;  // up to 16 threads
  if (M % 4 != 0) {
    SPARSE_LOG(WARNING) << "M must be a multiple of 4! Test case skipped.";
    return;
  }
  const size_t dst_size = M * pad_n;
  uint8_t* const src = new uint8_t[M * N];
  for (auto i = 0; i < M * N; ++i) src[i] = i % UINT8_MAX;

  // get true data
  uint8_t* const dst_ref = new uint8_t[dst_size];
  std::fill_n(dst_ref, dst_size, -3);
#pragma omp parallel for collapse(3)
  for (auto j = 0; j < N; j += 48)
    for (auto i = 0; i < M; i += 4)
      for (auto ii = 0; ii < 4; ++ii)
#pragma omp simd
        for (auto jj = 0; jj < 48; ++jj) {
          auto value = (i + ii >= M || j + jj >= N) ? 0 : src[(i + ii) * ld_src + j + jj];
          dst_ref[j * M + i * 48 + jj * 4 + ii] = value;
        }

  // run kernel
  const n_thread_t with_n_thread(1);
  uint8_t* const dst = aligned_allocator_t<uint8_t>::allocate(dst_size);
  int32_t* const dst_sum = aligned_allocator_t<int32_t>::allocate(max_sum_size);
  std::fill_n(dst, dst_size, -3);
  std::fill_n(dst_sum, max_sum_size, -1);
  auto jit_ker = new jit_seq_cpy_48x4({t.sum, true, t.sum_pad_val});
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());
  int real_num_threads = 0;
  bool sum_append = false;
  const int ld_dst = jit_seq_cpy_48x4::dst_step(M);
#pragma omp parallel for firstprivate(sum_append)
  for (dim_t i = 0; i < M; i += 4) {
    jit_seq_cpy_48x4 ::rt_data_t rt_data;
    rt_data.src = src + i * N;
    rt_data.dst = dst + i * 48;
    rt_data.dst_sum = dst_sum + pad_n * omp_get_thread_num();
    rt_data.sum_append = sum_append;
    rt_data.N = N;
    rt_data.ld_src = ld_src;
    rt_data.ld_dst = ld_dst;
    (*jit_ker)(&rt_data);
    sum_append = true;
    if (i == 0) real_num_threads = omp_get_num_threads();
  }
  real_num_threads = std::min(real_num_threads, ceil_div(M, 4));

  ASSERT_TRUE(compare_data<uint8_t>(dst, dst_size, dst_ref, dst_size, 0));

  if (t.sum) {
    // collaps sum
    for (dim_t i = 1; i < real_num_threads; ++i) {
#pragma omp simd
      for (dim_t j = 0; j < N; ++j) {
        dst_sum[j] += dst_sum[i * pad_n + j];
      }
    }

    //  sum true data
    int32_t* dst_sum_ref = aligned_allocator_t<int32_t>::allocate(pad_n, true);
#pragma omp parallel for
    for (auto j = 0; j < N; ++j)
      for (auto i = 0; i < M; ++i) {
        dst_sum_ref[j] += src[i * N + j];
      }
    for (auto j = N; j < pad_n; ++j) dst_sum_ref[j] = t.sum_pad_val;

    ASSERT_TRUE(compare_data<int32_t>(dst_sum, pad_n, dst_sum_ref, pad_n, 0));
    aligned_allocator_t<int32_t>::deallocate(dst_sum_ref);
  }

  delete[] src;
  delete[] dst_ref;
  aligned_allocator_t<uint8_t>::deallocate(dst);
  aligned_allocator_t<int32_t>::deallocate(dst_sum);
  delete jit_ker;
}

INSTANTIATE_TEST_SUITE_P(SparseLib, MMVNNIP2031P2013SEQCPYBTest,
                         ::testing::ValuesIn(std::vector<seqcpy_param_t>{
                             // M N sum sum_pad_val
                             {4, 48},
                             {128, 768, true},
                             {64, 48, true},
                             {4, 1, true},
                             {4, 17, true, 100},
                             {4, 33, true, 100},
                             {64, 47, true},
                             {64, 47, true, 100},
                             {64, 49, true, INT32_MAX},
                             {64, 65, true, 100},
                         }),
                         to_string);
}  // namespace jd
