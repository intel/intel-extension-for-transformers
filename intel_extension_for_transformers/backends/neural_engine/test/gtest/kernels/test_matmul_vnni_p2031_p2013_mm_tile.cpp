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
#include "jit_domain/jit_matmul_vnni_8xkx48.hpp"

#define OMP_NUM_THREADS "OMP_NUM_THREADS"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;

// test mm_8xkx48
struct mm_param_t {
  dim_t K;
  dim_t ld_dst;
  uint8_t bias_lshift;
  bool binary_add;
  std::vector<postop_attr> postop_list;
};

inline static std::string mmTestParam2str(testing::TestParamInfo<mm_param_t> tpi) {
  std::vector<std::string> params;
  params.push_back(std::to_string(tpi.param.K));
  params.push_back(std::to_string(tpi.param.ld_dst));
  params.push_back("lsh" + std::to_string(tpi.param.bias_lshift));
  if (tpi.param.binary_add) params.push_back("badd");
  for (postop_attr& p_attr : tpi.param.postop_list) {
    params.push_back("post" + std::string(postop_alg_name.at(p_attr.op_alg)));
    params.push_back(std::string(data_type_name.at(p_attr.dt)));
    params.push_back(num2id(p_attr.alpha));
    params.push_back(num2id(p_attr.beta));
    params.push_back(num2id(p_attr.scale));
  }
  return join_str(params, "_");
}
class MMVNNIP2031P2013MMTileTest : public testing::TestWithParam<mm_param_t> {
 protected:
  MMVNNIP2031P2013MMTileTest() {}
  virtual ~MMVNNIP2031P2013MMTileTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

template <typename dt_dst>
inline void exec_mmtile(jit_matmul_vnni_8xkx48_t* ker, const uint8_t* src0, const int8_t* src1, const int32_t* bias,
                        const float* src_b0, dt_dst* dst) {
  jit_matmul_vnni_8xkx48_t::rt_data_t<dt_dst> rt_data;
  rt_data.src0 = src0;
  rt_data.src1 = src1;
  rt_data.bias = bias;
  rt_data.src_b0 = src_b0;
  rt_data.dst = dst;
  (*ker)(&rt_data);
}

TEST_P(MMVNNIP2031P2013MMTileTest, ) {
  const mm_param_t t = testing::TestWithParam<mm_param_t>::GetParam();
  const data_type dst_type = t.postop_list.size() != 0 ? t.postop_list.back().dt : data_type::fp32;
  const dim_t M = 8, K = t.K, N = 48;
  const size_t dst_buf_size = M * t.ld_dst;
  const size_t dst_buf_bytes = dst_buf_size * type_size.at(dst_type);
  SPARSE_LOG_IF(FATAL, t.ld_dst < N) << "ld_dst must be gretaer then N";
  uint8_t* const src0 = new uint8_t[M * K];
  init_vector(src0, M * K, 0, 10);
  int8_t* const src1 = new int8_t[K * N];
  init_vector(src1, N * K, -10, 10);
  float* src_b0 = nullptr;
  if (t.binary_add) {
    src_b0 = new float[dst_buf_size];
    init_vector(src_b0, dst_buf_size, -10, 10);
  }

  int32_t* const bias = new int32_t[N];
  init_vector(bias, N, 0, 100);
  float scale[1];
  init_vector(scale, 1);

  // get true data
  void* const dst0_ref = new char[dst_buf_bytes];
  memset(dst0_ref, 3, dst_buf_bytes);  // something special to check if the kernel keep the additional memory untouched
  const auto dst0_ref_u8 = reinterpret_cast<uint8_t*>(dst0_ref);
  const auto dst0_ref_s8 = reinterpret_cast<int8_t*>(dst0_ref);
  const auto dst0_ref_f32 = reinterpret_cast<float*>(dst0_ref);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float value = -(bias[j] << t.bias_lshift);
      for (int k = 0; k < K; ++k) {
        float l = src0[i * 4 + k % 4 + (k / 4) * 4 * M];
        float r = src1[j * 4 + k % 4 + (k / 4) * 4 * N];
        value += l * r;
      }
      value *= scale[0];
      if (t.binary_add) value += src_b0[i * t.ld_dst + j];
      value = apply_postop_list(value, t.postop_list);
      switch (dst_type) {
        case data_type::u8:
          dst0_ref_u8[i * t.ld_dst + j] = static_cast<uint8_t>(value);
          break;
        case data_type::s8:
          dst0_ref_s8[i * t.ld_dst + j] = static_cast<int8_t>(value);
          break;
        case data_type::fp32:
          dst0_ref_f32[i * t.ld_dst + j] = value;
          break;
        default:
          SPARSE_LOG(FATAL) << "Unexpected dst type!";
          break;
      }
    }
  }

  // run kernel
  void* const dst0 = new char[dst_buf_bytes];
  memset(dst0, 3, dst_buf_bytes);  // something special to check if the kernel keep the additional memory untouched
  auto jit_ker = new jit_matmul_vnni_8xkx48_t(jit_matmul_vnni_8xkx48_t::param_t{
      t.K,
      t.ld_dst,
      scale[0],
      t.bias_lshift,
      t.binary_add,
      dst_type,
      t.postop_list,
      M,
      N,
  });
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());

  switch (dst_type) {
    case data_type::u8:
      exec_mmtile<uint8_t>(jit_ker, src0, src1, bias, src_b0, reinterpret_cast<uint8_t*>(dst0));
      ASSERT_TRUE(compare_data<uint8_t>(dst0, dst_buf_size, dst0_ref, dst_buf_size, 8e-3));
      break;
    case data_type::s8:
      exec_mmtile<int8_t>(jit_ker, src0, src1, bias, src_b0, reinterpret_cast<int8_t*>(dst0));
      ASSERT_TRUE(compare_data<int8_t>(dst0, dst_buf_size, dst0_ref, dst_buf_size, 8e-3));
      break;
    case data_type::fp32:
      exec_mmtile<float>(jit_ker, src0, src1, bias, src_b0, reinterpret_cast<float*>(dst0));
      ASSERT_TRUE(compare_data<float>(dst0, dst_buf_size, dst0_ref, dst_buf_size, 8e-3));
      break;
    default:
      SPARSE_LOG(FATAL) << "Unexpected dst type!";
      break;
  }
  delete[] src0;
  delete[] src1;
  if (src_b0 != nullptr) delete[] src_b0;
  delete[] bias;
  delete[] reinterpret_cast<char*>(dst0);
  delete[] reinterpret_cast<char*>(dst0_ref);
  delete jit_ker;
}
INSTANTIATE_TEST_SUITE_P(
    SparseLib, MMVNNIP2031P2013MMTileTest,
    ::testing::ValuesIn(std::vector<mm_param_t>{
        {16, 48, 2, false, {}},                                                                   // basic
        {64, 64, 7, true, {}},                                                                    // ld_dst != N
        {64, 65, 7, true, {}},                                                                    // ld_dst != N
        {64, 64, 7, true, {{dt::u8, postop_type::eltwise, postop_alg::quantize, 0.0, 0.0, 10}}},  // post quant s8
        {64, 64, 7, true, {{dt::u8, postop_type::eltwise, postop_alg::quantize, 128, 0.0, 10}}},  // post quant u8
    }),
    mmTestParam2str);

}  // namespace jd
