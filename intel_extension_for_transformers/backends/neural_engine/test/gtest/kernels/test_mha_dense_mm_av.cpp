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
#include "jit_domain/jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab.hpp"
#include "singleton.hpp"
#include "unit_test_utils.hpp"

namespace jd {
using dt = data_type;
struct MMAV_param_t {
  int M;
  int K;
  int N;
  int ld_dst;
  int K_pad;  // used to calculate address of blocks of src0/src1
  data_type dst_dt;
};
inline static std::string MMAVTestParam2str(testing::TestParamInfo<MMAV_param_t> tpi) {
  std::vector<std::string> params;
  params.push_back(std::to_string(tpi.param.M));
  params.push_back(std::to_string(tpi.param.K));
  params.push_back(std::to_string(tpi.param.N));
  params.push_back("ld" + std::to_string(tpi.param.ld_dst));
  params.push_back("kpad" + std::to_string(tpi.param.K_pad));
  params.push_back(data_type_name.at(tpi.param.dst_dt));
  return join_str(params, "_");
}
class MhaDenseMMAVTest : public testing::TestWithParam<MMAV_param_t> {
 protected:
  MhaDenseMMAVTest() {}
  virtual ~MhaDenseMMAVTest() {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(MhaDenseMMAVTest, ) {
  if (!isa_available(avx512_core_amx)) {
    SPARSE_LOG(ERROR) << "AMX unavailable!";
    return;
  }
  MMAV_param_t t = testing::TestWithParam<MMAV_param_t>::GetParam();
  const auto M = t.M;
  const auto K = t.K;
  const auto N = t.N;
  const auto K_pad = t.K_pad;
  const auto ld_dst = t.ld_dst;
  if (M % 16 != 0 || N % 16 != 0 || K % 64 != 0 || K_pad % 64 != 0 || K_pad < K) {
    SPARSE_LOG(ERROR) << "Unexpected test arg";
    return;
  }
  const data_type dt_dst = t.dst_dt;
  const auto dst_size = M * ld_dst;
  const auto dst_bytes = dst_size * type_size.at(dt_dst);
  uint8_t* src0 = new uint8_t[M * K_pad];
  int8_t* src1 = new int8_t[K_pad * N];
  for (auto i = 0; i < M * K_pad; ++i) src0[i] = i % (UINT8_MAX - 1);
  for (auto i = 0; i < K_pad * N; ++i) src1[i] = i % (UINT8_MAX - 1);
  float src_scale = 2e-2f;
  float src_zp = -4;

  // get true data
  char* dst_ref = new char[dst_bytes];
  std::fill(dst_ref, dst_ref + dst_bytes, 3);
  const auto dst_ref_s8 = reinterpret_cast<int8_t*>(dst_ref);
  const auto dst_ref_u8 = reinterpret_cast<uint8_t*>(dst_ref);
  const auto dst_ref_f32 = reinterpret_cast<float*>(dst_ref);
  for (auto i = 0; i < M; i += 16)
    for (auto j = 0; j < N; j += 16)
      for (auto ii = 0; ii < 16; ++ii)
        for (auto jj = 0; jj < 16; ++jj) {
          float value = 0;
          for (int k = 0; k < K; k += 64)
            for (int kk = 0; kk < 64; kk += 4)
              for (int kkk = 0; kkk < 4; ++kkk) {
                const float l = src0[i * K_pad + ii * 64 + k * 16 + kk + kkk];
                const float r = src1[j * K_pad + jj * 4 + (k + kk) * 16 + kkk];
                value += l * r;
              }
          value = value * src_scale + src_zp;

          const auto dst_idx = (i + ii) * ld_dst + j + jj;
          switch (dt_dst) {
            case data_type::fp32:
              dst_ref_f32[dst_idx] = value;
              break;
            case data_type::u8:
              dst_ref_u8[dst_idx] = value < 0           ? 0  //
                                    : value > UINT8_MAX ? UINT8_MAX
                                                        : static_cast<uint8_t>(std::roundf(value));
              break;
            case data_type::s8:
              dst_ref_s8[dst_idx] = value < INT8_MIN   ? INT8_MIN
                                    : value > INT8_MAX ? INT8_MAX
                                                       : static_cast<int8_t>(std::roundf(value));
              break;
            default:
              SPARSE_LOG(FATAL) << "Unexpected dst type!";
              break;
          }
        }

  // prepare amx
  auto amx_config_ = Singleton<amx_tile_config_t>::GetInstance();
  if (amx_config_ == nullptr) {
    SPARSE_LOG(ERROR) << "Unable to config AMX! Skipping...";
    return;
  }
  amx_config_->amx_tile_configure(0, tile_param_t(16, 16, 64, false, 4));

  // run kernel
  char* dst = aligned_allocator_t<char>::allocate(dst_bytes);
  std::fill(dst, dst + dst_bytes, 3);
  auto jit_ker = new jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab({M, K_pad, N, ld_dst, t.dst_dt});
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());
  jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t rt_data{src0, src1, dst, K, src_scale, src_zp};
  (*jit_ker)(&rt_data);

  // release amx
  amx_config_->amx_tile_release(0);

  // check results
  switch (dt_dst) {
    case data_type::fp32:
      ASSERT_TRUE(compare_data<float>(dst, dst_size, dst_ref, dst_size, 1e-3));
      break;
    case data_type::u8:
      ASSERT_TRUE(compare_data<uint8_t>(dst, dst_size, dst_ref, dst_size, 0));
      break;
    case data_type::s8:
      ASSERT_TRUE(compare_data<int8_t>(dst, dst_size, dst_ref, dst_size, 0));
      break;
    default:
      SPARSE_LOG(FATAL) << "Unexpected dst type!";
      break;
  }

  delete[] src0;
  delete[] src1;
  delete[] dst_ref;
  aligned_allocator_t<char>::deallocate(dst);
  delete jit_ker;
}
INSTANTIATE_TEST_SUITE_P(SparseLib, MhaDenseMMAVTest,
                         ::testing::ValuesIn(std::vector<MMAV_param_t>{
                             // M K N ld_dst kpad dst_dt
                             {16, 64, 16, 16, 64, dt::fp32},         // base heaszie 16
                             {32, 64, 32, 32, 128, dt::fp32},        // base heaszie 32
                             {32, 128, 32, 32, 128, dt::fp32},       // seqlen 128
                             {32, 384, 64, 64 * 16, 384, dt::fp32},  // 16 heads
                             {32, 384, 64, 64 * 16, 384, dt::u8},    // postop
                             {32, 384, 64, 64 * 16, 384, dt::s8},    // postop
                         }),
                         MMAVTestParam2str);

}  // namespace jd
