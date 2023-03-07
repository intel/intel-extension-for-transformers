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
#include "jit_domain/jit_softmax_Ab16a.hpp"
#include "unit_test_utils.hpp"

namespace jd {
using dt = data_type;

using softmax_param_t = jit_softmax_Ab16a::params;
inline static std::string SoftmaxTestParam2str(testing::TestParamInfo<softmax_param_t> tpi) {
  std::vector<std::string> params;
  params.push_back(std::to_string(tpi.param.att_tail));
  params.push_back(std::to_string(tpi.param.sl_pad64));
  params.push_back(tpi.param.output_type);
  return join_str(params, "_");
}
class MhaDenseSoftmaxTest : public testing::TestWithParam<softmax_param_t> {
 protected:
  MhaDenseSoftmaxTest() {}
  virtual ~MhaDenseSoftmaxTest() {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(MhaDenseSoftmaxTest, ) {
  if (!isa_available(avx512_core_vbmi) || !isa_available(avx512_core_fp16)) {
    SPARSE_LOG(ERROR) << "VBMI or FP16 unavailable!";
    return;
  }

  softmax_param_t t = testing::TestWithParam<softmax_param_t>::GetParam();
  const auto M = 16;
  const auto N = t.att_tail == 0 ? t.sl_pad64 : t.sl_pad64 - 64 + t.att_tail;
  const auto dt_dst = t.output_type == "u8" ? dt::u8 : dt::s8;

  const auto dst_size = M * t.sl_pad64;
  int32_t* src = new int32_t[dst_size];
  for (auto i = 0; i < dst_size; ++i) src[i] = i % (UINT8_MAX - 1);
  float src_QK_rescale_ = .2f;
  float src_softmax_rescale_ = 127.8f;

  // get true data
  char* const dst_ref = new char[dst_size];
  const auto dst_ref_u8 = reinterpret_cast<uint8_t*>(dst_ref);
  const auto dst_ref_s8 = reinterpret_cast<int8_t*>(dst_ref);
  std::fill(dst_ref, dst_ref + dst_size, 3);
  float* tmp_ref = new float[dst_size];
  for (auto i = 0; i < M; ++i) {
    int32_t max_i = INT32_MIN;
    for (auto j = 0; j < N; j += 16)
      for (auto jj = 0; jj < 16; ++jj)  //
        max_i = std::max(max_i, src[i * 16 + j * M + jj]);
    float sum_exp = 0;
    for (auto j = 0; j < N; j += 16)
      for (auto jj = 0; jj < 16; ++jj) {
        float exp_ij = std::exp((src[i * 16 + j * M + jj] - max_i) * src_QK_rescale_);
        sum_exp += exp_ij;
        tmp_ref[i * t.sl_pad64 + j + jj] = exp_ij;
      }
    for (auto j = 0; j < N; j += 64)
      for (auto jj = 0; jj < 64; ++jj) {
        int dst_idx = i * 64 + j * M + jj;
        float value = tmp_ref[i * t.sl_pad64 + j + jj] / sum_exp * src_softmax_rescale_;
        switch (dt_dst) {
          case dt::s8:
            dst_ref_s8[dst_idx] = value > INT8_MAX   ? INT8_MAX
                                  : value < INT8_MIN ? INT8_MIN
                                                     : static_cast<int8_t>(std::roundf(value));
            break;
          case dt::u8:
            dst_ref_u8[dst_idx] = value > UINT8_MAX ? UINT8_MAX : value < 0 ? 0 : static_cast<uint8_t>(value + .5f);
            break;
          default:
            SPARSE_LOG(FATAL) << "Unexpected dst type!";
        }
      }
  }

  // run kernel
  char* dst = aligned_allocator_t<char>::allocate(dst_size);
  std::fill(dst, dst + dst_size, 3);
  auto jit_ker = new jit_softmax_Ab16a(t);
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());
  jit_softmax_Ab16a::rt_data_t rt_data{
      src,
      reinterpret_cast<uint8_t*>(dst),
      t.sl_pad64 / 16,
      fp32_to_fp16(src_softmax_rescale_),
      .src_badd = nullptr,
      .ld_badd = 0,
      src_QK_rescale_,
  };
  (*jit_ker)(&rt_data);

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

  delete[] src;
  delete[] tmp_ref;
  delete[] dst_ref;
  aligned_allocator_t<char>::deallocate(dst);
  delete jit_ker;
}
INSTANTIATE_TEST_SUITE_P(SparseLib, MhaDenseSoftmaxTest,
                         ::testing::ValuesIn(std::vector<softmax_param_t>{
                             // att_tail sl_pad64 has_badd output_type
                             softmax_param_t{0, 64, false, "s8"},  // basic
                         }),
                         SoftmaxTestParam2str);

}  // namespace jd
