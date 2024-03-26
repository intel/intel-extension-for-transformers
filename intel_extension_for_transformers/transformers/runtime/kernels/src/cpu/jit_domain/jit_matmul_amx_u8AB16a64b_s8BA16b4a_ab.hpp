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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMUL_AMX_U8AB16A64B_S8BA16B4A_AB_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMUL_AMX_U8AB16A64B_S8BA16B4A_AB_HPP_

#include <glog/logging.h>

#include <array>
#include <vector>

#include "jit_eltwise_injector.hpp"
#include "jit_generator.hpp"
#include "src/utils.hpp"

namespace jd {

/**
 * @brief jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab does matrix multiplication to a non-transposed and a transposed matrix,
 * saving result in plain format of f32 or any other data output by the last postop.
 *
 * Note that amx config should be set in advance!
 */
class jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab : public jit_generator {
 public:
  struct param_t {
    int M;
    int K_pad;  // used to calculate address of blocks of src0/src1
    int N;
    int ld_dst;
    data_type dst_dt;
  };

  struct rt_data_t {
    const uint8_t* src0;
    const int8_t* src1;
    void* dst;
    int K;
    float rescale, zp;  // out = value * rescale + zp
  };

  explicit jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab(const jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::param_t& param)
      : jit_generator(),
        M(param.M),
        K_pad(param.K_pad),
        N(param.N),
        dt_dst(param.dst_dt),
        ld_dst(param.ld_dst),
        lb_dst(ld_dst * type_size.at(dt_dst)) {
    SPARSE_LOG_IF(FATAL, is_all_of({data_type::u8, data_type::s8, data_type::fp32, data_type::bf16},
                                   [dst = dt_dst](auto t) { return t != dst; }))
        << "Unexpected dt_dst";
    SPARSE_LOG_IF(ERROR, N > ld_dst) << "N > ld_dst which may lead to unexpected behavior!";
    SPARSE_LOG_IF(FATAL, K_pad % 64 != 0) << "Currently only support K as a multiple of 64";
    SPARSE_LOG_IF(FATAL, M % 16 != 0) << "Currently only support M as a multiple of 16";
    SPARSE_LOG_IF(FATAL, N % 16 != 0) << "Currently only support N as a multiple of 16";
  }
  virtual ~jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab() {}

 private:
  void generate() override;
  static constexpr auto TH_ = 2;
  static constexpr auto TW_ = 2;
  const int M, K_pad, N;
  const data_type dt_dst;
  const int ld_dst;  // dst leading dimension
  const int lb_dst;  // #bytes on dst leading dimension
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMUL_AMX_U8AB16A64B_S8BA16B4A_AB_HPP_
