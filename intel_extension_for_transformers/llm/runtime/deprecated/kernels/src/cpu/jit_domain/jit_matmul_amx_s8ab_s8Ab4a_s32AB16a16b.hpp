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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMUL_AMX_S8AB_S8AB4A_S32AB16A16B_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMUL_AMX_S8AB_S8AB4A_S32AB16A16B_HPP_

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <vector>

#include "kernels/amx_utils.hpp"
#include "jit_generator.hpp"
#include "src/utils.hpp"

namespace jd {

/**
 * @brief jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b does matrix multiplication to a non-transposed and a transposed matrix,
 * saving result in blocked format AB16a16b with 0 padding.
 *
 * Note that amx config should be set in advance!
 */
class jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b : public jit_generator {
 public:
  struct param_t {
    int M;
    int K;
    int N;
    int ld_src0;
    /**
     * @brief Guaranteed amx config before executing this jit kernel.
     *
     * If pre_amx_cfg fits what the kernel needs or set to nullptr, the kernel will not touch the amx config while
     * executing. Otherwise, the kernel will temporally reset the amx config while perserving it when exiting.
     *
     */
    const tile_param_t* pre_amx_cfg;
  };

  struct rt_data_t {
    const int8_t* src0;
    const int8_t* src1;
    int32_t* dst;
  };

  explicit jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b(const jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::param_t& param)
      : jit_generator(),
        M(param.M),
        K(param.K),
        N(param.N),
        ld_src0(param.ld_src0),
        pre_amx_cfg_(param.pre_amx_cfg),
        required_amx_cfg_(16, 16, std::min(64, K), false, 4) {
    SPARSE_LOG_IF(ERROR, K > ld_src0) << "K > ld_src0 which may lead to unexpected behavior!";
    SPARSE_LOG_IF(FATAL, K > 64 && K % 64 != 0) << "Currently only support K <= 64 or K as a multiple of 64";
    SPARSE_LOG_IF(FATAL, K % 4 != 0) << "Currently only support K as a multiple of 4";
    SPARSE_LOG_IF(FATAL, M % 16 != 0) << "Currently only support M as a multiple of 16";
    SPARSE_LOG_IF(FATAL, N % 16 != 0) << "Currently only support N as a multiple of 16";
  }
  virtual ~jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b() {}

 private:
  void generate() override;
  static constexpr auto TH_ = 2;
  static constexpr auto TW_ = 2;
  const int M, K, N, ld_src0;

  Xbyak::Label L_amx_cfg;
  const tile_param_t* const pre_amx_cfg_;
  const tile_param_t required_amx_cfg_;
  tileconfig_t reqired_tile_cfg_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_MATMUL_AMX_S8AB_S8AB4A_S32AB16A16B_HPP_
