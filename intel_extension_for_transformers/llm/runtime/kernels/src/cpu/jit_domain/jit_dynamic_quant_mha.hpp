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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_DYNAMIC_QUANT_MHA_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_DYNAMIC_QUANT_MHA_HPP_

#include <array>
#include <vector>

#include "kernels/amx_utils.hpp"
#include "jit_generator.hpp"
#include "regs_pool.hpp"
#include "src/utils.hpp"

namespace jd {

/**
 * @brief jit_mmsoftmax_batch_amx_s8_ab_BA16b4a_u8_16x does matrix multiplication to a non-transposed and a transposed
 * matrix, performing binary add with fp32, calculating softmax to each row, saving result in u8 plain format, and does
 * all the above several times for a batch. The 2 source matrices are DYNAMIC-PERCHANNEL-QUANTIZED and the dst matrix is
 * STATICALLY scaled by 1/UINT8_MAX.
 */
class jit_mmsoftmax_batch_amx_s8_ab_BA16b4a_u8_16x : public jit_generator {
 public:
  struct param_t {
    bool has_bias;
    int K;  // codegen-time problem size; set to non-positive value for runtime determination

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
    const float* scale_src0;  // src0_f32(i,k)=src0_s8(i,k)*scale_src0(i); of size M(16)
    const float* scale_src1;  // src1_f32(k,j)=src1_s8(k,j)*scale_src1(j); of size N
    const float* src_bias;    // aka att_mask in MHA; of size `pad_to(N, 16)`; ignored when !has_bias
    uint8_t* dst;             // of size M(16) x N
    // float* dst_scale;      // dst_scale is 1/255 implicitly
    // float* dst_zp;         // dst_zp is 0 implicitly
    float scale;  // a scale applied to the output of matmul
    int K;        // will be used as pad_to(K, 64)
    int N;        // will be used as pad_to(N, 16)
    int ld_src0;
    int ld_src1;  // step of each K * 16 block
    int ld_dst;
    int batch_size;
    size_t batchstep_src0;
    size_t batchstep_src0scale;  // set to 0 when scale reused among batches
    size_t batchstep_src1;
    size_t batchstep_src1scale;
    size_t batchstep_dst;
  };

  explicit jit_mmsoftmax_batch_amx_s8_ab_BA16b4a_u8_16x(const param_t& param)
      : jit_generator(),
        has_bias(param.has_bias),
        K(param.K),
        pre_amx_cfg_(param.pre_amx_cfg),
        required_amx_cfg_(16, 16, 64, false, 4) {}
  virtual ~jit_mmsoftmax_batch_amx_s8_ab_BA16b4a_u8_16x() {}

 private:
  void generate() override;
  static constexpr auto TH_ = 1;
  static constexpr auto TW_ = 2;
  static constexpr auto M = 16;

  /**
   * @brief perform matmul & exp & sum, store as 16x16 blocks of fp32
   *
   * @param rp register pool pointer
   * @param reg_tmpf32 a "caller saved" regitster with pointer pointer to tmp memory for the exp result
   * @param zmm_expsum std::array<Xbyak::Zmm, 16UL> registers to hold sum of exp to be reduced
   */
  void mm_exp_sum(regs_pool* const rp, const Reg64& reg_tmpf32, const std::array<Zmm, 16UL> zmm_expsum);
  /**
   * @brief perform scale with zmm_expscale and store as u8
   *
   * @param rp register pool pointer
   * @param reg_tmpf32 a "caller saved" regitster with pointer pointer to tmp memory for the exp result
   * @param zmm_expscale 16 registers with scale for each row broadcasted
   */
  void quant255_store(regs_pool* const rp, const Xbyak::Reg64& reg_tmpf32,
                      const std::array<Xbyak::Zmm, 16UL>& zmm_expscale);

  const bool has_bias;
  const int K;

  const tile_param_t* const pre_amx_cfg_;  // pointer to assumed amx cfg before executing this jit
  const tile_param_t required_amx_cfg_;    // required amx cfg for runing AMX instructions in this jit
  tileconfig_t reqired_tile_cfg_;          // corresponding palette of  required_amx_cfg_

  Xbyak::Label L_amx_cfg;
  Xbyak::Label l_255, l_log2ef, l_ln2, l_halff, l_poly_c[3];
};

class jit_mmexp_amx_s8_ab_BA16b4a_u8_16x : public jit_generator {
 public:
  struct param_t {
    bool has_bias;
    int K;  // codegen-time problem size; set to non-positive value for runtime determination

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
    const float* scale_src0;  // src0_f32(i,k)=src0_s8(i,k)*scale_src0(i); of size M(16)
    const float* scale_src1;  // src1_f32(k,j)=src1_s8(k,j)*scale_src1(j); of size N
    const float* src_bias;    // aka att_mask in MHA; of size `pad_to(N, 16)`; ignored when !has_bias
    float* dst;               // of size M(16) x N
    float* dst_sum;           // of size 16; i.e. sumexp, one for each line
    float* dst_max;           // of size 16; i.e. maxexp, one for each line
    float scale;              // a scale applied to the output of matmul
    int K;                    // will be used as pad_to(K, 64)
    int N;                    // will be used as pad_to(N, 16)
    int ld_src0;
    int ld_src1;  // step of each K * 16 block
  };

  explicit jit_mmexp_amx_s8_ab_BA16b4a_u8_16x(const param_t& param)
      : jit_generator(),
        has_bias(param.has_bias),
        K(param.K),
        pre_amx_cfg_(param.pre_amx_cfg),
        required_amx_cfg_(16, 16, 64, false, 4) {}
  virtual ~jit_mmexp_amx_s8_ab_BA16b4a_u8_16x() {}

 private:
  void generate() override;
  static constexpr auto TH_ = 1;
  static constexpr auto TW_ = 2;
  static constexpr auto M = 16;

  /**
   * @brief perform matmul & exp & sum, store as 16x16 blocks of fp32
   *
   * @param rp register pool pointer
   * @param zmm_expsum std::array<Xbyak::Zmm, 16UL>& registers to hold sum of exp to be reduced
   * @param addr_expmax Xbyak::RegExp& pointer to memory to hold max of exp to be reduced
   */
  void mm_exp_sum(regs_pool* const rp, const std::array<Zmm, 16UL>& zmm_expsum, const Xbyak::RegExp& addr_expmax);

  const bool has_bias;
  const int K;

  const tile_param_t* const pre_amx_cfg_;  // pointer to assumed amx cfg before executing this jit
  const tile_param_t required_amx_cfg_;    // required amx cfg for runing AMX instructions in this jit
  tileconfig_t reqired_tile_cfg_;          // corresponding palette of  required_amx_cfg_

  Xbyak::Label L_amx_cfg;
  Xbyak::Label l_255, l_log2ef, l_ln2, l_halff, l_poly_c[3];
};

/**
 * @brief jit_mm_batch_amx_u8s8_ab_AB16a4b_dynamic_quant_16x does matrix multiplication to a plain and a reordered
 * matrix, calculating per-channel dynq10n to each row, saving result in s8 plain format, and does all the above several
 * times for a batch. The left matrix is STATICALLY scaled by 1/UINT8_MAX and the right and dst matrices are
 * DYNAMIC-PERCHANNEL-QUANTIZED .
 */
class jit_mm_batch_amx_u8s8_ab_AB16a4b_dynamic_quant_16x : public jit_generator {
 public:
  struct param_t {
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
    const uint8_t* src0;
    const int8_t* src1;
    const float* scale_src1;  // src1_f32(k,j)=src1_s8(k,j)*scale_src1(j); of size N
    int8_t* dst;              // of size M(16) x N
    float* dst_scale;         // dst_f32(i,j)=dst_s8(i,j)*scale_src1(i); of size M(16)
    // float* dst_zp;         // dst_zp reserved
    int K;  // will be used as pad_to(K, 64)
    int N;  // will be used as pad_to(N, 16)
    int ld_src0;
    int ld_src1;  // step of each K * 16 block
    int ld_dst;
    int batch_size;
    size_t batchstep_src0;
    size_t batchstep_src1;
    size_t batchstep_src1scale;
    size_t batchstep_dst;
  };

  explicit jit_mm_batch_amx_u8s8_ab_AB16a4b_dynamic_quant_16x(const param_t& param)
      : jit_generator(), pre_amx_cfg_(param.pre_amx_cfg), required_amx_cfg_(16, 16, 64, false, 4) {}
  virtual ~jit_mm_batch_amx_u8s8_ab_AB16a4b_dynamic_quant_16x() {}

 private:
  void generate() override;
  static constexpr auto TH_ = 1;
  static constexpr auto TW_ = 3;
  static constexpr auto M = 16;

  /**
   * @brief perform 16xkxn matmul & dequant & absmax, store as 16x16 blocks of fp32
   *
   * @param rp register pool pointer
   * @param reg_tmpf32 a "caller saved" regitster with pointer pointer to tmp memory for the exp result
   * @param zmm_absmax registers with absmax of dst to be updated
   */
  void mm_absmax(regs_pool* const rp, const Xbyak::Reg64& reg_tmpf32, std::array<Xbyak::Zmm, 16UL> zmm_absmax);
  /**
   * @brief perform scale with zmm_rcpscale and store as s8
   *
   * @param rp register pool pointer
   * @param reg_tmpf32 a "caller saved" regitster with pointer pointer to tmp memory for the f32 result
   * @param zmm_rcpscale 16 registers with reciprocal scale for each row broadcasted
   */
  void quant_store(regs_pool* const rp, const Xbyak::Reg64& reg_tmpf32,
                   const std::array<Xbyak::Zmm, 16UL>& zmm_rcpscale);

  const tile_param_t* const pre_amx_cfg_;
  const tile_param_t required_amx_cfg_;
  tileconfig_t reqired_tile_cfg_;

  Xbyak::Label L_amx_cfg;
  Xbyak::Label l_127f, l_rcp255, l_float_epsilon;
};

class jit_scale_mm_amx_u8s8_ab_BA16b_16x : public jit_generator {
 public:
  struct param_t {
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
    const float* src0;
    const float* prescale_src0;  // of size 16; u8 = static_cast<u8>(src0 * prescale_src0)
    const float* scale_src0;     // of size 16; f32 = scale_src0 * u8
    const int8_t* src1;
    const float* scale_src1;  // src1_f32(k,j)=src1_s8(k,j)*scale_src1(j); of size N
    float* dst;               // of size M(16) x pad(N, 16*TW)
    float* absmax_dst;        // of size M(16)
    // float* dst_zp;         // dst_zp reserved
    int K;        // will be used as pad_to(K, 64)
    int N;        // will be used as pad_to(N, 16)
    int ld_src1;  // step of each K * 16 block
    int ld_dst;   // leading dim of dst (in elements)
  };

  explicit jit_scale_mm_amx_u8s8_ab_BA16b_16x(const param_t& param)
      : jit_generator(), pre_amx_cfg_(param.pre_amx_cfg), required_amx_cfg_(16, 16, 64, false, 4) {}
  virtual ~jit_scale_mm_amx_u8s8_ab_BA16b_16x() {}

 private:
  void generate() override;
  static constexpr auto TH_ = 1;
  static constexpr auto TW_ = 3;
  static constexpr auto M = 16;

  /**
   * @brief perform 16xkxn matmul & dequant & absmax, store as 16x16 blocks of fp32
   *
   * @param rp register pool pointer
   * @param zmm_absmax registers with absmax of dst to be updated
   */
  void mm_absmax(regs_pool* const rp, std::array<Xbyak::Zmm, 16UL> zmm_absmax);

  const tile_param_t* const pre_amx_cfg_;
  const tile_param_t required_amx_cfg_;
  tileconfig_t reqired_tile_cfg_;

  Xbyak::Label L_amx_cfg;
  Xbyak::Label l_127f, l_float_epsilon;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_DYNAMIC_QUANT_MHA_HPP_
