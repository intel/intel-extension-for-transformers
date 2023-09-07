//  Copyright (c) 2022 Intel Corporation
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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_TRANSPOSE_MHA_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_TRANSPOSE_MHA_HPP_

#include <memory>
#include <vector>

#include "src/cpu/cpu_isa.hpp"
#include "src/cpu/jit_domain/jit_mm_exp_vnni_mxkx48.hpp"
#include "src/cpu/jit_domain/jit_seq_cpy_2x8x8.hpp"
#include "src/cpu/jit_domain/jit_seq_cpy_48x4.hpp"
#include "src/cpu/jit_domain/jit_transpose_mha.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "operator_desc.hpp"
#include "kernels/transpose_mha_types.hpp"
#include "src/utils.hpp"

namespace jd {
class transpose_mha_k_t;

class transpose_mha_kd_t : public kernel_desc_t {
 public:
  explicit transpose_mha_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::transpose_mha), op_desc_(op_desc) {}

  virtual ~transpose_mha_kd_t() {}

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(transpose_mha_k_t, transpose_mha_kd_t);

 public:
  const operator_desc& get_operator_desc() const override { return op_desc_; }

 private:
  operator_desc op_desc_;
};

class transpose_mha_k_t : public kernel_t {
 public:
  using kd_t = transpose_mha_kd_t;
  explicit transpose_mha_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) {
    TransCopy8x8_1B.reset(new TransposeCopy8x8_1B_kernel());
    SeqVnniCopy32.reset(new SeqCopy_1B_avx512_Nx4_Temp());
    SeqVnniWCopy32.reset(new SeqCopy_1B_avx512_Nx2_Temp());
    MHA_amx_step1_k32.reset(new MHA_s8s8s8_row_amx_32x32_batchk_binary_exp(32));
    MHA_amx_step1_k64.reset(new MHA_s8s8s8_row_amx_32x32_batchk_binary_exp(64));
    MHA_vnni_step1.reset(new MHA_s8s8s8_row_vnni_8x32_batchk_binary_exp());
    MHA_amx_step2.reset(new MHA_norm_quantize_reorder_prescale_packed(8, 32));
    MHA_vnni_step2.reset(new MHA_norm_quantize_reorder_vnniw_prescale_packed(8, 32));
    MHA_amx_step3_ktile32.reset(new MHA_Matmul_s8u8u8_amx_32x32(32));
    MHA_amx_step3_ktile64.reset(new MHA_Matmul_s8u8u8_amx_32x32(64));
    MHA_vnni_step3.reset(new MHA_Matmul_s8u8u8_vnni_word_8x32());

    kernel_set.push_back(TransCopy8x8_1B);
    kernel_set.push_back(SeqVnniCopy32);
    kernel_set.push_back(SeqVnniWCopy32);
    kernel_set.push_back(MHA_amx_step1_k32);
    kernel_set.push_back(MHA_amx_step1_k64);
    kernel_set.push_back(MHA_vnni_step1);
    kernel_set.push_back(MHA_amx_step2);
    kernel_set.push_back(MHA_vnni_step2);
    kernel_set.push_back(MHA_amx_step3_ktile32);
    kernel_set.push_back(MHA_amx_step3_ktile64);
    kernel_set.push_back(MHA_vnni_step3);
  }
  virtual ~transpose_mha_k_t() {
    //  aligned_free(mTmp);
  }
  // Delete move constructor and move operator
  transpose_mha_k_t(transpose_mha_k_t&& other) = delete;
  transpose_mha_k_t& operator=(transpose_mha_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  transpose_mha_k_t(const transpose_mha_k_t& other) = delete;
  transpose_mha_k_t& operator=(const transpose_mha_k_t& other) = delete;

  enum ker_idx {
    trans_cpy,
    vnni_cpy_Nx4,
    vnni_cpy_Nx2,
    mha_amx_step1_k32,
    mha_amx_step1_k64,
    mha_vnni_step1,
    mha_amx_step2,
    mha_vnni_step2,
    mha_amx_step3_ktile32,
    mha_amx_step3_ktile64,
    mha_vnni_step3
  };
  enum impl {
    amx,
    vnni_w,
    vnni_b,
    undef,
  };

  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;
  bool execute_vnnib(const std::vector<const void*>& rt_data) const;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  // uint8_t* mTmp;
  impl impl_ = impl::undef;
  static constexpr int Size2M = 1 << 21;

  std::shared_ptr<TransposeCopy8x8_1B_kernel> TransCopy8x8_1B;
  std::shared_ptr<SeqCopy_1B_avx512_Nx4_Temp> SeqVnniCopy32;
  std::shared_ptr<SeqCopy_1B_avx512_Nx2_Temp> SeqVnniWCopy32;
  std::shared_ptr<MHA_s8s8s8_row_amx_32x32_batchk_binary_exp> MHA_amx_step1_k32;
  std::shared_ptr<MHA_s8s8s8_row_amx_32x32_batchk_binary_exp> MHA_amx_step1_k64;
  std::shared_ptr<MHA_s8s8s8_row_vnni_8x32_batchk_binary_exp> MHA_vnni_step1;
  std::shared_ptr<MHA_norm_quantize_reorder_prescale_packed> MHA_amx_step2;
  std::shared_ptr<MHA_norm_quantize_reorder_vnniw_prescale_packed> MHA_vnni_step2;
  std::shared_ptr<MHA_Matmul_s8u8u8_amx_32x32> MHA_amx_step3_ktile32;
  std::shared_ptr<MHA_Matmul_s8u8u8_amx_32x32> MHA_amx_step3_ktile64;
  std::shared_ptr<MHA_Matmul_s8u8u8_vnni_word_8x32> MHA_vnni_step3;
  std::vector<std::shared_ptr<MHA_kernel>> kernel_set;
  std::unique_ptr<jit_seq_cpy_2x8x8> ker_seq_cpy_k_;
  std::unique_ptr<jit_seq_cpy_48x4> ker_seq_cpy_q_;
  std::unique_ptr<jit_mm_exp_vnni_mxkx48_t> ker_kxq_;
  std::unique_ptr<MHA_norm_quantize_reorder_vnnib_prescale_packed> ker_scale_trans;
  std::unique_ptr<MHA_Matmul_s8u8u8_vnni_byte_8x48> ker_vxa_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_TRANSPOSE_MHA_HPP_
