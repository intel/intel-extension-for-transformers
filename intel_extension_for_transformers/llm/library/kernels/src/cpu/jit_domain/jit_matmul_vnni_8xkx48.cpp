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

#include "jit_matmul_vnni_8xkx48.hpp"

#define GET_OFF(field) offsetof(jit_matmul_vnni_8xkx48_t::rt_data_t<void>, field)

namespace jd {
jit_matmul_vnni_8xkx48_t::jit_matmul_vnni_8xkx48_t(const param_t& param)
    : jit_generator(),
      param_(param),
      M_(param.dst_M),
      N_(param.dst_N),
      TH_(M_),
      TW_(ceil_div(N_, 16L)),
      dsize_dst(get_data_size(param.dt_dst)),
      lb_src_b0(param.ld_dst * dsize_src_b0),
      lb_dst(param.ld_dst * dsize_dst) {
  SPARSE_LOG_IF(FATAL, param.K < 0 || param.K % 4 != 0) << "K must be multiple of 4";
  SPARSE_LOG_IF(FATAL,
                param_.dt_dst != (param_.postop_attrs.size() == 0 ? data_type::fp32 : param_.postop_attrs.back().dt))
      << "Output type must match the last postop if applied!";
  SPARSE_LOG_IF(FATAL, M_ > 8 || N_ > 48) << "Output dimention too large!";
  SPARSE_LOG_IF(FATAL, M_ == 0 || N_ == 0) << "Output dimention can not be zero!";
  eltwise_inj_.eltwise_injector_init(this, param_.postop_attrs);
}

Xbyak::Zmm jit_matmul_vnni_8xkx48_t::dst_tile_Vmm(int i, int j) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS;
  const int& alloc_idx = alloc_start - j * TH_ - i;
  return Xbyak::Zmm(alloc_idx);
}
Xbyak::Zmm jit_matmul_vnni_8xkx48_t::TW_Vmm(int j) { return Xbyak::Zmm(j); }

void jit_matmul_vnni_8xkx48_t::generate() {
  inLocalLabel();  // use local label for multiple instance
  preamble();

  // move in bias
  mov(reg_tmp, ptr[parambase + GET_OFF(bias)]);
  if (param_.bias_lshift != 0) vxorps(Xmm(vreg_temp), Xmm(vreg_temp), Xmm(vreg_temp));
  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      if (i == 0) {
        if (param_.bias_lshift != 0) {
          vpslld(dst_tile_Vmm(i, j), zword[reg_tmp + j * BYTES_ZMM], param_.bias_lshift);
          vpsubd(dst_tile_Vmm(i, j), vreg_temp, dst_tile_Vmm(i, j));
        } else {
          vmovdqu32(dst_tile_Vmm(i, j), ptr[reg_tmp + j * BYTES_ZMM]);
        }
      } else {
        vmovdqu32(dst_tile_Vmm(i, j), dst_tile_Vmm(0, j));
      }
    }
  }

  // inner prod
  mov(reg_src0, ptr[parambase + GET_OFF(src0)]);
  mov(reg_src1, ptr[parambase + GET_OFF(src1)]);
  mov(reg_ksize, param_.K);
  xor_(reg_iterk, reg_iterk);
  L(kloop);
  for (int j = 0; j < TW_; ++j) {  // move in src1
    vmovdqu8(TW_Vmm(j), ptr[reg_src1 + j * BYTES_ZMM]);
  }
  for (int i = 0; i < TH_; ++i) {
    vpbroadcastd(vreg_temp, ptr[reg_src0 + i * 4]);  // move in src0
    for (int j = 0; j < TW_; ++j) {
      vpdpbusds(dst_tile_Vmm(i, j), vreg_temp, TW_Vmm(j));
    }
  }

  lea(reg_src0, ptr[reg_src0 + 8 * 4]);          // src0 is paded with 8
  lea(reg_src1, ptr[reg_src1 + 3 * BYTES_ZMM]);  // src1 is paded with 48
  lea(reg_iterk, ptr[reg_iterk + 4]);
  cmp(reg_iterk, reg_ksize);  // k iteration variable
  jb(kloop);

  // scale & binary_add(f32) & postpo
  if (N_ % VEC != 0) {
    mov(reg_tmp.cvt16(), (1 << (N_ % VEC)) - 1);
    kmovw(mask_n, reg_tmp.cvt16());
  }
  mov(reg_dst, ptr[parambase + GET_OFF(dst)]);
  auto reg_scale = reg_src0.cvt32();
  auto& reg_src_b0 = reg_src1;
  mov(reg_scale, bit_cast<uint32_t, float>(param_.scale));
  const auto& vreg_scale = vreg_temp;
  vpbroadcastd(vreg_scale, reg_scale);
  if (param_.binary_add) mov(reg_src_b0, ptr[parambase + GET_OFF(src_b0)]);
  for (int i = 0; i < TH_; ++i) {
    for (int j = 0; j < TW_; ++j) {
      vcvtdq2ps(dst_tile_Vmm(i, j) | T_rn_sae, dst_tile_Vmm(i, j));  // s32 => f32
      if (param_.binary_add) {                                       // f32 x scale + badd
        vfmadd213ps(dst_tile_Vmm(i, j), vreg_scale, zword[reg_src_b0 + i * lb_src_b0 + j * BYTES_ZMM]);
      } else {
        vmulps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j), vreg_scale);
      }
      if (param_.postop_attrs.size() != 0)  // postop
        eltwise_inj_.vector_compute(dst_tile_Vmm(i, j), param_.postop_attrs);
      const bool needs_mask = N_ % VEC != 0 && j == TW_ - 1;
      const Opmask& mask = needs_mask ? mask_n : k0;
      switch (param_.dt_dst) {  // move out
        case data_type::u8:
          vpmovusdb(ptr[reg_dst + i * lb_dst + j * VEC] | mask, dst_tile_Vmm(i, j));
          break;
        case data_type::s8:
          vpmovsdb(ptr[reg_dst + i * lb_dst + j * VEC] | mask, dst_tile_Vmm(i, j));
          break;
        case data_type::fp32:
          vmovups(ptr[reg_dst + i * lb_dst + j * BYTES_ZMM] | mask, dst_tile_Vmm(i, j));
          break;
        default:
          SPARSE_LOG(FATAL) << "Unexpected dst type";
          break;
      }
    }
  }
  postamble();
  outLocalLabel();  // end of local label
  eltwise_inj_.prepare_table();
}
}  // namespace jd
