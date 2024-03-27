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

#include "jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab.hpp"

#include <memory>

#include "regs_pool.hpp"

#define GET_OFF(field) offsetof(jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t, field)

namespace jd {
void jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::generate() {
  const auto tmp_dst_size = TH_ * TW_ * BYTES_TMM;
  regs_pool rp(this, 1, {9, 19, 0}, tmp_dst_size + 64);  // 64 extra for alignment
  std::shared_ptr<void> use_loacl_label = {(inLocalLabel(), nullptr), [&](...) { outLocalLabel(); }};

  const auto reg_src0 = rp.reg<Reg64>();
  const auto reg_src1 = rp.reg<Reg64>();
  const auto reg_dst = rp.reg<Reg64>();
  const auto reg_lb_dst = rp.reg<Reg64>();
  const auto reg_stride64 = rp.reg<Reg64>();
  const auto reg_dst_tmp = rp.reg<Reg64>();
  const auto reg_tmp = rp.reg<Reg64>();
  const auto reg_iterk = rp.reg<Reg64>();
  const auto reg_maxk = rp.reg<Reg64>();
  const auto vreg_0f = rp.reg<Zmm>();
  const auto vreg_scale = rp.reg<Zmm>();
  const auto vreg_zp = rp.reg<Zmm>();
  const auto reg_tmp32 = reg_tmp.cvt32();
  if (dt_dst == data_type::u8) {
    mov(reg_tmp32, 0);  // 0L is same as 0.f in binary
    vpbroadcastd(vreg_0f, reg_tmp32);
  }

  const std::array<Tmm, 4> tmm_dst{tmm0, tmm1, tmm2, tmm3};
  const std::array<Tmm, 2> tmm_src0{tmm4, tmm5};
  const std::array<Tmm, 2> tmm_src1{tmm6, tmm7};

  mov(reg_src0, ptr[rp.p[0] + GET_OFF(src0)]);
  mov(reg_src1, ptr[rp.p[0] + GET_OFF(src1)]);
  mov(reg_dst, ptr[rp.p[0] + GET_OFF(dst)]);
  mov(reg_maxk.cvt32(), dword[rp.p[0] + GET_OFF(K)]);
  mov(reg_lb_dst, lb_dst);
  mov(reg_stride64, 64);
  vpbroadcastd(vreg_scale, dword[rp.p[0] + GET_OFF(rescale)]);
  vpbroadcastd(vreg_zp, dword[rp.p[0] + GET_OFF(zp)]);

  // align stack tmp mem
  mov(reg_dst_tmp, rsp);
  and_(reg_dst_tmp, -64);
  add(reg_dst_tmp, 64);

  for (int idx_m = 0; idx_m < M; idx_m += TH_ * 16) {
    for (int idx_n = 0; idx_n < N; idx_n += TW_ * 16) {
      // clear dst
      for (int i = 0; i < TH_; ++i)
        for (int j = 0; j < TW_; ++j) {
          tilezero(tmm_dst[i * TW_ + j]);
        }

      // for (int idx_k = 0; idx_k < K; idx_k += 64)
      xor_(reg_iterk.cvt32(), reg_iterk.cvt32());
      Xbyak::Label l_kloop;
      L(l_kloop);
      {
        // load src0
        imul(reg_tmp, reg_iterk, 16);
        lea(reg_tmp, ptr[reg_src0 + reg_tmp + idx_m * K_pad]);  // reg_src0 with src0_offset
        for (int i = 0; i < TH_; ++i) {
          if (idx_m + i * 16 < M) {
            tileloadd(tmm_src0[i], ptr[reg_tmp + reg_stride64 + i * 16 * K_pad]);
          }
        }

        // load src1 && dot prod
        imul(reg_tmp, reg_iterk, 16);
        lea(reg_tmp, ptr[reg_src1 + reg_tmp + idx_n * K_pad]);  // reg_src0 with src0_offset
        for (int i = 0; i < TH_; ++i)
          for (int j = 0; j < TW_; ++j) {
            if (idx_m + i * 16 < M && idx_n + j * 16 < N) {
              const auto TI = i * TW_ + j;
              if (i == 0) tileloadd(tmm_src1[j], ptr[reg_tmp + reg_stride64 + j * 16 * K_pad]);
              tdpbusd(tmm_dst[TI], tmm_src0[i], tmm_src1[j]);
            }
          }
      }
      lea(reg_iterk, ptr[reg_iterk + 64]);
      cmp(reg_iterk, reg_maxk);
      jl(l_kloop);

      // quant & store
      const auto vreg_post = rp.regs<Xbyak::Zmm, 16>();
      for (int i = 0; i < TH_; ++i)
        for (int j = 0; j < TW_; ++j)
          if (idx_m + i * 16 < M && idx_n + j * 16 < N) {
            const auto TI = i * TW_ + j;
            const auto dst_tmp_offset = TI * BYTES_TMM;

            tilestored(ptr[reg_dst_tmp + reg_stride64 + dst_tmp_offset], tmm_dst[TI]);
            for (int ii = 0; ii < 16; ++ii) {
              vmovdqa32(vreg_post[ii], zword[reg_dst_tmp + dst_tmp_offset + ii * BYTES_ZMM]);
              vcvtdq2ps(vreg_post[ii] | T_rn_sae, vreg_post[ii]);
              vfmadd213ps(vreg_post[ii], vreg_scale, vreg_zp);
              const auto dst_disp = (idx_m + i * 16 + ii) * lb_dst + (idx_n + j * 16) * type_size.at(dt_dst);
              switch (dt_dst) {  // move out
                case data_type::u8:
                  vmaxps(vreg_post[ii], vreg_post[ii], vreg_0f);
                  vcvtps2udq(vreg_post[ii], vreg_post[ii]);
                  vpmovusdb(ptr[reg_dst + dst_disp], vreg_post[ii]);
                  break;
                case data_type::s8:
                  vcvtps2dq(vreg_post[ii], vreg_post[ii]);
                  vpmovsdb(ptr[reg_dst + dst_disp], vreg_post[ii]);
                  break;
                case data_type::fp32:
                  vmovups(ptr[reg_dst + dst_disp], vreg_post[ii]);
                  break;
                case data_type::bf16:
                  vcvtneps2bf16(Ymm(vreg_post[ii].getIdx()), vreg_post[ii]);
                  vmovdqu16(yword[reg_dst + dst_disp], Ymm(vreg_post[ii].getIdx()));
                  break;
                default:
                  SPARSE_LOG(FATAL) << "Unexpected dst type";
                  break;
              }
            }
          }
    }
  }
}
}  // namespace jd
