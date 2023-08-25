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

#include "jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b.hpp"

#include <memory>

#include "regs_pool.hpp"

#define GET_OFF(field) offsetof(jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t, field)

namespace jd {
void jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::generate() {
  bool need_cfg_amx = pre_amx_cfg_ != nullptr && *pre_amx_cfg_ != required_amx_cfg_;

  std::shared_ptr<void> use_loacl_label = {(inLocalLabel(), nullptr), [&](...) { outLocalLabel(); }};
  {
    regs_pool rp(this, 1, {5, 0, 0}, need_cfg_amx ? sizeof(tileconfig_t) : 0);
    std::shared_ptr<void> local_cfg;
    if (need_cfg_amx) {  // create a local amx config environment
      local_cfg = {(sttilecfg(ptr[rsp]), ldtilecfg(ptr[rip + L_amx_cfg]), nullptr), [&](...) { ldtilecfg(ptr[rsp]); }};
    }

    const auto reg_src0 = rp.reg<Reg64>();
    const auto reg_src1 = rp.reg<Reg64>();
    const auto reg_dst = rp.reg<Reg64>();
    const auto reg_ld_src0 = rp.reg<Reg64>();
    const auto reg_stride64 = rp.reg<Reg64>();
    const std::array<Tmm, 4> tmm_dst{tmm0, tmm1, tmm2, tmm3};
    const std::array<Tmm, 2> tmm_src0{tmm4, tmm5};
    const std::array<Tmm, 2> tmm_src1{tmm6, tmm7};

    mov(reg_src0, ptr[rp.p[0] + GET_OFF(src0)]);
    mov(reg_src1, ptr[rp.p[0] + GET_OFF(src1)]);
    mov(reg_dst, ptr[rp.p[0] + GET_OFF(dst)]);
    mov(reg_ld_src0, ld_src0);
    mov(reg_stride64, 64);

    for (int idx_m = 0; idx_m < M; idx_m += TH_ * 16) {
      for (int idx_n = 0; idx_n < N; idx_n += TH_ * 16) {
        // clear
        for (int i = 0; i < TH_; ++i)
          for (int j = 0; j < TW_; ++j)
            if (idx_m + i * 16 < M && idx_n + j * 16 < N) tilezero(tmm_dst[i * TW_ + j]);

        // dp (loop k)
        for (int k = 0; k < K; k += 64)
          for (int i = 0; i < TH_; ++i)
            for (int j = 0; j < TW_; ++j)
              if (idx_m + i * 16 < M && idx_n + j * 16 < N) {
                const auto src0_offset = (idx_m + i * 16) * ld_src0 + k;
                if (j == 0) tileloadd(tmm_src0[i], ptr[reg_src0 + reg_ld_src0 + src0_offset]);

                const auto src1_offset = (idx_n + j * 16) * K + k * 16;
                if (i == 0) tileloadd(tmm_src1[j], ptr[reg_src1 + reg_stride64 + src1_offset]);

                tdpbssd(tmm_dst[i * TW_ + j], tmm_src0[i], tmm_src1[j]);

                // store
                const auto dst_offset = (idx_m + i * 16) * N + (idx_n + j * 16) * 16;
                if (k + 64 >= K)
                  tilestored(ptr[reg_dst + reg_stride64 + dst_offset * sizeof(int32_t)], tmm_dst[i * TW_ + j]);
              }
      }
    }
  }  // end of call stack

  if (need_cfg_amx) {
    configure_tiles(required_amx_cfg_, &reqired_tile_cfg_);
    align(sizeof(tileconfig_t));
    L(L_amx_cfg);
    db(reinterpret_cast<const uint8_t*>(&reqired_tile_cfg_), sizeof(tileconfig_t));
  }
}
}  // namespace jd
