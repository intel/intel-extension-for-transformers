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

#include "jit_matmul_vnni_Ba4b_Ab4a_ba.hpp"

#define GET_OFF(field) offsetof(ssd::matmul_u8_data_t, field)

namespace jd {
inline std::vector<Xbyak::Ymm> jit_matmul_vnni_Ba4b_Ab4a_ba_t::get_Ymm(int start, int num) const {
  std::vector<Xbyak::Ymm> result(num);
  for (int i = 0; i < num; ++i) {
    result[i] = Xbyak::Ymm(start + i);
  }
  return result;
}

Xbyak::Zmm jit_matmul_vnni_Ba4b_Ab4a_ba_t::dst_tile_Vmm(int i, int j) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS;
  const int& alloc_idx = alloc_start - j * TH_ - i;
  return Xbyak::Zmm(alloc_idx);
}
Xbyak::Zmm jit_matmul_vnni_Ba4b_Ab4a_ba_t::src0_tile_Vmm(int i) { return Xbyak::Zmm(i); }

void jit_matmul_vnni_Ba4b_Ab4a_ba_t::calc_THxTKxTW() {}

void jit_matmul_vnni_Ba4b_Ab4a_ba_t::generate() {
  inLocalLabel();  // use local label for multiple instance
  preamble();

  mov(reg_src0, ptr[parambase + GET_OFF(src0)]);
  mov(reg_src1, ptr[parambase + GET_OFF(src1)]);
  mov(reg_ksize, param_.K);
  xor_(reg_iterk, reg_iterk);
  assert(param_.K % (TK_ * VNNI_ADJ) == 0);

  for (int j = 0; j < TW_; j++)
    for (int i = 0; i < TH_; i++) {
      vxorps(dst_tile_Vmm(i, j), dst_tile_Vmm(i, j));
    }

  int src0_group = 0;
  int src1_group = 0;

  // load src0 for the first run
  for (int i = 0; i < TH_; ++i) {
    vmovups(src0_tile_Vmm(i * 2 + src0_group), zword[reg_src0 + i * BYTES_ZMM]);
  }
  // load src1 for the first run
  vpbroadcastd(vreg_temp[src1_group], dword[reg_src1]);

  L(kloop);

  for (int ik = 0; ik < TK_; ++ik) {
    // load src0 for next run
    for (int i = 0; i < TH_; ++i) {
      vmovups(src0_tile_Vmm(i * 2 + (1 - src0_group)), zword[reg_src0 + ((ik + 1) * TH_ + i) * BYTES_ZMM]);
    }

    for (int j = 0; j < TW_; ++j) {
      // load src1 for next run
      vpbroadcastd(vreg_temp[1 - src1_group], dword[reg_src1 + (j + 1) * VNNI_ADJ + ik * TW_ * VNNI_ADJ]);

      // inner prod
      for (int i = 0; i < TH_; ++i) {
        vpdpbusds(dst_tile_Vmm(i, j), src0_tile_Vmm(i * 2 + src0_group), vreg_temp[src1_group]);
      }
      src1_group = 1 - src1_group;
    }
    // update src0_group state
    src0_group = 1 - src0_group;
  }

  add(reg_src0, TK_ * TH_ * BYTES_ZMM);
  add(reg_src1, TK_ * TW_ * VNNI_ADJ * dsize_src1);
  add(reg_iterk, TK_ * VNNI_ADJ);
  cmp(reg_iterk, reg_ksize);  // k iteration variable
  jb(kloop);

  // postop & store results
  const Xbyak::Zmm& vreg_scale = zmm0;
  const Xbyak::Zmm& vreg_zp = zmm1;
  const Xbyak::Zmm& vreg_zero = zmm2;
  mov(reg_tmp, qword[parambase + GET_OFF(scale)]);
  vbroadcastss(vreg_scale, dword[reg_tmp]);  // move in scale.
  mov(reg_tmp, qword[parambase + GET_OFF(zp)]);
  vbroadcastss(vreg_zp, dword[reg_tmp]);    // move in zp.
  vpxord(vreg_zero, vreg_zero, vreg_zero);  // 0 in fp32 is 0x0
  // auto& reg_ld_dst = reg_tmp;
  mov(reg_dst, ptr[parambase + GET_OFF(dst)]);
  // mov(reg_ld_dst, ld_dst);

  for (int j = 0; j < TW_; ++j) {
    for (int i = 0; i < TH_; ++i) {
      // TODO(zhe1wang): replace with eltwise injector supporting runtime args
      vcvtdq2ps(dst_tile_Vmm(i, j) | T_rn_sae, dst_tile_Vmm(i, j));       // s32->fp32
      vfmadd132ps(dst_tile_Vmm(i, j), vreg_zp, vreg_scale);               // multiplies scaler and add zp
      vcmpleps(reg_k1, vreg_zero, dst_tile_Vmm(i, j));                    // mask of everything greater than 0
      vcvtps2udq(dst_tile_Vmm(i, j) | T_z | reg_k1, dst_tile_Vmm(i, j));  // fp32->u32
      vpmovusdb(ptr[reg_dst + ld_dst * j + VNNI_GROUPS * dsize_dst * i], dst_tile_Vmm(i, j));  // store result
    }
  }

  postamble();
  outLocalLabel();  // end of local label
}
}  // namespace jd
