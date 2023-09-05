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

#include "jit_matmul_vnni_noperm_p2031_p1302.hpp"

#define GET_OFF(field) offsetof(ssd::matmul_u8_data_t, field)

namespace jd {
inline std::vector<Xbyak::Ymm> jit_matmul_vnni_noperm_p2031_p1302_t::get_Ymm(int start, int num) const {
  std::vector<Xbyak::Ymm> result(num);
  for (int i = 0; i < num; ++i) {
    result[i] = Xbyak::Ymm(start + i);
  }
  return result;
}

Xbyak::Zmm jit_matmul_vnni_noperm_p2031_p1302_t::dst_tile_Vmm(int j) {
  const int& alloc_start = VREG_NUMS - 1 - USED_VREGS;
  const int& alloc_idx = alloc_start - j;
  return Xbyak::Zmm(alloc_idx);
}

void jit_matmul_vnni_noperm_p2031_p1302_t::transpose8_ps(const Xbyak::Ymm mat[8], const Xbyak::Ymm tmp[8]) {
  vunpcklps(tmp[0], mat[0], mat[1]);
  vunpcklps(tmp[1], mat[2], mat[3]);
  vunpckhps(tmp[2], mat[0], mat[1]);
  vunpcklps(tmp[3], mat[4], mat[5]);
  vunpcklps(mat[0], mat[6], mat[7]);

  vshufps(tmp[4], tmp[0], tmp[1], 0x4e);
  vblendps(mat[1], tmp[0], tmp[4], 0xcc);
  vshufps(tmp[0], tmp[3], mat[0], 0x4e);

  vunpckhps(tmp[5], mat[2], mat[3]);
  vblendps(mat[2], tmp[3], tmp[0], 0xCC);
  vblendps(mat[3], tmp[4], tmp[1], 0xCC);
  vperm2f128(tmp[4], mat[1], mat[2], 0x20);
  // tmp[4] ready

  vunpckhps(tmp[7], mat[4], mat[5]);
  vblendps(mat[4], tmp[0], mat[0], 0xcc);
  vunpckhps(tmp[6], mat[6], mat[7]);
  vperm2f128(mat[7], mat[3], mat[4], 0x20);
  // mat[7] ready

  vshufps(mat[5], tmp[2], tmp[5], 0x4e);
  vblendps(mat[6], mat[5], tmp[5], 0xcc);
  vshufps(tmp[5], tmp[7], tmp[6], 0x4e);
  vblendps(tmp[2], tmp[2], mat[5], 0xcc);
  vblendps(tmp[7], tmp[7], tmp[5], 0xcc);
  vperm2f128(tmp[0], tmp[2], tmp[7], 0x020);
  // tmp[0] ready

  vblendps(tmp[6], tmp[5], tmp[6], 0xcc);
  vperm2f128(tmp[5], mat[6], tmp[6], 0x20);
  // tmp[5] ready

  vperm2f128(tmp[1], mat[1], mat[2], 0x31);
  // tmp[1] ready
  vperm2f128(tmp[3], mat[3], mat[4], 0x31);
  // tmp[3] ready

  vperm2f128(tmp[7], tmp[2], tmp[7], 0x31);
  // tmp[7] ready
  vperm2f128(tmp[6], mat[6], tmp[6], 0x31);
  // tmp[6] ready
  vmovaps(mat[1], mat[7]);
  vmovaps(mat[7], tmp[6]);
  vmovaps(mat[6], tmp[7]);
  vmovaps(mat[5], tmp[3]);
  vmovaps(mat[4], tmp[1]);
  vmovaps(mat[3], tmp[5]);
  vmovaps(mat[2], tmp[0]);
  vmovaps(mat[0], tmp[4]);
}

void jit_matmul_vnni_noperm_p2031_p1302_t::calc_THxTKxTW() {
  const auto src_Ymm = get_Ymm(0, 8), tmp_Ymm = get_Ymm(8, 8);
  constexpr int dim_transpose = 8;

  for (int j = 0; j < TK_ * VNNI_ADJ; j += dim_transpose * VNNI_ADJ) {
    for (int i = 0; i < TH_ * VNNI_GROUPS; i += dim_transpose) {
      for (int ii = 0; ii < dim_transpose; ii++) {
        vmovups(src_Ymm[ii], ptr[reg_src0 + j + ld_src0 * (ii + i)]);
      }
      transpose8_ps(src_Ymm.data(), tmp_Ymm.data());
      for (int ii = 0; ii < dim_transpose; ii++) {
        vmovups(ptr[rsp + start_transpose_src0 + (j / 4 + ii) * 64 + i * 4], src_Ymm[ii]);
      }
    }
    for (int i = 0; i < TW_; i += dim_transpose) {
      for (int ii = 0; ii < dim_transpose; ii++) {
        vmovups(src_Ymm[ii], ptr[reg_src1 + j + ld_src1 * (i + ii)]);
      }
      transpose8_ps(src_Ymm.data(), tmp_Ymm.data());
      for (int ii = 0; ii < dim_transpose; ii++) {
        vmovups(ptr[rsp + start_transpose_src1 + (j / 4 + ii) * 64 + i * 4], src_Ymm[ii]);
      }
    }

    // Tile product (output in col-major)
    const auto vreg_src0_col = zmm0;
    for (int ik = 0; ik < dim_transpose; ik++) {
      vmovups(vreg_src0_col, ptr[rsp + start_transpose_src0 + (j / 4 + ik) * 64]);
      for (int in = 0; in < TW_; in++) {
        vpdpbusds(dst_tile_Vmm(in), vreg_src0_col, zword_b[rsp + start_transpose_src1 + in * 4 + (j / 4 + ik) * 64]);
      }
    }
  }
}

void jit_matmul_vnni_noperm_p2031_p1302_t::generate() {
  inLocalLabel();  // use local label for multiple instance
  preamble();
  sub(rsp, stack_size);

  mov(reg_src0, ptr[parambase + GET_OFF(src0)]);
  mov(reg_src1, ptr[parambase + GET_OFF(src1)]);
  mov(reg_ksize, param_.K);
  xor_(reg_iterk, reg_iterk);

  mov(reg_ld_src0, ld_src0);
  mov(reg_ld_src1, ld_src1);

  for (int i = 0; i < TW_; i++) {
    vxorps(dst_tile_Vmm(i), dst_tile_Vmm(i));
  }

  L(kloop);
  calc_THxTKxTW();
  add(reg_src0, TK_ * VNNI_ADJ * dsize_src0);
  add(reg_src1, TK_ * VNNI_ADJ * dsize_src1);
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
  auto& reg_ld_dst = reg_tmp;
  mov(reg_dst, ptr[parambase + GET_OFF(dst)]);
  mov(reg_ld_dst, ld_dst);

  for (int j = 0; j < TW_; ++j) {
    // TODO(zhe1wang): replace with eltwise injector supporting runtime args
    vcvtdq2ps(dst_tile_Vmm(j) | T_rn_sae, dst_tile_Vmm(j));       // s32->fp32
    vfmadd132ps(dst_tile_Vmm(j), vreg_zp, vreg_scale);            // multiplies scaler and add zp
    vcmpleps(reg_k1, vreg_zero, dst_tile_Vmm(j));                 // mask of everything greater than 0
    vcvtps2udq(dst_tile_Vmm(j) | T_z | reg_k1, dst_tile_Vmm(j));  // fp32->u32
    vpmovusdb(ptr[reg_dst + ld_dst * j], dst_tile_Vmm(j));        // store result
  }
  add(rsp, stack_size);

  postamble();
  outLocalLabel();  // end of local label
}
}  // namespace jd
