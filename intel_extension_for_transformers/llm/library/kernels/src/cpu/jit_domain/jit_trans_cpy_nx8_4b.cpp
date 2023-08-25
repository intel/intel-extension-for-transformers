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
#include "jit_trans_cpy_nx8_4b.hpp"

#define GET_OFF(tparam, field) offsetof(jit_transpose_nx8_4b<tparam>::rt_data_t, field)

namespace jd {

template <int tile_m>
inline std::vector<Xbyak::Ymm> jit_transpose_nx8_4b<tile_m>::get_Ymm(int start, int num) const {
  std::vector<Xbyak::Ymm> result(num);
  for (int i = 0; i < num; ++i) {
    result[i] = Xbyak::Ymm(start + i);
  }
  return result;
}

template <int tile_m>
void jit_transpose_nx8_4b<tile_m>::transpose8_ps(const Xbyak::Ymm mat[8], const Xbyak::Ymm tmp[8]) {
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

template <int tile_m>
void jit_transpose_nx8_4b<tile_m>::generate() {
  preamble();
  inLocalLabel();  // use local label for multiple instance

  mov(reg_src, ptr[parambase + GET_OFF(tile_m, src)]);
  mov(reg_dst, ptr[parambase + GET_OFF(tile_m, dst)]);
  mov(reg_ksize, param_.K);

  const auto src_Ymm = get_Ymm(0, 8), tmp_Ymm = get_Ymm(8, 8);
  xor_(reg_iterk, reg_iterk);
  L(k_loop);
  for (int j = 0; j < tile_m; j += dim_transpose) {
    for (int ii = 0; ii < dim_transpose; ii++) {
      vmovdqu8(src_Ymm[ii], ptr[reg_src + param_.ld_src * (j + ii)]);
    }
    transpose8_ps(src_Ymm.data(), tmp_Ymm.data());
    for (int ii = 0; ii < 8; ii++) {
      vmovdqu8(ptr[reg_dst + ii * tile_m * VNNI_ADJ + j * VNNI_ADJ], src_Ymm[ii]);
    }
  }
  add(reg_dst, VNNI_ADJ * dim_transpose * tile_m);
  add(reg_src, VNNI_ADJ * dim_transpose);
  add(reg_iterk, VNNI_ADJ * dim_transpose);
  cmp(reg_iterk, reg_ksize);
  jb(k_loop);

  outLocalLabel();  // end of local label
  postamble();
}

template class jit_transpose_nx8_4b<8>;
template class jit_transpose_nx8_4b<32>;
}  // namespace jd
