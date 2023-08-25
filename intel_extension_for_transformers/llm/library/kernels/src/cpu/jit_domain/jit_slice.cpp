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

#include "jit_slice.hpp"

#define GET_OFF(field) offsetof(rt_data_t, field)

namespace jd {

template <bool USE_AVX512>
void jit_slice_t::copy_continuously(regs_pool* const rp, const Reg64 dst, const Reg64 src) {
  using VMM = std::conditional_t<USE_AVX512, Zmm, Ymm>;
  constexpr auto BYTES_VMM = USE_AVX512 ? BYTES_ZMM : BYTES_YMM;
  const auto loops = copy_size / BYTES_VMM;

  const auto vreg_xs = rp->reg<VMM>();
  for (size_t m = 0; m < loops; ++m) {
    const auto offset = m * BYTES_VMM;
    vmovups(vreg_xs, ptr[src + offset]);
    vmovups(ptr[dst + offset], vreg_xs);
  }

  // tail
  const int tail_bytes = copy_size % BYTES_VMM;
  const auto bytes_mask = (1LL << tail_bytes) - 1;
  const auto extend_tail_mask = rp->reg<Opmask>();
  const auto tmp_r64 = rp->reg<Reg64>();
  if (tail_bytes != 0) {
    const auto offset = loops * BYTES_VMM;
    if (USE_AVX512) {
      mov(tmp_r64, bytes_mask);
      kmovq(extend_tail_mask, tmp_r64);
      vmovdqu8(vreg_xs | extend_tail_mask, ptr[src + offset]);
      vmovdqu8(ptr[dst + offset] | extend_tail_mask, vreg_xs);
    } else {
      vmov_avx2(dst + offset, src + offset, tail_bytes, Xmm(vreg_xs.getIdx()), tmp_r64);
    }
  }
}

template <bool USE_AVX512>
void jit_slice_t::copy_by_step(regs_pool* const rp, const Reg64 dst, const Reg64 src) {
  using VMM = std::conditional_t<USE_AVX512, Zmm, Ymm>;
  constexpr auto BYTES_VMM = USE_AVX512 ? BYTES_ZMM : BYTES_YMM;
  const auto loops = copy_size / BYTES_VMM;

  const auto tmp_r64 = rp->reg<Reg64>();
  for (size_t m = 0; m < loops; ++m) {
    const auto src_offset = m * BYTES_VMM;
    const auto dst_offset = m * BYTES_VMM / 2;
    if (USE_AVX512) {
      const auto vreg_xs = rp->reg<VMM>();
      vmovups(vreg_xs, ptr[src + src_offset]);
      switch (dt_size) {
        case 4:
          vpmovqd(ptr[dst + dst_offset], vreg_xs);
          break;
        case 2:
          vpmovdw(ptr[dst + dst_offset], vreg_xs);
          break;
        case 1:
          vpmovwb(ptr[dst + dst_offset], vreg_xs);
          break;
        default:
          SPARSE_LOG(FATAL) << "Unexpected dt_size";
      }
    } else {
      const auto xmms = rp->regs<Xmm, 4>();
      switch (dt_size) {
        case 4:
          vmovd(xmms[0], dword[src + src_offset + 0]);
          vmovd(xmms[1], dword[src + src_offset + 16]);
          vpinsrd(xmms[0], xmms[0], dword[src + src_offset + 8], 1);
          vpinsrd(xmms[1], xmms[1], dword[src + src_offset + 24], 1);
          vpunpcklqdq(xmms[0], xmms[0], xmms[1]);
          vmovups(xword[dst + dst_offset], xmms[0]);
          break;
        case 2:
          vmovd(xmms[0], dword[src + src_offset + 0]);
          vmovd(xmms[1], dword[src + src_offset + 8]);
          vpinsrw(xmms[0], xmms[0], word[src + src_offset + 4], 1);
          vpinsrw(xmms[1], xmms[1], word[src + src_offset + 12], 1);
          vmovd(xmms[2], dword[src + src_offset + 16]);
          vmovd(xmms[3], dword[src + src_offset + 24]);
          vpinsrw(xmms[2], xmms[2], word[src + src_offset + 20], 1);
          vpinsrw(xmms[3], xmms[3], word[src + src_offset + 28], 1);
          vpunpckldq(xmms[0], xmms[0], xmms[1]);
          vpunpckldq(xmms[2], xmms[2], xmms[3]);
          vpunpcklqdq(xmms[0], xmms[0], xmms[2]);
          vmovups(xword[dst + dst_offset], xmms[0]);
          break;
        case 1:
          for (size_t ii = 0; ii < BYTES_VMM / 2; ++ii) {
            movzx(tmp_r64.cvt32(), word[src + src_offset + ii * 2]);
            mov(byte[dst + dst_offset + ii], tmp_r64.cvt8());
          }
          break;
        default:
          SPARSE_LOG(FATAL) << "Unexpected dt_size";
      }
    }
  }
  // tail
  const auto tail_mask = rp->reg<Opmask>();
  const auto tail = copy_size % BYTES_VMM / dt_size / 2;  // in terms of #elemets in dst
  const int mask = (1LL << tail) - 1;
  if (tail != 0) {
    const auto src_offset = loops * BYTES_VMM;
    const auto dst_offset = loops * BYTES_VMM / 2;
    if (USE_AVX512) {
      const auto vreg_xs = rp->reg<VMM>();
      mov(tmp_r64.cvt32(), mask);
      kmovd(tail_mask, tmp_r64.cvt32());
      vmovups(vreg_xs, ptr[src + src_offset]);
      switch (dt_size) {
        case 4:
          vpmovqd(ptr[dst + dst_offset] | tail_mask, vreg_xs);
          break;
        case 2:
          vpmovdw(ptr[dst + dst_offset] | tail_mask, vreg_xs);
          break;
        case 1:
          vpmovwb(ptr[dst + dst_offset] | tail_mask, vreg_xs);
          break;
        default:
          SPARSE_LOG(FATAL) << "Unexpected dt_size";
      }
    } else {
      const auto xmms = rp->regs<Xmm, 4>();
      SPARSE_LOG_IF(FATAL, tail <= 0 || tail >= BYTES_VMM / dt_size) << "Unexpected tail length!";
      switch (dt_size) {
        case 4:
          for (size_t ii = 0; ii < tail; ++ii) {  // TODO(Yucheng): Can we use overlapping for the tail?
            mov(tmp_r64, qword[src + src_offset + ii * 8]);
            mov(dword[dst + dst_offset + ii * 4], tmp_r64.cvt32());
          }
          break;
        case 2:
          for (size_t ii = 0; ii < tail; ++ii) {
            mov(tmp_r64.cvt32(), dword[src + src_offset + ii * 4]);
            mov(word[dst + dst_offset + ii * 2], tmp_r64.cvt16());
          }
          break;
        case 1:
          for (size_t ii = 0; ii < tail; ++ii) {
            movzx(tmp_r64.cvt32(), word[src + src_offset + ii * 2]);
            mov(byte[dst + dst_offset + ii * 1], tmp_r64.cvt8());
          }
          break;
        default:
          SPARSE_LOG(FATAL) << "Unexpected dt_size";
      }
    }
  }
}

template <bool USE_AVX512>
void jit_slice_t::generate_() {
  const auto use_by_step = inner_size == 1 && step > 1;
  const auto rp_flags = USE_AVX512 ? regs_pool::DefaultFlags : regs_pool::DisableEvex;
  regs_pool rp(this, 1, {3, (USE_AVX512 || !use_by_step) ? 1 : 4, 1}, 0, rp_flags);
  const auto src_addr = rp.reg<Reg64>();
  const auto dst_addr = rp.reg<Reg64>();
  mov(src_addr, ptr[rp.p[0] + GET_OFF(src)]);
  mov(dst_addr, ptr[rp.p[0] + GET_OFF(dst)]);

  if (inner_size > 1 && step > 1)
    prefetchnta(ptr[inner_size * step * dt_size + src_addr]);
  else
    prefetchnta(ptr[inner_size * src_axis_size * dt_size + src_addr]);

  if (use_by_step)
    copy_by_step<USE_AVX512>(&rp, dst_addr, src_addr);
  else
    copy_continuously<USE_AVX512>(&rp, dst_addr, src_addr);
}

void jit_slice_t::generate() { use_avx512 ? generate_<true>() : generate_<false>(); }
}  // namespace jd
