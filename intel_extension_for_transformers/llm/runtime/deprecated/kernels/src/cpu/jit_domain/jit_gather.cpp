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

#include "jit_gather.hpp"

#include "regs_pool.hpp"

#define GET_OFF(field) offsetof(rt_data_t, field)

namespace jd {
void jit_gather_t::generate() { param_.use_avx512 ? generate_<true>() : generate_<false>(); }

template <bool USE_AVX512>
void jit_gather_t::generate_() {
  using VMM = std::conditional_t<USE_AVX512, Zmm, Ymm>;
  constexpr auto BYTES_VMM = USE_AVX512 ? BYTES_ZMM : BYTES_YMM;
  constexpr auto stack_size = USE_AVX512 ? 0 : BYTES_VMM;
  const auto rp_flags = USE_AVX512 ? regs_pool::DefaultFlags : regs_pool::DisableEvex;

  regs_pool rp(this, 1, {3 + static_cast<int>(binaryop_attrs_.size()), 1, 2}, stack_size, rp_flags, BYTES_VMM);
  const auto src_addr = rp.reg<Reg64>();
  mov(src_addr, ptr[rp.p[0] + GET_OFF(src)]);
  {
    const auto idx_addr = rp.reg<Reg64>();
    mov(idx_addr, ptr[rp.p[0] + GET_OFF(idx)]);

    const auto gather_idx = rp.reg<Reg64>();
    mov(gather_idx.cvt32(), dword[idx_addr + 4]);
    imul(gather_idx.cvt32(), gather_idx, param_.inner_size * param_.dt_size);
    prefetchnta(ptr[gather_idx + src_addr]);

    mov(gather_idx.cvt32(), dword[idx_addr]);
    imul(gather_idx, gather_idx, param_.inner_size * param_.dt_size);
    add(src_addr, gather_idx);
  }
  const auto dst_addr = rp.reg<Reg64>();
  mov(dst_addr, ptr[rp.p[0] + GET_OFF(dst)]);
  const auto binaryop_addr = rp.regs<Reg64>(binaryop_attrs_.size());
  for (size_t i = 0; i < binaryop_attrs_.size(); i++) mov(binaryop_addr[i], ptr[rp.p[0] + GET_OFF(binaryop_addrs)]);

  const auto vmm_elements = BYTES_VMM / param_.dt_size;
  const auto loops = param_.inner_size / vmm_elements;
  for (size_t m = 0; m < loops; ++m) {
    const auto offset = m * BYTES_VMM;
    const auto xs = rp.reg<VMM>();
    vmovups(xs, ptr[src_addr + offset]);
    for (size_t i = 0; i < binaryop_attrs_.size(); i++)
      binary_injector.compute_vector(xs, binaryop_addr[i] + offset, binaryop_attrs_[i]);
    vmovups(ptr[dst_addr + offset], xs);
  }
  // tail
  const auto tail_size = param_.inner_size % vmm_elements;
  const auto reg_tmp = rp.reg<Reg64>();
  const auto tail_mask = rp.reg<Opmask>();
  const auto extend_tail_mask = rp.reg<Opmask>();
  if (tail_size) {
    if (binaryop_attrs_.size() > 0 && USE_AVX512) {  // TODO(Zhe): binary injector without EVEX
      mov(reg_tmp, (1ULL << tail_size) - 1);
      kmovq(tail_mask, reg_tmp);
      binary_injector.set_mask(tail_mask);
    }

    auto tail_bytes = tail_size * param_.dt_size;
    if (USE_AVX512) {
      mov(reg_tmp, (1ULL << tail_bytes) - 1);
      kmovq(extend_tail_mask, reg_tmp);
    }
    const auto offset = loops * BYTES_VMM;
    const auto has_binary = binaryop_attrs_.size() != 0;
    const auto xs = rp.reg<VMM>();
    if (USE_AVX512) {
      vmovdqu8(xs | extend_tail_mask, ptr[src_addr + offset]);
    } else {  // mask is available only with EVEX encoding
      vmov_avx2((has_binary ? rsp : dst_addr + offset), src_addr + offset, tail_bytes, Xmm(xs.getIdx()), reg_tmp);
      if (has_binary) vmovaps(xs, ptr[rsp]);
    }

    for (size_t i = 0; i < binaryop_attrs_.size(); i++)
      binary_injector.compute_vector(xs, binaryop_addr[i] + offset, binaryop_attrs_[i], true);

    if (USE_AVX512) {
      vmovdqu8(ptr[dst_addr + offset] | extend_tail_mask, xs);
    } else if (has_binary) {
      vmovaps(ptr[rsp], xs);
      vmov_avx2(dst_addr + offset, rsp, tail_bytes, Xmm(xs.getIdx()), reg_tmp);
    }
  }
}
}  // namespace jd
