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
#ifndef ENGINE_SPARSELIB_SRC_CPU_CPU_ISA_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_CPU_ISA_HPP_

#include "src/utils.hpp"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace jd {

// Maximum number of features + hints that can be specified via bits
static constexpr int cpu_isa_total_bits = sizeof(unsigned) * 8;

enum cpu_isa_bit_t : unsigned {
  // Fill in features from least significant bit to most significant bit
  // begin from avx512, isa < avx512 will be dsiptached to reference
  // for more details abount AVX512-ISA supported status in different architectures, pls refer to this
  // page:https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512
  avx512_vbmi = 1u << 5,
  avx512_core_bit = 1u << 6,
  avx512_core_vnni_bit = 1u << 7,
  avx512_core_bf16_bit = 1u << 8,
  amx_tile_bit = 1u << 9,
  amx_int8_bit = 1u << 10,
  amx_bf16_bit = 1u << 11,
  avx_vnni_bit = 1u << 12,
  avx512_core_fp16_bit = 1u << 13,
};

enum cpu_isa_t : unsigned {
  isa_any = 0u,
  avx512_core = avx512_core_bit,
  avx512_core_vnni = avx512_core_vnni_bit | avx512_core,
  avx512_core_bf16 = avx512_core_bf16_bit | avx512_core_vnni,
  avx512_core_vbmi = avx512_vbmi,
  amx_tile = amx_tile_bit,
  amx_int8 = amx_int8_bit | amx_tile,
  amx_bf16 = amx_bf16_bit | amx_tile,
  avx512_core_bf16_amx_int8 = avx512_core_bf16 | amx_int8,
  avx512_core_bf16_amx_bf16 = avx512_core_bf16 | amx_bf16,
  avx512_core_amx = amx_int8 | amx_bf16,
  avx512_core_fp16 = avx512_core_fp16_bit,
};

bool init_amx();

inline const Xbyak::util::Cpu& cpu() {
  static const Xbyak::util::Cpu cpu_;
  return cpu_;
}

set_once_before_first_get_setting_t<bool>& amx_setting();

static inline bool isa_available(const cpu_isa_t cpu_isa) {
  using Cpu = Xbyak::util::Cpu;

  switch (cpu_isa) {
    case avx512_core:
      return cpu().has(Cpu::tAVX512F) && cpu().has(Cpu::tAVX512BW) && cpu().has(Cpu::tAVX512VL) &&
             cpu().has(Cpu::tAVX512DQ);
    case avx512_vbmi:
      return cpu().has(Cpu::tAVX512_VBMI);
    case avx512_core_vnni:
      return cpu().has(Cpu::tAVX512F) && cpu().has(Cpu::tAVX512BW) && cpu().has(Cpu::tAVX512VL) &&
             cpu().has(Cpu::tAVX512DQ) && cpu().has(Cpu::tAVX512_VNNI);
    case avx512_core_bf16:
      return cpu().has(Cpu::tAVX512_BF16);
    case amx_tile:
      return cpu().has(Cpu::tAMX_TILE) && amx_setting().get();
    case amx_int8:
      return isa_available(amx_tile) && cpu().has(Cpu::tAMX_INT8);
    case amx_bf16:
      return isa_available(amx_tile) && cpu().has(Cpu::tAMX_BF16);
    case avx512_core_bf16_amx_int8:
      return isa_available(avx512_core_bf16) && isa_available(amx_int8);
    case avx512_core_bf16_amx_bf16:
      return isa_available(avx512_core_bf16) && isa_available(amx_bf16);
    case avx512_core_amx:
      return isa_available(avx512_core_bf16_amx_int8) && isa_available(avx512_core_bf16_amx_bf16) &&
             cpu().has(Cpu::tAVX512_FP16);
    case avx512_core_fp16:
      return cpu().has(Cpu::tAVX512_FP16);
    case isa_any:
      return true;
  }
  return false;
}

}  // namespace jd

#endif  // ENGINE_SPARSELIB_SRC_CPU_CPU_ISA_HPP_
