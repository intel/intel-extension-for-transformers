//  Copyright (c) 2021 Intel Corporation
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

#ifndef ENGINE_EXECUTOR_INCLUDE_ISA_HPP_
#define ENGINE_EXECUTOR_INCLUDE_ISA_HPP_

#include "glog/logging.h"
#include "oneapi/dnnl/dnnl.hpp"

namespace executor {

/**
* The ISAs are partially ordered:
* - SSE41 < AVX < AVX2,
* - AVX2 < AVX512_CORE < AVX512_CORE_VNNI < AVX512_CORE_BF16 < AVX512_CORE_AMX,
* - AVX2 < AVX2_VNNI.
* Suppose to use AVX-512 for getting best performance when doing kernel tuning.
* For more supported isa details, please see OneDNN docs.
* https://oneapi-src.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html
* https://oneapi-src.github.io/oneDNN/enum_dnnl_cpu_isa_t.html#enum-dnnl-cpu-isa-t
*
*/

enum class isa {
  unspport = 0x0,
  avx512_core = 0x1,
  avx512_core_vnni = 0x3,
  avx512_core_bf16 = 0x7,
  avx512_core_amx = 0xf,
};

inline isa get_max_isa() {
  dnnl::cpu_isa dnnl_isa = dnnl::get_effective_cpu_isa();
  switch (dnnl_isa) {
    case (dnnl::cpu_isa::avx512_core):
      return isa::avx512_core;
    case (dnnl::cpu_isa::avx512_core_vnni):
      return isa::avx512_core_vnni;
    case (dnnl::cpu_isa::avx512_core_bf16):
      return isa::avx512_core_bf16;
    case (dnnl::cpu_isa::avx512_core_amx):
      return isa::avx512_core_amx;
    default:
      LOG(FATAL) << "Only support device with Intel AVX-512!";
      return isa::unspport;
  }
}
}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_ISA_HPP_
