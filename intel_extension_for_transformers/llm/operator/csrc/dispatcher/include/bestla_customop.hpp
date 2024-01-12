//  Copyright (c) 2023 Intel Corporation
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
#pragma once
#include <cassert>
#include <cstddef>
#include <type_traits>
#include "bestla/bestla.h"
#include "bestla/kernel_wrapper.h"

template <typename Param, typename DST_T, BTLA_ISA ISA_T>
inline BTLA_CODE alphabeta_dt_cvt_process(float* tmp_dst, const int cachestep, const int M_offset, const int N_offset,
                                           const int M, const int N, const Param& _param) {
  auto DOffset = M_offset * _param.ldd + N_offset;
  auto dptr = reinterpret_cast<float*>(_param.D) + DOffset;
  bestla::kernel::wrapper::AlphaBetaF32F32::template forward<ISA_T>(_param.alpha, tmp_dst, cachestep, _param.beta, dptr,
                                                                   _param.ldd, tmp_dst, cachestep, M, N);

  auto COffset = M_offset * _param.ldc + N_offset;
  auto cptr = reinterpret_cast<DST_T*>(_param.C) + COffset;
  if constexpr (std::is_same_v<DST_T, float>) {
    return bestla::kernel::wrapper::Memcpy2D::template forward<ISA_T, float, DST_T>(tmp_dst, cptr, M, N, cachestep,
                                                                                   _param.ldc, NULL);
  }
  if constexpr (std::is_same_v<DST_T, bestla::utils::bf16>) {
    return bestla::kernel::wrapper::Memcpy2DFp32CvtBf16::template forward<ISA_T>(
        (void*)tmp_dst, (void*)cptr, M, N, cachestep * sizeof(float), _param.ldc * sizeof(DST_T), false);
  }
  assert(false);
}

template <BTLA_ISA ISA_T, typename DST_T>
class AlphaBetaProcess {
 public:
  struct Param {
    void *C, *D;
    int ldc, ldd;
    float alpha, beta;
  };
  BTLA_CODE forward(float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param, void* tmpcache = nullptr, size_t cachesize = -1) {
    return alphabeta_dt_cvt_process<Param, DST_T, ISA_T>(cacheptr, cachestep, M_offset, N_offset, M, N, _param);
  }
};

template <BTLA_ISA ISA_T>
using AlphaBetaProcessStoreFp32 = AlphaBetaProcess<ISA_T, float>;
template <BTLA_ISA ISA_T>
using AlphaBetaProcessStoreBf16 = AlphaBetaProcess<ISA_T, bestla::utils::bf16>;
