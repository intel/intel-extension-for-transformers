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

#include "jblas/jit_blas_prologue_b.h"
#include "jblas/jit_blas_wrapper.h"

namespace jblas {
template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T>
using tLauncher_Fp_F32F32 =
    jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, jblas::prologue_a::gemm::ActivationKBlockBaseF32,
                                         Wei_T, jblas::epilogue::gemm::CompFp32BlockEpilogue,
                                         jblas::epilogue::gemm::AccumulatorWriteBackFp32>;

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T>
using tLauncher_Int8_F32F32 =
    jblas::wrapper::gemm::LauncherIntKBlock<GemmCore_T::ISA, GemmCore_T,
                                            jblas::prologue_a::gemm::ActivationF32KBlockQuantize, Wei_T,
                                            jblas::epilogue::gemm::AccumulatorWriteBackFp32>;

using tAVX2 = jblas::gemm::SCoreRowNAvx2<24, 4>;
using tAVX_VNNI = jblas::gemm::ICoreRowNAvxvnni<24, 4>;
using tAVX512F = jblas::gemm::SCoreRowNAvx512f<48, 8>;
using tAVX512_VNNI = jblas::gemm::ICoreRowNAvx512vnni<48, 8>;
using tAMX_BF16 = jblas::gemm::HCoreRowNAmxbf16<48, 16>;
using tAVX512_BF16 = jblas::gemm::HCoreRowNAvx512bf16<48, 8>;
using tAVX512_FP16 = jblas::gemm::HCoreRowNAvx512fp16<96, 8>;
using tAMX_INT8_US = jblas::gemm::ICoreRowNAmxint8<48, 16>;
using tAMX_INT8_SS = jblas::gemm::ICoreRowNAmxint8SS<48, 16>;

using tAVX_VNNI_KBlock = jblas::gemm::ICoreRowNAvxvnniKBlock<24, 2>;
using tAVX512_VNNI_KBlock = jblas::gemm::ICoreRowNAvx512vnniKBlock<48, 4>;
using tAMX_INT8_US_KBlock = jblas::gemm::ICoreRowNAmxint8KBlock<48, 16>;
using tAMX_INT8_SS_KBlock = jblas::gemm::ICoreRowNAmxint8SSKBlock<48, 16>;

template <class GC_T, JBLAS_ISA ISA_T>
using tWeiNInt = jblas::prologue_b::gemm::WeightKBlockNInteger<GC_T, ISA_T>;
template <class GC_T, JBLAS_ISA ISA_T>
using tWeiS8 = jblas::prologue_b::gemm::WeightKBlockS8<GC_T, ISA_T>;
template <class GC_T, JBLAS_ISA ISA_T>
using tWeiS4 = jblas::prologue_b::gemm::WeightKBlockS4<GC_T, ISA_T>;
template <class GC_T, JBLAS_ISA ISA_T>
using tWeiF4 = jblas::prologue_b::gemm::WeightKBlockF4<GC_T, ISA_T>;
template <class GC_T, JBLAS_ISA ISA_T>
using tWeiF8 = jblas::prologue_b::gemm::WeightKBlockF8<GC_T, ISA_T>;

template <class GC_T, JBLAS_ISA ISA_T>
using tActKBaseF32 = jblas::prologue_a::gemm::ActivationKBlockBaseF32<GC_T, ISA_T>;

constexpr uint64_t Fp32Cores[] = {tAVX2::ID, tAVX512F::ID};
constexpr uint64_t Bf16Cores[] = {tAMX_BF16::ID};
constexpr uint64_t Fp16Cores[] = {tAVX512_FP16::ID};
constexpr uint64_t Int8Cores[] = {tAVX_VNNI::ID, tAVX512F::ID, tAVX512_VNNI::ID, tAMX_INT8_US::ID, tAMX_INT8_SS::ID};
constexpr uint64_t FloatCores[] = {tAVX2::ID, tAVX512F::ID, tAMX_BF16::ID, tAVX512_FP16::ID};
constexpr uint64_t AllKBlockCores[] = {tAVX2::ID,
                                       tAVX512F::ID,
                                       tAMX_BF16::ID,
                                       tAVX512_FP16::ID,
                                       tAVX_VNNI_KBlock::ID,
                                       tAVX512_VNNI_KBlock::ID,
                                       tAMX_INT8_US_KBlock::ID,
                                       tAMX_INT8_SS_KBlock::ID};

}  // namespace jblas
