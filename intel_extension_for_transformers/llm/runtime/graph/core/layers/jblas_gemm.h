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

/*++
Module Name:

    jblas_gemm.h

Abstract:

    C APIs of BesTLA GEMMs.
--*/

#pragma once

#include "data_types.h"
#include "jblas/jit_blas.h"

struct JBLAS_GEMM_DATA_PACKED_PARAMS {
  const float* A = nullptr; /**< address of A (float32 matrix)*/
  const void* B = nullptr;  /**< address of B (packed nbits blob)*/
  float* C = nullptr;       /**< address of result matrix */
  int lda = 0;           /**< leading dimension of A */
  int ldc = 0;              /**< leading dimension of C*/
};

size_t JblasGemmPackBSize(size_t N, size_t K, size_t BlkSize, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype,
                          bool isAsym, ne_comp_type CompType);

bool JblasGemmQuantPackBTrans(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb, size_t BlkSize,
                              JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype, bool isAsym, ne_comp_type CompType,
                              void* ThreadPool);
// QData:  K*N quantized int8 weight
// Scales: K/BlkSize * N scales
// Zp:     K/BlkSize * N zero points
bool JblasGemmPackB(void* PackedBuf, const int8_t* QData, const float* Scales, const int8_t* Zp, size_t N, size_t K,
                    size_t ldb, size_t BlkSize, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype, bool isAsym,
                    ne_comp_type CompType, void* ThreadPool);

bool JblasGemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, void* ThreadPool);

bool JblasGemmBatchDriver(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                          const JBLAS_GEMM_DATA_PACKED_PARAMS* DataParams, int8_t* WorkSpace, void* ThreadPool);
