/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    jblas_gemm.h

Abstract:

    Currently only support Q4 gemm.
--*/

#pragma once

#include "data_types.h"

struct JBLAS_GEMM_DATA_PACKED_PARAMS{
  const float* A = nullptr; /**< address of A (float32 matrix)*/
  const void* B = nullptr;  /**< address of B (packed nbits blob)*/
  float* C = nullptr;       /**< address of result matrix */
  size_t lda = 0;           /**< leading dimension of A */
  size_t ldc = 0;           /**< leading dimension of C*/
};

size_t JblasQ4GemmPackBSize(size_t N, size_t K, size_t BlkSize, bool isAsym, ne_comp_type CompType);

bool JblasQ4GemmPackB(void* PackedBuf, const uint8_t* QData, const float* Scale, const uint8_t* Zp, size_t N, size_t K,
                      size_t ldb, size_t BlkSize, bool isAsym, bool lastCall, ne_comp_type CompType, void* ThreadPool);

bool JblasQ4GemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb,
                        void* ThreadPool);

bool JblasQ4GemmBatchDriver(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                            const JBLAS_GEMM_DATA_PACKED_PARAMS* DataParams, int8_t* WorkSpace,
                            void* ThreadPool);
