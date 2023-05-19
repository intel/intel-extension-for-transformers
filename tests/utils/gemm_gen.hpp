/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <mkl.h>
#include "common/core/core.hpp"
#include <type_traits>

using namespace gpu::xetla;

template <class dtype_a, class dtype_b, class blas_type = float>
void get_gemm_gold(const size_t m, const size_t n, const size_t k,
        const mem_layout layout_a, const mem_layout layout_b, dtype_a *A,
        dtype_b *B, blas_type *C, const CBLAS_LAYOUT layout = CblasRowMajor) {
    std::vector<blas_type> tmp_A(A, A + m * k);
    std::vector<blas_type> tmp_B(B, B + k * n);

    CBLAS_TRANSPOSE transa, transb;
    transa = transb = CblasNoTrans;
    size_t lda, ldb, ldc;

    if (layout == CblasRowMajor) {
        ldc = n > 1 ? n : 1;
        if (layout_a == mem_layout::col_major) transa = CblasTrans;
        if (layout_b == mem_layout::col_major) transb = CblasTrans;
        if (transa == CblasNoTrans)
            lda = k > 1 ? k : 1;
        else
            lda = m > 1 ? m : 1;
        if (transb == CblasNoTrans)
            ldb = n > 1 ? n : 1;
        else
            ldb = k > 1 ? k : 1;
    } else {
        ldc = m > 1 ? m : 1;
        if (layout_a == mem_layout::row_major) transa = CblasTrans;
        if (layout_b == mem_layout::row_major) transb = CblasTrans;
        if (transa == CblasNoTrans)
            lda = m > 1 ? m : 1;
        else
            lda = k > 1 ? k : 1;
        if (transb == CblasNoTrans)
            ldb = k > 1 ? k : 1;
        else
            ldb = n > 1 ? n : 1;
    }

    if constexpr (std::is_same<remove_const_t<blas_type>, float>::value)
        cblas_sgemm(layout, transa, transb, m, n, k, 1.0, tmp_A.data(), lda,
                tmp_B.data(), ldb, 0.0, C, ldc);
    else if constexpr (std::is_same<remove_const_t<blas_type>, double>::value)
        cblas_dgemm(layout, transa, transb, m, n, k, 1.0, tmp_A.data(), lda,
                tmp_B.data(), ldb, 0.0, C, ldc);
}
