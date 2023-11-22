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
#include "../include/jblas_gemm_dispatcher.hpp"
#include "../include/dispatcher_utils.hpp"
#include "jblas/jit_blas.h"
#include "jblas/jit_blas_epilogue.h"
#include "jblas/jit_blas_gemm.h"
#include "jblas/jit_blas_prologue_a.h"
#include <c10/util/Exception.h>
#include <torch/types.h>

namespace jblas_gemm {

template <class GemmCore, template <class _T, JBLAS_ISA> class PrologueA, template <JBLAS_ISA ISA> class Epilogue>
void do_gemm(jblas_gemm_runtime_ctx* ctx) {}

void dispatch_jblas_gemm(jblas_gemm_runtime_ctx* ctx) {
  TORCH_CHECK(
      ctx->matA->scalar_type() == ctx->matB->scalar_type() && ctx->matA->scalar_type() == ctx->matC->scalar_type(),
      "QBits: data-type of matA matB matC must be equal in gemm op.");
  if (ctx->matA->scalar_type() == torch::kFloat32) {
    if (dispatcher_utils::check_avx512f()) {
      return do_gemm<jblas::gemm::SCoreRowNAvx512f<48, 8>, jblas::prologue_a::gemm::ActivationBase,
                     jblas::epilogue::gemm::AccumulatorWriteBackFp32>(ctx);
    }
  }
}
}  // namespace jblas_gemm