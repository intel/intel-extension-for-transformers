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
#include "jblas/jit_blas_parallel.h"
#include "jblas/jit_blas_prologue_a.h"
#include "jblas/jit_blas_prologue_b.h"
#include "jblas/jit_blas_utils.h"
#include "jblas/jit_blas_wrapper.h"
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <torch/types.h>
#include <string>

namespace jblas_gemm {

template <class GemmCore, template <class _T, JBLAS_ISA> class PrologueA, template <JBLAS_ISA ISA> class Epilogue,
          typename DT>
void do_gemm(jblas_gemm_runtime_ctx* ctx) {
  using Launcher = jblas::wrapper::gemm::LauncherBase<GemmCore::ISA, GemmCore, PrologueA,
                                                      jblas::prologue_b::gemm::WeightPack, Epilogue>;
  Launcher launcher;
  using Parallel = jblas::parallel::gemm::SchedulerBase<GemmCore>;
  auto packw = launcher.mProB.createStorage(ctx->n, ctx->k);
  auto tmpbuf = jblas::utils::amalloc<int8_t>(packw.mSize);
  packw.assign(tmpbuf);
  if (ctx->matB_trans) {
    launcher.mProB.packWeightTranspose(ctx->n, ctx->k, {reinterpret_cast<DT*>(ctx->matB->data_ptr()), ctx->k, &packw},
                                       &dispatcher_utils::DefaultThreading);
  } else {
    launcher.mProB.packWeight(ctx->n, ctx->k, {reinterpret_cast<DT*>(ctx->matB->data_ptr()), ctx->n, &packw},
                              &dispatcher_utils::DefaultThreading);
  }
  typename Launcher::Param args{ctx->m,
                                ctx->n,
                                ctx->k,
                                {reinterpret_cast<DT*>(ctx->matA->data_ptr()), ctx->k},
                                {reinterpret_cast<DT*>(ctx->matB->data_ptr()), ctx->n, &packw},
                                {reinterpret_cast<DT*>(ctx->matC->data_ptr()), ctx->n}};
  jblas::parallel::GemmBaseRun<Parallel>(launcher, args, &dispatcher_utils::DefaultThreading);
  jblas::utils::afree(tmpbuf);
}

void dispatch_jblas_gemm(jblas_gemm_runtime_ctx* ctx) {
  TORCH_CHECK(
      ctx->matA->scalar_type() == ctx->matB->scalar_type() && ctx->matA->scalar_type() == ctx->matC->scalar_type(),
      "QBits: data-type of matA matB matC must be equal in gemm op.");
  if (ctx->matA->scalar_type() == torch::kFloat32) {
    if (dispatcher_utils::check_avx512f()) {
      return do_gemm<jblas::gemm::SCoreRowNAvx512f<48, 8>, jblas::prologue_a::gemm::ActivationBase,
                     jblas::epilogue::gemm::AccumulatorWriteBackFp32, float>(ctx);
    }
    if (dispatcher_utils::check_avx2()) {
      return do_gemm<jblas::gemm::SCoreRowNAvx2<48, 2>, jblas::prologue_a::gemm::ActivationBase,
                     jblas::epilogue::gemm::AccumulatorWriteBackFp32, float>(ctx);
    }
  }
  if (ctx->matA->scalar_type() == torch::kBFloat16) {
    if (dispatcher_utils::check_amx()) {
      return do_gemm<jblas::gemm::HCoreRowNAmxbf16<64, 16>, jblas::prologue_a::gemm::ActivationBase,
                     jblas::epilogue::gemm::AccumulatorWriteBackFp32Bf16, jblas::utils::bf16>(ctx);
    }
  }
  TORCH_CHECK(false, "QBits: unsupported config in gemm op, data_type:", dispatcher_utils::get_torch_dt_name(ctx->matA),
              ", AVX2:", dispatcher_utils::check_avx2(), ", AVX512F:", dispatcher_utils::check_avx512f(),
              ", AMX_BF16:", dispatcher_utils::check_amx());
}
}  // namespace jblas_gemm
