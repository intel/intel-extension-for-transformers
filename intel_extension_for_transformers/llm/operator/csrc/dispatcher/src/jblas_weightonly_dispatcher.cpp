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
#include "../include/jblas_weightonly_dispatcher.hpp"
#include "../include/jblas_customop.hpp"
#include <omp.h>
#include "jblas/jit_blas.h"
#include "jblas/jit_blas_epilogue.h"
#include "jblas/jit_blas_gemm.h"
#include "jblas/jit_blas_parallel.h"
#include "jblas/jit_blas_prologue_b.h"
#include "jblas/jit_blas_prologue_a.h"
#include "jblas/jit_blas_storage.h"
#include "jblas/jit_blas_wrapper.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>

namespace woq {

inline void set_nk(woq_runtime_ctx* ctx, torch::Tensor* tensor) {
  ctx->n = ctx->transpose ? static_cast<int>(tensor->sizes()[0]) : static_cast<int>(tensor->sizes()[1]);
  ctx->k = ctx->transpose ? static_cast<int>(tensor->sizes()[1]) : static_cast<int>(tensor->sizes()[0]);
}

static std::map<std::string, JBLAS_DTYPE> wei2jblasdt_map{{"s4clip_scalef32", JBLAS_DTYPE::S4_CLIP},
                                                          {"s4fullrange_scalef32", JBLAS_DTYPE::S4_FULLRANGE},
                                                          {"nf4_scalef32", JBLAS_DTYPE::F4_NF4},
                                                          {"fp4bnb_scalef32", JBLAS_DTYPE::F4_BNB},
                                                          {"fp4e2m1_scalef32", JBLAS_DTYPE::F4_E2M1}};
static void* woq_workspace = nullptr;
static int64_t workspace_size = 0;

template <typename T>
concept quant_PrologueA = requires {
  requires !std::is_same_v<T, float>;
  requires !std::is_same_v<T, jblas::utils::bf16>;
};

template <class Launcher>
void woq_dequantize(woq_config_param* p, woq_runtime_ctx* ctx) {
  if (dispatcher_utils::initer.verbose) dispatcher_utils::timer.start();
  using PrologueB = typename Launcher::PrologueB;
  static PrologueB kernel;
  if (ctx->transpose) {
    kernel.unpackTransposeWeight(ctx->deseries_wei->mN, ctx->deseries_wei->mK, ctx->deseries_wei,
                                 ctx->output->data_ptr<float>(), ctx->deseries_wei->mK,
                                 &dispatcher_utils::DefaultThreading);
  } else {
    kernel.unpackWeight(ctx->deseries_wei->mN, ctx->deseries_wei->mK, ctx->deseries_wei, ctx->output->data_ptr<float>(),
                        ctx->deseries_wei->mN, &dispatcher_utils::DefaultThreading);
  }
}

template <class Launcher>
void woq_quantize(woq_config_param* p, woq_runtime_ctx* ctx) {
  if (dispatcher_utils::initer.verbose) dispatcher_utils::timer.start();
  using WType = typename Launcher::PrologueB::StorageWeight;
  WType packedw(0);
  static Launcher launcher;
  if constexpr (std::is_same_v<WType, jblas::storage::gemm::StorageWeightKBlockS8>) {
    packedw = launcher.mProB.createStorage(ctx->n, ctx->k, ctx->blocksize, jblas::utils::jblas_dtype<float>,
                                           jblas::utils::jblas_dtype<float>, false);
  } else if constexpr (std::is_same_v<WType, jblas::storage::gemm::StorageWeightKBlockS4>) {
    packedw = launcher.mProB.createStorage(ctx->n, ctx->k, ctx->blocksize, wei2jblasdt_map[p->weight_type],
                                           jblas::utils::jblas_dtype<float>, jblas::utils::jblas_dtype<float>, false);
  } else if constexpr (std::is_same_v<WType, jblas::storage::gemm::StorageWeightKBlockF4>) {
    packedw = launcher.mProB.createStorage(ctx->n, ctx->k, ctx->blocksize, wei2jblasdt_map[p->weight_type],
                                           jblas::utils::jblas_dtype<float>);
  } else {
    assert(0);
  }
  *(ctx->output) = torch::empty(packedw.mSize, torch::kInt8);
  packedw.assign(ctx->output->data_ptr<int8_t>());
  if (ctx->transpose) {
    launcher.mProB.packTransposeWeight(ctx->n, ctx->k, ctx->weight->data_ptr<float>(), ctx->k, &packedw,
                                       &dispatcher_utils::DefaultThreading);
  } else {
    launcher.mProB.packWeight(ctx->n, ctx->k, ctx->weight->data_ptr<float>(), ctx->n, &packedw,
                              &dispatcher_utils::DefaultThreading);
  }
  if (dispatcher_utils::initer.verbose) {
    dispatcher_utils::timer.stop();
    auto cost_time = dispatcher_utils::timer.get_elapsed_time();
    LOG(INFO) << "QBits quantize verbose\nn:" << ctx->n << " k:" << ctx->k << " weight_type:" << p->weight_type
              << " blocksize:" << ctx->blocksize << " src_type:" << dispatcher_utils::get_torch_dt_name(ctx->weight)
              << " execute time:" << cost_time << "ms";
  }
}
template <class Launcher, class ParamA>
void do_compute(woq_config_param* p, woq_runtime_ctx* ctx, ParamA param_a) {
  if (dispatcher_utils::initer.verbose) dispatcher_utils::timer.start();
  static Launcher launcher;
  using EpiParam = typename Launcher::EpiParam;
  EpiParam param_epi = {ctx->output->data_ptr(), ctx->bias->data_ptr(), ctx->ldo, 0, ctx->alpha, ctx->beta};
  using GemmCore = typename Launcher::GemmCore;
  if constexpr (GemmCore::ISA == JblasAMX_INT8 || GemmCore::ISA == JblasAVX512_VNNI || GemmCore::ISA == JblasAVX_VNNI) {
    using Parallel = jblas::parallel::gemm::SchedulerKBlockS<GemmCore>;
    jblas::utils::GemmProblem gp(1, ctx->m, ctx->n, ctx->k, ctx->blocksize);
    typename Launcher::Param args{gp, param_a,
                                  dynamic_cast<jblas::storage::gemm::IWeightKBlockBase*>(ctx->deseries_wei), param_epi};
    jblas::parallel::GemmRunWithA<Parallel>(launcher, args, &dispatcher_utils::DefaultThreading);
  } else {
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore>;
    using StorageWeight = typename Launcher::PrologueB::StorageWeight;
    StorageWeight* packedw = dynamic_cast<StorageWeight*>(ctx->deseries_wei);
    jblas::utils::GemmProblem gp(1, ctx->m, ctx->n, ctx->k, ctx->blocksize);
    typename Launcher::Param args{gp,
                                  param_a,
                                  dynamic_cast<jblas::storage::gemm::IWeightKBlockBase*>(ctx->deseries_wei),
                                  {packedw->template SPtr<int8_t>(), packedw->SDtype(), packedw->CStep()},
                                  param_epi};
    jblas::parallel::GemmRun<Parallel>(launcher, args, &dispatcher_utils::DefaultThreading);
  }
  if (dispatcher_utils::initer.verbose) {
    dispatcher_utils::timer.stop();
    auto cost_time = dispatcher_utils::timer.get_elapsed_time();
    LOG(INFO) << "QBits linear verbose\nm:" << ctx->m << " n:" << ctx->deseries_wei->mN
              << " k:" << ctx->deseries_wei->mK << " weight_type:" << p->weight_type
              << " compute_type:" << p->compute_type << " blocksize:" << ctx->blocksize
              << " src_type:" << dispatcher_utils::get_torch_dt_name(ctx->activation)
              << " dst_type:" << dispatcher_utils::get_torch_dt_name(ctx->output) << " execute time:" << cost_time
              << "ms";
  }
}

template <class Launcher>
void parse_paramA(woq_config_param* p, woq_runtime_ctx* ctx) {
  using PrologueA = typename Launcher::PrologueA;
  using ParamA = typename PrologueA::Param;
  using SrcType = typename PrologueA::SRCType;
  if constexpr (quant_PrologueA<typename PrologueA::AType>) {
    static PrologueA kernel;
    void* workspace = woq_workspace == nullptr ? NULL : woq_workspace;
    size_t need_size;
    void* tmpbuf = NULL;
    auto get_workspace = [&] {
      if (workspace != NULL) {
        TORCH_CHECK(workspace_size >= need_size, "Qbits: workspace size should large than ", need_size, " bytes");
        return workspace;
      } else {
        tmpbuf = jblas::utils::amalloc<int8_t>(need_size);
        return tmpbuf;
      }
    };
    auto quantA = kernel.createStorage(ctx->m, ctx->deseries_wei->mK, ctx->blocksize, false);
    need_size = quantA.mSize;
    quantA.assign(reinterpret_cast<int8_t*>(get_workspace()));
    kernel.quantize({reinterpret_cast<SrcType*>(ctx->activation->data_ptr()), ctx->deseries_wei->mK, &quantA}, ctx->m,
                    ctx->deseries_wei->mK, &dispatcher_utils::DefaultThreading);
    ParamA param_a = {reinterpret_cast<SrcType*>(ctx->activation->data_ptr()), ctx->deseries_wei->mK, &quantA};
    return do_compute<Launcher, ParamA>(p, ctx, param_a);
  } else {
    ParamA param_a = {reinterpret_cast<SrcType*>(ctx->activation->data_ptr()), ctx->deseries_wei->mK};
    return do_compute<Launcher, ParamA>(p, ctx, param_a);
  }
}

template <class Launcher>
void woq_gemm(woq_config_param* p, woq_runtime_ctx* ctx) {
  return parse_paramA<Launcher>(p, ctx);
}

template <WOQ_TASK TASK, class Launcher>
void execute_task(woq_config_param* p, woq_runtime_ctx* ctx) {
  switch (TASK) {
    case WOQ_QUANTIZE:
      return woq_quantize<Launcher>(p, ctx);
    case WOQ_DEQUANTIZE:
      return woq_dequantize<Launcher>(p, ctx);
    case WOQ_LINEAR:
      return woq_gemm<Launcher>(p, ctx);
  }
}

template <WOQ_TASK TASK, class GemmCore, template <class _T, JBLAS_ISA> class PrologueB,
          template <class _T, JBLAS_ISA> class PrologueA, template <JBLAS_ISA> class Epilogue>
void parse_launcher(woq_config_param* p, woq_runtime_ctx* ctx) {
  if constexpr (GemmCore::ISA == JblasAMX_INT8 || GemmCore::ISA == JblasAVX512_VNNI || GemmCore::ISA == JblasAVX_VNNI) {
    using Launcher = jblas::wrapper::gemm::LauncherIntKBlock<GemmCore::ISA, GemmCore, PrologueA, PrologueB, Epilogue>;
    return execute_task<TASK, Launcher>(p, ctx);
  } else {
    using Launcher = jblas::wrapper::gemm::LauncherKBlock<GemmCore::ISA, GemmCore, PrologueA, PrologueB,
                                                          jblas::epilogue::gemm::CompFp32BlockEpilogue, Epilogue>;
    return execute_task<TASK, Launcher>(p, ctx);
  }
}

template <WOQ_TASK TASK, class GemmCore, template <class _T, JBLAS_ISA> class PrologueB,
          template <class _T, JBLAS_ISA> class PrologueA, dispatcher_utils::QBITS_DT ACT_DT>
void parse_store(woq_config_param* p, woq_runtime_ctx* ctx) {
  auto constexpr ISA = GemmCore::ISA;
  if constexpr (ACT_DT == dispatcher_utils::QBITS_FP32) {  // DT of output always equal to ACT
    return parse_launcher<TASK, GemmCore, PrologueB, PrologueA, AlphaBetaProcessStoreFp32>(p, ctx);
  }
  if constexpr (ACT_DT == dispatcher_utils::QBITS_BF16) {
    return parse_launcher<TASK, GemmCore, PrologueB, PrologueA, AlphaBetaProcessStoreBf16>(p, ctx);
  }
}

template <WOQ_TASK TASK, class GemmCore, template <class _T, JBLAS_ISA> class PrologueB>
void parse_activation(woq_config_param* p, woq_runtime_ctx* ctx) {
  using namespace jblas::prologue_a::gemm;
  if (p->src_dt == dispatcher_utils::QBITS_FP32) {
    if constexpr (GemmCore::ISA == JblasAMX_INT8 || GemmCore::ISA == JblasAVX512_VNNI ||
                  GemmCore::ISA == JblasAVX_VNNI) {
      return parse_store<TASK, GemmCore, PrologueB, ActivationF32KBlockQuantize, dispatcher_utils::QBITS_FP32>(p, ctx);
    } else {
      if constexpr (GemmCore::ISA == JblasAMX_BF16) {
        return parse_store<TASK, GemmCore, PrologueB, ActivationConverterFp32, dispatcher_utils::QBITS_FP32>(p, ctx);
      } else {
        return parse_store<TASK, GemmCore, PrologueB, ActivationBase, dispatcher_utils::QBITS_FP32>(p, ctx);
      }
    }
  }
  if (p->src_dt == dispatcher_utils::QBITS_BF16) {
    if constexpr (GemmCore::ISA == JblasAMX_INT8 || GemmCore::ISA == JblasAVX512_VNNI ||
                  GemmCore::ISA == JblasAVX_VNNI) {
      return parse_store<TASK, GemmCore, PrologueB, ActivationBf16KBlockQuantize, dispatcher_utils::QBITS_BF16>(p, ctx);
    } else {
      if constexpr (GemmCore::ISA == JblasAMX_BF16) {
        return parse_store<TASK, GemmCore, PrologueB, ActivationBase, dispatcher_utils::QBITS_BF16>(p, ctx);
      } else {
        return parse_store<TASK, GemmCore, PrologueB, ActivationConverterBf16, dispatcher_utils::QBITS_BF16>(p, ctx);
      }
    }
  }
}

template <WOQ_TASK TASK, class GemmCore>
void parse_weight(woq_config_param* p, woq_runtime_ctx* ctx) {
  using namespace jblas::prologue_b::gemm;
  if (p->weight_type == "s8_scalef32") {
    return parse_activation<TASK, GemmCore, WeightKBlockS8>(p, ctx);
  }
  if (p->weight_type == "s4clip_scalef32" || p->weight_type == "s4fullrange_scalef32") {
    return parse_activation<TASK, GemmCore, WeightKBlockS4>(p, ctx);
  }
  if (p->weight_type == "nf4_scalef32" || p->weight_type == "fp4bnb_scalef32" || p->weight_type == "fp4e2m1_scalef32") {
    if constexpr (GemmCore::ISA != JblasAMX_INT8 && GemmCore::ISA != JblasAVX512_VNNI && GemmCore::ISA != JblasAVX_VNNI)
      return parse_activation<TASK, GemmCore, WeightKBlockF4>(p, ctx);
  }
  TORCH_CHECK(false,
              "Qbits: unsupported jblas_config, compute_type==" + p->compute_type + " weight_type==" + p->weight_type);
}

template <WOQ_TASK TASK>
void parse_gemm_core_online(woq_config_param* p, woq_runtime_ctx* ctx) {
  set_nk(ctx, ctx->weight);
  ctx->blocksize = ctx->blocksize == -1 ? ctx->k : ctx->blocksize;
  if (p->compute_type == "int8") {
    if (dispatcher_utils::check_amx() && ctx->blocksize % jblas::gemm::ICoreRowNAmxint8KBlock<48, 16>::KTILE == 0) {
      return parse_weight<TASK, jblas::gemm::ICoreRowNAmxint8KBlock<48, 16>>(p, ctx);
    }
    if (dispatcher_utils::check_avx512_vnni() &&
        ctx->blocksize % jblas::gemm::ICoreRowNAvx512vnniKBlock<48, 4>::KTILE == 0) {
      return parse_weight<TASK, jblas::gemm::ICoreRowNAvx512vnniKBlock<48, 4>>(p, ctx);
    }
    if (dispatcher_utils::check_avx_vnni() && ctx->blocksize % jblas::gemm::ICoreRowNAvxvnniKBlock<48, 2>::KTILE == 0) {
      return parse_weight<TASK, jblas::gemm::ICoreRowNAvxvnniKBlock<48, 2>>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: Illegal config in int8 compute_type, blocksize:", ctx->blocksize,
                ", ISA support vnni:", dispatcher_utils::check_avx_vnni());
  }
  if (p->compute_type == "fp32") {
    if (dispatcher_utils::check_avx512f()) {
      return parse_weight<TASK, jblas::gemm::SCoreRowNAvx512f<48, 8>>(p, ctx);
    }
    if (dispatcher_utils::check_avx2()) {
      return parse_weight<TASK, jblas::gemm::SCoreRowNAvx2<48, 2>>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: device ISA must support AVX2 when compute_type==fp32");
  }
  if (p->compute_type == "bf16") {
    if (dispatcher_utils::check_amx()) {
      return parse_weight<TASK, jblas::gemm::HCoreRowNAmxbf16<64, 16>>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: device ISA must support AMX-BF16 when compute_type==bf16");
  }
  TORCH_CHECK(false, "Qbits: unsupported jblas_config, compute_type:", p->compute_type,
              ", weight_type:", p->weight_type + ", blocksize:", ctx->blocksize);
}

template <WOQ_TASK TASK>
void parse_gemm_core_offline(woq_config_param* p, woq_runtime_ctx* ctx) {
  ctx->deseries_wei = jblas::storage::gemm::PackedWeightParser::deserialBuffer(ctx->weight->data_ptr());
  ctx->blocksize = dynamic_cast<jblas::storage::gemm::IWeightKBlockBase*>(ctx->deseries_wei)->mBlockSize;
  auto NTile = jblas::gemm::CoreAttr::get_mask_val(ctx->deseries_wei->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK,
                                                   jblas::gemm::CoreAttr::NTILE_SHIFT);
  auto CType = jblas::gemm::CoreAttr::get_mask_val(ctx->deseries_wei->mCoreId, jblas::gemm::CoreAttr::COMP_MASK,
                                                   jblas::gemm::CoreAttr::COMP_SHIFT);
  if (CType == uint32_t(jblas::gemm::CompType::COMP_INT8_US_INT32)) {
    if (NTile == jblas::gemm::ICoreRowNAmxint8KBlock<48, 16>::NTILE && dispatcher_utils::check_amx()) {
      return parse_weight<TASK, jblas::gemm::ICoreRowNAmxint8KBlock<48, 16>>(p, ctx);
    }
  }
  if (CType == uint32_t(jblas::gemm::CompType::COMP_INT8_US_FP32)) {
    if (NTile == jblas::gemm::ICoreRowNAvx512vnniKBlock<48, 4>::NTILE && dispatcher_utils::check_avx512_vnni()) {
      return parse_weight<TASK, jblas::gemm::ICoreRowNAvx512vnniKBlock<48, 4>>(p, ctx);
    }
    if (NTile == jblas::gemm::ICoreRowNAvxvnniKBlock<48, 2>::NTILE && dispatcher_utils::check_avx_vnni()) {
      return parse_weight<TASK, jblas::gemm::ICoreRowNAvxvnniKBlock<48, 2>>(p, ctx);
    }
  }
  if (CType == uint32_t(jblas::gemm::CompType::COMP_FP32)) {
    if (NTile == jblas::gemm::SCoreRowNAvx512f<48, 8>::NTILE && dispatcher_utils::check_avx512f()) {
      return parse_weight<TASK, jblas::gemm::SCoreRowNAvx512f<48, 8>>(p, ctx);
    }
    if (NTile == jblas::gemm::SCoreRowNAvx2<48, 2>::NTILE && dispatcher_utils::check_avx2()) {
      return parse_weight<TASK, jblas::gemm::SCoreRowNAvx2<48, 2>>(p, ctx);
    }
  }
  if (CType == uint32_t(jblas::gemm::CompType::COMP_BF16_FP32)) {
    if (NTile == jblas::gemm::HCoreRowNAmxbf16<64, 16>::NTILE && dispatcher_utils::check_amx()) {
      return parse_weight<TASK, jblas::gemm::HCoreRowNAmxbf16<64, 16>>(p, ctx);
    }
  }
  TORCH_CHECK(false, "Qbits: parse packweight fail, NTile:", NTile, ", CType:", CType,
              ", AMX:", dispatcher_utils::check_amx(), ", AVX512_VNNI:", dispatcher_utils::check_avx512_vnni(),
              ", AVX_VNNI:", dispatcher_utils::check_avx_vnni(), ", AVX512F:", dispatcher_utils::check_avx512f(),
              ", AVX2:", dispatcher_utils::check_avx2());
}

template <WOQ_TASK TASK>
void parse_gemm_core(woq_config_param* p, woq_runtime_ctx* ctx) {
  if constexpr (TASK == WOQ_QUANTIZE)
    return parse_gemm_core_online<TASK>(p, ctx);
  else
    return parse_gemm_core_offline<TASK>(p, ctx);
}

void dispatch_woq_task(woq_config_param* p, woq_runtime_ctx* ctx, WOQ_TASK task) {
  switch (task) {
    case WOQ_QUANTIZE:
      return parse_gemm_core<WOQ_QUANTIZE>(p, ctx);
    case WOQ_DEQUANTIZE:
      return parse_gemm_core<WOQ_DEQUANTIZE>(p, ctx);
    case WOQ_LINEAR:
      return parse_gemm_core<WOQ_LINEAR>(p, ctx);
  }
}
void set_woq_workspace(torch::Tensor* workspace) {
  woq_workspace = workspace->data_ptr();
  workspace_size = workspace->element_size() * workspace->numel();
}
}  // namespace woq
