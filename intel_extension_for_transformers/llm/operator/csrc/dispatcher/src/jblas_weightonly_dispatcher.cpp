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
#include "../include/dispatcher_utils.hpp"
#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include "jblas/jit_blas.h"
#include "jblas/jit_blas_epilogue.h"
#include "jblas/jit_blas_gemm.h"
#include "jblas/jit_blas_prologue.h"
#include "jblas/jit_blas_utils.h"
#include "jblas/jit_blas_weight_compression.h"
#include "jblas/jit_blas_wrapper.h"

#define INTERFACE_TEMPLATE                                            \
  template <class _Launcher_T, template <class _T> class _Parallel_T> \
  class Interface
#define LAUNCHER_TEMPLATE                                                                              \
  template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, JBLAS_ISA> class _PrologueA_T, \
            template <class _T, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T> \
  class Launcher

class env_initer {
 public:
  env_initer() {
    if (check_amx()) jblas::utils::request_perm_xtile_data();
    verbose = std::getenv("QBITS_VERBOSE") != nullptr;
    FLAGS_caffe2_log_level = 0;
  }
  bool verbose;
};
static env_initer initer;

template <typename T>
concept quant_PrologueA = requires {
  requires !std::is_same_v<T, float>;
  requires !std::is_same_v<T, jblas::utils::bf16>;
};

template <typename T>
concept normal_PrologueA = requires {
  requires !std::is_same_v<T, int8_t>;
  requires !std::is_same_v<T, uint8_t>;
};

template <typename T>
concept perchannel_Gemmcore = std::is_same_v<T, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI> ||
                              std::is_same_v<T, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8> ||
                              std::is_same_v<T, jblas::gemm::GemmCore_Row_NN_2x48_AVX_VNNI>;

template <typename T>
concept int8_cmptype_kblock_Gemmcore =
    std::is_same_v<T, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK> ||
    std::is_same_v<T, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK> ||
    std::is_same_v<T, jblas::gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK>;

static void* jblas_workspace = nullptr;
static int64_t workspace_size = 0;
static dispatcher_utils::Timer timer;

inline void set_nk(qbits_runtime_ctx* ctx, torch::Tensor* tensor) {
  ctx->n = ctx->transpose ? tensor->sizes()[0] : tensor->sizes()[1];
  ctx->k = ctx->transpose ? tensor->sizes()[1] : tensor->sizes()[0];
}

template <class KERNEL>
void qbits_quantize(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using PrologueB = typename KERNEL::WeightType;
  static PrologueB compress_kernel;
  set_nk(ctx, ctx->weight);

  if (initer.verbose) timer.start();
  auto do_quant = [&](typename PrologueB::StorageWeight* ptr) {
    *(ctx->output) = torch::zeros(ptr->mSize, torch::kInt8);
    ptr->assign(ctx->output->data_ptr<int8_t>());
    if (ctx->transpose)
      compress_kernel.packTransposeWeight(ctx->n, ctx->k, ctx->weight->data_ptr<float>(), ctx->k, ptr);
    else
      compress_kernel.packWeight(ctx->n, ctx->k, ctx->weight->data_ptr<float>(), ctx->n, ptr);
  };
  if constexpr (!perchannel_Gemmcore<typename KERNEL::GemmCore>) {
    auto storage = compress_kernel.createStorage(ctx->n, ctx->k, ctx->blocksize);
    do_quant(&storage);
  } else {
    auto storage = compress_kernel.createStorage(ctx->n, ctx->k, false);
    do_quant(&storage);
  }
  if (initer.verbose) {
    timer.stop();
    auto cost_time = timer.get_elapsed_time();
    LOG(INFO) << "QBits quantize verbose\nn:" << ctx->n << " k:" << ctx->k << " weight_type:" << p->weight_type
              << " blocksize:" << ctx->blocksize << " src_type:" << dispatcher_utils::get_torch_dt_name(ctx->weight)
              << " execute time:" << cost_time << "ms";
  }
}

template <class KERNEL>
void qbits_dequantize(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using PrologueB = typename KERNEL::WeightType;
  static PrologueB decompress_kernel;
  set_nk(ctx, ctx->output);
  if (initer.verbose) timer.start();
  if (ctx->transpose)
    decompress_kernel.unpackTransposeWeight(int(ctx->n), int(ctx->k), ctx->deseries_wei, ctx->output->data_ptr<float>(),
                                            int(ctx->k));
  else
    decompress_kernel.unpackWeight(int(ctx->n), int(ctx->k), ctx->deseries_wei, ctx->output->data_ptr<float>(),
                                   int(ctx->n));
  if (initer.verbose) {
    timer.stop();
    auto cost_time = timer.get_elapsed_time();
    LOG(INFO) << "QBits dequantize verbose\nn:" << ctx->n << " k:" << ctx->k << " weight_type:" << p->weight_type
              << " blocksize:" << ctx->blocksize << " dst_type:" << dispatcher_utils::get_torch_dt_name(ctx->output)
              << " execute time:" << cost_time << "ms";
  }
}

template <class KERNEL, class ParamA, class ParamC>
void do_compute(qbits_config_param* p, qbits_runtime_ctx* ctx, const ParamA param_a, const ParamC param_c) {
  if (initer.verbose) timer.start();
  static KERNEL gemm_kernel;
  if constexpr (!perchannel_Gemmcore<typename KERNEL::GemmCore>)
    gemm_kernel.compute({int(ctx->m), int(ctx->n), int(ctx->k), param_a, ctx->deseries_wei, param_c});
  else
    gemm_kernel.template compute<true, false>(
        {int(ctx->m), int(ctx->n), int(ctx->k), param_a, ctx->deseries_wei, param_c});
  if (initer.verbose) {
    timer.stop();
    auto cost_time = timer.get_elapsed_time();
    LOG(INFO) << "QBits linear verbose\nm:" << ctx->m << " n:" << ctx->n << " k:" << ctx->k
              << " weight_type:" << p->weight_type << " compute_type:" << p->compute_type
              << " blocksize:" << ctx->blocksize << " src_type:" << dispatcher_utils::get_torch_dt_name(ctx->activation)
              << " dst_type:" << dispatcher_utils::get_torch_dt_name(ctx->output) << " execute time:" << cost_time
              << "ms";
  }
}

template <class KERNEL, class ParamA>
void parse_paramC(qbits_config_param* p, qbits_runtime_ctx* ctx, ParamA param_a) {
  using ParamC = typename KERNEL::Epilogue::Param;
  if constexpr (!perchannel_Gemmcore<typename KERNEL::GemmCore>) {
    ParamC param_c = {ctx->output->data_ptr(), ctx->bias->data_ptr(), ctx->ldo, 0, ctx->alpha, ctx->beta};
    return do_compute<KERNEL, ParamA, ParamC>(p, ctx, param_a, param_c);
  } else {
    if constexpr (std::is_same_v<typename KERNEL::GemmCore, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8>) {
      ParamC param_c = {ctx->output->data_ptr(),
                        ctx->ldo,
                        param_a.Q->mSPtr,
                        param_a.Q->mCStep,
                        dynamic_cast<typename KERNEL::WeightType::StorageWeight*>(ctx->deseries_wei)->mSPtr,
                        ctx->bias->data_ptr(),
                        0,
                        ctx->alpha,
                        ctx->beta};
      return do_compute<KERNEL, ParamA, ParamC>(p, ctx, param_a, param_c);
    }
    if constexpr (std::is_same_v<typename KERNEL::GemmCore, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI> ||
                  std::is_same_v<typename KERNEL::GemmCore, jblas::gemm::GemmCore_Row_NN_2x48_AVX_VNNI>) {
      ParamC param_c = {ctx->output->data_ptr(),
                        ctx->ldo,
                        param_a.Q->mZPtr,
                        param_a.Q->mSPtr,
                        param_a.Q->mCStep,
                        dynamic_cast<typename KERNEL::WeightType::StorageWeight*>(ctx->deseries_wei)->mRPtr,
                        dynamic_cast<typename KERNEL::WeightType::StorageWeight*>(ctx->deseries_wei)->mSPtr,
                        ctx->bias->data_ptr(),
                        0,
                        ctx->alpha,
                        ctx->beta};
      return do_compute<KERNEL, ParamA, ParamC>(p, ctx, param_a, param_c);
    }
  }
}

template <class KERNEL>
void parse_paramA(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using PrologueA = typename KERNEL::ActivationType;
  using ParamA = typename PrologueA::Param;
  using SrcType = typename PrologueA::SRCType;
  if constexpr (normal_PrologueA<typename KERNEL::ActivationType::AType>) {
    ParamA param_a = {reinterpret_cast<SrcType*>(ctx->activation->data_ptr()), ctx->lda};
    return parse_paramC<KERNEL, ParamA>(p, ctx, param_a);
  }
  if constexpr (quant_PrologueA<typename KERNEL::ActivationType::AType>) {
    static KERNEL gemm_kernel;
    void* workspace = jblas_workspace == nullptr ? NULL : jblas_workspace;
    size_t need_size;
    void* tmpbuf = NULL;
    auto get_workspace = [&](int need_size) {
      if (workspace != NULL) {
        TORCH_CHECK(workspace_size >= need_size,
                    "Qbits: workspace size should large than " + std::to_string(need_size) + " bytes");
        return workspace;
      } else {
        tmpbuf = jblas::utils::amalloc<int8_t>(need_size);
        return tmpbuf;
      }
    };
    if constexpr (!perchannel_Gemmcore<typename KERNEL::GemmCore>) {
      auto quantA = gemm_kernel.getActivationPtr()->createStorage(ctx->m, ctx->k, ctx->blocksize);
      auto need_size = quantA.mSize;
      quantA.assign(reinterpret_cast<int8_t*>(get_workspace(need_size)));
      ParamA param_a = {reinterpret_cast<SrcType*>(ctx->activation->data_ptr()), ctx->lda, &quantA};
      parse_paramC<KERNEL, ParamA>(p, ctx, param_a);
    } else {
      auto quantA = gemm_kernel.getActivationPtr()->createStorage(ctx->m, ctx->k);
      auto need_size = quantA.mSize;
      quantA.assign(reinterpret_cast<int8_t*>(get_workspace(need_size)));
      ParamA param_a = {reinterpret_cast<SrcType*>(ctx->activation->data_ptr()), ctx->lda, &quantA};
      parse_paramC<KERNEL, ParamA>(p, ctx, param_a);
    }
    if (tmpbuf != NULL) jblas::utils::afree(tmpbuf);
  }
}

template <class KERNEL>
void qbits_gemm(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  return parse_paramA<KERNEL>(p, ctx);
}

template <QBITS_TASK TASK, class KERNEL>
void execute_task(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  switch (TASK) {
    case QBITS_QUANTIZE:
      return qbits_quantize<KERNEL>(p, ctx);
    case QBITS_DEQUANTIZE:
      return qbits_dequantize<KERNEL>(p, ctx);
    case QBITS_LINEAR:
      return qbits_gemm<KERNEL>(p, ctx);
  }
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE, class Gemmcore, template <class _T> class Parallel,
          JBLAS_ISA ISA, template <class _T, JBLAS_ISA> class PrologueB, template <class _T, JBLAS_ISA> class PrologueA>
void parse_store(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  if (p->dst_dt == QBITS_FP32) {
    using namespace jblas::epilogue::gemm;
    if constexpr (perchannel_Gemmcore<Gemmcore>) {
      if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI> ||
                    std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_2x48_AVX_VNNI>)
        return execute_task<
            TASK, Interface<Launcher<ISA, Gemmcore, PrologueA, PrologueB, ZpDequantInt32AlphaBetaStoreFp32>, Parallel>>(
            p, ctx);
      if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8>)
        return execute_task<
            TASK, Interface<Launcher<ISA, Gemmcore, PrologueA, PrologueB, DequantInt32AlphaBetaStoreFp32>, Parallel>>(
            p, ctx);
    } else {
      return execute_task<
          TASK, Interface<Launcher<ISA, Gemmcore, PrologueA, PrologueB, AlphaBetaProcessStoreFp32>, Parallel>>(p, ctx);
    }
  }
  if (p->dst_dt == QBITS_BF16) {
    if constexpr (perchannel_Gemmcore<Gemmcore>) {
      if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI> ||
                    std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_2x48_AVX_VNNI>)
        return execute_task<
            TASK, Interface<Launcher<ISA, Gemmcore, PrologueA, PrologueB, ZpDequantInt32AlphaBetaStoreBf16>, Parallel>>(
            p, ctx);
      if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8>)
        return execute_task<
            TASK, Interface<Launcher<ISA, Gemmcore, PrologueA, PrologueB, DequantInt32AlphaBetaStoreBf16>, Parallel>>(
            p, ctx);
    } else {
      return execute_task<
          TASK, Interface<Launcher<ISA, Gemmcore, PrologueA, PrologueB, AlphaBetaProcessStoreBf16>, Parallel>>(p, ctx);
    }
  }
  TORCH_CHECK(false, "Qbits: unsupported dst data type.");
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE, class Gemmcore, template <class _T> class Parallel,
          JBLAS_ISA ISA, template <class _T, JBLAS_ISA> class PrologueB>
void parse_activation(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using namespace jblas::prologue::gemm;
  if (p->src_dt == QBITS_FP32) {
    if constexpr (std::is_same_v<Gemmcore, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK>)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationF32S8KBlockQuantize>(
          p, ctx);
    if constexpr (std::is_same_v<Gemmcore, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK> ||
                  std::is_same_v<Gemmcore, jblas::gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK>)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationF32U8KBlockQuantize>(
          p, ctx);
    if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F> ||
                  std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_2x48_AVX2>)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationBase>(p, ctx);
    if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16>)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationConverterFp32>(p,
                                                                                                                 ctx);
    if constexpr (perchannel_Gemmcore<Gemmcore>) {
      if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8>)
        return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationFp32SymS8Quantize>(
            p, ctx);
      if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI> ||
                    std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_2x48_AVX_VNNI>)
        return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationFp32AsymU8Quantize>(
            p, ctx);
    }
  }
  if (p->src_dt == QBITS_BF16) {
    if constexpr (std::is_same_v<Gemmcore, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK>)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationBf16S8KBlockQuantize>(
          p, ctx);
    if constexpr (std::is_same_v<Gemmcore, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK>)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationBf16U8KBlockQuantize>(
          p, ctx);
    if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F> ||
                  std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_2x48_AVX2>)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationConverterBf16>(p,
                                                                                                                 ctx);
    if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16>)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationBase>(p, ctx);
    if constexpr (perchannel_Gemmcore<Gemmcore>) {
      if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8>)
        return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationBf16SymS8Quantize>(
            p, ctx);
      if constexpr (std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI> ||
                    std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_2x48_AVX_VNNI>)
        return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationBf16AsymU8Quantize>(
            p, ctx);
    }
  }
  TORCH_CHECK(false, "Qbits: unsupported src data type in current config, compute_type==" + p->compute_type +
                         " weight_type==" + p->weight_type);
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE, class Gemmcore, template <class _T> class Parallel,
          JBLAS_ISA ISA>
void parse_weight(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using namespace jblas::prologue::weight_comp::gemm_kblcok;
  if (p->weight_type == "s8_scalef32") {
    if constexpr (perchannel_Gemmcore<Gemmcore>)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, WeightS8ScaleFp32PerChannelN>(p, ctx);
    if constexpr (!std::is_same_v<Gemmcore, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16> &&
                  !perchannel_Gemmcore<Gemmcore>)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, WeightS8ScaleFp32>(p, ctx);
  }
  if (p->weight_type == "s4clip_scalef32") {
    if constexpr (perchannel_Gemmcore<Gemmcore>)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, WeightS4ClipScaleFp32PerN>(p, ctx);
    if constexpr (!perchannel_Gemmcore<Gemmcore>)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, WeightS4ClipScaleFp32>(p, ctx);
  }
  if (p->weight_type == "s4fullrange_scalef32") {
    if constexpr (!perchannel_Gemmcore<Gemmcore>)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, WeightS4FullRangeScaleFp32>(p, ctx);
  }
  if (p->weight_type == "fp4bnb_scalef32") {
    if constexpr (!perchannel_Gemmcore<Gemmcore> && !int8_cmptype_kblock_Gemmcore<Gemmcore>)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, WeightFp4BnbScaleFp32>(p, ctx);
  }
  if (p->weight_type == "fp4e2m1_scalef32") {
    if constexpr (!perchannel_Gemmcore<Gemmcore> && !int8_cmptype_kblock_Gemmcore<Gemmcore>)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, WeightFp4E2M1ScaleFp32>(p, ctx);
  }
  if (p->weight_type == "nf4_scalef32") {
    if constexpr (!perchannel_Gemmcore<Gemmcore> && !int8_cmptype_kblock_Gemmcore<Gemmcore>)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, WeightNf4ScaleFp32>(p, ctx);
  }
  TORCH_CHECK(false,
              "Qbits: unsupported jblas_config, compute_type==" + p->compute_type + " weight_type==" + p->weight_type);
}

template <QBITS_TASK TASK>
void parse_gemm_core_online(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  bool per_channel_quant = ctx->blocksize == -1 ? true : false;
  if (per_channel_quant) {
    TORCH_CHECK(p->compute_type == "int8", "Qbits: compute type must be int8 when enable per_channel quantization.");
    if (check_amx())
      return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB,
                          jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                          jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8, jblas::utils::parallel::Parallel2DGemm,
                          JblasAMX_INT8>(p, ctx);
    if (check_avx512_vnni())
      return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB,
                          jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                          jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, jblas::utils::parallel::Parallel2DGemm,
                          JblasAVX512_VNNI>(p, ctx);
    if (check_avx_vnni())
      return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB,
                          jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                          jblas::gemm::GemmCore_Row_NN_2x48_AVX_VNNI, jblas::utils::parallel::Parallel2DGemm,
                          JblasAVX_VNNI>(p, ctx);
  }
  if (p->compute_type == "int8") {
    if (check_amx()) {
      if (ctx->blocksize % (jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK::KTILE * 2) == 0)
        return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                            jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
                            jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
                            jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAMX_INT8>(p, ctx);
      else if (ctx->blocksize % (jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK::KTILE * 2) == 0)
        return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                            jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
                            jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
                            jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAVX512_VNNI>(p, ctx);
    }
    if (check_avx512_vnni() &&
        ctx->blocksize % (jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK::KTILE * 2) == 0) {
      return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                          jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
                          jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
                          jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAVX512_VNNI>(p, ctx);
    }
    if (check_avx_vnni() &&
        ctx->blocksize % (jblas::gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK::KTILE * 2) == 0) {
      return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                          jblas::wrapper::gemm_kblock::GemmLauncherKBlock,
                          jblas::gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK,
                          jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAVX_VNNI>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: Illegal config in int8 compute_type: blocksize:", ctx->blocksize,
                " ISA largger than vnni:", check_avx512_vnni());
  }
  if (p->compute_type == "fp32") {
    if (check_avx512f()) {
      return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight,
                          jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                          jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, jblas::utils::parallel::Parallel2DGemm,
                          JblasAVX512F>(p, ctx);
    }
    if (check_avx2()) {
      return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight,
                          jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                          jblas::gemm::GemmCore_Row_NN_2x48_AVX2, jblas::utils::parallel::Parallel2DGemm, JblasAVX2>(
          p, ctx);
    }
    TORCH_CHECK(false, "Qbits: device ISA must lagger than AVX2 when compute_type==fp32");
  }
  if (p->compute_type == "bf16") {
    if (check_amx()) {
      return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight,
                          jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                          jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16, jblas::utils::parallel::Parallel2DGemm,
                          JblasAMX_BF16>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: device ISA must support AMX-BF16 when compute_type==bf16");
  }
  TORCH_CHECK(false, "Qbits: unsupported jblas_config, compute_type==" + p->compute_type +
                         " weight_type==" + p->weight_type + " blocksize==" + std::to_string(ctx->blocksize));
}
template <QBITS_TASK TASK>
void parse_gemm_core_offline(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  ctx->deseries_wei =
      jblas::prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(ctx->weight->data_ptr<int8_t>());
  auto gemm_core_type = ctx->deseries_wei->mCoreType;
  auto blocksize = ctx->deseries_wei->mBlockSize;
  ctx->blocksize = blocksize;
  switch (gemm_core_type) {
    case jblas::gemm::GemmCoreType::AMX_INT8_16x48_KBLOCK:
    case jblas::gemm::GemmCoreType::AVX512_VNNI_3x48_KBLOCK:
      assert(p->compute_type == "int8");
      // TODO(zhe): potential bug, quantize in vnni machine, compute on amx machine.
      if (check_amx() && blocksize % (jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK::KTILE * 2) == 0) {
        return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                            jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
                            jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
                            jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAMX_INT8>(p, ctx);
      } else if (check_avx512_vnni() &&
                 blocksize % (jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK::KTILE * 2) == 0) {
        return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                            jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
                            jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
                            jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAVX512_VNNI>(p, ctx);
      }
      TORCH_CHECK(false, "Qbits: Illegal config in int8 compute_type: blocksize:", blocksize,
                  " ISA largger than avx512-vnni:", check_avx512_vnni());
      break;
    case jblas::gemm::GemmCoreType::AVX_VNNI_1x48_KBLOCK:
      assert(p->compute_type == "int8");
      if (check_avx_vnni() &&
          ctx->blocksize % (jblas::gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK::KTILE * 2) == 0)
        return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                            jblas::wrapper::gemm_kblock::GemmLauncherKBlock,
                            jblas::gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK,
                            jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAVX_VNNI>(p, ctx);
      TORCH_CHECK(false, "Qbits: Illegal config in int8 compute_type: blocksize:", blocksize,
                  " ISA largger than avx-vnni:", check_avx_vnni());
      break;
    case jblas::gemm::GemmCoreType::AVX512F_8x48:
      assert(p->compute_type == "fp32");
      if (check_avx512f()) {
        return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight,
                            jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                            jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, jblas::utils::parallel::Parallel2DGemm,
                            JblasAVX512F>(p, ctx);
      }
      TORCH_CHECK(false, "Qbits: device ISA must support AVX512F when GemmCore==AVX512F_8x48");
      break;
    case jblas::gemm::GemmCoreType::AVX2_2X48:
      assert(p->compute_type == "fp32");
      if (check_avx2()) {
        return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight,
                            jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                            jblas::gemm::GemmCore_Row_NN_2x48_AVX2, jblas::utils::parallel::Parallel2DGemm, JblasAVX2>(
            p, ctx);
      }
      TORCH_CHECK(false, "Qbits: device ISA must support AVX2 when GemmCore==AVX2_2x48");
      break;
    case jblas::gemm::GemmCoreType::AMX_BF16_16x64:
      assert(p->compute_type == "bf16");
      if (check_amx()) {
        return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight,
                            jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                            jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16, jblas::utils::parallel::Parallel2DGemm,
                            JblasAMX_BF16>(p, ctx);
      }
      TORCH_CHECK(false, "Qbits: device ISA must support AMX-BF16 when compute_type==bf16");
    case jblas::gemm::GemmCoreType::AVX512_VNNI_8x48:
      assert(p->compute_type == "int8");
      if (check_avx512_vnni())
        return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB,
                            jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                            jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, jblas::utils::parallel::Parallel2DGemm,
                            JblasAVX512_VNNI>(p, ctx);
      TORCH_CHECK(false, "Qbits: device ISA must lagger than AVX512_VNNI when GemmCore==Row_NN_8x48_AVX512_VNNI");
      break;
    case jblas::gemm::GemmCoreType::AVX_VNNI_2x48:
      assert(p->compute_type == "int8");
      if (check_avx_vnni())
        return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB,
                            jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                            jblas::gemm::GemmCore_Row_NN_2x48_AVX_VNNI, jblas::utils::parallel::Parallel2DGemm,
                            JblasAVX_VNNI>(p, ctx);
      TORCH_CHECK(false, "Qbits: device ISA must lagger than AVX_VNNI when GemmCore==Row_NN_2x48_AVX_VNNI");
      break;
    case jblas::gemm::GemmCoreType::AMX_INT8_16x48:
      if (check_amx())
        return parse_weight<TASK, jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB,
                            jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
                            jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8, jblas::utils::parallel::Parallel2DGemm,
                            JblasAMX_INT8>(p, ctx);
      TORCH_CHECK(false, "Qbits: device ISA must support AMX_INT8 when GemmCore==Row_NN_16x48_AMX_S8S8");
    default:
      break;
  }

  TORCH_CHECK(false, "Qbits: unrecognized gemm core");
}

template <QBITS_TASK TASK>
void parse_gemm_core(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  if constexpr (TASK == QBITS_QUANTIZE)
    return parse_gemm_core_online<TASK>(p, ctx);
  else
    return parse_gemm_core_offline<TASK>(p, ctx);
}

void task_dispatcher(qbits_config_param* p, qbits_runtime_ctx* ctx, QBITS_TASK task) {
  if (task == QBITS_QUANTIZE) return parse_gemm_core<QBITS_QUANTIZE>(p, ctx);
  if (task == QBITS_DEQUANTIZE) return parse_gemm_core<QBITS_DEQUANTIZE>(p, ctx);
  if (task == QBITS_LINEAR) return parse_gemm_core<QBITS_LINEAR>(p, ctx);
}

void set_jblas_workspace(torch::Tensor* workspace) {
  jblas_workspace = workspace->data_ptr();
  workspace_size = workspace->element_size() * workspace->numel();
}
