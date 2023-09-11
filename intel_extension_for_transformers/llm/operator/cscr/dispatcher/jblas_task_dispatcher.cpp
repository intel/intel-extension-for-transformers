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

#include "jblas_task_dispatcher.hpp"

#define INTERFACE_TEMPLATE                                            \
  template <class _Launcher_T, template <class _T> class _Parallel_T> \
  class Interface
#define LAUNCHER_TEMPLATE                                      \
  template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T,            \
            template <class _T, JBLAS_ISA> class _PrologueA_T, \
            template <class _T, JBLAS_ISA> class _PrologueB_T, \
            template <JBLAS_ISA> class _Epilogue_T>            \
  class Launcher

inline bool check_amx() {
  return jblas::utils::parallel::CpuDevice::getInstance()->AMX_BF16();
}
inline bool check_avx512_vnni() {
  return jblas::utils::parallel::CpuDevice::getInstance()->AVX512_VNNI();
}
inline bool check_avx512f() {
  return jblas::utils::parallel::CpuDevice::getInstance()->AVX512F();
}

class env_initer {
 public:
  env_initer() { if(check_amx()) jblas::utils::request_perm_xtile_data(); }
};
static env_initer initer;

inline void set_nk(qbits_runtime_ctx* ctx, torch::Tensor* tensor) {
  ctx->n = ctx->transpose ? tensor->sizes()[0] : tensor->sizes()[1];
  ctx->k = ctx->transpose ? tensor->sizes()[1] : tensor->sizes()[0];
}

template <class KERNEL>
void qbits_quantize(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using PrologueB = typename KERNEL::WeightType;
  static PrologueB compress_kernel;
  set_nk(ctx, ctx->weight);

  auto ptr = (typename PrologueB::StorageWeight*)compress_kernel.createStorage(
      ctx->n, ctx->k, ctx->blocksize);
  if (ctx->transpose)
    compress_kernel.packTransposeWeight(
        ctx->n, ctx->k, ctx->weight->data_ptr<float>(), ctx->k, ptr);
  else
    compress_kernel.packWeight(ctx->n, ctx->k, ctx->weight->data_ptr<float>(),
                               ctx->n, ptr);
  auto size = ptr->getSerializedSize();
  *(ctx->output) = torch::zeros(size, torch::kInt8);
  ptr->serializeToBuffer(ctx->output->data_ptr<int8_t>());
}

template <class KERNEL>
void qbits_dequantize(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using PrologueB = typename KERNEL::WeightType;
  static PrologueB decompress_kernel;
  set_nk(ctx, ctx->output);
  auto parse_wei =
      dynamic_cast<typename PrologueB::StorageWeight*>(ctx->deseries_wei);
  TORCH_CHECK(parse_wei != nullptr, "unresolved compressed weight.");
  if (ctx->transpose)
    decompress_kernel.unpackTransposeWeight(
        ctx->n, ctx->k, parse_wei, ctx->output->data_ptr<float>(), ctx->k);
  else
    decompress_kernel.unpackWeight(ctx->n, ctx->k, parse_wei,
                                   ctx->output->data_ptr<float>(), ctx->n);
}

template <class KERNEL, JBLAS_ISA ISA>
void qbits_gemm(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  static KERNEL gemm_kernel;
  if (p->src_dt == QBITS_FP32 && p->dst_dt == QBITS_FP32) {
    if constexpr (ISA != JblasAMX_INT8 && ISA != JblasAVX512_VNNI) {
      gemm_kernel.compute({ctx->m, ctx->n, ctx->k,
                           ctx->activation->data_ptr<float>(), ctx->lda,
                           ctx->deseries_wei, ctx->output->data_ptr<float>(),
                           ctx->bias->data_ptr<float>(), ctx->ldo, 0,
                           ctx->alpha, ctx->beta, NULL});
    } else {
      auto quantA = gemm_kernel.getActivationPtr()->createStorage(
          ctx->m, ctx->k, ctx->blocksize,
          NULL);  // TODO(zhe): pass by python user, config the workspace buffer
                  // & size.
      gemm_kernel.compute({ctx->m, ctx->n, ctx->k,
                           ctx->activation->data_ptr<float>(), ctx->lda, quantA,
                           ctx->deseries_wei, ctx->output->data_ptr<float>(),
                           ctx->bias->data_ptr<float>(), ctx->ldo, 0,
                           ctx->alpha, ctx->beta, NULL});
      delete quantA;
    }
    return;
  }
  TORCH_CHECK(false, "unsupported src & dst data_type combination.")
}

template <QBITS_TASK TASK, class KERNEL, JBLAS_ISA ISA>
void execute_task(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  switch (TASK) {
    case QBITS_QUANTIZE:
      return qbits_quantize<KERNEL>(p, ctx);
    case QBITS_DEQUANTIZE:
      return qbits_dequantize<KERNEL>(p, ctx);
    case QBITS_LINEAR:
      return qbits_gemm<KERNEL, ISA>(p, ctx);
  }
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE,
          class Gemmcore, template <class _T> class Parallel, JBLAS_ISA ISA,
          template <class _T, JBLAS_ISA> class PrologueB,
          template <class _T, JBLAS_ISA> class PrologueA>
void parse_store(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  if (p->dst_dt == QBITS_FP32) {
    using namespace jblas::epilogue::gemm;
    return execute_task<TASK,
                        Interface<Launcher<ISA, Gemmcore, PrologueA, PrologueB,
                                           AlphaBetaProcessFp32>,
                                  Parallel>,
                        ISA>(p, ctx);
  }
  TORCH_CHECK(false, "unsupported dst data type.");
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE,
          class Gemmcore, template <class _T> class Parallel, JBLAS_ISA ISA,
          template <class _T, JBLAS_ISA> class PrologueB>
void parse_activation(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using namespace jblas::prologue::gemm;
  if (p->compute_type == "int8" && p->src_dt == QBITS_FP32) {
    if constexpr (ISA == JblasAMX_INT8)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA,
                         PrologueB, ActivationF32S8KBlockQuantize>(p, ctx);
    if constexpr (ISA == JblasAVX512_VNNI)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA,
                         PrologueB, ActivationF32U8KBlockQuantize>(p, ctx);
  }
  if (p->compute_type == "fp32") {
    if constexpr (ISA == JblasAVX512F)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA,
                         PrologueB, ActivationBase>(p, ctx);
  }
  if (p->compute_type == "bf16") {
    if constexpr (ISA == JblasAMX_BF16)
      return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA,
                         PrologueB, ActivationConverterFp32>(p, ctx);
  }
  TORCH_CHECK(false, "unsupported src data type.");
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE,
          class Gemmcore, template <class _T> class Parallel, JBLAS_ISA ISA>
void parse_weight(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using namespace jblas::prologue::weight_comp::gemm_kblcok;
  if (p->weight_type == "s8_scalef32") {
    if constexpr (ISA != JblasAMX_BF16)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel,
                              ISA, WeightS8ScaleFp32>(p, ctx);
  }
  if (p->weight_type == "s4clip_scalef32") {
    return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA,
                            WeightS4ClipScaleFp32>(p, ctx);
  }
  if (p->weight_type == "s4fullrange_scalef32") {
    return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA,
                            WeightS4FullRangeScaleFp32>(p, ctx);
  }
  if (p->weight_type == "fp4bnb_scalef32") {
    if constexpr (ISA != JblasAMX_INT8 && ISA != JblasAVX512_VNNI)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel,
                              ISA, WeightFp4BnbScaleFp32>(p, ctx);
  }
  if (p->weight_type == "fp4e2m1_scalef32") {
    if constexpr (ISA != JblasAMX_INT8 && ISA != JblasAVX512_VNNI)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel,
                              ISA, WeightFp4E2M1ScaleFp32>(p, ctx);
  }
  if (p->weight_type == "nf4_scalef32") {
    if constexpr (ISA != JblasAMX_INT8 && ISA != JblasAVX512_VNNI)
      return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel,
                              ISA, WeightNf4ScaleFp32>(p, ctx);
  }
  TORCH_CHECK(false, "unsupported jblas_config, compute_type==" +
                         p->compute_type + " weight_type==" + p->weight_type);
}

template <QBITS_TASK TASK>
void parse_gemm_core_online(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  if (p->compute_type == "int8") {
    if (check_amx()) {
      if (ctx->blocksize % 128 == 0)
        return parse_weight<
            TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
            jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
            jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
            jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAMX_INT8>(
            p, ctx);
      else
        return parse_weight<
            TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
            jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
            jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
            jblas::utils::parallel::Parallel2DGemmKBlockFixed,
            JblasAVX512_VNNI>(p, ctx);
    }
    if (check_avx512_vnni()) {
      return parse_weight<
          TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
          jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
          jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
          jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAVX512_VNNI>(
          p, ctx);
    }
    TORCH_CHECK(false,
                "device ISA must lagger than VNNI when compute_type==int8");
  }
  if (p->compute_type == "fp32") {
    if (check_avx512f()) {
      return parse_weight<
          TASK, jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight,
          jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
          jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
          jblas::utils::parallel::Parallel2DGemm, JblasAVX512F>(p, ctx);
    }
    TORCH_CHECK(false,
                "device ISA must lagger than AVX512F when compute_type==fp32");
  }
  if (p->compute_type == "bf16") {
    if (check_amx()) {
      return parse_weight<
          TASK, jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight,
          jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
          jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
          jblas::utils::parallel::Parallel2DGemm, JblasAMX_BF16>(p, ctx);
    }
    TORCH_CHECK(false,
                "device ISA must support AMX-BF16 when compute_type==bf16");
  }
  TORCH_CHECK(false, "unsupported jblas_config, compute_type==" +
                         p->compute_type + " weight_type==" + p->weight_type);
}
template <QBITS_TASK TASK>
void parse_gemm_core_offline(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  ctx->deseries_wei =
      jblas::prologue::weight_comp::gemm_kblcok::PackedWeightParser::
          deserialBuffer(ctx->weight->data_ptr<int8_t>(), false);
  auto gemm_core_type = ctx->deseries_wei->mCoreType;
  auto wbtmp = dynamic_cast<jblas::prologue::weight_comp::PackedWeightKBlock*>(
      ctx->deseries_wei);
  auto blocksize = wbtmp->mBlockSize;
  ctx->blocksize = blocksize;
  switch (gemm_core_type) {
    case jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK:
    case jblas::gemm::GemmCoreType::AVX512_VNNI_8X48:
    case jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK:
      assert(p->compute_type == "int8");
      if (check_amx() &&
          blocksize % (jblas::gemm::kblock::
                           GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK::KTILE *
                       2) ==
              0) {
        return parse_weight<
            TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
            jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
            jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
            jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAMX_INT8>(
            p, ctx);
      } else if (check_avx512_vnni() &&
                 blocksize %
                         (jblas::gemm::kblock::
                              GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK::KTILE *
                          2) ==
                     0) {
        return parse_weight<
            TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
            jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
            jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
            jblas::utils::parallel::Parallel2DGemmKBlockFixed,
            JblasAVX512_VNNI>(p, ctx);
      }
      TORCH_CHECK(false,
                  "Illegal config in int8 compute_type: blocksize:", blocksize,
                  " ISA largger than vnni:", check_avx512_vnni());
      break;
    case jblas::gemm::GemmCoreType::AVX512F_8X48:
      assert(p->compute_type == "fp32");
      if (check_avx512f()) {
        return parse_weight<
            TASK, jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight,
            jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
            jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
            jblas::utils::parallel::Parallel2DGemm, JblasAVX512F>(p, ctx);
      }
      TORCH_CHECK(
          false, "device ISA must lagger than AVX512F when compute_type==fp32");
      break;
    case jblas::gemm::GemmCoreType::AMX_BF16_16x64:
      assert(p->compute_type == "bf16");
      if (check_amx()) {
        return parse_weight<
            TASK, jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight,
            jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight,
            jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
            jblas::utils::parallel::Parallel2DGemm, JblasAMX_BF16>(p, ctx);
      }
      TORCH_CHECK(false,
                  "device ISA must support AMX-BF16 when compute_type==bf16");
    default:
      break;
  }

  TORCH_CHECK(false, "unrecognized gemm core");
}

template <QBITS_TASK TASK>
void parse_gemm_core(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  if constexpr (TASK == QBITS_QUANTIZE)
    return parse_gemm_core_online<TASK>(p, ctx);
  else
    return parse_gemm_core_offline<TASK>(p, ctx);
}

void task_dispatcher(qbits_config_param* p, qbits_runtime_ctx* ctx,
                     QBITS_TASK task) {
  if (task == QBITS_QUANTIZE) return parse_gemm_core<QBITS_QUANTIZE>(p, ctx);
  if (task == QBITS_DEQUANTIZE)
    return parse_gemm_core<QBITS_DEQUANTIZE>(p, ctx);
  if (task == QBITS_LINEAR) return parse_gemm_core<QBITS_LINEAR>(p, ctx);
}