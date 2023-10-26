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
#include "jblas_common.hpp"

using namespace jblas;
using namespace ne_jblas;

unsigned long long jblas_f32f32_get_workspace_size(int _m, int _n, int _k, void* wptr) {
  // maximum padding
  int constexpr padding = 128;
  size_t s = size_t(_m) * utils::padto((size_t)_k, padding) * 4;
  return s;
}

namespace {

namespace avx512f {
JBLAS_ISA constexpr DefaultISA = JblasAVX512F;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using Default = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
                                                             jblas::prologue::gemm::ActivationBase, ProB,
                                                             jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemm>;

}  // namespace avx512f

namespace amx_bf16 {
JBLAS_ISA constexpr DefaultISA = JblasAMX_BF16;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using Default = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<DefaultISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
                                                             jblas::prologue::gemm::ActivationConverterFp32, ProB,
                                                             jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemm>;

}  // namespace amx_bf16

namespace amx_int8 {
JBLAS_ISA constexpr DefaultISA = JblasAMX_INT8;
template <template <class GC, JBLAS_ISA ISA> class ProB>
using KBlockFp32Fp32 = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
        jblas::prologue::gemm::ActivationF32S8KBlockQuantize, ProB, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using PerNFp32Fp32 = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<DefaultISA, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8,
                                                             jblas::prologue::gemm::ActivationFp32SymS8Quantize, ProB,
                                                             jblas::epilogue::gemm::DequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;

}  // namespace amx_int8

namespace avx512_vnni {
JBLAS_ISA constexpr DefaultISA = JblasAVX512_VNNI;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using KBlockFp32Fp32 = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmLauncherKBlock<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_4x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize, ProB, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using KBlockFp32Fp32Next = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize, ProB, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using PerNFp32Fp32 = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,
                                                             jblas::prologue::gemm::ActivationFp32AsymU8Quantize, ProB,
                                                             jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;
}  // namespace avx512_vnni

namespace avx_vnni {
JBLAS_ISA constexpr DefaultISA = JblasAVX_VNNI;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using KBlockFp32Fp32 = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmLauncherKBlock<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize, ProB, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

// template <template <class GC, JBLAS_ISA ISA> class ProB>
// using KBlockFp32Fp32Next = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
//     jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
//         DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
//         jblas::prologue::gemm::ActivationF32U8KBlockQuantize, ProB, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
//     jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

// template <template <class GC, JBLAS_ISA ISA> class ProB>
// using PerNFp32Fp32 = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
//     jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<DefaultISA,
//     jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,
//                                                              jblas::prologue::gemm::ActivationFp32AsymU8Quantize,
//                                                              ProB, jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
//     jblas::utils::parallel::Parallel2DGemm>;
}  // namespace avx_vnni

namespace avx2 {
JBLAS_ISA constexpr DefaultISA = JblasAVX2;
template <template <class GC, JBLAS_ISA ISA> class ProB>
using Default = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<DefaultISA, jblas::gemm::GemmCore_Row_NN_2x48_AVX2,
                                                             jblas::prologue::gemm::ActivationBase, ProB,
                                                             jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemm>;
}  // namespace avx2
}  // namespace

static JBLAS_CODE jblas_s4fp32kblock_f32f32_forward(float* activation, SS4Fp32* weiptr, float* output, int _m, int _n,
                                                    int _k, int lda, int ldo, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (weiptr->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && weiptr->mBlockSize % 128 == 0) {
      using GemmKernel = amx_int8::KBlockFp32Fp32<WeiS4ClipFp32>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, ldo});
    } else {
      if (weiptr->mBlockSize % 8 == 0) {
        if (_cd->AVX512_VNNI()) {
          if (_m <= 32) {
            using GemmKernel = avx512_vnni::KBlockFp32Fp32Next<WeiS4ClipFp32>;
            static GemmKernel kernel;
            auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
            quanA.assign((int8_t*)workspace);
            ret = kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, ldo});
          } else {
            using GemmKernel = avx512_vnni::KBlockFp32Fp32<WeiS4ClipFp32>;
            static GemmKernel kernel;
            auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
            quanA.assign((int8_t*)workspace);
            ret = kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, ldo});
          }
        } else if (_cd->AVX_VNNI()) {
          using GemmKernel = avx_vnni::KBlockFp32Fp32<WeiS4ClipFp32>;
          static GemmKernel kernel;
          auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
          quanA.assign((int8_t*)workspace);
          ret = kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, ldo});
        }
      }
    }
  } else if (weiptr->mCoreType == GcCompFp32::TYPE) {
    if (_cd->AVX512F()) {
      using GemmKernel = avx512f::Default<WeiS4ClipFp32>;
      static GemmKernel kernel;
      ret = kernel.compute({_m, _n, _k, activation, lda, weiptr, output, ldo});
    } else if (_cd->AVX2()) {
      using GemmKernel = avx2::Default<WeiS4ClipFp32>;
      static GemmKernel kernel;
      ret = kernel.compute({_m, _n, _k, activation, lda, weiptr, output, ldo});
    }
  } else if (weiptr->mCoreType == GcCompBf16::TYPE) {
    if (_cd->AMX_BF16()) {
      using GemmKernel = amx_bf16::Default<WeiS4ClipFp32>;
      static GemmKernel kernel;
      ret = kernel.compute({_m, _n, _k, activation, lda, weiptr, output, ldo});
    }
  }
  return ret;
}

static JBLAS_CODE jblas_s8fp32kblock_f32f32_forward(float* activation, SS8Fp32* weiptr, float* output, int _m, int _n,
                                                    int _k, int lda, int ldo, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (weiptr->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && weiptr->mBlockSize % 128 == 0) {
      using GemmKernel = amx_int8::KBlockFp32Fp32<WeiS8Fp32>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, ldo});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = avx512_vnni::KBlockFp32Fp32<WeiS8Fp32>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, ldo});
    } else if (_cd->AVX_VNNI()) {
      using GemmKernel = avx_vnni::KBlockFp32Fp32<WeiS8Fp32>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, ldo});
    }
  } else if (weiptr->mCoreType == GcCompFp32::TYPE) {
    if (_cd->AVX512F()) {
      using GemmKernel = avx512f::Default<WeiS8Fp32>;
      static GemmKernel kernel;
      ret = kernel.compute({_m, _n, _k, activation, lda, weiptr, output, ldo});
    } else if (_cd->AVX2()) {
      using GemmKernel = avx2::Default<WeiS8Fp32>;
      static GemmKernel kernel;
      ret = kernel.compute({_m, _n, _k, activation, lda, weiptr, output, ldo});
    }
  }
  return ret;
}

static JBLAS_CODE jblas_s8fp32perN_f32f32_forward(float* activation, SS8Fp32PerN* weiptr, float* output, int _m, int _n,
                                                  int _k, int lda, int ldo, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  assert(weiptr->mBlockSize == _k);
  if (weiptr->mCoreType == GcCompInt8::TYPE) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = amx_int8::PerNFp32Fp32<WeiS8Fp32PerN>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute<true, false>(
          {_m, _n, _k, activation, lda, &quanA, weiptr, output, ldo, quanA.mSPtr, quanA.mCStep, weiptr->mSPtr});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = avx512_vnni::PerNFp32Fp32<WeiS8Fp32PerN>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute<true, false>(
          {_m,
           _n,
           _k,
           activation,
           lda,
           &quanA,
           weiptr,
           {output, ldo, quanA.mCStep, quanA.mSPtr, weiptr->mSPtr, quanA.mZPtr, weiptr->mRPtr}});
    }
  }
  return ret;
}

static JBLAS_CODE jblas_s4fp32perN_f32f32_forward(float* activation, SS4Fp32PerN* weiptr, float* output, int _m, int _n,
                                                  int _k, int lda, int ldo, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  assert(weiptr->mBlockSize == _k);
  if (weiptr->mCoreType == GcCompInt8::TYPE) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = amx_int8::PerNFp32Fp32<WeiS4ClipFp32PerN>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute<true, false>(
          {_m, _n, _k, activation, lda, &quanA, weiptr, output, ldo, quanA.mSPtr, quanA.mCStep, weiptr->mSPtr});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = avx512_vnni::PerNFp32Fp32<WeiS4ClipFp32PerN>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute<true, false>(
          {_m,
           _n,
           _k,
           activation,
           lda,
           &quanA,
           weiptr,
           {output, ldo, quanA.mCStep, quanA.mSPtr, weiptr->mSPtr, quanA.mZPtr, weiptr->mRPtr}});
    }
  }
  return ret;
}

// f32f32: activation & output dtype
void jblas_f32f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda, int ldo,
                          void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto wtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(weiptr);
  if (wtmp != nullptr) {
    if (wtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32)) {
      ret = jblas_s4fp32kblock_f32f32_forward(activation, dynamic_cast<SS4Fp32*>(wtmp), output, _m, _n, _k, lda, ldo,
                                              workspace);
    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32)) {
      ret = jblas_s8fp32kblock_f32f32_forward(activation, dynamic_cast<SS8Fp32*>(wtmp), output, _m, _n, _k, lda, ldo,
                                              workspace);
    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
      ret = jblas_s8fp32perN_f32f32_forward(activation, dynamic_cast<SS8Fp32PerN*>(wtmp), output, _m, _n, _k, lda, ldo,
                                            workspace);
    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
      ret = jblas_s4fp32perN_f32f32_forward(activation, dynamic_cast<SS4Fp32PerN*>(wtmp), output, _m, _n, _k, lda, ldo,
                                            workspace);
    }
  }
  assert(ret == JblasSuccess);
  safe_delete(wtmp);
}

bool jblas_fusion_add_f32f32_support(void* weiptr, int _m, int _n, int _k) {
  GetCPUDevice();
  bool support = false;
  auto wtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(weiptr);
  if (wtmp) {
    if (wtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32)) {
      constexpr size_t EleNum = sizeof(GcCompInt8KBlockSet) / sizeof(GcCompInt8KBlockSet[0]);
      support = contains(wtmp->mCoreType, GcCompInt8KBlockSet, EleNum);
      support &= hasISA(GcCompInt8KBlockSet, EleNum);
    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32)) {
      constexpr size_t EleNum = sizeof(GcCompInt8KBlockSet) / sizeof(GcCompInt8KBlockSet[0]);
      support = contains(wtmp->mCoreType, GcCompInt8KBlockSet, EleNum);
      support &= hasISA(GcCompInt8KBlockSet, EleNum);
    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
      constexpr size_t EleNum = sizeof(GcCompInt8Set) / sizeof(GcCompInt8Set[0]);
      support = contains(wtmp->mCoreType, GcCompInt8Set, EleNum);
      support &= hasISA(GcCompInt8Set, EleNum);
    }
  }
  safe_delete(wtmp);
  return support;
}

JBLAS_CODE jblas_fusion_add_s4fp32_f32f32_forward(float* activation, SS4Fp32* weiptr, float* bias, float* output,
                                                  int _m, int _n, int _k, int lda, int ldo, bool broadcast_bias,
                                                  void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (weiptr->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && weiptr->mBlockSize % 128 == 0) {
      using GemmKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
          custom::wrapper::kblock::amx_int8::AddGemmSKernelDynamicS4KBlock,
          jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, bias, ldo, broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX512_VNNI() && weiptr->mBlockSize % 8 == 0) {
      if (_m <= 32) {
        using GemmKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
            custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS4KBlockNext,
            jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
        static GemmKernel kernel;
        auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
        quanA.assign((int8_t*)workspace);
        ret =
            kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, bias, ldo, broadcast_bias ? 0 : ldo});
      } else {
        using GemmKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
            custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS4KBlock,
            jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
        static GemmKernel kernel;
        auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
        quanA.assign((int8_t*)workspace);
        ret =
            kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, bias, ldo, broadcast_bias ? 0 : ldo});
      }
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_add_s8fp32_f32f32_forward(float* activation, SS8Fp32* weiptr, float* bias, float* output,
                                                  int _m, int _n, int _k, int lda, int ldo, bool broadcast_bias,
                                                  void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (weiptr->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && weiptr->mBlockSize % 128 == 0) {
      using GemmKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
          custom::wrapper::kblock::amx_int8::AddGemmSKernelDynamicS8KBlock,
          jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, bias, ldo, broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
          custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS8KBlock,
          jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute({_m, _n, _k, activation, lda, &quanA, weiptr, output, bias, ldo, broadcast_bias ? 0 : ldo});
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_add_s8fp32pern_f32f32_forward(float* activation, SS8Fp32PerN* weiptr, float* bias,
                                                      float* output, int _m, int _n, int _k, int lda, int ldo,
                                                      bool broadcast_bias, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (weiptr->mCoreType == GcCompInt8::TYPE) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
          custom::wrapper::kblock::amx_int8::AddGemmDynamicS8PerN, jblas::utils::parallel::Parallel2DGemm>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute<true, false>({_m, _n, _k, activation, lda, &quanA, weiptr, output, ldo, quanA.mSPtr,
                                         quanA.mCStep, weiptr->mSPtr, bias, broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
          custom::wrapper::kblock::avx512_vnni::AddGemmDynamicS8PerN, jblas::utils::parallel::Parallel2DGemm>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute<true, false>(
          {_m,
           _n,
           _k,
           activation,
           lda,
           &quanA,
           weiptr,
           {{output, ldo, quanA.mCStep, quanA.mSPtr, weiptr->mSPtr, quanA.mZPtr, weiptr->mRPtr},
            bias,
            broadcast_bias ? 0 : ldo}});
    }
  }
  return ret;
}

void jblas_fusion_add_f32f32_forward(float* activation, void* weiptr, float* bias, float* output, int _m, int _n,
                                     int _k, int lda, int ldo, bool broadcast_bias, void* workspace) {
  auto ret = JblasRuntimeError;
  auto wtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(weiptr);
  if (wtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_fusion_add_s4fp32_f32f32_forward(activation, (SS4Fp32*)wtmp, bias, output, _m, _n, _k, lda, ldo,
                                                 broadcast_bias, workspace);
  } else if (wtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_fusion_add_s8fp32_f32f32_forward(activation, (SS8Fp32*)wtmp, bias, output, _m, _n, _k, lda, ldo,
                                                 broadcast_bias, workspace);
  } else if (wtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
    ret = jblas_fusion_add_s8fp32pern_f32f32_forward(activation, (SS8Fp32PerN*)wtmp, bias, output, _m, _n, _k, lda, ldo,
                                                     broadcast_bias, workspace);
  }
  assert(ret == JblasSuccess);
  safe_delete(wtmp);
}
