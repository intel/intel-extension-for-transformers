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

unsigned long long jblas_fusion_QKV_f32f32_get_workspace_size(int _m, int _n, int _k, void* w1ptr) {
  // maximum padding
  int constexpr padding = 128;
  size_t s = size_t(_m) * utils::padto((size_t)_k, padding) * 4;
  return s;
}

namespace {
namespace transformer {
namespace avx512_vnni {
static JBLAS_ISA constexpr DefaultISA = JblasAVX512_VNNI;
using QKVGemmDynamicS4Fp32KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmLauncherKBlock<DefaultISA,
                                                    jblas::gemm::kblock::GemmCore_Row_NN_4x48_AVX512_VNNI_KBLOCK,
                                                    jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
                                                    jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
                                                    jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
using QKVGemmDynamicS4Fp32KBlockNext = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
using QKVGemmDynamicS8Fp32KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmLauncherKBlock<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_4x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
using QKVGemmDynamicS8Fp32PerN = jblas::wrapper::transformer::QKVGemmInterfacePackWeightParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, jblas::prologue::gemm::ActivationFp32AsymU8Quantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32PerChannelN,
        jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;
using QKVGemmDynamicS4ClipFp32PerN = jblas::wrapper::transformer::QKVGemmInterfacePackWeightParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, jblas::prologue::gemm::ActivationFp32AsymU8Quantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32PerN,
        jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;
}  // namespace avx512_vnni
namespace avx_vnni {
static JBLAS_ISA constexpr DefaultISA = JblasAVX_VNNI;
using QKVGemmDynamicS4Fp32KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmLauncherKBlock<DefaultISA,
                                                    jblas::gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK,
                                                    jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
                                                    jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
                                                    jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
using QKVGemmDynamicS8Fp32KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmLauncherKBlock<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
// using QKVGemmDynamicS8Fp32PerN = jblas::wrapper::transformer::QKVGemmInterfacePackWeightParallelAB<
//     jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
//         DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,
//         jblas::prologue::gemm::ActivationFp32AsymU8Quantize,
//         jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32PerChannelN,
//         jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
//     jblas::utils::parallel::Parallel2DGemm>;
// using QKVGemmDynamicS4ClipFp32PerN = jblas::wrapper::transformer::QKVGemmInterfacePackWeightParallelAB<
//     jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
//         DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,
//         jblas::prologue::gemm::ActivationFp32AsymU8Quantize,
//         jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32PerN,
//         jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
//     jblas::utils::parallel::Parallel2DGemm>;
}  // namespace avx_vnni
namespace amx_int8 {
static JBLAS_ISA constexpr DefaultISA = JblasAMX_INT8;
using QKVGemmDynamicS4Fp32KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
        jblas::prologue::gemm::ActivationF32S8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
using QKVGemmDynamicS8Fp32KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
        jblas::prologue::gemm::ActivationF32S8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
using QKVGemmDynamicS8Fp32PerN = jblas::wrapper::transformer::QKVGemmInterfacePackWeightParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8, jblas::prologue::gemm::ActivationFp32SymS8Quantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32PerChannelN,
        jblas::epilogue::gemm::DequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;
using QKVGemmDynamicS4ClipFp32PerN = jblas::wrapper::transformer::QKVGemmInterfacePackWeightParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8, jblas::prologue::gemm::ActivationFp32SymS8Quantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32PerN,
        jblas::epilogue::gemm::DequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;
}  // namespace amx_int8
}  // namespace transformer
}  // namespace

bool jblas_fusion_QKV_f32f32_support(void* wqptr, void* wkptr, void* wvptr, int _m, int _n, int _k) {
  auto wqtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wqptr);
  auto wktmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wkptr);
  auto wvtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wvptr);
  bool support = false;
  if (wqtmp != nullptr && wktmp != nullptr && wvtmp != nullptr) {
    prologue::gemm::WeightBase* tmps[3] = {wqtmp, wktmp, wvtmp};
    auto sameKernel = samePackedWeight(tmps, 3);
    if (sameKernel) {
      if (wqtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32) ||
          wqtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32)) {
        constexpr size_t EleNum = sizeof(GcCompInt8KBlockSet) / sizeof(GcCompInt8KBlockSet[0]);
        support = contains(wqtmp->mCoreType, GcCompInt8KBlockSet, EleNum);
        support &= hasISA(GcCompInt8KBlockSet, EleNum);
      } else if (wqtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32PerChannelN) ||
                 wqtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
        constexpr size_t EleNum = sizeof(GcCompInt8Set) / sizeof(GcCompInt8Set[0]);
        support = contains(wqtmp->mCoreType, GcCompInt8Set, EleNum);
        support &= hasISA(GcCompInt8Set, EleNum);
      }
    }
  }
  safe_delete(wqtmp);
  safe_delete(wktmp);
  safe_delete(wvtmp);
  return support;
}

JBLAS_CODE jblas_QKVs4fp32_f32f32_forward(float* activation, SS4Fp32* wqptr, SS4Fp32* wkptr, SS4Fp32* wvptr,
                                          float* output, int _m, int _n, int _k, int lda, int ldo, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (wqptr->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && wqptr->mBlockSize % 128 == 0) {
      using GemmKernel = transformer::amx_int8::QKVGemmDynamicS4Fp32KBlock;
      static GemmKernel kernel;
      GemmKernel::WeightType::Param wparams[3]{
          wqptr,
          wkptr,
          wvptr,
      };
      GemmKernel::CParam oparams[3]{
          {output, ldo},
          {output + _m * _n, ldo},
          {output + 2 * _m * _n, ldo},
      };
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, wqptr->mBlockSize);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute({_m, _n, _k, 3, activation, lda, &quanA, wparams, oparams, NULL});
    } else if (wqptr->mBlockSize % 8 == 0) {
      if (_cd->AVX512_VNNI()) {
        if (_m <= 32) {
          using GemmKernel = transformer::avx512_vnni::QKVGemmDynamicS4Fp32KBlockNext;
          static GemmKernel kernel;
          GemmKernel::WeightType::Param wparams[3]{
              wqptr,
              wkptr,
              wvptr,
          };
          GemmKernel::CParam oparams[3]{
              {output, ldo},
              {output + _m * _n, ldo},
              {output + 2 * _m * _n, ldo},
          };
          auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, wqptr->mBlockSize);
          quanA.assign((int8_t*)workspace);
          ret = kernel.compute({_m, _n, _k, 3, activation, lda, &quanA, wparams, oparams, NULL});
        } else {
          using GemmKernel = transformer::avx512_vnni::QKVGemmDynamicS4Fp32KBlock;
          static GemmKernel kernel;
          GemmKernel::WeightType::Param wparams[3]{
              wqptr,
              wkptr,
              wvptr,
          };
          GemmKernel::CParam oparams[3]{
              {output, ldo},
              {output + _m * _n, ldo},
              {output + 2 * _m * _n, ldo},
          };
          auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, wqptr->mBlockSize);
          quanA.assign((int8_t*)workspace);
          ret = kernel.compute({_m, _n, _k, 3, activation, lda, &quanA, wparams, oparams, NULL});
        }
      } else if (_cd->AVX_VNNI()) {
        using GemmKernel = transformer::avx_vnni::QKVGemmDynamicS4Fp32KBlock;
        static GemmKernel kernel;
        GemmKernel::WeightType::Param wparams[3]{
            wqptr,
            wkptr,
            wvptr,
        };
        GemmKernel::CParam oparams[3]{
            {output, ldo},
            {output + _m * _n, ldo},
            {output + 2 * _m * _n, ldo},
        };
        auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, wqptr->mBlockSize);
        quanA.assign((int8_t*)workspace);
        ret = kernel.compute({_m, _n, _k, 3, activation, lda, &quanA, wparams, oparams, NULL});
      }
    }
  }
  return ret;
}

JBLAS_CODE jblas_QKVs8fp32_f32f32_forward(float* activation, SS8Fp32* wqptr, SS8Fp32* wkptr, SS8Fp32* wvptr,
                                          float* output, int _m, int _n, int _k, int lda, int ldo, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (wqptr->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && wqptr->mBlockSize % 128 == 0) {
      using GemmKernel = transformer::amx_int8::QKVGemmDynamicS8Fp32KBlock;
      static GemmKernel kernel;
      GemmKernel::WeightType::Param wparams[3]{
          wqptr,
          wkptr,
          wvptr,
      };
      GemmKernel::CParam oparams[3]{
          {output, ldo},
          {output + _m * _n, ldo},
          {output + 2 * _m * _n, ldo},
      };
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, wqptr->mBlockSize);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute({_m, _n, _k, 3, activation, lda, &quanA, wparams, oparams, NULL});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = transformer::avx512_vnni::QKVGemmDynamicS8Fp32KBlock;
      static GemmKernel kernel;
      GemmKernel::WeightType::Param wparams[3]{
          wqptr,
          wkptr,
          wvptr,
      };
      GemmKernel::CParam oparams[3]{
          {output, ldo},
          {output + _m * _n, ldo},
          {output + 2 * _m * _n, ldo},
      };
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, wqptr->mBlockSize);
      quanA.assign((int8_t*)workspace);
      ret = kernel.compute({_m, _n, _k, 3, activation, lda, &quanA, wparams, oparams, NULL});
    }
  }
  return ret;
}

JBLAS_CODE jblas_QKVs8fp32pern_f32f32_forward(float* activation, SS8Fp32PerN* wqptr, SS8Fp32PerN* wkptr,
                                              SS8Fp32PerN* wvptr, float* output, int _m, int _n, int _k, int lda,
                                              int ldo, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (wqptr->mCoreType == GcCompInt8::TYPE) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = transformer::amx_int8::QKVGemmDynamicS8Fp32PerN;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k);
      quanA.assign((int8_t*)workspace);
      GemmKernel::WeightType::Param wparams[3]{
          wqptr,
          wkptr,
          wvptr,
      };
      GemmKernel::CParam oparams[3]{
          {output, ldo, quanA.mSPtr, quanA.mCStep, wqptr->mSPtr},
          {output + _m * _n, ldo, quanA.mSPtr, quanA.mCStep, wkptr->mSPtr},
          {output + 2 * _m * _n, ldo, quanA.mSPtr, quanA.mCStep, wvptr->mSPtr},
      };
      ret = kernel.compute<true, false>({_m, _n, _k, 3, activation, lda, &quanA, wparams, oparams, NULL});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = transformer::avx512_vnni::QKVGemmDynamicS8Fp32PerN;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k);
      quanA.assign((int8_t*)workspace);
      GemmKernel::WeightType::Param wparams[3]{
          wqptr,
          wkptr,
          wvptr,
      };
      GemmKernel::CParam oparams[3]{
          {output, ldo, quanA.mCStep, quanA.mSPtr, wqptr->mSPtr, quanA.mZPtr, wqptr->mRPtr},
          {output + _m * _n, ldo, quanA.mCStep, quanA.mSPtr, wkptr->mSPtr, quanA.mZPtr, wkptr->mRPtr},
          {output + 2 * _m * _n, ldo, quanA.mCStep, quanA.mSPtr, wvptr->mSPtr, quanA.mZPtr, wvptr->mRPtr},
      };
      ret = kernel.compute<true, false>({_m, _n, _k, 3, activation, lda, &quanA, wparams, oparams, NULL});
    }
  }
  return ret;
}

JBLAS_CODE jblas_QKVs4clipfp32pern_f32f32_forward(float* activation, SS4Fp32PerN* wqptr, SS4Fp32PerN* wkptr,
                                                  SS4Fp32PerN* wvptr, float* output, int _m, int _n, int _k, int lda,
                                                  int ldo, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (wqptr->mCoreType == GcCompInt8::TYPE) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = transformer::amx_int8::QKVGemmDynamicS4ClipFp32PerN;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k);
      quanA.assign((int8_t*)workspace);
      GemmKernel::WeightType::Param wparams[3]{
          wqptr,
          wkptr,
          wvptr,
      };
      GemmKernel::CParam oparams[3]{
          {output, ldo, quanA.mSPtr, quanA.mCStep, wqptr->mSPtr},
          {output + _m * _n, ldo, quanA.mSPtr, quanA.mCStep, wkptr->mSPtr},
          {output + 2 * _m * _n, ldo, quanA.mSPtr, quanA.mCStep, wvptr->mSPtr},
      };
      ret = kernel.compute<true, false>({_m, _n, _k, 3, activation, lda, &quanA, wparams, oparams, NULL});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = transformer::avx512_vnni::QKVGemmDynamicS4ClipFp32PerN;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k);
      quanA.assign((int8_t*)workspace);
      GemmKernel::WeightType::Param wparams[3]{
          wqptr,
          wkptr,
          wvptr,
      };
      GemmKernel::CParam oparams[3]{
          {output, ldo, quanA.mCStep, quanA.mSPtr, wqptr->mSPtr, quanA.mZPtr, wqptr->mRPtr},
          {output + _m * _n, ldo, quanA.mCStep, quanA.mSPtr, wkptr->mSPtr, quanA.mZPtr, wkptr->mRPtr},
          {output + 2 * _m * _n, ldo, quanA.mCStep, quanA.mSPtr, wvptr->mSPtr, quanA.mZPtr, wvptr->mRPtr},
      };
      ret = kernel.compute<true, false>({_m, _n, _k, 3, activation, lda, &quanA, wparams, oparams, NULL});
    }
  }
  return ret;
}

// f32f32: activation & output dtype
void jblas_fusion_QKV_f32f32_forward(float* activation, void* wqptr, void* wkptr, void* wvptr, float* output, int _m,
                                     int _n, int _k, int lda, int ldo, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto wqtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wqptr);
  auto wktmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wkptr);
  auto wvtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wvptr);
  // must check support before forward, there is no need to check support twice.
  if (wqtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_QKVs4fp32_f32f32_forward(activation, dynamic_cast<SS4Fp32*>(wqtmp), dynamic_cast<SS4Fp32*>(wktmp),
                                         dynamic_cast<SS4Fp32*>(wvtmp), output, _m, _n, _k, lda, ldo, workspace);
  } else if (wqtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_QKVs8fp32_f32f32_forward(activation, dynamic_cast<SS8Fp32*>(wqtmp), dynamic_cast<SS8Fp32*>(wktmp),
                                         dynamic_cast<SS8Fp32*>(wvtmp), output, _m, _n, _k, lda, ldo, workspace);
  } else if (wqtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
    ret = jblas_QKVs8fp32pern_f32f32_forward(activation, dynamic_cast<SS8Fp32PerN*>(wqtmp),
                                             dynamic_cast<SS8Fp32PerN*>(wktmp), dynamic_cast<SS8Fp32PerN*>(wvtmp),
                                             output, _m, _n, _k, lda, ldo, workspace);
  } else if (wqtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
    ret = jblas_QKVs4clipfp32pern_f32f32_forward(activation, dynamic_cast<SS4Fp32PerN*>(wqtmp),
                                                 dynamic_cast<SS4Fp32PerN*>(wktmp), dynamic_cast<SS4Fp32PerN*>(wvtmp),
                                                 output, _m, _n, _k, lda, ldo, workspace);
  }
  assert(ret == JblasSuccess);
  safe_delete(wqtmp);
  safe_delete(wktmp);
  safe_delete(wvtmp);
}
