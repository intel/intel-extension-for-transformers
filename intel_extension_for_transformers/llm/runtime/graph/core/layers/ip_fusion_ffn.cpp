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

unsigned long long jblas_fusion_FFN_f32f32_get_workspace_size(int seq, int fin, int fmid, int fout, void* w1ptr,
                                                              void* w2ptr) {
  // lazy size: maximum padding
  int constexpr padding = 128;
  size_t s = size_t(seq) * utils::padto((size_t)fin, padding) * 4;
  s += size_t(seq) * utils::padto((size_t)fmid, padding) * 4;
  return s;
}

bool jblas_fusion_FFN_SiLu_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid, int fout) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  auto w3tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w3ptr, 0);
  bool support = false;
  if (w1tmp != nullptr && w2tmp != nullptr && w3tmp != nullptr) {
    prologue::PackedWeight* tmps[3] = {w1tmp, w2tmp, w3tmp};
    auto sameKernel = samePackedWeight(tmps, 3);
    if (sameKernel) {
      if (sameKernel) {
        if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32) ||
            w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
          constexpr jblas::gemm::GemmCoreType cores[] = {
              jblas::gemm::GemmCoreType::AMX_INT8_16x48_KBLOCK, jblas::gemm::GemmCoreType::AVX512_VNNI_3x48_KBLOCK,
              jblas::gemm::GemmCoreType::AVX512F_8x48, jblas::gemm::GemmCoreType::AMX_BF16_16x48,
              jblas::gemm::GemmCoreType::AVX2_2X48};
          constexpr size_t EleNum = sizeof(cores) / sizeof(GcCompInt8KBlockSet[0]);
          support = contains(w1tmp->mCoreType, cores, EleNum);
          support &= hasISA(cores, EleNum);
        } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32PerChannelN) ||
                   w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
          constexpr size_t EleNum = sizeof(GcCompInt8Set) / sizeof(GcCompInt8Set[0]);
          support = contains(w1tmp->mCoreType, GcCompInt8Set, EleNum);
          support &= hasISA(GcCompInt8Set, EleNum);
        }
      }
    }
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  safe_delete(w3tmp);
  return support;
}

JBLAS_CODE jblas_fusion_FFN_SiLu_s4fp32_f32f32_forward(float* activation, SS4Fp32* w1ptr, SS4Fp32* w2ptr,
                                                       SS4Fp32* w3ptr, float* tmp1, float* tmp2, float* output, int seq,
                                                       int fin, int fmid, int fout, void* workspace) {
  GetCPUDevice();
  auto ret = JblasNotSupport;
  if (w1ptr->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && w1ptr->mBlockSize % 128 == 0) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::GemmSKernelDynamicS4KBlock;
      using SiluGemmKernel = custom::wrapper::kblock::amx_int8::SiluGemmSKernelDynamicS4KBlock;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
      using DQuantParam = GemmKernel::PrologueA::QParam;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;

      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin, w1ptr->mBlockSize);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize, (int8_t*)workspace + offset);
      ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, quanA1, tmp1, ldtmp1, quanA2, w1ptr,
                            w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,   tmp2, ldtmp2, NULL});
      delete quanA1;
      delete quanA2;

    } else if (_cd->AVX512_VNNI() && w1ptr->mBlockSize % 8 == 0) {
      if (seq <= 32) {
        using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS4KBlockNext;
        using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmSKernelDynamicS4KBlockNext;
        using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
        using DQuantParam = GemmKernel::PrologueA::QParam;
        static FusedInter finter;
        int lda = fin;
        int ldtmp1 = fmid;
        int ldtmp2 = fmid;
        int ldo = fout;

        auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize, (int8_t*)workspace);
        auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin, w1ptr->mBlockSize);
        auto quanA2 =
            finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize, (int8_t*)workspace + offset);
        ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, quanA1, tmp1, ldtmp1, quanA2, w1ptr,
                              w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,   tmp2, ldtmp2, NULL});
        delete quanA1;
        delete quanA2;
      } else {
        using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS4KBlock;
        using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmSKernelDynamicS4KBlock;
        using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
        using DQuantParam = GemmKernel::PrologueA::QParam;
        static FusedInter finter;
        int lda = fin;
        int ldtmp1 = fmid;
        int ldtmp2 = fmid;
        int ldo = fout;

        auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize, (int8_t*)workspace);
        auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin, w1ptr->mBlockSize);
        auto quanA2 =
            finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize, (int8_t*)workspace + offset);
        ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, quanA1, tmp1, ldtmp1, quanA2, w1ptr,
                              w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,   tmp2, ldtmp2, NULL});
        delete quanA1;
        delete quanA2;
      }
    }
  } else if (w1ptr->mCoreType == GcCompFp32::TYPE) {
    if (_cd->AVX512F()) {
      using GemmKernel = custom::wrapper::kblock::avx512f::GemmS4KBlock;
      using SiluGemmKernel = custom::wrapper::kblock::avx512f::SiluGemmS4KBlock;
      using FusedInter = custom::wrapper::transformer::FPFFNFusedInterface<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;
      GemmKernel::AParam paramA = {activation, lda};
      SiluGemmKernel::BParam paramW1 = {w1ptr};
      GemmKernel::BParam paramW2 = {w2ptr};
      GemmKernel::BParam paramW3 = {w3ptr};
      SiluGemmKernel::EpiParam param1 = {tmp1, ldtmp1};
      GemmKernel::EpiParam param2 = {output, ldo, NULL};
      GemmKernel::EpiParam param3 = {tmp2, ldtmp2, NULL};
      ret = finter.compute({seq, fin, fmid, fout, paramA, paramW1, paramW2, paramW3, param1, param2, param3});
    } else if (_cd->AVX2()) {
      using GemmKernel = custom::wrapper::kblock::avx2::GemmS4KBlock;
      using SiluGemmKernel = custom::wrapper::kblock::avx2::SiluGemmS4KBlock;
      using FusedInter = custom::wrapper::transformer::FPFFNFusedInterface<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;
      GemmKernel::AParam paramA = {activation, lda};
      SiluGemmKernel::BParam paramW1 = {w1ptr};
      GemmKernel::BParam paramW2 = {w2ptr};
      GemmKernel::BParam paramW3 = {w3ptr};
      SiluGemmKernel::EpiParam param1 = {tmp1, ldtmp1};
      GemmKernel::EpiParam param2 = {output, ldo, NULL};
      GemmKernel::EpiParam param3 = {tmp2, ldtmp2, NULL};
      ret = finter.compute({seq, fin, fmid, fout, paramA, paramW1, paramW2, paramW3, param1, param2, param3});
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_FFN_SiLu_s8fp32_f32f32_forward(float* activation, SS8Fp32* w1ptr, SS8Fp32* w2ptr,
                                                       SS8Fp32* w3ptr, float* tmp1, float* tmp2, float* output, int seq,
                                                       int fin, int fmid, int fout, void* workspace) {
  GetCPUDevice();
  auto ret = JblasNotSupport;
  if (w1ptr->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && w1ptr->mBlockSize % 128 == 0) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::GemmSKernelDynamicS8KBlock;
      using SiluGemmKernel = custom::wrapper::kblock::amx_int8::SiluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
      using DQuantParam = GemmKernel::PrologueA::QParam;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;

      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin, w1ptr->mBlockSize);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize, (int8_t*)workspace + offset);
      ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, quanA1, tmp1, ldtmp1, quanA2, w1ptr,
                            w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,   tmp2, ldtmp2, NULL});
      delete quanA1;
      delete quanA2;

    } else if (_cd->AVX512_VNNI() && w1ptr->mBlockSize % 4 == 0) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS8KBlock;
      using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
      using DQuantParam = GemmKernel::PrologueA::QParam;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;

      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin, w1ptr->mBlockSize);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize, (int8_t*)workspace + offset);
      ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, quanA1, tmp1, ldtmp1, quanA2, w1ptr,
                            w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,   tmp2, ldtmp2, NULL});
      delete quanA1;
      delete quanA2;
    }
  } else if (w1ptr->mCoreType == GcCompFp32::TYPE) {
    if (_cd->AVX512F()) {
      using GemmKernel = custom::wrapper::kblock::avx512f::GemmS8KBlock;
      using SiluGemmKernel = custom::wrapper::kblock::avx512f::SiluGemmS8KBlock;
      using FusedInter = custom::wrapper::transformer::FPFFNFusedInterface<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;
      GemmKernel::AParam paramA = {activation, lda};
      SiluGemmKernel::BParam paramW1 = {w1ptr};
      GemmKernel::BParam paramW2 = {w2ptr};
      GemmKernel::BParam paramW3 = {w3ptr};
      SiluGemmKernel::EpiParam param1 = {tmp1, ldtmp1};
      GemmKernel::EpiParam param2 = {output, ldo, NULL};
      GemmKernel::EpiParam param3 = {tmp2, ldtmp2, NULL};
      ret = finter.compute({seq, fin, fmid, fout, paramA, paramW1, paramW2, paramW3, param1, param2, param3});
    } else if (_cd->AVX2()) {
      using GemmKernel = custom::wrapper::kblock::avx2::GemmS8KBlock;
      using SiluGemmKernel = custom::wrapper::kblock::avx2::SiluGemmS8KBlock;
      using FusedInter = custom::wrapper::transformer::FPFFNFusedInterface<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;
      GemmKernel::AParam paramA = {activation, lda};
      SiluGemmKernel::BParam paramW1 = {w1ptr};
      GemmKernel::BParam paramW2 = {w2ptr};
      GemmKernel::BParam paramW3 = {w3ptr};
      SiluGemmKernel::EpiParam param1 = {tmp1, ldtmp1};
      GemmKernel::EpiParam param2 = {output, ldo, NULL};
      GemmKernel::EpiParam param3 = {tmp2, ldtmp2, NULL};
      ret = finter.compute({seq, fin, fmid, fout, paramA, paramW1, paramW2, paramW3, param1, param2, param3});
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_FFN_SiLu_s8fp32pern_f32f32_forward(float* activation, SS8Fp32PerN* w1ptr, SS8Fp32PerN* w2ptr,
                                                           SS8Fp32PerN* w3ptr, float* tmp1, float* tmp2, float* output,
                                                           int seq, int fin, int fmid, int fout, void* workspace) {
  GetCPUDevice();
  auto ret = JblasNotSupport;
  if (w1ptr->mCoreType == GcCompInt8::TYPE) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::GemmDynamicS8PerN;
      using SiluGemmKernel = custom::wrapper::kblock::amx_int8::SiluGemmDynamicS8PerN;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterfacePerN<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, (int8_t*)workspace + offset);
      ret = finter.compute({seq,    fin,           fmid,          fout,          activation,   lda,
                            quanA1, tmp1,          ldtmp1,        quanA2,        w1ptr,        w2ptr,
                            w3ptr,  tmp1,          ldtmp1,        quanA1->mSPtr, quanA1->lds,  w1ptr->mSPtr,
                            output, ldo,           quanA2->mSPtr, quanA2->lds,   w2ptr->mSPtr, tmp2,
                            ldtmp2, quanA1->mSPtr, quanA1->lds,   w3ptr->mSPtr});
      delete quanA1;
      delete quanA2;

    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmDynamicS8PerN;
      using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmDynamicS8PerN;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterfacePerN<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;

      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, (int8_t*)workspace + offset);
      ret = finter.compute({seq,    fin,           fmid,          fout,        activation,   lda,          quanA1,
                            tmp1,   ldtmp1,        quanA2,        w1ptr,       w2ptr,        w3ptr,        tmp1,
                            ldtmp1, quanA1->mZPtr, quanA1->mSPtr, quanA1->lds, w1ptr->mRPtr, w1ptr->mSPtr, output,
                            ldo,    quanA2->mZPtr, quanA2->mSPtr, quanA2->lds, w2ptr->mRPtr, w2ptr->mSPtr, tmp2,
                            ldtmp2, quanA1->mZPtr, quanA1->mSPtr, quanA1->lds, w3ptr->mRPtr, w3ptr->mSPtr});
      delete quanA1;
      delete quanA2;
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_FFN_SiLu_s4clipfp32pern_f32f32_forward(float* activation, SS4Fp32PerN* w1ptr,
                                                               SS4Fp32PerN* w2ptr, SS4Fp32PerN* w3ptr, float* tmp1,
                                                               float* tmp2, float* output, int seq, int fin, int fmid,
                                                               int fout, void* workspace) {
  GetCPUDevice();
  auto ret = JblasNotSupport;
  if (w1ptr->mCoreType == GcCompInt8::TYPE) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::GemmDynamicS4ClipPerN;
      using SiluGemmKernel = custom::wrapper::kblock::amx_int8::SiluGemmDynamicS4ClipPerN;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterfacePerN<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, (int8_t*)workspace + offset);
      ret = finter.compute({seq,    fin,           fmid,          fout,          activation,   lda,
                            quanA1, tmp1,          ldtmp1,        quanA2,        w1ptr,        w2ptr,
                            w3ptr,  tmp1,          ldtmp1,        quanA1->mSPtr, quanA1->lds,  w1ptr->mSPtr,
                            output, ldo,           quanA2->mSPtr, quanA2->lds,   w2ptr->mSPtr, tmp2,
                            ldtmp2, quanA1->mSPtr, quanA1->lds,   w3ptr->mSPtr});
      delete quanA1;
      delete quanA2;

    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmDynamicS4ClipPerN;
      using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmDynamicS4ClipPerN;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterfacePerN<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;

      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, (int8_t*)workspace + offset);
      ret = finter.compute({seq,    fin,           fmid,          fout,        activation,   lda,          quanA1,
                            tmp1,   ldtmp1,        quanA2,        w1ptr,       w2ptr,        w3ptr,        tmp1,
                            ldtmp1, quanA1->mZPtr, quanA1->mSPtr, quanA1->lds, w1ptr->mRPtr, w1ptr->mSPtr, output,
                            ldo,    quanA2->mZPtr, quanA2->mSPtr, quanA2->lds, w2ptr->mRPtr, w2ptr->mSPtr, tmp2,
                            ldtmp2, quanA1->mZPtr, quanA1->mSPtr, quanA1->lds, w3ptr->mRPtr, w3ptr->mSPtr});
      delete quanA1;
      delete quanA2;
    }
  }
  return ret;
}

void jblas_fusion_FFN_SiLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                          float* tmp2, float* output, int seq, int fin, int fmid, int fout,
                                          void* workspace) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  auto w3tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w3ptr, 0);
  auto ret = JblasRuntimeError;

  // must check support before forward, there is no need to check support twice.
  if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_fusion_FFN_SiLu_s4fp32_f32f32_forward(activation, dynamic_cast<SS4Fp32*>(w1tmp),
                                                      dynamic_cast<SS4Fp32*>(w2tmp), dynamic_cast<SS4Fp32*>(w3tmp),
                                                      tmp1, tmp2, output, seq, fin, fmid, fout, workspace);
  } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_fusion_FFN_SiLu_s8fp32_f32f32_forward(activation, dynamic_cast<SS8Fp32*>(w1tmp),
                                                      dynamic_cast<SS8Fp32*>(w2tmp), dynamic_cast<SS8Fp32*>(w3tmp),
                                                      tmp1, tmp2, output, seq, fin, fmid, fout, workspace);
  } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
    ret = jblas_fusion_FFN_SiLu_s8fp32pern_f32f32_forward(
        activation, dynamic_cast<SS8Fp32PerN*>(w1tmp), dynamic_cast<SS8Fp32PerN*>(w2tmp),
        dynamic_cast<SS8Fp32PerN*>(w3tmp), tmp1, tmp2, output, seq, fin, fmid, fout, workspace);
  } else if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
    ret = jblas_fusion_FFN_SiLu_s4clipfp32pern_f32f32_forward(
        activation, dynamic_cast<SS4Fp32PerN*>(w1tmp), dynamic_cast<SS4Fp32PerN*>(w2tmp),
        dynamic_cast<SS4Fp32PerN*>(w3tmp), tmp1, tmp2, output, seq, fin, fmid, fout, workspace);
  }
  assert(ret == JblasSuccess);
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  safe_delete(w3tmp);
}

bool jblas_fusion_FFN_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  return false;
}

void jblas_fusion_FFN_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp1, float* output,
                                          int seq, int fin, int fmid, int fout, void* workspace) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  if (w1tmp->mCoreType == GcCompInt8KBlock::TYPE) {
    using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS4KBlock;
    using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::GeluGemmSKernelDynamicS4KBlock;
    using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
    static FusedInter finter;
    int lda = fin;
    int ldtmp1 = fmid;
    int ldo = fout;
    /*auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, NULL);
    auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w1tmp->mBlockSize, NULL);
    finter.compute({seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, tmp1, ldtmp1, output, ldo});*/
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
}

bool jblas_fusion_FFN_Add_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  bool support = false;
  if (w1tmp != nullptr && w2tmp != nullptr) {
    prologue::PackedWeight* tmps[2] = {w1tmp, w2tmp};
    auto sameKernel = samePackedWeight(tmps, 2);
    if (sameKernel) {
      if (sameKernel) {
        if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32) ||
            w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
          constexpr jblas::gemm::GemmCoreType cores[] = {
              jblas::gemm::GemmCoreType::AMX_INT8_16x48_KBLOCK, jblas::gemm::GemmCoreType::AVX512_VNNI_3x48_KBLOCK,
              jblas::gemm::GemmCoreType::AVX512F_8x48, jblas::gemm::GemmCoreType::AMX_BF16_16x48,
              jblas::gemm::GemmCoreType::AVX2_2X48};
          constexpr size_t EleNum = sizeof(cores) / sizeof(cores[0]);
          support = contains(w1tmp->mCoreType, cores, EleNum);
          support &= hasISA(cores, EleNum);
        } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32PerChannelN) ||
                   w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
          constexpr size_t EleNum = sizeof(GcCompInt8Set) / sizeof(GcCompInt8Set[0]);
          support = contains(w1tmp->mCoreType, GcCompInt8Set, EleNum);
          support &= hasISA(GcCompInt8Set, EleNum);
        }
      }
    }
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  return support;
}

JBLAS_CODE jblas_fusion_FFN_Add_GeLu_s4fp32_f32f32_forward(float* activation, SS4Fp32* w1tmp, SS4Fp32* w2tmp,
                                                           float* b1ptr, float* b2ptr, float* tmp1, float* output,
                                                           int seq, int fin, int fmid, int fout, bool broadcast_bias,
                                                           void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && w1tmp->mBlockSize % 128 == 0) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::AddGemmSKernelDynamicS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::amx_int8::AddGeluGemmSKernelDynamicS4KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin, w1tmp->mBlockSize);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize, (int8_t*)workspace + offset);
      ret = finter.compute({seq,        fin,    fmid,   fout,
                            activation, lda,    quanA1, tmp1,
                            ldtmp1,     quanA2, w1tmp,  w2tmp,
                            tmp1,       b1ptr,  ldtmp1, broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,  ldo,    broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmSKernelDynamicS4KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin, w1tmp->mBlockSize);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize, (int8_t*)workspace + offset);
      ret = finter.compute({seq,        fin,    fmid,   fout,
                            activation, lda,    quanA1, tmp1,
                            ldtmp1,     quanA2, w1tmp,  w2tmp,
                            tmp1,       b1ptr,  ldtmp1, broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,  ldo,    broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    }
  } else if (w1tmp->mCoreType == GcCompFp32::TYPE) {
    if (_cd->AVX512F()) {
      using GemmKernel = custom::wrapper::kblock::avx512f::AddGemmS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx512f::AddGeluGemmS4KBlock;
      using FusedInter = custom::wrapper::transformer::FpGeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      ret = finter.compute({seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, tmp1, b1ptr, ldtmp1,
                            broadcast_bias ? 0 : ldtmp1, output, b2ptr, ldo, broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX2()) {
      using GemmKernel = custom::wrapper::kblock::avx2::AddGemmS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx2::AddGeluGemmS4KBlock;
      using FusedInter = custom::wrapper::transformer::FpGeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      ret = finter.compute({seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, tmp1, b1ptr, ldtmp1,
                            broadcast_bias ? 0 : ldtmp1, output, b2ptr, ldo, broadcast_bias ? 0 : ldo});
    }
  } else if (w1tmp->mCoreType == GcCompBf16::TYPE) {
    if (_cd->AMX_BF16()) {
      using GemmKernel = custom::wrapper::kblock::amx_bf16::AddGemmS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::amx_bf16::AddGeluGemmS4KBlock;
      using FusedInter = custom::wrapper::transformer::FpGeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      ret = finter.compute({seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, tmp1, b1ptr, ldtmp1,
                            broadcast_bias ? 0 : ldtmp1, output, b2ptr, ldo, broadcast_bias ? 0 : ldo});
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_FFN_Add_GeLu_s8fp32_f32f32_forward(float* activation, SS8Fp32* w1tmp, SS8Fp32* w2tmp,
                                                           float* b1ptr, float* b2ptr, float* tmp1, float* output,
                                                           int seq, int fin, int fmid, int fout, bool broadcast_bias,
                                                           void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && w1tmp->mBlockSize % 128 == 0) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::AddGemmSKernelDynamicS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::amx_int8::AddGeluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin, w1tmp->mBlockSize);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize, (int8_t*)workspace + offset);
      ret = finter.compute({seq,        fin,    fmid,   fout,
                            activation, lda,    quanA1, tmp1,
                            ldtmp1,     quanA2, w1tmp,  w2tmp,
                            tmp1,       b1ptr,  ldtmp1, broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,  ldo,    broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin, w1tmp->mBlockSize);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize, (int8_t*)workspace + offset);
      ret = finter.compute({seq,        fin,    fmid,   fout,
                            activation, lda,    quanA1, tmp1,
                            ldtmp1,     quanA2, w1tmp,  w2tmp,
                            tmp1,       b1ptr,  ldtmp1, broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,  ldo,    broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    }
  } else if (w1tmp->mCoreType == GcCompFp32::TYPE) {
    if (_cd->AVX512F()) {
      using GemmKernel = custom::wrapper::kblock::avx512f::AddGemmS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx512f::AddGeluGemmS8KBlock;
      using FusedInter = custom::wrapper::transformer::FpGeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      ret = finter.compute({seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, tmp1, b1ptr, ldtmp1,
                            broadcast_bias ? 0 : ldtmp1, output, b2ptr, ldo, broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX2()) {
      using GemmKernel = custom::wrapper::kblock::avx2::AddGemmS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx2::AddGeluGemmS8KBlock;
      using FusedInter = custom::wrapper::transformer::FpGeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      ret = finter.compute({seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, tmp1, b1ptr, ldtmp1,
                            broadcast_bias ? 0 : ldtmp1, output, b2ptr, ldo, broadcast_bias ? 0 : ldo});
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_FFN_Add_GeLu_s8fp32pern_f32f32_forward(float* activation, SS8Fp32PerN* w1tmp,
                                                               SS8Fp32PerN* w2tmp, float* b1ptr, float* b2ptr,
                                                               float* tmp1, float* output, int seq, int fin, int fmid,
                                                               int fout, bool broadcast_bias, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == GcCompInt8::TYPE) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::AddGemmDynamicS8PerN;
      using GeluGemmKernel = custom::wrapper::kblock::amx_int8::AddGeluGemmDynamicS8PerN;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterfacePerN<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, (int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            quanA1,
                            tmp1,
                            ldtmp1,
                            quanA2,
                            w1tmp,
                            w2tmp,
                            tmp1,
                            ldtmp1,
                            quanA1->mSPtr,
                            quanA1->lds,
                            w1tmp->mSPtr,
                            b1ptr,
                            broadcast_bias ? 0 : ldtmp1,
                            output,
                            ldo,
                            quanA2->mSPtr,
                            quanA2->lds,
                            w2tmp->mSPtr,
                            b2ptr,
                            broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmDynamicS8PerN;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmDynamicS8PerN;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterfacePerN<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, (int8_t*)workspace + offset);
      ret = finter.compute({seq,           fin,         fmid,
                            fout,          activation,  lda,
                            quanA1,        tmp1,        ldtmp1,
                            quanA2,        w1tmp,       w2tmp,
                            tmp1,          ldtmp1,      quanA1->mZPtr,
                            quanA1->mSPtr, quanA1->lds, w1tmp->mRPtr,
                            w1tmp->mSPtr,  b1ptr,       broadcast_bias ? 0 : ldtmp1,
                            output,        ldo,         quanA2->mZPtr,
                            quanA2->mSPtr, quanA2->lds, w2tmp->mRPtr,
                            w2tmp->mSPtr,  b2ptr,       broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_FFN_Add_GeLu_s4clipfp32pern_f32f32_forward(float* activation, SS4Fp32PerN* w1tmp,
                                                                   SS4Fp32PerN* w2tmp, float* b1ptr, float* b2ptr,
                                                                   float* tmp1, float* output, int seq, int fin,
                                                                   int fmid, int fout, bool broadcast_bias,
                                                                   void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == GcCompInt8::TYPE) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::AddGemmDynamicS4ClipPerN;
      using GeluGemmKernel = custom::wrapper::kblock::amx_int8::AddGeluGemmDynamicS4ClipPerN;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterfacePerN<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, (int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            quanA1,
                            tmp1,
                            ldtmp1,
                            quanA2,
                            w1tmp,
                            w2tmp,
                            tmp1,
                            ldtmp1,
                            quanA1->mSPtr,
                            quanA1->lds,
                            w1tmp->mSPtr,
                            b1ptr,
                            broadcast_bias ? 0 : ldtmp1,
                            output,
                            ldo,
                            quanA2->mSPtr,
                            quanA2->lds,
                            w2tmp->mSPtr,
                            b2ptr,
                            broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmDynamicS4ClipPerN;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmDynamicS4ClipPerN;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterfacePerN<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, (int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : finter.getActivationPtr()->getWorkSpaceSize(seq, fin);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, (int8_t*)workspace + offset);
      ret = finter.compute({seq,           fin,         fmid,
                            fout,          activation,  lda,
                            quanA1,        tmp1,        ldtmp1,
                            quanA2,        w1tmp,       w2tmp,
                            tmp1,          ldtmp1,      quanA1->mZPtr,
                            quanA1->mSPtr, quanA1->lds, w1tmp->mRPtr,
                            w1tmp->mSPtr,  b1ptr,       broadcast_bias ? 0 : ldtmp1,
                            output,        ldo,         quanA2->mZPtr,
                            quanA2->mSPtr, quanA2->lds, w2tmp->mRPtr,
                            w2tmp->mSPtr,  b2ptr,       broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    }
  }
  return ret;
}

void jblas_fusion_FFN_Add_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* b1ptr, float* b2ptr,
                                              float* tmp1, float* output, int seq, int fin, int fmid, int fout,
                                              bool broadcast_bias, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret =
        jblas_fusion_FFN_Add_GeLu_s4fp32_f32f32_forward(activation, (SS4Fp32*)w1tmp, (SS4Fp32*)w2tmp, b1ptr, b2ptr,
                                                        tmp1, output, seq, fin, fmid, fout, broadcast_bias, workspace);
  } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
    ret =
        jblas_fusion_FFN_Add_GeLu_s8fp32_f32f32_forward(activation, (SS8Fp32*)w1tmp, (SS8Fp32*)w2tmp, b1ptr, b2ptr,
                                                        tmp1, output, seq, fin, fmid, fout, broadcast_bias, workspace);
  } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
    ret = jblas_fusion_FFN_Add_GeLu_s8fp32pern_f32f32_forward(activation, (SS8Fp32PerN*)w1tmp, (SS8Fp32PerN*)w2tmp,
                                                              b1ptr, b2ptr, tmp1, output, seq, fin, fmid, fout,
                                                              broadcast_bias, workspace);
  } else if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
    ret = jblas_fusion_FFN_Add_GeLu_s4clipfp32pern_f32f32_forward(activation, (SS4Fp32PerN*)w1tmp, (SS4Fp32PerN*)w2tmp,
                                                                  b1ptr, b2ptr, tmp1, output, seq, fin, fmid, fout,
                                                                  broadcast_bias, workspace);
  }
  assert(ret == JblasSuccess);
  safe_delete(w1tmp);
  safe_delete(w2tmp);
}
