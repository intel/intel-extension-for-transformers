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

/*++

Module Name:

    jblas_gemm.cpp

Abstract:

    C APIs of BesTLA GEMMs.
--*/

#include "jblas_gemm.h"

#include "jblas_defs.h"

using namespace jblas;

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T>
void JblasGemmCompF32(const int M, const int N, const int K, const float* A, const int lda,
                      jblas::storage::gemm::IWeightBase* _B, float* C, const int ldc, int8_t* WorkSpace,
                      jblas::parallel::IThreading* th) {
  if (M <= 32) {
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher = tLauncher_Fp_F32F32<GemmCore_T, Wei_T>;
    static Launcher kernel;
    auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
    auto reduceA = kernel.mProA.createStorage(M, K, B->mBlockSize);
    if (B->IsAsym()) {
      reduceA.assign(WorkSpace);
    }
    typename Launcher::BEpiParam blkargs{
        B->template SPtr<int8_t>(),    B->SDtype(), B->CStep(), B->template ZPtr<int8_t>(),
        reduceA.template RPtr<float>(), reduceA.lda};

    typename Launcher::Param args{M, N, K, B->mBlockSize, {A, K, &reduceA}, {B}, blkargs, {C, N}};
    if (B->IsAsym()) {
      jblas::parallel::GemmKBlockRunWithA<Parallel>(kernel, args, th);
    } else {
      jblas::parallel::GemmKBlockRun<Parallel>(kernel, args, th);
    }
  } else {
    using Parallel = jblas::parallel::gemm::SchedulerBase<GemmCore_T>;
    using Launcher = jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T,
                                                        jblas::prologue_a::gemm::ActivationKBlockBaseF32, Wei_T,
                                                        jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
    static Launcher kernel;
    auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);

    typename Launcher::Param args{M, N, K, {A, K}, {B}, {C, N}};
    jblas::parallel::GemmBaseRun<Parallel>(kernel, args, th);
  }
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T>
void JblasGemmCompInt8(const int M, const int N, const int K, const float* A, const int lda,
                       jblas::storage::gemm::IWeightBase* _B, float* C, const int ldc, int8_t* WorkSpace,
                       jblas::parallel::IThreading* th) {
  using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
  using Launcher = tLauncher_Int8_F32F32<GemmCore_T, Wei_T>;
  auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
  static Launcher kernel;
  auto quanA = kernel.mProA.createStorage(M, K, B->mBlockSize, B->IsAsym());
  quanA.assign(WorkSpace);
  typename Launcher::Param args{M,
                                N,
                                K,
                                B->mBlockSize,
                                {A, K, &quanA},
                                {B},
                                {B->template SPtr<int8_t>(), B->SDtype(), B->CStep(), quanA.template SPtr<float>(),
                                 quanA.CStep(), quanA.template ZPtr<uint8_t>(), B->template RPtr<float>(), B->RDtype(),
                                 B->template ZPtr<int8_t>(), quanA.template RPtr<float>(), B->mBlockSize},
                                {C, N}};
  jblas::parallel::GemmKBlockRunWithA<Parallel>(kernel, args, th);
}

bool JblasGemmBatchDriver(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                          const JBLAS_GEMM_DATA_PACKED_PARAMS* DataParams, int8_t* WorkSpace, void* ThreadPool) {
  GetCPUDevice();
  auto pth = reinterpret_cast<jblas::parallel::IThreading*>(ThreadPool);
  bool processed = true;
  for (size_t i = 0; i < BatchN; i++) {
    auto ptr = jblas::storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(DataParams[i].B));
    if (ptr) {
      auto coretype = ptr->mCoreId;
      auto NTile = jblas::gemm::CoreAttr::get_mask_val(ptr->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK,
                                                       jblas::gemm::CoreAttr::NTILE_SHIFT);
      auto CType = jblas::gemm::CoreAttr::get_mask_val(ptr->mCoreId, jblas::gemm::CoreAttr::COMP_MASK,
                                                       jblas::gemm::CoreAttr::COMP_SHIFT);
      if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
        if (CType == uint32_t(gemm::CompType::COMP_FP32)) {
          if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
            JblasGemmCompF32<tAVX512F, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                 DataParams[i].ldc, WorkSpace, pth);
            goto __END;
          }
          if (NTile == tAVX2::NTILE && _cd->AVX2()) {
            JblasGemmCompF32<tAVX2, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                              DataParams[i].ldc, WorkSpace, pth);
            goto __END;
          }
        }
        if (CType == uint32_t(gemm::CompType::COMP_BF16_FP32)) {
          if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
            JblasGemmCompF32<tAMX_BF16, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                  DataParams[i].ldc, WorkSpace, pth);
            goto __END;
          }
        }
        if (CType == uint32_t(gemm::CompType::COMP_INT8_US_INT32)) {
          if (NTile == tAMX_INT8_US::NTILE && _cd->AMX_INT8()) {
            JblasGemmCompInt8<tAMX_INT8_US, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                      DataParams[i].ldc, WorkSpace, pth);
            goto __END;
          }
          if (NTile == tAVX512_VNNI::NTILE && _cd->AVX512_VNNI()) {
            JblasGemmCompInt8<tAVX512_VNNI, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                      DataParams[i].ldc, WorkSpace, pth);
            goto __END;
          }
          if (NTile == tAVX_VNNI::NTILE && _cd->AVX_VNNI()) {
            JblasGemmCompInt8<tAVX_VNNI, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                   DataParams[i].ldc, WorkSpace, pth);
            goto __END;
          }
        }
        if (CType == uint32_t(gemm::CompType::COMP_INT8_SS_INT32)) {
          if (NTile == tAMX_INT8_SS::NTILE && _cd->AMX_INT8()) {
            JblasGemmCompInt8<tAMX_INT8_SS, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                      DataParams[i].ldc, WorkSpace, pth);
            goto __END;
          }
        }
      }
      if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
        if (CType == uint32_t(gemm::CompType::COMP_FP32)) {
          if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
            JblasGemmCompF32<tAVX512F, tWeiF4>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                               DataParams[i].ldc, WorkSpace, pth);
            goto __END;
          }
          if (NTile == tAVX2::NTILE && _cd->AVX2()) {
            JblasGemmCompF32<tAVX2, tWeiF4>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                            DataParams[i].ldc, WorkSpace, pth);
            goto __END;
          }
        }
        if (CType == uint32_t(gemm::CompType::COMP_BF16_FP32)) {
          if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
            JblasGemmCompF32<tAMX_BF16, tWeiF4>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                DataParams[i].ldc, WorkSpace, pth);
            goto __END;
          }
        }
      }
    __END:
      delete ptr;
    } else {
      processed = false;
      break;
    }
  }
  return processed;
}

template <typename T>
static size_t JblasBuSize(int block_size, size_t N, size_t K, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype,
                          bool isAsym) {
  static T launcher;
  using WType = typename T::PrologueB::StorageWeight;
  WType stor(0);
  if constexpr (std::is_same_v<typename T::PrologueB,
                               jblas::prologue_b::gemm::WeightKBlockNInteger<typename T::GemmCore, T::ISA>>) {
    stor = launcher.mProB.createStorage(N, K, block_size, QuantType, ScaleDtype, JBLAS_DTYPE::BF16, isAsym);
  } else {
    stor = launcher.mProB.createStorage(N, K, block_size, QuantType, ScaleDtype);
  }

  // Reduce dtype set to bf16
  return stor.mSize;
}
template <template <class, JBLAS_ISA> class Wei_T>
static size_t JblasGemmPackBSizeLocal(size_t N, size_t K, size_t BlkSize, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype,
                                      bool isAsym, ne_comp_type CompType) {
  GetCPUDevice();
  // from low precision to high precision
  switch (CompType) {
    case NE_COMP_INT8:
      if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS::KTILE == 0) {
        return JblasBuSize<tLauncher_Int8_F32F32<tAMX_INT8_SS, Wei_T>>(int(BlkSize), N, K, QuantType, ScaleDtype,
                                                                       isAsym);
      }
      if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI::KTILE == 0) {
        return JblasBuSize<tLauncher_Int8_F32F32<tAVX512_VNNI, Wei_T>>(int(BlkSize), N, K, QuantType, ScaleDtype,
                                                                       isAsym);
      }
      if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI::KTILE == 0) {
        return JblasBuSize<tLauncher_Int8_F32F32<tAVX_VNNI, Wei_T>>(int(BlkSize), N, K, QuantType, ScaleDtype, isAsym);
      }
    case NE_COMP_F16:
    case NE_COMP_BF16:
      if (_cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
        return JblasBuSize<tLauncher_Int8_F32F32<tAMX_BF16, Wei_T>>(int(BlkSize), N, K, QuantType, ScaleDtype, isAsym);
      }
    case NE_COMP_F32:
    case NE_COMP_UNDEF:  // currently only f32 activation
      if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
        return JblasBuSize<tLauncher_Fp_F32F32<tAVX512F, Wei_T>>(int(BlkSize), N, K, QuantType, ScaleDtype, isAsym);
      }
      if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
        return JblasBuSize<tLauncher_Fp_F32F32<tAVX2, Wei_T>>(int(BlkSize), N, K, QuantType, ScaleDtype, isAsym);
      }
      break;
    default:
      return 0;
  }
  return 0;
}

size_t JblasGemmPackBSize(size_t N, size_t K, size_t BlkSize, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype,
                          bool isAsym, ne_comp_type CompType) {
  switch (QuantType) {
    case JBLAS_DTYPE::S4_CLIP:
    case JBLAS_DTYPE::S4_FULLRANGE:
    case JBLAS_DTYPE::S8:
      return JblasGemmPackBSizeLocal<jblas::prologue_b::gemm::WeightKBlockNInteger>(N, K, BlkSize, QuantType,
                                                                                    ScaleDtype, isAsym, CompType);
    case JBLAS_DTYPE::F4_BNB:
    case JBLAS_DTYPE::F4_E2M1:
    case JBLAS_DTYPE::F4_NF4:
      return JblasGemmPackBSizeLocal<jblas::prologue_b::gemm::WeightKBlockF4>(N, K, BlkSize, QuantType, ScaleDtype,
                                                                              isAsym, CompType);
    default:
      return 0;
  }
  return 0;
}

template <typename T>
void JblaGemmQuantPackBTrans(void* PackedBuf, int BlkSize, const float* FpData, int N, int K, JBLAS_DTYPE QuantType,
                             JBLAS_DTYPE ScaleDtype, bool IsAsym, int ldb, void* ThreadPool) {
  static T launcher;
  using WType = typename T::PrologueB::StorageWeight;
  WType stor(0);
  if constexpr (std::is_same_v<typename T::PrologueB,
                               jblas::prologue_b::gemm::WeightKBlockNInteger<typename T::GemmCore, T::ISA>>) {
    stor = launcher.mProB.createStorage(N, K, BlkSize, QuantType, ScaleDtype, JBLAS_DTYPE::BF16, IsAsym);
  } else {
    stor = launcher.mProB.createStorage(N, K, BlkSize, QuantType, ScaleDtype);
  }
  stor.assign((int8_t*)PackedBuf);
  auto pth = reinterpret_cast<jblas::parallel::IThreading*>(ThreadPool);
  launcher.mProB.packTransposeWeight(N, K, FpData, ldb, &stor, pth);
}

template <template <class, JBLAS_ISA> class Wei_T>
static bool JblasGemmQuantPackBTransLocal(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb,
                                          size_t BlkSize, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype, bool isAsym,
                                          ne_comp_type CompType, void* ThreadPool) {
  GetCPUDevice();
  switch (CompType) {
    case NE_COMP_INT8:
      if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS::KTILE == 0) {
        JblaGemmQuantPackBTrans<tLauncher_Int8_F32F32<tAMX_INT8_SS, Wei_T>>(
            PackedBuf, int(BlkSize), FpData, int(N), int(K), QuantType, ScaleDtype, isAsym, int(ldb), ThreadPool);
        return true;
      }
      if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI::KTILE == 0) {
        JblaGemmQuantPackBTrans<tLauncher_Int8_F32F32<tAVX512_VNNI, Wei_T>>(
            PackedBuf, int(BlkSize), FpData, int(N), int(K), QuantType, ScaleDtype, isAsym, int(ldb), ThreadPool);
        return true;
      }
      if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI::KTILE == 0) {
        JblaGemmQuantPackBTrans<tLauncher_Int8_F32F32<tAVX_VNNI, Wei_T>>(
            PackedBuf, int(BlkSize), FpData, int(N), int(K), QuantType, ScaleDtype, isAsym, int(ldb), ThreadPool);
        return true;
      }
    case NE_COMP_F16:
    case NE_COMP_BF16:
      if (_cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
        JblaGemmQuantPackBTrans<tLauncher_Fp_F32F32<tAMX_BF16, Wei_T>>(
            PackedBuf, int(BlkSize), FpData, int(N), int(K), QuantType, ScaleDtype, isAsym, int(ldb), ThreadPool);
        return true;
      }
    case NE_COMP_F32:
    case NE_COMP_UNDEF:  // currently only f32 activation
      if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
        JblaGemmQuantPackBTrans<tLauncher_Fp_F32F32<tAVX512F, Wei_T>>(
            PackedBuf, int(BlkSize), FpData, int(N), int(K), QuantType, ScaleDtype, isAsym, int(ldb), ThreadPool);
        return true;
      }
      if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
        JblaGemmQuantPackBTrans<tLauncher_Fp_F32F32<tAVX2, Wei_T>>(PackedBuf, int(BlkSize), FpData, int(N), int(K),
                                                                   QuantType, ScaleDtype, isAsym, int(ldb), ThreadPool);
        return true;
      }
    default:
      return false;
  }
  return false;
}

bool JblasGemmQuantPackBTrans(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb, size_t BlkSize,
                              JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype, bool isAsym, ne_comp_type CompType,
                              void* ThreadPool) {
  switch (QuantType) {
    case JBLAS_DTYPE::S4_CLIP:
    case JBLAS_DTYPE::S4_FULLRANGE:
    case JBLAS_DTYPE::S8:
      return JblasGemmQuantPackBTransLocal<jblas::prologue_b::gemm::WeightKBlockNInteger>(
          PackedBuf, FpData, N, K, ldb, BlkSize, QuantType, ScaleDtype, isAsym, CompType, ThreadPool);
    case JBLAS_DTYPE::F4_BNB:
    case JBLAS_DTYPE::F4_E2M1:
    case JBLAS_DTYPE::F4_NF4:
      return JblasGemmQuantPackBTransLocal<jblas::prologue_b::gemm::WeightKBlockF4>(
          PackedBuf, FpData, N, K, ldb, BlkSize, QuantType, ScaleDtype, isAsym, CompType, ThreadPool);
    default:
      return 0;
  }
  return 0;
}

bool JblasGemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, void* ThreadPool) {
  auto ptr = jblas::storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(PackedBuf));
  auto pth = reinterpret_cast<jblas::parallel::IThreading*>(ThreadPool);
  GetCPUDevice();
  if (ptr) {
    auto NTile = jblas::gemm::CoreAttr::get_mask_val(ptr->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK,
                                                     jblas::gemm::CoreAttr::NTILE_SHIFT);
    auto CType = jblas::gemm::CoreAttr::get_mask_val(ptr->mCoreId, jblas::gemm::CoreAttr::COMP_MASK,
                                                     jblas::gemm::CoreAttr::COMP_SHIFT);
    if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto sptr = reinterpret_cast<jblas::storage::gemm::StorageWeightKBlockNInteger*>(ptr);
      if (CType == uint32_t(jblas::gemm::CompType::COMP_FP32)) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAVX512F, tAVX512F::ISA> proB;
          proB.unpackWeight(int(N), int(K), sptr, FpData, int(ldb), pth);
          goto __END;
        }
        if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAVX2, tAVX2::ISA> proB;
          proB.unpackWeight(int(N), int(K), sptr, FpData, int(ldb), pth);
          goto __END;
        }
      }
      if (CType == uint32_t(jblas::gemm::CompType::COMP_INT8_US_INT32)) {
        if (NTile == tAMX_INT8_US::NTILE && _cd->AMX_INT8()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAMX_INT8_US, tAMX_INT8_US::ISA> proB;
          proB.unpackWeight(int(N), int(K), sptr, FpData, int(ldb), pth);
          goto __END;
        }
        if (NTile == tAVX512_VNNI::NTILE && _cd->AVX512_VNNI()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAVX512_VNNI, tAVX512_VNNI::ISA> proB;
          proB.unpackWeight(int(N), int(K), sptr, FpData, int(ldb), pth);
          goto __END;
        }
        if (NTile == tAVX_VNNI::NTILE && _cd->AVX_VNNI()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAVX_VNNI, tAVX_VNNI::ISA> proB;
          proB.unpackWeight(int(N), int(K), sptr, FpData, int(ldb), pth);
          goto __END;
        }
      }
      if (CType == uint32_t(jblas::gemm::CompType::COMP_INT8_SS_INT32)) {
        if (NTile == tAMX_INT8_SS::NTILE && _cd->AMX_INT8()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAMX_INT8_SS, tAMX_INT8_SS::ISA> proB;
          proB.unpackWeight(int(N), int(K), sptr, FpData, int(ldb), pth);
          goto __END;
        }
      }
      if (CType == uint32_t(jblas::gemm::CompType::COMP_BF16_FP32)) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAMX_BF16, tAMX_BF16::ISA> proB;
          proB.unpackWeight(int(N), int(K), sptr, FpData, int(ldb), pth);
          goto __END;
        }
      }
    }
    if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
      if (CType == uint32_t(jblas::gemm::CompType::COMP_FP32)) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          static jblas::prologue_b::gemm::WeightKBlockF4<tAVX512F, tAVX512F::ISA> proB;
          proB.unpackWeight(int(N), int(K), ptr, FpData, int(ldb), pth);
          goto __END;
        }
        if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          static jblas::prologue_b::gemm::WeightKBlockF4<tAVX2, tAVX2::ISA> proB;
          proB.unpackWeight(int(N), int(K), ptr, FpData, int(ldb), pth);
          goto __END;
        }
      }
      if (CType == uint32_t(jblas::gemm::CompType::COMP_BF16_FP32)) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          static jblas::prologue_b::gemm::WeightKBlockF4<tAMX_BF16, tAMX_BF16::ISA> proB;
          proB.unpackWeight(int(N), int(K), ptr, FpData, int(ldb), pth);
          goto __END;
        }
      }
    }
  __END:
    delete ptr;
    return true;
  }
  return false;
}
