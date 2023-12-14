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
#include <cstdint>

#include "jblas_defs.h"

using namespace jblas;  // NOLINT

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T>
void JblasGemmCompF32(const int M, const int N, const int K, const float* A, const int lda,
                      jblas::storage::gemm::IWeightBase* _B, float* C, const int ldc, int8_t* WorkSpace,
                      jblas::parallel::IThreading* th) {
  if (M <= 16) {
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher = tLauncher_Fp_F32F32<GemmCore_T, Wei_T>;
    static Launcher kernel;
    auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
    auto reduceA = kernel.mProA.createStorage(M, K, B->mBlockSize);
    if (B->IsAsym()) {
      reduceA.assign(WorkSpace);
    }
    typename Launcher::BEpiParam blkargs{
        B->template SPtr<int8_t>(),     B->SDtype(), B->CStep(), B->template ZPtr<int8_t>(),
        reduceA.template RPtr<float>(), reduceA.lda};
    utils::GemmProblem gp(1, M, N, K, B->mBlockSize);
    typename Launcher::Param args{gp, {A, K, &reduceA, B->ShfIndice()}, {B}, blkargs, {C, N}};
    if (B->IsAsym()) {
      jblas::parallel::GemmRunWithA<Parallel>(kernel, args, th);
    } else {
      jblas::parallel::GemmRun<Parallel>(kernel, args, th);
    }
  } else {
    using Parallel = jblas::parallel::gemm::SchedulerBase<GemmCore_T>;
    using Launcher = jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T,
                                                        jblas::prologue_a::gemm::ShuffleActivationKBlockBaseF32, Wei_T,
                                                        jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
    static Launcher kernel;
    auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
    utils::GemmProblem gp(1, M, N, K);

    typename Launcher::Param args{gp, {A, K, nullptr, B->ShfIndice()}, {B}, {C, N}};
    jblas::parallel::GemmRun<Parallel>(kernel, args, th);
  }
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T>
void JblasGemmCompInt8(const int M, const int N, const int K, const float* A, const int lda,
                       jblas::storage::gemm::IWeightBase* _B, float* C, const int ldc, int8_t* WorkSpace,
                       jblas::parallel::IThreading* th) {
  using Parallel = jblas::parallel::gemm::SchedulerKBlockS<GemmCore_T>;
  using Launcher = tLauncher_Int8_F32F32<GemmCore_T, Wei_T>;
  auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
  utils::GemmProblem gp(1, M, N, K, B->mBlockSize);
  static Launcher kernel;
  auto quanA = kernel.mProA.createStorage(M, K, B->mBlockSize, B->IsAsym());
  quanA.assign(WorkSpace);
  typename Launcher::Param args{gp, {A, K, &quanA, B->ShfIndice()}, {B}, {C, N}};
  jblas::parallel::GemmRunWithA<Parallel>(kernel, args, th);
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
      auto PackRow = jblas::gemm::CoreAttr::get_packrow(ptr->mCoreId);
      auto CType = jblas::gemm::CoreAttr::get_comp(ptr->mCoreId);
      auto btype = static_cast<jblas::gemm::CompType>(jblas::gemm::CompTypeHelper::get_B(CType));
      if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
        auto bptr = reinterpret_cast<jblas::storage::gemm::IWeightKBlockBase*>(ptr);
        auto BlkSize = bptr->mBlockSize;
        if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
          if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
            JblasGemmCompF32<tAVX512F, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                 DataParams[i].ldc, WorkSpace, pth);
          } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
            JblasGemmCompF32<tAVX2, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                              DataParams[i].ldc, WorkSpace, pth);
          }
        }
        if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
          if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
            if (M <= tAVX512_BF16::MTILE) {
              static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
              JblasGemmCompF32<tAVX512_BF16, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                       DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              JblasGemmCompF32<tAMX_BF16, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                    DataParams[i].ldc, WorkSpace, pth);
            }
          }
        }
        if (btype == jblas::gemm::CompType::tS8 && PackRow == 4) {
          // Do we need US for AMX_INT8
          if (NTile == tAMX_INT8_SS_KBlock::NTILE && _cd->AMX_INT8() && BlkSize % tAMX_INT8_SS_KBlock::KTILE == 0) {
            static_assert(tAMX_INT8_SS_KBlock::NTILE == tAVX512_VNNI_KBlock::NTILE);
            if (M <= tAVX512_VNNI_KBlock::MTILE) {
              JblasGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                               DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            } else {
              JblasGemmCompInt8<tAMX_INT8_SS_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                               DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
            }

          } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI() &&
                     BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
            JblasGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                             DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
          } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
            JblasGemmCompInt8<tAVX_VNNI_KBlock, tWeiNInt>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr,
                                                          DataParams[i].C, DataParams[i].ldc, WorkSpace, pth);
          }
        }
      }
      if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
        auto bptr = reinterpret_cast<jblas::storage::gemm::IWeightKBlockBase*>(ptr);
        auto BlkSize = bptr->mBlockSize;
        if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
          if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
            JblasGemmCompF32<tAVX512F, tWeiF4>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                               DataParams[i].ldc, WorkSpace, pth);
          } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
            JblasGemmCompF32<tAVX2, tWeiF4>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                            DataParams[i].ldc, WorkSpace, pth);
          }
        }
        if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
          if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
            if (M <= tAVX512_BF16::MTILE) {
              static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
              JblasGemmCompF32<tAVX512_BF16, tWeiF4>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                     DataParams[i].ldc, WorkSpace, pth);
            } else {
              JblasGemmCompF32<tAMX_BF16, tWeiF4>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                  DataParams[i].ldc, WorkSpace, pth);
            }
          }
        }
      }
      if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF8) {
        auto bptr = reinterpret_cast<jblas::storage::gemm::IWeightKBlockBase*>(ptr);
        auto BlkSize = bptr->mBlockSize;
        if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
          if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
            JblasGemmCompF32<tAVX512F, tWeiF8>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                               DataParams[i].ldc, WorkSpace, pth);
          } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
            JblasGemmCompF32<tAVX2, tWeiF8>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                            DataParams[i].ldc, WorkSpace, pth);
          }
        }
        if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
          if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
            if (M <= tAVX512_BF16::MTILE) {
              static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
              JblasGemmCompF32<tAVX512_BF16, tWeiF8>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                     DataParams[i].ldc, WorkSpace, pth);
            } else {
              JblasGemmCompF32<tAMX_BF16, tWeiF8>(M, N, K, DataParams[i].A, DataParams[i].lda, ptr, DataParams[i].C,
                                                  DataParams[i].ldc, WorkSpace, pth);
            }
          }
        }
      }
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
                          bool isAsym, int* shuffle_indice) {
  static T launcher;
  using WType = typename T::PrologueB::StorageWeight;
  WType stor(0);
  if constexpr (std::is_same_v<typename T::PrologueB,
                               jblas::prologue_b::gemm::WeightKBlockNInteger<typename T::GemmCore, T::ISA>>) {
    stor = launcher.mProB.createStorage(N, K, block_size, QuantType, ScaleDtype, JBLAS_DTYPE::BF16, isAsym);
    if (shuffle_indice != nullptr) {
      launcher.mProB.enableShuffle(&stor);
    }
  } else {
    stor = launcher.mProB.createStorage(N, K, block_size, QuantType, ScaleDtype);
    (void)(shuffle_indice);
  }

  // Reduce dtype set to bf16
  return stor.mSize;
}
template <template <class, JBLAS_ISA> class Wei_T>
static size_t JblasGemmPackBSizeLocal(size_t N, size_t K, size_t BlkSize, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype,
                                      bool isAsym, ne_comp_type CompType, int* shuffle_indice) {
  GetCPUDevice();
  auto dtype_type = jblas::utils::jblas_dtype_type(QuantType);
  auto constexpr dtype_int = jblas::utils::jblas_dtype_type(JBLAS_DTYPE::TypeInt);
  // from low precision to high precision
  switch (CompType) {
    case NE_COMP_INT8:
      if (dtype_type == dtype_int && !isAsym) {  // asym int8 is not optimized, so fall through to others.
        if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS_KBlock::KTILE == 0) {
          return JblasBuSize<tLauncher_Int8_F32F32<tAMX_INT8_SS_KBlock, Wei_T>>(
              static_cast<int>(BlkSize), N, K, QuantType, ScaleDtype, isAsym, shuffle_indice);
        }
        if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          return JblasBuSize<tLauncher_Int8_F32F32<tAVX512_VNNI_KBlock, Wei_T>>(
              static_cast<int>(BlkSize), N, K, QuantType, ScaleDtype, isAsym, shuffle_indice);
        }
        if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          return JblasBuSize<tLauncher_Int8_F32F32<tAVX_VNNI_KBlock, Wei_T>>(static_cast<int>(BlkSize), N, K, QuantType,
                                                                             ScaleDtype, isAsym, shuffle_indice);
        }
      }
    case NE_COMP_F16:
    case NE_COMP_BF16:
      if (_cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
        return JblasBuSize<tLauncher_Int8_F32F32<tAMX_BF16, Wei_T>>(static_cast<int>(BlkSize), N, K, QuantType,
                                                                    ScaleDtype, isAsym, shuffle_indice);
      }
    case NE_COMP_F32:
    case NE_COMP_UNDEF:  // currently only f32 activation
      if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
        return JblasBuSize<tLauncher_Fp_F32F32<tAVX512F, Wei_T>>(static_cast<int>(BlkSize), N, K, QuantType, ScaleDtype,
                                                                 isAsym, shuffle_indice);
      }
      if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
        return JblasBuSize<tLauncher_Fp_F32F32<tAVX2, Wei_T>>(static_cast<int>(BlkSize), N, K, QuantType, ScaleDtype,
                                                              isAsym, shuffle_indice);
      }
      break;
    default:
      return 0;
  }
  return 0;
}

size_t JblasGemmPackBSize(size_t N, size_t K, size_t BlkSize, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype,
                          bool isAsym, ne_comp_type CompType, int* shuffle_indice) {
  switch (QuantType) {
    case JBLAS_DTYPE::S4_CLIP:
    case JBLAS_DTYPE::S4_FULLRANGE:
    case JBLAS_DTYPE::S8:
      return JblasGemmPackBSizeLocal<jblas::prologue_b::gemm::WeightKBlockNInteger>(
          N, K, BlkSize, QuantType, ScaleDtype, isAsym, CompType, shuffle_indice);
    case JBLAS_DTYPE::F8_E4M3:
    case JBLAS_DTYPE::F8_E5M2:
      return JblasGemmPackBSizeLocal<jblas::prologue_b::gemm::WeightKBlockF8>(N, K, BlkSize, QuantType, ScaleDtype,
                                                                              isAsym, CompType, shuffle_indice);
    case JBLAS_DTYPE::F4_BNB:
    case JBLAS_DTYPE::F4_E2M1:
    case JBLAS_DTYPE::F4_NF4:
      return JblasGemmPackBSizeLocal<jblas::prologue_b::gemm::WeightKBlockF4>(N, K, BlkSize, QuantType, ScaleDtype,
                                                                              isAsym, CompType, shuffle_indice);
    default:
      return 0;
  }
  return 0;
}

template <typename T>
void JblaGemmQuantPackB(void* PackedBuf, int BlkSize, const float* FpData, int N, int K, JBLAS_DTYPE QuantType,
                        JBLAS_DTYPE ScaleDtype, bool IsAsym, int ldb, bool IsTrans, void* ThreadPool) {
  static T launcher;
  using WType = typename T::PrologueB::StorageWeight;
  WType stor(0);
  if constexpr (std::is_same_v<typename T::PrologueB,
                               jblas::prologue_b::gemm::WeightKBlockNInteger<typename T::GemmCore, T::ISA>>) {
    stor = launcher.mProB.createStorage(N, K, BlkSize, QuantType, ScaleDtype, JBLAS_DTYPE::BF16, IsAsym);
  } else {
    stor = launcher.mProB.createStorage(N, K, BlkSize, QuantType, ScaleDtype);
  }
  stor.assign(reinterpret_cast<int8_t*>(PackedBuf));
  auto pth = reinterpret_cast<jblas::parallel::IThreading*>(ThreadPool);
  if (IsTrans) {
    launcher.mProB.packTransposeWeight(N, K, FpData, ldb, &stor, pth);
  } else {
    launcher.mProB.packWeight(N, K, FpData, ldb, &stor, pth);
  }
}

template <template <class, JBLAS_ISA> class Wei_T>
static bool JblasGemmQuantPackBLocal(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb,
                                     size_t BlkSize, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype, bool isAsym,
                                     ne_comp_type CompType, bool isTrans, void* ThreadPool) {
  GetCPUDevice();
  auto dtype_type = jblas::utils::jblas_dtype_type(QuantType);
  auto constexpr dtype_int = jblas::utils::jblas_dtype_type(JBLAS_DTYPE::TypeInt);
  switch (CompType) {
    case NE_COMP_INT8:
      if (dtype_type == dtype_int && !isAsym) {  // asym int8 is not optimized, so fall through to others.
        if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS_KBlock::KTILE == 0) {
          JblaGemmQuantPackB<tLauncher_Int8_F32F32<tAMX_INT8_SS_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
              ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
          return true;
        }
        if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          JblaGemmQuantPackB<tLauncher_Int8_F32F32<tAVX512_VNNI_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
              ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
          return true;
        }
        if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          JblaGemmQuantPackB<tLauncher_Int8_F32F32<tAVX_VNNI_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
              ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
          return true;
        }
      }
    case NE_COMP_F16:
    case NE_COMP_BF16:
      if (_cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
        JblaGemmQuantPackB<tLauncher_Fp_F32F32<tAMX_BF16, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
            ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
        return true;
      }
    case NE_COMP_F32:
    case NE_COMP_UNDEF:  // currently only f32 activation
      if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
        JblaGemmQuantPackB<tLauncher_Fp_F32F32<tAVX512F, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
            ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
        return true;
      }
      if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
        JblaGemmQuantPackB<tLauncher_Fp_F32F32<tAVX2, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), FpData, static_cast<int>(N), static_cast<int>(K), QuantType,
            ScaleDtype, isAsym, static_cast<int>(ldb), isTrans, ThreadPool);
        return true;
      }
    default:
      return false;
  }
  return false;
}

bool JblasGemmQuantPackB(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb, size_t BlkSize,
                         JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype, bool isAsym, ne_comp_type CompType,
                         bool isTrans, void* ThreadPool) {
  switch (QuantType) {
    case JBLAS_DTYPE::S4_CLIP:
    case JBLAS_DTYPE::S4_FULLRANGE:
    case JBLAS_DTYPE::S8:
      return JblasGemmQuantPackBLocal<jblas::prologue_b::gemm::WeightKBlockNInteger>(
          PackedBuf, FpData, N, K, ldb, BlkSize, QuantType, ScaleDtype, isAsym, CompType, isTrans, ThreadPool);
    case JBLAS_DTYPE::F8_E5M2:
    case JBLAS_DTYPE::F8_E4M3:
      return JblasGemmQuantPackBLocal<jblas::prologue_b::gemm::WeightKBlockF8>(
          PackedBuf, FpData, N, K, ldb, BlkSize, QuantType, ScaleDtype, isAsym, CompType, isTrans, ThreadPool);
    case JBLAS_DTYPE::F4_BNB:
    case JBLAS_DTYPE::F4_E2M1:
    case JBLAS_DTYPE::F4_NF4:
      return JblasGemmQuantPackBLocal<jblas::prologue_b::gemm::WeightKBlockF4>(
          PackedBuf, FpData, N, K, ldb, BlkSize, QuantType, ScaleDtype, isAsym, CompType, isTrans, ThreadPool);
    default:
      return false;
  }
  return false;
}

template <typename T>
void JblaGemmPackBImpl(void* PackedBuf, int BlkSize, const int8_t* QData, const float* Scales, const int8_t* Zp, int N,
                       int K, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype, bool IsAsym, int ldb, int* shuffle_indice,
                       void* ThreadPool) {
  static T launcher;
  using WType = typename T::PrologueB::StorageWeight;
  auto pth = reinterpret_cast<jblas::parallel::IThreading*>(ThreadPool);
  WType stor(0);
  if constexpr (std::is_same_v<typename T::PrologueB,
                               jblas::prologue_b::gemm::WeightKBlockNInteger<typename T::GemmCore, T::ISA>>) {
    stor = launcher.mProB.createStorage(N, K, BlkSize, QuantType, ScaleDtype, JBLAS_DTYPE::BF16, IsAsym);
    if (shuffle_indice != nullptr) {
      launcher.mProB.enableShuffle(&stor);
      stor.assign(reinterpret_cast<int8_t*>(PackedBuf));
      launcher.mProB.setShuffleIndices(shuffle_indice, &stor, pth);
    } else {
      stor.assign(reinterpret_cast<int8_t*>(PackedBuf));
    }
  } else {
    (void)(shuffle_indice);
    stor = launcher.mProB.createStorage(N, K, BlkSize, QuantType, ScaleDtype);
    stor.assign(reinterpret_cast<int8_t*>(PackedBuf));
  }
  launcher.mProB.packQWeight(N, K, QData, ldb, Scales, IsAsym ? Zp : nullptr, &stor, pth);
}

template <template <class, JBLAS_ISA> class Wei_T>
static bool JblasGemmPackBLocal(void* PackedBuf, const int8_t* QData, const float* Scales, const int8_t* Zp, size_t N,
                                size_t K, size_t ldb, size_t BlkSize, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype,
                                bool isAsym, ne_comp_type CompType, int* shuffle_indice, void* ThreadPool) {
  GetCPUDevice();
  auto dtype_type = jblas::utils::jblas_dtype_type(QuantType);
  auto constexpr dtype_int = jblas::utils::jblas_dtype_type(JBLAS_DTYPE::TypeInt);
  if (dtype_type != dtype_int) {
    return false;
  }
  switch (CompType) {
    case NE_COMP_INT8:
      if (dtype_type == dtype_int && !isAsym) {  // asym int8 is not optimized, so fall through to others.
        if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS_KBlock::KTILE == 0) {
          JblaGemmPackBImpl<tLauncher_Int8_F32F32<tAMX_INT8_SS_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
              QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
          return true;
        }
        if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          JblaGemmPackBImpl<tLauncher_Int8_F32F32<tAVX512_VNNI_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
              QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
          return true;
        }
        if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          JblaGemmPackBImpl<tLauncher_Int8_F32F32<tAVX_VNNI_KBlock, Wei_T>>(
              PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
              QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
          return true;
        }
      }
    case NE_COMP_F16:
    case NE_COMP_BF16:
      if (_cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
        JblaGemmPackBImpl<tLauncher_Fp_F32F32<tAMX_BF16, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
            QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
        return true;
      }
    case NE_COMP_F32:
    case NE_COMP_UNDEF:  // currently only f32 activation
      if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
        JblaGemmPackBImpl<tLauncher_Fp_F32F32<tAVX512F, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
            QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
        return true;
      }
      if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
        JblaGemmPackBImpl<tLauncher_Fp_F32F32<tAVX2, Wei_T>>(
            PackedBuf, static_cast<int>(BlkSize), QData, Scales, Zp, static_cast<int>(N), static_cast<int>(K),
            QuantType, ScaleDtype, isAsym, static_cast<int>(ldb), shuffle_indice, ThreadPool);
        return true;
      }
    default:
      return false;
  }
  return false;
}

bool JblasGemmPackB(void* PackedBuf, const int8_t* QData, const float* Scales, const int8_t* Zp, size_t N, size_t K,
                    size_t ldb, size_t BlkSize, JBLAS_DTYPE QuantType, JBLAS_DTYPE ScaleDtype, bool isAsym,
                    ne_comp_type CompType, int* shuffle_indice, void* ThreadPool) {
  // only for integer quant
  switch (QuantType) {
    case JBLAS_DTYPE::S4_CLIP:
    case JBLAS_DTYPE::S4_FULLRANGE:
    case JBLAS_DTYPE::S8:
      return JblasGemmPackBLocal<jblas::prologue_b::gemm::WeightKBlockNInteger>(PackedBuf, QData, Scales, Zp, N, K, ldb,
                                                                                BlkSize, QuantType, ScaleDtype, isAsym,
                                                                                CompType, shuffle_indice, ThreadPool);
    default:
      return false;
  }
  return false;
}

bool JblasGemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, void* ThreadPool) {
  auto ptr = jblas::storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(PackedBuf));
  auto pth = reinterpret_cast<jblas::parallel::IThreading*>(ThreadPool);
  GetCPUDevice();
  if (ptr) {
    auto NTile = jblas::gemm::CoreAttr::get_mask_val(ptr->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK,
                                                     jblas::gemm::CoreAttr::NTILE_SHIFT);
    auto PackRow = jblas::gemm::CoreAttr::get_packrow(ptr->mCoreId);
    auto CType = jblas::gemm::CoreAttr::get_comp(ptr->mCoreId);
    auto btype = static_cast<jblas::gemm::CompType>(jblas::gemm::CompTypeHelper::get_B(CType));
    if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto sptr = reinterpret_cast<jblas::storage::gemm::StorageWeightKBlockNInteger*>(ptr);
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAVX512F, tAVX512F::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAVX2, tAVX2::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        }
      }
      if (btype == jblas::gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_SS_KBlock::NTILE && _cd->AMX_INT8()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAMX_INT8_SS_KBlock, tAMX_INT8_SS_KBlock::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAVX512_VNNI_KBlock, tAVX512_VNNI_KBlock::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAVX_VNNI_KBlock, tAVX_VNNI_KBlock::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          static jblas::prologue_b::gemm::WeightKBlockNInteger<tAMX_BF16, tAMX_BF16::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), sptr, FpData, static_cast<int>(ldb), pth);
        }
      }
    }
    if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          static jblas::prologue_b::gemm::WeightKBlockF4<tAVX512F, tAVX512F::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), ptr, FpData, static_cast<int>(ldb), pth);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          static jblas::prologue_b::gemm::WeightKBlockF4<tAVX2, tAVX2::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), ptr, FpData, static_cast<int>(ldb), pth);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          static jblas::prologue_b::gemm::WeightKBlockF4<tAMX_BF16, tAMX_BF16::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), ptr, FpData, static_cast<int>(ldb), pth);
        }
      }
    }
    if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF8) {
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          static jblas::prologue_b::gemm::WeightKBlockF8<tAVX512F, tAVX512F::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), ptr, FpData, static_cast<int>(ldb), pth);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          static jblas::prologue_b::gemm::WeightKBlockF8<tAVX2, tAVX2::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), ptr, FpData, static_cast<int>(ldb), pth);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          static jblas::prologue_b::gemm::WeightKBlockF8<tAMX_BF16, tAMX_BF16::ISA> proB;
          proB.unpackWeight(static_cast<int>(N), static_cast<int>(K), ptr, FpData, static_cast<int>(ldb), pth);
        }
      }
    }
    delete ptr;
    return true;
  }
  return false;
}
