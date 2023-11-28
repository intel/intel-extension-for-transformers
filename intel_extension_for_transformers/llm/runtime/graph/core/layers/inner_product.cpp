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
#include "jblas_gemm.h"

using namespace jblas;
using namespace ne_jblas;

unsigned long long jblas_f32f32_get_workspace_size(int _m, int _n, int _k, void* wptr) {
  // maximum padding
  int constexpr padding = 128;
  size_t s = size_t(_m) * utils::padto((size_t)_k, padding) * 4;
  return s;
}

// f32f32: activation & output dtype
void jblas_f32f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda, int ldo,
                          void* workspace) {
  auto ret = JblasRuntimeError;
  JBLAS_GEMM_DATA_PACKED_PARAMS param{activation, weiptr, output, lda, ldo};
  if (!JblasGemmBatchDriver(_m, _n, _k, 1, &param, reinterpret_cast<int8_t*>(workspace), get_threading())) {
    printf("Err: invalid parameters\n");
    assert(0);
  }
}

namespace ip_add {
template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T, template <class, JBLAS_ISA> class Act_T>
void JblasGemmCompF32(const int M, const int N, const int K, const float* A, const int lda,
                      jblas::storage::gemm::WeightBase* _B, float* C, const int ldc, float* bias, bool broadcast_bias,
                      int8_t* WorkSpace, jblas::parallel::IThreading* th) {
  if (M <= 32) {
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher =
        jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T,
                                             jblas::epilogue::gemm::CompFp32BlockEpilogue, custom::epilogue::AddFp32>;
    static Launcher kernel;
    auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
    auto reduceA = kernel.mProA.createStorage(M, K, B->mBlockSize);
    if (B->mIsAsym) {
      reduceA.assign(WorkSpace);
    }
    typename Launcher::BEpiParam blkargs{
        B->template SPtr<int8_t>(),    B->mScaT,   B->mCStep, B->template ZPtr<int8_t>(),
        reduceA.template get<float>(), reduceA.lda};

    typename Launcher::Param args{
        M, N, K, B->mBlockSize, {A, lda, &reduceA}, {B}, blkargs, {C, bias, ldc, broadcast_bias ? 0 : ldc}};
    if (B->mIsAsym) {
      jblas::parallel::GemmKBlockRunWithA<Parallel>(kernel, args, th);
    } else {
      jblas::parallel::GemmKBlockRun<Parallel>(kernel, args, th);
    }
  } else {
    using Parallel = jblas::parallel::gemm::SchedulerBase<GemmCore_T>;
    using Launcher =
        jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, Act_T, jblas::prologue_b::gemm::WeightKBlockS4,
                                           custom::epilogue::AddFp32>;
    auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
    static Launcher kernel;
    typename Launcher::Param args{M, N, K, {A, lda}, {B}, {C, bias, ldc, broadcast_bias ? 0 : ldc}};
    jblas::parallel::GemmBaseRun<Parallel>(kernel, args, th);
  }
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T>
void JblasGemmCompInt8(const int M, const int N, const int K, const float* A, const int lda,
                       jblas::storage::gemm::WeightBase* _B, float* C, const int ldc, float* bias, bool broadcast_bias,
                       int8_t* WorkSpace, jblas::parallel::IThreading* th) {
  using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
  using Launcher =
      jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T,
                                           jblas::prologue_a::gemm::ActivationF32KBlockQuantize, Wei_T,
                                           jblas::epilogue::gemm::CompInt8BlockEpilogue, custom::epilogue::AddFp32>;

  auto B = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_B);
  static Launcher kernel;
  auto quanA = kernel.mProA.createStorage(M, K, B->mBlockSize, B->mIsAsym);
  quanA.assign(WorkSpace);
  typename Launcher::Param args{M,
                                N,
                                K,
                                B->mBlockSize,
                                {A, lda, &quanA},
                                {B},
                                {B->template SPtr<int8_t>(), B->mScaT, B->mCStep, quanA.template SPtr<float>(),
                                 quanA.mCStep, quanA.template ZPtr<uint8_t>(), B->template RPtr<float>(), B->mRedT,
                                 B->template ZPtr<int8_t>(), quanA.template RPtr<float>(), B->mBlockSize},
                                {C, bias, ldc, broadcast_bias ? 0 : ldc}};
  jblas::parallel::GemmKBlockRunWithA<Parallel>(kernel, args, th);
}
}  // namespace ip_add

bool jblas_fusion_add_f32f32_support(void* weiptr, int _m, int _n, int _k) {
  GetCPUDevice();
  bool support = false;
  auto wtmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(weiptr);
  if (wtmp) {
    if (wtmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
      constexpr size_t EleNum = sizeof(AllCores) / sizeof(AllCores[0]);
      support = contains(wtmp->mCoreId, AllCores, EleNum);
      support &= hasISA(AllCores, EleNum);
    } else if (wtmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
      constexpr size_t EleNum = sizeof(AllCores) / sizeof(AllCores[0]);
      support = contains(wtmp->mCoreId, AllCores, EleNum);
      support &= hasISA(AllCores, EleNum);
    }
  }
  safe_delete(wtmp);
  return support;
}

void jblas_fusion_add_f32f32_forward(float* activation, void* weiptr, float* bias, float* output, int _m, int _n,
                                     int _k, int lda, int ldo, bool broadcast_bias, void* _workspace) {
  GetCPUDevice();
  auto pth = get_threading();
  auto ptr = jblas::storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(weiptr));
  auto workspace = reinterpret_cast<int8_t*>(_workspace);
  if (ptr) {
    auto coretype = ptr->mCoreId;
    auto NTile = jblas::gemm::CoreAttr::get_mask_val(ptr->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK,
                                                     jblas::gemm::CoreAttr::NTILE_SHIFT);
    auto CType = jblas::gemm::CoreAttr::get_mask_val(ptr->mCoreId, jblas::gemm::CoreAttr::COMP_MASK,
                                                     jblas::gemm::CoreAttr::COMP_SHIFT);
    if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
      if (CType == uint32_t(gemm::CompType::COMP_FP32)) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          ip_add::JblasGemmCompF32<tAVX512F, tWeiNInt, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                   broadcast_bias, workspace, pth);
          goto __END;
        }
        if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          ip_add::JblasGemmCompF32<tAVX2, tWeiNInt, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                broadcast_bias, workspace, pth);
          goto __END;
        }
      }
      if (CType == uint32_t(gemm::CompType::COMP_BF16_FP32)) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          ip_add::JblasGemmCompF32<tAMX_BF16, tWeiNInt, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                    broadcast_bias, workspace, pth);
          goto __END;
        }
      }
      if (CType == uint32_t(gemm::CompType::COMP_INT8_US_INT32)) {
        if (NTile == tAMX_INT8_US::NTILE && _cd->AMX_INT8()) {
          ip_add::JblasGemmCompInt8<tAMX_INT8_US, tWeiNInt>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                          broadcast_bias, workspace, pth);
          goto __END;
        }
        if (NTile == tAVX512_VNNI::NTILE && _cd->AVX512_VNNI()) {
          ip_add::JblasGemmCompInt8<tAVX512_VNNI, tWeiNInt>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                          broadcast_bias, workspace, pth);
          goto __END;
        }
        if (NTile == tAVX_VNNI::NTILE && _cd->AVX_VNNI()) {
          ip_add::JblasGemmCompInt8<tAVX_VNNI, tWeiNInt>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                       broadcast_bias, workspace, pth);
          goto __END;
        }
      }
      if (CType == uint32_t(gemm::CompType::COMP_INT8_SS_INT32)) {
        if (NTile == tAMX_INT8_SS::NTILE && _cd->AMX_INT8()) {
          ip_add::JblasGemmCompInt8<tAMX_INT8_SS, tWeiNInt>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                          broadcast_bias, workspace, pth);
          goto __END;
        }
      }
    }
    if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
      if (CType == uint32_t(gemm::CompType::COMP_FP32)) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          ip_add::JblasGemmCompF32<tAVX512F, tWeiF4, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                   broadcast_bias, workspace, pth);
          goto __END;
        }
        if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          ip_add::JblasGemmCompF32<tAVX2, tWeiF4, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                broadcast_bias, workspace, pth);
          goto __END;
        }
      }
      if (CType == uint32_t(gemm::CompType::COMP_BF16_FP32)) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          ip_add::JblasGemmCompF32<tAMX_BF16, tWeiF4, tActKBaseF32>(_m, _n, _k, activation, lda, ptr, output, ldo, bias,
                                                                    broadcast_bias, workspace, pth);
          goto __END;
        }
      }
    }
  __END:
    delete ptr;
  } else {
    printf("Wrong Input\n");
    assert(0);
  }
}
