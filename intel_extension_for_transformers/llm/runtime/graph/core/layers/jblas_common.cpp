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
#include "jblas/jit_blas_weight_compression.h"
using namespace jblas;
using namespace ne_jblas;

void jblas_init() {
  GetCPUDevice();
  if (_cd->AMX_BF16() || _cd->AMX_INT8()) {
    utils::request_perm_xtile_data();
  }
  _cd->print();
}

void jblas_timer(bool _init) {
  static utils::timer<utils::microseconds> tr;
  if (_init)
    tr.start();
  else
    printf("time :%f us\n", tr.stop());
}

int jblas_set_threads(int _nth) {
  jblas::utils::parallel::CpuDevice::getInstance()->setThreads(_nth);
  return jblas::utils::parallel::CpuDevice::getInstance()->getThreads();
}

template <template <class, JBLAS_ISA> class Wei_T, class GC_T>
void jblas_unpackweight(void* wptr, int n, int k, float* fp32data, int ld) {
  GetCPUDevice();
  using ProB_AVX512 = Wei_T<GC_T, JblasAVX512F>;
  using Prob_AVX2 = Wei_T<GC_T, JblasAVX2>;
  if (_cd->AVX512F()) {
    static ProB_AVX512 prob;
    prob.unpackWeight(n, k, wptr, fp32data, ld);
    return;
  }
  if (_cd->AVX2()) {
    static Prob_AVX2 prob;
    prob.unpackWeight(n, k, wptr, fp32data, ld);
    return;
  }
}

void jblas_unpackweight_fp32(void* wptr, int n, int k, float* fp32data, int ld) {
  auto wtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wptr);
  if (wtmp != nullptr) {
    if (wtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32)) {
      if (wtmp->mCoreType == GcCompFp32::TYPE) {
        jblas_unpackweight<WeiS4ClipFp32, GcCompFp32>(wtmp, n, k, fp32data, ld);
      }
      if (wtmp->mCoreType == GcCompInt8::TYPE || wtmp->mCoreType == GcCompInt8KBlock::TYPE) {
        jblas_unpackweight<WeiS4ClipFp32, GcCompInt8>(wtmp, n, k, fp32data, ld);
      }
    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32)) {
      if (wtmp->mCoreType == GcCompFp32::TYPE) {
        jblas_unpackweight<WeiS8Fp32, GcCompFp32>(wtmp, n, k, fp32data, ld);
      }
      if (wtmp->mCoreType == GcCompInt8::TYPE || wtmp->mCoreType == GcCompInt8KBlock::TYPE) {
        jblas_unpackweight<WeiS8Fp32, GcCompInt8>(wtmp, n, k, fp32data, ld);
      }
    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
      if (wtmp->mCoreType == GcCompFp32::TYPE) {
        jblas_unpackweight<WeiS8Fp32PerN, GcCompFp32>(wtmp, n, k, fp32data, ld);
      }
      if (wtmp->mCoreType == GcCompInt8::TYPE || wtmp->mCoreType == GcCompInt8KBlock::TYPE) {
        jblas_unpackweight<WeiS8Fp32PerN, GcCompInt8>(wtmp, n, k, fp32data, ld);
      }
    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
      if (wtmp->mCoreType == GcCompFp32::TYPE) {
        jblas_unpackweight<WeiS4ClipFp32PerN, GcCompFp32>(wtmp, n, k, fp32data, ld);
      }
      if (wtmp->mCoreType == GcCompInt8::TYPE || wtmp->mCoreType == GcCompInt8KBlock::TYPE) {
        jblas_unpackweight<WeiS4ClipFp32PerN, GcCompInt8>(wtmp, n, k, fp32data, ld);
      }
    }
  }
  safe_delete(wtmp);
}

template <template <class, JBLAS_ISA> class Wei_T, class GC_T>
void jblas_packweight(const float* fp32data, void* dstptr, int n, int k, int ld, void* srcptr) {
  GetCPUDevice();
  using ProB_AVX512 = Wei_T<GC_T, JblasAVX512F>;
  using Prob_AVX2 = Wei_T<GC_T, JblasAVX2>;
  using ST = typename Prob_AVX2::StorageWeight;
  static Prob_AVX2 prob;
  ST tmp(gemm::GemmCoreType::Undef);
  auto src = (ST*)srcptr;
  if constexpr (std::is_same_v<Prob_AVX2, WeiS4ClipFp32<GC_T, JblasAVX2>> ||
                std::is_same_v<Prob_AVX2, WeiS8Fp32<GC_T, JblasAVX2>>) {
    tmp = prob.createStorage(n, k, src->mBlockSize, src->mIsAsym);
  }
  if constexpr (std::is_same_v<Prob_AVX2, WeiS4ClipFp32PerN<GC_T, JblasAVX2>> ||
                std::is_same_v<Prob_AVX2, WeiS8Fp32PerN<GC_T, JblasAVX2>>) {
    tmp = prob.createStorage(n, k, src->mIsAsym);
  }
  tmp.assign((int8_t*)dstptr);
  if (_cd->AVX512F()) {
    static ProB_AVX512 prob;
    prob.packWeight(n, k, fp32data, ld, &tmp);
    return;
  }
  if (_cd->AVX2()) {
    prob.packWeight(n, k, fp32data, ld, &tmp);
    return;
  }
}

void jblas_packweight_copyattr(const float* f32ptr, void* dstpr, int n, int k, int ld, void* srcptr) {
  auto wtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(srcptr);
  if (wtmp != nullptr) {
    if (wtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32)) {
      if (wtmp->mCoreType == GcCompFp32::TYPE) {
        jblas_packweight<WeiS4ClipFp32, GcCompFp32>(f32ptr, dstpr, n, k, ld, wtmp);
      }
      if (wtmp->mCoreType == GcCompInt8KBlock::TYPE) {
        jblas_packweight<WeiS4ClipFp32, GcCompInt8KBlock>(f32ptr, dstpr, n, k, ld, wtmp);
      }
    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32)) {
      if (wtmp->mCoreType == GcCompFp32::TYPE) {
        jblas_packweight<WeiS8Fp32, GcCompFp32>(f32ptr, dstpr, n, k, ld, wtmp);
      }
      if (wtmp->mCoreType == GcCompInt8KBlock::TYPE) {
        jblas_packweight<WeiS8Fp32, GcCompInt8KBlock>(f32ptr, dstpr, n, k, ld, wtmp);
      }
    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
      if (wtmp->mCoreType == GcCompFp32::TYPE) {
        jblas_packweight<WeiS8Fp32PerN, GcCompFp32>(f32ptr, dstpr, n, k, ld, wtmp);
      }
      if (wtmp->mCoreType == GcCompInt8::TYPE) {
        jblas_packweight<WeiS8Fp32PerN, GcCompInt8>(f32ptr, dstpr, n, k, ld, wtmp);
      }

    } else if (wtmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
      if (wtmp->mCoreType == GcCompFp32::TYPE) {
        jblas_packweight<WeiS4ClipFp32PerN, GcCompFp32>(f32ptr, dstpr, n, k, ld, wtmp);
      }
      if (wtmp->mCoreType == GcCompInt8::TYPE) {
        jblas_packweight<WeiS4ClipFp32PerN, GcCompInt8>(f32ptr, dstpr, n, k, ld, wtmp);
      }
    }
  }
  safe_delete(wtmp);
}
