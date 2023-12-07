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
  get_threading()->set_threads(_nth);
  return get_threading()->num_threads();
}

jblas::parallel::IThreading* get_threading() {
#ifdef _OPENMP
  static parallel::OMPThreading DefaultThreading(4);
#else
  static parallel::StdThreading DefaultThreading(4);
#endif  // _OPNEMP
  return &DefaultThreading;
}

void jblas_unpackweight_fp32(void* wptr, int n, int k, float* fp32data, int ld) {
  JblasGemmUnPackB(fp32data, wptr, static_cast<size_t>(n), static_cast<size_t>(k), static_cast<size_t>(ld),
                   get_threading());
}

void jblas_packweight_copyattr(const float* f32ptr, void* dstptr, int n, int k, int ld, void* srcptr) {
  auto wtmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(srcptr);
  if (wtmp != nullptr) {
    auto proID = wtmp->mPrologueID;
    if (wtmp->mPrologueID != JBLAS_PROLOGUEB_IDS::WeightPack) {
      auto kwtmp = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(wtmp);
      auto coreID = wtmp->mCoreId;
      auto comptype = gemm::CoreAttr::get_comp(coreID);
      auto btype = static_cast<gemm::CompType>(gemm::CompTypeHelper::get_B(comptype));
      ne_comp_type ne_comptype{ne_comp_type::NE_COMP_UNDEF};
      if (btype == gemm::CompType::tBF16) {
        ne_comptype = ne_comp_type::NE_COMP_BF16;
      }
      if (btype == gemm::CompType::tS8) {
        ne_comptype = ne_comp_type::NE_COMP_INT8;
      }
      if (btype == gemm::CompType::tFP32) {
        ne_comptype = ne_comp_type::NE_COMP_F32;
      }
      if (kwtmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
        auto niptr = reinterpret_cast<storage::gemm::StorageWeightKBlockNInteger*>(kwtmp);

        JblasGemmQuantPackB(dstptr, f32ptr, n, k, ld, niptr->mBlockSize, niptr->mDType, niptr->SDtype(),
                            niptr->IsAsym(), ne_comptype, false, get_threading());
      } else if (kwtmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
        auto f4ptr = reinterpret_cast<storage::gemm::StorageWeightKBlockF4*>(kwtmp);
        JblasGemmQuantPackB(dstptr, f32ptr, n, k, ld, f4ptr->mBlockSize, f4ptr->mDType, f4ptr->SDtype(), false,
                            ne_comptype, false, get_threading());
      }
    }
  }
  safe_delete(wtmp);
}
