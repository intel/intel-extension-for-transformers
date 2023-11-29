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

namespace ip_qkv {

template <class Parallel_T, class Launch_T>
void GemmRun_QKV(Launch_T& launcher, const typename Launch_T::Param* args, parallel::IThreading* th) {
  device::CpuBase cb;
  Parallel_T para({th->num_threads(), args[0].problem, cb.mL2Cache, cb.mL1Cache});
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename Parallel_T::ThreadProblem thdp{tidx};
    para.getIndex(thdp);
    if (thdp.valid) {
      for (size_t i = 0; i < 3; i++) {
        launcher.run(args[i], thdp);
      }
    }
  });
}

template <class Parallel_T, class Launch_T>
void GemmRunWithA_QKV(Launch_T& launcher, const typename Launch_T::Param* args, parallel::IThreading* th) {
  device::CpuBase cb;
  Parallel_T para({th->num_threads(), args[0].problem, cb.mL2Cache, cb.mL1Cache});
  using AParall = typename Launch_T::PrologueA::Parallel;
  auto apara = launcher.mProA.createParallel(th->num_threads(), args[0].problem);
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename AParall::ThreadProblem thdpA{tidx};
    apara.getIndex(thdpA);
    if (thdpA.valid) {
      launcher.mProA.run(args[0].paramA, thdpA);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp{tidx};
    para.getIndex(thdp);
    if (thdp.valid) {
      for (size_t i = 0; i < 3; i++) {
        launcher.run(args[i], thdp);
      }
    }
  });
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T>
void JblasGemmCompF32(const int M, const int N, const int K, const float* A, const int lda,
                      jblas::storage::gemm::IWeightBase* _BQ, jblas::storage::gemm::IWeightBase* _BK,
                      jblas::storage::gemm::IWeightBase* _BV, float* C, const int ldc, int8_t* WorkSpace,
                      jblas::parallel::IThreading* th) {
  if (M <= 32) {
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher = tLauncher_Fp_F32F32<GemmCore_T, Wei_T>;
    static Launcher kernel;
    auto BQ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_BQ);
    auto BK = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_BK);
    auto BV = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_BV);
    auto reduceA = kernel.mProA.createStorage(M, K, BQ->mBlockSize);
    if (BQ->IsAsym()) {
      reduceA.assign(WorkSpace);
    }
    auto cstep = static_cast<int>(BQ->CStep());
    typename Launcher::BEpiParam blkargs[3]{{BQ->template SPtr<int8_t>(), BQ->SDtype(), cstep,
                                             BQ->template ZPtr<int8_t>(), reduceA.template RPtr<float>(), reduceA.lda},
                                            {BK->template SPtr<int8_t>(), BK->SDtype(), cstep,
                                             BK->template ZPtr<int8_t>(), reduceA.template RPtr<float>(), reduceA.lda},
                                            {BV->template SPtr<int8_t>(), BV->SDtype(), cstep,
                                             BV->template ZPtr<int8_t>(), reduceA.template RPtr<float>(), reduceA.lda}};
    utils::GemmProblem gp(1, M, N, K, BQ->mBlockSize);  // If mixed blocksize, change it to three instances.
    typename Launcher::Param args[3]{{gp, {A, lda, &reduceA}, {BQ}, blkargs[0], {C, ldc}},
                                     {gp, {A, lda, &reduceA}, {BK}, blkargs[1], {C + M * ldc, ldc}},
                                     {gp, {A, lda, &reduceA}, {BV}, blkargs[2], {C + M * ldc * 2, ldc}}};
    if (BQ->IsAsym()) {
      GemmRunWithA_QKV<Parallel>(kernel, args, th);
    } else {
      GemmRun_QKV<Parallel>(kernel, args, th);
    }
  } else {
    using Parallel = jblas::parallel::gemm::SchedulerBase<GemmCore_T>;
    using Launcher = jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T,
                                                        jblas::prologue_a::gemm::ActivationKBlockBaseF32, Wei_T,
                                                        jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
    static Launcher kernel;
    auto BQ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_BQ);
    auto BK = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_BK);
    auto BV = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_BV);

    utils::GemmProblem gp(1, M, N, K);
    typename Launcher::Param args[3]{{gp, {A, K}, {BQ}, {C, ldc}},
                                     {gp, {A, K}, {BK}, {C + M * ldc, ldc}},
                                     {gp, {A, K}, {BV}, {C + M * ldc * 2, ldc}}};
    GemmRun_QKV<Parallel>(kernel, args, th);
  }
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T>
void JblasGemmCompInt8(const int M, const int N, const int K, const float* A, const int lda,
                       jblas::storage::gemm::IWeightBase* _BQ, jblas::storage::gemm::IWeightBase* _BK,
                       jblas::storage::gemm::IWeightBase* _BV, float* C, const int ldc, int8_t* WorkSpace,
                       jblas::parallel::IThreading* th) {
  using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
  using Launcher = tLauncher_Int8_F32F32<GemmCore_T, Wei_T>;
  auto BQ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_BQ);
  auto BK = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_BK);
  auto BV = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(_BV);
  static Launcher kernel;
  auto quanA = kernel.mProA.createStorage(M, K, BQ->mBlockSize, BQ->IsAsym());
  quanA.assign(WorkSpace);
  utils::GemmProblem gp(1, M, N, K, BQ->mBlockSize);  // If mixed blocksize, change it to three instances.
  typename Launcher::Param args[3]{
      {gp,
       {A, K, &quanA},
       {BQ},
       {BQ->template SPtr<int8_t>(), BQ->SDtype(), BQ->CStep(), quanA.template SPtr<float>(), quanA.CStep(),
        quanA.template ZPtr<uint8_t>(), BQ->template RPtr<float>(), BQ->RDtype(), BQ->template ZPtr<int8_t>(),
        quanA.template RPtr<float>(), BQ->mBlockSize},
       {C, N}},
      {gp,
       {A, K, &quanA},
       {BK},
       {BK->template SPtr<int8_t>(), BK->SDtype(), BK->CStep(), quanA.template SPtr<float>(), quanA.CStep(),
        quanA.template ZPtr<uint8_t>(), BK->template RPtr<float>(), BK->RDtype(), BK->template ZPtr<int8_t>(),
        quanA.template RPtr<float>(), BK->mBlockSize},
       {C, N}},
      {gp,
       {A, K, &quanA},
       {BV},
       {BV->template SPtr<int8_t>(), BV->SDtype(), BV->CStep(), quanA.template SPtr<float>(), quanA.CStep(),
        quanA.template ZPtr<uint8_t>(), BV->template RPtr<float>(), BV->RDtype(), BV->template ZPtr<int8_t>(),
        quanA.template RPtr<float>(), BV->mBlockSize},
       {C, N}}};
  GemmRunWithA_QKV<Parallel>(kernel, args, th);
}
}  // namespace ip_qkv

unsigned long long jblas_fusion_QKV_f32f32_get_workspace_size(int _m, int _n, int _k, void* w1ptr) {
  // maximum padding
  // we can parse w1ptr to get a accurate size, but not necessary
  int constexpr padding = 128;
  size_t s = size_t(_m) * utils::padto((size_t)_k, padding) * 4;
  return s;
}

bool jblas_fusion_QKV_f32f32_support(void* wqptr, void* wkptr, void* wvptr, int _m, int _n, int _k) {
  GetCPUDevice();
  bool support = false;
  auto wqtmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(wqptr);
  auto wktmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(wkptr);
  auto wvtmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(wvptr);
  if (wqtmp && wktmp && wvtmp) {
    storage::gemm::IWeightBase* wset[] = {wqtmp, wktmp, wvtmp};
    if (samePackedWeight(wset, 3)) {
      if (wqtmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
        constexpr size_t EleNum = sizeof(AllCores) / sizeof(AllCores[0]);
        support = contains(wqtmp->mCoreId, AllCores, EleNum);
        support &= hasISA(AllCores, EleNum);
      } else if (wqtmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
        constexpr size_t EleNum = sizeof(AllCores) / sizeof(AllCores[0]);
        support = contains(wqtmp->mCoreId, AllCores, EleNum);
        support &= hasISA(AllCores, EleNum);
      }
    }
  }
  safe_delete(wqtmp);
  safe_delete(wktmp);
  safe_delete(wvtmp);
  return support;
}

// f32f32: activation & output dtype
void jblas_fusion_QKV_f32f32_forward(float* activation, void* wqptr, void* wkptr, void* wvptr, float* output, int _m,
                                     int _n, int _k, int lda, int ldo, void* _workspace) {
  GetCPUDevice();
  auto wqtmp = storage::gemm::PackedWeightParser::deserialBuffer(wqptr);
  auto wktmp = storage::gemm::PackedWeightParser::deserialBuffer(wkptr);
  auto wvtmp = storage::gemm::PackedWeightParser::deserialBuffer(wvptr);
  // must check support before forward, there is no need to check support twice.
  auto ptr = wqtmp;
  auto coretype = ptr->mCoreId;
  auto NTile = jblas::gemm::CoreAttr::get_mask_val(ptr->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK,
                                                   jblas::gemm::CoreAttr::NTILE_SHIFT);
  auto CType = jblas::gemm::CoreAttr::get_mask_val(ptr->mCoreId, jblas::gemm::CoreAttr::COMP_MASK,
                                                   jblas::gemm::CoreAttr::COMP_SHIFT);
  auto pth = get_threading();
  auto workspace = reinterpret_cast<int8_t*>(_workspace);
  if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
    if (CType == uint32_t(gemm::CompType::COMP_FP32)) {
      if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
        ip_qkv::JblasGemmCompF32<tAVX512F, tWeiNInt>(_m, _n, _k, activation, lda, wqtmp, wktmp, wvtmp, output, ldo,
                                                     workspace, pth);
        goto __END;
      }
      if (NTile == tAVX2::NTILE && _cd->AVX2()) {
        ip_qkv::JblasGemmCompF32<tAVX2, tWeiNInt>(_m, _n, _k, activation, lda, wqtmp, wktmp, wvtmp, output, ldo,
                                                  workspace, pth);
        goto __END;
      }
    }
    if (CType == uint32_t(gemm::CompType::COMP_BF16_FP32)) {
      if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
        ip_qkv::JblasGemmCompF32<tAMX_BF16, tWeiNInt>(_m, _n, _k, activation, lda, wqtmp, wktmp, wvtmp, output, ldo,
                                                      workspace, pth);
        goto __END;
      }
    }
    if (CType == uint32_t(gemm::CompType::COMP_INT8_US_INT32)) {
      if (NTile == tAMX_INT8_US::NTILE && _cd->AMX_INT8()) {
        ip_qkv::JblasGemmCompInt8<tAMX_INT8_US, tWeiNInt>(_m, _n, _k, activation, lda, wqtmp, wktmp, wvtmp, output, ldo,
                                                          workspace, pth);
        goto __END;
      }
      if (NTile == tAVX512_VNNI::NTILE && _cd->AVX512_VNNI()) {
        ip_qkv::JblasGemmCompInt8<tAVX512_VNNI, tWeiNInt>(_m, _n, _k, activation, lda, wqtmp, wktmp, wvtmp, output, ldo,
                                                          workspace, pth);
        goto __END;
      }
      if (NTile == tAVX_VNNI::NTILE && _cd->AVX_VNNI()) {
        ip_qkv::JblasGemmCompInt8<tAVX_VNNI, tWeiNInt>(_m, _n, _k, activation, lda, wqtmp, wktmp, wvtmp, output, ldo,
                                                       workspace, pth);
        goto __END;
      }
    }
    if (CType == uint32_t(gemm::CompType::COMP_INT8_SS_INT32)) {
      if (NTile == tAMX_INT8_SS::NTILE && _cd->AMX_INT8()) {
        ip_qkv::JblasGemmCompInt8<tAMX_INT8_SS, tWeiNInt>(_m, _n, _k, activation, lda, wqtmp, wktmp, wvtmp, output, ldo,
                                                          workspace, pth);
        goto __END;
      }
    }
  }
  if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
    if (CType == uint32_t(gemm::CompType::COMP_FP32)) {
      if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
        ip_qkv::JblasGemmCompF32<tAVX512F, tWeiF4>(_m, _n, _k, activation, lda, wqtmp, wktmp, wvtmp, output, ldo,
                                                   workspace, pth);
        goto __END;
      }
      if (NTile == tAVX2::NTILE && _cd->AVX2()) {
        ip_qkv::JblasGemmCompF32<tAVX2, tWeiF4>(_m, _n, _k, activation, lda, wqtmp, wktmp, wvtmp, output, ldo,
                                                workspace, pth);
        goto __END;
      }
    }
    if (CType == uint32_t(gemm::CompType::COMP_BF16_FP32)) {
      if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
        ip_qkv::JblasGemmCompF32<tAMX_BF16, tWeiF4>(_m, _n, _k, activation, lda, wqtmp, wktmp, wvtmp, output, ldo,
                                                    workspace, pth);
        goto __END;
      }
    }
  }
__END:
  safe_delete(wqtmp);
  safe_delete(wktmp);
  safe_delete(wvtmp);
}
