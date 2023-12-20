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

using namespace jblas;     // NOLINT
using namespace ne_jblas;  // NOLINT

unsigned long long jblas_fusion_FFN_f32f32_get_workspace_size(int seq, int fin, int fmid, int fout,  // NOLINT
                                                              void* w1ptr, void* w2ptr) {
  // lazy size: maximum padding
  int constexpr padding = 128;
  size_t s = static_cast<size_t>(seq) * utils::padto(static_cast<size_t>(fin), padding) * 4;
  s += static_cast<size_t>(seq) * utils::padto(static_cast<size_t>(fmid), padding) * 4;
  return s;
}

namespace ffn_2w {

template <class Parallel_T, class Launch_T1, class Launch_T2>
void GemmRunWithA_ffn(Launch_T1* launcher1, Launch_T2* launcher2, const typename Launch_T1::Param& args1,
                      const typename Launch_T2::Param& args2, parallel::IThreading* th) {
  GetCPU();
  Parallel_T para1({th->num_threads(), args1.problem, _cb->getL2Cache(), _cb->getL1Cache()});
  Parallel_T para2({th->num_threads(), args2.problem, _cb->getL2Cache(), _cb->getL1Cache()});
  using AParall1 = typename Launch_T1::PrologueA::Parallel;
  using AParall2 = typename Launch_T2::PrologueA::Parallel;
  auto apara1 = launcher1->mProA.createParallel(th->num_threads(), args1.problem);
  auto apara2 = launcher2->mProA.createParallel(th->num_threads(), args2.problem);
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para1.print();
    para2.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename AParall1::ThreadProblem thdpA1{tidx};
    apara1.getIndex(thdpA1);
    if (thdpA1.valid) {
      launcher1->mProA.run(args1.paramA, thdpA1);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp1{tidx};
    para1.getIndex(thdp1);
    if (thdp1.valid) {
      launcher1->run(args1, thdp1);
    }
    th->sync();
    typename AParall2::ThreadProblem thdpA2{tidx};
    apara2.getIndex(thdpA2);
    if (thdpA2.valid) {
      launcher2->mProA.run(args2.paramA, thdpA2);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp2{tidx};
    para2.getIndex(thdp2);
    if (thdp2.valid) {
      launcher2->run(args2, thdp2);
    }
  });
}

template <class Parallel_T, class Launch_T1, class Launch_T2>
void GemmRunWithA_ffn(Launch_T1* launcher1, Launch_T2* launcher2, const typename Launch_T1::Param& args1_P,
                      const typename Launch_T1::Param& args1_E, const typename Launch_T2::Param& args2_P,
                      const typename Launch_T2::Param& args2_E, parallel::IThreading* th) {
  GetCPU();
  Parallel_T para1_P(
      {th->num_threads() - _cb->getECoreNum(), args1_P.problem, _cb->getL2Cache_P(), _cb->getL1Cache_P()});
  Parallel_T para2_P(
      {th->num_threads() - _cb->getECoreNum(), args2_P.problem, _cb->getL2Cache_P(), _cb->getL1Cache_P()});
  Parallel_T para1_E({_cb->getECoreNum(), args1_E.problem, _cb->getL2Cache_E(), _cb->getL1Cache_E()});
  Parallel_T para2_E({_cb->getECoreNum(), args2_E.problem, _cb->getL2Cache_E(), _cb->getL1Cache_E()});
  using AParall1 = typename Launch_T1::PrologueA::Parallel;
  using AParall2 = typename Launch_T2::PrologueA::Parallel;
  auto apara1_P = launcher1->mProA.createParallel(th->num_threads() - _cb->getECoreNum(), args1_P.problem);
  auto apara2_P = launcher2->mProA.createParallel(th->num_threads() - _cb->getECoreNum(), args2_P.problem);
  auto apara1_E = launcher1->mProA.createParallel(_cb->getECoreNum(), args1_E.problem);
  auto apara2_E = launcher2->mProA.createParallel(_cb->getECoreNum(), args2_E.problem);
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para1_P.print();
    para2_E.print();
    para1_P.print();
    para2_E.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    _cb->core_bond(tidx);
    int core_idx = _cb->getCoreidx(tidx);
    typename AParall1::ThreadProblem thdpA1{core_idx};
    if (_cb->getCoreType(tidx) == _cb->E_CORE) {
      apara1_E.getIndex(thdpA1);
      if (thdpA1.valid) launcher1->mProA.run(args1_E.paramA, thdpA1);
    } else {
      apara1_P.getIndex(thdpA1);
      if (thdpA1.valid) launcher1->mProA.run(args1_P.paramA, thdpA1);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp1{core_idx};
    if (_cb->getCoreType(tidx) == _cb->E_CORE) {
      para1_E.getIndex(thdp1);
      if (thdp1.valid) launcher1->run(args1_E, thdp1);
    } else {
      para1_P.getIndex(thdp1);
      if (thdp1.valid) launcher1->run(args1_P, thdp1);
    }
    th->sync();
    typename AParall2::ThreadProblem thdpA2{core_idx};
    if (_cb->getCoreType(tidx) == _cb->E_CORE) {
      apara2_E.getIndex(thdpA2);
      if (thdpA2.valid) launcher2->mProA.run(args2_E.paramA, thdpA2);
    } else {
      apara2_P.getIndex(thdpA2);
      if (thdpA2.valid) launcher2->mProA.run(args2_P.paramA, thdpA2);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp2{core_idx};
    if (_cb->getCoreType(tidx) == _cb->E_CORE) {
      para2_E.getIndex(thdp2);
      if (thdp2.valid) launcher2->run(args2_E, thdp2);
    } else {
      para2_P.getIndex(thdp2);
      if (thdp2.valid) launcher2->run(args2_P, thdp2);
    }
  });
}

template <class Parallel_T, class Launch_T1, class Launch_T2>
void GemmRun_ffn(Launch_T1* launcher1, Launch_T2* launcher2, const typename Launch_T1::Param& args1,
                 const typename Launch_T2::Param& args2, parallel::IThreading* th) {
  GetCPU();
  Parallel_T para1({th->num_threads(), args1.problem, _cb->getL2Cache(), _cb->getL1Cache()});
  Parallel_T para2({th->num_threads(), args2.problem, _cb->getL2Cache(), _cb->getL1Cache()});
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para1.print();
    para2.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename Parallel_T::ThreadProblem thdp1{tidx};
    para1.getIndex(thdp1);
    if (thdp1.valid) {
      launcher1->run(args1, thdp1);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp2{tidx};
    para2.getIndex(thdp2);
    if (thdp2.valid) {
      launcher2->run(args2, thdp2);
    }
  });
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T, template <class, JBLAS_ISA> class Act_T,
          template <JBLAS_ISA> class Epi_T1, template <JBLAS_ISA> class Epi_T2>
void JblasGemmCompF32(float* activation, jblas::storage::gemm::IWeightBase* w1ptr,
                      jblas::storage::gemm::IWeightBase* w2ptr, float* tmp, float* output, int seq, int fin, int fmid,
                      int fout, void* workspace, jblas::parallel::IThreading* th,
                      typename Epi_T1<GemmCore_T::ISA>::Param epi_prama1,
                      typename Epi_T2<GemmCore_T::ISA>::Param epi_prama2) {
  if (seq <= 16) {
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher_epi = jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T,
                                                              jblas::epilogue::gemm::CompFp32BlockEpilogue, Epi_T1>;
    using Launcher = jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T,
                                                          jblas::epilogue::gemm::CompFp32BlockEpilogue, Epi_T2>;
    static Launcher_epi kernel_epi;
    static Launcher kernel;
    auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
    auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);

    auto reduceA1 = kernel_epi.mProA.createStorage(seq, fin, w1ptr_->mBlockSize);
    auto reordA1 = kernel_epi.mProA.createReorderStorage(seq, fin, w1ptr_->mBlockSize);
    utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
    auto WS = reinterpret_cast<int8_t*>(workspace);
    if (w1ptr_->IsAsym()) {
      reduceA1.assign(WS);
      WS += reduceA1.mSize;
    }
    if (w1ptr_->ShfIndice()) {
      reordA1.assign(WS);
    }
    typename Launcher_epi::BEpiParam blkargs1{
        w1ptr_->template SPtr<int8_t>(), w1ptr_->SDtype(), w1ptr_->CStep(), w1ptr_->template ZPtr<int8_t>(),
        reduceA1.template RPtr<float>(), reduceA1.lda};
    typename Launcher_epi::Param args1{
        gp1, {activation, fin, &reduceA1, w1ptr_->ShfIndice(), &reordA1}, {w1ptr_}, blkargs1, {epi_prama1}};

    auto reduceA2 = kernel.mProA.createStorage(seq, fmid, w2ptr_->mBlockSize);
    auto reordA2 = kernel_epi.mProA.createReorderStorage(seq, fin, w2ptr_->mBlockSize);
    utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
    WS = reinterpret_cast<int8_t*>(workspace);
    if (w2ptr_->IsAsym()) {
      reduceA2.assign(reinterpret_cast<int8_t*>(WS));
      WS += reduceA1.mSize;
    }
    if (w2ptr_->ShfIndice()) {
      reordA2.assign(WS);
    }
    typename Launcher::BEpiParam blkargs2{
        w2ptr_->template SPtr<int8_t>(), w2ptr_->SDtype(), w2ptr_->CStep(), w2ptr_->template ZPtr<int8_t>(),
        reduceA2.template RPtr<float>(), reduceA2.lda};
    typename Launcher::Param args2{
        gp2, {tmp, fmid, &reduceA2, w2ptr_->ShfIndice(), &reordA2}, {w2ptr_}, blkargs2, epi_prama2};

    if (w1ptr_->IsAsym() || w1ptr_->ShfIndice()) {
      GemmRunWithA_ffn<Parallel>(&kernel_epi, &kernel, args1, args2, th);
    } else {
      GemmRun_ffn<Parallel>(&kernel_epi, &kernel, args1, args2, th);
    }
  } else {
    using Parallel = jblas::parallel::gemm::SchedulerBase<GemmCore_T>;
    using Launcher_epi = jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T, Epi_T1>;
    using Launcher = jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T, Epi_T2>;
    auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
    auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
    utils::GemmProblem gp1(1, seq, fmid, fin);
    utils::GemmProblem gp2(1, seq, fout, fmid);
    static Launcher_epi kernel_epi;
    static Launcher kernel;
    auto reordA1 = kernel_epi.mProA.createReorderStorage(seq, fin, w1ptr_->mBlockSize);
    auto reordA2 = kernel_epi.mProA.createReorderStorage(seq, fin, w2ptr_->mBlockSize);
    typename Launcher_epi::Param args1{
        gp1, {activation, fin, nullptr, w1ptr_->ShfIndice(), &reordA1}, {w1ptr_}, epi_prama1};
    typename Launcher::Param args2{gp2, {tmp, fmid, nullptr, w2ptr_->ShfIndice(), &reordA2}, {w2ptr_}, epi_prama2};
    auto WS = reinterpret_cast<int8_t*>(workspace);
    if (w1ptr_->ShfIndice()) {
      reordA1.assign(WS);
      reordA2.assign(WS);
      GemmRunWithA_ffn<Parallel>(&kernel_epi, &kernel, args1, args2, th);
    } else {
      GemmRun_ffn<Parallel>(&kernel_epi, &kernel, args1, args2, th);
    }
  }
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T, template <JBLAS_ISA> class Epi_T1,
          template <JBLAS_ISA> class Epi_T2>
void JblasGemmCompInt8(float* activation, jblas::storage::gemm::IWeightBase* w1ptr,
                       jblas::storage::gemm::IWeightBase* w2ptr, float* tmp, float* output, int seq, int fin, int fmid,
                       int fout, void* workspace, jblas::parallel::IThreading* th,
                       typename Epi_T1<GemmCore_T::ISA>::Param epi_prama1,
                       typename Epi_T2<GemmCore_T::ISA>::Param epi_prama2) {
  using Parallel = jblas::parallel::gemm::SchedulerKBlockS<GemmCore_T>;
  using Launcher_epi = jblas::wrapper::gemm::LauncherIntKBlock<
      GemmCore_T::ISA, GemmCore_T, jblas::prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T1>;
  using Launcher = jblas::wrapper::gemm::LauncherIntKBlock<
      GemmCore_T::ISA, GemmCore_T, jblas::prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T2>;
  auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
  auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
  static Launcher_epi kernel_epi;
  static Launcher kernel;
  auto WS = reinterpret_cast<int8_t*>(workspace);
  GetCPU();
  if (_cb->isHybrid()) {
    int offset = seq - int(seq / (1 + _cb->getPE()));
    utils::GemmProblem gp1_P(1, offset, fmid, fin, w1ptr_->mBlockSize);
    utils::GemmProblem gp2_P(1, offset, fout, fmid, w2ptr_->mBlockSize);
    utils::GemmProblem gp1_E(1, seq - offset, fmid, fin, w1ptr_->mBlockSize);
    utils::GemmProblem gp2_E(1, seq - offset, fout, fmid, w2ptr_->mBlockSize);
    auto quanA1_P = kernel_epi.mProA.createStorage(offset, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
    quanA1_P.assign(WS);
    auto quanA2_P = kernel.mProA.createStorage(offset, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
    quanA2_P.assign(WS);
    auto quanA1_E = kernel_epi.mProA.createStorage(seq - offset, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
    quanA1_E.assign(WS + quanA1_P.mSize);
    auto quanA2_E = kernel.mProA.createStorage(seq - offset, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
    quanA2_E.assign(WS + quanA2_P.mSize);
    assert(w1ptr_->ShfIndice() == nullptr);
    assert(w2ptr_->ShfIndice() == nullptr);
    typename Launcher_epi::Param args1_P{gp1_P, {activation, fin, &quanA1_P}, {w1ptr_}, epi_prama1};
    typename Launcher::Param args2_P{gp2_P, {tmp, fmid, &quanA2_P}, {w2ptr_}, epi_prama2};
    typename Launcher_epi::Param args1_E{
        gp1_E, {activation + offset * fin, fin, &quanA1_E}, {w1ptr_}, epi_prama1.offset(offset)};
    typename Launcher::Param args2_E{
        gp2_E, {tmp + offset * fmid, fmid, &quanA2_E}, {w2ptr_}, epi_prama2.offset(offset)};
    GemmRunWithA_ffn<Parallel>(&kernel_epi, &kernel, args1_P, args1_E, args2_P, args2_E, th);
  } else {
    utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
    utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
    auto quanA1 = kernel_epi.mProA.createStorage(seq, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
    quanA1.assign(WS);
    WS += quanA1.mSize;
    auto reordA1 = kernel_epi.mProA.createReorderStorage(seq, fin, w1ptr_->mBlockSize);
    if (w1ptr_->ShfIndice()) {
      reordA1.assign(WS);
    }
    WS = reinterpret_cast<int8_t*>(workspace);
    auto quanA2 = kernel.mProA.createStorage(seq, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
    quanA2.assign(WS);
    WS += quanA2.mSize;
    auto reordA2 = kernel.mProA.createReorderStorage(seq, fmid, w2ptr_->mBlockSize);
    if (w2ptr_->ShfIndice()) {
      reordA2.assign(WS);
    }
    typename Launcher_epi::Param args1{
        gp1, {activation, fin, &quanA1, w1ptr_->ShfIndice(), &reordA1}, {w1ptr_}, epi_prama1};
    typename Launcher::Param args2{gp2, {tmp, fmid, &quanA2, w2ptr_->ShfIndice(), &reordA2}, {w2ptr_}, epi_prama2};
    GemmRunWithA_ffn<Parallel>(&kernel_epi, &kernel, args1, args2, th);
  }
}

bool jblas_fusion_ffn_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  GetCPUDevice();
  auto w1tmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w1ptr);
  auto w2tmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w2ptr);
  bool support = false;
  if (w1tmp != nullptr && w2tmp != nullptr) {
    auto sameKernel = samePackedWeight(w1tmp, w2tmp);
    if (sameKernel) {
      if (w1tmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
        constexpr size_t EleNum = sizeof(AllKBlockCores) / sizeof(AllKBlockCores[0]);
        support = contains(w1tmp->mCoreId, AllKBlockCores, EleNum);
        support &= hasISA(AllKBlockCores, EleNum);
      } else if (w1tmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
        constexpr size_t EleNum = sizeof(FloatCores) / sizeof(FloatCores[0]);
        support = contains(w1tmp->mCoreId, FloatCores, EleNum);
        support &= hasISA(FloatCores, EleNum);
      } else if (w1tmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF8) {
        constexpr size_t EleNum = sizeof(FloatCores) / sizeof(FloatCores[0]);
        support = contains(w1tmp->mCoreId, FloatCores, EleNum);
        support &= hasISA(FloatCores, EleNum);
      }
    }
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  return support;
}

template <template <JBLAS_ISA> class epilogue1, template <JBLAS_ISA> class epilogue2, typename Epi_args1,
          typename Epi_args2>
void jblas_fusion_ffn_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp, float* output, int seq,
                                     int fin, int fmid, int fout, void* workspace, Epi_args1 epi_args1,
                                     Epi_args2 epi_args2) {
  GetCPUDevice();
  auto pth = ne_jblas::ne_threading::get();
  auto ptr1 = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w1ptr);
  auto ptr2 = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w2ptr);
  auto _workspace = reinterpret_cast<int8_t*>(workspace);
  if (ptr1) {
    auto coretype = ptr1->mCoreId;
    auto NTile = jblas::gemm::CoreAttr::get_mask_val(ptr1->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK,
                                                     jblas::gemm::CoreAttr::NTILE_SHIFT);
    auto PackRow = jblas::gemm::CoreAttr::get_packrow(ptr1->mCoreId);
    auto CType = jblas::gemm::CoreAttr::get_comp(ptr1->mCoreId);
    auto btype = static_cast<jblas::gemm::CompType>(jblas::gemm::CompTypeHelper::get_B(CType));
    if (ptr1->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto bptr = reinterpret_cast<jblas::storage::gemm::IWeightKBlockBase*>(ptr1);
      auto BlkSize = bptr->mBlockSize;
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          JblasGemmCompF32<tAVX512F, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          JblasGemmCompF32<tAVX2, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          if (seq <= tAVX512_BF16::MTILE) {
            static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
            JblasGemmCompF32<tAVX512_BF16, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          } else {
            JblasGemmCompF32<tAMX_BF16, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          }
        }
      }
      if (btype == jblas::gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_SS_KBlock::NTILE && _cd->AMX_INT8() && BlkSize % tAMX_INT8_SS_KBlock::KTILE == 0) {
          if (seq <= tAVX512_VNNI_KBlock::MTILE) {
            static_assert(tAVX512_VNNI_KBlock::NTILE == tAMX_INT8_SS_KBlock::NTILE);
            JblasGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          } else {
            JblasGemmCompInt8<tAMX_INT8_SS_KBlock, tWeiNInt, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          }

        } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI() &&
                   BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          JblasGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          JblasGemmCompInt8<tAVX_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        }
      }
    }
    if (ptr1->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
      auto bptr = reinterpret_cast<jblas::storage::gemm::IWeightKBlockBase*>(ptr1);
      auto BlkSize = bptr->mBlockSize;
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          JblasGemmCompF32<tAVX512F, tWeiF4, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          JblasGemmCompF32<tAVX2, tWeiF4, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          if (seq <= tAVX512_BF16::MTILE) {
            static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
            JblasGemmCompF32<tAVX512_BF16, tWeiF4, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          } else {
            JblasGemmCompF32<tAMX_BF16, tWeiF4, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          }
        }
      }
    }
    if (ptr1->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF8) {
      auto bptr = reinterpret_cast<jblas::storage::gemm::IWeightKBlockBase*>(ptr1);
      auto BlkSize = bptr->mBlockSize;
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          JblasGemmCompF32<tAVX512F, tWeiF8, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          JblasGemmCompF32<tAVX2, tWeiF8, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          if (seq <= tAVX512_BF16::MTILE) {
            static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
            JblasGemmCompF32<tAVX512_BF16, tWeiF8, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          } else {
            JblasGemmCompF32<tAMX_BF16, tWeiF8, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          }
        }
      }
    }
    delete ptr1;
    delete ptr2;
  } else {
    printf("Wrong Input\n");
    assert(0);
  }
}
}  // namespace ffn_2w

namespace ffn_3w {

template <class Parallel_T, class Launch_T1, class Launch_T2, class Launch_T3>
void GemmRunWithA_ffn(Launch_T1* launcher1, Launch_T2* launcher2, Launch_T3* launcher3,
                      const typename Launch_T1::Param& args1, const typename Launch_T2::Param& args2,
                      const typename Launch_T3::Param& args3, parallel::IThreading* th) {
  GetCPU();
  Parallel_T para1({th->num_threads(), args1.problem, _cb->getL2Cache(), _cb->getL1Cache()});
  Parallel_T para3({th->num_threads(), args3.problem, _cb->getL2Cache(), _cb->getL1Cache()});
  using AParall1 = typename Launch_T1::PrologueA::Parallel;
  using AParall3 = typename Launch_T3::PrologueA::Parallel;
  auto apara1 = launcher1->mProA.createParallel(th->num_threads(), args1.problem);
  auto apara3 = launcher3->mProA.createParallel(th->num_threads(), args3.problem);
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para1.print();
    para3.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename AParall1::ThreadProblem thdpA1{tidx};
    apara1.getIndex(thdpA1);
    if (thdpA1.valid) {
      launcher1->mProA.run(args1.paramA, thdpA1);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp1{tidx};
    para1.getIndex(thdp1);
    if (thdp1.valid) {
      launcher1->run(args1, thdp1);
      launcher2->run(args2, thdp1);
    }
    th->sync();
    typename AParall3::ThreadProblem thdpA3{tidx};
    apara3.getIndex(thdpA3);
    if (thdpA3.valid) {
      launcher3->mProA.run(args3.paramA, thdpA3);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp3{tidx};
    para3.getIndex(thdp3);
    if (thdp3.valid) {
      launcher3->run(args3, thdp3);
    }
  });
}

template <class Parallel_T, class Launch_T1, class Launch_T2, class Launch_T3>
void GemmRunWithA_ffn(Launch_T1* launcher1, Launch_T2* launcher2, Launch_T3* launcher3,
                      const typename Launch_T1::Param& args1_P, const typename Launch_T1::Param& args1_E,
                      const typename Launch_T2::Param& args2_P, const typename Launch_T2::Param& args2_E,
                      const typename Launch_T3::Param& args3_P, const typename Launch_T3::Param& args3_E,
                      parallel::IThreading* th) {
  GetCPU();
  Parallel_T para1_P(
      {th->num_threads() - _cb->getECoreNum(), args1_P.problem, _cb->getL2Cache_P(), _cb->getL1Cache_P()});
  Parallel_T para3_P(
      {th->num_threads() - _cb->getECoreNum(), args3_P.problem, _cb->getL2Cache_P(), _cb->getL1Cache_P()});
  Parallel_T para1_E({_cb->getECoreNum(), args1_E.problem, _cb->getL2Cache_E(), _cb->getL1Cache_E()});
  Parallel_T para3_E({_cb->getECoreNum(), args3_E.problem, _cb->getL2Cache_E(), _cb->getL1Cache_E()});
  using AParall1 = typename Launch_T1::PrologueA::Parallel;
  using AParall3 = typename Launch_T3::PrologueA::Parallel;
  auto apara1_P = launcher1->mProA.createParallel(th->num_threads() - _cb->getECoreNum(), args1_P.problem);
  auto apara3_P = launcher3->mProA.createParallel(th->num_threads() - _cb->getECoreNum(), args3_P.problem);
  auto apara1_E = launcher1->mProA.createParallel(_cb->getECoreNum(), args1_E.problem);
  auto apara3_E = launcher3->mProA.createParallel(_cb->getECoreNum(), args3_E.problem);
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para1_P.print();
    para3_E.print();
    para1_P.print();
    para3_E.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    _cb->core_bond(tidx);
    int core_idx = _cb->getCoreidx(tidx);
    typename AParall1::ThreadProblem thdpA1{core_idx};
    if (_cb->getCoreType(tidx) == _cb->E_CORE) {
      apara1_E.getIndex(thdpA1);
      if (thdpA1.valid) launcher1->mProA.run(args1_E.paramA, thdpA1);
    } else {
      apara1_P.getIndex(thdpA1);
      if (thdpA1.valid) launcher1->mProA.run(args1_P.paramA, thdpA1);
    }
    th->sync();

    typename Parallel_T::ThreadProblem thdp1{core_idx};
    if (_cb->getCoreType(tidx) == _cb->E_CORE) {
      para1_E.getIndex(thdp1);
      if (thdp1.valid) {
        launcher1->run(args1_E, thdp1);
        launcher2->run(args2_E, thdp1);
      }
    } else {
      para1_P.getIndex(thdp1);
      if (thdp1.valid) {
        launcher1->run(args1_P, thdp1);
        launcher2->run(args2_P, thdp1);
      }
    }
    th->sync();

    typename AParall3::ThreadProblem thdpA3{core_idx};
    if (_cb->getCoreType(tidx) == _cb->E_CORE) {
      apara3_E.getIndex(thdpA3);
      if (thdpA3.valid) launcher3->mProA.run(args3_E.paramA, thdpA3);
    } else {
      apara3_P.getIndex(thdpA3);
      if (thdpA3.valid) launcher3->mProA.run(args3_P.paramA, thdpA3);
    }
    th->sync();

    typename Parallel_T::ThreadProblem thdp3{core_idx};
    if (_cb->getCoreType(tidx) == _cb->E_CORE) {
      para3_E.getIndex(thdp3);
      if (thdp3.valid) launcher3->run(args3_E, thdp3);
    } else {
      para3_P.getIndex(thdp3);
      if (thdp3.valid) launcher3->run(args3_P, thdp3);
    }
  });
}

template <class Parallel_T, class Launch_T1, class Launch_T2, class Launch_T3>
void GemmRun_ffn(Launch_T1* launcher1, Launch_T2* launcher2, Launch_T3* launcher3,
                 const typename Launch_T1::Param& args1, const typename Launch_T2::Param& args2,
                 const typename Launch_T3::Param& args3, parallel::IThreading* th) {
  GetCPU();
  Parallel_T para1({th->num_threads(), args1.problem, _cb->getL2Cache(), _cb->getL1Cache()});
  Parallel_T para3({th->num_threads(), args3.problem, _cb->getL2Cache(), _cb->getL1Cache()});
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para1.print();
    para3.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename Parallel_T::ThreadProblem thdp1{tidx};
    para1.getIndex(thdp1);
    if (thdp1.valid) {
      launcher1->run(args1, thdp1);
      launcher2->run(args2, thdp1);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp3{tidx};
    para3.getIndex(thdp3);
    if (thdp3.valid) {
      launcher3->run(args3, thdp3);
    }
  });
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T, template <class, JBLAS_ISA> class Act_T,
          template <JBLAS_ISA> class Epi_T1, template <JBLAS_ISA> class Epi_T2>
void JblasGemmCompF32(float* activation, jblas::storage::gemm::IWeightBase* w1ptr,
                      jblas::storage::gemm::IWeightBase* w2ptr, jblas::storage::gemm::IWeightBase* w3ptr, float* tmp1,
                      float* tmp2, float* output, int seq, int fin, int fmid, int fout, void* workspace,
                      jblas::parallel::IThreading* th, typename Epi_T1<GemmCore_T::ISA>::Param epi_prama1,
                      typename Epi_T2<GemmCore_T::ISA>::Param epi_prama2) {
  if (seq <= 16) {
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher_epi = jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T,
                                                              jblas::epilogue::gemm::CompFp32BlockEpilogue, Epi_T1>;
    using Launcher_mul =
        jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T,
                                             jblas::epilogue::gemm::CompFp32BlockEpilogue, custom::epilogue::MulFp32>;
    using Launcher = jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T,
                                                          jblas::epilogue::gemm::CompFp32BlockEpilogue, Epi_T2>;
    static Launcher_epi kernel_epi;
    static Launcher_mul kernel_mul;
    static Launcher kernel;
    auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
    auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
    auto w3ptr_ = reinterpret_cast<typename Launcher_mul::PrologueB::StorageWeight*>(w3ptr);
    assert(w1ptr_->ShfIndice() == nullptr);
    assert(w2ptr_->ShfIndice() == nullptr);
    assert(w3ptr_->ShfIndice() == nullptr);
    auto reduceA1 = kernel_epi.mProA.createStorage(seq, fin, w1ptr_->mBlockSize);
    utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
    if (w1ptr_->IsAsym()) {
      reduceA1.assign(reinterpret_cast<int8_t*>(workspace));
    }
    typename Launcher_epi::BEpiParam blkargs1{
        w1ptr_->template SPtr<int8_t>(), w1ptr_->SDtype(), w1ptr_->CStep(), w1ptr_->template ZPtr<int8_t>(),
        reduceA1.template RPtr<float>(), reduceA1.lda};
    typename Launcher_epi::Param args1{gp1, {activation, fin, &reduceA1}, {w1ptr_}, blkargs1, epi_prama1};

    auto reduceA2 = kernel.mProA.createStorage(seq, fmid, w2ptr_->mBlockSize);
    utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
    if (w2ptr_->IsAsym()) {
      reduceA2.assign(reinterpret_cast<int8_t*>(workspace));
    }
    typename Launcher::BEpiParam blkargs2{
        w2ptr_->template SPtr<int8_t>(), w2ptr_->SDtype(), w2ptr_->CStep(), w2ptr_->template ZPtr<int8_t>(),
        reduceA2.template RPtr<float>(), reduceA2.lda};
    typename Launcher::Param args2{gp2, {tmp2, fmid, &reduceA2}, {w2ptr_}, blkargs2, epi_prama2};

    utils::GemmProblem gp3(1, seq, fmid, fin, w3ptr_->mBlockSize);
    typename Launcher_mul::BEpiParam blkargs3{
        w3ptr_->template SPtr<int8_t>(), w3ptr_->SDtype(), w3ptr_->CStep(), w3ptr_->template ZPtr<int8_t>(),
        reduceA1.template RPtr<float>(), reduceA1.lda};
    typename Launcher_mul::Param args3{gp3, {activation, fin, &reduceA1}, {w3ptr_}, blkargs3, {tmp2, tmp1, fmid, fmid}};

    if (w1ptr_->IsAsym()) {
      GemmRunWithA_ffn<Parallel>(&kernel_epi, &kernel_mul, &kernel, args1, args3, args2, th);
    } else {
      GemmRun_ffn<Parallel>(&kernel_epi, &kernel_mul, &kernel, args1, args3, args2, th);
    }
  } else {
    using Parallel = jblas::parallel::gemm::SchedulerBase<GemmCore_T>;
    using Launcher_epi = jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T, Epi_T1>;
    using Launcher_mul =
        jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T, custom::epilogue::MulFp32>;
    using Launcher = jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T, Epi_T2>;
    auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
    auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
    auto w3ptr_ = reinterpret_cast<typename Launcher_mul::PrologueB::StorageWeight*>(w3ptr);
    assert(w1ptr_->ShfIndice() == nullptr);
    assert(w2ptr_->ShfIndice() == nullptr);
    assert(w3ptr_->ShfIndice() == nullptr);
    utils::GemmProblem gp1(1, seq, fmid, fin);
    utils::GemmProblem gp2(1, seq, fout, fmid);
    utils::GemmProblem gp3(1, seq, fmid, fin);
    static Launcher_epi kernel_epi;
    static Launcher_mul kernel_mul;
    static Launcher kernel;
    typename Launcher_epi::Param args1{gp1, {activation, fin, nullptr}, {w1ptr_}, epi_prama1};
    typename Launcher::Param args2{gp2, {tmp2, fmid, nullptr}, {w2ptr_}, epi_prama2};
    typename Launcher_mul::Param args3{gp3, {activation, fin, nullptr}, {w3ptr_}, {tmp2, tmp1, fmid, fmid}};
    GemmRun_ffn<Parallel>(&kernel_epi, &kernel_mul, &kernel, args1, args3, args2, th);
  }
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T, template <JBLAS_ISA> class Epi_T1,
          template <JBLAS_ISA> class Epi_T2>
void JblasGemmCompInt8(float* activation, jblas::storage::gemm::IWeightBase* w1ptr,
                       jblas::storage::gemm::IWeightBase* w2ptr, jblas::storage::gemm::IWeightBase* w3ptr, float* tmp1,
                       float* tmp2, float* output, int seq, int fin, int fmid, int fout, void* workspace,
                       jblas::parallel::IThreading* th, typename Epi_T1<GemmCore_T::ISA>::Param epi_prama1,
                       typename Epi_T2<GemmCore_T::ISA>::Param epi_prama2) {
  using Parallel = jblas::parallel::gemm::SchedulerKBlockS<GemmCore_T>;
  using Launcher_epi = jblas::wrapper::gemm::LauncherIntKBlock<
      GemmCore_T::ISA, GemmCore_T, jblas::prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T1>;
  using Launcher_mul =
      jblas::wrapper::gemm::LauncherIntKBlock<GemmCore_T::ISA, GemmCore_T,
                                              jblas::prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T,
                                              custom::epilogue::MulFp32>;
  using Launcher = jblas::wrapper::gemm::LauncherIntKBlock<
      GemmCore_T::ISA, GemmCore_T, jblas::prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T2>;
  auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
  auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
  auto w3ptr_ = reinterpret_cast<typename Launcher_mul::PrologueB::StorageWeight*>(w3ptr);
  static Launcher_epi kernel_epi;
  static Launcher_mul kernel_mul;
  static Launcher kernel;
  auto WS = reinterpret_cast<int8_t*>(workspace);
  GetCPU();
  if (_cb->isHybrid()) {
    int offset = seq - int(seq / (1 + _cb->getPE()));
    // utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
    // _cb->getP_problem(gp1);
    utils::GemmProblem gp1_P(1, offset, fmid, fin, w1ptr_->mBlockSize);
    utils::GemmProblem gp2_P(1, offset, fout, fmid, w2ptr_->mBlockSize);
    utils::GemmProblem gp3_P(1, offset, fmid, fin, w3ptr_->mBlockSize);
    utils::GemmProblem gp1_E(1, seq - offset, fmid, fin, w1ptr_->mBlockSize);
    utils::GemmProblem gp2_E(1, seq - offset, fout, fmid, w2ptr_->mBlockSize);
    utils::GemmProblem gp3_E(1, seq - offset, fmid, fin, w3ptr_->mBlockSize);
    auto quanA1_P = kernel_epi.mProA.createStorage(offset, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
    quanA1_P.assign(WS);
    auto quanA2_P = kernel.mProA.createStorage(offset, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
    quanA2_P.assign(WS);
    auto quanA1_E = kernel_epi.mProA.createStorage(seq - offset, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
    quanA1_E.assign(WS + quanA1_P.mSize);
    auto quanA2_E = kernel.mProA.createStorage(seq - offset, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
    quanA2_E.assign(WS + quanA2_P.mSize);
    assert(w1ptr_->ShfIndice() == nullptr);
    assert(w2ptr_->ShfIndice() == nullptr);
    assert(w3ptr_->ShfIndice() == nullptr);
    typename Launcher_epi::Param args1_P{gp1_P, {activation, fin, &quanA1_P}, {w1ptr_}, epi_prama1};
    typename Launcher::Param args2_P{gp2_P, {tmp2, fmid, &quanA2_P}, {w2ptr_}, epi_prama2};
    typename Launcher_epi::Param args3_P{gp3_P, {activation, fin, &quanA1_P}, {w3ptr_}, {tmp2, tmp1, fmid, fmid}};
    typename Launcher_epi::Param args1_E{
        gp1_E, {activation + offset * fin, fin, &quanA1_E}, {w1ptr_}, epi_prama1.offset(offset)};
    typename Launcher::Param args2_E{
        gp2_E, {tmp2 + offset * fmid, fmid, &quanA2_E}, {w2ptr_}, epi_prama2.offset(offset)};
    typename Launcher_mul::Param args3_E{gp3_E,
                                         {activation + offset * fin, fin, &quanA1_E},
                                         {w3ptr_},
                                         {tmp2 + fmid * offset, tmp1 + fmid * offset, fmid, fmid}};
    GemmRunWithA_ffn<Parallel>(&kernel_epi, &kernel_mul, &kernel, args1_P, args1_E, args3_P, args3_E, args2_P, args2_E,
                               th);
  } else {
    utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
    utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
    utils::GemmProblem gp3(1, seq, fmid, fin, w3ptr_->mBlockSize);
    auto quanA1 = kernel_epi.mProA.createStorage(seq, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
    quanA1.assign(WS);
    auto quanA2 = kernel.mProA.createStorage(seq, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
    quanA2.assign(WS);
    assert(w1ptr_->ShfIndice() == nullptr);
    assert(w2ptr_->ShfIndice() == nullptr);
    assert(w3ptr_->ShfIndice() == nullptr);
    typename Launcher_epi::Param args1{gp1, {activation, fin, &quanA1}, {w1ptr_}, epi_prama1};
    typename Launcher::Param args2{gp2, {tmp2, fmid, &quanA2}, {w2ptr_}, epi_prama2};
    typename Launcher_mul::Param args3{gp3, {activation, fin, &quanA1}, {w3ptr_}, {tmp2, tmp1, fmid, fmid}};
    GemmRunWithA_ffn<Parallel>(&kernel_epi, &kernel_mul, &kernel, args1, args3, args2, th);
  }
}

bool jblas_fusion_ffn_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid, int fout) {
  GetCPUDevice();
  auto w1tmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w1ptr);
  auto w2tmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w2ptr);
  auto w3tmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w3ptr);
  bool support = false;
  if (w1tmp != nullptr && w2tmp != nullptr && w3tmp != nullptr) {
    jblas::storage::gemm::IWeightBase* tmps[3] = {w1tmp, w2tmp, w3tmp};
    auto sameKernel = samePackedWeight(tmps, 3);
    if (sameKernel) {
      if (w1tmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
        auto w1ptr = reinterpret_cast<jblas::storage::gemm::StorageWeightKBlockNInteger*>(w1tmp);
        if (w1ptr->ShfIndice()) {
          return false;  // Do not support 3w ffn fusion for activation shuffle
        }
        constexpr size_t EleNum = sizeof(AllKBlockCores) / sizeof(AllKBlockCores[0]);
        support = contains(w1tmp->mCoreId, AllKBlockCores, EleNum);
        support &= hasISA(AllKBlockCores, EleNum);
      } else if (w1tmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
        constexpr size_t EleNum = sizeof(FloatCores) / sizeof(FloatCores[0]);
        support = contains(w1tmp->mCoreId, FloatCores, EleNum);
        support &= hasISA(FloatCores, EleNum);
      } else if (w1tmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF8) {
        constexpr size_t EleNum = sizeof(FloatCores) / sizeof(FloatCores[0]);
        support = contains(w1tmp->mCoreId, FloatCores, EleNum);
        support &= hasISA(FloatCores, EleNum);
      }
    }
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  safe_delete(w3tmp);
  return support;
}

template <template <JBLAS_ISA> class epilogue1, template <JBLAS_ISA> class epilogue2, typename Epi_args1,
          typename Epi_args2>
void jblas_fusion_ffn_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1, float* tmp2,
                                     float* output, int seq, int fin, int fmid, int fout, void* workspace,
                                     Epi_args1 epi_args1, Epi_args2 epi_args2) {
  GetCPUDevice();
  auto pth = ne_jblas::ne_threading::get();
  auto ptr1 = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w1ptr);
  auto ptr2 = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w2ptr);
  auto ptr3 = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w3ptr);
  auto _workspace = reinterpret_cast<int8_t*>(workspace);
  if (ptr1) {
    auto coretype = ptr1->mCoreId;
    auto NTile = jblas::gemm::CoreAttr::get_mask_val(ptr1->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK,
                                                     jblas::gemm::CoreAttr::NTILE_SHIFT);
    auto PackRow = jblas::gemm::CoreAttr::get_packrow(ptr1->mCoreId);
    auto CType = jblas::gemm::CoreAttr::get_comp(ptr1->mCoreId);
    auto btype = static_cast<jblas::gemm::CompType>(jblas::gemm::CompTypeHelper::get_B(CType));
    if (ptr1->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto bptr = reinterpret_cast<jblas::storage::gemm::IWeightKBlockBase*>(ptr1);
      auto BlkSize = bptr->mBlockSize;
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          JblasGemmCompF32<tAVX512F, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
              epi_args2);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          JblasGemmCompF32<tAVX2, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1,
                                                                                tmp2, output, seq, fin, fmid, fout,
                                                                                workspace, pth, epi_args1, epi_args2);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          if (seq <= tAVX512_BF16::MTILE) {
            static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
            JblasGemmCompF32<tAVX512_BF16, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          } else {
            JblasGemmCompF32<tAMX_BF16, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          }
        }
      }
      if (btype == jblas::gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_SS_KBlock::NTILE && _cd->AMX_INT8() && BlkSize % tAMX_INT8_SS_KBlock::KTILE == 0) {
          if (seq <= tAVX512_VNNI_KBlock::MTILE) {
            static_assert(tAVX512_VNNI_KBlock::NTILE == tAMX_INT8_SS_KBlock::NTILE);
            JblasGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          } else {
            JblasGemmCompInt8<tAMX_INT8_SS_KBlock, tWeiNInt, epilogue1, epilogue2>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          }

        } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI() &&
                   BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          JblasGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1,
                                                                                 tmp2, output, seq, fin, fmid, fout,
                                                                                 workspace, pth, epi_args1, epi_args2);
        } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          JblasGemmCompInt8<tAVX_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1, tmp2,
                                                                              output, seq, fin, fmid, fout, workspace,
                                                                              pth, epi_args1, epi_args2);
        }
      }
    }
    if (ptr1->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
      auto bptr = reinterpret_cast<jblas::storage::gemm::IWeightKBlockBase*>(ptr1);
      auto BlkSize = bptr->mBlockSize;
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          JblasGemmCompF32<tAVX512F, tWeiF4, tActKBaseF32, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1,
                                                                                 tmp2, output, seq, fin, fmid, fout,
                                                                                 workspace, pth, epi_args1, epi_args2);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          JblasGemmCompF32<tAVX2, tWeiF4, tActKBaseF32, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1, tmp2,
                                                                              output, seq, fin, fmid, fout, workspace,
                                                                              pth, epi_args1, epi_args2);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          if (seq <= tAVX512_BF16::MTILE) {
            static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
            JblasGemmCompF32<tAVX512_BF16, tWeiF4, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          } else {
            JblasGemmCompF32<tAMX_BF16, tWeiF4, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          }
        }
      }
    }
    if (ptr1->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF8) {
      auto bptr = reinterpret_cast<jblas::storage::gemm::IWeightKBlockBase*>(ptr1);
      auto BlkSize = bptr->mBlockSize;
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          JblasGemmCompF32<tAVX512F, tWeiF8, tActKBaseF32, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1,
                                                                                 tmp2, output, seq, fin, fmid, fout,
                                                                                 workspace, pth, epi_args1, epi_args2);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          JblasGemmCompF32<tAVX2, tWeiF8, tActKBaseF32, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1, tmp2,
                                                                              output, seq, fin, fmid, fout, workspace,
                                                                              pth, epi_args1, epi_args2);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          if (seq <= tAVX512_BF16::MTILE) {
            static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
            JblasGemmCompF32<tAVX512_BF16, tWeiF8, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          } else {
            JblasGemmCompF32<tAMX_BF16, tWeiF8, tActKBaseF32, epilogue1, epilogue2>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          }
        }
      }
    }
    delete ptr1;
    delete ptr2;
    delete ptr3;
  } else {
    printf("Wrong Input\n");
    assert(0);
  }
}
}  // namespace ffn_3w

bool jblas_fusion_FFN_SiLu_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid, int fout) {
  return ffn_3w::jblas_fusion_ffn_f32f32_support(w1ptr, w2ptr, w3ptr, seq, fin, fmid, fout);
}

void jblas_fusion_FFN_SiLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                          float* tmp2, float* output, int seq, int fin, int fmid, int fout,
                                          void* workspace) {
  float silu_alpha = -1.0f;
  jblas::epilogue::gemm::ParamAccumulatorWriteBack<float> epi_args1 = {tmp1, fmid, &silu_alpha};
  jblas::epilogue::gemm::ParamAccumulatorWriteBack<float> epi_args2 = {output, fout};
  ffn_3w::jblas_fusion_ffn_f32f32_forward<jblas::epilogue::gemm::AccumulatorWriteBackWithSwishFp32,
                                          jblas::epilogue::gemm::AccumulatorWriteBackFp32>(
      activation, w1ptr, w2ptr, w3ptr, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, epi_args1, epi_args2);
}

bool jblas_fusion_FFN_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  return ffn_2w::jblas_fusion_ffn_f32f32_support(w1ptr, w2ptr, seq, fin, fmid, fout);
}

void jblas_fusion_FFN_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp1, float* output,
                                          int seq, int fin, int fmid, int fout, void* workspace) {
  jblas::epilogue::gemm::ParamAccumulatorWriteBack<float> epi_args1 = {tmp1, fmid, nullptr};
  jblas::epilogue::gemm::ParamAccumulatorWriteBack<float> epi_args2 = {output, fout};
  ffn_2w::jblas_fusion_ffn_f32f32_forward<jblas::epilogue::gemm::AccumulatorWriteBackWithGeluFp32,
                                          jblas::epilogue::gemm::AccumulatorWriteBackFp32>(
      activation, w1ptr, w2ptr, tmp1, output, seq, fin, fmid, fout, workspace, epi_args1, epi_args2);
}

bool jblas_fusion_FFN_Add_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  return ffn_2w::jblas_fusion_ffn_f32f32_support(w1ptr, w2ptr, seq, fin, fmid, fout);
}

void jblas_fusion_FFN_Add_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* b1ptr, float* b2ptr,
                                              float* tmp1, float* output, int seq, int fin, int fmid, int fout,
                                              bool broadcast_bias, void* workspace) {
  custom::epilogue::ParamAdd_Gelu<float> epi_args1 = {tmp1, b1ptr, fmid, broadcast_bias ? 0 : fmid};
  custom::epilogue::ParamAdd<float> epi_args2 = {output, b2ptr, fout, broadcast_bias ? 0 : fout};
  ffn_2w::jblas_fusion_ffn_f32f32_forward<custom::epilogue::Add_GeluFp32, custom::epilogue::AddFp32>(
      activation, w1ptr, w2ptr, tmp1, output, seq, fin, fmid, fout, workspace, epi_args1, epi_args2);
}
