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

namespace ffn_silu {

template <class Parallel_T, class Launch_T1, class Launch_T2, class Launch_T3>
void GemmRunWithA_ffn(Launch_T1& launcher1, Launch_T2& launcher2, Launch_T3& launcher3,
                      const typename Launch_T1::Param& args1, const typename Launch_T2::Param& args2,
                      const typename Launch_T3::Param& args3, parallel::IThreading* th) {
  device::CpuBase cb;
  Parallel_T para1({th->num_threads(), args1.problem, cb.mL2Cache, cb.mL1Cache});
  Parallel_T para3({th->num_threads(), args3.problem, cb.mL2Cache, cb.mL1Cache});
  using AParall1 = typename Launch_T1::PrologueA::Parallel;
  using AParall3 = typename Launch_T3::PrologueA::Parallel;
  auto apara1 = launcher1.mProA.createParallel(th->num_threads(), args1.problem);
  auto apara3 = launcher3.mProA.createParallel(th->num_threads(), args3.problem);
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
      launcher1.mProA.run(args1.paramA, thdpA1);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp1{tidx};
    para1.getIndex(thdp1);
    if (thdp1.valid) {
      launcher1.run(args1, thdp1);
      launcher2.run(args2, thdp1);
    }
    th->sync();
    typename AParall3::ThreadProblem thdpA3{tidx};
    apara3.getIndex(thdpA3);
    if (thdpA3.valid) {
      launcher3.mProA.run(args3.paramA, thdpA3);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp3{tidx};
    para3.getIndex(thdp3);
    if (thdp3.valid) {
      launcher3.run(args3, thdp3);
    }
  });
}

template <class Parallel_T, class Launch_T1, class Launch_T2, class Launch_T3>
void GemmRun_ffn(Launch_T1& launcher1, Launch_T2& launcher2, Launch_T3& launcher3,
                 const typename Launch_T1::Param& args1, const typename Launch_T2::Param& args2,
                 const typename Launch_T3::Param& args3, parallel::IThreading* th) {
  device::CpuBase cb;
  Parallel_T para1({th->num_threads(), args1.problem, cb.mL2Cache, cb.mL1Cache});
  Parallel_T para3({th->num_threads(), args3.problem, cb.mL2Cache, cb.mL1Cache});
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
      launcher1.run(args1, thdp1);
      launcher2.run(args2, thdp1);
    }
    th->sync();
    typename Parallel_T::ThreadProblem thdp3{tidx};
    para3.getIndex(thdp3);
    if (thdp3.valid) {
      launcher3.run(args3, thdp3);
    }
  });
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T, template <class, JBLAS_ISA> class Act_T>
void JblasGemmCompF32(float* activation, jblas::storage::gemm::IWeightBase* w1ptr,
                      jblas::storage::gemm::IWeightBase* w2ptr, jblas::storage::gemm::IWeightBase* w3ptr, float* tmp1,
                      float* tmp2, float* output, int seq, int fin, int fmid, int fout, void* workspace,
                      jblas::parallel::IThreading* th) {
  if (seq <= 32) {
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher_silu =
        jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T,
                                             jblas::epilogue::gemm::CompFp32BlockEpilogue,
                                             jblas::epilogue::gemm::AccumulatorWriteBackWithSwishFp32>;
    using Launcher_mul =
        jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T,
                                             jblas::epilogue::gemm::CompFp32BlockEpilogue, custom::epilogue::MulFp32>;
    using Launcher = jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, Act_T, Wei_T,
                                                          jblas::epilogue::gemm::CompFp32BlockEpilogue,
                                                          jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
    static Launcher_silu kernel_silu;
    static Launcher_mul kernel_mul;
    static Launcher kernel;
    auto w1ptr_ = reinterpret_cast<typename Launcher_silu::PrologueB::StorageWeight*>(w1ptr);
    auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
    auto w3ptr_ = reinterpret_cast<typename Launcher_mul::PrologueB::StorageWeight*>(w3ptr);

    auto reduceA1 = kernel_silu.mProA.createStorage(seq, fin, w1ptr_->mBlockSize);
    utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
    if (w1ptr_->IsAsym()) {
      reduceA1.assign(reinterpret_cast<int8_t*>(workspace));
    }
    typename Launcher_silu::BEpiParam blkargs1{
        w1ptr_->template SPtr<int8_t>(), w1ptr_->SDtype(), w1ptr_->CStep(), w1ptr_->template ZPtr<int8_t>(),
        reduceA1.template RPtr<float>(), reduceA1.lda};
    float silu_alpha = -1.0;
    typename Launcher_silu::Param args1{
        gp1, {activation, fin, &reduceA1}, {w1ptr_}, blkargs1, {tmp1, fmid, &silu_alpha}};

    auto reduceA2 = kernel.mProA.createStorage(seq, fmid, w2ptr_->mBlockSize);
    utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
    if (w2ptr_->IsAsym()) {
      reduceA2.assign(reinterpret_cast<int8_t*>(workspace));
    }
    typename Launcher::BEpiParam blkargs2{
        w2ptr_->template SPtr<int8_t>(), w2ptr_->SDtype(), w2ptr_->CStep(), w2ptr_->template ZPtr<int8_t>(),
        reduceA2.template RPtr<float>(), reduceA2.lda};
    typename Launcher::Param args2{gp2, {tmp2, fmid, &reduceA2}, {w2ptr_}, blkargs2, {output, fout}};

    utils::GemmProblem gp3(1, seq, fmid, fin, w3ptr_->mBlockSize);
    typename Launcher_mul::BEpiParam blkargs3{
        w3ptr_->template SPtr<int8_t>(), w3ptr_->SDtype(), w3ptr_->CStep(), w3ptr_->template ZPtr<int8_t>(),
        reduceA1.template RPtr<float>(), reduceA1.lda};
    typename Launcher_mul::Param args3{gp3, {activation, fin, &reduceA1}, {w3ptr_}, blkargs3, {tmp2, tmp1, fmid, fmid}};

    if (w1ptr_->IsAsym()) {
      GemmRunWithA_ffn<Parallel>(kernel_silu, kernel_mul, kernel, args1, args3, args2, th);
    } else {
      GemmRun_ffn<Parallel>(kernel_silu, kernel_mul, kernel, args1, args3, args2, th);
    }
  } else {
    using Parallel = jblas::parallel::gemm::SchedulerBase<GemmCore_T>;
    using Launcher_silu =
        jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, Act_T, jblas::prologue_b::gemm::WeightKBlockS4,
                                           jblas::epilogue::gemm::AccumulatorWriteBackWithSwishFp32>;
    using Launcher_mul =
        jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, Act_T, jblas::prologue_b::gemm::WeightKBlockS4,
                                           custom::epilogue::MulFp32>;
    using Launcher =
        jblas::wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, Act_T, jblas::prologue_b::gemm::WeightKBlockS4,
                                           jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
    auto w1ptr_ = reinterpret_cast<typename Launcher_silu::PrologueB::StorageWeight*>(w1ptr);
    auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
    auto w3ptr_ = reinterpret_cast<typename Launcher_mul::PrologueB::StorageWeight*>(w3ptr);
    utils::GemmProblem gp1(1, seq, fmid, fin);
    utils::GemmProblem gp2(1, seq, fout, fmid);
    utils::GemmProblem gp3(1, seq, fmid, fin);
    static Launcher_silu kernel_silu;
    static Launcher_mul kernel_mul;
    static Launcher kernel;
    float silu_alpha = -1.0;
    typename Launcher_silu::Param args1{gp1, {activation, fin}, {w1ptr_}, {tmp1, fmid, &silu_alpha}};
    typename Launcher::Param args2{gp2, {tmp2, fmid}, {w2ptr_}, {output, fout}};
    typename Launcher_mul::Param args3{gp3, {activation, fin}, {w3ptr_}, {tmp2, tmp1, fmid, fmid}};
    GemmRun_ffn<Parallel>(kernel_silu, kernel_mul, kernel, args1, args3, args2, th);
  }
}

template <class GemmCore_T, template <class, JBLAS_ISA> class Wei_T>
void JblasGemmCompInt8(float* activation, jblas::storage::gemm::IWeightBase* w1ptr,
                       jblas::storage::gemm::IWeightBase* w2ptr, jblas::storage::gemm::IWeightBase* w3ptr, float* tmp1,
                       float* tmp2, float* output, int seq, int fin, int fmid, int fout, void* workspace,
                       jblas::parallel::IThreading* th) {
  using Parallel = jblas::parallel::gemm::SchedulerKBlockS<GemmCore_T>;
  using Launcher_silu =
      jblas::wrapper::gemm::LauncherIntKBlock<GemmCore_T::ISA, GemmCore_T,
                                              jblas::prologue_a::gemm::ActivationF32KBlockQuantize, Wei_T,
                                              jblas::epilogue::gemm::AccumulatorWriteBackWithSwishFp32>;
  using Launcher_mul = jblas::wrapper::gemm::LauncherIntKBlock<GemmCore_T::ISA, GemmCore_T,
                                                               jblas::prologue_a::gemm::ActivationF32KBlockQuantize,
                                                               Wei_T, custom::epilogue::MulFp32>;
  using Launcher = jblas::wrapper::gemm::LauncherIntKBlock<GemmCore_T::ISA, GemmCore_T,
                                                           jblas::prologue_a::gemm::ActivationF32KBlockQuantize, Wei_T,
                                                           jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
  auto w1ptr_ = reinterpret_cast<typename Launcher_silu::PrologueB::StorageWeight*>(w1ptr);
  auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
  auto w3ptr_ = reinterpret_cast<typename Launcher_mul::PrologueB::StorageWeight*>(w3ptr);
  utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
  utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
  utils::GemmProblem gp3(1, seq, fmid, fin, w3ptr_->mBlockSize);
  static Launcher_silu kernel_silu;
  static Launcher_mul kernel_mul;
  static Launcher kernel;
  auto quanA1 = kernel_silu.mProA.createStorage(seq, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
  quanA1.assign(reinterpret_cast<int8_t*>(workspace));
  auto quanA2 = kernel.mProA.createStorage(seq, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
  quanA2.assign(reinterpret_cast<int8_t*>(workspace));
  float silu_alpha = -1.0;
  typename Launcher_silu::Param args1{gp1, {activation, fin, &quanA1}, {w1ptr_}, {tmp1, fmid, &silu_alpha}};
  typename Launcher::Param args2{gp2, {tmp2, fmid, &quanA2}, {w2ptr_}, {output, fout}};
  typename Launcher_mul::Param args3{gp3, {activation, fin, &quanA1}, {w3ptr_}, {tmp2, tmp1, fmid, fmid}};
  GemmRunWithA_ffn<Parallel>(kernel_silu, kernel_mul, kernel, args1, args3, args2, th);
}
}  // namespace ffn_silu

bool jblas_fusion_FFN_SiLu_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid, int fout) {
  GetCPUDevice();
  auto w1tmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w1ptr);
  auto w2tmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w2ptr);
  auto w3tmp = jblas::storage::gemm::PackedWeightParser::deserialBuffer(w3ptr);
  bool support = false;
  if (w1tmp != nullptr && w2tmp != nullptr && w3tmp != nullptr) {
    jblas::storage::gemm::IWeightBase* tmps[3] = {w1tmp, w2tmp, w3tmp};
    auto sameKernel = samePackedWeight(tmps, 3);
    if (w1tmp) {
      if (w1tmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockNInteger) {
        constexpr size_t EleNum = sizeof(AllKBlockCores) / sizeof(AllKBlockCores[0]);
        support = contains(w1tmp->mCoreId, AllKBlockCores, EleNum);
        support &= hasISA(AllKBlockCores, EleNum);
      } else if (w1tmp->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
        constexpr size_t EleNum = sizeof(AllKBlockCores) / sizeof(AllKBlockCores[0]);
        support = contains(w1tmp->mCoreId, AllKBlockCores, EleNum);
        support &= hasISA(AllKBlockCores, EleNum);
      }
    }
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  safe_delete(w3tmp);
  return support;
}

void jblas_fusion_FFN_SiLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                          float* tmp2, float* output, int seq, int fin, int fmid, int fout,
                                          void* workspace) {
  GetCPUDevice();
  auto pth = get_threading();
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
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          ffn_silu::JblasGemmCompF32<tAVX512F, tWeiNInt, tActKBaseF32>(activation, ptr1, ptr2, ptr3, tmp1, tmp2, output,
                                                                       seq, fin, fmid, fout, workspace, pth);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          ffn_silu::JblasGemmCompF32<tAVX2, tWeiNInt, tActKBaseF32>(activation, ptr1, ptr2, ptr3, tmp1, tmp2, output,
                                                                    seq, fin, fmid, fout, workspace, pth);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          if (seq <= tAVX512_BF16::MTILE) {
            static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
            ffn_silu::JblasGemmCompF32<tAVX512_BF16, tWeiNInt, tActKBaseF32>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth);
          } else {
            ffn_silu::JblasGemmCompF32<tAMX_BF16, tWeiNInt, tActKBaseF32>(activation, ptr1, ptr2, ptr3, tmp1, tmp2,
                                                                          output, seq, fin, fmid, fout, workspace, pth);
          }
        }
      }
      if (btype == jblas::gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_SS_KBlock::NTILE && _cd->AMX_INT8()) {
          if (seq <= tAVX512_VNNI_KBlock::MTILE) {
            static_assert(tAVX512_VNNI_KBlock::NTILE == tAMX_INT8_SS_KBlock::NTILE);
            ffn_silu::JblasGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt>(activation, ptr1, ptr2, ptr3, tmp1, tmp2, output,
                                                                       seq, fin, fmid, fout, workspace, pth);
          } else {
            ffn_silu::JblasGemmCompInt8<tAMX_INT8_SS_KBlock, tWeiNInt>(activation, ptr1, ptr2, ptr3, tmp1, tmp2, output,
                                                                       seq, fin, fmid, fout, workspace, pth);
          }

        } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI()) {
          ffn_silu::JblasGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt>(activation, ptr1, ptr2, ptr3, tmp1, tmp2, output,
                                                                     seq, fin, fmid, fout, workspace, pth);
        } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI()) {
          ffn_silu::JblasGemmCompInt8<tAVX_VNNI_KBlock, tWeiNInt>(activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq,
                                                                  fin, fmid, fout, workspace, pth);
        }
      }
    }
    if (ptr1->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockF4) {
      if (btype == jblas::gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          ffn_silu::JblasGemmCompF32<tAVX512F, tWeiF4, tActKBaseF32>(activation, ptr1, ptr2, ptr3, tmp1, tmp2, output,
                                                                     seq, fin, fmid, fout, workspace, pth);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          ffn_silu::JblasGemmCompF32<tAVX2, tWeiF4, tActKBaseF32>(activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq,
                                                                  fin, fmid, fout, workspace, pth);
        }
      }
      if (btype == jblas::gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          if (seq <= tAVX512_BF16::MTILE) {
            static_assert(tAVX512_BF16::NTILE == tAMX_BF16::NTILE);
            ffn_silu::JblasGemmCompF32<tAVX512_BF16, tWeiNInt, tActKBaseF32>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth);
          } else {
            ffn_silu::JblasGemmCompF32<tAMX_BF16, tWeiF4, tActKBaseF32>(activation, ptr1, ptr2, ptr3, tmp1, tmp2,
                                                                        output, seq, fin, fmid, fout, workspace, pth);
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

#if 0
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
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1ptr,
                            w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,    tmp2, ldtmp2, NULL});

    } else if (w1ptr->mBlockSize % 8 == 0) {
      if (_cd->AVX512_VNNI()) {
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
          auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize);
          quanA1.assign((int8_t*)workspace);
          auto offset = workspace == NULL ? 0 : quanA1.mSize;
          auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize);
          quanA2.assign((int8_t*)workspace + offset);
          ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1ptr,
                                w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,    tmp2, ldtmp2, NULL});
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
          auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize);
          quanA1.assign((int8_t*)workspace);
          auto offset = workspace == NULL ? 0 : quanA1.mSize;
          auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize);
          quanA2.assign((int8_t*)workspace + offset);
          ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1ptr,
                                w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,    tmp2, ldtmp2, NULL});
        }
      } else if (_cd->AVX_VNNI()) {
        using GemmKernel = custom::wrapper::kblock::avx_vnni::GemmSKernelDynamicS4KBlock;
        using SiluGemmKernel = custom::wrapper::kblock::avx_vnni::SiluGemmSKernelDynamicS4KBlock;
        using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
        using DQuantParam = GemmKernel::PrologueA::QParam;
        static FusedInter finter;
        int lda = fin;
        int ldtmp1 = fmid;
        int ldtmp2 = fmid;
        int ldo = fout;
        auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize);
        quanA1.assign((int8_t*)workspace);
        auto offset = workspace == NULL ? 0 : quanA1.mSize;
        auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize);
        quanA2.assign((int8_t*)workspace + offset);
        ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1ptr,
                              w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,    tmp2, ldtmp2, NULL});
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
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1ptr,
                            w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,    tmp2, ldtmp2, NULL});
    } else if (w1ptr->mBlockSize % 4 == 0) {
      if (_cd->AVX512_VNNI()) {
        using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS8KBlock;
        using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmSKernelDynamicS8KBlock;
        using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
        using DQuantParam = GemmKernel::PrologueA::QParam;
        static FusedInter finter;
        int lda = fin;
        int ldtmp1 = fmid;
        int ldtmp2 = fmid;
        int ldo = fout;
        auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize);
        quanA1.assign((int8_t*)workspace);
        auto offset = workspace == NULL ? 0 : quanA1.mSize;
        auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize);
        quanA2.assign((int8_t*)workspace + offset);
        ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1ptr,
                              w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,    tmp2, ldtmp2, NULL});
      } else if (_cd->AVX_VNNI()) {
        using GemmKernel = custom::wrapper::kblock::avx_vnni::GemmSKernelDynamicS8KBlock;
        using SiluGemmKernel = custom::wrapper::kblock::avx_vnni::SiluGemmSKernelDynamicS8KBlock;
        using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
        using DQuantParam = GemmKernel::PrologueA::QParam;
        static FusedInter finter;
        int lda = fin;
        int ldtmp1 = fmid;
        int ldtmp2 = fmid;
        int ldo = fout;
        auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize);
        quanA1.assign((int8_t*)workspace);
        auto offset = workspace == NULL ? 0 : quanA1.mSize;
        auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize);
        quanA2.assign((int8_t*)workspace + offset);
        ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1ptr,
                              w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,    tmp2, ldtmp2, NULL});
      }
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
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,     fin,          fmid,          fout,          activation,    lda,
                            &quanA1, tmp1,         ldtmp1,        &quanA2,       w1ptr,         w2ptr,
                            w3ptr,   tmp1,         ldtmp1,        quanA1.mSPtr,  quanA1.mCStep, w1ptr->mSPtr,
                            output,  ldo,          quanA2.mSPtr,  quanA2.mCStep, w2ptr->mSPtr,  tmp2,
                            ldtmp2,  quanA1.mSPtr, quanA1.mCStep, w3ptr->mSPtr});

    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmDynamicS8PerN;
      using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmDynamicS8PerN;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterfacePerN<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            &quanA1,
                            tmp1,
                            ldtmp1,
                            &quanA2,
                            w1ptr,
                            w2ptr,
                            w3ptr,
                            {tmp1, ldtmp1, quanA1.mCStep, quanA1.mSPtr, w1ptr->mSPtr, quanA1.mZPtr, w1ptr->mRPtr},
                            {output, ldo, quanA2.mCStep, quanA2.mSPtr, w2ptr->mSPtr, quanA2.mZPtr, w2ptr->mRPtr},
                            {tmp2, ldtmp2, quanA1.mCStep, quanA1.mSPtr, w3ptr->mSPtr, quanA1.mZPtr, w3ptr->mRPtr}});
    } else if (_cd->AVX_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx_vnni::GemmDynamicS8PerN;
      using SiluGemmKernel = custom::wrapper::kblock::avx_vnni::SiluGemmDynamicS8PerN;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterfacePerN<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            &quanA1,
                            tmp1,
                            ldtmp1,
                            &quanA2,
                            w1ptr,
                            w2ptr,
                            w3ptr,
                            {tmp1, ldtmp1, quanA1.mCStep, quanA1.mSPtr, w1ptr->mSPtr, quanA1.mZPtr, w1ptr->mRPtr},
                            {output, ldo, quanA2.mCStep, quanA2.mSPtr, w2ptr->mSPtr, quanA2.mZPtr, w2ptr->mRPtr},
                            {tmp2, ldtmp2, quanA1.mCStep, quanA1.mSPtr, w3ptr->mSPtr, quanA1.mZPtr, w3ptr->mRPtr}});
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
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,     fin,          fmid,          fout,          activation,    lda,
                            &quanA1, tmp1,         ldtmp1,        &quanA2,       w1ptr,         w2ptr,
                            w3ptr,   tmp1,         ldtmp1,        quanA1.mSPtr,  quanA1.mCStep, w1ptr->mSPtr,
                            output,  ldo,          quanA2.mSPtr,  quanA2.mCStep, w2ptr->mSPtr,  tmp2,
                            ldtmp2,  quanA1.mSPtr, quanA1.mCStep, w3ptr->mSPtr});

    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmDynamicS4ClipPerN;
      using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmDynamicS4ClipPerN;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterfacePerN<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            &quanA1,
                            tmp1,
                            ldtmp1,
                            &quanA2,
                            w1ptr,
                            w2ptr,
                            w3ptr,
                            {tmp1, ldtmp1, quanA1.mCStep, quanA1.mSPtr, w1ptr->mSPtr, quanA1.mZPtr, w1ptr->mRPtr},
                            {output, ldo, quanA2.mCStep, quanA2.mSPtr, w2ptr->mSPtr, quanA2.mZPtr, w2ptr->mRPtr},
                            {tmp2, ldtmp2, quanA1.mCStep, quanA1.mSPtr, w3ptr->mSPtr, quanA1.mZPtr, w3ptr->mRPtr}});
    } else if (_cd->AVX_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx_vnni::GemmDynamicS4ClipPerN;
      using SiluGemmKernel = custom::wrapper::kblock::avx_vnni::SiluGemmDynamicS4ClipPerN;
      using FusedInter = custom::wrapper::transformer::FFNFusedInterfacePerN<SiluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldtmp2 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            &quanA1,
                            tmp1,
                            ldtmp1,
                            &quanA2,
                            w1ptr,
                            w2ptr,
                            w3ptr,
                            {tmp1, ldtmp1, quanA1.mCStep, quanA1.mSPtr, w1ptr->mSPtr, quanA1.mZPtr, w1ptr->mRPtr},
                            {output, ldo, quanA2.mCStep, quanA2.mSPtr, w2ptr->mSPtr, quanA2.mZPtr, w2ptr->mRPtr},
                            {tmp2, ldtmp2, quanA1.mCStep, quanA1.mSPtr, w3ptr->mSPtr, quanA1.mZPtr, w3ptr->mRPtr}});
    }
  }
  return ret;
}

void jblas_fusion_FFN_SiLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                          float* tmp2, float* output, int seq, int fin, int fmid, int fout,
                                          void* workspace) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr);
  auto w3tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w3ptr);
  auto ret = JblasRuntimeError;

  // must check support before forward, there is no need to check support twice.
  if (w1tmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_fusion_FFN_SiLu_s4fp32_f32f32_forward(activation, dynamic_cast<SS4Fp32*>(w1tmp),
                                                      dynamic_cast<SS4Fp32*>(w2tmp), dynamic_cast<SS4Fp32*>(w3tmp),
                                                      tmp1, tmp2, output, seq, fin, fmid, fout, workspace);
  } else if (w1tmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_fusion_FFN_SiLu_s8fp32_f32f32_forward(activation, dynamic_cast<SS8Fp32*>(w1tmp),
                                                      dynamic_cast<SS8Fp32*>(w2tmp), dynamic_cast<SS8Fp32*>(w3tmp),
                                                      tmp1, tmp2, output, seq, fin, fmid, fout, workspace);
  } else if (w1tmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
    ret = jblas_fusion_FFN_SiLu_s8fp32pern_f32f32_forward(
        activation, dynamic_cast<SS8Fp32PerN*>(w1tmp), dynamic_cast<SS8Fp32PerN*>(w2tmp),
        dynamic_cast<SS8Fp32PerN*>(w3tmp), tmp1, tmp2, output, seq, fin, fmid, fout, workspace);
  } else if (w1tmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
    ret = jblas_fusion_FFN_SiLu_s4clipfp32pern_f32f32_forward(
        activation, dynamic_cast<SS4Fp32PerN*>(w1tmp), dynamic_cast<SS4Fp32PerN*>(w2tmp),
        dynamic_cast<SS4Fp32PerN*>(w3tmp), tmp1, tmp2, output, seq, fin, fmid, fout, workspace);
  }
  assert(ret == JblasSuccess);
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  safe_delete(w3tmp);
}

JBLAS_CODE jblas_fusion_FFN_GeLu_s4fp32_f32f32_forward(float* activation, SS4Fp32* w1tmp, SS4Fp32* w2tmp, float* tmp1,
                                                       float* output, int seq, int fin, int fmid, int fout,
                                                       void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && w1tmp->mBlockSize % 128 == 0) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::GemmSKernelDynamicS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::amx_int8::GeluGemmSKernelDynamicS4KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      GeluGemmKernel::AParam paramA = {activation, lda, &quanA1};
      GemmKernel::AParam paramA2 = {tmp1, ldtmp1, &quanA2};
      GeluGemmKernel::BParam paramW1 = {w1tmp};
      GemmKernel::BParam paramW2 = {w2tmp};
      GeluGemmKernel::EpiParam param1 = {tmp1, ldtmp1};
      GemmKernel::EpiParam param2 = {output, ldo};
      ret = finter.compute({seq, fin, fmid, fout, paramA, paramA2, paramW1, paramW2, param1, param2});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::GeluGemmSKernelDynamicS4KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq, fin, fmid, fout, activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1tmp, w2tmp, tmp1,
                            ldtmp1, output, ldo});
    } else if (_cd->AVX_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx_vnni::GemmSKernelDynamicS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx_vnni::GeluGemmSKernelDynamicS4KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq, fin, fmid, fout, activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1tmp, w2tmp, tmp1,
                            ldtmp1, output, ldo});
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_FFN_GeLu_s8fp32_f32f32_forward(float* activation, SS8Fp32* w1tmp, SS8Fp32* w2tmp, float* tmp1,
                                                       float* output, int seq, int fin, int fmid, int fout,
                                                       void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && w1tmp->mBlockSize % 128 == 0) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::GemmSKernelDynamicS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::amx_int8::GeluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq, fin, fmid, fout, activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1tmp, w2tmp, tmp1,
                            ldtmp1, output, ldo});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::GeluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq, fin, fmid, fout, activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1tmp, w2tmp, tmp1,
                            ldtmp1, output, ldo});
    } else if (_cd->AVX_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx_vnni::GemmSKernelDynamicS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx_vnni::GeluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq, fin, fmid, fout, activation, lda, &quanA1, tmp1, ldtmp1, &quanA2, w1tmp, w2tmp, tmp1,
                            ldtmp1, output, ldo});
    }
  }
  return ret;
}

void jblas_fusion_FFN_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp1, float* output,
                                          int seq, int fin, int fmid, int fout, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr);
  if (w1tmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_fusion_FFN_GeLu_s4fp32_f32f32_forward(activation, (SS4Fp32*)w1tmp, (SS4Fp32*)w2tmp, tmp1, output, seq,
                                                      fin, fmid, fout, workspace);
  } else if (w1tmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_fusion_FFN_GeLu_s8fp32_f32f32_forward(activation, (SS8Fp32*)w1tmp, (SS8Fp32*)w2tmp, tmp1, output, seq,
                                                      fin, fmid, fout, workspace);
  }
  assert(ret == JblasSuccess);
  safe_delete(w1tmp);
  safe_delete(w2tmp);
}
#endif
void jblas_fusion_FFN_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp1, float* output,
                                          int seq, int fin, int fmid, int fout, void* workspace) {}

bool jblas_fusion_FFN_Add_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  return false;
}

#if 0
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
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,        fin,     fmid,    fout,
                            activation, lda,     &quanA1, tmp1,
                            ldtmp1,     &quanA2, w1tmp,   w2tmp,
                            tmp1,       b1ptr,   ldtmp1,  broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,   ldo,     broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmSKernelDynamicS4KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,        fin,     fmid,    fout,
                            activation, lda,     &quanA1, tmp1,
                            ldtmp1,     &quanA2, w1tmp,   w2tmp,
                            tmp1,       b1ptr,   ldtmp1,  broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,   ldo,     broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx_vnni::AddGemmSKernelDynamicS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx_vnni::AddGeluGemmSKernelDynamicS4KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,        fin,     fmid,    fout,
                            activation, lda,     &quanA1, tmp1,
                            ldtmp1,     &quanA2, w1tmp,   w2tmp,
                            tmp1,       b1ptr,   ldtmp1,  broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,   ldo,     broadcast_bias ? 0 : ldo});
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
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,        fin,     fmid,    fout,
                            activation, lda,     &quanA1, tmp1,
                            ldtmp1,     &quanA2, w1tmp,   w2tmp,
                            tmp1,       b1ptr,   ldtmp1,  broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,   ldo,     broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,        fin,     fmid,    fout,
                            activation, lda,     &quanA1, tmp1,
                            ldtmp1,     &quanA2, w1tmp,   w2tmp,
                            tmp1,       b1ptr,   ldtmp1,  broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,   ldo,     broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx_vnni::AddGemmSKernelDynamicS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx_vnni::AddGeluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2tmp->mBlockSize);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,        fin,     fmid,    fout,
                            activation, lda,     &quanA1, tmp1,
                            ldtmp1,     &quanA2, w1tmp,   w2tmp,
                            tmp1,       b1ptr,   ldtmp1,  broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,   ldo,     broadcast_bias ? 0 : ldo});
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
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            &quanA1,
                            tmp1,
                            ldtmp1,
                            &quanA2,
                            w1tmp,
                            w2tmp,
                            tmp1,
                            ldtmp1,
                            quanA1.mSPtr,
                            quanA1.mCStep,
                            w1tmp->mSPtr,
                            b1ptr,
                            broadcast_bias ? 0 : ldtmp1,
                            output,
                            ldo,
                            quanA2.mSPtr,
                            quanA2.mCStep,
                            w2tmp->mSPtr,
                            b2ptr,
                            broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmDynamicS8PerN;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmDynamicS8PerN;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterfacePerN<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            &quanA1,
                            tmp1,
                            ldtmp1,
                            &quanA2,
                            w1tmp,
                            w2tmp,
                            {{tmp1, ldtmp1, quanA1.mCStep, quanA1.mSPtr, w1tmp->mSPtr, quanA1.mZPtr, w1tmp->mRPtr},
                             b1ptr,
                             broadcast_bias ? 0 : ldtmp1},
                            {{output, ldo, quanA2.mCStep, quanA2.mSPtr, w2tmp->mSPtr, quanA2.mZPtr, w2tmp->mRPtr},
                             b2ptr,
                             broadcast_bias ? 0 : ldo}});
    } else if (_cd->AVX_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx_vnni::AddGemmDynamicS8PerN;
      using GeluGemmKernel = custom::wrapper::kblock::avx_vnni::AddGeluGemmDynamicS8PerN;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterfacePerN<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            &quanA1,
                            tmp1,
                            ldtmp1,
                            &quanA2,
                            w1tmp,
                            w2tmp,
                            {{tmp1, ldtmp1, quanA1.mCStep, quanA1.mSPtr, w1tmp->mSPtr, quanA1.mZPtr, w1tmp->mRPtr},
                             b1ptr,
                             broadcast_bias ? 0 : ldtmp1},
                            {{output, ldo, quanA2.mCStep, quanA2.mSPtr, w2tmp->mSPtr, quanA2.mZPtr, w2tmp->mRPtr},
                             b2ptr,
                             broadcast_bias ? 0 : ldo}});
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
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            &quanA1,
                            tmp1,
                            ldtmp1,
                            &quanA2,
                            w1tmp,
                            w2tmp,
                            tmp1,
                            ldtmp1,
                            quanA1.mSPtr,
                            quanA1.mCStep,
                            w1tmp->mSPtr,
                            b1ptr,
                            broadcast_bias ? 0 : ldtmp1,
                            output,
                            ldo,
                            quanA2.mSPtr,
                            quanA2.mCStep,
                            w2tmp->mSPtr,
                            b2ptr,
                            broadcast_bias ? 0 : ldo});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmDynamicS4ClipPerN;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmDynamicS4ClipPerN;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterfacePerN<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            &quanA1,
                            tmp1,
                            ldtmp1,
                            &quanA2,
                            w1tmp,
                            w2tmp,
                            {{tmp1, ldtmp1, quanA1.mCStep, quanA1.mSPtr, w1tmp->mSPtr, quanA1.mZPtr, w1tmp->mRPtr},
                             b1ptr,
                             broadcast_bias ? 0 : ldtmp1},
                            {{output, ldo, quanA2.mCStep, quanA2.mSPtr, w2tmp->mSPtr, quanA2.mZPtr, w2tmp->mRPtr},
                             b2ptr,
                             broadcast_bias ? 0 : ldo}});
    } else if (_cd->AVX_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx_vnni::AddGemmDynamicS4ClipPerN;
      using GeluGemmKernel = custom::wrapper::kblock::avx_vnni::AddGeluGemmDynamicS4ClipPerN;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterfacePerN<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin);
      quanA1.assign((int8_t*)workspace);
      auto offset = workspace == NULL ? 0 : quanA1.mSize;
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid);
      quanA2.assign((int8_t*)workspace + offset);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            &quanA1,
                            tmp1,
                            ldtmp1,
                            &quanA2,
                            w1tmp,
                            w2tmp,
                            {{tmp1, ldtmp1, quanA1.mCStep, quanA1.mSPtr, w1tmp->mSPtr, quanA1.mZPtr, w1tmp->mRPtr},
                             b1ptr,
                             broadcast_bias ? 0 : ldtmp1},
                            {{output, ldo, quanA2.mCStep, quanA2.mSPtr, w2tmp->mSPtr, quanA2.mZPtr, w2tmp->mRPtr},
                             b2ptr,
                             broadcast_bias ? 0 : ldo}});
    }
  }
  return ret;
}

void jblas_fusion_FFN_Add_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* b1ptr, float* b2ptr,
                                              float* tmp1, float* output, int seq, int fin, int fmid, int fout,
                                              bool broadcast_bias, void* workspace) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr);
  if (w1tmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret =
        jblas_fusion_FFN_Add_GeLu_s4fp32_f32f32_forward(activation, (SS4Fp32*)w1tmp, (SS4Fp32*)w2tmp, b1ptr, b2ptr,
                                                        tmp1, output, seq, fin, fmid, fout, broadcast_bias, workspace);
  } else if (w1tmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32)) {
    ret =
        jblas_fusion_FFN_Add_GeLu_s8fp32_f32f32_forward(activation, (SS8Fp32*)w1tmp, (SS8Fp32*)w2tmp, b1ptr, b2ptr,
                                                        tmp1, output, seq, fin, fmid, fout, broadcast_bias, workspace);
  } else if (w1tmp->mPrologueID == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
    ret = jblas_fusion_FFN_Add_GeLu_s8fp32pern_f32f32_forward(activation, (SS8Fp32PerN*)w1tmp, (SS8Fp32PerN*)w2tmp,
                                                              b1ptr, b2ptr, tmp1, output, seq, fin, fmid, fout,
                                                              broadcast_bias, workspace);
  } else if (w1tmp->mPrologueID == int(WeightCompType::WeightS4ClipScaleFp32PerChannelN)) {
    ret = jblas_fusion_FFN_Add_GeLu_s4clipfp32pern_f32f32_forward(activation, (SS4Fp32PerN*)w1tmp, (SS4Fp32PerN*)w2tmp,
                                                                  b1ptr, b2ptr, tmp1, output, seq, fin, fmid, fout,
                                                                  broadcast_bias, workspace);
  }
  assert(ret == JblasSuccess);
  safe_delete(w1tmp);
  safe_delete(w2tmp);
}
#endif
void jblas_fusion_FFN_Add_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* b1ptr, float* b2ptr,
                                              float* tmp1, float* output, int seq, int fin, int fmid, int fout,
                                              bool broadcast_bias, void* workspace) {}
