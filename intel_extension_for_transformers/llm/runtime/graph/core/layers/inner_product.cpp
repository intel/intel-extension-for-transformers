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
#include "layers/inner_product.h"
#include "layers/ele_wise.h"
#include "jblas/jit_blas_weight_compression.h"
#include "jblas/jit_blas_transformer.h"

using namespace jblas;
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
template <typename T>
static inline void safe_delete(T* ptr) {
  if (ptr) {
    delete ptr;
  }
}

namespace {
using WeightCompType = prologue::weight_comp::gemm_kblcok::WeightCompType;

using SS4Fp32 = prologue::weight_comp::gemm_kblcok::StorageWeightS4ScaleFp32;
using SS8Fp32 = prologue::weight_comp::gemm_kblcok::StorageWeightS8ScaleFp32;
using SS8Fp32PerN = prologue::weight_comp::gemm_kblcok::StorageWeightS8ScaleFp32PerChannelN;

template <class GC, JBLAS_ISA ISA>
using WS4Fp32 = jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32<GC, ISA>;

template <class GC, JBLAS_ISA ISA>
using WS4Bf16 = jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleBf16<GC, ISA>;

template <class GC, JBLAS_ISA ISA>
using WS8Fp32 = jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32<GC, ISA>;

template <class GC, JBLAS_ISA ISA>
using WS8Fp32PerN = jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32PerChannelN<GC, ISA>;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using AMX_INT8_KBLOCK_Fp32Fp32 = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        JblasAMX_INT8, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
        jblas::prologue::gemm::ActivationF32S8KBlockQuantize, ProB, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using AMX_INT8_PerN_Fp32Fp32 = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<JblasAMX_INT8, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8,
                                                             jblas::prologue::gemm::ActivationFp32SymS8Quantize, ProB,
                                                             jblas::epilogue::gemm::DequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using AVX512_VNNI_KBLOCK_Fp32Fp32 = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize, ProB, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using AVX512_VNNI_PerN_Fp32Fp32 = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        JblasAVX512_VNNI, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,
        jblas::prologue::gemm::ActivationFp32AsymU8Quantize, ProB, jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;

}  // namespace

static JBLAS_CODE jblas_s4fp32kblock_f32f32_forward(float* activation, SS4Fp32* weiptr, float* output, int _m, int _n,
                                                    int _k, int lda, int ldo) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (weiptr->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK ||
      weiptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
    if (_cd->AMX_INT8() && weiptr->mBlockSize % 128 == 0) {
      using GemmKernel = AMX_INT8_KBLOCK_Fp32Fp32<WS4Fp32>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, activation, lda, quanA, weiptr, output, ldo});
      delete quanA;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = AVX512_VNNI_KBLOCK_Fp32Fp32<WS4Fp32>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, activation, lda, quanA, weiptr, output, ldo});
      delete quanA;
    }
  }
  return ret;
}

static JBLAS_CODE jblas_s8fp32kblock_f32f32_forward(float* activation, SS8Fp32* weiptr, float* output, int _m, int _n,
                                                    int _k, int lda, int ldo) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (weiptr->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK ||
      weiptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
    if (_cd->AMX_INT8() && weiptr->mBlockSize % 128 == 0) {
      using GemmKernel = AMX_INT8_KBLOCK_Fp32Fp32<WS8Fp32>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, activation, lda, quanA, weiptr, output, ldo});
      delete quanA;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = AVX512_VNNI_KBLOCK_Fp32Fp32<WS8Fp32>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, activation, lda, quanA, weiptr, output, ldo});
      delete quanA;
    }
  }
  return ret;
}

static JBLAS_CODE jblas_s8fp32perN_f32f32_forward(float* activation, SS8Fp32PerN* weiptr, float* output, int _m, int _n,
                                                  int _k, int lda, int ldo) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  assert(weiptr->mBlockSize == _k);
  if (weiptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
      weiptr->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16x48_SS) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = AMX_INT8_PerN_Fp32Fp32<WS8Fp32PerN>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, NULL);
      ret = kernel.compute<true, false>(
          {_m, _n, _k, activation, lda, quanA, weiptr, output, ldo, quanA->mSPtr, quanA->lds, weiptr->mSPtr});
      delete quanA;
    } else if (_cd->AVX512_VNNI()) {
      assert(false);
      using GemmKernel = AVX512_VNNI_PerN_Fp32Fp32<WS8Fp32>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, NULL);
      ret = kernel.compute<true, false>({_m, _n, _k, activation, lda, quanA, weiptr, output, ldo, quanA->mZPtr,
                                         quanA->mSPtr, quanA->lds, weiptr->mRPtr, weiptr->mSPtr});
      delete quanA;
    }
  }
  return ret;
}

// f32f32: activation & output dtype
void jblas_f32f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda, int ldo) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto wtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(weiptr, 0);
  if (wtmp != nullptr) {
    if (wtmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
      ret = jblas_s4fp32kblock_f32f32_forward(activation, dynamic_cast<SS4Fp32*>(wtmp), output, _m, _n, _k, lda, ldo);
    } else if (wtmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
      ret = jblas_s8fp32kblock_f32f32_forward(activation, dynamic_cast<SS8Fp32*>(wtmp), output, _m, _n, _k, lda, ldo);
    } else if (wtmp->mType == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
      ret = jblas_s8fp32perN_f32f32_forward(activation, dynamic_cast<SS8Fp32PerN*>(wtmp), output, _m, _n, _k, lda, ldo);
    }
  }
  assert(ret == JblasSuccess);
  safe_delete(wtmp);
}

namespace custom {
namespace epilogue {
template <JBLAS_ISA ISA_T, typename _T>
class Silu {
 public:
  struct Param {
    _T* C;
    int ldc;
  };

  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
#if __AVX512F__
    if (jblas::utils::isa_base<ISA_T>::avx512f) {
      // TODO
    }
#endif
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        cptr[i * _param.ldc + j] = ne_silu_f32(cacheptr[i * cachestep + j]);
      }
    }
    return JblasSuccess;
  }
};
template <JBLAS_ISA ISA_T>
using SiluFp32 = Silu<ISA_T, float>;

template <JBLAS_ISA ISA_T, typename _T>
class Gelu {
 public:
  struct Param {
    _T* C;
    int ldc;
  };

  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    // for (int i = 0; i < M; i++) {
    //   ne_vec_gelu_f32(N, cptr + i * _param.ldc, cacheptr + i * cachestep);
    // }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        cptr[i * _param.ldc + j] = ne_gelu_f32(cacheptr[i * cachestep + j]);
      }
    }
    return JblasSuccess;
  }
};

template <JBLAS_ISA ISA_T>
using GeluFp32 = Gelu<ISA_T, float>;

template <JBLAS_ISA ISA_T, typename _T>
class Add {
 public:
  struct Param {
    _T *C, *D;
    int ldc, ldd;
  };

  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    // for (int i = 0; i < M; i++) {
    //   ne_vec_add_f32(N, cptr + i * _param.ldc,dptr + i * _param.ldd, cacheptr + i * cachestep);
    // }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        cptr[i * _param.ldc + j] = dptr[i * _param.ldd + j] + cacheptr[i * cachestep + j];
      }
    }
    return JblasSuccess;
  }
};
template <JBLAS_ISA ISA_T>
using AddFp32 = Add<ISA_T, float>;

template <JBLAS_ISA ISA_T, typename _T>
class Add_Gelu {
 public:
  struct Param {
    _T *C, *D;
    int ldc, ldd;
  };

  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    for (int i = 0; i < M; i++) {
      ne_vec_add_f32(N, cptr + i * _param.ldc, dptr + i * _param.ldd, cacheptr + i * cachestep);
      // ne_vec_gelu_f32(N, cptr + i * _param.ldc, cptr + i * _param.ldc);
    }
    using GeluKernel = jblas::epilogue::gemm::AccumulatorWriteBackWithGeluFp32<ISA_T>;
    GeluKernel ker;
    typename GeluKernel::Param param{_param.C, _param.ldc, NULL};
    auto ret = ker.forward(cptr, _param.ldc, M_offset, N_offset, M, N, param);
    return ret;
  }
};
template <JBLAS_ISA ISA_T>
using Add_GeluFp32 = Add_Gelu<ISA_T, float>;
}  // namespace epilogue
namespace wrapper {
namespace transformer {
template <class _SiluLauncher_T, class _Launcher_T>
class FFNFusedInterface {
 public:
  static_assert(std::is_same<typename _Launcher_T::AParam, typename _SiluLauncher_T::AParam>::value,
                "Prologue A param of the 2 Launcher (w/wo SILU) should be the same.");
  struct Arguments {
    const int Seq, Fin, FMid, FOut;
    const typename _Launcher_T::AParam paramA;
    const typename _Launcher_T::AParam paramA2;
    const typename _SiluLauncher_T::BParam paramW1;
    const typename _Launcher_T::BParam paramW2, paramW3;
    const typename _SiluLauncher_T::EpiParam param1;
    const typename _Launcher_T::EpiParam param2, param3;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActConfig = typename _SiluLauncher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = jblas::utils::parallel::Parallel2DGemmKBlockFixed<GemmCore>;
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }
  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramW1.packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }
    // dynamic quantization: Seq*Fin
    auto cb = utils::CpuBase();
    auto paraA = mLauncher.mProA.createParallel(_param.Seq, _param.Fin, bptr->mBlockSize);
    auto paraA2 = mLauncher.mProA.createParallel(_param.Seq, _param.FMid, bptr->mBlockSize);

    Parallel _paral = Parallel();   // w1&w3 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, bptr->mBlockSize, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, bptr->mBlockSize, cb.mNumThreads);

    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mLauncher.mProA.launch(_param.paramA, tidx, paraA);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          Config _config{rowidx,     colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                         cb.mL2Cache};
          mActLauncher.launch(
              _actconfig, {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL});
          mLauncher.launch(_config,
                           {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW3, _param.param3, NULL});
          int row_r = jblas::utils::remainsize(rowidx, _paral.mRows, rowsize);
          int col_r = jblas::utils::remainsize(colidx, _paral.mCols, colsize);

          // TODO(Yu): replace the naive inplace eltwise mul
          for (int i = 0; i < row_r; i++) {
            for (int j = 0; j < col_r; j++) {
              _param.param1.C[(rowidx + i) * _param.param1.ldc + colidx + j] *=
                  _param.param3.C[(rowidx + i) * _param.param3.ldc + colidx + j];
            }
          }
        }
      }
#pragma omp barrier
      mLauncher.mProA.launch(_param.paramA2, tidx, paraA2);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config,
                           {_param.Seq, _param.FOut, _param.FMid, _param.paramA2, _param.paramW2, _param.param2, NULL});
        }
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
  _SiluLauncher_T mActLauncher;
};

template <class _GeluLauncher_T, class _Launcher_T>
class GeluFusedInterface {
 public:
  struct Arguments {
    const int Seq, Fin, FMid, FOut;
    const typename _GeluLauncher_T::AParam paramA;
    const typename _Launcher_T::AParam paramA2;
    const typename _GeluLauncher_T::BParam paramW1;
    const typename _Launcher_T::BParam paramW2;
    const typename _GeluLauncher_T::EpiParam param1;
    const typename _Launcher_T::EpiParam param2;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActConfig = typename _GeluLauncher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = jblas::utils::parallel::Parallel2DGemmKBlockFixed<GemmCore>;
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }

  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramW1.packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }
    // dynamic quantization: Seq*Fin
    auto paraA = mActLauncher.mProA.createParallel(_param.Seq, _param.Fin, bptr->mBlockSize);
    auto paraA2 = mLauncher.mProA.createParallel(_param.Seq, _param.FMid, bptr->mBlockSize);
    auto cb = utils::CpuBase();
    Parallel _paral = Parallel();   // w1 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, bptr->mBlockSize, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, bptr->mBlockSize, cb.mNumThreads);

    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mActLauncher.mProA.launch(_param.paramA, tidx, paraA);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          mActLauncher.launch(
              _actconfig, {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL});
        }
      }
#pragma omp barrier
      mLauncher.mProA.launch(_param.paramA2, tidx, paraA2);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config,
                           {_param.Seq, _param.FOut, _param.FMid, _param.paramA2, _param.paramW2, _param.param2, NULL});
        }
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
  _GeluLauncher_T mActLauncher;
};

template <class _GeluLauncher_T, class _Launcher_T>
class FpGeluFusedInterface {
 public:
  struct Arguments {
    const int Seq, Fin, FMid, FOut;
    const typename _GeluLauncher_T::AParam paramA;
    const typename _GeluLauncher_T::BParam paramW1;
    const typename _Launcher_T::BParam paramW2;
    const typename _GeluLauncher_T::EpiParam param1;
    const typename _Launcher_T::EpiParam param2;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActConfig = typename _GeluLauncher_T::ParallelConfig;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = jblas::utils::parallel::Parallel2DGemmKBlockFixed<GemmCore>;

  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramW1.packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }
    auto cb = utils::CpuBase();
    Parallel _paral = Parallel();   // w1 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, bptr->mBlockSize, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, bptr->mBlockSize, cb.mNumThreads);

    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          mActLauncher.launch(
              _actconfig, {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL});
        }
      }

#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config, {_param.Seq,
                                     _param.FOut,
                                     _param.FMid,
                                     {_param.param1.C, _param.param1.ldc},
                                     _param.paramW2,
                                     _param.param2,
                                     NULL});
        }
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
  _GeluLauncher_T mActLauncher;
};

}  // namespace transformer
namespace kblock {
namespace avx512_vnni {
template <template <class GC, JBLAS_ISA ISA> class ProB, template <JBLAS_ISA ISA> class Epi>
using DynamicGemm = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
    jblas::prologue::gemm::ActivationF32U8KBlockQuantize, ProB, Epi>;
using GemmSKernelDynamicS4KBlock = DynamicGemm<WS4Fp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmSKernelDynamicS4KBlock = DynamicGemm<WS4Fp32, custom::epilogue::SiluFp32>;
using GeluGemmSKernelDynamicS4KBlock = DynamicGemm<WS4Fp32, custom::epilogue::GeluFp32>;
using AddGeluGemmSKernelDynamicS4KBlock = DynamicGemm<WS4Fp32, custom::epilogue::Add_GeluFp32>;
using AddGemmSKernelDynamicS4KBlock = DynamicGemm<WS4Fp32, custom::epilogue::AddFp32>;

using GemmSKernelDynamicS8KBlock = DynamicGemm<WS8Fp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmSKernelDynamicS8KBlock = DynamicGemm<WS8Fp32, custom::epilogue::SiluFp32>;
using GeluGemmSKernelDynamicS8KBlock = DynamicGemm<WS8Fp32, custom::epilogue::GeluFp32>;
using AddGeluGemmSKernelDynamicS8KBlock = DynamicGemm<WS8Fp32, custom::epilogue::Add_GeluFp32>;
using AddGemmSKernelDynamicS8KBlock = DynamicGemm<WS8Fp32, custom::epilogue::AddFp32>;
}  // namespace avx512_vnni
namespace amx_int8 {
template <template <class GC, JBLAS_ISA ISA> class ProB, template <JBLAS_ISA ISA> class Epi>
using DynamicGemm = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAMX_INT8, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
    jblas::prologue::gemm::ActivationF32S8KBlockQuantize, ProB, Epi>;

using GemmSKernelDynamicS4KBlock = DynamicGemm<WS4Fp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmSKernelDynamicS4KBlock = DynamicGemm<WS4Fp32, custom::epilogue::SiluFp32>;
using GeluGemmSKernelDynamicS4KBlock = DynamicGemm<WS4Fp32, custom::epilogue::GeluFp32>;
using AddGeluGemmSKernelDynamicS4KBlock = DynamicGemm<WS4Fp32, custom::epilogue::Add_GeluFp32>;
using AddGemmSKernelDynamicS4KBlock = DynamicGemm<WS4Fp32, custom::epilogue::AddFp32>;

using GemmSKernelDynamicS8KBlock = DynamicGemm<WS8Fp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmSKernelDynamicS8KBlock = DynamicGemm<WS8Fp32, custom::epilogue::SiluFp32>;
using GeluGemmSKernelDynamicS8KBlock = DynamicGemm<WS8Fp32, custom::epilogue::GeluFp32>;
using AddGeluGemmSKernelDynamicS8KBlock = DynamicGemm<WS8Fp32, custom::epilogue::Add_GeluFp32>;
using AddGemmSKernelDynamicS8KBlock = DynamicGemm<WS8Fp32, custom::epilogue::AddFp32>;
}  // namespace amx_int8
}  // namespace kblock
}  // namespace wrapper
}  // namespace custom

namespace {
namespace transformer {
namespace avx512_vnni {
static JBLAS_ISA constexpr DefaultISA = JblasAVX512_VNNI;
using QKVGemmDynamicS4Fp32KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
using QKVGemmDynamicS8Fp32KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
}  // namespace avx512_vnni
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
}  // namespace amx_int8
}  // namespace transformer
}  // namespace
JBLAS_CODE jblas_QKVs4fp32_f32f32_forward(float* activation, SS4Fp32* wqptr, SS4Fp32* wkptr, SS4Fp32* wvptr,
                                          float* output, int _m, int _n, int _k, int lda, int ldo) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  float alpha = 1.f, beta = 0.f;
  if (wqptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK ||
      wqptr->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK) {
    if (_cd->AMX_INT8() && wqptr->mBlockSize % 128 == 0) {
      using GemmKernel = jblas::wrapper::transformer_default::weight_comp::amx_int8::QKVGemmDynamicS4Fp32KBlock;
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
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, wqptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, 3, activation, lda, quanA, wparams, oparams, NULL});
      delete quanA;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = jblas::wrapper::transformer_default::weight_comp::avx512_vnni::QKVGemmDynamicS4Fp32KBlock;
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
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, wqptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, 3, activation, lda, quanA, wparams, oparams, NULL});
      delete quanA;
    }
  }
  return ret;
}

JBLAS_CODE jblas_QKVs8fp32_f32f32_forward(float* activation, SS8Fp32* wqptr, SS8Fp32* wkptr, SS8Fp32* wvptr,
                                          float* output, int _m, int _n, int _k, int lda, int ldo) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  float alpha = 1.f, beta = 0.f;
  if (wqptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK ||
      wqptr->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK) {
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
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, wqptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, 3, activation, lda, quanA, wparams, oparams, NULL});
      delete quanA;
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
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, wqptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, 3, activation, lda, quanA, wparams, oparams, NULL});
      delete quanA;
    }
  }
  return ret;
}

template <typename T>
static bool contains(const T& val, const T* set, size_t len) {
  for (size_t i = 0; i < len; i++) {
    if (val == set[i]) {
      return true;
    }
  }
  return false;
}

static bool hasISA(const jblas::gemm::GemmCoreType* set, size_t len) {
  GetCPUDevice();
  bool support = false;
  for (size_t i = 0; i < len; i++) {
    switch (set[i]) {
      case jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK:
      case jblas::gemm::GemmCoreType::AMX_INT8_16x48:
      case jblas::gemm::GemmCoreType::AMX_INT8_16x64:
      case jblas::gemm::GemmCoreType::AMX_INT8_16x48_SS:
        support |= _cd->AMX_INT8();
        break;
      case jblas::gemm::GemmCoreType::AMX_BF16_16x48:
      case jblas::gemm::GemmCoreType::AMX_BF16_16x64:
        support |= _cd->AMX_BF16();
        break;
      case jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK:
      case jblas::gemm::GemmCoreType::AVX512_VNNI_8X48:
        support |= _cd->AVX512_VNNI();
        break;
      case jblas::gemm::GemmCoreType::AVX512_FP16_8x64:
      case jblas::gemm::GemmCoreType::AVX512_FP16_8x96:
        support |= _cd->AVX512_FP16();
        break;
      case jblas::gemm::GemmCoreType::AVX512F_8X48:
        support |= _cd->AVX512F();
        break;
      case jblas::gemm::GemmCoreType::AVX2_4X24:
        support |= _cd->AVX2();
        break;
      default:
        break;
    }
    if (support) {
      break;
    }
  }
  return support;
}

static inline bool samePackedWeight(prologue::PackedWeight* ptr0, prologue::PackedWeight* ptr1) {
  return ptr0->mCoreType == ptr1->mCoreType && ptr0->mType == ptr1->mType;
}

static inline bool samePackedWeight(prologue::PackedWeight** ptrs, size_t len) {
  assert(len >= 2);
  bool sameKernel = samePackedWeight(ptrs[0], ptrs[1]);
  if (sameKernel) {
    for (size_t i = 2; i < len; i++) {
      sameKernel &= samePackedWeight(ptrs[0], ptrs[i]);
    }
  }
  return sameKernel;
}

bool jblas_fusion_QKV_f32f32_support(void* wqptr, void* wkptr, void* wvptr, int _m, int _n, int _k) {
  auto wqtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wqptr, 0);
  auto wktmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wkptr, 0);
  auto wvtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wvptr, 0);
  bool support = false;
  if (wqtmp != nullptr && wktmp != nullptr && wvtmp != nullptr) {
    prologue::PackedWeight* tmps[3] = {wqtmp, wktmp, wvtmp};
    auto sameKernel = samePackedWeight(tmps, 3);
    if (sameKernel) {
      if (wqtmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
        constexpr jblas::gemm::GemmCoreType Cores[] = {jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK,
                                                       jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK};
        constexpr size_t EleNum = sizeof(Cores) / sizeof(Cores[0]);
        support = contains(wqtmp->mCoreType, Cores, EleNum);
        support &= hasISA(Cores, EleNum);
      } else if (wqtmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
        constexpr jblas::gemm::GemmCoreType Cores[] = {jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK,
                                                       jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK};
        constexpr size_t EleNum = sizeof(Cores) / sizeof(Cores[0]);
        support = contains(wqtmp->mCoreType, Cores, EleNum);
        support &= hasISA(Cores, EleNum);
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
                                     int _n, int _k, int lda, int ldo) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto wqtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wqptr, 0);
  auto wktmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wkptr, 0);
  auto wvtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(wvptr, 0);
  // must check support before forward, there is no need to check support twice.
  if (wqtmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_QKVs4fp32_f32f32_forward(activation, dynamic_cast<SS4Fp32*>(wqtmp), dynamic_cast<SS4Fp32*>(wktmp),
                                         dynamic_cast<SS4Fp32*>(wvtmp), output, _m, _n, _k, lda, ldo);
  } else if (wqtmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_QKVs8fp32_f32f32_forward(activation, dynamic_cast<SS8Fp32*>(wqtmp), dynamic_cast<SS8Fp32*>(wktmp),
                                         dynamic_cast<SS8Fp32*>(wvtmp), output, _m, _n, _k, lda, ldo);
  }
  assert(ret == JblasSuccess);
  safe_delete(wqtmp);
  safe_delete(wktmp);
  safe_delete(wvtmp);
}

bool jblas_fusion_add_f32f32_support(void* weiptr, int _m, int _n, int _k) {
  GetCPUDevice();
  bool support = false;
  auto wtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(weiptr, 0);
  if (wtmp) {
    if (wtmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
      constexpr jblas::gemm::GemmCoreType Cores[] = {jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK,
                                                     jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK};
      constexpr size_t EleNum = sizeof(Cores) / sizeof(Cores[0]);
      support = contains(wtmp->mCoreType, Cores, EleNum);
      support &= hasISA(Cores, EleNum);
    } else if (wtmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
      constexpr jblas::gemm::GemmCoreType Cores[] = {jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK,
                                                     jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK};
      constexpr size_t EleNum = sizeof(Cores) / sizeof(Cores[0]);
      support = contains(wtmp->mCoreType, Cores, EleNum);
      support &= hasISA(Cores, EleNum);
    }
  }
  safe_delete(wtmp);
  return support;
}

JBLAS_CODE jblas_fusion_add_s4fp32_f32f32_forward(float* activation, SS4Fp32* weiptr, float* bias, float* output,
                                                  int _m, int _n, int _k, int lda, int ldo, bool broadcast_bias) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (weiptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK ||
      weiptr->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK) {
    if (_cd->AMX_INT8() && weiptr->mBlockSize % 128 == 0) {
      using GemmKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
          custom::wrapper::kblock::amx_int8::AddGemmSKernelDynamicS4KBlock,
          jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, activation, lda, quanA, weiptr, output, bias, ldo, broadcast_bias ? 0 : ldo});
      delete quanA;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
          custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS4KBlock,
          jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, activation, lda, quanA, weiptr, output, bias, ldo, broadcast_bias ? 0 : ldo});
      delete quanA;
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_add_s8fp32_f32f32_forward(float* activation, SS8Fp32* weiptr, float* bias, float* output,
                                                  int _m, int _n, int _k, int lda, int ldo, bool broadcast_bias) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (weiptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK ||
      weiptr->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK) {
    if (_cd->AMX_INT8() && weiptr->mBlockSize % 128 == 0) {
      using GemmKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
          custom::wrapper::kblock::amx_int8::AddGemmSKernelDynamicS8KBlock,
          jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, activation, lda, quanA, weiptr, output, bias, ldo, broadcast_bias ? 0 : ldo});
      delete quanA;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
          custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS8KBlock,
          jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
      static GemmKernel kernel;
      auto quanA = kernel.getActivationPtr()->createStorage(_m, _k, weiptr->mBlockSize, NULL);
      ret = kernel.compute({_m, _n, _k, activation, lda, quanA, weiptr, output, bias, ldo, broadcast_bias ? 0 : ldo});
      delete quanA;
    }
  }
  return ret;
}

void jblas_fusion_add_f32f32_forward(float* activation, void* weiptr, float* bias, float* output, int _m, int _n,
                                     int _k, int lda, int ldo, bool broadcast_bias) {
  auto ret = JblasRuntimeError;
  auto wtmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(weiptr, 0);
  if (wtmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_fusion_add_s4fp32_f32f32_forward(activation, (SS4Fp32*)wtmp, bias, output, _m, _n, _k, lda, ldo,
                                                 broadcast_bias);
  } else if (wtmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_fusion_add_s8fp32_f32f32_forward(activation, (SS8Fp32*)wtmp, bias, output, _m, _n, _k, lda, ldo,
                                                 broadcast_bias);
  }
  assert(ret == JblasSuccess);
  safe_delete(wtmp);
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
        if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
          constexpr jblas::gemm::GemmCoreType Cores[] = {jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK,
                                                         jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK};
          constexpr size_t EleNum = sizeof(Cores) / sizeof(Cores[0]);
          support = contains(w1tmp->mCoreType, Cores, EleNum);
          support &= hasISA(Cores, EleNum);
        } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
          constexpr jblas::gemm::GemmCoreType Cores[] = {jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK,
                                                         jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK};
          constexpr size_t EleNum = sizeof(Cores) / sizeof(Cores[0]);
          support = contains(w1tmp->mCoreType, Cores, EleNum);
          support &= hasISA(Cores, EleNum);
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
                                                       int fin, int fmid, int fout) {
  if (w1ptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK ||
      w1ptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48) {
    using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS4KBlock;
    using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmSKernelDynamicS4KBlock;
    using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
    static FusedInter finter;
    int lda = fin;
    int ldtmp1 = fmid;
    int ldtmp2 = fmid;
    int ldo = fout;

    auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize, NULL);
    auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize, NULL);
    auto ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, quanA1, tmp1, ldtmp1, quanA2, w1ptr,
                               w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,   tmp2, ldtmp2, NULL});
    delete quanA1;
    delete quanA2;
    return ret;
  }
  return JblasNotSupport;
}
JBLAS_CODE jblas_fusion_FFN_SiLu_s8fp32_f32f32_forward(float* activation, SS8Fp32* w1ptr, SS8Fp32* w2ptr,
                                                       SS8Fp32* w3ptr, float* tmp1, float* tmp2, float* output, int seq,
                                                       int fin, int fmid, int fout) {
  if (w1ptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK ||
      w1ptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48) {
    using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS8KBlock;
    using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmSKernelDynamicS8KBlock;
    using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
    static FusedInter finter;
    int lda = fin;
    int ldtmp1 = fmid;
    int ldtmp2 = fmid;
    int ldo = fout;
    auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize, NULL);
    auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize, NULL);
    auto ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, quanA1, tmp1, ldtmp1, quanA2, w1ptr,
                               w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,   tmp2, ldtmp2, NULL});
    delete quanA1;
    delete quanA2;
    return ret;
  }
  return JblasNotSupport;
}

void jblas_fusion_FFN_SiLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                          float* tmp2, float* output, int seq, int fin, int fmid, int fout) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  auto w3tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w3ptr, 0);
  auto ret = JblasRuntimeError;

  // must check support before forward, there is no need to check support twice.
  if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_fusion_FFN_SiLu_s4fp32_f32f32_forward(activation, dynamic_cast<SS4Fp32*>(w1tmp),
                                                      dynamic_cast<SS4Fp32*>(w2tmp), dynamic_cast<SS4Fp32*>(w3tmp),
                                                      tmp1, tmp2, output, seq, fin, fmid, fout);
  } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_fusion_FFN_SiLu_s8fp32_f32f32_forward(activation, dynamic_cast<SS8Fp32*>(w1tmp),
                                                      dynamic_cast<SS8Fp32*>(w2tmp), dynamic_cast<SS8Fp32*>(w3tmp),
                                                      tmp1, tmp2, output, seq, fin, fmid, fout);
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
                                          int seq, int fin, int fmid, int fout) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  if (w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
      w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
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
        if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
          constexpr jblas::gemm::GemmCoreType Cores[] = {jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK,
                                                         jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK};
          constexpr size_t EleNum = sizeof(Cores) / sizeof(Cores[0]);
          support = contains(w1tmp->mCoreType, Cores, EleNum);
          support &= hasISA(Cores, EleNum);
        } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
          constexpr jblas::gemm::GemmCoreType Cores[] = {jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK,
                                                         jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK};
          constexpr size_t EleNum = sizeof(Cores) / sizeof(Cores[0]);
          support = contains(w1tmp->mCoreType, Cores, EleNum);
          support &= hasISA(Cores, EleNum);
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
                                                           int seq, int fin, int fmid, int fout, bool broadcast_bias) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
      w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
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
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, NULL);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w1tmp->mBlockSize, NULL);
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
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, NULL);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w1tmp->mBlockSize, NULL);
      ret = finter.compute({seq,        fin,    fmid,   fout,
                            activation, lda,    quanA1, tmp1,
                            ldtmp1,     quanA2, w1tmp,  w2tmp,
                            tmp1,       b1ptr,  ldtmp1, broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,  ldo,    broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_FFN_Add_GeLu_s8fp32_f32f32_forward(float* activation, SS8Fp32* w1tmp, SS8Fp32* w2tmp,
                                                           float* b1ptr, float* b2ptr, float* tmp1, float* output,
                                                           int seq, int fin, int fmid, int fout, bool broadcast_bias) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
      w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
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
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, NULL);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w1tmp->mBlockSize, NULL);
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
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, NULL);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w1tmp->mBlockSize, NULL);
      ret = finter.compute({seq,        fin,    fmid,   fout,
                            activation, lda,    quanA1, tmp1,
                            ldtmp1,     quanA2, w1tmp,  w2tmp,
                            tmp1,       b1ptr,  ldtmp1, broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,  ldo,    broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    }
  }
  return ret;
}

void jblas_fusion_FFN_Add_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* b1ptr, float* b2ptr,
                                              float* tmp1, float* output, int seq, int fin, int fmid, int fout,
                                              bool broadcast_bias) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_fusion_FFN_Add_GeLu_s4fp32_f32f32_forward(activation, (SS4Fp32*)w1tmp, (SS4Fp32*)w2tmp, b1ptr, b2ptr,
                                                          tmp1, output, seq, fin, fmid, fout, broadcast_bias);
  } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_fusion_FFN_Add_GeLu_s8fp32_f32f32_forward(activation, (SS8Fp32*)w1tmp, (SS8Fp32*)w2tmp, b1ptr, b2ptr,
                                                          tmp1, output, seq, fin, fmid, fout, broadcast_bias);
  }
  assert(ret == JblasSuccess);
  safe_delete(w1tmp);
  safe_delete(w2tmp);
}
