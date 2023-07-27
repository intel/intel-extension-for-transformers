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
void jblas_weights4block_f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda,
                                     int ldo) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto wtmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(weiptr, 0);
  if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK ||
      wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
      wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
    auto wbtmp = dynamic_cast<prologue::weight_comp::PackedWeightKBlock*>(wtmp);
    if (_cd->AMX_INT8() && wbtmp->mBlockSize % 128 == 0) {
      using GemmKernel = jblas::wrapper::gemm_default::weight_comp::amx_int8::GemmSKernelDynamicS4KBlock;
      static GemmKernel kernel;
      ret = kernel.compute({_m, _n, _k, activation, lda, wtmp, output, ldo});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = jblas::wrapper::gemm_default::weight_comp::avx512_vnni::GemmSKernelDynamicS4KBlock;
      static GemmKernel kernel;
      ret = kernel.compute({_m, _n, _k, activation, lda, wtmp, output, ldo});
    }
  } else if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512F_8X48) {
    using GemmKernel = jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4KBlock;
    float alpha = 1.f, beta = 0.f;
    static GemmKernel kernel;
    if (_cd->AVX512F()) {
      ret = kernel.compute({_m, _n, _k, activation, lda, wtmp, output, output, ldo, ldo, alpha, beta});
    }
  } else if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_BF16_16x64) {
    using GemmKernel = jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS4KBlock;
    static GemmKernel kernel;
    if (_cd->AMX_BF16()) {
      ret = kernel.compute({_m, _n, _k, activation, lda, wtmp, output, ldo});
    }
  }
  assert(ret == JblasSuccess);
  delete wtmp;
}

namespace custom {
namespace epilogue {
template <typename _T>
class Silu {
 public:
  struct Param {
    _T* C;
    int ldc;
  };

  template <JBLAS_ISA ISA_T>
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

template <typename _T>
class Gelu {
 public:
  struct Param {
    _T* C;
    int ldc;
  };

  template <JBLAS_ISA ISA_T>
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

template <typename _T>
class Add {
 public:
  struct Param {
    _T *C, *D;
    int ldc, ldd;
  };

  template <JBLAS_ISA ISA_T>
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

template <typename _T>
class Add_Gelu {
 public:
  struct Param {
    _T *C, *D;
    int ldc, ldd;
  };

  template <JBLAS_ISA ISA_T>
  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    for (int i = 0; i < M; i++) {
      ne_vec_add_f32(N, cptr + i * _param.ldc, dptr + i * _param.ldd, cacheptr + i * cachestep);
      ne_vec_gelu_f32(N, cptr + i * _param.ldc, cptr + i * _param.ldc);
    }
    /*for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        cptr[i * _param.ldc + j] = dptr[i * _param.ldd + j] + cacheptr[i * cachestep + j];
        cptr[i * _param.ldc + j] = ne_gelu_f32(cptr[i * _param.ldc + j]);
      }
    }*/
    return JblasSuccess;
  }
};

}  // namespace epilogue
namespace wrapper {
namespace transformer {
template <class _SiluLauncher_T, class _Launcher_T>
class FFNFusedInterface {
 public:
  static_assert(std::is_same<typename _Launcher_T::AParam, typename _SiluLauncher_T::AParam>::value);
  struct Arguments {
    const int Seq, Fin, FMid, FOut;
    const typename _Launcher_T::AParam paramA;
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

  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramW1.packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }
    // dynamic quantization: Seq*Fin
    auto paraA = mLauncher.mProA.createParallel(_param.Seq, _param.Fin, bptr->mBlockSize);
    auto quanA = mLauncher.mProA.createObj(_param.Seq, _param.Fin, bptr->mBlockSize);
    auto cb = utils::CpuBase();
    Parallel _paral = Parallel();   // w1&w3 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, bptr->mBlockSize, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, bptr->mBlockSize, cb.mNumThreads);

    auto paraA2 = mLauncher.mProA.createParallel(_param.Seq, _param.FMid, bptr->mBlockSize);
    auto quanA2 = mLauncher.mProA.createObj(_param.Seq, _param.FMid, bptr->mBlockSize);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mLauncher.mProA.template quantizeT<_Launcher_T::RT_ISA>(_param.paramA, tidx, quanA, paraA);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          Config _config{rowidx,     colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                         cb.mL2Cache};
          mActLauncher.launch(_actconfig,
                              {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL},
                              quanA);
          mLauncher.launch(_config,
                           {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW3, _param.param3, NULL},
                           quanA);
          int row_r = jblas::utils::remainsize(rowidx, _paral.mRows, rowsize);
          int col_r = jblas::utils::remainsize(colidx, _paral.mCols, colsize);

          for (int i = 0; i < row_r; i++) {
            for (int j = 0; j < col_r; j++) {
              _param.param1.C[(rowidx + i) * _param.param1.ldc + colidx + j] *=
                  _param.param3.C[(rowidx + i) * _param.param3.ldc + colidx + j];
            }
          }
        }
      }
#pragma omp barrier
      mLauncher.mProA.template quantizeT<_Launcher_T::RT_ISA>({_param.param1.C, _param.param1.ldc}, tidx, quanA2,
                                                              paraA2);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config,
                           {_param.Seq, _param.FOut, _param.FMid, _param.paramA, _param.paramW2, _param.param2, NULL},
                           quanA2);
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

  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramW1.packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }
    // dynamic quantization: Seq*Fin
    auto paraA = mActLauncher.mProA.createParallel(_param.Seq, _param.Fin, bptr->mBlockSize);
    auto quanA = mActLauncher.mProA.createObj(_param.Seq, _param.Fin, bptr->mBlockSize);
    auto cb = utils::CpuBase();
    Parallel _paral = Parallel();   // w1 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, bptr->mBlockSize, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, bptr->mBlockSize, cb.mNumThreads);

    auto paraA2 = mLauncher.mProA.createParallel(_param.Seq, _param.FMid, bptr->mBlockSize);
    auto quanA2 = mLauncher.mProA.createObj(_param.Seq, _param.FMid, bptr->mBlockSize);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mActLauncher.mProA.template quantizeT<_Launcher_T::RT_ISA>(_param.paramA, tidx, quanA, paraA);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          mActLauncher.launch(_actconfig,
                              {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL},
                              quanA);
        }
      }
#pragma omp barrier
      mLauncher.mProA.template quantizeT<_Launcher_T::RT_ISA>({_param.param1.C, _param.param1.ldc}, tidx, quanA2,
                                                              paraA2);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config,
                           {_param.Seq, _param.FOut, _param.FMid, _param.paramA, _param.paramW2, _param.param2, NULL},
                           quanA2);
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
using GemmSKernelDynamicS4KBlock = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
    jblas::prologue::gemm::ActivationF32U8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
    jblas::epilogue::gemm::AccumulatorWriteBack<float, float>>;
using SiluGemmSKernelDynamicS4KBlock = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
    jblas::prologue::gemm::ActivationF32U8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
    custom::epilogue::Silu<float>>;
using GeluGemmSKernelDynamicS4KBlock = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
    jblas::prologue::gemm::ActivationF32U8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
    custom::epilogue::Gelu<float>>;
using AddGeluGemmSKernelDynamicS4KBlock = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
    jblas::prologue::gemm::ActivationF32U8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
    custom::epilogue::Add_Gelu<float>>;
using AddGemmSKernelDynamicS4KBlock = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
    jblas::prologue::gemm::ActivationF32U8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
    custom::epilogue::Add<float>>;
}  // namespace avx512_vnni
namespace amx_int8 {
using GemmSKernelDynamicS4KBlock = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAMX_INT8, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
    jblas::prologue::gemm::ActivationF32S8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
    jblas::epilogue::gemm::AccumulatorWriteBack<float, float>>;
using SiluGemmSKernelDynamicS4KBlock = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAMX_INT8, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
    jblas::prologue::gemm::ActivationF32S8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
    custom::epilogue::Silu<float>>;
using GeluGemmSKernelDynamicS4KBlock = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAMX_INT8, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
    jblas::prologue::gemm::ActivationF32S8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
    custom::epilogue::Gelu<float>>;
using AddGeluGemmSKernelDynamicS4KBlock = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAMX_INT8, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
    jblas::prologue::gemm::ActivationF32S8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
    custom::epilogue::Add_Gelu<float>>;
using AddGemmSKernelDynamicS4KBlock = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAMX_INT8, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
    jblas::prologue::gemm::ActivationF32S8KBlockQuantize, jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
    custom::epilogue::Add<float>>;
}  // namespace amx_int8
}  // namespace kblock
}  // namespace wrapper
}  // namespace custom

void jblas_weightcomp_QKV_f32_forward(float* activation, void* wqptr, void* wkptr, void* wvptr, float* output, int _m,
                                      int _n, int _k, int lda, int ldo) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto wqtmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(wqptr, 0);
  auto wktmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(wkptr, 0);
  auto wvtmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(wvptr, 0);
  float alpha = 1.f, beta = 0.f;
  if (wqtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK ||
      wqtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
      wqtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16X48_KBLOCK) {
    auto wbtmp = dynamic_cast<prologue::weight_comp::PackedWeightKBlock*>(wqtmp);
    if (_cd->AMX_INT8() && wbtmp->mBlockSize % 128 == 0) {
      using GemmKernel = jblas::wrapper::transformer_default::weight_comp::amx_int8::QKVGemmSKernelDynamicS4KBlock;
      static GemmKernel kernel;
      GemmKernel::WeightType::Param wparams[3]{
          wqtmp,
          wktmp,
          wvtmp,
      };
      GemmKernel::CParam oparams[3]{
          {output, ldo},
          {output + _m * _n, ldo},
          {output + 2 * _m * _n, ldo},
      };
      ret = kernel.compute2({_m, _n, _k, 3, activation, lda, wparams, oparams, NULL});
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = jblas::wrapper::transformer_default::weight_comp::avx512_vnni::QKVGemmSKernelDynamicS4KBlock;
      static GemmKernel kernel;
      GemmKernel::WeightType::Param wparams[3]{
          wqtmp,
          wktmp,
          wvtmp,
      };
      GemmKernel::CParam oparams[3]{
          {output, ldo},
          {output + _m * _n, ldo},
          {output + 2 * _m * _n, ldo},
      };
      ret = kernel.compute2({_m, _n, _k, 3, activation, lda, wparams, oparams, NULL});
    }
  } else if (wqtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_BF16_16x64) {
    using GemmKernel = jblas::wrapper::transformer_default::weight_comp::amx_bf16::QKVGemm;
    static GemmKernel kernel;
    GemmKernel::WeightType::Param wparams[3]{
        wqtmp,
        wktmp,
        wvtmp,
    };
    GemmKernel::CParam oparams[3]{
        {output, ldo},
        {output + _m * _n, ldo},
        {output + 2 * _m * _n, ldo},
    };
    if (_cd->AMX_BF16()) {
      ret = kernel.compute({_m, _n, _k, 3, activation, lda, wparams, oparams, NULL});
    }
  } else if (wqtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512F_8X48) {
    using GemmKernel = jblas::wrapper::transformer_default::weight_comp::avx512_f::QKVGemm;
    static GemmKernel kernel;
    GemmKernel::WeightType::Param wparams[3]{
        wqtmp,
        wktmp,
        wvtmp,
    };
    GemmKernel::CParam oparams[3]{
        {output, ldo},
        {output + _m * _n, ldo},
        {output + 2 * _m * _n, ldo},
    };
    if (_cd->AVX512F()) {
      ret = kernel.compute({_m, _n, _k, 3, activation, lda, wparams, oparams, NULL});
    }
  }
  assert(ret == JblasSuccess);
  delete wqtmp;
  delete wktmp;
  delete wvtmp;
}

void jblas_weightcomp_FFN_SiLu_f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                           float* tmp2, float* output, int seq, int fin, int fmid, int fout) {
  auto w1tmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(w2ptr, 0);
  auto w3tmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(w3ptr, 0);
  if (w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK ||
      w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48) {
    using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS4KBlock;
    using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmSKernelDynamicS4KBlock;
    using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
    static FusedInter finter;
    int lda = fin;
    int ldtmp1 = fmid;
    int ldtmp2 = fmid;
    int ldo = fout;
    finter.compute(
        {seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, w3tmp, tmp1, ldtmp1, output, ldo, tmp2, ldtmp2});
  }
  delete w1tmp;
  delete w2tmp;
  delete w3tmp;
}

void jblas_weightcomp_FFN_GeLu_f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp1, float* output,
                                           int seq, int fin, int fmid, int fout) {
  auto w1tmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(w2ptr, 0);
  if (w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
      w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
    using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS4KBlock;
    using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::GeluGemmSKernelDynamicS4KBlock;
    using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
    static FusedInter finter;
    int lda = fin;
    int ldtmp1 = fmid;
    int ldo = fout;
    finter.compute({seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, tmp1, ldtmp1, output, ldo});
  }
  delete w1tmp;
  delete w2tmp;
}

void jblas_weightcomp_FFN_Add_GeLu_f32_forward(float* activation, void* w1ptr, void* w2ptr, float* b1ptr, float* b2ptr,
                                               float* tmp1, float* output, int seq, int fin, int fmid, int fout) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto w1tmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(w2ptr, 0);
  if (w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
      w1tmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
    auto wbtmp = dynamic_cast<prologue::weight_comp::PackedWeightKBlock*>(w1tmp);
    if (_cd->AMX_INT8() && wbtmp->mBlockSize % 128 == 0) {
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
      ret = finter.compute(
          {seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, tmp1, b1ptr, ldtmp1, ldtmp1, output, b2ptr, ldo, ldo});
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
      ret = finter.compute(
          {seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, tmp1, b1ptr, ldtmp1, ldtmp1, output, b2ptr, ldo, ldo});
    }
  }
  assert(ret = JblasSuccess);
  delete w1tmp;
  delete w2tmp;
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
