#pragma once
#include "jit_blas_weight_compression.h"
#include "jit_blas_wrapper.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace wrapper {
namespace transformer {

// compared with BatchGemm, QKV has the same activation matrix.
// This is the reason why making it a new template.
template <class _Launcher_T, template <class _T> class _Parallel_T>
class QKVGemmInterfaceKBlockPackWeight {
 public:
  struct Arguments {
    const int M, N, K, Batch;
    const typename _Launcher_T::AParam paramA;
    const typename _Launcher_T::BParam* paramsB;
    const typename _Launcher_T::EpiParam* paramsC;
    void* workspace;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using QuanParam = typename _Launcher_T::QuanAParam;
  using Parallel = _Parallel_T<GemmCore>;
  Parallel createParallel(int M = 0, int N = 0, int K = 0, int Batch = 1, int KBlock = 0) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(M, N, K, KBlock, cb.mNumThreads);
    return _paral;
  }
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }
  WeightType* getWeightPtr() { return &mLauncher.mProB; }
  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param, Parallel _paral = Parallel()) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramsB[0].packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }
    auto quanA =
        mLauncher.mProA.template quantize<_Launcher_T::RT_ISA>(_param.paramA, _param.M, _param.K, bptr->mBlockSize);
    auto cb = utils::CpuBase();
    if (_paral.update(_param.M, _param.N, _param.K, bptr->mBlockSize, cb.mNumThreads)) {
      static bool dbgprint = false;
      if (dbgprint) {
        _paral.print();
        dbgprint = false;
      }
    }
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        Config _config{rowidx,     colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                       cb.mL2Cache};
        for (size_t i = 0; i < _param.Batch; i++) {
          mLauncher.launch(
              _config,
              {_param.M, _param.N, _param.K, _param.paramA, _param.paramsB[i], _param.paramsC[i], _param.workspace},
              quanA);
        }
      }
    }
    return JblasSuccess;
  }

  JBLAS_CODE compute2(const Arguments& _param, Parallel _paral = Parallel()) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramsB[0].packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }

    auto cb = utils::CpuBase();
    if (_paral.update(_param.M, _param.N, _param.K, bptr->mBlockSize, cb.mNumThreads)) {
      static bool dbgprint = false;
      if (dbgprint) {
        _paral.print();
        dbgprint = false;
      }
    }
    auto paraA = mLauncher.mProA.createParallel(_param.M, _param.K, bptr->mBlockSize);
    auto quanA = mLauncher.mProA.createObj(_param.M, _param.K, bptr->mBlockSize);

    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mLauncher.mProA.template quantizeT<_Launcher_T::RT_ISA>(_param.paramA, tidx, quanA, paraA);
#pragma omp barrier
      launchT(_param, tidx, _paral, quanA, cb.mL2Cache);
    }
    return JblasSuccess;
  }

 protected:
  void launchT(const Arguments& _param, int tidx, Parallel& _paral, QuanParam& quanA, size_t l2cache) {
    int colidx, rowidx, rowsize, colsize;
    _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    if (rowsize > 0 && colsize > 0) {
      Config _config{rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                     l2cache};
      for (size_t i = 0; i < _param.Batch; i++) {
        mLauncher.launch(
            _config,
            {_param.M, _param.N, _param.K, _param.paramA, _param.paramsB[i], _param.paramsC[i], _param.workspace},
            quanA);
      }
    }
  }

  _Launcher_T mLauncher;
};

}  // namespace transformer
namespace transformer_default {
namespace weight_comp {
namespace avx512_vnni {
static JBLAS_ISA constexpr DefaultISA = JblasAVX512_VNNI;

using QKVGemmSKernelDynamicS4KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm::WeightS4_KBlock, jblas::epilogue::gemm::AccumulateWriteBack<float>>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

}  // namespace avx512_vnni
}  // namespace weight_comp
}  // namespace transformer_default
}  // namespace wrapper
}  // namespace jblas
