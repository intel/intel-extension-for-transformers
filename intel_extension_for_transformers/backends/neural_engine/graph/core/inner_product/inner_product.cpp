#include "inner_product/inner_product.h"
#include "jblas/jit_blas_weight_compression.h"

using namespace jblas;

void jblas_weights4block_f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda,
                                     int ldo) {
  auto wtmp = prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(weiptr, 0);
  if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
      wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
    using GemmKernel = jblas::wrapper::gemm_default::weight_comp::avx512_vnni::GemmSKernelDynamicS4KBlock;
    static GemmKernel kernel;
    auto ret = kernel.compute({_m, _n, _k, activation, lda, wtmp, output, ldo});
  } else if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512F_8X48) {
    using GemmKernel = jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4KBlock;
    float alpha = 1.f, beta = 0.f;
    static GemmKernel kernel;
    auto ret = kernel.compute({_m, _n, _k, activation, lda, wtmp, output, output, ldo, ldo, alpha, beta});
  }
  delete wtmp;
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
