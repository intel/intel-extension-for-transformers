#include "inner_product/inner_product.h"
#include "jblas/jit_blas_weight_compression.h"

using namespace jblas;

void jblas_weights4block_f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda,
                                     int ldo) {
  using GemmKernel = jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4KBlock;
  using WeightType = GemmKernel::WeightType;
  auto wtmp = WeightType::PackedWeightBase::deserialBuffer(weiptr, 0);
  float alpha = 1.f, beta = 0.f;
  static GemmKernel kernel;
  auto ret = kernel.compute({_m, _n, _k, activation, lda, wtmp, output, output, ldo, ldo, alpha, beta});
  delete wtmp;
}
