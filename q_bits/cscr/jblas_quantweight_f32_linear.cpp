#include "jblas_quantweight_f32_linear.hpp"

#include <torch/script.h>
#include <chrono>
#include "jblas/jit_blas_transformer.h"
#include "jblas/jit_blas_weight_compression.h"

void quantweight_f32_linear_launcher(const torch::Tensor& activation, const torch::Tensor& weight, float* bias_ptr,
                                     torch::Tensor& output, const std::string& compute_type,
                                     const std::string& quant_type, int64_t m, int64_t n, int64_t k, int64_t lda,
                                     int64_t ldo, bool need_bias) {
  bool q_bit_verbose = std::getenv("QBIT_VERBOSE") != nullptr;
  auto begin = std::chrono::high_resolution_clock::now();
  float alpha = 1.f, beta = 0.f;
  if (need_bias) beta = 1.f;
  auto wtmp = jblas::prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(weight.data_ptr<int8_t>(), 0);
  int bits = quant_type == "s8" ? 8 : 4;

  auto s8_linear = [&] {
    if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_BF16_16x64) {
      LINEAR_EXECUTE(jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS8KBlock)
    } else {
      LINEAR_EXECUTE(jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS8KBlock)
    }
  };

  BIT4_LINEAR(s4_clip_linear, jblas::wrapper::gemm_default::weight_comp::amx_int8::GemmSKernelDynamicS4ClipKBlock,
              jblas::wrapper::gemm_default::weight_comp::avx512_vnni::GemmSKernelDynamicS4ClipKBlock,
              jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS4ClipKBlock,
              jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4ClipKBlock)

  BIT4_LINEAR(s4_fullrange_linear,
              jblas::wrapper::gemm_default::weight_comp::amx_int8::GemmSKernelDynamicS4FullRangeKBlock,
              jblas::wrapper::gemm_default::weight_comp::avx512_vnni::GemmSKernelDynamicS4FullRangeKBlock,
              jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS4FullRangeKBlock,
              jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4FullRangeKBlock)

  if (quant_type == "s8") s8_linear();
  if (quant_type == "s4_clip") s4_clip_linear();
  if (quant_type == "s4_fullrange") s4_fullrange_linear();

  if (q_bit_verbose) {
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e6;
    std::cout << "m: " << m << " n: " << n << " k: " << k << " with_bias: " << need_bias
              << " cmp_type: " << cmp_type[wtmp->mType] << " quant_type: " << quant_type
              << " gemm_core_type: " << gemm_core_type[static_cast<int>(wtmp->mCoreType)] << " time: " << time << "ms"
              << std::endl;
  }
  delete wtmp;
}