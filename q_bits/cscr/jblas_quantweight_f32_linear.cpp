#include "jblas_quantweight_f32_linear.hpp"

#include <torch/script.h>
#include <chrono>
#include "jblas/jit_blas_transformer.h"
#include "jblas/jit_blas_weight_compression.h"

#define LINEAR_EXECUTE(KER)  \
  static KER kernel;         \
  auto ret = kernel.compute( \
      {m, n, k, activation.data_ptr<float>(), lda, wtmp, output.data_ptr<float>(), bias_ptr, ldo, 0, alpha, beta});

void quantweight_f32_linear_launcher(const torch::Tensor& activation, const torch::Tensor& weight, float* bias_ptr,
                                     torch::Tensor& output, const std::string& compute_type, int64_t bits, int64_t m,
                                     int64_t n, int64_t k, int64_t lda, int64_t ldo, bool need_bias) {
  bool q_bit_verbose = std::getenv("QBIT_VERBOSE") != nullptr;
  static std::map<int, std::string> cmp_type = {{1, "S8_F32"}, {2, "S4_F32"}, {3, "S4_BF16"}};
  auto begin = std::chrono::high_resolution_clock::now();
  float alpha = 1.f, beta = 0.f;
  if (need_bias) beta = 1.f;

  auto wtmp = jblas::prologue::weight_comp::gemm::CompressedPackedWeight::deserialBuffer(weight.data_ptr<int8_t>(), 0);
  if (compute_type == "int8") {
    TORCH_CHECK(bits == 4, "quantization bits must be 4 when compute_type==int8");
    if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16x48) {
      LINEAR_EXECUTE(jblas::wrapper::gemm_default::weight_comp::amx_int8::GemmSKernelDynamicS4KBlock)
    } else {
      LINEAR_EXECUTE(jblas::wrapper::gemm_default::weight_comp::avx512_vnni::GemmKernelDynamicQuantS4KBlock)
    }
  } else {
    if (bits == 4) {
      if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_BF16_16x64) {
        LINEAR_EXECUTE(jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS4KBlock)
      } else {
        LINEAR_EXECUTE(jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4KBlock)
      }
    } else {
      if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_BF16_16x64) {
        LINEAR_EXECUTE(jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS8KBlock)
      } else {
        LINEAR_EXECUTE(jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS8KBlock)
      }
    }
  }

  if (q_bit_verbose) {
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e6;
    std::cout << "m: " << m << " n: " << n << " k: " << k << " with_bias: " << need_bias
              << " cmp_type: " << cmp_type[wtmp->mType] << " time: " << time << "ms" << std::endl;
  }
  delete wtmp;
}