#pragma once

#include <torch/script.h>

static std::vector<std::string> cmp_type = {"UNDEF", "S8_F32", "S4_F32", "S4_BF16"};
static std::vector<std::string> gemm_core_type = {"Undef",
                                                  "AVX2_4X24",
                                                  "AVX512F_8X48",
                                                  "AVX512_VNNI_8X48",
                                                  "AMX_BF16_16x64",
                                                  "AMX_BF16_16x48",
                                                  "AMX_INT8_16x64",
                                                  "AMX_INT8_16x48",
                                                  "AVX512_VNNI_3X48_KBLOCK",
                                                  "AMX_INT8_16X48_KBLOCK",
                                                  "AVX512_FP16_8x64",
                                                  "AVX512_FP16_8x96"};

#define LINEAR_EXECUTE(KER)  \
  static KER kernel;         \
  auto ret = kernel.compute( \
      {m, n, k, activation.data_ptr<float>(), lda, wtmp, output.data_ptr<float>(), bias_ptr, ldo, 0, alpha, beta});

#define BIT4_FULL_CMPTYPE_LINEAR(NAME, AMX_INT8_KER, VNNI_INT8_KER, AMX_BF16_KER, AVX512F_FP32_KER) \
  auto NAME = [&]() {                                                                               \
    if (compute_type == "int8") {                                                                   \
      TORCH_CHECK(bits == 4, "quantization bits must be 4 when compute_type==int8");                \
      if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_INT8_16x48) {                           \
        LINEAR_EXECUTE(AMX_INT8_KER)                                                                \
      } else {                                                                                      \
        LINEAR_EXECUTE(VNNI_INT8_KER)                                                               \
      }                                                                                             \
    } else {                                                                                        \
      if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_BF16_16x64) {                           \
        LINEAR_EXECUTE(AMX_BF16_KER)                                                                \
      } else {                                                                                      \
        LINEAR_EXECUTE(AVX512F_FP32_KER)                                                            \
      }                                                                                             \
    }                                                                                               \
  };

#define BIT4_FP32_CMPTYPE_LINEAR(NAME, AMX_BF16_KER, AVX512F_FP32_KER)                                   \
  auto NAME = [&]() {                                                                                    \
    TORCH_CHECK(compute_type == "fp32", std::string(#NAME) + std::string(" compute_type must be fp32")); \
    if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AMX_BF16_16x64) {                                  \
      LINEAR_EXECUTE(AMX_BF16_KER)                                                                       \
    } else {                                                                                             \
      LINEAR_EXECUTE(AVX512F_FP32_KER)                                                                   \
    }                                                                                                    \
  };

void quantweight_f32_linear_launcher(const torch::Tensor& activation, const torch::Tensor& weight, float* bias_ptr,
                                     torch::Tensor& output, const std::string& compute_type,
                                     const std::string& quant_type, int64_t m, int64_t n, int64_t k, int64_t lda,
                                     int64_t ldo, bool need_bias);