#pragma once

#include <torch/script.h>
#include "jblas/jit_blas_weight_compression.h"

using CompType = jblas::prologue::weight_comp::gemm::WeightCompType;
using PackedWeight = jblas::prologue::PackedWeight;

struct MyHash {
  std::size_t operator()(const std::tuple<int, std::string, std::string>& k) const {
    return std::hash<int>()(std::get<0>(k)) ^ (std::hash<std::string>()(std::get<1>(k))) ^
           std::hash<std::string>()(std::get<2>(k));
  }
};

static std::unordered_map<std::tuple<int, std::string, std::string>, CompType, MyHash> NE_FTYPE_MAP = {
    // bits, alg, scale dtype -> weicomptype
    {{4, "sym", "fp32"}, CompType::S4_F32},
    {{4, "sym", "bf16"}, CompType::S4_Bf16},
    {{8, "sym", "fp32"}, CompType::S8_F32}};

#define COMPUTE_DICPATCH(KER) \
  KER kernel;                 \
  packedw = compressWeight<KER>(&kernel, transpose, n, k, Fp32Wei.data_ptr<float>(), blocksize, type);

#define BIT4_QUANTIZE(NAME, AMX_INT8_KER, VNNI_INT8_KER, AMX_BF16_KER, AVX512F_FP32_KER)                 \
  auto NAME = [&] {                                                                                      \
    if (compute_type == "int8") {                                                                        \
      TORCH_CHECK(check_amx() || check_vnni(), "ISA must lagger than AVX_VNNI when compute_type==int8"); \
      TORCH_CHECK(bits == 4, "quantization bits must be 4 when compute_type==int8");                     \
      if (check_amx()) {                                                                                 \
        jblas::utils::request_perm_xtile_data();                                                         \
        COMPUTE_DICPATCH(AMX_INT8_KER);                                                                  \
      } else {                                                                                           \
        COMPUTE_DICPATCH(VNNI_INT8_KER);                                                                 \
      }                                                                                                  \
    } else {                                                                                             \
      TORCH_CHECK(check_avx512f, "ISA must lagger than AVX_512F when compute_type==fp32");               \
      if (check_amx()) {                                                                                 \
        jblas::utils::request_perm_xtile_data();                                                         \
        COMPUTE_DICPATCH(AMX_BF16_KER);                                                                  \
      } else {                                                                                           \
        COMPUTE_DICPATCH(AVX512F_FP32_KER);                                                              \
      }                                                                                                  \
    }                                                                                                    \
  };

torch::Tensor quant_launcher(const torch::Tensor& Fp32Wei, bool transpose, const std::string& alg, int64_t block_size,
                             const std::string& compute_type, const std::string& quant_type);
