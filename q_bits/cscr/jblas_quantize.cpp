#include "jblas_quantize.hpp"
#include "jblas_utils.hpp"
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

template <typename KER>
PackedWeight* compressWeight(KER* ker, bool transpose, const int N, const int K, const float* B, int blocksize,
                             CompType type) {
  auto weiptr = ker->getWeightPtr();
  if (transpose) {
    return weiptr->compressWeightTranspose(N, K, B, K, blocksize, type);
  } else {
    return weiptr->compressWeight(N, K, B, N, blocksize, type);
  }
}

torch::Tensor quant_launcher(const torch::Tensor& Fp32Wei, bool transpose, int64_t bits, const std::string& alg,
                             int64_t blocksize, const std::string& compute_type) {
  TORCH_CHECK(compute_type == "int8" || compute_type == "fp32", "unsupported compute_type, must be int8/fp32");
  TORCH_CHECK(alg == "sym", "unsupported alg, only support sym currently.");
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4/8");
  std::string scale_dtype = bits == 4 ? "bf16" : "fp32";
  TORCH_CHECK(Fp32Wei.sizes().size() == 2, "dim of weight dosen't meet requirement, must be 2.");
  int k = Fp32Wei.sizes()[0];
  int n = Fp32Wei.sizes()[1];
  if (transpose) {
    int tmp = k;
    k = n;
    n = tmp;
  }
  jblas::prologue::PackedWeight* packedw = NULL;
  auto type = NE_FTYPE_MAP[std::make_tuple(bits, alg, scale_dtype)];
  if (compute_type == "int8") {
    TORCH_CHECK(check_amx() || check_vnni(), "ISA must lagger than AVX_VNNI when compute_type==int8");
    TORCH_CHECK(bits == 4, "quantization bits must be 4 when compute_type==int8");
    if (check_amx()) {
      jblas::utils::request_perm_xtile_data();
      COMPUTE_DICPATCH(jblas::wrapper::gemm_default::weight_comp::amx_int8::GemmSKernelDynamicS4KBlock);
    } else {
      COMPUTE_DICPATCH(jblas::wrapper::gemm_default::weight_comp::avx512_vnni::GemmSKernelDynamicS4KBlock);
    }
  } else {
    TORCH_CHECK(check_avx512f, "ISA must lagger than AVX_512F when compute_type==fp32");
    if (bits == 4) {
      if (check_amx()) {
        jblas::utils::request_perm_xtile_data();
        COMPUTE_DICPATCH(jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS4KBlock);
      } else {
        COMPUTE_DICPATCH(jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4KBlock);
      }
    } else {
      if (check_amx()) {
        jblas::utils::request_perm_xtile_data();
        COMPUTE_DICPATCH(jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS8KBlock);
      } else {
        COMPUTE_DICPATCH(jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS8KBlock);
      }
    }
  }
  auto tsize = packedw->getSerializedSize();
  torch::Tensor output = torch::zeros(tsize, torch::kInt8);
  packedw->serializeToBuffer(output.data_ptr<int8_t>());
  delete packedw;
  return output;
}
