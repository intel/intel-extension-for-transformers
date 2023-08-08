#include "jblas_quantize.hpp"

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

template <typename KER>
PackedWeight* computeDispatch(KER* ker, float* fp32wei, bool transpose, const int N, const int K, int blocksize,
                              CompType type) {
  auto cd = jblas::utils::parallel::CpuDevice::getInstance();
  if (cd->AVX512F()) {
    return compressWeight<KER>(ker, transpose, N, K, fp32wei, blocksize, type);
  } else {
    return compressWeight<KER>(ker, transpose, N, K, fp32wei, blocksize, type);
  }
}

torch::Tensor quant_launcher(const torch::Tensor& Fp32Wei, bool transpose, int64_t bits, const std::string& alg,
                             int64_t blocksize, const std::string& compute_type) {
  TORCH_CHECK(compute_type == "int8" || compute_type == "fp32", "unsupported compute_type, must be int8/fp32");
  TORCH_CHECK(alg == "sym", "unsupported alg, only support sym currently.");
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4/8");
  std::string scale_dtype = bits == 4 ? "bf16" : "fp32";
  // std::string scale_dtype = "fp32";
  TORCH_CHECK(Fp32Wei.sizes().size() == 2, "dim of weight dosen't meet requirement, must be 2.");
  int k = Fp32Wei.sizes()[0];
  int n = Fp32Wei.sizes()[1];
  if (transpose) {
    int tmp = k;
    k = n;
    n = tmp;
  }
  using S4GemmKernel = jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4KBlock;
  using S4GemmVnniKernel = jblas::wrapper::gemm_default::weight_comp::avx512_vnni::GemmKernelDynamicQuantS4KBlock;
  using S8GemmKernel = jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS8KBlock;
  S4GemmKernel s4kernel;
  S8GemmKernel s8kernel;
  S4GemmVnniKernel vnnikernel;
  jblas::prologue::PackedWeight* packedw = NULL;
  auto type = NE_FTYPE_MAP[std::make_tuple(bits, alg, scale_dtype)];
  if (compute_type == "int8") {
    TORCH_CHECK(bits == 4, "only support 4bit quant when compute_type==int8");
    packedw =
        computeDispatch<S4GemmVnniKernel>(&vnnikernel, Fp32Wei.data_ptr<float>(), transpose, n, k, blocksize, type);
  } else {
    if (bits == 4) {
      packedw = computeDispatch<S4GemmKernel>(&s4kernel, Fp32Wei.data_ptr<float>(), transpose, n, k, blocksize, type);
    } else {
      packedw = computeDispatch<S8GemmKernel>(&s8kernel, Fp32Wei.data_ptr<float>(), transpose, n, k, blocksize, type);
    }
  }
  auto tsize = packedw->getSerializedSize();
  torch::Tensor output = torch::zeros(tsize, torch::kInt8);
  packedw->serializeToBuffer(output.data_ptr<int8_t>());
  delete packedw;
  return output;
}
