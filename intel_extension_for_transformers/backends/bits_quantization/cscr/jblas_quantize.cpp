#include "jblas_quantize.hpp"

#include <torch/script.h>

#include "jblas/jit_blas_weight_compression.h"

using CompType = jblas::prologue::weight_comp::gemm::WeightCompType;

struct quant_params {
  int nthread = 1;
  int32_t bits = 4;
  std::string alg = "sym";
  int32_t block_size = 32;
  std::string scale_dtype = "fp32";
  std::string gemm_isa = "none";
};

struct MyHash {
  std::size_t operator()(
      const std::tuple<int, std::string, std::string>& k) const {
    return std::hash<int>()(std::get<0>(k)) ^
           (std::hash<std::string>()(std::get<1>(k))) ^
           std::hash<std::string>()(std::get<2>(k));
  }
};

static std::unordered_map<
    std::tuple<int, std::string, std::string>,
    //   jblas::prologue::weight_comp::gemm::WeightCompType,
    CompType, MyHash>
    NE_FTYPE_MAP = {
        // bits, alg, scale dtype -> weicomptype
        {{4, "sym", "fp32"}, CompType::S4_F32},
        {{4, "sym", "bf16"}, CompType::S4_Bf16},
        {{8, "sym", "fp32"}, CompType::S8_F32}};

torch::Tensor quant_launcher(const torch::Tensor& Fp32Wei, int64_t nthread,
                             int64_t bits, const std::string& alg,
                             int64_t block_size, const std::string& scale_dtype,
                             const std::string& gemm_isa) {
  TORCH_CHECK(alg == "sym", "unsupported alg, only support sym currently.");
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4/8");
  TORCH_CHECK(scale_dtype == "fp32" || scale_dtype == "bf16",
              "scale_dtype must be fp32/bf16");
  TORCH_CHECK(Fp32Wei.sizes().size() == 2,
              "dim of weight dosen't meet requirement, must be 2.");
  int k_ = Fp32Wei.sizes()[0];
  int n_ = Fp32Wei.sizes()[1];
  using GemmKernel =
      jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4KBlock;
  using GemmVnniKernel = jblas::wrapper::gemm_default::weight_comp::
      avx512_vnni::GemmKernelDynamicQuantS4KBlock;
  GemmKernel kernel;
  GemmVnniKernel vnnikernel;
  jblas::prologue::PackedWeight* packedw = NULL;

  auto cd = jblas::utils::parallel::CpuDevice::getInstance();
  auto type = NE_FTYPE_MAP[std::make_tuple(bits, alg, scale_dtype)];
  if (gemm_isa == "vnni") {
    if (cd->AVX512F()) {
      packedw =
          vnnikernel.getWeightPtr()->compressWeightTranspose<JblasAVX512F>(
              n_, k_, Fp32Wei.data_ptr<float>(), k_, block_size, type);
    } else {
      packedw = vnnikernel.getWeightPtr()->compressWeightTranspose<JblasNoSIMD>(
          n_, k_, Fp32Wei.data_ptr<float>(), k_, block_size, type);
    }
  } else {
    if (cd->AVX512F()) {
      packedw = kernel.getWeightPtr()->compressWeightTranspose<JblasAVX512F>(
          n_, k_, Fp32Wei.data_ptr<float>(), k_, block_size, type);
    } else {
      packedw = kernel.getWeightPtr()->compressWeightTranspose<JblasNoSIMD>(
          n_, k_, Fp32Wei.data_ptr<float>(), k_, block_size, type);
    }
  }

  auto tsize = packedw->getSerializedSize();
  torch::Tensor output = torch::zeros(tsize, torch::kInt8);
  packedw->serializeToBuffer(output.data_ptr<int8_t>());
  delete packedw;
  return output;
}