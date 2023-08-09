#include "jblas_quantize.hpp"
#include <torch/script.h>

bool check_amx() { return jblas::utils::parallel::CpuDevice::getInstance()->AMX_BF16(); }
bool check_vnni() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX_VNNI(); }
bool check_avx512f() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX512F(); }

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

torch::Tensor quant_launcher(const torch::Tensor& Fp32Wei, bool transpose, const std::string& alg, int64_t blocksize,
                             const std::string& compute_type, const std::string& quant_type) {
  TORCH_CHECK(compute_type == "int8" || compute_type == "fp32", "unsupported compute_type, must be int8/fp32");
  TORCH_CHECK(alg == "sym", "unsupported alg, only support sym currently.");
  TORCH_CHECK(quant_type == "s8" || quant_type == "s4_clip" || quant_type == "s4_fullrange", "unsupported quant_type.");
  TORCH_CHECK(Fp32Wei.sizes().size() == 2, "dim of weight dosen't meet requirement, must be 2.");
  std::string scale_dtype = quant_type == "s8" ? "fp32" : "bf16";
  int bits = quant_type == "s8" ? 8 : 4;
  int k = Fp32Wei.sizes()[0];
  int n = Fp32Wei.sizes()[1];
  if (transpose) {
    int tmp = k;
    k = n;
    n = tmp;
  }
  jblas::prologue::PackedWeight* packedw = NULL;
  auto type = NE_FTYPE_MAP[std::make_tuple(bits, alg, scale_dtype)];

  auto process_s8_quantize = [&] {
    TORCH_CHECK(compute_type == "fp32", "compute_type must be fp32 when execute s8-linear.");
    TORCH_CHECK(check_avx512f(), "ISA must lagger than AVX_512F when compute_type==fp32");
    if (check_amx()) {
      jblas::utils::request_perm_xtile_data();
      COMPUTE_DICPATCH(jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS8KBlock);
    } else {
      COMPUTE_DICPATCH(jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS8KBlock);
    }
  };

  BIT4_QUANTIZE(process_s4_clip_quantize,
                jblas::wrapper::gemm_default::weight_comp::amx_int8::GemmSKernelDynamicS4ClipKBlock,
                jblas::wrapper::gemm_default::weight_comp::avx512_vnni::GemmSKernelDynamicS4ClipKBlock,
                jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS4ClipKBlock,
                jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4ClipKBlock)

  BIT4_QUANTIZE(process_s4_fullrange_quantize,
                jblas::wrapper::gemm_default::weight_comp::amx_int8::GemmSKernelDynamicS4FullRangeKBlock,
                jblas::wrapper::gemm_default::weight_comp::avx512_vnni::GemmSKernelDynamicS4FullRangeKBlock,
                jblas::wrapper::gemm_default::weight_comp::amx_bf16::GemmKernelS4FullRangeKBlock,
                jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4FullRangeKBlock)

  if (quant_type == "s8") process_s8_quantize();
  if (quant_type == "s4_clip") process_s4_clip_quantize();
  if (quant_type == "s4_fullrange") process_s4_fullrange_quantize();

  auto tsize = packedw->getSerializedSize();
  torch::Tensor output = torch::zeros(tsize, torch::kInt8);
  packedw->serializeToBuffer(output.data_ptr<int8_t>());
  delete packedw;
  return output;
}
