#include <torch/extension.h>
#include <ipex.h>
#include "utils.hpp"

torch::Tensor quantize(float* weight, int k, int n, int blksize, bool transpose, std::string weight_type,
               std::string cmpt_type) {
  CompressWei4Bit compress_wei(k, n, blksize);
  torch::Tensor ret = torch::zeros(compress_wei.get_serialize_size(), torch::kInt8);
  //void* ret = malloc(compress_wei.get_serialize_size());
  assert(!transpose);
  if (weight_type == "s4fullrange_scalef32") {
    std::vector<int8_t> s8quant_tmp(k * n);
    float* scale = reinterpret_cast<float*>(compress_wei.get_scale_ptr());
    s8_quant_row_blk(weight, s8quant_tmp.data(), k, n, n, n, scale, blksize);
    int4x2* wei = reinterpret_cast<int4x2*>(compress_wei.get_4bit_wei_ptr());
    compress_s8_s4(s8quant_tmp.data(), wei, k, n, n, n);
    compress_wei.serialize(ret.data_ptr<int8_t>());
  } else {
    assert(0);
  }
  return ret;
}

static torch::Tensor gbits_quantize(const torch::Tensor &weight, bool transpose,
                                    int64_t block_size,
                                    const std::string &compute_type,
                                    const std::string &weight_type) {
  torch::Tensor output =
      quantize(weight.data_ptr<float>(), weight.sizes()[0], weight.sizes()[1],
               block_size, transpose, weight_type, compute_type);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gbits_quantize, "gbits_quantize forward (XPU)");
}
