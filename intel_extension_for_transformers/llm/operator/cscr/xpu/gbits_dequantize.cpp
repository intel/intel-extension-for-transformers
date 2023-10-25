#include <torch/extension.h>
#include <ipex.h>
#include "utils.hpp"

static void gbits_dequantize(const torch::Tensor compressed_weight,
                             torch::Tensor &dequantize_weight, bool transpose,
                             const std::string &compute_type,
                             const std::string &weight_type) {
  queue q;
  CompressWei4Bit obj(compressed_weight.data_ptr<int8_t>());
  dequant_dispatch(q, &obj, dequantize_weight, transpose, compute_type,
                   weight_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gbits_dequantize, "gbits_dequantize forward (XPU)");
}