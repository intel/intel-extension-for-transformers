#include <torch/extension.h>
#include <ipex.h>
#include "xetla.hpp"
#include "tests/utils/utils.hpp"
#include "utils.hpp"

static void gbits_linear(const torch::Tensor &activation,
                         const torch::Tensor weight, const torch::Tensor &bias,
                         torch::Tensor &output, int64_t ldo, bool with_bias,
                         const std::string &compute_type,
                         const std::string &weight_type) {

  // Turn on the profiling property to facilitate subsequent profiling
  sycl::property_list properties{sycl::property::queue::enable_profiling()};

  // Define SYCL queue
  auto queue = sycl::queue(properties);
  torch::Tensor revert_weight;
  if (compute_type == "fp32")
    revert_weight = torch::zeros(activation.sizes()[1] * ldo, torch::kFloat32);
  else
    revert_weight = torch::zeros(activation.sizes()[1] * ldo, torch::kFloat16);
  CompressWei4Bit obj(weight.data_ptr<int8_t>());

  dequant_dispatch(queue, &obj, revert_weight, false, compute_type,
                   weight_type);
  linear_dispatch(queue, activation, revert_weight, bias, output, ldo,
                  with_bias, compute_type, weight_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gbits_linear, "gbits_linear forward (XPU)");
}