#include <ipex.h>
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

void xetla_linear_fp16(sycl::queue queue, fp16 *A, fp16 *B, fp16 *C,
                       uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k);

static void gbits_linear(const torch::Tensor &activation,
                         const torch::Tensor weight, const torch::Tensor &bias,
                         torch::Tensor &output, int64_t ldo, bool with_bias,
                         const std::string &compute_type,
                         const std::string &weight_type) {
  // Turn on the profiling property to facilitate subsequent profiling
  sycl::property_list properties{};

  // Define SYCL queue
  auto queue = sycl::queue(properties);
  // torch::Tensor revert_weight;
  // if (compute_type == "fp32")
  //   revert_weight = torch::zeros(activation.sizes()[1] * ldo,
  //   torch::kFloat32);
  // else
  //   revert_weight = torch::zeros(activation.sizes()[1] * ldo,
  //   torch::kFloat16);
  // CompressWei4Bit obj(weight.data_ptr<int8_t>());

  // dequant_dispatch(queue, &obj, revert_weight, false, compute_type,
  //                  weight_type);
  // linear_dispatch(queue, activation, revert_weight, bias, output, ldo,
  //                 with_bias, compute_type, weight_type);
  uint32_t matrix_m = activation.sizes()[0];
  uint32_t matrix_n = ldo;
  uint32_t matrix_k = activation.sizes()[1];
  fp16 *A = activation.data_ptr<fp16>();
  fp16 *B = weight.data_ptr<fp16>();
  fp16 *C = output.data_ptr<fp16>();
  xetla_linear_fp16(queue, A, B, C, matrix_m, matrix_n, matrix_k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gbits_linear, "gbits_linear forward (XPU)");
}
