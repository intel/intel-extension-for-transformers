#include "dequant_utils.hpp"

static void gbits_dequantize(const torch::Tensor compressed_weight,
                             torch::Tensor &dequantize_weight, bool transpose,
                             const std::string &compute_type,
                             const std::string &weight_type) {
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto q = xpu::get_queue_from_stream(c10_stream);

  CompressWei4Bit obj(compressed_weight.data_ptr<int8_t>());
  if (compute_type == "fp32") {
    gpu_dequant<float>(q, &obj, dequantize_weight.data_ptr<float>(),
                       transpose, compute_type, weight_type);
  }
  else {
    gpu_dequant<at::Half>(q, &obj, dequantize_weight.data_ptr<at::Half>(),
                        transpose, compute_type, weight_type);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gbits_dequantize, "gbits_dequantize forward (XPU)");
}