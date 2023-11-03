//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

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
  gpu_dequant(q, &obj, dequantize_weight.data_ptr<float>(), transpose,
                     compute_type, weight_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gbits_dequantize, "gbits_dequantize forward (XPU)");
}
