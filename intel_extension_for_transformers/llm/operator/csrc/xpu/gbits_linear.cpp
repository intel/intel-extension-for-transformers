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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif
using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

void xetla_linear_fp16(sycl::queue queue, fp16 *A, CompressWei4Bit *B, fp16 *C,
                       uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k,
                       bool with_bias, float *bias);

void xetla_linear_fp32(sycl::queue queue, float *A, CompressWei4Bit *B,
                       float *C, uint32_t matrix_m, uint32_t matrix_n,
                       uint32_t matrix_k, bool with_bias, float *bias);

template <typename DST_T>
void gpu_dequant(sycl::queue &q, CompressWei4Bit *compress_wei,
                 DST_T *dequant_weight, bool transpose,
                 const std::string &compute_type,
                 const std::string &weight_type);

static void gbits_linear(const torch::Tensor &activation,
                         const torch::Tensor weight, const torch::Tensor &bias,
                         torch::Tensor &output, int64_t ldo, bool with_bias,
                         const std::string &compute_type,
                         const std::string &weight_type) {
  // Turn on the profiling property to facilitate subsequent profiling
  sycl::property_list properties{};

  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto queue = xpu::get_queue_from_stream(c10_stream);

  uint32_t matrix_m = activation.sizes()[0];
  uint32_t matrix_n = ldo;
  uint32_t matrix_k = activation.sizes()[1];

  torch::Tensor revert_weight;
  CompressWei4Bit obj(weight.data_ptr<int8_t>());
  if (compute_type == "fp32") {
    auto *A = reinterpret_cast<float *>(activation.data_ptr<float>());
    auto *C = reinterpret_cast<float *>(output.data_ptr<float>());
    auto *D = reinterpret_cast<float *>(bias.data_ptr<float>());
    xetla_linear_fp32(queue, A, &obj, C, matrix_m, matrix_n, matrix_k,
                      with_bias, D);
  } else {
    auto *A = reinterpret_cast<fp16 *>(activation.data_ptr<at::Half>());
    auto *C = reinterpret_cast<fp16 *>(output.data_ptr<at::Half>());
    auto *D = reinterpret_cast<float *>(bias.data_ptr<float>());
    xetla_linear_fp16(queue, A, &obj, C, matrix_m, matrix_n, matrix_k,
                      with_bias, D);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gbits_linear, "gbits_linear forward (XPU)");
}
