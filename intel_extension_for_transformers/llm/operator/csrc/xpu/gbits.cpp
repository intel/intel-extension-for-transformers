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

#include "utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

void* workspace = nullptr;
int64_t workspace_size = 0;

void xetla_linear_fp16_bias(sycl::queue queue, fp16 *A, int8_t *B, fp16 *C,
                            uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k,
                            int dequant_s, fp16 *bias, void *workspace, int64_t workspace_size);

void xetla_linear_fp16(sycl::queue queue, fp16 *A, int8_t *B, fp16 *C,
                       uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k,
                       int dequant_s, void *workspace, int64_t workspace_size);

static void gbits_linear(const torch::Tensor &activation,
                         const torch::Tensor weight, const torch::Tensor &bias,
                         torch::Tensor &output, int64_t ldo, bool with_bias,
                         const std::string &compute_type,
                         const std::string &weight_type, int blksize) {
  // Turn on the profiling property to facilitate subsequent profiling
  sycl::property_list properties{};

  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto queue = xpu::get_queue_from_stream(c10_stream);
  uint32_t matrix_m = activation.sizes()[0];
  uint32_t matrix_n = ldo;
  uint32_t matrix_k = activation.sizes()[1];
  if (initer.verbose) timer.start();
  if (activation.dtype() == torch::kFloat32) {
    std::cout << "Only support fp16 input now, aborting ... "
              << std::endl;
    exit(0);
  } else {
    auto *A = reinterpret_cast<fp16 *>(activation.data_ptr<at::Half>());
    auto *C = reinterpret_cast<fp16 *>(output.data_ptr<at::Half>());
    if (with_bias) {
      auto *D = reinterpret_cast<fp16 *>(bias.data_ptr<at::Half>());
      xetla_linear_fp16_bias(queue, A, weight.data_ptr<int8_t>(), C, matrix_m, matrix_n,
                             matrix_k, blksize, D, workspace, workspace_size);
    } else {
      xetla_linear_fp16(queue, A, weight.data_ptr<int8_t>(), C, matrix_m, matrix_n,
                        matrix_k, blksize, workspace, workspace_size);
    }
  }
  if (initer.verbose) {
    timer.stop();
    std::cout << "GPU linear cost"
          << timer.get_elapsed_time() << "ms"
          << std::endl;
  }
}

static void gbits_dequantize(const torch::Tensor compressed_weight,
                             torch::Tensor &dequantize_weight, bool transpose,
                             const std::string &compute_type,
                             const std::string &weight_type, int blk) {
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto q = xpu::get_queue_from_stream(c10_stream);
  auto context = q.get_info<sycl::info::queue::context>();
  bool isDevicePointer = sycl::get_pointer_type(
    compressed_weight.data_ptr(), context) == sycl::usm::alloc::device;
  if (sycl::get_pointer_type(dequantize_weight.data_ptr(), context) != sycl::usm::alloc::device) {
    std::cout << "gbits_dequantize requires the dequantized weight on xpu" << std::endl;
    exit(0);
  }
  int n = transpose ? dequantize_weight.sizes()[0] : dequantize_weight.sizes()[1];
  int k = transpose ? dequantize_weight.sizes()[1] : dequantize_weight.sizes()[0];
  CompressWei4Bit obj(compressed_weight.data_ptr<int8_t>(), k, n, blk);
  if (initer.verbose) timer.start();
  if (dequantize_weight.dtype() == torch::kFloat32)
    gpu_dequant(q, &obj, dequantize_weight.data_ptr<float>(), transpose,
                       compute_type, weight_type, isDevicePointer);
  else
    gpu_dequant(q, &obj, dequantize_weight.data_ptr<c10::Half>(), transpose,
                       compute_type, weight_type, isDevicePointer);
  if (initer.verbose) {
    timer.stop();
    std::cout << "GPU dequant cost"
          << timer.get_elapsed_time() << "ms"
          << std::endl;
  }
}

template <typename T>
torch::Tensor quantize(T *weight, int k, int n, int blksize,
                       std::string weight_type, std::string cmpt_type, bool trans) {
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto q = xpu::get_queue_from_stream(c10_stream);
  auto context = q.get_info<sycl::info::queue::context>();
  bool isDevicePointer = sycl::get_pointer_type(weight, context) == sycl::usm::alloc::device;
  T *host_weight;
  if (isDevicePointer) {
    host_weight = (T *)malloc(k * n * sizeof(T));
    q.memcpy((void *)host_weight, (void *)weight, k * n * sizeof(T)).wait();
  } else {
    host_weight = weight;
  }
  CompressWei4Bit compress_wei(k, n, blksize);
  torch::Tensor ret =
      torch::zeros(compress_wei.get_buf_size(), torch::kInt8);
  if (weight_type == "s4fullrange_scalef32") {
    std::vector<int8_t> s8quant_tmp(k * n);
    c10::Half *scale = reinterpret_cast<c10::Half *>(compress_wei.get_scale_ptr());
    if (trans)
      s8_quant_row_blk(host_weight, s8quant_tmp.data(), k, n, k, n, scale, blksize, trans);
    else
      s8_quant_row_blk(host_weight, s8quant_tmp.data(), k, n, n, n, scale, blksize, trans);
    gblas::int4x2 *wei =
        reinterpret_cast<gblas::int4x2 *>(compress_wei.get_4bit_wei_ptr());
    compress_s8_s4(s8quant_tmp.data(), wei, k, n, n, n);
    compress_wei.serialize(ret.data_ptr<int8_t>());
    if (isDevicePointer) free(host_weight);
  } else {
    assert(0);
  }
  return ret;
}

static torch::Tensor gbits_quantize(const torch::Tensor &weight, bool transpose,
                                    int64_t block_size,
                                    const std::string &compute_type,
                                    const std::string &weight_type) {
  int n = transpose ? weight.sizes()[0] : weight.sizes()[1];
  int k = transpose ? weight.sizes()[1] : weight.sizes()[0];
  torch::Tensor output;
  if (k < block_size) {
    if (initer.verbose)
      std::cout << "blocksize is smaller than k, take k as blocksize "
                << std::endl;
    block_size = k;
  }
  if (block_size % 16 != 0) {
      std::cout << "blocksize must be divisible by 16 "
                << std::endl;
      exit(0);
  }
  if (weight.is_meta()) {
    output = torch::zeros(k * n / 2 + k / block_size * n * sizeof(fp16), torch::kInt8);
    return output;
  }
  if (initer.verbose) timer.start();

  if (weight.dtype() == torch::kFloat32) {
    output = quantize(weight.data_ptr<float>(), k, n, block_size, weight_type,
                      compute_type, transpose);
  } else {
    output = quantize(weight.data_ptr<c10::Half>(), k, n, block_size, weight_type,
                      compute_type, transpose);
  }

  if (initer.verbose) {
    timer.stop();
    std::cout << "GPU quant cost"
          << timer.get_elapsed_time() << "ms"
          << std::endl;
  }
  return output;
}

static void set_workspace(torch::Tensor& workspace_tensor) {
  workspace = workspace_tensor.data_ptr();
  workspace_size = workspace_tensor.element_size() * workspace_tensor.numel();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &gbits_linear, "gbits_linear forward (XPU)");
  m.def("quantize", &gbits_quantize, "gbits_quantize forward (XPU)");
  m.def("dequantize", &gbits_dequantize, "gbits_dequantize forward (XPU)");
  m.def("set_workspace", &set_workspace, "set_workspace (XPU)");
}
