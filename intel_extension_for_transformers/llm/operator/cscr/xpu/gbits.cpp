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
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <map>
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

//#include "gblas/esimd_test_utils.hpp"
#include "customop.hpp"

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif


static void gbits_dequantize(const torch::Tensor compressed_weight, torch::Tensor& dequantize_weight, bool transpose,
                             const std::string& compute_type, const std::string& weight_type) {
  queue q;
  CompressWei4Bit obj(compressed_weight.data_ptr<int8_t>());
  dequant_dispatch(q, &obj, dequantize_weight, transpose, compute_type, weight_type);
}


static torch::Tensor gbits_quantize(const torch::Tensor& weight, bool transpose, int64_t block_size,
                                    const std::string& compute_type, const std::string& weight_type) {
  torch::Tensor output = quantize(weight.data_ptr<float>(), weight.sizes()[0], weight.sizes()[1], block_size, transpose, weight_type, compute_type);
  return output;
}

static void gbits_linear(const torch::Tensor& activation, const torch::Tensor weight, const torch::Tensor& bias,
                         torch::Tensor& output, int64_t ldo, bool with_bias, const std::string& compute_type,
                         const std::string& weight_type) {

    // Turn on the profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    // Define SYCL queue
    auto queue = sycl::queue(properties);
    torch::Tensor revert_weight;
    if (compute_type == "fp32")
      revert_weight = torch::zeros(activation.sizes()[1] * ldo, torch::kFloat32);
    else
      revert_weight = torch::zeros(activation.sizes()[1] * ldo, torch::kFloat16);
    CompressWei4Bit obj(weight.data_ptr<int8_t>());

    dequant_dispatch(queue, &obj, revert_weight, false, compute_type, weight_type);
    linear_dispatch(queue, activation, revert_weight, bias, output, ldo, with_bias, compute_type, weight_type);
}

TORCH_LIBRARY(weight_only_gblasop, m) {
  m.def("gbits_linear", &gbits_linear);
  m.def("gbits_quantize", &gbits_quantize);
  m.def("gbits_dequantize", &gbits_dequantize);
}
