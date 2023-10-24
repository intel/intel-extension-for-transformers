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

#include "gblas/esimd_test_utils.hpp"
#include "customop.hpp"
//#include "xetla.hpp"
//#include "tests/utils/utils.hpp"

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

//static void gbits_linear(const torch::Tensor& activation, const torch::Tensor weight, const torch::Tensor& bias,
//                         torch::Tensor& output, int64_t ldo, bool with_bias, const std::string& compute_type,
//                         const std::string& weight_type) {
//
//    // GEMM input size
//    uint32_t matrix_m = activation.sizes()[0];
//    uint32_t matrix_n = ldo;
//    uint32_t matrix_k = activation.sizes()[1];
//
//    uint32_t size_a = matrix_m * matrix_k;
//    uint32_t size_b = matrix_k * matrix_n;
//    uint32_t size_c = matrix_m * matrix_n;
//
//    using data_type_a = bf16;
//    using data_type_b = bf16;
//    using data_type_c = bf16;
//    using data_type_acc = float;
//
//    // Turn on the profiling property to facilitate subsequent profiling
//    sycl::property_list properties {sycl::property::queue::enable_profiling()};
//
//    // Define SYCL queue, context and device
//    auto queue = sycl::queue(properties);
//    auto context = queue.get_info<info::queue::context>();
//    auto device = queue.get_info<info::queue::device>();
//
//    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";
//
//    torch::Tensor revert_weight = torch::zeros(size_b, torch::kFloat16);
//    CompressWei4Bit obj(weight.data_ptr<int8_t>());
//    dequant_dispatch(queue, &obj, revert_weight, false, compute_type, weight_type);
//
//    //auto A = activation.data_ptr<__bf16>();
//    //auto B = revert_weight.data_ptr<__bf16>();
//    //auto C = output.data_ptr<__bf16>();
//    auto A = alloc_device_and_init<data_type_a>(
//            size_a,
//            [](data_type_a *data, size_t idx) {
//                data[idx] = static_cast<data_type_a>(random_float());
//            },
//            queue, device, context);
//    auto B = alloc_device_and_init<data_type_b>(
//            size_b,
//            [](data_type_b *data, size_t idx) {
//                data[idx] = static_cast<data_type_b>(random_float());
//            },
//            queue, device, context);
//    auto C = alloc_device_and_init<data_type_c>(
//            size_c,
//            [](data_type_c *data, size_t idx) {
//                data[idx] = static_cast<data_type_c>(0.0f);
//            },
//            queue, device, context);
//
//    // Define the shape of workgroup and subgroup
//    // It's tunable parameters based on different input shape and hardware for
//    // better performance
//    constexpr uint32_t wg_tile_m = 256;
//    constexpr uint32_t wg_tile_n = 256;
//    constexpr uint32_t sg_tile_m = 32;
//    constexpr uint32_t sg_tile_n = 64;
//
//    // Workload mapping, linear mapping will be used in the code
//    // Suppose it is divisible.
//    uint32_t group_range_m = matrix_m / wg_tile_m;
//    uint32_t group_range_n = matrix_n / wg_tile_n;
//
//    // Each subgroup will be executed in one hardware thread
//    // Calculate how many threads in a workgroup
//    uint32_t thread_range_m = wg_tile_m / sg_tile_m;
//    uint32_t thread_range_n = wg_tile_n / sg_tile_n;
//
//    // leading dimension
//    uint32_t lda = matrix_k;
//    uint32_t ldb = matrix_n;
//    uint32_t ldc = matrix_n;
//
//    // Ndrange and workgroup shape
//    cl::sycl::range<3> group_range {1, group_range_m, group_range_n};
//    cl::sycl::range<3> local_range {1, thread_range_m, thread_range_n};
//
//    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);
//
//    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
//
//    auto gpu_event = queue.submit([&](handler &cgh) {
//        // GPU kernel
//        cgh.parallel_for(nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
//            using namespace gpu::xetla;
//            using namespace gpu::xetla::group;
//            using namespace gpu::xetla::kernel;
//            using namespace gpu::xetla::subgroup;
//
//            // wrap the nd_range to XeTLA range
//            xetla_exec_item<3> ei(item);
//
//            // Step 1: basic computation information
//            // define A, B and accumulator datatype
//            // Using float as accumuator for better accuracy
//            using compute_attr = compute_attr_t<data_type_a, data_type_b,
//                    data_type_acc>;
//
//            // Performance tuning setting based on different shapes
//            static constexpr uint32_t periodic_sync_interval = 0;
//            static constexpr uint32_t prefetch_distance = 0;
//            // should larger than 8
//            static constexpr uint32_t k_stride = 32;
//            using perf_tuning_knob = perf_tuning_knob_t<k_stride,
//                    prefetch_distance, periodic_sync_interval>;
//
//            // specific the computation, performance tuning and computation core
//            using compute_policy
//                    = compute_policy_unaligned_xmx<compute_attr,
//                            perf_tuning_knob, gpu_arch::Xe>;
//
//            // Step 2: define the memory layout & location of input/output
//            // this setting could be used to optimize the data re-use in shared
//            // local memory
//            using mem_desc_input_a = mem_desc_t<data_type_a,
//                    mem_layout::row_major, mem_space::global>;
//            using mem_desc_input_b = mem_desc_t<data_type_b,
//                    mem_layout::row_major, mem_space::global>;
//            using mem_desc_output_c = mem_desc_t<data_type_c,
//                    mem_layout::row_major, mem_space::global>;
//
//            // Step 3: define mirco-kernel's configuration
//            using tile_shape = tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n,
//                    sg_tile_m>;
//            using gemm_t = gemm_t<compute_policy, tile_shape,
//                    mem_desc_input_a, mem_desc_input_b>;
//            gemm_t gemm;
//
//            // Step 4: epilogue function to overwrite the result
//            using epilogue_t
//                    = epilogue_t<epilogue_policy_unaligned<gpu_arch::Xe>,
//                            tile_shape, mem_desc_output_c>;
//
//            // Step 5: define the shared local memory usages
//            // developers have the responsibility to set
//            // shared loacal memory through XeTLA API
//            static constexpr uint32_t barrier_count = gemm_t::barrier_count;
//            static constexpr uint32_t slm_size = gemm_t::slm_size;
//            xetla_nbarrier_init<barrier_count>();
//            xetla_local_init<slm_size>();
//
//            // Step 6: ecah workgroup gets it individual index to start computation
//            int start_n = ei.get_group(2) * wg_tile_n;
//            int start_m = ei.get_group(1) * wg_tile_m;
//            // no slicing in K direction so start from zero for all WG
//            int start_k = 0;
//
//            // Each workgroup will compute all data in K based on no k_sliciing
//            // The developer can set how much data a subgroup compute by k_stride
//            uint32_t wg_tile_k = matrix_k;
//            uint32_t inner_loop_count = wg_tile_k / k_stride;
//
//            // Step 7: define the workgroup start point for each workgroup
//            mem_desc_input_a md_a(
//                    {A}, {matrix_k, matrix_m, lda}, {start_k, start_m});
//            mem_desc_input_b md_b(
//                    {B}, {matrix_n, matrix_k, ldb}, {start_n, start_k});
//            mem_desc_output_c md_c(
//                    {C}, {matrix_n, matrix_m, ldc}, {start_n, start_m});
//
//            // Step 8: real calculation with accumulator varibales which suppose
//            // will be in register.
//            gemm_t::matAcc_t matAcc;
//            matAcc.init(0);
//
//            gemm_t::arguments_t gemm_args(md_a, md_b, inner_loop_count);
//
//            // the results is in the matAcc rather than real output C
//            gemm_t::work_group_t g(ei.get_local_linear_id());
//            gemm(g, matAcc, gemm_args);
//
//            // Step 9: write the results from matACC to real output C
//            epilogue_t epilogue;
//            epilogue(g, matAcc, md_c);
//        });
//    });
//    gpu_event.wait();
//}

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
