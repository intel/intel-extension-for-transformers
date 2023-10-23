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
#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

#define JJ 8
#define II 2
#define KK 1
#define KKK 8
#define JJJ 8
#define III 32

static void gbits_dequantize(const torch::Tensor compressed_weight, torch::Tensor& dequantize_weight, bool transpose,
                             const std::string& compute_type, const std::string& weight_type, int64_t blksize) {
  queue q;
  int K = dequantize_weight.sizes()[0];
  int N = dequantize_weight.sizes()[1];
  CompressWei4Bit obj(K, N, blksize, false);
  obj.deserialize(compressed_weight.data_ptr<int8_t>());
  gpu_dequant(q, &obj, dequantize_weight.data_ptr<float>(), transpose, compute_type, weight_type);
}


static torch::Tensor gbits_quantize(const torch::Tensor& weight, bool transpose, int64_t block_size,
                                    const std::string& compute_type, const std::string& weight_type) {
  torch::Tensor output = quantize(weight.data_ptr<float>(), weight.sizes()[0], weight.sizes()[1], block_size, transpose, weight_type, compute_type);
  return output;
}

static void gbits_linear(const torch::Tensor& activation, const torch::Tensor weight, const torch::Tensor& bias,
                         torch::Tensor& output, int64_t ldo, bool with_bias, const std::string& compute_type,
                         const std::string& weight_type, int64_t blksize) {
  sycl::property_list properties {sycl::property::queue::enable_profiling()};
  auto q = sycl::queue(properties);
  unsigned long TOTAL_I = activation.sizes()[0];
  unsigned long TOTAL_J = ldo;
  unsigned long TOTAL_K = activation.sizes()[1];

  // dequant
  torch::Tensor revert_weight = torch::zeros(TOTAL_K * TOTAL_J, torch::kFloat32);
  CompressWei4Bit obj(TOTAL_K, TOTAL_J, blksize, false);
  obj.deserialize(weight.data_ptr<int8_t>());
  gpu_dequant(q, &obj, revert_weight.data_ptr<float>(), false, compute_type, weight_type);
  //std::cout << "finish dequant " << "\n";

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  unsigned long I = TOTAL_I / (II * III);
  unsigned long J = TOTAL_J / (JJ * JJJ);
  range<2> GlobalRange(JJ * J, II * I);
  range<2> LocalRange(JJ, II);
  sycl::image<2> imgA(activation.data_ptr<float>(), sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, sycl::range<2>{TOTAL_K / 4, TOTAL_I});
  sycl::image<2> imgB(revert_weight.data_ptr<float>(), sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, sycl::range<2>{TOTAL_J / 4, TOTAL_K});
  sycl::image<2> imgC(output.data_ptr<float>(), sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, sycl::range<2>{TOTAL_J / 4, TOTAL_I});

  auto e = q.submit([&](handler &cgh) {
  auto A = imgA.get_access<uint4, access::mode::read>(cgh);
  auto B = imgB.get_access<uint4, access::mode::read>(cgh);
  auto _Out = imgC.get_access<uint4, access::mode::write>(cgh);
  cgh.parallel_for<class Test>(
      nd_range{GlobalRange, LocalRange},
      [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
        static const CONSTANT char FMT[] = "58: %d,59:%d\n";
        static const CONSTANT char FMT1[] = "pos=%d,value=%f\n";
        static const CONSTANT char FMT2[] = "error\n";
        using namespace sycl::ext::intel::esimd;
        int _A_extent_0 = TOTAL_K;
        int _B_extent_0 = TOTAL_J;
        int _Out_extent_0 = TOTAL_J;
        int _X_s0_i___block_id_y = ndi.get_group(1);
        int _X_s0_j___block_id_x = ndi.get_group(0);
        int ___thread_id_y = ndi.get_local_id(1);
        int ___thread_id_x = ndi.get_local_id(0);
        simd<float, 256> _Z;
        simd<float, 256> _Y;
        simd<float, 256> _X;
        simd<float, 256> _A_im_buf;
        simd<float, 64> _B_im_buf;
        _Z.select<256, 1>(0) = 0.000000;
        for (int _X_s0_k = 0; _X_s0_k < 0 + (_A_extent_0 / 8); _X_s0_k++) {
          int _54 = (_X_s0_k * 8);
          int _55 = (((_X_s0_j___block_id_x * 8) + ___thread_id_x) * 8);
          _B_im_buf.select<64, 1>(0).bit_cast_view<float, 8, 8>() =
              media_block_load<float, 8, 8>(B, (_55 * 4), (_54 + 0));
          // for (size_t posa = 0; posa < 64; posa++) {
          //   if (_B_im_buf[posa] != 1) {
          //     sycl::ext::oneapi::experimental::printf(FMT2);
          //   }
          // }
          int _56 = (((_X_s0_i___block_id_y * 2) + ___thread_id_y) * 32);
          int _57 = (_X_s0_k * 8);
          _A_im_buf.select<256, 1>(0)
              .bit_cast_view<float, 32, 8>()
              .select<8, 1, 8, 1>(0, 0) =
              media_block_load<float, 8, 8>(A, (_57 * 4), (_56 + 0));
          _A_im_buf.select<256, 1>(0)
              .bit_cast_view<float, 32, 8>()
              .select<8, 1, 8, 1>(8, 0) =
              media_block_load<float, 8, 8>(A, (_57 * 4), (_56 + 8));
          _A_im_buf.select<256, 1>(0)
              .bit_cast_view<float, 32, 8>()
              .select<8, 1, 8, 1>(16, 0) =
              media_block_load<float, 8, 8>(A, (_57 * 4), (_56 + 16));
          _A_im_buf.select<256, 1>(0)
              .bit_cast_view<float, 32, 8>()
              .select<8, 1, 8, 1>(24, 0) =
              media_block_load<float, 8, 8>(A, (_57 * 4), (_56 + 24));
      // for (size_t posa = 0; posa < 256; posa++) {
      //   if (_A_im_buf[posa] != 1) {
      //     sycl::ext::oneapi::experimental::printf(FMT2);
      //   }
      // }

      // for (int posa = 0; posa < 256; posa++) {
      //   sycl::ext::oneapi::experimental::printf(FMT1, posa,
      //   _A_im_buf[posa]);
      // }
#pragma unroll
          for (int _X_s0_iii = 0; _X_s0_iii < 0 + 32; _X_s0_iii++) {
        // tile -> vnni_format
#pragma unroll
            for (int _X_s0_kkk = 0; _X_s0_kkk < 0 + 8; _X_s0_kkk++) {
              _X.select<8, 1>((_X_s0_iii * 8)) =
                  _A_im_buf.select<8, 0>(((_X_s0_iii * 8) + _X_s0_kkk));
              _Y.select<8, 1>((_X_s0_iii * 8)) =
                  _B_im_buf.select<8, 1>((_X_s0_kkk * 8));
              _Z.select<8, 1>((_X_s0_iii * 8)) =
                  (_Z.select<8, 1>((_X_s0_iii * 8)) +
                   (_X.select<8, 1>((_X_s0_iii * 8)) *
                    _Y.select<8, 1>((_X_s0_iii * 8))));
            }  // for _X_s0_kkk
          }    // for _X_s0_iii
        }      // for _X_s0_k
        int _58 = (((_X_s0_i___block_id_y * 2) + ___thread_id_y) * 32);
        int _59 = (((_X_s0_j___block_id_x * 8) + ___thread_id_x) * 8);
        // sycl::ext::oneapi::experimental::printf(FMT, _58, _59);
        // for (int pos = 0; pos < 256; pos++) {
        //   sycl::ext::oneapi::experimental::printf(FMT1, pos, _Z[pos]);
        // }
        // for (size_t posa = 0; posa < 256; posa++) {
        //   if (_Z[posa] != 8) {
        //     sycl::ext::oneapi::experimental::printf(FMT2);
        //   }
        // }
        media_block_store<float, 8, 8>(_Out, (_59 * 4), (_58 + 0),
                                       _Z.select<256, 1>(0)
                                           .bit_cast_view<float, 32, 8>()
                                           .select<8, 1, 8, 1>(0, 0));
        media_block_store<float, 8, 8>(_Out, (_59 * 4), (_58 + 8),
                                       _Z.select<256, 1>(0)
                                           .bit_cast_view<float, 32, 8>()
                                           .select<8, 1, 8, 1>(8, 0));
        media_block_store<float, 8, 8>(_Out, (_59 * 4), (_58 + 16),
                                       _Z.select<256, 1>(0)
                                           .bit_cast_view<float, 32, 8>()
                                           .select<8, 1, 8, 1>(16, 0));
        media_block_store<float, 8, 8>(_Out, (_59 * 4), (_58 + 24),
                                       _Z.select<256, 1>(0)
                                           .bit_cast_view<float, 32, 8>()
                                           .select<8, 1, 8, 1>(24, 0));
      }  // kernel kernel_X
    );
  });
  e.wait();
}


TORCH_LIBRARY(weight_only_gblasop, m) {
  m.def("gbits_linear", &gbits_linear);
  m.def("gbits_quantize", &gbits_quantize);
  m.def("gbits_dequantize", &gbits_dequantize);
}
