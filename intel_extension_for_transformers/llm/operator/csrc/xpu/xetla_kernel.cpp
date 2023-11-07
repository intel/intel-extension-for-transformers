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

#include "common.hpp"
#include "xetla.hpp"
#define DEVICE_MEM_ALIGNMENT (64)

static constexpr size_t wg_tile_m = 8;
static constexpr size_t wg_tile_n = 16;
static constexpr size_t sg_tile_m = 8;
static constexpr size_t sg_tile_n = 32;
static constexpr size_t num_buffer = 64;
static constexpr size_t local_kslicing = 4;
static constexpr size_t global_kslicing = 1;
static constexpr uint32_t periodic_sync_interval = 0;
static constexpr uint32_t prefetch_distance = 0;
static constexpr gpu::xetla::mem_layout layout_a =
    gpu::xetla::mem_layout::row_major;
static constexpr gpu::xetla::mem_layout layout_b =
    gpu::xetla::mem_layout::row_major;
using data_type_zero_pt = gpu::xetla::int4x2;
using data_type_scale = gpu::xetla::fp16;
using data_type_acc_in = gpu::xetla::fp16;
using data_type_acc = float;
using data_type_b = gpu::xetla::int4x2;

struct linear_param {
  int dequant_s;
  uint32_t matrix_m;
  uint32_t matrix_n;
  uint32_t matrix_k;
  size_t size_a;
  size_t size_b;
  size_t size_scale_m;
  size_t size_scale_n;
  size_t size_scale;
  size_t size_zero_pt_m;
  size_t size_zero_pt_n;
  size_t size_zero_pt;
  size_t size_c;
  size_t size_d;
  size_t size_acc;
  size_t size_cnt;
  linear_param(uint32_t m, uint32_t n, uint32_t k, int blk) :
    matrix_m(m), matrix_n(n), matrix_k(k), dequant_s(blk) {
    size_a = matrix_m * matrix_k;
    size_b = matrix_k * matrix_n / 2;
    size_d = 1 * matrix_n;
    size_scale_m = matrix_k / dequant_s;
    size_scale_n = matrix_n;
    size_scale = size_scale_m * size_scale_n;
    size_zero_pt_m = matrix_k / dequant_s;
    size_zero_pt_n = matrix_n / 2;
    size_zero_pt = size_zero_pt_m * size_zero_pt_n;
    size_c = matrix_m * matrix_n;
  }
};

template <typename T, int dequant_s, int sg_tile_k>
void xetla_linear(sycl::queue queue, T *A, CompressWei4Bit *B, T *C,
                  uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k) {
  using data_type_a = T;
  using data_type_c = T;
  linear_param p(matrix_m, matrix_n, matrix_k, dequant_s);
  auto context = queue.get_info<sycl::info::queue::context>();
  auto device = queue.get_info<sycl::info::queue::device>();

  if (initer.verbose)
    std::cout << "Running on " << device.get_info<sycl::info::device::name>()
            << "\n";

  using tile_shape = gpu::xetla::group::tile_shape_t<wg_tile_n, wg_tile_m,
                                                     sg_tile_n, sg_tile_m>;
  using mem_desc_a_t =
      gpu::xetla::mem_desc_t<data_type_a, gpu::xetla::mem_layout::row_major,
                             gpu::xetla::mem_space::global>;
  using mem_desc_b_t =
      gpu::xetla::mem_desc_t<data_type_b, gpu::xetla::mem_layout::row_major,
                             gpu::xetla::mem_space::global>;
  using mem_desc_c_t =
      gpu::xetla::mem_desc_t<data_type_c, gpu::xetla::mem_layout::row_major,
                             gpu::xetla::mem_space::global>;

  using compute_attr =
      gpu::xetla::group::compute_attr_t<data_type_acc_in, data_type_acc_in,
                                        data_type_acc>;
  using perf_tuning_knob =
      gpu::xetla::group::perf_tuning_knob_t<sg_tile_k, prefetch_distance,
                                            periodic_sync_interval>;
  using compute_policy = gpu::xetla::group::compute_policy_bit4_dequantize_xmx<
      compute_attr, perf_tuning_knob,
      gpu::xetla::group::quant_type::S4_FULLRANGE, data_type_scale, dequant_s,
      gpu::xetla::gpu_arch::Arc>;
  using gemm_t = gpu::xetla::group::gemm_t<compute_policy, tile_shape,
                                           mem_desc_a_t, mem_desc_b_t>;
  using epilogue_t = gpu::xetla::group::epilogue_t<
      gpu::xetla::group::epilogue_policy_unaligned<gpu::xetla::gpu_arch::Arc>,
      tile_shape, mem_desc_c_t>;
  using group_swizzle =
      gpu::xetla::kernel::group_swizzle_default<gpu::xetla::gpu_arch::Arc>;
  using gemm_op_t = gpu::xetla::kernel::gemm_universal_t<
      gpu::xetla::kernel::dispatch_policy_int4_dequantize_kslicing<
          group_swizzle, global_kslicing, local_kslicing>,
      gemm_t, epilogue_t>;
  p.size_acc = gemm_op_t::get_acc_buf_size(p.matrix_m, p.matrix_n);
  p.size_cnt = gemm_op_t::get_cnt_buf_size(p.matrix_m, p.matrix_n);

  auto *A_d = static_cast<data_type_a *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_a * sizeof(data_type_a), device, context));
  auto *B_d = static_cast<data_type_b *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_b * sizeof(data_type_b), device, context));
  auto *C_d = static_cast<data_type_c *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_c * sizeof(data_type_c), device, context));
  auto *scale_d = static_cast<data_type_scale *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_scale * sizeof(data_type_scale), device,
      context));
  auto *Acc_d = static_cast<data_type_acc *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_acc * sizeof(data_type_acc), device, context));
  auto *Cnt_d = static_cast<uint32_t *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_cnt * sizeof(uint32_t), device, context));
  queue.memcpy((void *)A_d, (void *)A, p.size_a * sizeof(data_type_a)).wait();
  queue
      .memcpy((void *)B_d, (void *)(B->get_4bit_wei_ptr()),
              p.size_b * sizeof(data_type_b))
      .wait();
  queue.memcpy((void *)C_d, (void *)C, p.size_c * sizeof(data_type_c)).wait();
  queue
      .memcpy((void *)scale_d, (void *)(B->get_scale_ptr()),
              p.size_scale * sizeof(data_type_scale))
      .wait();

  // set up gemm arguments
  typename gemm_op_t::arguments_t gemm_arg(
      p.matrix_m, p.matrix_k, p.matrix_n, A_d,
      p.matrix_k, B_d, p.matrix_n, C_d,
      p.matrix_n, scale_d, p.matrix_n, Acc_d, Cnt_d);
  cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
  if (!gemm_op_t::can_implement(gemm_arg)) {
    std::cout << "The arguments cannot be supported, aborting ... "
              << std::endl;
    exit(0);
  }

  size_t ops = 2 * p.matrix_m * p.matrix_n * p.matrix_k;
  auto e_esimd = queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      // allocate slm and nbarrier resource
      gpu::xetla::slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(item, gemm_arg);
    });
  });
  e_esimd.wait();

  queue.memcpy((void *)C, (void *)C_d, p.size_c * sizeof(data_type_c)).wait();
  free(A_d, context);
  free(B_d, context);
  free(C_d, context);
  free(scale_d, context);
  free(Acc_d, context);
  free(Cnt_d, context);
}

template <typename T, int dequant_s, int sg_tile_k>
void xetla_linear_bias(sycl::queue queue, T *A, CompressWei4Bit *B, T *C,
                  uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k,
                  float *D) {
  using data_type_a = T;
  using data_type_c = T;
  linear_param p(matrix_m, matrix_n, matrix_k, dequant_s);
  auto context = queue.get_info<sycl::info::queue::context>();
  auto device = queue.get_info<sycl::info::queue::device>();

  if (initer.verbose)
    std::cout << "Running on " << device.get_info<sycl::info::device::name>()
            << "\n";

  using tile_shape = gpu::xetla::group::tile_shape_t<wg_tile_n, wg_tile_m,
                                                     sg_tile_n, sg_tile_m>;


  using mem_desc_a_t =
      gpu::xetla::mem_desc_t<data_type_a, gpu::xetla::mem_layout::row_major,
                             gpu::xetla::mem_space::global>;
  using mem_desc_b_t =
      gpu::xetla::mem_desc_t<data_type_b, gpu::xetla::mem_layout::row_major,
                             gpu::xetla::mem_space::global>;
  using mem_desc_c_t =
      gpu::xetla::mem_desc_t<data_type_c, gpu::xetla::mem_layout::row_major,
                             gpu::xetla::mem_space::global>;

  using compute_attr =
      gpu::xetla::group::compute_attr_t<data_type_acc_in, data_type_acc_in,
                                        data_type_acc>;
  using perf_tuning_knob =
      gpu::xetla::group::perf_tuning_knob_t<sg_tile_k, prefetch_distance,
                                            periodic_sync_interval>;
  using compute_policy = gpu::xetla::group::compute_policy_bit4_dequantize_xmx<
      compute_attr, perf_tuning_knob,
      gpu::xetla::group::quant_type::S4_FULLRANGE, data_type_scale, dequant_s,
      gpu::xetla::gpu_arch::Arc>;
  using gemm_t = gpu::xetla::group::gemm_t<compute_policy, tile_shape,
                                           mem_desc_a_t, mem_desc_b_t>;
  using bias_op_t =
      gpu::xetla::subgroup::bias_add_op_t<float, gpu::xetla::gpu_arch::Arc>;
  using tile_op_t =
      gpu::xetla::subgroup::chained_tile_op_t<bias_op_t>;
  using bias_epilogue_policy_t = gpu::xetla::group::epilogue_policy_tile_op<tile_op_t,
      gpu::xetla::gpu_arch::Arc>;
  using epilogue_t = gpu::xetla::group::epilogue_t<
      gpu::xetla::group::epilogue_policy_tile_op<tile_op_t, gpu::xetla::gpu_arch::Arc>,
      tile_shape, mem_desc_c_t>;
  using group_swizzle =
      gpu::xetla::kernel::group_swizzle_default<gpu::xetla::gpu_arch::Arc>;
  using gemm_op_t = gpu::xetla::kernel::gemm_universal_t<
      gpu::xetla::kernel::dispatch_policy_int4_dequantize_kslicing<
          group_swizzle, global_kslicing, local_kslicing>,
      gemm_t, epilogue_t>;

  p.size_acc = gemm_op_t::get_acc_buf_size(p.matrix_m, p.matrix_n);
  p.size_cnt = gemm_op_t::get_cnt_buf_size(p.matrix_m, p.matrix_n);
  
  auto *A_d = static_cast<data_type_a *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_a * sizeof(data_type_a), device, context));
  auto *B_d = static_cast<data_type_b *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_b * sizeof(data_type_b), device, context));
  auto *C_d = static_cast<data_type_c *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_c * sizeof(data_type_c), device, context));
  auto *scale_d = static_cast<data_type_scale *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_scale * sizeof(data_type_scale), device,
      context));
  auto *D_d = static_cast<data_type_acc *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_d * sizeof(data_type_acc), device, context));
  auto *Acc_d = static_cast<data_type_acc *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_acc * sizeof(data_type_acc), device, context));
  auto *Cnt_d = static_cast<uint32_t *>(aligned_alloc_device(
      DEVICE_MEM_ALIGNMENT, p.size_cnt * sizeof(uint32_t), device, context));
  queue.memcpy((void *)A_d, (void *)A, p.size_a * sizeof(data_type_a)).wait();
  queue
      .memcpy((void *)B_d, (void *)(B->get_4bit_wei_ptr()),
              p.size_b * sizeof(data_type_b))
      .wait();
  queue.memcpy((void *)C_d, (void *)C, p.size_c * sizeof(data_type_c)).wait();
  queue
      .memcpy((void *)scale_d, (void *)(B->get_scale_ptr()),
              p.size_scale * sizeof(data_type_scale))
      .wait();
  queue.memcpy((void *)D_d, (void *)D, p.size_d * sizeof(data_type_acc)).wait();

  bias_op_t::shape_t bias_add_shape(p.matrix_n, 1, p.matrix_n);
  using epilogue_args_t = epilogue_t::arguments_t;  
  epilogue_args_t ecpilogue_args({{D_d, bias_add_shape}});

  // set up gemm arguments
  typename gemm_op_t::arguments_t gemm_arg(
      p.matrix_m, p.matrix_k, p.matrix_n, A_d,
      p.matrix_k, B_d, p.matrix_n, C_d,
      p.matrix_n, scale_d, p.matrix_n, Acc_d, Cnt_d, ecpilogue_args);
  cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
  if (!gemm_op_t::can_implement(gemm_arg)) {
    std::cout << "The arguments cannot be supported, aborting ... "
              << std::endl;
    exit(0);
  }

  size_t ops = 2 * p.matrix_m * p.matrix_n * p.matrix_k;
  auto e_esimd = queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      // allocate slm and nbarrier resource
      gpu::xetla::slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(item, gemm_arg);
    });
  });
  e_esimd.wait();

  queue.memcpy((void *)C, (void *)C_d, p.size_c * sizeof(data_type_c)).wait();
  free(A_d, context);
  free(B_d, context);
  free(C_d, context);
  free(D_d, context);
  free(scale_d, context);
  free(Acc_d, context);
  free(Cnt_d, context);
}

template <typename T>
void xetla_linear_base(sycl::queue queue, T *A, CompressWei4Bit *B,
                       T *C, uint32_t matrix_m, uint32_t matrix_n,
                       uint32_t matrix_k, int dequant_s) {
  switch (dequant_s) {
    case 16:
      return xetla_linear<T, 16, 16>(queue, A, B, C, matrix_m, matrix_n, matrix_k);
    case 32:
      return xetla_linear<T, 32, 32>(queue, A, B, C, matrix_m, matrix_n, matrix_k);
    case 64:
      return xetla_linear<T, 64, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k);
    case 128:
      return xetla_linear<T, 128, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k);
    case 256:
      return xetla_linear<T, 256, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k);
    case 512:
      return xetla_linear<T, 512, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k);
    case 1024:
      return xetla_linear<T, 1024, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k);
    case 2048:
      return xetla_linear<T, 2048, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k);
    default:
      std::cout << "blocksize must be divisible by 16 and in [16, 2048]"
                << std::endl;
      exit(0);
  }
}

template <typename T>
void xetla_linear_bias_base(sycl::queue queue, T *A, CompressWei4Bit *B,
                            T *C, uint32_t matrix_m, uint32_t matrix_n,
                            uint32_t matrix_k, int dequant_s, float *D) {
  switch (dequant_s) {
    case 16:
      return xetla_linear_bias<T, 16, 16>(queue, A, B, C, matrix_m, matrix_n, matrix_k, D);
    case 32:
      return xetla_linear_bias<T, 32, 32>(queue, A, B, C, matrix_m, matrix_n, matrix_k, D);
    case 64:
      return xetla_linear_bias<T, 64, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k, D);
    case 128:
      return xetla_linear_bias<T, 128, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k, D);
    case 256:
      return xetla_linear_bias<T, 256, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k, D);
    case 512:
      return xetla_linear_bias<T, 512, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k, D);
    case 1024:
      return xetla_linear_bias<T, 1024, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k, D);
    case 2048:
      return xetla_linear_bias<T, 2048, 64>(queue, A, B, C, matrix_m, matrix_n, matrix_k, D);
    default:
      std::cout << "blocksize must be divisible by 16 and in [16, 2048]"
                << std::endl;
      exit(0);
  }
}

void xetla_linear_fp16_bias(sycl::queue queue, fp16 *A, CompressWei4Bit *B, fp16 *C,
                            uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k,
                            int dequant_s, float *bias) {
  return xetla_linear_bias_base<fp16>(queue, A, B, C, matrix_m, matrix_n, matrix_k,
                                dequant_s, bias);
}

void xetla_linear_fp32_bias(sycl::queue queue, float *A, CompressWei4Bit *B,
                            float *C, uint32_t matrix_m, uint32_t matrix_n,
                            uint32_t matrix_k, int dequant_s, float *bias) {
  return xetla_linear_bias_base<float>(queue, A, B, C, matrix_m, matrix_n, matrix_k,
                                dequant_s, bias);
}

void xetla_linear_fp16(sycl::queue queue, fp16 *A, CompressWei4Bit *B, fp16 *C,
                       uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k,
                       int dequant_s) {
  return xetla_linear_base<fp16>(queue, A, B, C, matrix_m, matrix_n, matrix_k, dequant_s);
}

void xetla_linear_fp32(sycl::queue queue, float *A, CompressWei4Bit *B,
                       float *C, uint32_t matrix_m, uint32_t matrix_n,
                       uint32_t matrix_k, int dequant_s) {
  if (dequant_s == 16) {
    std::cout << "blocksize must be divisible by 16 and in [32, 2048] for fp32 compute_type" 
              << std::endl;
    exit(0);
  }
  return xetla_linear_base<float>(queue, A, B, C, matrix_m, matrix_n, matrix_k, dequant_s);
}
