#include "xetla.hpp"
#include "common.hpp"
#define DEVICE_MEM_ALIGNMENT (64)

template <typename T>
void xetla_linear(sycl::queue queue, T *A, CompressWei4Bit *B,
                  T *C, uint32_t matrix_m, uint32_t matrix_n,
                  uint32_t matrix_k, bool with_bias, float *D) {
    static constexpr size_t wg_tile_m = 8;
    static constexpr size_t wg_tile_n = 256;
    static constexpr size_t sg_tile_m = 8;
    static constexpr size_t sg_tile_n = 16;
    static constexpr size_t sg_tile_k = 32;
    static constexpr size_t dequant_s = 128;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 1;
    static constexpr size_t global_kslicing = 1;
    static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
    static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
    using data_type_zero_pt = gpu::xetla::int4x2;
    using data_type_scale = gpu::xetla::fp16;
    using data_type_acc_in = gpu::xetla::fp16;
    using data_type_acc = float;
    using data_type_a = T;
    using data_type_b = gpu::xetla::int4x2;
    using data_type_c = T;


    size_t size_a = matrix_m * matrix_k;
    size_t size_b = matrix_k * matrix_n / 2;
    size_t size_d = 1 * matrix_n;

    size_t size_scale_m = matrix_k / dequant_s;
    size_t size_scale_n = matrix_n;
    size_t size_scale = size_scale_m * size_scale_n;

    size_t size_zero_pt_m = matrix_k / dequant_s;
    size_t size_zero_pt_n = matrix_n / 2;
    size_t size_zero_pt = size_zero_pt_m * size_zero_pt_n;

    size_t size_c = matrix_m * matrix_n;
    uint32_t lda = matrix_k;
    uint32_t ldb = matrix_n;
    uint32_t ldc = matrix_n;
    uint32_t ld_scale = size_scale_n;
    uint32_t ld_zero_pt = size_zero_pt_n;

    auto context = queue.get_info<sycl::info::queue::context>();
    auto device = queue.get_info<sycl::info::queue::device>();

    std::cout << "Running on " << device.get_info<sycl::info::device::name>() << "\n";

    using tile_shape = gpu::xetla::group::tile_shape_t<wg_tile_n, wg_tile_m,
            sg_tile_n, sg_tile_m>;
    static constexpr uint32_t periodic_sync_interval = 0;
    static constexpr uint32_t prefetch_distance = 0;

    data_type_acc *D_d = static_cast<data_type_acc *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_d * sizeof(data_type_acc), device, context));
    if (with_bias) {
        queue.memcpy((void *)D_d, (void *)D, size_d * sizeof(data_type_acc)).wait();
    }

    using mem_desc_a_t = gpu::xetla::mem_desc_t<data_type_a, gpu::xetla::mem_layout::row_major,
            gpu::xetla::mem_space::global>;
    using mem_desc_b_t = gpu::xetla::mem_desc_t<data_type_b, gpu::xetla::mem_layout::row_major,
            gpu::xetla::mem_space::global>;
    using mem_desc_c_t = gpu::xetla::mem_desc_t<data_type_c, gpu::xetla::mem_layout::row_major,
            gpu::xetla::mem_space::global>;

    using compute_attr = gpu::xetla::group::compute_attr_t<data_type_acc_in,
            data_type_acc_in, data_type_acc>;
    using perf_tuning_knob = gpu::xetla::group::perf_tuning_knob_t<sg_tile_k,
            prefetch_distance, periodic_sync_interval>;
    using compute_policy
            = gpu::xetla::group::compute_policy_bit4_dequantize_xmx<compute_attr,
                    perf_tuning_knob,
                    gpu::xetla::group::quant_type::S4_FULLRANGE,
                    data_type_scale, dequant_s, gpu::xetla::gpu_arch::Xe>;
    using gemm_t = gpu::xetla::group::gemm_t<compute_policy, tile_shape,
            mem_desc_a_t, mem_desc_b_t>;

    using bias_op_t = gpu::xetla::subgroup::bias_add_op_t<float, gpu::xetla::gpu_arch::Xe>;;
    using tile_op_t = gpu::xetla::subgroup::chained_tile_op_t<
            bias_op_t // apply elementwise BiasAdd
            >;
    using epilogue_t = gpu::xetla::group::epilogue_t<
            gpu::xetla::group::epilogue_policy_unaligned<tile_op_t, gpu::xetla::gpu_arch::Xe>, tile_shape,
            mem_desc_c_t>;
    using group_swizzle = gpu::xetla::kernel::group_swizzle_default<gpu::xetla::gpu_arch::Xe>;
    using gemm_op_t = gpu::xetla::kernel::gemm_universal_t<
            gpu::xetla::kernel::dispatch_policy_int4_dequantize_kslicing<
                    group_swizzle, global_kslicing, local_kslicing>,
            gemm_t, epilogue_t>;
    bias_op_t::shape_t bias_add_shape(matrix_n, 1, matrix_n);
    // [ReLuBias] pass arguments of chained_tile_op_t<> to epilogue_args
    using epilogue_args_t = epilogue_t::arguments_t;
    epilogue_args_t epilogue_args({//epilogue_args init list
            // [ReLuBias] 2. bias_add_op_t
            // It accepts the base pointer to matrix D, and its dimensions
            {D_d, bias_add_shape}});
    size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);

    //Define and initialize the data required for the calculation
    auto *Acc_h = static_cast<data_type_acc *>(
            malloc_host(size_acc * sizeof(data_type_acc), context));
    auto *Cnt_h = static_cast<uint32_t *>(
            malloc_host(size_cnt * sizeof(uint32_t), context));

    for (unsigned i = 0; i < size_acc; ++i) {
        Acc_h[i] = 0;
    }
    for (unsigned i = 0; i < size_cnt; ++i) {
        Cnt_h[i] = 0;
    }

    auto *A_d = static_cast<data_type_a *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_a * sizeof(data_type_a), device, context));
    auto *B_d = static_cast<data_type_b *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_b * sizeof(data_type_b), device, context));
    auto *C_d = static_cast<data_type_c *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_c * sizeof(data_type_c), device, context));
    auto *Acc_d = static_cast<data_type_acc *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_acc * sizeof(data_type_acc), device, context));
    auto *Cnt_d
            = static_cast<uint32_t *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_cnt * sizeof(uint32_t), device, context));
    auto *scale_d = static_cast<data_type_scale *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_scale * sizeof(data_type_scale), device, context));

    queue.memcpy((void *)A_d, (void *)A, size_a * sizeof(data_type_a)).wait();
    queue.memcpy((void *)B_d, (void *)(B->get_4bit_wei_ptr()), size_b * sizeof(data_type_b)).wait();
    queue.memcpy((void *)C_d, (void *)C, size_c * sizeof(data_type_c)).wait();
    queue.memcpy((void *)Acc_d, (void *)Acc_h, size_acc * sizeof(data_type_acc))
            .wait();
    queue.memcpy((void *)Cnt_d, (void *)Cnt_h, size_cnt * sizeof(uint32_t))
            .wait();
    queue.memcpy((void *)scale_d, (void *)(B->get_scale_ptr()),
                 size_scale * sizeof(data_type_scale))
            .wait();

    // set up gemm arguments
    typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A_d,
            matrix_k, B_d, matrix_n, C_d, matrix_n, scale_d, matrix_n, Acc_d,
            Cnt_d, epilogue_args);
    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
    if (!gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        exit(0);
    }

    size_t ops = 2 * matrix_m * matrix_n * matrix_k;
    auto e_esimd = queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
                nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                    // allocate slm and nbarrier resource
                    gpu::xetla::slm_barrier_init<gemm_op_t>();
                    gemm_op_t gemm_op;
                    gemm_op(item, gemm_arg);
                });
    });
    e_esimd.wait();

    std::vector<fp16> dequantize_b(matrix_k * matrix_n, 0);
    for (int i = 0; i < matrix_k / dequant_s; i++) {
        for (int j = 0; j < matrix_n / 2; j++) {
            int start_in = i * dequant_s * matrix_n / 2 + j;
            int start_zero_pt = i * size_zero_pt_n + j;
            int start_out = i * dequant_s * matrix_n + j * 2;
            int start_scale = i * size_scale_n + j * 2;
            for (int ii = 0; ii < dequant_s; ii++) {
                uint8_t data_in = reinterpret_cast<uint8_t *>(B->get_scale_ptr())[start_in + ii * matrix_n / 2];
                int8_t data_0 = int8_t(data_in & 0x0f) - 8;
                int8_t data_1 = int8_t(data_in >> 4) - 8;
                dequantize_b[start_out + ii * matrix_n]
                        = fp16(data_0) * reinterpret_cast<data_type_scale *>(B->get_scale_ptr())[start_scale];
                dequantize_b[start_out + ii * matrix_n + 1]
                        = fp16(data_1) * reinterpret_cast<data_type_scale *>(B->get_scale_ptr())[start_scale + 1];
            }
        }
    }

    queue.memcpy((void *)C, (void *)C_d, size_c * sizeof(data_type_c)).wait();

    free(A_d, context);
    free(B_d, context);
    free(C_d, context);
    free(D_d, context);
    free(scale_d, context);
    free(Acc_h, context);
    free(Cnt_h, context);
    free(Acc_d, context);
    free(Cnt_d, context);
}

void xetla_linear_fp16(sycl::queue queue, fp16* A, CompressWei4Bit* B, fp16* C,
                       uint32_t matrix_m, uint32_t matrix_n,
                       uint32_t matrix_k, bool with_bias, float *bias) {
  return xetla_linear(queue, A, B, C, matrix_m, matrix_n, matrix_k, with_bias, bias);
}

void xetla_linear_fp32(sycl::queue queue, float* A, CompressWei4Bit* B, float* C,
                       uint32_t matrix_m, uint32_t matrix_n,
                       uint32_t matrix_k, bool with_bias, float *bias) {
  return xetla_linear(queue, A, B, C, matrix_m, matrix_n, matrix_k, with_bias, bias);
}
