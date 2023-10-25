#include "xetla.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

template <typename data_type_a = fp16, typename data_type_b = fp16,
          typename data_type_c = fp16,
          gpu::xetla::gpu_arch arch_tag_ = gpu::xetla::gpu_arch::Arc>
void xetla_linear(sycl::queue queue, data_type_a* A, data_type_b* B,
                  data_type_c* C, uint32_t matrix_m, uint32_t matrix_n,
                  uint32_t matrix_k) {
  uint32_t size_a = matrix_m * matrix_k;
  uint32_t size_b = matrix_k * matrix_n;
  uint32_t size_c = matrix_m * matrix_n;

  //   using data_type_a = float;
  //   using data_type_b = float;
  //   using data_type_c = float;
  using data_type_acc = float;

  constexpr uint32_t wg_tile_m = 256;
  constexpr uint32_t wg_tile_n = 256;
  constexpr uint32_t sg_tile_m = 32;
  constexpr uint32_t sg_tile_n = 64;

  // Workload mapping, linear mapping will be used in the code
  // Suppose it is divisible.
  uint32_t group_range_m = matrix_m / wg_tile_m;
  uint32_t group_range_n = matrix_n / wg_tile_n;

  // Each subgroup will be executed in one hardware thread
  // Calculate how many threads in a workgroup
  uint32_t thread_range_m = wg_tile_m / sg_tile_m;
  uint32_t thread_range_n = wg_tile_n / sg_tile_n;

  // leading dimension
  uint32_t lda = matrix_k;
  uint32_t ldb = matrix_n;
  uint32_t ldc = matrix_n;

  // Ndrange and workgroup shape
  cl::sycl::range<3> group_range{1, group_range_m, group_range_n};
  cl::sycl::range<3> local_range{1, thread_range_m, thread_range_n};

  cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);
  auto device = queue.get_info<sycl::info::queue::device>();
  std::cout << "Running on " << device.get_info<sycl::info::device::name>()
            << "\n";
  auto gpu_event = queue.submit([&](sycl::handler& cgh) {
    // GPU kernel
    cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      using namespace gpu::xetla;
      using namespace gpu::xetla::group;
      using namespace gpu::xetla::kernel;
      using namespace gpu::xetla::subgroup;

      // wrap the nd_range to XeTLA range

      // Step 1: basic computation information
      // define A, B and accumulator datatype
      // Using float as accumuator for better accuracy
      using compute_attr =
          compute_attr_t<data_type_a, data_type_b, data_type_acc>;

      // Performance tuning setting based on different shapes
      static constexpr uint32_t periodic_sync_interval = 0;
      static constexpr uint32_t prefetch_distance = 3;
      // should larger than 8
      static constexpr uint32_t k_stride = 32;
      using perf_tuning_knob = perf_tuning_knob_t<k_stride, prefetch_distance,
                                                  periodic_sync_interval>;

      // specific the computation, performance tuning and computation core
      using compute_policy =
          compute_policy_unaligned_xmx<compute_attr, perf_tuning_knob,
                                       arch_tag_>;

      // Step 2: define the memory layout & location of input/output
      // this setting could be used to optimize the data re-use in shared
      // local memory
      using mem_desc_input_a =
          mem_desc_t<data_type_a, mem_layout::row_major, mem_space::global>;
      using mem_desc_input_b =
          mem_desc_t<data_type_b, mem_layout::row_major, mem_space::global>;
      using mem_desc_output_c =
          mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>;

      // Step 3: define mirco-kernel's configuration
      using tile_shape =
          tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;
      using gemm_t = gemm_t<compute_policy, tile_shape, mem_desc_input_a,
                            mem_desc_input_b>;
      gemm_t gemm;

      // Step 4: epilogue function to overwrite the result
      using epilogue_t = epilogue_t<epilogue_policy_unaligned<arch_tag_>,
                                    tile_shape, mem_desc_output_c>;

      // Step 5: define the shared local memory usages
      // developers have the responsibility to set
      // shared loacal memory through XeTLA API
      static constexpr uint32_t barrier_count = gemm_t::barrier_count;
      static constexpr uint32_t slm_size = gemm_t::slm_size;
      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();

      // Step 6: ecah workgroup gets it individual index to start computation
      int start_n = item.get_group(2) * wg_tile_n;
      int start_m = item.get_group(1) * wg_tile_m;
      // no slicing in K direction so start from zero for all WG
      int start_k = 0;

      // Each workgroup will compute all data in K based on no k_sliciing
      // The developer can set how much data a subgroup compute by k_stride
      uint32_t wg_tile_k = matrix_k;
      uint32_t inner_loop_count = wg_tile_k / k_stride;

      // Step 7: define the workgroup start point for each workgroup
      mem_desc_input_a md_a({A}, {matrix_k, matrix_m, lda}, {start_k, start_m});
      mem_desc_input_b md_b({B}, {matrix_n, matrix_k, ldb}, {start_n, start_k});
      mem_desc_output_c md_c({C}, {matrix_n, matrix_m, ldc},
                             {start_n, start_m});

      // Step 8: real calculation with accumulator varibales which suppose
      // will be in register.
      typename gemm_t::matAcc_t matAcc;
      matAcc.init(0);

      typename gemm_t::arguments_t gemm_args(md_a, md_b, inner_loop_count);

      // the results is in the matAcc rather than real output C
      typename gemm_t::work_group_t g(item.get_local_linear_id());
      gemm(g, matAcc, gemm_args);

      // Step 9: write the results from matACC to real output C
      epilogue_t epilogue;
      epilogue(g, matAcc, md_c);
    });
  });
  gpu_event.wait();
}

void xetla_linear_fp16(sycl::queue queue, fp16* A, fp16* B, fp16* C,
                       uint32_t matrix_m, uint32_t matrix_n,
                       uint32_t matrix_k) {
  return xetla_linear(queue, A, B, C, matrix_m, matrix_n, matrix_k);
}
