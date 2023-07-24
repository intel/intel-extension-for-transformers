/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "tests/utils/utils.hpp"
#include "xetla.hpp"

enum class kslicing_impl_t { none = 0, global = 1, local = 2 };

template <kslicing_impl_t kslicing_type = kslicing_impl_t::none>
void basic_gemm_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    //GEMM input size
    uint32_t matrix_m = 4096;
    uint32_t matrix_n = 4096;
    uint32_t matrix_k = 4096;

    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c
            = std::conditional_t<kslicing_type == kslicing_impl_t::global,
                    float, bf16>;
    using data_type_acc = float;

    //Turn on the profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    //Define SYCL queue, context and device
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto A = alloc_device_and_init<data_type_a>(
            size_a,
            [](data_type_a *data, size_t idx) {
                data[idx] = static_cast<data_type_a>(random_float());
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type_b>(
            size_b,
            [](data_type_b *data, size_t idx) {
                data[idx] = static_cast<data_type_b>(random_float());
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type_c>(
            size_c,
            [](data_type_c *data, size_t idx) {
                data[idx] = static_cast<data_type_c>(0.0f);
            },
            queue, device, context);

    //Define the shape of workgroup and subgroup
    //It's tunable parameters based on different input shape and hardware for better performance
    constexpr uint32_t wg_tile_m
            = (kslicing_type != kslicing_impl_t::local) ? 256 : 64;
    constexpr uint32_t wg_tile_n
            = (kslicing_type != kslicing_impl_t::local) ? 256 : 128;
    constexpr uint32_t sg_tile_m
            = (kslicing_type != kslicing_impl_t::local) ? 32 : 16;
    constexpr uint32_t sg_tile_n
            = (kslicing_type != kslicing_impl_t::local) ? 64 : 32;

    //There are implicit requirement for sg_tile_k range
    constexpr uint32_t sg_tile_k = 32;

    // Org the compute shape for sub-matrix
    using tile_shape
            = xetla::group::tile_shape_t<wg_tile_n, // workgroup size in dim0
                    wg_tile_m, //	workgroup size in dim1
                    sg_tile_n, //	subgroup size in dim0
                    sg_tile_m>; //	subgroup size in dim1

    // Mirco-kernel configuration
    using brgemm_config = xetla::group::brgemm_selector_t<
            data_type_a, // input datatype for A
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for A
            mem_layout::row_major, // memory layout for B
            mem_space::global, // memory reading from global mem for A
            mem_space::global, // memory reading from global mem for B
            8, // buffer alignment for A, in unit of element
            8, // buffer alignment for B, in unit of element
            data_type_acc, // accumulator data type for intermediate resutls
            tile_shape, // computation tile shape
            sg_tile_k, // elements in each iteration
            mma_engine::xmx, // compute engine
            gpu_arch::Xe> // GPU arch
            ::brgemm;

    using update_method
            = std::conditional_t<kslicing_type == kslicing_impl_t::global,
                    result_reduce_sum, result_overwrite>;
    using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_default<update_method, gpu_arch::Xe>,
            tile_shape,
            mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>>;

    // specify the range k_w/k_s by setting the corresponding ratio
    // splitk using global memory
    constexpr int splitk_global_ratio
            = (kslicing_type == kslicing_impl_t::global) ? 2 : 1;
    // splitk using local memory
    constexpr int splitk_local_ratio
            = (kslicing_type == kslicing_impl_t::local) ? 2 : 1;

    using dispatch_policy = std::conditional_t<kslicing_type
                    == kslicing_impl_t::none,
            gpu::xetla::kernel::dispatch_policy_default<gpu_arch::Xe>,
            gpu::xetla::kernel::dispatch_policy_kslicing<splitk_global_ratio,
                    splitk_local_ratio, gpu_arch::Xe>>;

    using gemm_op_t
            = xetla::kernel::gemm_t<dispatch_policy, brgemm_config, epilogue_t>;

    if constexpr (kslicing_type != kslicing_impl_t::none) {
        std::cout << "basic_gemm with "
                  << (kslicing_type == kslicing_impl_t::global ? "global"
                                                               : "local")
                  << " cooperation" << std::endl;
    }

    // set up gemm arguments
    typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A,
            matrix_k, B, matrix_n, C, matrix_n);

    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
    if (!gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        FAIL();
    }

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof("basic_gemm", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        if constexpr (kslicing_type == kslicing_impl_t::global) {
            queue.memset(C, 0, size_c * sizeof(data_type_c));
        }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                xetla_exec_item<3> ei(item);
                // allocate slm and nbarrier resource
                slm_barrier_init<gemm_op_t>();
                gemm_op_t gemm_op;
                gemm_op(ei, gemm_arg);
            });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
        }
    }

    ASSERT_EQ(0,
            gemm_result_validate(A, B, C, 1, matrix_m, matrix_k, matrix_n,
                    queue, context, mem_layout::row_major,
                    mem_layout::row_major));

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
}

int main() {
    // An example code for calculating matrix multiplication using
    // GEMM API:
    //   C = A x B
    // The resulted matrix C is partitioned by the group range
    // in to multiple blocks. The block matrix
    //  C<i_w, j_w>
    // is computed by the workgroup with id: (0, i_w, j_w).
    // (i_w, j_w) is an element in range specified by group range.
    // Each thread with index (0, i_s, j_s) inside the same workgroup
    // is responsible for a sub block of matrix multiplication, which is
    //   C<i_w, j_w>[i_s*sg_m:(i_s+1):sg_m,j_s*sg_n:(j_s+1)*sg_n]

    // Alternatively, some threads can cooperate on the same sub block
    // matrix given the same (i_s, j_s), i.e. the index space is extended
    // from (0, i_s, j_s) to (k_s, i_s, j_s).

    // Another method to achieve the same effect is to extend the index space
    // in group range, i.e. from (0, i_w, j_w) to (k_w, i_w, j_w)

    // More detailed description referring to the cooperation (kslicing) could
    // be found in the example 06_splitk_brgemm with custom implementation

    // basic gemm
    basic_gemm_run<kslicing_impl_t::none>(10);

    // basic gemm with workgroup cooperation
    // basic_gemm_run<kslicing_impl_t::global>(10);

    // basic gemm with thread cooperation
    // basic_gemm_run<kslicing_impl_t::local>(10);
    return (0);
}
