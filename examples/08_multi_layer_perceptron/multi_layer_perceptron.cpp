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
#include <algorithm>
#include "tests/utils/utils.hpp"
#include "xetla.hpp"

using namespace cl::sycl;
using namespace gpu::xetla;

//GEMM input size
static constexpr uint32_t matrix_m = 1024;
static constexpr uint32_t matrix_n = 512;
static constexpr uint32_t matrix_k = 128;
static constexpr uint32_t matrix_l = 64;

static constexpr uint32_t size_a = matrix_m * matrix_k;
static constexpr uint32_t size_b = matrix_m * matrix_n;
static constexpr uint32_t size_c = matrix_m * matrix_l;
static constexpr uint32_t size_w = matrix_k * matrix_n;
static constexpr uint32_t size_v = matrix_n * matrix_l;

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_w, typename data_type_v,
        typename data_type_acc = float>
int mlp_result_validate(data_type_a *A_device, data_type_b *B_device,
        data_type_c *C_device, data_type_w *W_device, data_type_v *V_device,
        uint32_t m, uint32_t k, uint32_t n, uint32_t l, sycl::queue queue,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_w_ = mem_layout::row_major,
        mem_layout mem_layout_v_ = mem_layout::row_major) {
    auto A = alloc_host_and_copy<data_type_a>(A_device, size_a, queue);
    auto B = alloc_host_and_copy<data_type_b>(B_device, size_b, queue);
    auto C = alloc_host_and_copy<data_type_c>(C_device, size_c, queue);
    auto W = alloc_host_and_copy<data_type_w>(W_device, size_w, queue);
    auto V = alloc_host_and_copy<data_type_v>(V_device, size_v, queue);

    bool is_col_major_a = mem_layout_a_ == mem_layout::col_major;
    bool is_col_major_w = mem_layout_w_ == mem_layout::col_major;
    bool is_col_major_v = mem_layout_v_ == mem_layout::col_major;
    buff_cmp::buff_vals<data_type_b> data_layer1(B, m, n, n);
    std::vector<data_type_acc> gold_B(m * n, 0);
    get_gemm_gold<data_type_a, data_type_w, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_w_, A, W, gold_B.data());
    std::transform(gold_B.cbegin(), gold_B.cend(), gold_B.begin(),
            [](data_type_acc b) { return b > 0.0f ? b : 0.0f; });
    buff_cmp::buff_vals<data_type_b, data_type_acc> other_layer1(
            gold_B.data(), m, n, n);

    bool result = buff_cmp::xetla_buff_cmp(
            data_layer1, other_layer1, "mlp validation (Layer1)");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");

    if (result) {
        buff_cmp::buff_vals<data_type_c> data_layer2(C, m, l, l);
        std::vector<data_type_acc> gold_C(m * l, 0);
        get_gemm_gold<data_type_b, data_type_v, data_type_acc>(
                m, l, n, mem_layout_a_, mem_layout_v_, B, V, gold_C.data());
        buff_cmp::buff_vals<data_type_c, data_type_acc> other_layer2(
                gold_C.data(), m, l, l);

        bool result = buff_cmp::xetla_buff_cmp(
                data_layer2, other_layer2, "mlp validation (Layer2)");

        std::cout << (!result ? "FAILED\n" : "PASSED\n");

    } else {
        std::cout << "Layer2 validation skipped due to failure in Layer1\n";
    }

    free(A);
    free(B);
    free(C);
    free(W);
    free(V);

    return result ? 0 : 1;
}

void mlp_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
    using data_type_w = bf16;
    using data_type_v = bf16;
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
                data[idx] = static_cast<data_type_b>(0.0f);
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type_c>(
            size_c,
            [](data_type_c *data, size_t idx) {
                data[idx] = static_cast<data_type_c>(0.0f);
            },
            queue, device, context);
    auto W = alloc_device_and_init<data_type_w>(
            size_w,
            [](data_type_w *data, size_t idx) {
                data[idx] = static_cast<data_type_w>(random_float());
            },
            queue, device, context);
    auto V = alloc_device_and_init<data_type_v>(
            size_v,
            [](data_type_v *data, size_t idx) {
                data[idx] = static_cast<data_type_v>(random_float());
            },
            queue, device, context);

    //Define the shape of workgroup and subgroup
    //It's tunable parameters based on different input shape and hardware for better performance
    constexpr uint32_t wg_tile_m = 256;
    constexpr uint32_t wg_tile_n = 256;
    constexpr uint32_t sg_tile_m = 32;
    constexpr uint32_t sg_tile_n = 64;

    //There are implicit requirement for sg_tile_k range
    constexpr uint32_t sg_tile_k = 32;

    //Workload mapping, linear mapping will be used in the code
    constexpr uint32_t group_range_m = matrix_m / wg_tile_m;
    constexpr uint32_t group_range_n = matrix_n / wg_tile_n;

    //Each subgroup will be executed in one hardware thread
    //Calculate how many threads in a workgroup
    constexpr uint32_t thread_range_m = wg_tile_m / sg_tile_m;
    constexpr uint32_t thread_range_n = wg_tile_n / sg_tile_n;

    // [MLP] tile parameters for Layer2
    constexpr uint32_t wg_tile_m_layer2 = matrix_m / group_range_m;
    constexpr uint32_t wg_tile_n_layer2 = matrix_l / group_range_n;
    constexpr uint32_t sg_tile_m_layer2 = 16;
    constexpr uint32_t sg_tile_n_layer2 = 16;
    constexpr uint32_t sg_tile_k_layer2 = 8;

    //Ndrange and workgroup shape
    cl::sycl::range<3> GroupRange {1, group_range_m, group_range_n};
    cl::sycl::range<3> LocalRange {1, thread_range_m, thread_range_n};

    cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k
            + 2 * static_cast<long>(matrix_m) * matrix_n * matrix_l;
    profiling_helper prof("mlp", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                using namespace gpu::xetla;
                using namespace gpu::xetla::group;
                using namespace gpu::xetla::kernel;
                using namespace gpu::xetla::subgroup;

                // [MLP] Layer1
                {

                    // Org the compute shape for sub-matrix
                    // [MLP] define tile_shape for Layer1
                    // This is used for loading block matrix for multiplication:
                    //   B<i, j> = sum_{k} A<i, k> x W<k, j>
                    //           = A<i, :> x W<:, j>
                    // where:
                    //   <r, c>: block matrix selector
                    // Note:
                    //   - checkout 06_splitk_brgemm example for more description

                    // [MLP] Suppose for each xetla_exec_item<3> ei,
                    // the index tuple (i, j) assigned to it has the following value:
                    //   i <- ei.get_group(2)
                    //   j <- ei.get_group(1)
                    // Then the block matrix could be expressed using numpy notation:
                    //   A<i, k> = A[Q:E, F:G]
                    //   W<k, j> = W[X:Y, S:H]
                    // where:
                    //   Q = j * wg_tile_m
                    //   E = (j + 1) * wg_tile_m
                    //   F = k * sg_tile_k
                    //   G = (k + 1) * sg_tile_k
                    //   X = k * sg_tile_k
                    //   Y = (k + 1) * sg_tile_k
                    //   S = i * wg_tile_n
                    //   H = (i + 1) * wg_tile_n
                    using tile_shape
                            = tile_shape_t<wg_tile_n, // workgroup size in dim0
                                    wg_tile_m, //   workgroup size in dim1
                                    sg_tile_n, //   subgroup size in dim0
                                    sg_tile_m>; //  subgroup size in dim1

                    // Mirco-kernel configuration
                    using brgemm_config = brgemm_selector_t<
                            data_type_a, // input datatype for A
                            data_type_w, // input datatype for W
                            mem_layout::row_major, // memory layout for A
                            mem_layout::row_major, // memory layout for W
                            mem_space::
                                    global, // memory reading from global mem for A
                            mem_space::
                                    global, // memory reading from global mem for W
                            8, // buffer alignment for A, in unit of element
                            8, // buffer alignment for W, in unit of element
                            data_type_acc, // accumulator data type for intermediate resutls
                            tile_shape, // computation tile shape
                            sg_tile_k, // elements in each iteration
                            mma_engine::xmx, // compute engine
                            gpu_arch::Xe> // GPU arch
                            ::brgemm;

                    // [MLP] Use relu_op_t as activation function
                    using mem_desc_output_t = mem_desc_t<data_type_b,
                            mem_layout::row_major, mem_space::global>;
                    using epilogue_t = epilogue_t<
                            epilogue_policy_tile_op<
                                    chained_tile_op_t<relu_op_t>,
                                    result_overwrite, gpu_arch::Xe>,
                            tile_shape, mem_desc_output_t>;
                    // [MLP] Define tile_op arguments
                    using epilogue_tile_op_args_t
                            = epilogue_t::tile_op_t::arguments_t;
                    // [MLP] ReLU accepts no arguments
                    epilogue_tile_op_args_t tile_op_args {};

                    using gemm_op_t
                            = gemm_t<dispatch_policy_default<gpu_arch::Xe>,
                                    brgemm_config, epilogue_t>;

                    typename gemm_op_t::arguments_t arg(matrix_m, matrix_k,
                            matrix_n, A, matrix_k, W, matrix_n, B, matrix_n,
                            tile_op_args);

                    slm_barrier_init<gemm_op_t>();

                    gemm_op_t gemm_op;

                    xetla_exec_item<3> ei(item);
                    gemm_op(ei, arg);
                }

                // [MLP] Layer2
                {

                    // Org the compute shape for sub-matrix
                    // [MLP] The work-item xetla_exec_item<3> ei is shared
                    // between Layer1 and Layer2. However, due to different
                    // memory size of the input matrices and output matrices,
                    // the tile_shape needs to be reconfigured.
                    // Old task:
                    //   B<i, j> = A<i, :> x W<:, j>
                    // where:
                    //   shape(A) = [m, k]
                    //   shape(W) = [k, n]
                    //   shape(B) = [m, n]
                    // New task:
                    //   C<i, j> = B<i, :> x V<:, j>
                    // where:
                    //   shape(B) = [m, n]
                    //   shape(V) = [n, l]
                    //   shape(C) = [m, l]
                    // Note:
                    //   - the block matrix size B<i, j> differs between tasks,
                    //     which is dependent on tile_shape

                    // [MLP] We show again the index space:
                    //   (i, j) ∈ I x J
                    // where:
                    //   I = {i: 0, 1, ..., group_range_n - 1}
                    //   J = {j: 0, 1, ..., group_range_m - 1}

                    // [MLP] The reconfiguration should find a set of parameters
                    //   - wg_tile_n_layer2
                    //   - wg_tile_m_layer2
                    //   - sg_tile_n_layer2
                    //   - sg_tile_m_layer2
                    //   - sg_tile_k_layer2
                    // that satisfy the following requirements:
                    //   ; load all rows from B
                    //   wg_tile_m_layer2 * U(J) = m
                    //   ; load all columns from V
                    //   wg_tile_n_layer2 * U(I) = l
                    //   ; accumulate step size in dimension n
                    //   sg_tile_k_layer2 <= n
                    //   ; remapping invariant Z from index space to tile_shape
                    //   Z = O * P = L * D
                    // where:
                    //   U: X |-> max X + 1
                    //   O = wg_tile_n_layer2 / sg_tile_n_layer2
                    //   P = wg_tile_m_layer2 / sg_tile_m_layer2
                    //   L = thread_range_m
                    //   D = thread_range_n
                    //   Z: invariant that all tile_shape definition should satisfy
                    using tile_shape = tile_shape_t<
                            wg_tile_n_layer2, // workgroup size in dim0
                            wg_tile_m_layer2, //   workgroup size in dim1
                            sg_tile_n_layer2, //   subgroup size in dim0
                            sg_tile_m_layer2>; //  subgroup size in dim1

                    // Mirco-kernel configuration
                    using brgemm_config = brgemm_selector_t<
                            data_type_b, // input datatype for B
                            data_type_v, // input datatype for V
                            mem_layout::row_major, // memory layout for B
                            mem_layout::row_major, // memory layout for V
                            mem_space::
                                    global, // memory reading from global mem for B
                            mem_space::
                                    global, // memory reading from global mem for V
                            8, // buffer alignment for B, in unit of element
                            8, // buffer alignment for V, in unit of element
                            data_type_acc, // accumulator data type for intermediate resutls
                            tile_shape, // computation tile shape
                            sg_tile_k, // elements in each iteration
                            mma_engine::xmx, // compute engine
                            gpu_arch::Xe> // GPU arch
                            ::brgemm;

                    using epilogue_t = epilogue_t<
                            epilogue_policy_tile_op<chained_tile_op_t<>,
                                    result_overwrite, gpu_arch::Xe>,
                            tile_shape,
                            mem_desc_t<data_type_c, mem_layout::row_major,
                                    mem_space::global>>;

                    using gemm_op_t
                            = gemm_t<dispatch_policy_default<gpu_arch::Xe>,
                                    brgemm_config, epilogue_t>;

                    // [MLP] specify Layer2 tensor dimensions
                    typename gemm_op_t::arguments_t arg(matrix_m, matrix_n,
                            matrix_l, B, matrix_n, V, matrix_l, C, matrix_l);

                    slm_barrier_init<gemm_op_t>();

                    gemm_op_t gemm_op;

                    xetla_exec_item<3> ei(item);
                    gemm_op(ei, arg);
                }
            });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
        }
    }

    ASSERT_EQ(0,
            mlp_result_validate(A, B, C, W, V, matrix_m, matrix_k, matrix_n,
                    matrix_l, queue, mem_layout::row_major,
                    mem_layout::row_major, mem_layout::row_major));

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
    free(W, context);
    free(V, context);
}

int main() {
    // This example implements a two layer MLP with ReLU as activation function.
    // It will be shown how to remap the index space of each work-item used for
    // Layer1 to a new type of task in Layer2.

    // Description:
    // A MLP is a chained matrix multiplication with activation function applied to
    // their intermediate results:
    //   MLP: A |-> C = Layer2(Layer1(A))
    //                = ReLU(A x W) x V
    // where:
    //   A: input data
    //   shape(A) = [m, k]
    //   shape(C) = [m, l]

    // The index space is designed to execute the following task:
    //   Layer1: A |-> B = ReLU(A x W)
    // where:
    //   A: input tensor
    //   W: weight tensor of Layer1
    //   shape(A) = [m, k]
    //   shape(W) = [k, n]

    // After the first stage, the same thread is to be remapped to the following task:
    //   Layer2: B |-> C = B x V
    // where:
    //    B: output tensor of Layer1
    //    V: weight tensor of Layer2
    //    shape(B) = [m, n]
    //    shape(V) = [n, l]

    // Note:
    //   - comments related to mlp will have the "[MLP]" prefix

    mlp_run(10);
    return (0);
}
