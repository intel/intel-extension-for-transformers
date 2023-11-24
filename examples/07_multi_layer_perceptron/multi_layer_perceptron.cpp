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
#include "multi_layer_perceptron.hpp"
#include "tests/utils/utils.hpp"

using namespace cl::sycl;
using namespace gpu::xetla;

// MLP input size
static constexpr uint32_t matrix_m = 8192;
static constexpr uint32_t matrix_n = 256;
static constexpr uint32_t matrix_k = 256;
static constexpr uint32_t matrix_l = 256;

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
        uint32_t m, uint32_t k, uint32_t n, uint32_t l, sycl::queue &queue,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_w_ = mem_layout::row_major,
        mem_layout mem_layout_v_ = mem_layout::row_major) {
    auto A = alloc_host_and_copy<data_type_a>(A_device, size_a, queue);
    auto B = alloc_host_and_copy<data_type_b>(B_device, size_b, queue);
    auto C = alloc_host_and_copy<data_type_c>(C_device, size_c, queue);
    auto W = alloc_host_and_copy<data_type_w>(W_device, size_w, queue);
    auto V = alloc_host_and_copy<data_type_v>(V_device, size_v, queue);

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
                data[idx] = static_cast<data_type_a>(random_float() - 0.5f);
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
                data[idx] = static_cast<data_type_w>(random_float() - 0.5f);
            },
            queue, device, context);
    auto V = alloc_device_and_init<data_type_v>(
            size_v,
            [](data_type_v *data, size_t idx) {
                data[idx] = static_cast<data_type_v>(random_float() - 0.5f);
            },
            queue, device, context);

    // Define the shape of workgroup and subgroup
    // It's tunable parameters based on different input shape and hardware for better performance
    // For MLP we need wg_tile_n of both layer1 and layer2 to match the matrix_n of layer1 and layer2
    // layer1 tiling config:
    constexpr uint32_t wg_tile_m_layer1 = 128;
    constexpr uint32_t wg_tile_n_layer1 = 256;
    constexpr uint32_t sg_tile_m_layer1 = 32;
    constexpr uint32_t sg_tile_n_layer1 = 32;

    // layer2 tiling config:
    constexpr uint32_t wg_tile_m_layer2 = wg_tile_m_layer1;
    constexpr uint32_t wg_tile_n_layer2 = 256;
    constexpr uint32_t sg_tile_m_layer2 = 32;
    constexpr uint32_t sg_tile_n_layer2 = 32;

    //There are implicit requirement for sg_tile_k range
    constexpr uint32_t wg_tile_k = 32;
    static constexpr uint32_t sync_freq = 1;
    static constexpr uint32_t stages = 3;

    // Org the compute shape for sub-matrix
    using wg_shape_layer1 = shape<wg_tile_n_layer1, wg_tile_m_layer1>;
    using sg_shape_layer1 = shape<sg_tile_n_layer1, sg_tile_m_layer1>;

    // Mirco-kernel configuration
    using epilogue_policy_layer1 = xetla::group::epilogue_policy_tile_op<
            xetla::subgroup::chained_tile_op_t<gpu::xetla::subgroup::relu_op_t>,
            gpu_arch::Xe>;
    using layer1_tune_option
            = dict_t<elem_v_t<tune_key::PARAM_OPTIMZER_TYPE,
                             tune_key_value::PARAM_OPTIMZER_DECISION_TREE>,
                    elem_t_t<tune_key::EPILOGUE_POLICY, epilogue_policy_layer1>,
                    elem_t_t<tune_key::SG_TILE_SHAPE, sg_shape_layer1>,
                    elem_v_t<tune_key::PREFETCH_DISTANCE, stages>,
                    elem_v_t<tune_key::PERIODIC_SYNC_INTERVAL, sync_freq>>;
    using gemm_layer1_t = xetla::group::default_gemm_selector_t<
            data_type_a, // input datatype for A
            mem_layout::row_major, // memory layout for A
            8, // leading dimension for A, in unit of element
            mem_space::global, // memory reading from global mem for A
            data_type_w, // input datatype for W
            mem_layout::row_major, // memory layout for W
            8, // leading dimension for W, in unit of element
            mem_space::global, // memory reading from global mem for W
            data_type_acc, // accumulator data type for intermediate resutls
            wg_shape_layer1, // computation tile shape
            wg_tile_k, // elements in each iteration
            gpu_arch::Xe, // GPU arch
            layer1_tune_option>;

    using epilogue_layer1_t = xetla::group::default_epilogue_selector_t<
            data_type_b, // onput datatype for B
            mem_layout::row_major, // memory layout for B
            8, // leading dimension for B, in unit of element
            mem_space::global, // memory writing to global mem for B
            wg_shape_layer1, // computation tile shape
            wg_tile_k, // elements in each iteration
            gpu_arch::Xe, // GPU arch
            layer1_tune_option>;

    using wg_shape_layer2 = shape<wg_tile_n_layer2, wg_tile_m_layer2>;
    using sg_shape_layer2 = shape<sg_tile_n_layer2, sg_tile_m_layer2>;

    // Mirco-kernel configuration
    using layer2_tune_option
            = dict_t<elem_v_t<tune_key::PARAM_OPTIMZER_TYPE,
                             tune_key_value::PARAM_OPTIMZER_DECISION_TREE>,
                    elem_t_t<tune_key::SG_TILE_SHAPE, sg_shape_layer2>,
                    elem_v_t<tune_key::PREFETCH_DISTANCE, stages>,
                    elem_v_t<tune_key::PERIODIC_SYNC_INTERVAL, sync_freq>>;
    using gemm_layer2_t = xetla::group::default_gemm_selector_t<
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for B
            8, // leading dimension for B, in unit of element
            mem_space::global, // memory reading from global mem for B
            data_type_v, // input datatype for V
            mem_layout::row_major, // memory layout for V
            8, // leading dimension for V, in unit of element
            mem_space::global, // memory reading from global mem for V
            data_type_acc, // accumulator data type for intermediate resutls
            wg_shape_layer2, // computation tile shape
            wg_tile_k, // elements in each iteration
            gpu_arch::Xe, // GPU arch
            layer2_tune_option>;

    using epilogue_layer2_t = xetla::group::default_epilogue_selector_t<
            data_type_c, // onput datatype for C
            mem_layout::row_major, // memory layout for C
            8, // leading dimension for C, in unit of element
            mem_space::global, // memory writing to global mem for C
            wg_shape_layer2, // computation tile shape
            wg_tile_k, // elements in each iteration
            gpu_arch::Xe, // GPU arch
            layer2_tune_option>;

    using mlp_op_t = xetla::kernel::multi_layer_perceptron_t<gemm_layer1_t,
            epilogue_layer1_t, gemm_layer2_t, epilogue_layer2_t, gpu_arch::Xe>;

    // set up mlp arguments
    // for relu we don't need to set arguments
    typename mlp_op_t::arguments_t mlp_arg(matrix_m, matrix_k, matrix_n,
            matrix_m, matrix_n, matrix_l, A, matrix_k, W, matrix_n, B, matrix_n,
            V, matrix_l, C, matrix_l);
    cl::sycl::nd_range<3> nd_range = mlp_op_t::get_nd_range(mlp_arg);

    if (!mlp_op_t::can_implement(mlp_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        free(A, context);
        free(B, context);
        free(C, context);
        free(W, context);
        free(V, context);
        FAIL();
    }
    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k
            + 2 * static_cast<long>(matrix_m) * matrix_n * matrix_l;
    profiling_helper prof("mlp", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                // allocate slm and nbarrier resource
                slm_barrier_init<mlp_op_t>();
                mlp_op_t mlp_op;
                mlp_op(item, mlp_arg);
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
