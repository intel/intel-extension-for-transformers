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
#pragma once

#include "xetla.hpp"

using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::subgroup;

class gru_config_t {
public:
    using dtype_in = bf16;
    using dtype_acc = float;
    /// layer_size = 3
    static constexpr uint32_t layer_size = 3;
    /// sequence_length = 64
    static constexpr uint32_t sequence_length = 2;
    /// batch_size = 512
    static constexpr uint32_t batch_size = 512;
    /// input_size = 384
    static constexpr uint32_t input_size = 384;
    /// hidden_size = 688;
    static constexpr uint32_t hidden_size = 704;
    /// launch config
    static constexpr uint32_t wg_tile_m = 64;
    static constexpr uint32_t wg_tile_n = 128;
    static constexpr uint32_t sg_tile_m = 16;
    static constexpr uint32_t sg_tile_n = 16;
    static constexpr uint32_t sg_tile_k = 32;
};

template <typename T>
struct fused_config_t {
    uint32_t input_size;
    uint32_t hidden_size;
    uint32_t batch_size;
    uint32_t sequence_length = 1;
    T *layer_ptr
            = nullptr; /// layer_input = sequence_length x batch_size x input_size
    T *hx_ptr = nullptr; /// h_x input = batch_size x hidden_size
    T *W_ir_ptr = nullptr;
    T *W_hr_ptr = nullptr;
    T *W_iz_ptr = nullptr;
    T *W_hz_ptr = nullptr;
    T *W_in_ptr = nullptr;
    T *W_hn_ptr = nullptr;
    T *cell_out_ptr
            = nullptr; /// cell output = sequence_length x batch_size x hidden_size
    T *layer_output
            = nullptr; /// layer_output = layer_size x batch_size x hidden_size
    T *one_cell_ptr = nullptr;
};

#define CONFIG_SETTING(m, k, n) \
    boundary_n = (start_n + wg_tile_n) > n ? n : (start_n + wg_tile_n); \
    matrix_n = n; \
    start_x_b = start_n; \
    start_y_b = start_k;

#define GEMM_CALL(id, acc_id, ptr_a, ptr_b) \
    mem_desc_a.init({ptr_a}, \
            {boundary_k_##id, boundary_m, \
                    is_col_major_a ? matrix_m : matrix_k_##id}, \
            {start_x_a, start_y_a}); \
    mem_desc_b.init({ptr_b}, \
            {boundary_n, boundary_k_##id, \
                    is_col_major_b ? matrix_k_##id : matrix_n}, \
            {start_x_b, start_y_b}); \
    gemm_args.init(mem_desc_a, mem_desc_b, inner_loop_count_##id); \
    op(g, matAcc_##acc_id, gemm_args); \
    SW_BARRIER();

#define MATC_STORE(ptr_c) \
    mem_desc_c.init( \
            {ptr_c}, {boundary_n, boundary_m, matrix_n}, {start_n, start_m}); \
    epilogue(g, matAcc_0, mem_desc_c, epilogue_args);

template <typename T, typename Act_T, uint32_t wg_tile_m, uint32_t wg_tile_n,
        uint32_t sg_tile_m, uint32_t sg_tile_n, uint32_t sg_tile_k,
        mem_layout layout_input = mem_layout::row_major,
        mem_layout layout_weight = mem_layout::row_major,
        mem_layout layout_out = mem_layout::row_major,
        mem_space mem_loc_input = mem_space::global,
        mem_space mem_loc_weight = mem_space::global,
        mem_space mem_loc_out = mem_space::global,
        uint32_t periodic_sync_interval = 0>
struct gru_layer {
    static constexpr uint32_t prefetch_distance = 3;
    using perf_tuning_knob = perf_tuning_knob_t<sg_tile_k, prefetch_distance,
            periodic_sync_interval>;

    using compute_attr = group::compute_attr_t<T, T, Act_T>;
    using compute_policy = compute_policy_default_xmx<compute_attr,
            perf_tuning_knob, gpu_arch::Xe>;
    using mem_desc_a_t = mem_desc_t<T, layout_input, mem_loc_input>;
    using mem_desc_b_t = mem_desc_t<T, layout_weight, mem_loc_weight>;
    // Org the compute shape for sub-matrix
    using tile_shape = tile_shape_t<wg_tile_n, // workgroup size in N dim
            wg_tile_m, //	workgroup size in M dim
            sg_tile_n, //	subgroup size in N dim
            sg_tile_m>; //	subgroup size in M dim

    static constexpr bool is_col_major_a
            = layout_input == mem_layout::col_major;
    static constexpr bool is_col_major_b
            = layout_weight == mem_layout::col_major;
    using gemm_op
            = gemm_t<compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;
    using work_group_t = typename gemm_op::work_group_t;
    using gemm_arguments = typename gemm_op::arguments_t;
    using matAcc_t = typename gemm_op::matAcc_t;

    using mem_desc_c_t = mem_desc_t<T, layout_out, mem_loc_out>;

    // define arguments for each epilogue_tile_op in chained_tile_op_t<>

    using epilogue_t = epilogue_t<epilogue_policy_default<gpu_arch::Xe>,
            tile_shape, mem_desc_c_t>;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    using matC_tile_desc_t = tile_desc_t<matAcc_t::tile_size_x,
            matAcc_t::tile_size_y, matAcc_t::block_size_x,
            matAcc_t::block_size_y, reg_layout::tiled>;
    using mat_hidden_t = tile_t<T, matC_tile_desc_t>;
    using matC_t = tile_t<T, matC_tile_desc_t>;
    using mat_hidden_payload_t = mem_payload_t<mem_desc_a_t, matC_tile_desc_t,
            msg_type_v<matC_tile_desc_t, mem_loc_input>, gpu_arch::Xe>;
    using matC_payload_t = mem_payload_t<mem_desc_c_t, matC_tile_desc_t,
            msg_type::block_2d, gpu_arch::Xe>;
    using sigmoid_t = typename subgroup::sigmoid_op_t;
    using tanh_t = typename subgroup::tanh_op_t;
    static void inline call(sycl::nd_item<3> &item, fused_config_t<T> *args) {
        gemm_op op;
        sigmoid_t sigmoid;
        tanh_t tanh;
        // declare two accumulators to stroe the results of two GEMMs
        // and its activation
        matAcc_t matAcc_0, matAcc_1;
        matC_t matC;
        matC_payload_t matC_payload;
        gemm_arguments gemm_args;
        mat_hidden_t mat_hidden;
        mat_hidden_payload_t mat_hidden_payload;
        mem_desc_a_t mem_desc_a;
        mem_desc_b_t mem_desc_b;
        mem_desc_c_t mem_desc_c;
        epilogue_t epilogue;
        epilogue_args_t epilogue_args {};

        uint32_t batch_size, input_size, hidden_size, seq_len;
        batch_size = args->batch_size;
        input_size = args->input_size;
        hidden_size = args->hidden_size;
        seq_len = args->sequence_length;

        uint32_t matrix_n = hidden_size;
        uint32_t matrix_m = batch_size;
        uint32_t matrix_k_0 = input_size;
        uint32_t matrix_k_1 = hidden_size;
        int start_x_b, start_y_b, start_x_a, start_y_a;
        uint32_t boundary_n, boundary_m, boundary_k_0, boundary_k_1;
        uint32_t wg_tile_k_0, wg_tile_k_1;
        wg_tile_k_0 = input_size;
        wg_tile_k_1 = hidden_size;
        boundary_k_0 = wg_tile_k_0;
        boundary_k_1 = wg_tile_k_1;

        // layer_0:
        // hidden out matrix = 512 x 704
        // matmul input 512 x 384 : 384 x 704
        // matmul hidden 512 x 704 : 704 x 704

        // layer_1 , layar_2:
        // hidden out matrix = 512 x 704
        // matmul input 512 x 704 : 704 x 704
        // matmul hidden 512 x 704 : 704 x 704
        // two GEMMs will have different loop counts on k dim
        uint32_t inner_loop_count_0 = (wg_tile_k_0 + sg_tile_k - 1) / sg_tile_k;
        uint32_t inner_loop_count_1 = (wg_tile_k_1 + sg_tile_k - 1) / sg_tile_k;

        int start_m = item.get_group(1) * wg_tile_m;

        boundary_m = (start_m + wg_tile_m) > batch_size ? batch_size
                                                        : (start_m + wg_tile_m);

        int start_k = 0;

        start_x_a = start_k;
        start_y_a = start_m;
        int io_size = batch_size * hidden_size;
        int pre_layer_size = batch_size * input_size;
        work_group_t g(item.get_local_linear_id());
        for (uint32_t seq_id = 0; seq_id < seq_len; ++seq_id) {
            for (int j = (hidden_size + wg_tile_n - 1) / wg_tile_n - 1; j >= 0;
                    j--) {
                int start_n = (j)*wg_tile_n;
                CONFIG_SETTING(batch_size, -1, hidden_size);
                matAcc_0.init(0);
                SW_BARRIER();

                // calculate reset gate: r_t = \sigmoid(X_t x W_ir + h_{t - 1} x W_hr)
                // acc0 = X_t x W_ir
                // acc0 += h_{t - 1} x W_hr
                // acc0 = sigmoid(acc0)
                // Mathematically elemwise_op is a map that applies to each element:
                //   elemwise_op: [m, n] -> [m, n], acc |-> tile_op_t(acc)
                GEMM_CALL(0, 0, args->layer_ptr + seq_id * pre_layer_size,
                        args->W_ir_ptr);
                GEMM_CALL(1, 0, args->hx_ptr, args->W_hr_ptr);
                sigmoid(matAcc_0, 0);
                // calculate new gate : n_t = tanh(X_t x W_in + r_t * (h_{t - 1} x
                // W_hn)) acc1 = h_{t - 1} x W_hn acc0 *= acc1 acc0 += X_t x W_in acc0 =
                // tanh(acc0) Mathematically elemwise_op is a map that applies to each
                // element:
                //   elemwise_op: [m, n] -> [m, n], acc |-> tile_op_t(acc)
                matAcc_1.init(0);
                GEMM_CALL(1, 1, args->hx_ptr, args->W_hn_ptr);
                matAcc_0.reg = matAcc_1.reg * matAcc_0.reg;
                GEMM_CALL(0, 0, args->layer_ptr + seq_id * pre_layer_size,
                        args->W_in_ptr);

                tanh(matAcc_0, 0);
                // calculate input gate z_t = \sigma(X_t x W_iz + h_{t - 1} x W_hz)
                // acc1 = X_t x W_iz
                // acc1 += h_{t - 1} x W_hz
                // acc1 = sigmoid(acc1)
                // Mathematically elemwise_op is a map that applies to each element:
                //   elemwise_op: [m, n] -> [m, n], acc |-> tile_op_t(acc)
                matAcc_1.init(0);
                GEMM_CALL(1, 1, args->hx_ptr, args->W_hz_ptr);
                GEMM_CALL(0, 1, args->layer_ptr + seq_id * pre_layer_size,
                        args->W_iz_ptr);
                sigmoid(matAcc_1, 0);
                // calculate h_t = (1 - z_t) n_t + z_t h_{t - 1} NOTICE z_t in Acc1,
                // n_t in Acc0 reload h_{t - 1}
                // acc0 = acc0 * (1 - acc1) + acc1 * h_{t -1}
                mem_desc_c.init({args->hx_ptr},
                        {boundary_n, boundary_m, matrix_n},
                        {start_n + gemm_op::get_matC_offset_x(g),
                                start_m + gemm_op::get_matC_offset_y(g)});
                mat_hidden_payload.init(mem_desc_c);
                tile_load<cache_hint::cached, cache_hint::cached>(
                        mat_hidden, mat_hidden_payload);
                matAcc_0.reg = matAcc_0.reg * (1 - matAcc_1.reg)
                        + matAcc_1.reg
                                * xetla_cvt<Act_T, T, matAcc_t::tile_elems>(
                                        mat_hidden.reg);
                SW_BARRIER();

                if (seq_id == seq_len - 1) {
                    MATC_STORE(args->layer_output);
                    SW_BARRIER();
                    __esimd_barrier();
                }
                MATC_STORE(args->cell_out_ptr + seq_id * io_size);
                SW_BARRIER();
                __esimd_barrier();

                MATC_STORE(args->one_cell_ptr + (seq_id % 2) * io_size);
                SW_BARRIER();
                __esimd_barrier();
            }
            args->hx_ptr = args->one_cell_ptr + (seq_id % 2) * io_size;
        }
    }
};

template <typename input_T, typename Act_T, uint32_t wg_tile_m_t,
        uint32_t wg_tile_n_t, uint32_t sg_tile_m_t, uint32_t sg_tile_n_t,
        uint32_t sg_tile_k_t>
struct kernel_xcoder_gru_fusion {
    /// @brief
    /// @param item Is the sycl::nd_item
    /// @param layer_ptr  input from previous layer i.e X_t
    /// @param h0_ptr     hx_ptr input i.e. h_{0}  shape = layer_size x batch_size
    /// x hidden_size weights
    /// @param W_ir_ptr   weights with input of reset gate, (input_weight_size,
    /// hidden_weight_size, ...)
    /// @param W_hr_ptr   weights with hidden input of reset gate, shape =
    /// layer_size x hidden_weight_size
    /// @param W_iz_ptr   weights with input of input gate, (input_weight_size,
    /// hidden_weight_size, ...)
    /// @param W_hz_ptr   weights with hidden input of input gate, shape =
    /// layer_size x hidden_weight_size
    /// @param W_in_ptr   weights with input of new gate, (input_weight_size,
    /// hidden_weight_size, ...)
    /// @param W_hn_ptr   weights with hidden input of new gate, shape =
    /// layer_size x hidden_weight_size output
    /// @param layer_out_ptr    the last cell per layer output, shape = layer_size
    /// x batch_size x hidden_size
    /// @param hidden_out_ptr   the last layer output for per gru cell, shape =
    /// sequence_length x batch_size x hidden_size
    static void inline run(sycl::nd_item<3> &item, input_T *layer_ptr,
            input_T *h0_ptr, input_T *W_ir_ptr, input_T *W_hr_ptr,
            input_T *W_iz_ptr, input_T *W_hz_ptr, input_T *W_in_ptr,
            input_T *W_hn_ptr, input_T *layer_out_ptr, input_T *hidden_out_ptr,
            input_T *ping_pong_buffer, input_T *ping_pong_cell, int batch_size,
            int input_size, int hidden_size, int sequence_length,
            int layer_size) {
        constexpr uint32_t fused_op_wg_m = wg_tile_m_t;
        constexpr uint32_t fused_op_wg_n = wg_tile_n_t;
        constexpr uint32_t fused_op_sg_m = sg_tile_m_t;
        constexpr uint32_t fused_op_sg_n = sg_tile_n_t;
        constexpr uint32_t fused_op_sg_k = sg_tile_k_t;

        using fused_op = gru_layer<input_T, Act_T, fused_op_wg_m, fused_op_wg_n,
                fused_op_sg_m, fused_op_sg_n, fused_op_sg_k>;

        fused_config_t<input_T> args;
        int layer_input_size = batch_size * input_size;
        int hidden_io_size = batch_size * hidden_size;
        int input_weight_size = input_size * hidden_size;
        int hidden_weight_size = hidden_size * hidden_size;
        int one_layer_size = sequence_length * batch_size * hidden_size;
        int ping = 0;
        int pong = 1;
        args.one_cell_ptr = ping_pong_cell;
        args.input_size = input_size;
        args.batch_size = batch_size;
        args.hidden_size = hidden_size;
        args.sequence_length = sequence_length;
        args.cell_out_ptr = layer_size == 1
                ? hidden_out_ptr
                : (ping_pong_buffer + ping * one_layer_size);
        args.layer_ptr = (layer_ptr);
        args.hx_ptr = (h0_ptr);
        args.layer_output = layer_out_ptr;
        args.W_ir_ptr = (W_ir_ptr);
        args.W_hr_ptr = (W_hr_ptr);
        args.W_iz_ptr = (W_iz_ptr);
        args.W_hz_ptr = (W_hz_ptr);
        args.W_in_ptr = (W_in_ptr);
        args.W_hn_ptr = (W_hn_ptr);
        SW_BARRIER();
        fused_op::call(item, &args);
        ping = (ping + 1) % 2;
        pong = (pong + 1) % 2;

        args.input_size = hidden_size;
        args.batch_size = batch_size;
        args.hidden_size = hidden_size;
        for (uint32_t layer_id = 1; layer_id < layer_size; ++layer_id) {
            args.layer_output = layer_out_ptr + layer_id * hidden_io_size;
            args.hx_ptr = (h0_ptr + layer_id * hidden_io_size);
            args.W_ir_ptr = (W_ir_ptr + (layer_id - 1) * hidden_weight_size
                    + input_weight_size);
            args.W_hr_ptr = (W_hr_ptr + layer_id * hidden_weight_size);
            args.W_iz_ptr = (W_iz_ptr + (layer_id - 1) * hidden_weight_size
                    + input_weight_size);
            args.W_hz_ptr = (W_hz_ptr + layer_id * hidden_weight_size);
            args.W_in_ptr = (W_in_ptr + (layer_id - 1) * hidden_weight_size
                    + input_weight_size);
            args.W_hn_ptr = (W_hn_ptr + layer_id * hidden_weight_size);
            args.cell_out_ptr = layer_id == layer_size - 1
                    ? hidden_out_ptr
                    : (ping_pong_buffer + ping * one_layer_size);
            args.layer_ptr = ((ping_pong_buffer + pong * one_layer_size));
            SW_BARRIER();
            fused_op::call(item, &args);
            ping = (ping + 1) % 2;
            pong = (pong + 1) % 2;
        }
    }
};
