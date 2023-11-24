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
#include "kernel_func.hpp"
#include "tests/utils/utils.hpp"

using namespace cl::sycl;
using namespace gpu::xetla;

template <typename data_type>
int validation(data_type *layer_inputs_device, data_type *h0_inputs_device,
        std::vector<data_type *> i_weights, std::vector<data_type *> h_weights,
        data_type *hidden_outputs_device, data_type *layer_outputs_device,
        uint32_t batch_size, uint32_t input_size, uint32_t hidden_size,
        uint32_t sequence_length, sycl::queue &queue, uint32_t layer_size = 1) {
    uint32_t layer_input_size = batch_size * input_size;
    uint32_t hidden_io_size = batch_size * hidden_size;
    uint32_t i_weight_size = input_size * hidden_size;
    uint32_t h_weight_size = hidden_size * hidden_size;
    uint32_t one_layer_size = hidden_io_size * sequence_length;
    uint32_t input_weight_size = input_size * hidden_size
            + (layer_size - 1) * hidden_size * hidden_size;

    data_type *ir_weights_device = i_weights[0];
    data_type *iz_weights_device = i_weights[1];
    data_type *in_weights_device = i_weights[2];

    data_type *hr_weights_device = h_weights[0];
    data_type *hz_weights_device = h_weights[1];
    data_type *hn_weights_device = h_weights[2];

    auto layer_inputs = alloc_host_and_copy<data_type>(
            layer_inputs_device, layer_input_size, queue);
    auto h0_inputs = alloc_host_and_copy<data_type>(
            h0_inputs_device, layer_size * batch_size * hidden_size, queue);
    auto hidden_outputs = alloc_host_and_copy<data_type>(hidden_outputs_device,
            sequence_length * batch_size * hidden_size, queue);
    auto layer_outputs = alloc_host_and_copy<data_type>(
            layer_outputs_device, layer_size * batch_size * hidden_size, queue);
    auto ir_weights = alloc_host_and_copy<data_type>(
            ir_weights_device, input_weight_size, queue);
    auto iz_weights = alloc_host_and_copy<data_type>(
            iz_weights_device, input_weight_size, queue);
    auto in_weights = alloc_host_and_copy<data_type>(
            in_weights_device, input_weight_size, queue);
    auto hr_weights = alloc_host_and_copy<data_type>(
            hr_weights_device, layer_size * hidden_size * hidden_size, queue);
    auto hz_weights = alloc_host_and_copy<data_type>(
            hz_weights_device, layer_size * hidden_size * hidden_size, queue);
    auto hn_weights = alloc_host_and_copy<data_type>(
            hn_weights_device, layer_size * hidden_size * hidden_size, queue);

    data_type *ping_pong_result
            = (data_type *)malloc(2 * one_layer_size * sizeof(data_type));
    data_type *ping_pong_cell
            = (data_type *)malloc(2 * hidden_io_size * sizeof(data_type));

    memset(ping_pong_result, 0, 2 * one_layer_size * sizeof(data_type));
    memset(ping_pong_cell, 0, 2 * hidden_io_size * sizeof(data_type));

    uint32_t err_cnt = 0;
    for (uint32_t layer_id = 0; layer_id < layer_size; layer_id++) {
        // inputs
        data_type *hidden_input = h0_inputs + layer_id * hidden_io_size;
        data_type *layer_input;
        if (layer_id == 0) {
            layer_input = layer_inputs;
        } else {
            layer_input = ping_pong_result + (layer_id % 2) * one_layer_size;
        }
        // i_weights and h_weights
        data_type *w_ir;
        data_type *w_iz;
        data_type *w_in;
        data_type *w_hr;
        data_type *w_hz;
        data_type *w_hn;
        if (layer_id == 0) {
            w_ir = ir_weights;
            w_iz = iz_weights;
            w_in = in_weights;
            w_hr = hr_weights;
            w_hz = hz_weights;
            w_hn = hn_weights;
        } else {
            w_ir = ir_weights + (layer_id - 1) * h_weight_size + i_weight_size;
            w_iz = iz_weights + (layer_id - 1) * h_weight_size + i_weight_size;
            w_in = in_weights + (layer_id - 1) * h_weight_size + i_weight_size;
            w_hr = hr_weights + layer_id * h_weight_size;
            w_hz = hz_weights + layer_id * h_weight_size;
            w_hn = hn_weights + layer_id * h_weight_size;
        }
        // output
        data_type *hidden_out = layer_id == layer_size - 1
                ? hidden_outputs
                : ping_pong_result + ((layer_id + 1) % 2) * one_layer_size;
        input_size = layer_id ? hidden_size : input_size;
        for (uint32_t seq = 0; seq < sequence_length; seq++) {
            data_type *one_cell = ping_pong_cell + (seq % 2) * hidden_io_size;
            for (uint32_t i = 0; i < batch_size; i++) {
                for (uint32_t j = 0; j < hidden_size; j++) {
                    float reset_gate = 0.0f;
                    float input_gate = 0.0f;
                    float new_gate = 0.0f;
                    float hgate_2 = 0.0f;
                    float result = 0.0f;
                    for (uint32_t kk = 0; kk < input_size; kk++) {
                        data_type a_temp = layer_input[i * input_size + kk];
                        data_type wi_0 = w_ir[kk * hidden_size + j];
                        data_type wi_1 = w_iz[kk * hidden_size + j];
                        data_type wi_2 = w_in[kk * hidden_size + j];
                        reset_gate = float(reset_gate)
                                + float(a_temp) * float(wi_0);
                        input_gate = float(input_gate)
                                + float(a_temp) * float(wi_1);
                        new_gate
                                = float(new_gate) + float(a_temp) * float(wi_2);
                    }
                    for (uint32_t kk = 0; kk < hidden_size; kk++) {
                        data_type b_temp = hidden_input[i * hidden_size + kk];
                        data_type wh_0 = w_hr[kk * hidden_size + j];
                        data_type wh_1 = w_hz[kk * hidden_size + j];
                        data_type wh_2 = w_hn[kk * hidden_size + j];
                        reset_gate = float(reset_gate)
                                + float(b_temp) * float(wh_0);
                        input_gate = float(input_gate)
                                + float(b_temp) * float(wh_1);
                        hgate_2 += float(b_temp) * float(wh_2);
                    }
                    reset_gate = 1 / (1 + std::exp(-float(reset_gate)));
                    input_gate = 1 / (1 + std::exp(-float(input_gate)));
                    new_gate = std::tanh(float(new_gate)
                            + float(reset_gate) * float(hgate_2));
                    result = float(1 - float(input_gate)) * float(new_gate)
                            + float(input_gate)
                                    * float(hidden_input[i * hidden_size + j]);
                    one_cell[hidden_size * i + j] = result;
                    if (layer_id != layer_size - 1) {
                        hidden_out[i * hidden_size + j] = result;
                    } else {
                        if (std::abs((float(result)
                                             - float(hidden_out[i * hidden_size
                                                     + j]))
                                    / float(hidden_out[i * hidden_size + j]))
                                > 0.03) {
                            if (++err_cnt < 100) {
                                std::cout << "failed at (" << layer_id << ", "
                                          << seq << ", " << i << ", " << j
                                          << "), "
                                          << " golden: " << float(result)
                                          << " != GPU: "
                                          << float(hidden_out[i * hidden_size
                                                     + j])
                                          << "\n";
                            }
                        }
                    }
                }
            }
            if (layer_id == 0) {
                layer_input += layer_input_size;
            } else {
                layer_input += hidden_io_size;
            }
            hidden_out += hidden_io_size;
            hidden_input = one_cell;
        }
    }
    free(ping_pong_cell);
    free(ping_pong_result);

    free(layer_inputs);
    free(h0_inputs);
    free(hidden_outputs);
    free(layer_outputs);
    free(ir_weights);
    free(iz_weights);
    free(in_weights);
    free(hr_weights);
    free(hz_weights);
    free(hn_weights);

    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(layer_size * batch_size * hidden_size
                                      * sequence_length
                              - err_cnt)
                             / (float)(layer_size * batch_size * hidden_size
                                     * sequence_length))
                        * 100.0f
                  << "% ("
                  << (layer_size * batch_size * hidden_size * sequence_length
                             - err_cnt)
                  << "/"
                  << layer_size * batch_size * hidden_size * sequence_length
                  << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt > 0 ? 1 : 0;
}

template <typename gru_config>
void gru_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    using data_type = typename gru_config::dtype_in;
    using data_type_act = typename gru_config::dtype_acc;

    size_t S = gru_config::layer_size;
    size_t L = gru_config::sequence_length;
    size_t N = gru_config::batch_size;
    size_t F = gru_config::input_size;
    size_t H = gru_config::hidden_size;

    // config
    size_t wg_tile_m = gru_config::wg_tile_m;
    size_t wg_tile_n = gru_config::wg_tile_n;
    size_t sg_tile_m = gru_config::sg_tile_m;
    size_t sg_tile_n = gru_config::sg_tile_n;
    size_t sg_tile_k = gru_config::sg_tile_k;
    // input and output size
    size_t layer_input_size = L * N * F;
    size_t hidden_input_size = S * (L + 1) * N * H;
    size_t input_weight_size = F * H + (S - 1) * H * H;
    size_t hidden_weight_size = S * H * H;
    size_t gate_size = S * L * N * H;

    //***********dpcpp runtime setup && buffer allocation start ************//
    // Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";
    std::vector<data_type *> i_weights, h_weights;
    /// malloc for inputs

    auto layer_inputs = alloc_device_and_init<data_type>(
            layer_input_size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(random_float());
            },
            queue, device, context);
    auto h0_inputs = alloc_device_and_init<data_type>(
            S * N * H,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(random_float());
            },
            queue, device, context);
    auto ir_weights = alloc_device_and_init<data_type>(
            input_weight_size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(0.001 * random_float());
            },
            queue, device, context);

    i_weights.push_back(ir_weights);

    auto iz_weights = alloc_device_and_init<data_type>(
            input_weight_size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(0.001 * random_float());
            },
            queue, device, context);

    i_weights.push_back(iz_weights);

    auto in_weights = alloc_device_and_init<data_type>(
            input_weight_size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(0.0001 * random_float());
            },
            queue, device, context);

    i_weights.push_back(in_weights);

    auto hr_weights = alloc_device_and_init<data_type>(
            hidden_weight_size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(0.001 * random_float());
            },
            queue, device, context);

    h_weights.push_back(hr_weights);

    auto hz_weights = alloc_device_and_init<data_type>(
            hidden_weight_size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(0.001 * random_float());
            },
            queue, device, context);

    h_weights.push_back(hz_weights);

    auto hn_weights = alloc_device_and_init<data_type>(
            hidden_weight_size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(0.001 * random_float());
            },
            queue, device, context);

    h_weights.push_back(hn_weights);

    auto hidden_outputs = alloc_device_and_init<data_type>(
            L * N * H, [](data_type *data, size_t idx) {}, queue, device,
            context);

    auto layer_outputs = alloc_device_and_init<data_type>(
            S * N * H, [](data_type *data, size_t idx) {}, queue, device,
            context);

    auto ping_pong_buffer = alloc_device_and_init<data_type>(
            2 * L * N * H, [](data_type *data, size_t idx) {}, queue, device,
            context);

    auto ping_pong_cell = alloc_device_and_init<data_type>(
            2 * N * H, [](data_type *data, size_t idx) {}, queue, device,
            context);

    //***********dpcpp runtime setup && buffer allocation start ************//

    cl::sycl::range<3> group_range {1, (N + wg_tile_m - 1) / wg_tile_m, 1};
    cl::sycl::range<3> local_range {1, (wg_tile_m + sg_tile_m - 1) / sg_tile_m,
            (wg_tile_n + sg_tile_n - 1) / sg_tile_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    std::cout << "Launch kernel:\n";
    std::cout << "group_num_x: " << 1
              << ", group_num_y: " << (N + wg_tile_m - 1) / wg_tile_m << "\n";
    std::cout << "group_size_x: " << (wg_tile_n + sg_tile_n - 1) / sg_tile_n
              << ", group_size_y: " << (wg_tile_m + sg_tile_m - 1) / sg_tile_m
              << std::endl;
    uint32_t warmup = 10;
    long ops = 3 * L * (2 * N * F * H + 2 * N * H * H)
            + (S - 1) * 12 * L * N * H * H;
    profiling_helper prof("gru", ops, "gflops");
    // esimd kernel prepratation and execution
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            cgh.parallel_for<gru_config>(
                    nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                        using xcoder_gru_op = kernel_xcoder_gru_fusion<
                                typename gru_config::dtype_in,
                                typename gru_config::dtype_acc,
                                gru_config::wg_tile_m, gru_config::wg_tile_n,
                                gru_config::sg_tile_m, gru_config::sg_tile_n,
                                gru_config::sg_tile_k>;
                        xcoder_gru_op::run(item, layer_inputs, h0_inputs,
                                ir_weights, hr_weights, iz_weights, hz_weights,
                                in_weights, hn_weights, layer_outputs,
                                hidden_outputs, ping_pong_buffer,
                                ping_pong_cell, gru_config::batch_size,
                                gru_config::input_size, gru_config::hidden_size,
                                gru_config::sequence_length,
                                gru_config::layer_size);
                    });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
        }
    }

    ASSERT_EQ(0,
            validation<data_type>(layer_inputs, h0_inputs, i_weights, h_weights,
                    hidden_outputs, layer_outputs, N, F, H, L, queue, S));

    // performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(layer_inputs, context);
    free(h0_inputs, context);

    free(ir_weights, context);
    free(iz_weights, context);
    free(in_weights, context);

    free(hr_weights, context);
    free(hz_weights, context);
    free(hn_weights, context);

    free(hidden_outputs, context);
    free(layer_outputs, context);
    free(ping_pong_buffer, context);
    free(ping_pong_cell, context);
}

int main() {
    // The purpose of this example is to illustrate the fusion API in XeTLA.
    // It allows the user to apply GRU based XeTLA.
    // Here provides some possible configurations using epilogue_t:
    // 2 GEMMs:
    //   r_t = X_t x W_ir + h_{t - 1} x W_hr
    // GEMM+sigmoid:
    //   r_t = sigmoid(r_t)
    // 2 GEMMs:
    //   n_t = X_t x W_in + r_t * (h_{t - 1} x W_hn)
    // GEMM+tanh:
    //   n_t = tanh(n_t)
    // 2 GEMMs:
    //   z_t = X_t x W_iz + h_{t - 1} x W_hz
    // GEMM+sigmoid:
    //   z_t = sigmoid(z_t)
    // result
    //   h_t = (1 - z_t) n_t + z_t h_{t - 1}
    // This example will implement the elementwise
    // operations, which demonstrates its maximal flexibility.
    // checkout op_functor.hpp & op.hpp for more elementwise ops

    // Note:
    //    x denotes matrix multiply and * denotes elementwise multiply
    gru_run<gru_config_t>(10);
    return 0;
}
