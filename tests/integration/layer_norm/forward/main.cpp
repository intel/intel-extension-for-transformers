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

#include "common.hpp"
#include "kernel_func.hpp"
#include "utils/utils.hpp"
#include "gtest/gtest.h"

using namespace cl::sycl;

template <typename test>
void ln_fwd_run() {

    size_t matrix_m = test::mat_m;
    size_t matrix_n = test::mat_n;
    float drop_out_ratio = 0.2f;
    float drop_out_scale = 1.f / (1.f - drop_out_ratio);
    int size_in = matrix_m * matrix_n;
    int size_out = matrix_m * matrix_n;
    int size_beta = matrix_n;
    int size_gamma = matrix_n;
    int size_mu = matrix_m;
    int size_rs = matrix_m;
    int size_mask = matrix_m * matrix_n;
    int size_bias = matrix_n;
    int size_resAdd = matrix_m * matrix_n;
    // in current design, one group process the entire row
    ASSERT_EQ(matrix_n, test::wg_n);

    using data_type_x = typename test::data_type_x;
    using data_type_y = typename test::data_type_y;
    using data_type_acc = typename test::data_type_acc;
    using data_type_weight = typename test::data_type_weight;

    queue queue {};
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto buffer_x = alloc_device_and_init<data_type_x>(
            size_in,
            [](data_type_x *data, size_t idx) {
                data[idx]
                        = static_cast<data_type_x>((random_float() - 0.5) * 10);
            },
            queue, device, context);
    auto buffer_bias_dropout_res = alloc_device_and_init<data_type_x>(
            size_in,
            [](data_type_x *data, size_t idx) {
                data[idx] = static_cast<data_type_x>(0);
            },
            queue, device, context);
    auto buffer_beta = alloc_device_and_init<data_type_weight>(
            size_beta,
            [](data_type_weight *data, size_t idx) {
                data[idx] = static_cast<data_type_weight>((idx * 3) % 17);
            },
            queue, device, context);
    auto buffer_gamma = alloc_device_and_init<data_type_weight>(
            size_gamma,
            [](data_type_weight *data, size_t idx) {
                data[idx] = static_cast<data_type_weight>((idx * 5) % 13);
            },
            queue, device, context);
    auto buffer_bias = alloc_device_and_init<data_type_x>(
            size_bias,
            [](data_type_x *data, size_t idx) {
                data[idx] = static_cast<data_type_x>(random_float() - 0.5);
            },
            queue, device, context);
    auto buffer_resAdd = alloc_device_and_init<data_type_x>(
            size_resAdd,
            [](data_type_x *data, size_t idx) {
                data[idx]
                        = static_cast<data_type_x>((random_float() - 0.5) * 10);
            },
            queue, device, context);
    auto buffer_mu = alloc_device_and_init<data_type_acc>(
            size_mu,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0);
            },
            queue, device, context);
    auto buffer_rs = alloc_device_and_init<data_type_acc>(
            size_rs,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0);
            },
            queue, device, context);
    auto buffer_y = alloc_device_and_init<data_type_y>(
            size_out,
            [](data_type_y *data, size_t idx) {
                data[idx] = static_cast<data_type_y>(0);
            },
            queue, device, context);
    uint32_t drop_threshold = drop_out_ratio * double(RAND_MAX);
    auto buffer_mask = alloc_device_and_init<uint8_t>(
            size_mask,
            [&drop_threshold](uint8_t *data, size_t idx) {
                data[idx] = static_cast<uint8_t>(
                        (generate_real_random<double>(0.0, double(RAND_MAX))
                                > drop_threshold)
                                ? 0
                                : 1);
            },
            queue, device, context);
    auto buffer_rand_offset = alloc_device_and_init<uint64_t>(
            1,
            [](uint64_t *data, size_t idx) {
                data[idx] = static_cast<uint64_t>(0);
            },
            queue, device, context);

    cl::sycl::range<3> group_range {1, test::wg_num_m, test::wg_num_n};
    cl::sycl::range<3> local_range {1,
            (test::wg_m + test::sg_m - 1) / test::sg_m,
            (test::wg_n + test::sg_n - 1) / test::sg_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    try {
        auto e_esimd = queue.submit([&](handler &cgh) {
            cgh.parallel_for<test>(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                constexpr bool store_for_bwd = true;
                using ln_fwd_func = ln_fwd_func_t<data_type_x, data_type_y,
                        data_type_weight, data_type_acc, test::wg_n, test::wg_m,
                        test::sg_n, test::sg_m, test::chunk_size,
                        test::wg_num_m, test::wg_num_n, test::ln_fused_op_kind,
                        store_for_bwd>;
                constexpr uint32_t slm_size = ln_fwd_func::slm_size;
                constexpr uint32_t barrier_count = ln_fwd_func::barrier_count;

                if constexpr (barrier_count != 0) {
                    xetla_nbarrier_init<barrier_count>();
                }
                if constexpr (slm_size != 0) { xetla_local_init<slm_size>(); }

                ln_fwd_func::call(item, buffer_x, buffer_y, matrix_m, matrix_n,
                        buffer_gamma, buffer_beta, buffer_rs, buffer_mu,
                        matrix_n, matrix_n, buffer_bias, buffer_resAdd,
                        buffer_mask, buffer_rand_offset, drop_out_ratio,
                        drop_out_scale, buffer_bias_dropout_res);
            });
        });
        e_esimd.wait();
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    auto buffer_x_host
            = alloc_host_and_copy<data_type_x>(buffer_x, size_in, queue);
    auto buffer_y_host
            = alloc_host_and_copy<data_type_y>(buffer_y, size_out, queue);
    auto buffer_gamma_host = alloc_host_and_copy<data_type_weight>(
            buffer_gamma, size_gamma, queue);
    auto buffer_beta_host = alloc_host_and_copy<data_type_weight>(
            buffer_beta, size_beta, queue);
    auto buffer_mu_host
            = alloc_host_and_copy<data_type_acc>(buffer_mu, size_mu, queue);
    auto buffer_rs_host
            = alloc_host_and_copy<data_type_acc>(buffer_rs, size_rs, queue);
    auto buffer_bias_host
            = alloc_host_and_copy<data_type_x>(buffer_bias, size_bias, queue);
    auto buffer_resAdd_host = alloc_host_and_copy<data_type_x>(
            buffer_resAdd, size_resAdd, queue);
    auto buffer_mask_host
            = alloc_host_and_copy<uint8_t>(buffer_mask, size_mask, queue);
    auto buffer_bias_dropout_res_host = alloc_host_and_copy<data_type_x>(
            buffer_bias_dropout_res, size_in, queue);

    // validation
    ASSERT_EQ(0,
            (ln_fwd_result_validate<data_type_x, data_type_weight, data_type_y,
                    data_type_acc>(buffer_x_host, buffer_y_host,
                    buffer_gamma_host, buffer_beta_host, buffer_mu_host,
                    buffer_rs_host, matrix_m, matrix_n, matrix_n, test::sg_n,
                    buffer_bias_host, buffer_resAdd_host, buffer_mask_host,
                    buffer_bias_dropout_res_host, drop_out_scale,
                    test::ln_fused_op_kind)));

    free(buffer_x, context);
    free(buffer_beta, context);
    free(buffer_gamma, context);
    free(buffer_mu, context);
    free(buffer_rs, context);
    free(buffer_y, context);
    free(buffer_mask, context);
    free(buffer_resAdd, context);
    free(buffer_bias, context);
    free(buffer_rand_offset, context);

    free(buffer_x_host);
    free(buffer_y_host);
    free(buffer_gamma_host);
    free(buffer_beta_host);
    free(buffer_mu_host);
    free(buffer_rs_host);
    free(buffer_bias_host);
    free(buffer_resAdd_host);
    free(buffer_mask_host);
    free(buffer_bias_dropout_res_host);
}

TEST(ln_fwd_0, esimd) {
    ln_fwd_run<ln_fwd_0>();
}

TEST(ln_fwd_1, esimd) {
    ln_fwd_run<ln_fwd_1>();
}

TEST(ln_fwd_2, esimd) {
    ln_fwd_run<ln_fwd_2>();
}

TEST(ln_fwd_3, esimd) {
    ln_fwd_run<ln_fwd_3>();
}

TEST(ln_fwd_4, esimd) {
    ln_fwd_run<ln_fwd_4>();
}

TEST(ln_fwd_5, esimd) {
    ln_fwd_run<ln_fwd_5>();
}

TEST(ln_fwd_6, esimd) {
    ln_fwd_run<ln_fwd_6>();
}

TEST(ln_fwd_7, esimd) {
    ln_fwd_run<ln_fwd_7>();
}

TEST(ln_fwd_8, esimd) {
    ln_fwd_run<ln_fwd_8>();
}

TEST(ln_fwd_0_fp16, esimd) {
    ln_fwd_run<ln_fwd_0_fp16>();
}

TEST(ln_fwd_1_fp16, esimd) {
    ln_fwd_run<ln_fwd_1_fp16>();
}

TEST(ln_fwd_2_fp16, esimd) {
    ln_fwd_run<ln_fwd_2_fp16>();
}

TEST(ln_fwd_3_fp16, esimd) {
    ln_fwd_run<ln_fwd_3_fp16>();
}

TEST(ln_fwd_0_chunked, esimd) {
    ln_fwd_run<ln_fwd_0_chunked>();
}

TEST(ln_fwd_1_chunked, esimd) {
    ln_fwd_run<ln_fwd_1_chunked>();
}

TEST(ln_fwd_2_chunked, esimd) {
    ln_fwd_run<ln_fwd_2_chunked>();
}
