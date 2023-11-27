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
void ln_bwd_run() {

    size_t matrix_m = test::mat_m;
    size_t matrix_n = test::mat_n;

    uint32_t final_mat_m = test::final_mat_m;
    uint32_t final_mat_n = test::final_mat_n;
    constexpr uint32_t final_wg_m = test::final_wg_m;
    constexpr uint32_t final_wg_n = test::final_wg_n;
    constexpr uint32_t final_sg_m = test::final_sg_m;
    constexpr uint32_t final_sg_n = test::final_sg_n;
    float drop_out_prob = 0.5f;
    float drop_out_scale_inv = 1.f - drop_out_prob;
    int size_dy_in = matrix_m * matrix_n;
    int size_x_in = matrix_m * matrix_n;
    int size_gamma_in = matrix_n;
    int size_mu = matrix_m;
    int size_rs = matrix_m;
    int size_mask = matrix_m * matrix_n;
    int size_dx_out = matrix_m * matrix_n;
    int size_grad_in = matrix_m * matrix_n;
    int size_dx_resAdd_out = matrix_m * matrix_n;
    int size_dgamma_acc = test::wg_num_m * matrix_n;
    int size_dbeta_acc = test::wg_num_m * matrix_n;
    int size_dbias_acc = test::wg_num_m * matrix_n;

    int size_dgamma = matrix_n;
    int size_dbeta = matrix_n;
    int size_dbias = matrix_n;
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

    auto dy_in = alloc_device_and_init<data_type_y>(
            size_dy_in,
            [](data_type_y *data, size_t idx) {
                data[idx]
                        = static_cast<data_type_y>((random_float() - 0.5) * 10);
            },
            queue, device, context);
    auto grad_in = alloc_device_and_init<data_type_y>(
            size_grad_in,
            [](data_type_y *data, size_t idx) {
                data[idx]
                        = static_cast<data_type_y>((random_float() - 0.5) * 10);
            },
            queue, device, context);
    auto x_in = alloc_device_and_init<data_type_x>(
            size_x_in,
            [](data_type_x *data, size_t idx) {
                data[idx]
                        = static_cast<data_type_x>((random_float() - 0.5) * 10);
            },
            queue, device, context);
    auto gamma_in = alloc_device_and_init<data_type_weight>(
            size_gamma_in,
            [](data_type_weight *data, size_t idx) {
                data[idx] = static_cast<data_type_weight>((idx * 5) % 13);
            },
            queue, device, context);
    auto buffer_mu = alloc_device_and_init<data_type_acc>(
            size_mu,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>((idx * 3) % 7);
            },
            queue, device, context);
    auto buffer_rs = alloc_device_and_init<data_type_acc>(
            size_rs,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(random_float());
            },
            queue, device, context);
    auto dx_out = alloc_device_and_init<data_type_x>(
            size_dx_out,
            [](data_type_x *data, size_t idx) {
                data[idx] = static_cast<data_type_x>(0);
            },
            queue, device, context);
    auto dx_resAdd_out = alloc_device_and_init<data_type_x>(
            size_dx_resAdd_out,
            [](data_type_x *data, size_t idx) {
                data[idx] = static_cast<data_type_x>(0);
            },
            queue, device, context);
    auto dgamma_acc = alloc_device_and_init<data_type_acc>(
            size_dgamma_acc,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0);
            },
            queue, device, context);
    auto dbeta_acc = alloc_device_and_init<data_type_acc>(
            size_dbeta_acc,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0);
            },
            queue, device, context);
    auto dbias_acc = alloc_device_and_init<data_type_acc>(
            size_dbias_acc,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0);
            },
            queue, device, context);
    auto dgamma = alloc_device_and_init<data_type_weight>(
            size_dgamma,
            [](data_type_weight *data, size_t idx) {
                data[idx] = static_cast<data_type_weight>(0);
            },
            queue, device, context);
    auto dbeta = alloc_device_and_init<data_type_weight>(
            size_dbeta,
            [](data_type_weight *data, size_t idx) {
                data[idx] = static_cast<data_type_weight>(0);
            },
            queue, device, context);
    auto dbias = alloc_device_and_init<data_type_x>(
            size_dbias,
            [](data_type_x *data, size_t idx) {
                data[idx] = static_cast<data_type_x>(0);
            },
            queue, device, context);

    uint32_t drop_threshold = drop_out_prob * double(RAND_MAX);
    auto buffer_mask = alloc_device_and_init<uint8_t>(
            size_mask,
            [&drop_threshold](uint8_t *data, size_t idx) {
                data[idx] = (generate_real_random<double>(0.0, double(RAND_MAX))
                                    > drop_threshold)
                        ? 0
                        : 1;
            },
            queue, device, context);

    cl::sycl::range<3> group_range {1, test::wg_num_m, test::wg_num_n};
    cl::sycl::range<3> local_range {1,
            (test::wg_m + test::sg_m - 1) / test::sg_m,
            (test::wg_n + test::sg_n - 1) / test::sg_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    // 3 buffers.
    cl::sycl::range<3> final_group_range {
            3, 1, (final_mat_n + final_wg_n - 1) / final_wg_n};
    cl::sycl::range<3> final_local_range {1,
            (final_wg_m + final_sg_m - 1) / final_sg_m,
            (final_wg_n + final_sg_n - 1) / final_sg_n};
    cl::sycl::nd_range<3> final_range(
            final_group_range * final_local_range, final_local_range);

    try {
        auto e_esimd_bwd0 = queue.submit([&](handler &cgh) {
            cgh.parallel_for<test>(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                using ln_bwd_func = ln_bwd_func_t<data_type_y, data_type_x,
                        data_type_weight, data_type_acc, test::wg_n, test::wg_m,
                        test::sg_n, test::sg_m, test::wg_num_m, test::wg_num_n,
                        test::ln_fused_op_kind>;

                constexpr uint32_t slm_size = ln_bwd_func::slm_size;
                constexpr uint32_t barrier_count = ln_bwd_func::barrier_count;
                if constexpr (barrier_count != 0) {
                    xetla_nbarrier_init<barrier_count>();
                }
                if constexpr (slm_size != 0) { xetla_local_init<slm_size>(); }

                ln_bwd_func::call(item, dy_in, x_in, gamma_in, matrix_m,
                        matrix_n, matrix_n, buffer_rs, buffer_mu, dx_out,
                        dgamma_acc, dbeta_acc, buffer_mask, dx_resAdd_out,
                        dbias_acc, matrix_n, drop_out_scale_inv, drop_out_prob,
                        grad_in);
            });
        });
        e_esimd_bwd0.wait();

        auto e_esimd_bwd1 = queue.submit([&](handler &cgh) {
            cgh.parallel_for(final_range, [=](nd_item<3> item) KERNEL_MAIN {
                using ln_bwd_final_func = ln_bwd_final_func_t<data_type_x,
                        data_type_weight, data_type_acc, final_wg_n, final_wg_m,
                        final_sg_n, final_sg_m>;
                constexpr uint32_t slm_size = ln_bwd_final_func::slm_size;
                constexpr uint32_t barrier_count
                        = ln_bwd_final_func::barrier_count;
                if constexpr (slm_size != 0) { xetla_local_init<slm_size>(); }
                if constexpr (barrier_count != 0) {
                    xetla_nbarrier_init<barrier_count>();
                }

                ln_bwd_final_func::call(item, dgamma_acc, dbeta_acc, dgamma,
                        dbeta, final_mat_m, final_mat_n, dbias_acc, dbias);
            });
        });
        e_esimd_bwd1.wait();
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    auto dy_in_host
            = alloc_host_and_copy<data_type_y>(dy_in, size_dy_in, queue);
    auto grad_in_host
            = alloc_host_and_copy<data_type_y>(grad_in, size_grad_in, queue);
    auto x_in_host = alloc_host_and_copy<data_type_x>(x_in, size_x_in, queue);
    auto gamma_in_host = alloc_host_and_copy<data_type_weight>(
            gamma_in, size_gamma_in, queue);
    auto buffer_mu_host
            = alloc_host_and_copy<data_type_acc>(buffer_mu, size_mu, queue);
    auto buffer_rs_host
            = alloc_host_and_copy<data_type_acc>(buffer_rs, size_rs, queue);
    auto dx_out_host
            = alloc_host_and_copy<data_type_x>(dx_out, size_dx_out, queue);
    auto dgamma_host
            = alloc_host_and_copy<data_type_weight>(dgamma, size_dgamma, queue);
    auto dbeta_host
            = alloc_host_and_copy<data_type_weight>(dbeta, size_dbeta, queue);
    auto dx_resAdd_out_host = alloc_host_and_copy<data_type_x>(
            dx_resAdd_out, size_dx_resAdd_out, queue);
    auto dbias_host
            = alloc_host_and_copy<data_type_x>(dbias, size_dbias, queue);
    auto buffer_mask_host
            = alloc_host_and_copy<uint8_t>(buffer_mask, size_mask, queue);

    // validation
    ASSERT_EQ(0,
            (ln_bwd_result_validate<data_type_x, data_type_weight, data_type_y,
                    data_type_acc>(dy_in_host, grad_in_host, x_in_host,
                    gamma_in_host, buffer_mu_host, buffer_rs_host, dx_out_host,
                    dgamma_host, dbeta_host, matrix_m, matrix_n, matrix_n,
                    test::sg_n, dx_resAdd_out_host, dbias_host,
                    buffer_mask_host, drop_out_scale_inv,
                    test::ln_fused_op_kind)));

    free(dy_in, context);
    free(x_in, context);
    free(gamma_in, context);
    free(buffer_mu, context);
    free(buffer_rs, context);
    free(dx_out, context);
    free(dgamma_acc, context);
    free(dbeta_acc, context);
    free(dgamma, context);
    free(dbeta, context);
    free(dbias_acc, context);
    free(dx_resAdd_out, context);
    free(dbias, context);
    free(buffer_mask, context);

    free(dy_in_host);
    free(grad_in_host);
    free(x_in_host);
    free(gamma_in_host);
    free(buffer_mu_host);
    free(buffer_rs_host);
    free(dx_out_host);
    free(dgamma_host);
    free(dbeta_host);
    free(dx_resAdd_out_host);
    free(dbias_host);
    free(buffer_mask_host);
}

TEST(ln_bwd_0_bf16, esimd) {
    ln_bwd_run<ln_bwd_0_bf16>();
}

TEST(ln_bwd_1_bf16, esimd) {
    ln_bwd_run<ln_bwd_1_bf16>();
}

TEST(ln_bwd_2_bf16, esimd) {
    ln_bwd_run<ln_bwd_2_bf16>();
}

TEST(ln_bwd_0_fp16, esimd) {
    ln_bwd_run<ln_bwd_0_fp16>();
}

TEST(ln_bwd_1_fp16, esimd) {
    ln_bwd_run<ln_bwd_1_fp16>();
}

TEST(ln_bwd_2_fp16, esimd) {
    ln_bwd_run<ln_bwd_2_fp16>();
}

TEST(ln_bwd_3_bf16, esimd) {
    ln_bwd_run<ln_bwd_3_bf16>();
}

TEST(ln_bwd_3_fp16, esimd) {
    ln_bwd_run<ln_bwd_3_fp16>();
}

TEST(ln_bwd_4_bf16, esimd) {
    ln_bwd_run<ln_bwd_4_bf16>();
}
