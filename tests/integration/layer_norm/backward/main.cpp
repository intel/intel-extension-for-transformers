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

    queue Queue {};
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    data_type_y *dy_in = static_cast<data_type_y *>(
            malloc_shared(size_dy_in * sizeof(data_type_y), Device, Context));
    data_type_y *grad_in = static_cast<data_type_y *>(
            malloc_shared(size_grad_in * sizeof(data_type_y), Device, Context));
    data_type_x *x_in = static_cast<data_type_x *>(
            malloc_shared(size_x_in * sizeof(data_type_x), Device, Context));
    data_type_weight *gamma_in = static_cast<data_type_weight *>(malloc_shared(
            size_gamma_in * sizeof(data_type_weight), Device, Context));
    data_type_acc *buffer_mu = static_cast<data_type_acc *>(
            malloc_shared(size_mu * sizeof(data_type_acc), Device, Context));
    data_type_acc *buffer_rs = static_cast<data_type_acc *>(
            malloc_shared(size_rs * sizeof(data_type_acc), Device, Context));

    data_type_x *dx_out = static_cast<data_type_x *>(
            malloc_shared(size_dx_out * sizeof(data_type_x), Device, Context));
    data_type_x *dx_resAdd_out = static_cast<data_type_x *>(malloc_shared(
            size_dx_resAdd_out * sizeof(data_type_x), Device, Context));
    data_type_acc *dgamma_acc = static_cast<data_type_acc *>(malloc_shared(
            size_dgamma_acc * sizeof(data_type_acc), Device, Context));
    data_type_acc *dbeta_acc = static_cast<data_type_acc *>(malloc_shared(
            size_dbeta_acc * sizeof(data_type_acc), Device, Context));
    data_type_acc *dbias_acc = static_cast<data_type_acc *>(malloc_shared(
            size_dbias_acc * sizeof(data_type_acc), Device, Context));

    data_type_weight *dgamma = static_cast<data_type_weight *>(malloc_shared(
            size_dgamma * sizeof(data_type_weight), Device, Context));
    data_type_weight *dbeta = static_cast<data_type_weight *>(malloc_shared(
            size_dbeta * sizeof(data_type_weight), Device, Context));
    data_type_x *dbias = static_cast<data_type_x *>(
            malloc_shared(size_dbias * sizeof(data_type_x), Device, Context));
    uint8_t *buffer_mask = static_cast<uint8_t *>(
            malloc_shared(size_mask * sizeof(uint8_t), Device, Context));

    for (unsigned i = 0; i < size_dy_in; ++i) {
        dy_in[i] = (random_float() - 0.5) * 10;
    }
    for (unsigned i = 0; i < size_grad_in; ++i) {
        grad_in[i] = (random_float() - 0.5) * 10;
    }
    for (unsigned i = 0; i < size_x_in; ++i) {
        x_in[i] = (random_float() - 0.5) * 10;
    }
    for (unsigned i = 0; i < size_gamma_in; ++i) {
        gamma_in[i] = (i * 5) % 13;
    }
    for (unsigned i = 0; i < size_mu; ++i) {
        buffer_mu[i] = (i * 3) % 7;
    }
    for (unsigned i = 0; i < size_rs; ++i) {
        buffer_rs[i] = random_float();
    }

    uint32_t drop_threshold = drop_out_prob * double(RAND_MAX);
    for (unsigned i = 0; i < size_mask; ++i) {
        buffer_mask[i] = (generate_random<double>(0.0, double(RAND_MAX))
                                 > drop_threshold)
                ? 0
                : 1;
    }
    for (unsigned i = 0; i < size_dx_out; ++i) {
        dx_out[i] = data_type_x(0);
    }
    for (unsigned i = 0; i < size_dgamma_acc; ++i) {
        dgamma_acc[i] = data_type_acc(0);
    }
    for (unsigned i = 0; i < size_dbeta_acc; ++i) {
        dbeta_acc[i] = data_type_acc(0);
    }
    for (unsigned i = 0; i < size_dbias_acc; ++i) {
        dbias_acc[i] = data_type_acc(0);
    }
    for (unsigned i = 0; i < size_dgamma; ++i) {
        dgamma[i] = data_type_weight(0);
    }
    for (unsigned i = 0; i < size_dbeta; ++i) {
        dbeta[i] = data_type_weight(0);
    }
    for (unsigned i = 0; i < size_dbias; ++i) {
        dbias[i] = data_type_x(0);
    }

    cl::sycl::range<3> GroupRange {1, test::wg_num_m, test::wg_num_n};
    cl::sycl::range<3> LocalRange {1,
            (test::wg_m + test::sg_m - 1) / test::sg_m,
            (test::wg_n + test::sg_n - 1) / test::sg_n};
    cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

    // 3 buffers.
    cl::sycl::range<3> final_GroupRange {
            3, 1, (final_mat_n + final_wg_n - 1) / final_wg_n};
    cl::sycl::range<3> final_LocalRange {1,
            (final_wg_m + final_sg_m - 1) / final_sg_m,
            (final_wg_n + final_sg_n - 1) / final_sg_n};
    cl::sycl::nd_range<3> final_Range(
            final_GroupRange * final_LocalRange, final_LocalRange);

    try {
        auto e_esimd_bwd0 = Queue.submit([&](handler &cgh) {
            cgh.parallel_for<
                    test>(Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
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

                xetla_exec_item<3> ei(item);
                ln_bwd_func::call(ei, dy_in, x_in, gamma_in, matrix_m, matrix_n,
                        matrix_n, buffer_rs, buffer_mu, dx_out, dgamma_acc,
                        dbeta_acc, buffer_mask, dx_resAdd_out, dbias_acc,
                        matrix_n, drop_out_scale_inv, drop_out_prob, grad_in);
            });
        });
        e_esimd_bwd0.wait();

        auto e_esimd_bwd1 = Queue.submit([&](handler &cgh) {
            cgh.parallel_for(
                    final_Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                        using ln_bwd_final_func = ln_bwd_final_func_t<
                                data_type_x, data_type_weight, data_type_acc,
                                final_wg_n, final_wg_m, final_sg_n, final_sg_m>;
                        constexpr uint32_t slm_size
                                = ln_bwd_final_func::slm_size;
                        constexpr uint32_t barrier_count
                                = ln_bwd_final_func::barrier_count;
                        if constexpr (slm_size != 0) {
                            xetla_local_init<slm_size>();
                        }
                        if constexpr (barrier_count != 0) {
                            xetla_nbarrier_init<barrier_count>();
                        }

                        xetla_exec_item<3> ei(item);
                        ln_bwd_final_func::call(ei, dgamma_acc, dbeta_acc,
                                dgamma, dbeta, final_mat_m, final_mat_n,
                                dbias_acc, dbias);
                    });
        });
        e_esimd_bwd1.wait();
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    // validation
    ASSERT_EQ(0,
            (ln_bwd_result_validate<data_type_x, data_type_weight, data_type_y,
                    data_type_acc>(dy_in, grad_in, x_in, gamma_in, buffer_mu,
                    buffer_rs, dx_out, dgamma, dbeta, matrix_m, matrix_n,
                    matrix_n, test::sg_n, dx_resAdd_out, dbias, buffer_mask,
                    drop_out_scale_inv, test::ln_fused_op_kind)));

    free(dy_in, Context);
    free(x_in, Context);
    free(gamma_in, Context);
    free(buffer_mu, Context);
    free(buffer_rs, Context);
    free(dx_out, Context);
    free(dgamma_acc, Context);
    free(dbeta_acc, Context);
    free(dgamma, Context);
    free(dbeta, Context);
    free(dbias_acc, Context);
    free(dx_resAdd_out, Context);
    free(dbias, Context);
    free(buffer_mask, Context);
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
