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

    queue Queue {};
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    data_type_x *buffer_x = static_cast<data_type_x *>(
            malloc_shared(size_in * sizeof(data_type_x), Device, Context));
    data_type_x *buffer_bias_dropout_res = static_cast<data_type_x *>(
            malloc_shared(size_in * sizeof(data_type_x), Device, Context));
    data_type_weight *buffer_beta
            = static_cast<data_type_weight *>(malloc_shared(
                    size_beta * sizeof(data_type_weight), Device, Context));
    data_type_weight *buffer_gamma
            = static_cast<data_type_weight *>(malloc_shared(
                    size_gamma * sizeof(data_type_weight), Device, Context));
    data_type_x *buffer_bias = static_cast<data_type_x *>(
            malloc_shared(size_bias * sizeof(data_type_x), Device, Context));
    data_type_x *buffer_resAdd = static_cast<data_type_x *>(
            malloc_shared(size_resAdd * sizeof(data_type_x), Device, Context));
    data_type_acc *buffer_mu = static_cast<data_type_acc *>(
            malloc_shared(size_mu * sizeof(data_type_acc), Device, Context));
    data_type_acc *buffer_rs = static_cast<data_type_acc *>(
            malloc_shared(size_rs * sizeof(data_type_acc), Device, Context));
    data_type_y *buffer_y = static_cast<data_type_y *>(
            malloc_shared(size_out * sizeof(data_type_y), Device, Context));
    uint8_t *buffer_mask = static_cast<uint8_t *>(
            malloc_shared(size_mask * sizeof(uint8_t), Device, Context));
    uint64_t *buffer_rand_offset = static_cast<uint64_t *>(
            malloc_shared(sizeof(uint64_t), Device, Context));
    buffer_rand_offset[0] = 0;
    for (unsigned i = 0; i < size_in; ++i) {
        buffer_x[i] = (random_float() - 0.5) * 10;
    }
    for (unsigned i = 0; i < size_in; ++i) {
        buffer_bias_dropout_res[i] = 0;
    }
    for (unsigned i = 0; i < size_bias; ++i) {
        buffer_bias[i] = (random_float() - 0.5);
    }
    for (unsigned i = 0; i < size_resAdd; ++i) {
        buffer_resAdd[i] = (random_float() - 0.5) * 10;
    }
    for (unsigned i = 0; i < size_beta; ++i) {
        buffer_beta[i] = (i * 3) % 17;
    }
    for (unsigned i = 0; i < size_gamma; ++i) {
        buffer_gamma[i] = (i * 5) % 13;
    }
    for (unsigned i = 0; i < size_out; ++i) {
        buffer_y[i] = data_type_y(0);
    }
    for (unsigned i = 0; i < size_mu; ++i) {
        buffer_mu[i] = data_type_acc(0);
    }
    for (unsigned i = 0; i < size_rs; ++i) {
        buffer_rs[i] = data_type_acc(0);
    }
    uint32_t drop_threshold = drop_out_ratio * double(RAND_MAX);
    for (unsigned i = 0; i < size_mask; ++i) {
        buffer_mask[i] = (generate_random<double>(0.0, double(RAND_MAX))
                                 > drop_threshold)
                ? 0
                : 1;
    }

    cl::sycl::range<3> GroupRange {1, test::wg_num_m, test::wg_num_n};
    cl::sycl::range<3> LocalRange {1,
            (test::wg_m + test::sg_m - 1) / test::sg_m,
            (test::wg_n + test::sg_n - 1) / test::sg_n};
    cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

    try {
        auto e_esimd = Queue.submit([&](handler &cgh) {
            cgh.parallel_for<
                    test>(Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                constexpr bool store_for_bwd = true;
                using ln_fwd_func = ln_fwd_func_t<data_type_x, data_type_y,
                        data_type_weight, data_type_acc, test::wg_n, test::wg_m,
                        test::sg_n, test::sg_m, test::wg_num_m, test::wg_num_n,
                        test::ln_fused_op_kind, store_for_bwd>;
                constexpr uint32_t slm_size = ln_fwd_func::slm_size;
                constexpr uint32_t barrier_count = ln_fwd_func::barrier_count;

                if constexpr (barrier_count != 0) {
                    xetla_nbarrier_init<barrier_count>();
                }
                if constexpr (slm_size != 0) { xetla_local_init<slm_size>(); }

                xetla_exec_item<3> ei(item);

                ln_fwd_func::call(ei, buffer_x, buffer_y, matrix_m, matrix_n,
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

    // validation
    ASSERT_EQ(0,
            (ln_fwd_result_validate<data_type_x, data_type_weight, data_type_y,
                    data_type_acc>(buffer_x, buffer_y, buffer_gamma,
                    buffer_beta, buffer_mu, buffer_rs, matrix_m, matrix_n,
                    matrix_n, test::sg_n, buffer_bias, buffer_resAdd,
                    buffer_mask, buffer_bias_dropout_res, drop_out_scale,
                    test::ln_fused_op_kind)));

    free(buffer_x, Context);
    free(buffer_beta, Context);
    free(buffer_gamma, Context);
    free(buffer_mu, Context);
    free(buffer_rs, Context);
    free(buffer_y, Context);
    free(buffer_mask, Context);
    free(buffer_resAdd, Context);
    free(buffer_bias, Context);
    free(buffer_rand_offset, Context);
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
