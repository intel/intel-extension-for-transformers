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

using namespace std::placeholders;

TEST(tile_elemwise_op_relu, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(tile_elemwise_op_validate<float, relu_op<float>>, _1,
                    _2, _3, 128, 16, 16);
    kernel_run<float,
            tile_elemwise_op_func<float, 128, 64, 128, 16, 16, 16, 16,
                    relu_op_t>>(nd_range, result_validate);
}

TEST(tile_elemwise_op_gelu_fwd, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(tile_elemwise_op_validate<float, gelu_op<float>>, _1,
                    _2, _3, 128, 16, 24);
    kernel_run<float,
            tile_elemwise_op_func<float, 128, 64, 128, 16, 24, 16, 16,
                    gelu_fwd_op_t>>(nd_range, result_validate);
}

TEST(tile_elemwise_op_gelu_fwd_w, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate
            = std::bind(tile_elemwise_op_validate<float, gelu_fwd_w_op<float>>,
                    _1, _2, _3, 128, 16, 24);
    kernel_run<float,
            tile_elemwise_op_func<float, 128, 64, 128, 16, 24, 16, 16,
                    gelu_fwd_w_op_t<float, gpu_arch::Xe>>>(
            nd_range, result_validate);
}

TEST(tile_elemwise_op_gelu_bwd, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_elemwise_gelu_bwd_validate<float>, _1, _2, _3, 128, 16, 24);
    kernel_run<float,
            tile_elemwise_op_func<float, 128, 64, 128, 16, 24, 16, 16,
                    gelu_bwd_op_t<float, gpu_arch::Xe>>>(
            nd_range, result_validate);
}

TEST(tile_elemwise_op_bias_add, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_elemwise_bias_add_validate<float>, _1, _2, _3, 128, 16, 24);
    kernel_run<float,
            tile_elemwise_op_func<float, 128, 64, 128, 16, 24, 16, 16,
                    bias_add_op_t<mem_desc_t<float, mem_layout::row_major,
                                          mem_space::global>,
                            gpu_arch::Xe>>>(nd_range, result_validate);
}

TEST(tile_elemwise_op_res_add, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_elemwise_res_add_validate<float>, _1, _2, _3, 128, 16, 24);
    kernel_run<float,
            tile_elemwise_op_func<float, 128, 64, 128, 16, 24, 16, 16,
                    elemwise_reduce_op_t<reduce_op::sum, float, gpu_arch::Xe>>>(
            nd_range, result_validate);
}

TEST(tile_elemwise_op_linear_op, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_elemwise_linear_op_validate<float>, _1, _2, _3, 128, 16, 32);
    kernel_run<float,
            tile_elemwise_op_func<float, 128, 64, 128, 16, 32, 16, 16,
                    linear_op_t<float, gpu_arch::Xe>>>(
            nd_range, result_validate);
}

TEST(tile_elemwise_op_linear_op_2, esimd) {
    cl::sycl::nd_range<1> nd_range({1}, {1});
    auto result_validate = std::bind(
            tile_elemwise_linear_op_validate<float>, _1, _2, _3, 128, 16, 24);
    kernel_run<float,
            tile_elemwise_op_func<float, 128, 64, 128, 16, 24, 16, 16,
                    linear_op_t<float, gpu_arch::Xe>>>(
            nd_range, result_validate);
}
