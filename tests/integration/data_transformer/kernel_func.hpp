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

template <typename dtype_in, typename dtype_out, typename dtype_acc, int wg_n,
        int wg_m, int sg_n, int sg_m, mem_layout mem_layout_in, int need_fp8_op>
KERNEL_FUNC inline void data_transformer_func(xetla_exec_item<3> &ei,
        dtype_in *buffer_in, dtype_out *buffer_out, int matrix_m, int matrix_n,
        int mat_in_ld, int mat_out_ld, dtype_acc *scale_ptr,
        dtype_acc *amax_ptr) {

    int gid_x = ei.get_group(2);
    int gid_y = ei.get_group(1);

    //input and output starting point for wg
    int wg_ld_start_x;
    int wg_ld_start_y;

    if constexpr (mem_layout_in == mem_layout::row_major) {
        wg_ld_start_x = gid_x * wg_n;
        wg_ld_start_y = gid_y * wg_m;
    } else {
        wg_ld_start_x = gid_y * wg_m;
        wg_ld_start_y = gid_x * wg_n;
    }

    int wg_st_start_x = gid_x * wg_n;
    int wg_st_start_y = gid_y * wg_m;

    using data_transformer_attr
            = gpu::xetla::kernel::data_transformer_attr_t<wg_n, wg_m, sg_n,
                    sg_m>;
    using data_transformer
            = gpu::xetla::kernel::xetla_data_transformer<dtype_in, dtype_out,
                    dtype_acc, data_transformer_attr, mem_layout_in,
                    need_fp8_op, gpu_arch::Xe>;

    typename data_transformer::arguments_t args;
    args.mat_in_ptr = buffer_in;
    args.mat_out_ptr = buffer_out;
    args.matrix_m = matrix_m;
    args.matrix_n = matrix_n;
    args.matrix_in_ld = mat_in_ld;
    args.matrix_out_ld = mat_out_ld;
    args.amax_ptr = amax_ptr;
    args.scale = scale_ptr;
    args.wg_ld_start_x = wg_ld_start_x;
    args.wg_ld_start_y = wg_ld_start_y;
    args.wg_st_start_x = wg_st_start_x;
    args.wg_st_start_y = wg_st_start_y;

    data_transformer::call(ei, &args);
}
