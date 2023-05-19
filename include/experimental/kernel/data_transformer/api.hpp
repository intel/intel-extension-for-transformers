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

/// @file
/// C++ API

#pragma once

#include "experimental/kernel/data_transformer/common.hpp"

namespace gpu::xetla::kernel {

/// @brief Is the data_transformer functor.
///
/// @tparam dtype_in_ Is the data type of input.
/// @tparam dtype_out_ Is the data type of output.
/// @tparam dtype_acc_
/// @tparam data_transformer_config_
/// @tparam mem_layout_in_ Indicates the input data col major or row major.
/// @tparam need_fp8_op
/// @tparam arch_ Is the HW generation.
template <typename dtype_in_, typename dtype_out_, typename dtype_acc_,
        typename data_transformer_config_, mem_layout mem_layout_in_,
        int need_fp8_op, gpu_arch arch_>
struct xetla_data_transformer {};

} // namespace gpu::xetla::kernel
