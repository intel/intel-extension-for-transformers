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

#include "group/softmax/common.hpp"

namespace gpu::xetla::group {

template <typename dtype_acc, gpu_arch arch_tag_ = gpu_arch::Xe>
struct softmax_policy_fwd {};

template <typename dtype_in, typename dtype_acc,
        gpu_arch arch_tag_ = gpu_arch::Xe>
struct softmax_policy_bwd {};

} // namespace gpu::xetla::group
