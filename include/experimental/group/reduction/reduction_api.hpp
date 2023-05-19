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

#include "subgroup/subgroup.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_reduction
/// @{

/// @brief This is the group row reduction(reduce_sum) + cooperative write out. Use slm to exchange the data.
/// For wg_size_y threads, at the beginning, everyone will keep one row of data; Then, they compose a wg_size_y * row_size 2D block in SLM;
/// After that, each thread will load a small wg_size_y * block_size block, do the local reduction and write to global memory
/// @tparam dtype_acc Is the data type to do the reduction
/// @tparam dtype_out Is the data type to write out
/// @tparam row_size Is the vector size per row
/// @tparam wg_size_x Is the wg size in x direction, is the number of parallel reductions in the wg.
/// @tparam wg_size_y Is the wg size in y direction, i.e. is the number of threads that participate in this reduction.
/// @tparam max_simd_len Is the max SIMD for scatter load. The limitation comes from the scattered load from local memory.
/// @tparam arch_ Is the HW generation.
template <typename dtype_acc, typename dtype_out, uint32_t row_size,
        uint32_t wg_size_x, uint32_t wg_size_y, uint32_t max_simd_len = 32,
        gpu_arch arch_ = gpu_arch::Xe>
struct group_row_reduce_store_t {};

/// @brief This is the group reduction. Use slm to exchange the data.
/// @tparam T Is the data type to do the reduction
/// @tparam SZ Is the vector size per item
/// @tparam N Is the number of independent sets for one subgroup to do the parallel all-reduction
/// @tparam Op Is the reduction op
/// @tparam N_SG Is the number of subgroups that participate in this reduction.
/// @tparam is_all_reduce Is the flag to enable all_reduce. If it is false, only sg_id 0 will have the updated result; otherwise all the N_SG subgroups will have the updated result.
/// @tparam arch_ Is the HW generation.
template <typename T, uint32_t SZ, uint32_t N, reduce_op Op, uint32_t N_SG,
        bool is_all_reduce = true, gpu_arch arch_ = gpu_arch::Xe>
struct group_reduce_t {};

/// @} xetla_reduction

} // namespace gpu::xetla::group
