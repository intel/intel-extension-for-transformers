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

#include "subgroup/tile/api.hpp"
#include "subgroup/tile/common.hpp"
#include "subgroup/tile/impl/load_xe.hpp"
#include "subgroup/tile/impl/payload_xe.hpp"
#include "subgroup/tile/impl/prefetch_xe.hpp"
#include "subgroup/tile/impl/store_xe.hpp"
#include "subgroup/tile/impl/tile_op_functor.hpp"

namespace gpu::xetla::subgroup {

/// @brief Is the dequantization op functor.
/// @tparam tile_op_t Is the dequantization method, share the same API with tile_op/chained_tile_op.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename tile_op_t, gpu_arch arch_tag, class enable = void>
struct dequant_op_t {};
/// @brief Is the quantization op functor, specialized for Xe architecture.
template <typename tile_op_t_, gpu_arch arch_tag>
struct dequant_op_t<tile_op_t_, arch_tag,
        std::enable_if_t<(arch_tag == gpu_arch::Xe)>> {
    //may need to add some limitations to tile_op used in dequant_op
    using tile_op_t = tile_op_t_;
    struct arguments_t {
        typename tile_op_t::arguments_t tile_op_args;
        inline arguments_t() = default;
        inline arguments_t(typename tile_op_t::arguments_t tile_op_args_)
            : tile_op_args(tile_op_args_) {}
    };
    template <typename mat_out_t, typename mat_in_t, typename coord_t>
    __XETLA_API KERNEL_FUNC void operator()(mat_out_t &mat_out,
            mat_in_t &mat_in, const coord_t &coord, const arguments_t &args) {
        elemwise_cvt(mat_out, mat_in);
        tile_op_t tile_op;
        tile_op(mat_out, coord, args.tile_op_args);
    }
};

/// @brief Is the quantization op functor.
/// @tparam tile_op_t Is the quantization method, share the same API with tile_op/chained_tile_op.
/// @tparam arch_tag Is the hardware architecture tag.
template <typename tile_op_t, gpu_arch arch_tag, class enable = void>
struct quant_op_t {};
/// @brief Is the quantization op functor, specialized for Xe architecture.
template <typename tile_op_t_, gpu_arch arch_tag>
struct quant_op_t<tile_op_t_, arch_tag,
        std::enable_if_t<(arch_tag == gpu_arch::Xe)>> {
    //may need to add some limitations to tile_op used in dequant_op
    using tile_op_t = tile_op_t_;

    struct arguments_t {
        typename tile_op_t::arguments_t tile_op_args;
        inline arguments_t() = default;
        inline arguments_t(typename tile_op_t::arguments_t tile_op_args_)
            : tile_op_args(tile_op_args_) {}
    };
    template <typename mat_out_t, typename mat_in_t,
            typename dtype_sat = typename mat_out_t::dtype, typename coord_t>
    __XETLA_API KERNEL_FUNC void operator()(mat_out_t &mat_out,
            mat_in_t &mat_in, const coord_t &coord, const arguments_t &args) {
        static_assert(is_same_layout<mat_out_t, mat_in_t>::value,
                " mat_in and mat_out should be the same layout");
        using matAcc_t = subgroup::tile_t<typename mat_in_t::dtype,
                typename mat_out_t::tile_desc>;
        using mat_sat_t
                = subgroup::tile_t<dtype_sat, typename mat_out_t::tile_desc>;

        matAcc_t matAcc;
        // to ensure there is no in-place changes in mat_in,
        // and compiler will optimize this if there is no usage for mat_in.
        elemwise_cvt(matAcc, mat_in);
        tile_op_t tile_op;
        tile_op(matAcc, coord, args.tile_op_args);
        mat_sat_t mat_sat;
        elemwise_cvt(mat_sat, matAcc);
        elemwise_cvt(mat_out, mat_sat);
    }

    template <typename dtype_sat, typename matAcc_t, typename coord_t>
    __XETLA_API KERNEL_FUNC void operator()(
            matAcc_t &matAcc, const coord_t &coord, const arguments_t &args) {
        operator()<matAcc_t, matAcc_t, dtype_sat>(matAcc, matAcc, coord, args);
    }
};

} // namespace gpu::xetla::subgroup
