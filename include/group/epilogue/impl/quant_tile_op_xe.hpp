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

#include "group/epilogue/api.hpp"
#include "group/epilogue/common.hpp"
#include "group/epilogue/epilogue_policy.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_epilogue
/// @{

/// @brief Is the epilogue functor specialized for epilogue_policy_quant_op and Xe architecture.
template <typename dequant_op_t_, typename dtype_dequant_, typename tile_op_t_,
        typename quant_op_t_, typename tile_shape_, typename mem_desc_c_t_,
        gpu_arch arch_tag_>
class epilogue_t<epilogue_policy_quant_op<dequant_op_t_, tile_op_t_,
                         quant_op_t_, arch_tag_, dtype_dequant_>,
        tile_shape_, mem_desc_c_t_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
public:
    using epilogue_policy = epilogue_policy_quant_op<dequant_op_t_, tile_op_t_,
            quant_op_t_, arch_tag_, dtype_dequant_>;
    using dequant_op_t = typename epilogue_policy::dequant_op_t;
    using quant_op_t = typename epilogue_policy::quant_op_t;
    using tile_op_t = typename epilogue_policy::tile_op_t;
    using dtype_dequant = typename epilogue_policy::dtype_dequant;
    using tile_shape = tile_shape_;
    using mem_desc_c_t = mem_desc_c_t_;
    static constexpr gpu_arch arch_tag = arch_tag_;
    static constexpr uint32_t barrier_count = 0;
    static constexpr uint32_t slm_size = mem_desc_c_t::is_local
            ? tile_shape::wg_tile_size_x * tile_shape::wg_tile_size_y
            : 0;

    /// @brief Epilogue arguments.
    struct arguments_t {
        /// @brief Is dequant_op arguments.
        typename dequant_op_t::arguments_t dequant_op_args;

        /// @brief Is tile_op arguments, could be a single
        /// tile_op argument or chained_tile_op_args.
        typename tile_op_t::arguments_t tile_op_args;

        /// @brief Is quant_op arguments.
        typename quant_op_t::arguments_t quant_op_args;

        /// @brief Constructs a new arguments t object.
        inline arguments_t() = default;

        /// @brief Constructs a new arguments t object.
        /// @param dequant_op_args_ Is dequant_op arguments.
        /// @param tile_op_args_ Is tile_op arguments, could be a single
        /// tile_op argument or chained_tile_op_args.
        /// @param quant_op_args_ Is quant_op arguments.
        inline arguments_t(typename dequant_op_t::arguments_t dequant_op_args_,
                typename tile_op_t::arguments_t tile_op_args_,
                typename quant_op_t::arguments_t quant_op_args_)
            : dequant_op_args(dequant_op_args_)
            , tile_op_args(tile_op_args_)
            , quant_op_args(quant_op_args_) {}
        inline arguments_t(const arguments_t &args)
            : dequant_op_args(args.dequant_op_args)
            , tile_op_args(args.tile_op_args)
            , quant_op_args(args.quant_op_args) {}
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // ~arguments_t(){}
        inline arguments_t &operator=(const arguments_t &args) {
            this->dequant_op_args = args.dequant_op_args;
            this->tile_op_args = args.tile_op_args;
            this->quant_op_args = args.quant_op_args;
            return *this;
        }

        /// @brief Explicit initialization function.
        /// @param dequant_op_args_ Is dequant_op arguments.
        /// @param tile_op_args_ Is tile_op arguments, could be a single
        /// tile_op argument or chained_tile_op_args.
        /// @param quant_op_args_ Is quant_op arguments.
        inline void init(typename dequant_op_t::arguments_t dequant_op_args_,
                typename tile_op_t::arguments_t tile_op_args_,
                typename quant_op_t::arguments_t quant_op_args_) {
            dequant_op_args = dequant_op_args_;
            tile_op_args = tile_op_args_;
            quant_op_args = quant_op_args_;
        }
    };

private:
    using work_group_t = typename tile_shape::work_group_t;
    static constexpr uint32_t sg_tile_y = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_x = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    using dtype_c = typename mem_desc_c_t::dtype;
    static constexpr mem_layout mem_layout_c = mem_desc_c_t::layout;
    static constexpr mem_space mem_space_c = mem_desc_c_t::space;

    /// @brief Updates tile base descriptor based on the tid.
    __XETLA_API static void update_sg_tile_tdesc(
            work_group_t &g, mem_desc_c_t &mem_desc_c) {
        int32_t sg_idx = g.get_id() % wg_size_x;
        int32_t sg_idy = g.get_id() / wg_size_x;
        int32_t tile_offset_x = sg_idx * sg_tile_x;
        int32_t tile_offset_y = sg_idy * sg_tile_y;
        mem_desc_c.update_coord(tile_offset_x, tile_offset_y);
    }

public:
    static constexpr msg_type msg_type_c
            = (mem_space_c == mem_space::global ? msg_type::block_2d
                                                : msg_type::scatter);

    /// @brief Default epilogue.
    /// 1) Call tile_op/chained_tile_op 2) Call quant_op
    /// 3) Overwrite/reduce_sum to memory.
    /// @tparam matAcc_t Is the type of the input tile.
    /// @param g Is the workgroup of the current tile.
    /// @param matAcc Is the input tile.
    /// @param mem_desc_c Is the memory description of matC, including base, shape and coordinate.
    /// @param args Is the additional arguments for epilogue.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g, matAcc_t &matAcc,
            mem_desc_c_t mem_desc_c, arguments_t args = {},
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        update_sg_tile_tdesc(g, mem_desc_c);
        using mat_tile_desc = typename matAcc_t::tile_desc;
        using mat_dequant_t = subgroup::tile_t<dtype_dequant, mat_tile_desc>;
        using matC_t = subgroup::tile_t<dtype_c, mat_tile_desc>;

        tile_op_t tile_op;
        quant_op_t quant_op;
        dequant_op_t dequant_op;
        //dequantize
        mat_dequant_t mat_dequant;
        dequant_op(mat_dequant, matAcc, mem_desc_c.coord, args.dequant_op_args);
        //post-op
        tile_op(mat_dequant, mem_desc_c.coord, args.tile_op_args, slm_base,
                nbarrier_base);
        //quantize
        matC_t matC;
        quant_op(matC, mat_dequant, mem_desc_c.coord, args.quant_op_args);

        using matC_payload_t = subgroup::mem_payload_t<mem_desc_c_t,
                mat_tile_desc, msg_type_c, arch_tag>;
        matC_payload_t matC_payload(mem_desc_c);
        subgroup::tile_store<cache_hint::streaming, cache_hint::write_back>(
                matC, matC_payload);
    }
};

/// @} xetla_epilogue

} // namespace gpu::xetla::group
