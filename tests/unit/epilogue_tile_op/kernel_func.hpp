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
using namespace gpu::xetla::subgroup;
using namespace gpu::xetla::group;

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight, typename tile_op_t>
struct tile_elemwise_op_func {
    using mem_desc_c_t
            = mem_desc_t<dtype, mem_layout::row_major, mem_space::global>;
    using matA_tile_desc_t
            = tile_desc_t<twidth, theight, bwidth, bheight, reg_layout::tiled>;
    using matA_t = tile_t<dtype, matA_tile_desc_t>;
    using matA_payload_t = mem_payload_t<mem_desc_c_t, matA_tile_desc_t,
            msg_type_v<matA_tile_desc_t, mem_space::global>, gpu_arch::Xe>;
    using tile_shape = tile_shape_t<twidth, theight, twidth, theight>;

    using epilogue_policy = epilogue_policy_tile_op<tile_op_t, gpu_arch::Xe>;
    using epilogue_t = epilogue_t<epilogue_policy, tile_shape, mem_desc_c_t>;
    using work_group_t = typename tile_shape::work_group_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        matA_t matA;
        matA_payload_t matA_payload;
        matA_payload.init(a, swidth, sheight, spitch, 0, 0);
        subgroup::tile_load(matA, matA_payload);
        matA.reg = (matA.reg - 50.f) / 100.f;
        mem_desc_c_t mem_desc_c({c}, {swidth, sheight, spitch}, {0, 0});
        work_group_t g(item->get_local_linear_id());
        epilogue_args_t epilogue_args {};
        epilogue_t epilogue;
        epilogue(g, matA, mem_desc_c, epilogue_args);
    }
};

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight>
struct tile_elemwise_op_func<dtype, swidth, sheight, spitch, twidth, theight,
        bwidth, bheight, gelu_fwd_w_op_t<dtype, gpu_arch::Xe>> {
    using mem_desc_b_t
            = mem_desc_t<dtype, mem_layout::row_major, mem_space::global>;
    using matA_tile_desc_t
            = tile_desc_t<twidth, theight, bwidth, bheight, reg_layout::tiled>;
    using matA_t = tile_t<dtype, matA_tile_desc_t>;
    using matA_payload_t = mem_payload_t<mem_desc_b_t, matA_tile_desc_t,
            msg_type_v<matA_tile_desc_t, mem_space::global>, gpu_arch::Xe>;
    using tile_shape = tile_shape_t<twidth, theight, twidth, theight>;
    using epilogue_policy
            = epilogue_policy_tile_op<gelu_fwd_w_op_t<dtype, gpu_arch::Xe>,
                    gpu_arch::Xe>;
    using epilogue_t = epilogue_t<epilogue_policy, tile_shape, mem_desc_b_t>;
    using work_group_t = typename tile_shape::work_group_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        matA_t matA;
        matA_payload_t matA_payload;
        matA_payload.init(a, swidth, sheight, spitch, 0, 0);
        subgroup::tile_load(matA, matA_payload);
        matA.reg = (matA.reg - 50.f) / 100.f;
        mem_desc_b_t mem_desc_b({b}, {swidth, sheight, spitch}, {0, 0});
        work_group_t g(item->get_local_linear_id());
        epilogue_args_t epilogue_args {{{c}, {swidth, sheight, spitch}}};
        epilogue_t epilogue;
        epilogue(g, matA, mem_desc_b, epilogue_args);
    }
};

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight>
struct tile_elemwise_op_func<dtype, swidth, sheight, spitch, twidth, theight,
        bwidth, bheight, gelu_bwd_op_t<dtype, gpu_arch::Xe>> {
    using mem_desc_c_t
            = mem_desc_t<dtype, mem_layout::row_major, mem_space::global>;
    using matA_tile_desc_t
            = tile_desc_t<twidth, theight, bwidth, bheight, reg_layout::tiled>;
    using matA_t = tile_t<dtype, matA_tile_desc_t>;
    using matA_payload_t = mem_payload_t<mem_desc_c_t, matA_tile_desc_t,
            msg_type_v<matA_tile_desc_t, mem_space::global>, gpu_arch::Xe>;
    using tile_shape = tile_shape_t<twidth, theight, twidth, theight>;
    using epilogue_policy
            = epilogue_policy_tile_op<gelu_bwd_op_t<dtype, gpu_arch::Xe>,
                    gpu_arch::Xe>;
    using epilogue_t = epilogue_t<epilogue_policy, tile_shape, mem_desc_c_t>;
    using work_group_t = typename tile_shape::work_group_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        matA_t matA;
        matA_payload_t matA_payload;
        matA_payload.init(a, swidth, sheight, spitch, 0, 0);
        subgroup::tile_load(matA, matA_payload);
        matA.reg = (matA.reg - 50.f) / 100.f;
        mem_desc_c_t mem_desc_c({c}, {swidth, sheight, spitch}, {0, 0});
        work_group_t g(item->get_local_linear_id());
        epilogue_args_t epilogue_args {{{b}, {swidth, sheight, spitch}}};
        epilogue_t epilogue;
        epilogue(g, matA, mem_desc_c, epilogue_args);
    }
};

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight>
struct tile_elemwise_op_func<dtype, swidth, sheight, spitch, twidth, theight,
        bwidth, bheight, bias_add_op_t<dtype, gpu_arch::Xe>> {
    using matAcc_t = tile_t<dtype,
            tile_desc_t<twidth, theight, bwidth, bheight, reg_layout::tiled>>;
    using tile_shape = tile_shape_t<twidth, theight, twidth, theight>;
    using mem_desc_c_t
            = mem_desc_t<dtype, mem_layout::row_major, mem_space::global>;
    using epilogue_policy
            = epilogue_policy_tile_op<bias_add_op_t<dtype, gpu_arch::Xe>,
                    gpu_arch::Xe>;
    using epilogue_t = epilogue_t<epilogue_policy, tile_shape, mem_desc_c_t>;
    using work_group_t = typename tile_shape::work_group_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        matAcc_t matAcc;
        matAcc.reg = 0;
        mem_desc_c_t mem_desc_c({c}, {swidth, sheight, spitch}, {0, 0});
        work_group_t g(item->get_local_linear_id());
        epilogue_args_t epilogue_args {{{a}, {swidth, sheight, spitch}}};
        epilogue_t epilogue;
        epilogue(g, matAcc, mem_desc_c, epilogue_args);
    }
};

template <reduce_op reduce_kind, typename dtype, int swidth, int sheight,
        int spitch, int twidth, int theight, int bwidth, int bheight>
struct tile_elemwise_op_func<dtype, swidth, sheight, spitch, twidth, theight,
        bwidth, bheight,
        elemwise_reduce_op_t<reduce_kind, dtype, gpu_arch::Xe>> {
    using matAcc_t = tile_t<dtype,
            tile_desc_t<twidth, theight, bwidth, bheight, reg_layout::tiled>>;
    using tile_shape = tile_shape_t<twidth, theight, twidth, theight>;
    using mem_desc_c_t
            = mem_desc_t<dtype, mem_layout::row_major, mem_space::global>;
    using epilogue_policy = epilogue_policy_tile_op<
            elemwise_reduce_op_t<reduce_kind, dtype, gpu_arch::Xe>,
            gpu_arch::Xe>;
    using epilogue_t = epilogue_t<epilogue_policy, tile_shape, mem_desc_c_t>;
    using work_group_t = typename tile_shape::work_group_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        matAcc_t matAcc;
        matAcc.reg = 0;
        mem_desc_c_t mem_desc_c({c}, {swidth, sheight, spitch}, {0, 0});
        work_group_t g(item->get_local_linear_id());
        epilogue_args_t epilogue_args {{{a}, {swidth, sheight, spitch}}};
        epilogue_t epilogue;
        epilogue(g, matAcc, mem_desc_c, epilogue_args);
    }
};

template <typename dtype, int swidth, int sheight, int spitch, int twidth,
        int theight, int bwidth, int bheight>
struct tile_elemwise_op_func<dtype, swidth, sheight, spitch, twidth, theight,
        bwidth, bheight, linear_op_t<dtype, gpu_arch::Xe>> {
    using matAcc_t = tile_t<dtype,
            tile_desc_t<twidth, theight, bwidth, bheight, reg_layout::tiled>>;
    using tile_shape = tile_shape_t<twidth, theight, twidth, theight>;
    using mem_desc_c_t
            = mem_desc_t<dtype, mem_layout::row_major, mem_space::global>;

    using epilogue_policy
            = epilogue_policy_tile_op<linear_op_t<dtype, gpu_arch::Xe>,
                    gpu_arch::Xe>;
    using epilogue_t = epilogue_t<epilogue_policy, tile_shape, mem_desc_c_t>;
    using work_group_t = typename tile_shape::work_group_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        matAcc_t matAcc;
        matAcc.reg = 0;
        mem_desc_c_t mem_desc_c {{c}, {swidth, sheight, spitch}, {0, 0}};
        work_group_t g(item->get_local_linear_id());
        epilogue_args_t epilogue_args({{a}, {swidth, sheight, spitch}, 0.4, 4});
        epilogue_t epilogue;
        epilogue(g, matAcc, mem_desc_c, epilogue_args);
    }
};
