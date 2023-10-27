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

#include "xetla.hpp"

namespace gpu::xetla::kernel {

template <typename gemm_layer1_t_, typename epilogue_layer1_t_,
        typename gemm_layer2_t_, typename epilogue_layer2_t_,
        gpu_arch arch_tag_>
class multi_layer_perceptron_t {
    using gemm_layer1_t = gemm_layer1_t_;
    using epilogue_layer1_t = epilogue_layer1_t_;
    using gemm_layer1_args_t = typename gemm_layer1_t::arguments_t;
    using epilogue_layer1_args_t = typename epilogue_layer1_t::arguments_t;

    using tile_shape_layer1 = typename gemm_layer1_t::tile_shape;
    static constexpr uint32_t wg_tile_m_layer1
            = tile_shape_layer1::wg_tile_size_y;
    static constexpr uint32_t wg_tile_n_layer1
            = tile_shape_layer1::wg_tile_size_x;
    static constexpr uint32_t sg_tile_m_layer1
            = tile_shape_layer1::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n_layer1
            = tile_shape_layer1::sg_tile_size_x;
    static constexpr uint32_t wg_size_y_layer1 = tile_shape_layer1::wg_size_y;
    static constexpr uint32_t wg_size_x_layer1 = tile_shape_layer1::wg_size_x;
    static constexpr uint32_t real_wg_tile_m_layer1
            = sg_tile_m_layer1 * wg_size_y_layer1;
    static constexpr uint32_t real_wg_tile_n_layer1
            = sg_tile_n_layer1 * wg_size_x_layer1;

    static constexpr uint32_t k_stride_layer1 = gemm_layer1_t::k_stride;
    using work_group_layer1_t = typename gemm_layer1_t::work_group_t;

    static constexpr gpu_arch arch_tag = arch_tag_;
    static_assert(
            arch_tag == gemm_layer1_t::arch_tag, "arch_tag should be the same");
    static_assert(arch_tag == epilogue_layer1_t::arch_tag,
            "arch_tag should be the same");
    static_assert(std::is_same<typename gemm_layer1_t::tile_shape,
                          typename epilogue_layer1_t::tile_shape>::value,
            "tile_shape should be the same");

    using mem_desc_a_t = typename gemm_layer1_t::mem_desc_a_t;
    using mem_desc_w_t = typename gemm_layer1_t::mem_desc_b_t;
    using mem_desc_b_t = typename epilogue_layer1_t::mem_desc_c_t;
    using matA_base_t = typename mem_desc_a_t::base_t;
    using matW_base_t = typename mem_desc_w_t::base_t;
    using matB_base_t = typename mem_desc_b_t::base_t;
    using dtype_a = typename mem_desc_a_t::dtype;
    using dtype_w = typename mem_desc_w_t::dtype;
    using dtype_b = typename mem_desc_b_t::dtype;
    using matAcc_layer1_t = typename gemm_layer1_t::matAcc_t;

    using gemm_layer2_t = gemm_layer2_t_;
    using epilogue_layer2_t = epilogue_layer2_t_;
    using gemm_layer2_args_t = typename gemm_layer2_t::arguments_t;
    using epilogue_layer2_args_t = typename epilogue_layer2_t::arguments_t;

    using tile_shape_layer2 = typename gemm_layer2_t::tile_shape;
    static constexpr uint32_t wg_tile_m_layer2
            = tile_shape_layer2::wg_tile_size_y;
    static constexpr uint32_t wg_tile_n_layer2
            = tile_shape_layer2::wg_tile_size_x;
    static constexpr uint32_t sg_tile_m_layer2
            = tile_shape_layer2::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n_layer2
            = tile_shape_layer2::sg_tile_size_x;
    static constexpr uint32_t wg_size_y_layer2 = tile_shape_layer2::wg_size_y;
    static constexpr uint32_t wg_size_x_layer2 = tile_shape_layer2::wg_size_x;
    static constexpr uint32_t real_wg_tile_m_layer2
            = sg_tile_m_layer2 * wg_size_y_layer2;
    static constexpr uint32_t real_wg_tile_n_layer2
            = sg_tile_n_layer2 * wg_size_x_layer2;

    static constexpr uint32_t k_stride_layer2 = gemm_layer2_t::k_stride;
    using work_group_layer2_t = typename gemm_layer2_t::work_group_t;

    static_assert(
            arch_tag == gemm_layer2_t::arch_tag, "arch_tag should be the same");
    static_assert(arch_tag == epilogue_layer2_t::arch_tag,
            "arch_tag should be the same");
    static_assert(std::is_same<typename gemm_layer2_t::tile_shape,
                          typename epilogue_layer2_t::tile_shape>::value,
            "tile_shape should be the same");

    // using mem_desc_b_t = typename gemm1_t::mem_desc_a_t;
    static_assert(std::is_same<typename epilogue_layer1_t::mem_desc_c_t,
                          typename gemm_layer2_t::mem_desc_a_t>::value,
            "the output of first gemm should be the left input og second "
            "gemm!");
    using mem_desc_v_t = typename gemm_layer2_t::mem_desc_b_t;
    using mem_desc_c_t = typename epilogue_layer2_t::mem_desc_c_t;
    using matV_base_t = typename mem_desc_v_t::base_t;
    using matC_base_t = typename mem_desc_c_t::base_t;
    using dtype_v = typename mem_desc_v_t::dtype;
    using dtype_c = typename mem_desc_c_t::dtype;
    using matAcc_layer2_t = typename gemm_layer2_t::matAcc_t;

public:
    struct arguments_t {
        /// @brief Is the size of the m dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_m_layer1;
        /// @brief Is the size of the k dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_k_layer1;
        /// @brief Is the size of the n dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_n_layer1;
        /// @brief Is the size of the m dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_m_layer2;
        /// @brief Is the size of the k dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_k_layer2;
        /// @brief Is the size of the n dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_n_layer2;
        /// @brief Is the leading dimension (pitch) size of the matrix A in memory.
        uint32_t matA_ld;
        /// @brief Is the leading dimension (pitch) size of the matrix W in memory.
        uint32_t matW_ld;
        /// @brief Is the leading dimension (pitch) size of the matrix B in memory.
        uint32_t matB_ld;
        /// @brief Is the leading dimension (pitch) size of the matrix V in memory.
        uint32_t matV_ld;
        /// @brief Is the leading dimension (pitch) size of the matrix C in memory.
        uint32_t matC_ld;
        /// @brief Is the base address of matrix A.
        matA_base_t matA_base;
        /// @brief Is the base address of matrix W.
        matW_base_t matW_base;
        /// @brief Is the base address of matrix B.
        matB_base_t matB_base;
        /// @brief Is the base address of matrix V.
        matV_base_t matV_base;
        /// @brief Is the base address of matrix C.
        matC_base_t matC_base;
        /// @brief Is the epilogue arguments of first gemm.
        epilogue_layer1_args_t epilogue_layer1_args;
        /// @brief Is the epilogue arguments of second gemm.
        epilogue_layer2_args_t epilogue_layer2_args;

        /// @brief Constructs arguments with default method.
        inline arguments_t() = default;

        /// @brief Set for device copyable
        static constexpr bool host_callable = true;

        /// @brief Constructs arguments with initialization list.
        /// @param matrix_m_layer1_ Is the size of the m dimension of the matrix multiplication (m x k x n).
        /// @param matrix_k_layer1_ Is the size of the k dimension of the matrix multiplication (m x k x n).
        /// @param matrix_n_layer1_ Is the size of the n dimension of the matrix multiplication (m x k x n).
        /// @param matrix_m_layer2_ Is the size of the m dimension of the matrix multiplication (m x k x n).
        /// @param matrix_k_layer2_ Is the size of the k dimension of the matrix multiplication (m x k x n).
        /// @param matrix_n_layer2_ Is the size of the n dimension of the matrix multiplication (m x k x n).
        /// @param matA_base_ Is the base address of matrix A.
        /// @param matA_ld_ Is the leading dimension (pitch) size of the matrix A in memory.
        /// @param matW_base_ Is the base address of matrix W.
        /// @param matW_ld_ Is the leading dimension (pitch) size of the matrix W in memory.
        /// @param matB_base_ Is the base address of matrix B.
        /// @param matB_ld_ Is the leading dimension (pitch) size of the matrix B in memory.
        /// @param matV_base_ Is the base address of matrix V.
        /// @param matV_ld_ Is the leading dimension (pitch) size of the matrix V in memory.
        /// @param matC_base_ Is the base address of matrix C.
        /// @param matC_ld_ Is the leading dimension (pitch) size of the matrix C in memory.
        /// @param epilogue_layer1_args_ Is the epilogue arguments of first gemm.
        /// @param epilogue_layer2_args_ Is the epilogue arguments of second gemm.
        inline arguments_t(uint32_t matrix_m_layer1_, uint32_t matrix_k_layer1_,
                uint32_t matrix_n_layer1_, uint32_t matrix_m_layer2_,
                uint32_t matrix_k_layer2_, uint32_t matrix_n_layer2_,
                matA_base_t matA_base_, uint32_t matA_ld_,
                matW_base_t matW_base_, uint32_t matW_ld_,
                matB_base_t matB_base_, uint32_t matB_ld_,
                matV_base_t matV_base_, uint32_t matV_ld_,
                matC_base_t matC_base_, uint32_t matC_ld_,
                epilogue_layer1_args_t epilogue_layer1_args_ = {},
                epilogue_layer2_args_t epilogue_layer2_args_ = {})
            : matrix_m_layer1(matrix_m_layer1_)
            , matrix_k_layer1(matrix_k_layer1_)
            , matrix_n_layer1(matrix_n_layer1_)
            , matrix_m_layer2(matrix_m_layer2_)
            , matrix_k_layer2(matrix_k_layer2_)
            , matrix_n_layer2(matrix_n_layer2_)
            , matA_base(matA_base_)
            , matA_ld(matA_ld_)
            , matW_base(matW_base_)
            , matW_ld(matW_ld_)
            , matB_base(matB_base_)
            , matB_ld(matB_ld_)
            , matV_base(matV_base_)
            , matV_ld(matV_ld_)
            , matC_base(matC_base_)
            , matC_ld(matC_ld_)
            , epilogue_layer1_args(epilogue_layer1_args_)
            , epilogue_layer2_args(epilogue_layer2_args_) {}
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // inline ~arguments_t(){}
        inline arguments_t(const arguments_t &args)
            : matrix_m_layer1(args.matrix_m_layer1)
            , matrix_k_layer1(args.matrix_k_layer1)
            , matrix_n_layer1(args.matrix_n_layer1)
            , matrix_m_layer2(args.matrix_m_layer2)
            , matrix_k_layer2(args.matrix_k_layer2)
            , matrix_n_layer2(args.matrix_n_layer2)
            , matA_base(args.matA_base)
            , matA_ld(args.matA_ld)
            , matW_base(args.matW_base)
            , matW_ld(args.matW_ld)
            , matB_base(args.matB_base)
            , matB_ld(args.matB_ld)
            , matV_base(args.matV_base)
            , matV_ld(args.matV_ld)
            , matC_base(args.matC_base)
            , matC_ld(args.matC_ld)
            , epilogue_layer1_args(args.epilogue_layer1_args)
            , epilogue_layer2_args(args.epilogue_layer2_args) {}
        inline arguments_t &operator=(const arguments_t &args) {
            this->matrix_m_layer1 = args.matrix_m_layer1;
            this->matrix_k_layer1 = args.matrix_k_layer1;
            this->matrix_n_layer1 = args.matrix_n_layer1;
            this->matrix_m_layer2 = args.matrix_m_layer2;
            this->matrix_k_layer2 = args.matrix_k_layer2;
            this->matrix_n_layer2 = args.matrix_n_layer2;
            this->matA_base = args.matA_base;
            this->matA_ld = args.matA_ld;
            this->matW_base = args.matW_base;
            this->matW_ld = args.matW_ld;
            this->matB_base = args.matB_base;
            this->matB_ld = args.matB_ld;
            this->matV_base = args.matV_base;
            this->matV_ld = args.matV_ld;
            this->matC_base = args.matC_base;
            this->matC_ld = args.matC_ld;
            this->epilogue_layer1_args = args.epilogue_layer1_args;
            this->epilogue_layer2_args = args.epilogue_layer2_args;
            return *this;
        }
    };

    /// @brief Gets named_barrier id consumption count.
    /// Users query and get a named_barrier id consumption count in compile time.
    /// @return The count of named barriers required.
    __XETLA_API static constexpr uint32_t get_barrier_count() {
        constexpr uint32_t count = gemm_layer1_t::barrier_count
                                + epilogue_layer1_t::barrier_count + 1
                        > gemm_layer2_t::barrier_count
                                + epilogue_layer2_t::barrier_count
                ? gemm_layer1_t::barrier_count
                        + epilogue_layer1_t::barrier_count + 1
                : gemm_layer2_t::barrier_count
                        + epilogue_layer2_t::barrier_count;
        static_assert(
                count <= 32, "The named_barrier count should be less than 32!");
        return count;
    }

    /// @brief Gets local memory size consumption.
    /// Users query and get a local memory consumption size in compile time.
    /// @return The size of local memory required.
    __XETLA_API static constexpr uint32_t get_slm_size() {
        // In this MLP example we don't use SLM for load/store or intermediate result storage
        // So the final slm size should equal to 0
        return 0;
    };

    /// @brief Host helper function to get the expected local range under the current MLP config.
    /// @return Expected local range.
    static cl::sycl::range<3> get_local_range() {
        // make sure first layer and second layer use same subgroup number.
        static_assert(work_group_layer1_t::size == work_group_layer2_t::size,
                "we should make sure first gemm and second gemm use same "
                "subgroup number!");
        uint32_t local_range_m
                = (wg_tile_m_layer2 + sg_tile_m_layer2 - 1) / sg_tile_m_layer2;
        uint32_t local_range_n
                = (wg_tile_n_layer2 + sg_tile_n_layer2 - 1) / sg_tile_n_layer2;
        std::cout << "Local range: {" << 1 << ", " << local_range_m << ", "
                  << local_range_n << "} \n";
        assert(local_range_m * local_range_n <= 32);
        return cl::sycl::range<3> {1, local_range_m, local_range_n};
    };

    /// @brief Host helper function to get the expected group range under the current MLP config.
    /// @param matrix_m Is the size of the m dimension of the matrix multiplication (m x k x n).
    /// @param matrix_n Is the size of the n dimension of the matrix multiplication (m x k x n).
    /// @return Expected group range.
    static cl::sycl::range<3> get_group_range(arguments_t &args) {
        // make sure first layer and second layer meet the condition to be fused.
        static_assert(wg_tile_m_layer1 == wg_tile_m_layer2,
                "first gemm and second gemm should have the same wg_tile_m");
        assert(args.matrix_m_layer1 == args.matrix_m_layer2);
        assert(((args.matrix_n_layer1 + wg_tile_n_layer1 - 1)
                       / wg_tile_n_layer1)
                        == 1
                && ((args.matrix_n_layer2 + wg_tile_n_layer2 - 1)
                           / wg_tile_n_layer2)
                        == 1);
        uint32_t group_range_m = (args.matrix_m_layer1 + wg_tile_m_layer1 - 1)
                / wg_tile_m_layer1;
        uint32_t group_range_n = (args.matrix_n_layer1 + wg_tile_n_layer1 - 1)
                / wg_tile_n_layer1;
        std::cout << "Group range: {1"
                  << ", " << group_range_m << ", " << group_range_n << "} \n";
        return cl::sycl::range<3> {1, group_range_m, group_range_n};
    };

    /// @brief Host helper function to get the expected nd_range under the current MLP config.
    /// @param args Is the MLP arguments for application-related runtime variables.
    /// @return Expected nd_range.
    static cl::sycl::nd_range<3> get_nd_range(arguments_t &args) {
        cl::sycl::range<3> local_range = get_local_range();
        cl::sycl::range<3> group_range = get_group_range(args);
        return cl::sycl::nd_range<3> {group_range * local_range, local_range};
    };

    /// @brief Check if the arguments can be implemented.
    /// @param args Is the MLP arguments for application-related runtime variables.
    /// @return Check result.
    static bool can_implement(arguments_t &args) {
        bool implementable = true;
        if (gemm_layer1_t::msg_type_a != msg_type::unaligned_2d) {
            if (gemm_layer1_t::msg_type_a == msg_type::block_2d) {
                implementable &= kernel::block_2d<gpu_arch::Xe,
                        dtype_a>::check_tensor((uint64_t)(args.matA_base.base),
                        args.matrix_k_layer1, args.matrix_m_layer1,
                        args.matA_ld);
            } else {
                implementable &= kernel::general_1d<gpu_arch::Xe,
                        dtype_a>::check_alignment(args.matA_base.base,
                        args.matA_ld);
            }
        }
        if (gemm_layer1_t::msg_type_b != msg_type::unaligned_2d) {
            if (gemm_layer1_t::msg_type_b == msg_type::block_2d) {
                implementable &= kernel::block_2d<gpu_arch::Xe,
                        dtype_w>::check_tensor((uint64_t)(args.matW_base.base),
                        args.matrix_n_layer1, args.matrix_k_layer1,
                        args.matW_ld);
            } else {
                implementable &= kernel::general_1d<gpu_arch::Xe,
                        dtype_w>::check_alignment(args.matW_base.base,
                        args.matW_ld);
            }
        }
        if (epilogue_layer1_t::msg_type_c != msg_type::unaligned_2d) {
            if (epilogue_layer1_t::msg_type_c == msg_type::block_2d) {
                implementable &= kernel::block_2d<gpu_arch::Xe,
                        dtype_b>::check_tensor((uint64_t)(args.matB_base.base),
                        args.matrix_n_layer1, args.matrix_m_layer1,
                        args.matB_ld);
            } else {
                implementable &= kernel::general_1d<gpu_arch::Xe,
                        dtype_b>::check_alignment(args.matB_base.base,
                        args.matB_ld);
            }
        }
        if (gemm_layer2_t::msg_type_a != msg_type::unaligned_2d) {
            if (gemm_layer2_t::msg_type_a == msg_type::block_2d) {
                implementable &= kernel::block_2d<gpu_arch::Xe,
                        dtype_b>::check_tensor((uint64_t)(args.matB_base.base),
                        args.matrix_k_layer2, args.matrix_m_layer2,
                        args.matB_ld);
            } else {
                implementable &= kernel::general_1d<gpu_arch::Xe,
                        dtype_a>::check_alignment(args.matB_base.base,
                        args.matB_ld);
            }
        }
        if (gemm_layer2_t::msg_type_b != msg_type::unaligned_2d) {
            if (gemm_layer2_t::msg_type_b == msg_type::block_2d) {
                implementable &= kernel::block_2d<gpu_arch::Xe,
                        dtype_v>::check_tensor((uint64_t)(args.matV_base.base),
                        args.matrix_n_layer2, args.matrix_k_layer2,
                        args.matV_ld);
            } else {
                implementable &= kernel::general_1d<gpu_arch::Xe,
                        dtype_v>::check_alignment(args.matV_base.base,
                        args.matV_ld);
            }
        }
        if (epilogue_layer2_t::msg_type_c != msg_type::unaligned_2d) {
            if (epilogue_layer2_t::msg_type_c == msg_type::block_2d) {
                implementable &= kernel::block_2d<gpu_arch::Xe,
                        dtype_c>::check_tensor((uint64_t)(args.matC_base.base),
                        args.matrix_n_layer2, args.matrix_m_layer2,
                        args.matC_ld);
            } else {
                implementable &= kernel::general_1d<gpu_arch::Xe,
                        dtype_c>::check_alignment(args.matC_base.base,
                        args.matC_ld);
            }
        }

        return implementable;
    }

    /// @brief Main execution function for MLP.
    /// The processing order is 1) set group-level base and boundary -> 2) gemm -> 3) epilogue.
    /// @param item Is the sycl::nd_item, returns execution related information, such as workgroup id, subgroup id...
    /// @param args Is the MLP arguments for application-related runtime variables.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    __XETLA_API KERNEL_FUNC void operator()(sycl::nd_item<3> &item,
            const arguments_t &args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        // set up workgroup level coordinates and boundaries
        int start_n = item.get_group(2) * wg_tile_n_layer1;
        int start_m = item.get_group(1) * wg_tile_m_layer1;
        int start_k = 0;
        uint32_t wg_tile_k = args.matrix_k_layer1;
        uint32_t boundary_n
                = (start_n + wg_tile_n_layer1) > args.matrix_n_layer1
                ? args.matrix_n_layer1
                : (start_n + wg_tile_n_layer1);
        uint32_t boundary_m
                = (start_m + wg_tile_m_layer1) > args.matrix_m_layer1
                ? args.matrix_m_layer1
                : (start_m + wg_tile_m_layer1);
        uint32_t boundary_k = wg_tile_k;

        uint32_t gemm_layer1_nbarr_base = nbarrier_base;
        uint32_t epilogue_layer1_nbarr_base
                = gemm_layer1_nbarr_base + gemm_layer1_t::barrier_count;
        uint32_t global_nbarr_base
                = epilogue_layer1_nbarr_base + epilogue_layer1_t::barrier_count;
        // Reuse named barrier
        uint32_t gemm_layer2_nbarr_base = nbarrier_base;
        uint32_t epilogue_layer2_nbarr_base
                = gemm_layer2_nbarr_base + gemm_layer2_t::barrier_count;

        uint32_t gemm_layer1_slm_base = slm_base;
        uint32_t epilogue_layer1_slm_base
                = gemm_layer1_slm_base + gemm_layer1_t::slm_size;
        uint32_t gemm_layer2_slm_base
                = epilogue_layer1_slm_base + epilogue_layer2_t::slm_size;
        uint32_t epilogue_layer2_slm_base
                = gemm_layer2_slm_base + gemm_layer2_t::slm_size;

        // set up arguments
        work_group_layer1_t g_layer1;
        g_layer1.init(item.get_local_linear_id());
        mem_desc_a_t mem_desc_a;
        mem_desc_w_t mem_desc_w;
        mem_desc_b_t mem_desc_b;
        //setup for matA
        mem_desc_a.init(args.matA_base, {boundary_k, boundary_m, args.matA_ld},
                {start_k, start_m});
        //setup for matB
        mem_desc_w.init(args.matW_base, {boundary_n, boundary_k, args.matW_ld},
                {start_n, start_k});
        //setup for matC
        mem_desc_b.init(args.matB_base, {boundary_n, boundary_m, args.matB_ld},
                {start_n, start_m});

        uint32_t inner_loop_count
                = (wg_tile_k + k_stride_layer1 - 1) / k_stride_layer1;
        gemm_layer1_args_t gemm_layer1_args(
                mem_desc_a, mem_desc_w, inner_loop_count);
        gemm_layer1_t gemm_layer1;
        epilogue_layer1_t epilogue_layer1;

        matAcc_layer1_t matAcc_layer1(0);
        gemm_layer1(g_layer1, matAcc_layer1, gemm_layer1_args,
                gemm_layer1_slm_base, gemm_layer1_nbarr_base);
        epilogue_layer1(g_layer1, matAcc_layer1, mem_desc_b,
                args.epilogue_layer1_args, epilogue_layer1_slm_base,
                epilogue_layer1_nbarr_base);

        // fence & barrier between two gemm
        xetla_fence();
        xetla_nbarrier_t<work_group_layer2_t::size, work_group_layer2_t::size>
                nbarrier_global;
        nbarrier_global.init_nbarrier(
                global_nbarr_base, nbarrier_role::producer_consumer);
        nbarrier_global.arrive_wait();

        // set up workgroup level coordinates and boundaries
        start_n = item.get_group(2) * wg_tile_n_layer2;
        start_m = item.get_group(1) * wg_tile_m_layer2;
        start_k = 0;
        wg_tile_k = args.matrix_k_layer2;
        boundary_n = (start_n + wg_tile_n_layer2) > args.matrix_n_layer2
                ? args.matrix_n_layer2
                : (start_n + wg_tile_n_layer2);
        boundary_m = (start_m + wg_tile_m_layer2) > args.matrix_m_layer2
                ? args.matrix_m_layer2
                : (start_m + wg_tile_m_layer2);
        boundary_k = wg_tile_k;

        // set up arguments
        // reuse mem_desc_b
        work_group_layer2_t g_layer2;
        g_layer2.init(item.get_local_linear_id());
        mem_desc_v_t mem_desc_v;
        mem_desc_c_t mem_desc_c;
        //setup for matA
        mem_desc_b.init(args.matB_base, {boundary_k, boundary_m, args.matB_ld},
                {start_k, start_m});
        //setup for matB
        mem_desc_v.init(args.matV_base, {boundary_n, boundary_k, args.matV_ld},
                {start_n, start_k});
        //setup for matC
        mem_desc_c.init(args.matC_base, {boundary_n, boundary_m, args.matC_ld},
                {start_n, start_m});

        inner_loop_count = (wg_tile_k + k_stride_layer2 - 1) / k_stride_layer2;
        gemm_layer2_args_t gemm_layer2_args(
                mem_desc_b, mem_desc_v, inner_loop_count);
        gemm_layer2_t gemm_layer2;
        epilogue_layer2_t epilogue_layer2;

        matAcc_layer2_t matAcc_layer2(0);
        gemm_layer2(g_layer2, matAcc_layer2, gemm_layer2_args,
                gemm_layer2_slm_base, gemm_layer2_nbarr_base);
        epilogue_layer2(g_layer2, matAcc_layer2, mem_desc_c,
                args.epilogue_layer2_args, epilogue_layer2_slm_base,
                epilogue_layer2_nbarr_base);
    }
};

/// @} xetla_MLP

} // namespace gpu::xetla::kernel
