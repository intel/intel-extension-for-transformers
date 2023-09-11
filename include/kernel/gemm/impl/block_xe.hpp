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

#include "kernel/gemm/api.hpp"
#include "kernel/gemm/common.hpp"
#include "kernel/gemm/dispatch_policy.hpp"

namespace gpu::xetla::kernel {

/// @addtogroup xetla_gemm
/// @{

/// @brief Persistent-Thread GEMM functor, specialized for Xe architecture.
///
/// @tparam brgemm_t_ Is the brgemm functor to compose a GEMM.
/// @tparam epilogue_t_ Is the epilogue functor to compose a GEMM.
/// @tparam wg_num_n_ Is the x-dir workgroup number of repeat block.
template <typename brgemm_t_, typename epilogue_t_, int wg_num_n_>
class gemm_t<dispatch_policy_block<wg_num_n_>, brgemm_t_, epilogue_t_> {
    using brgemm_t = brgemm_t_;
    using epilogue_t = epilogue_t_;
    using brgemm_args_t = typename brgemm_t::arguments_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    using tile_shape = typename brgemm_t::tile_shape;
    static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
    static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t k_stride = brgemm_t::k_stride;

    static constexpr int wg_num_n = wg_num_n_;
    using dispatch_policy = dispatch_policy_block<wg_num_n>;
    static constexpr uint32_t max_wg_num = dispatch_policy::max_wg_num;
    static constexpr int wg_num_m = dispatch_policy::wg_num_m;
    using work_group_t = typename brgemm_t::work_group_t;

    static constexpr gpu_arch arch_tag = gpu_arch::Xe;
    static_assert(
            arch_tag == brgemm_t::arch_tag, "arch_tag should be the same");
    static_assert(
            arch_tag == epilogue_t::arch_tag, "arch_tag should be the same");
    static_assert(std::is_same<typename brgemm_t::tile_shape,
                          typename epilogue_t::tile_shape>::value,
            "tile_shape should be the same");

    using mem_desc_a_t = typename brgemm_t::mem_desc_a_t;
    using mem_desc_b_t = typename brgemm_t::mem_desc_b_t;
    using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;
    using matA_base_t = typename mem_desc_a_t::base_t;
    using matB_base_t = typename mem_desc_b_t::base_t;
    using matC_base_t = typename mem_desc_c_t::base_t;
    using dtype_a = typename mem_desc_a_t::dtype;
    using dtype_b = typename mem_desc_b_t::dtype;
    using dtype_c = typename mem_desc_c_t::dtype;
    using matAcc_t = typename brgemm_t::matAcc_t;

public:
    /// @brief GEMM arguments.
    /// This is the interface for users to pass the application-related runtime variables.
    struct arguments_t {
        /// @brief Is the size of the m dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_m;
        /// @brief Is the size of the k dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_k;
        /// @brief Is the size of the n dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_n;
        /// @brief Is the leading dimension (pitch) size of the matrix A in memory.
        uint32_t matA_ld;
        /// @brief Is the leading dimension (pitch) size of the matrix B in memory.
        uint32_t matB_ld;
        /// @brief Is the leading dimension (pitch) size of the matrix C in memory.
        uint32_t matC_ld;
        /// @brief Is the base address of matrix A.
        matA_base_t matA_base;
        /// @brief Is the base address of matrix B.
        matB_base_t matB_base;
        /// @brief Is the base address of matrix C.
        matC_base_t matC_base;
        /// @brief Is the epilogue arguments.
        epilogue_args_t epilogue_args;

        /// @brief Constructs arguments with default method.
        inline arguments_t() = default;

        /// @brief Set for device copyable
        static constexpr bool host_callable = true;

        /// @brief Constructs arguments with initialization list.
        /// @param matrix_m_ Is the size of the m dimension of the matrix multiplication (m x k x n).
        /// @param matrix_k_ Is the size of the k dimension of the matrix multiplication (m x k x n).
        /// @param matrix_n_ Is the size of the n dimension of the matrix multiplication (m x k x n).
        /// @param matA_base_ Is the base address of matrix A.
        /// @param matA_ld_ Is the leading dimension (pitch) size of the matrix A in memory.
        /// @param matB_base_ Is the base address of matrix B.
        /// @param matB_ld_ Is the leading dimension (pitch) size of the matrix B in memory.
        /// @param matC_base_ Is the base address of matrix C.
        /// @param matC_ld_ Is the leading dimension (pitch) size of the matrix C in memory.
        /// @param epilogue_args_ Is the epilogue arguments.
        inline arguments_t(uint32_t matrix_m_, uint32_t matrix_k_,
                uint32_t matrix_n_, matA_base_t matA_base_, uint32_t matA_ld_,
                matB_base_t matB_base_, uint32_t matB_ld_,
                matC_base_t matC_base_, uint32_t matC_ld_,
                epilogue_args_t epilogue_args_ = {})
            : matrix_m(matrix_m_)
            , matrix_k(matrix_k_)
            , matrix_n(matrix_n_)
            , matA_base(matA_base_)
            , matA_ld(matA_ld_)
            , matB_base(matB_base_)
            , matB_ld(matB_ld_)
            , matC_base(matC_base_)
            , matC_ld(matC_ld_)
            , epilogue_args(epilogue_args_) {}
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // inline ~arguments_t(){}
        inline arguments_t(const arguments_t &args)
            : matrix_m(args.matrix_m)
            , matrix_k(args.matrix_k)
            , matrix_n(args.matrix_n)
            , matA_base(args.matA_base)
            , matA_ld(args.matA_ld)
            , matB_base(args.matB_base)
            , matB_ld(args.matB_ld)
            , matC_base(args.matC_base)
            , matC_ld(args.matC_ld)
            , epilogue_args(args.epilogue_args) {}
        inline arguments_t &operator=(const arguments_t &args) {
            this->matrix_m = args.matrix_m;
            this->matrix_k = args.matrix_k;
            this->matrix_n = args.matrix_n;
            this->matA_base = args.matA_base;
            this->matA_ld = args.matA_ld;
            this->matB_base = args.matB_base;
            this->matB_ld = args.matB_ld;
            this->matC_base = args.matC_base;
            this->matC_ld = args.matC_ld;
            this->epilogue_args = args.epilogue_args;
            return *this;
        }
    };

    /// @brief Gets named_barrier id consumption count.
    /// Users query and get a named_barrier id consumption count in compile time.
    /// @return The count of named barriers required.
    __XETLA_API static constexpr uint32_t get_barrier_count() {
        constexpr uint32_t count
                = brgemm_t::barrier_count + epilogue_t::barrier_count;
        static_assert(
                count <= 32, "The named_barrier count should be less than 32!");
        return count;
    }

    /// @brief Gets local memory size consumption.
    /// Users query and get a local memory consumption size in compile time.
    /// @return The size of local memory required.
    __XETLA_API static constexpr uint32_t get_slm_size() {
        constexpr uint32_t size = brgemm_t::slm_size + epilogue_t::slm_size;
        static_assert(size <= (128 * 1024),
                "The local memory size should be less than 128KB!");
        return size;
    };

    /// @brief Host helper function to get the expected local range under the current GEMM config.
    /// @return Expected local range.
    static cl::sycl::range<3> get_local_range() {
        uint32_t local_range_m = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
        uint32_t local_range_n = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
        std::cout << "Local range: {" << 1 << ", " << local_range_m << ", "
                  << local_range_n << "} \n";
        assert(local_range_m * local_range_n <= 32);
        return cl::sycl::range<3> {1, local_range_m, local_range_n};
    };

    /// @brief Host helper function to auto padding the group range under the current GEMM config.
    /// @param matrix_m Is the size of the m dimension of the matrix multiplication (m x k x n).
    /// @param matrix_n Is the size of the n dimension of the matrix multiplication (m x k x n).
    /// @return Expected group range.
    static cl::sycl::range<3> get_group_range(
            uint32_t matrix_m, uint32_t matrix_n) {
        uint32_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
        uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;

        group_range_m = (group_range_m + wg_num_m - 1) / wg_num_m * wg_num_m;
        group_range_n = (group_range_n + wg_num_n - 1) / wg_num_n * wg_num_n;

        std::cout << "Group range: {" << 1 << ", " << group_range_m << ", "
                  << group_range_n << "} \n";
        return cl::sycl::range<3> {1, group_range_m, group_range_n};
    };

    /// @brief Host helper function to get the expected nd_range under the current GEMM config.
    /// @param args Is the GEMM arguments for application-related runtime variables.
    /// @return Expected nd_range.
    static cl::sycl::nd_range<3> get_nd_range(arguments_t &args) {
        cl::sycl::range<3> local_range = get_local_range();
        cl::sycl::range<3> group_range
                = get_group_range(args.matrix_m, args.matrix_n);
        return cl::sycl::nd_range<3> {group_range * local_range, local_range};
    };

    /// @brief Check if the arguments can be implemented.
    /// @param args Is the GEMM arguments for application-related runtime variables.
    /// @return Check result.
    static bool can_implement(arguments_t &args) {
        bool implementable = true;
        if (brgemm_t::is_2d_block_a) {
            implementable
                    &= core::block_2d<gpu_arch::Xe, dtype_a>::check_tensor(
                            (uint64_t)(args.matA_base.base),
                            brgemm_t::is_col_major_a ? args.matrix_m
                                                     : args.matrix_k,
                            brgemm_t::is_col_major_a ? args.matrix_k
                                                     : args.matrix_m,
                            args.matA_ld);
        } else {
            implementable &= kernel::general_1d<gpu_arch::Xe,
                    dtype_a>::check_alignment(args.matA_base.base,
                    args.matA_ld);
        }
        if (brgemm_t::is_2d_block_b) {
            implementable
                    &= core::block_2d<gpu_arch::Xe, dtype_b>::check_tensor(
                            (uint64_t)(args.matB_base.base),
                            brgemm_t::is_col_major_b ? args.matrix_k
                                                     : args.matrix_n,
                            brgemm_t::is_col_major_b ? args.matrix_n
                                                     : args.matrix_k,
                            args.matB_ld);
        } else {
            implementable &= kernel::general_1d<gpu_arch::Xe,
                    dtype_b>::check_alignment(args.matB_base.base,
                    args.matB_ld);
        }
        if (epilogue_t::is_2d_block_c) {
            implementable
                    &= core::block_2d<gpu_arch::Xe, dtype_c>::check_tensor(
                            (uint64_t)(args.matC_base.base), args.matrix_n,
                            args.matrix_m, args.matC_ld);
        } else {
            implementable &= kernel::general_1d<gpu_arch::Xe,
                    dtype_c>::check_alignment(args.matC_base.base,
                    args.matC_ld);
        }

        return implementable;
    }

    /// @brief Main execution function for GEMM.
    /// The processing order is 1) set group-level base and boundary -> 2) brgemm -> 3) epilogue.
    /// @param ei Is the execution item, returns execution related information, such as workgroup id, subgroup id...
    /// @param args Is the GEMM arguments for application-related runtime variables.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    __XETLA_API KERNEL_FUNC void operator()(xetla_exec_item<3> &ei,
            const arguments_t &args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        // remap workgroups into a 2D block shape and re-coordinates workgroups
        uint32_t group_range_n = ei.get_group_range(2);
        uint32_t group_range_m = ei.get_group_range(1);
        uint32_t wg_repeat_n = group_range_n / wg_num_n;
        uint32_t wg_repeat_m = group_range_m / wg_num_m;
        uint32_t repeat_id = ei.get_group_linear_id() / max_wg_num;

        uint32_t repeat_id_n = repeat_id % wg_repeat_n;
        uint32_t repeat_id_m = repeat_id / wg_repeat_n;

        uint32_t repeat_start_n_0 = repeat_id_n * wg_num_n * wg_tile_n;
        uint32_t repeat_start_n_1
                = (wg_repeat_n - repeat_id_n - 1) * wg_num_n * wg_tile_n;

        uint32_t repeat_start_n
                = (repeat_id_m & 1) == 0 ? repeat_start_n_0 : repeat_start_n_1;
        uint32_t repeat_start_m = repeat_id_m * wg_num_m * wg_tile_m;

        uint32_t wg_inner_id = ei.get_group_linear_id() % max_wg_num;
        uint32_t wg_coord_n = wg_inner_id % wg_num_n;
        uint32_t wg_coord_m = wg_inner_id / wg_num_n;
        int start_n = repeat_start_n + wg_coord_n * wg_tile_n;
        int start_m = repeat_start_m + wg_coord_m * wg_tile_m;

        int start_k = 0;
        uint32_t wg_tile_k = args.matrix_k;

        uint32_t boundary_n = (start_n + wg_tile_n) > args.matrix_n
                ? args.matrix_n
                : (start_n + wg_tile_n);
        uint32_t boundary_m = (start_m + wg_tile_m) > args.matrix_m
                ? args.matrix_m
                : (start_m + wg_tile_m);
        uint32_t boundary_k = wg_tile_k;

        uint32_t brgemm_slm_base = slm_base;
        uint32_t brgemm_nbarr_base = nbarrier_base;
        uint32_t epilogue_slm_base = brgemm_slm_base + brgemm_t::slm_size;
        uint32_t epilogue_nbarr_base
                = brgemm_nbarr_base + brgemm_t::barrier_count;

        // set up arguments
        work_group_t g;
        g.init(ei.get_local_linear_id());
        mem_desc_a_t mem_desc_a;
        mem_desc_b_t mem_desc_b;
        mem_desc_c_t mem_desc_c;
        //setup for matA
        if constexpr (mem_desc_a_t::is_local) {
            mem_desc_a.init(
                    args.matA_base, {wg_tile_k, wg_tile_m, wg_tile_k}, {0, 0});
        } else {
            mem_desc_a.init(args.matA_base,
                    {boundary_k, boundary_m, args.matA_ld}, {start_k, start_m});
        }
        //setup for matB
        if constexpr (mem_desc_b_t::is_local) {
            mem_desc_b.init(
                    args.matB_base, {wg_tile_n, wg_tile_k, wg_tile_n}, {0, 0});
        } else {
            mem_desc_b.init(args.matB_base,
                    {boundary_n, boundary_k, args.matB_ld}, {start_n, start_k});
        }
        //setup for matC
        if constexpr (mem_desc_c_t::is_local) {
            mem_desc_c.init(
                    args.matC_base, {wg_tile_n, wg_tile_m, wg_tile_n}, {0, 0});
        } else {
            mem_desc_c.init(args.matC_base,
                    {boundary_n, boundary_m, args.matC_ld}, {start_n, start_m});
        }
        uint32_t inner_loop_count = (wg_tile_k + k_stride - 1) / k_stride;
        brgemm_args_t brgemm_args(mem_desc_a, mem_desc_b, inner_loop_count);
        brgemm_t brgemm;
        epilogue_t epilogue;

        matAcc_t matAcc(0);
        brgemm(g, matAcc, brgemm_args, brgemm_slm_base, brgemm_nbarr_base);
        epilogue(g, matAcc, mem_desc_c, args.epilogue_args, epilogue_slm_base,
                epilogue_nbarr_base);
    }
};

/// @} xetla_gemm

} // namespace gpu::xetla::kernel
