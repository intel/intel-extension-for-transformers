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

#include "experimental/kernel/gemm/common.hpp"
#include "experimental/kernel/gemm/dispatch_policy.hpp"

namespace gpu::xetla::kernel {

/// @addtogroup xetla_gemm
/// @{

/// @brief Is the GEMM functor, specialized in bit4 matB kslicing dispatch policy and Xe architecture.
///
/// @tparam global_kslicing_ratio_ Is the k dim split ratio between groups.
/// @tparam local_kslicing_ratio_ Is the k dim split ratio within a group.
/// @tparam brgemm_t_ Is the brgemm functor to compose a GEMM.
/// @tparam epilogue_t_ Is the epilogue functor to compose a GEMM.
template <int global_kslicing_ratio_, int local_kslicing_ratio_,
        typename brgemm_t_, typename epilogue_t_>
class gemm_t<dispatch_policy_int4_dequantize_kslicing<global_kslicing_ratio_,
                     local_kslicing_ratio_, gpu_arch::Xe>,
        brgemm_t_, epilogue_t_> {
    using brgemm_t = brgemm_t_;
    using epilogue_t = epilogue_t_;
    using brgemm_args_t = typename brgemm_t::arguments_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    using tile_shape = typename brgemm_t::tile_shape;
    static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
    static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t real_wg_tile_m = sg_tile_m * wg_size_y;
    static constexpr uint32_t real_wg_tile_n = sg_tile_n * wg_size_x;

    static constexpr uint32_t k_stride = brgemm_t::k_stride;
    static constexpr uint32_t dequant_s = brgemm_t::dequant_s;
    static constexpr uint32_t pack_ratio = brgemm_t::pack_ratio;
    using work_group_t = typename brgemm_t::work_group_t;
    static constexpr uint32_t work_group_size = work_group_t::size;

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
    using mem_desc_scale_t = typename brgemm_t::mem_desc_scale_t;
    using mem_desc_zero_pt_t = typename brgemm_t::mem_desc_zero_pt_t;
    using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;
    using matA_base_t = typename mem_desc_a_t::base_t;
    using matB_base_t = typename mem_desc_b_t::base_t;
    using matC_base_t = typename mem_desc_c_t::base_t;
    using scale_base_t = typename mem_desc_scale_t::base_t;
    using zero_pt_base_t = typename mem_desc_zero_pt_t::base_t;

    using dtype_a = typename mem_desc_a_t::dtype;
    using dtype_b = typename mem_desc_b_t::dtype;
    using dtype_c = typename mem_desc_c_t::dtype;
    using dtype_scale = typename mem_desc_scale_t::dtype;
    using dtype_zero_pt = typename mem_desc_zero_pt_t::dtype;
    using matAcc_t = typename brgemm_t::matAcc_t;

    static_assert(brgemm_t::compute_policy::is_int4_matB_policy,
            "should match with 4bit brgemm impl");

    using update_method = typename epilogue_t::update_method;
    static constexpr uint32_t global_kslicing_ratio = global_kslicing_ratio_;
    static constexpr uint32_t local_kslicing_ratio = local_kslicing_ratio_;
    static_assert((global_kslicing_ratio > 0) && (local_kslicing_ratio > 0),
            "min slicing ratio is 1");

    static_assert((local_kslicing_ratio & (local_kslicing_ratio - 1)) == 0,
            "local_kslicing_ratio should be power of 2!");
    static_assert(global_kslicing_ratio == 1
                    || std::is_same<remove_const_t<dtype_c>, float>::value
                    || std::is_same<remove_const_t<dtype_c>, int>::value,
            "for global_kslicing_ratio > 1, current we only support float or "
            "int for matC");
    static_assert(global_kslicing_ratio == 1
                    || std::is_same<update_method, result_reduce_sum>::value,
            "for global_kslicing_ratio > 1, the update method should be reduce "
            "sum");

    using kslicing_t = group::cooperative_reduce_t<reduce_op::sum, tile_shape,
            matAcc_t, local_kslicing_ratio, gpu_arch::Xe>;
    using mat_slice_t = typename kslicing_t::mat_slice_t;

    static constexpr uint32_t brgemm_nbarr_count = brgemm_t::barrier_count;
    static constexpr uint32_t brgemm_slm_size = brgemm_t::slm_size;

    static constexpr uint32_t epilogue_nbarr_count = epilogue_t::barrier_count;
    static constexpr uint32_t epilogue_slm_size = epilogue_t::slm_size;

    static constexpr uint32_t kslicing_nbarr_count = kslicing_t::barrier_count;
    static constexpr uint32_t kslicing_slm_size = kslicing_t::slm_size;

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

        scale_base_t scale_base;
        zero_pt_base_t zero_pt_base;
        uint32_t scale_ld;
        uint32_t zero_pt_ld;

        /// @brief Constructs arguments with default method.
        inline arguments_t() = default;

        /// @brief Set for device copyable
        static constexpr bool host_callable = true;

        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // ~arguments_t(){}

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
                scale_base_t scale_base_, uint32_t scale_ld_,
                zero_pt_base_t zero_pt_base_, uint32_t zero_pt_ld_,
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
            , scale_base(scale_base_)
            , scale_ld(scale_ld_)
            , zero_pt_base(zero_pt_base_)
            , zero_pt_ld(zero_pt_ld_)
            , epilogue_args(epilogue_args_) {}
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
            , scale_base(args.scale_base)
            , scale_ld(args.scale_ld)
            , zero_pt_base(args.zero_pt_base)
            , zero_pt_ld(args.zero_pt_ld)
            , epilogue_args(args.epilogue_args) {}
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // inline ~arguments_t(){}
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
            this->scale_base = args.scale_base;
            this->scale_ld = args.scale_ld;
            this->zero_pt_base = args.zero_pt_base;
            this->zero_pt_ld = args.zero_pt_ld;
            this->epilogue_args = args.epilogue_args;
            return *this;
        }
    };

    /// @brief Gets named_barrier id consumption count.
    /// Users query and get a named_barrier id consumption count in compile time.
    /// @return The count of named barriers required.
    __XETLA_API static constexpr uint32_t get_barrier_count() {
        constexpr uint32_t count = brgemm_nbarr_count * local_kslicing_ratio
                + kslicing_nbarr_count
                + epilogue_nbarr_count * local_kslicing_ratio;
        static_assert(
                count <= 32, "The named_barrier count should be less than 32!");
        return count;
    }

    /// @brief Gets local memory size consumption.
    /// Users query and get a local memory consumption size in compile time.
    /// @return The size of local memory required.
    __XETLA_API static constexpr uint32_t get_slm_size() {
        constexpr uint32_t size = brgemm_slm_size * local_kslicing_ratio
                + kslicing_slm_size + epilogue_slm_size * local_kslicing_ratio;
        static_assert(size <= (128 * 1024),
                "The local memory size should be less than 128KB!");
        return size;
    }

    /// @brief Host helper function to get the expected local range under the current GEMM config.
    /// @return Expected local range.
    static cl::sycl::range<3> get_local_range() {
        uint32_t local_range_m = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
        uint32_t local_range_n = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
        std::cout << "Local range: {" << local_kslicing_ratio << ", "
                  << local_range_m << ", " << local_range_n << "} \n";
        assert(local_range_m * local_range_n * local_kslicing_ratio <= 32);
        return cl::sycl::range<3> {
                local_kslicing_ratio, local_range_m, local_range_n};
    };

    /// @brief Host helper function to get the expected group range under the current GEMM config.
    /// @param matrix_m Is the size of the m dimension of the matrix multiplication (m x k x n).
    /// @param matrix_n Is the size of the n dimension of the matrix multiplication (m x k x n).
    /// @return Expected group range.
    static cl::sycl::range<3> get_group_range(
            uint32_t matrix_m, uint32_t matrix_n) {
        uint32_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
        uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;
        std::cout << "Group range: {" << global_kslicing_ratio << ", "
                  << group_range_m << ", " << group_range_n << "} \n";
        return cl::sycl::range<3> {
                global_kslicing_ratio, group_range_m, group_range_n};
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
            bool implementable_a = detail::check_2d_block_restriction<dtype_a>(
                    args.matA_base.base, args.matA_ld);
            if (!implementable_a) {
                std::cout << "matA is not well aligned!" << std::endl;
            }
            implementable &= implementable_a;
        } else {
            bool implementable_a = detail::check_dw_align<dtype_a>(
                    args.matA_base.base, args.matA_ld);
            if (!implementable_a) {
                std::cout << "matA is not well aligned!" << std::endl;
            }
            implementable &= implementable_a;
        }
        if (brgemm_t::is_2d_block_b) {
            // for 4bit matB
            bool implementable_b = detail::check_2d_block_restriction<dtype_b>(
                    args.matB_base.base, args.matB_ld / pack_ratio);
            if (!implementable_b) {
                std::cout << "matB is not well aligned!" << std::endl;
            }
            implementable &= implementable_b;
        } else {
            // for 4bit matB
            bool implementable_b = detail::check_dw_align<dtype_b>(
                    args.matB_base.base, args.matB_ld / pack_ratio);
            if (!implementable_b) {
                std::cout << "matB is not well aligned!" << std::endl;
            }
            implementable &= implementable_b;
        }
        if (epilogue_t::is_2d_block_c) {
            bool implementable_c = detail::check_2d_block_restriction<dtype_c>(
                    args.matC_base.base, args.matC_ld);
            if (!implementable_c) {
                std::cout << "matC is not well aligned!" << std::endl;
            }
            implementable &= implementable_c;
        } else {
            bool implementable_c = detail::check_dw_align<dtype_c>(
                    args.matC_base.base, args.matC_ld);
            if (!implementable_c) {
                std::cout << "matC is not well aligned!" << std::endl;
            }
            implementable &= implementable_c;
        }
        // check for int4x2
        implementable &= ((args.matB_ld % pack_ratio == 0)
                && (args.zero_pt_ld % pack_ratio == 0)
                && (args.matrix_n % pack_ratio == 0));

        return implementable;
    }

    /// @brief Main execution function for GEMM.
    /// The processing order is 1) set group-level base and boundary, split group to workgroups ->
    /// 2) #local_kslicing_ratio x brgemms -> 3) local kslicing -> 4) #local_kslicing_ratio x epilogues.
    /// @param ei Is the execution item, returns execution related information, such as workgroup id, subgroup id...
    /// @param args Is the GEMM arguments for application-related runtime variables.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    __XETLA_API KERNEL_FUNC void operator()(xetla_exec_item<3> &ei,
            const arguments_t &args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        // set up workgroup level coordinates and boundaries
        work_group_t g(ei.get_local_linear_id() % work_group_size);
        uint32_t wg_id = ei.get_local_linear_id() / work_group_size;
        int start_n = ei.get_group(2) * wg_tile_n;
        int start_m = ei.get_group(1) * wg_tile_m;
        int start_k = 0;
        uint32_t wg_tile_k = args.matrix_k;
        uint32_t boundary_n = (start_n + wg_tile_n) > args.matrix_n
                ? args.matrix_n
                : (start_n + wg_tile_n);
        uint32_t boundary_m = (start_m + wg_tile_m) > args.matrix_m
                ? args.matrix_m
                : (start_m + wg_tile_m);
        uint32_t boundary_k = wg_tile_k;
        if constexpr (global_kslicing_ratio > 1) {
            wg_tile_k = (wg_tile_k + global_kslicing_ratio - 1)
                    / global_kslicing_ratio;
            start_k = start_k + ei.get_group(0) * wg_tile_k;
            boundary_k = (start_k + wg_tile_k) > boundary_k
                    ? boundary_k
                    : (start_k + wg_tile_k);
        }
        if constexpr (local_kslicing_ratio > 1) {
            wg_tile_k = (wg_tile_k + local_kslicing_ratio - 1)
                    / local_kslicing_ratio;
            start_k = start_k + wg_id * wg_tile_k;
            boundary_k = (start_k + wg_tile_k) > boundary_k
                    ? boundary_k
                    : (start_k + wg_tile_k);
        }

        int start_x_scale = start_n;
        int start_y_scale = start_k / dequant_s;

        int start_x_zero_pt = start_n / pack_ratio;
        int start_y_zero_pt = start_k / dequant_s;

        // set up arguments
        uint32_t brgemm_slm_base = slm_base;
        uint32_t brgemm_nbarr_base = nbarrier_base;
        if constexpr (local_kslicing_ratio > 1) {
            brgemm_slm_base = slm_base + wg_id * brgemm_slm_size;
            brgemm_nbarr_base = nbarrier_base + wg_id * brgemm_nbarr_count;
        }
        uint32_t kslicing_slm_base
                = slm_base + local_kslicing_ratio * brgemm_slm_size;
        uint32_t kslicing_nbarr_base
                = nbarrier_base + local_kslicing_ratio * brgemm_nbarr_count;
        uint32_t epilogue_slm_base = kslicing_slm_base + kslicing_slm_size;
        uint32_t epilogue_nbarr_base
                = kslicing_nbarr_base + kslicing_nbarr_count;

        mem_desc_a_t mem_desc_a;
        mem_desc_b_t mem_desc_b;
        mem_desc_c_t mem_desc_c;
        //setup for matA

        mem_desc_a.init(args.matA_base, {boundary_k, boundary_m, args.matA_ld},
                {start_k, start_m});
        mem_desc_b.init(args.matB_base,
                {boundary_n / pack_ratio, boundary_k,
                        args.matB_ld / pack_ratio},
                {int(start_n / pack_ratio), start_k});
        mem_desc_scale_t mem_desc_scale(args.scale_base,
                {args.matrix_n, args.matrix_k / dequant_s, args.scale_ld},
                {start_x_scale, start_y_scale});
        mem_desc_zero_pt_t mem_desc_zero_pt(args.zero_pt_base,
                {args.matrix_n / pack_ratio, args.matrix_k / dequant_s,
                        args.zero_pt_ld / pack_ratio},
                {start_x_zero_pt, start_y_zero_pt});

        uint32_t inner_loop_count = (wg_tile_k + k_stride - 1) / k_stride;
        brgemm_args_t brgemm_args(mem_desc_a, mem_desc_b, inner_loop_count,
                mem_desc_scale, mem_desc_zero_pt);
        matAcc_t matAcc;
        matAcc.init(0);
        brgemm_t brgemm;
        brgemm(g, matAcc, brgemm_args, brgemm_slm_base, brgemm_nbarr_base);

        kslicing_t kslicing(wg_id);
        mat_slice_t mat_slice;
        kslicing(g, mat_slice, matAcc, kslicing_slm_base, kslicing_nbarr_base);

        //setup for matC
        //set up cooperative offset for matC store
        int32_t coop_offset_n = kslicing.coop_id_x * mat_slice_t::tile_size_x;
        int32_t coop_offset_m = kslicing.coop_id_y * mat_slice_t::tile_size_y;
        if constexpr (mem_desc_c_t::is_local) {
            mem_desc_c.init(args.matC_base,
                    {real_wg_tile_n, real_wg_tile_m, real_wg_tile_n},
                    {coop_offset_n, coop_offset_m});
        } else {
            mem_desc_c.init(args.matC_base,
                    {boundary_n, boundary_m, args.matC_ld},
                    {start_n + coop_offset_n, start_m + coop_offset_m});
        }
        epilogue_t epilogue;
        epilogue(g, mat_slice, mem_desc_c, args.epilogue_args,
                epilogue_slm_base, epilogue_nbarr_base);
    }
};

/// @} xetla_gemm

} // namespace gpu::xetla::kernel
