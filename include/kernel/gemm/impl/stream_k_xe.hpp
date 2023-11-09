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

/// @addtogroup xetla_gemm_universal
/// @{

/// @brief Default GEMM_UNIVERSAL functor, specialized for Xe architecture.
///
/// @tparam gemm_t_ Is the gemm functor to compose a GEMM_UNIVERSAL.
/// @tparam epilogue_t_ Is the epilogue functor to compose a GEMM_UNIVERSAL.
template <typename gemm_t_, typename epilogue_t_>
class gemm_universal_t<dispatch_policy_stream_k<gpu_arch::Xe>, gemm_t_,
        epilogue_t_> {
    using gemm_t = gemm_t_;
    using epilogue_t = epilogue_t_;
    using gemm_args_t = typename gemm_t::arguments_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;
    using dispatch_stream_k = dispatch_policy_stream_k<gpu_arch::Xe>;

    //Scratchspace to accumulate partials
    using mem_desc_d_t
            = mem_desc_t<float, mem_layout::row_major, mem_space::global>;
    //Workspace to sync across xecores
    using mem_desc_atomic_sync_t
            = mem_desc_t<uint32_t, mem_layout::row_major, mem_space::global>;

    using tile_shape = typename gemm_t::tile_shape;
    static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
    static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t real_wg_tile_m = sg_tile_m * wg_size_y;
    static constexpr uint32_t real_wg_tile_n = sg_tile_n * wg_size_x;
    //tile_k used in GEMMs
    static constexpr uint32_t k_stride = gemm_t::k_stride;

    using work_group_t = typename gemm_t::work_group_t;

    static constexpr gpu_arch arch_tag = gpu_arch::Xe;
    static_assert(arch_tag == gemm_t::arch_tag, "arch_tag should be the same");

    using mem_desc_a_t = typename gemm_t::mem_desc_a_t;
    using mem_desc_b_t = typename gemm_t::mem_desc_b_t;
    using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;
    using matA_base_t = typename mem_desc_a_t::base_t;
    using matB_base_t = typename mem_desc_b_t::base_t;
    using matC_base_t = typename mem_desc_c_t::base_t;
    using matD_base_t = typename mem_desc_d_t::base_t;
    using matatomic_sync_base_t = typename mem_desc_atomic_sync_t::base_t;
    using dtype_a = typename mem_desc_a_t::dtype;
    using dtype_b = typename mem_desc_b_t::dtype;
    using dtype_c = typename mem_desc_c_t::dtype;
    using matAcc_t = typename gemm_t::matAcc_t;
    using epilogue_stream_k_t = group::epilogue_stream_k_t<tile_shape,
            epilogue_t, mem_desc_d_t, mem_desc_atomic_sync_t>;

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
        /// @brief Is the leading dimension (pitch) size of the matrix D in memory.
        uint32_t matD_ld;
        /// @brief Is the leading dimension (pitch) size of the atomic_sync space in memory.
        uint32_t matatomic_sync_ld;
        /// @brief Is the base address of matrix A.
        matA_base_t matA_base;
        /// @brief Is the base address of matrix B.
        matB_base_t matB_base;
        /// @brief Is the base address of matrix C.
        matC_base_t matC_base;
        /// @brief Is the base address of matrix D
        matD_base_t matD_base;
        /// @brief Is the base address of groupsync buf
        matatomic_sync_base_t matatomic_sync_base;
        /// @brief Is the workgroup split streamk arguments.
        dispatch_stream_k stream_k_args;
        /// @brief Is the epilogue arguments.
        epilogue_args_t epilogue_args;

        /// @brief Constructs arguments with default method.
        inline arguments_t() = default;
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // ~arguments_t(){}

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
        /// @param matC_ld_ Is the leading  dimension (pitch) size of the matrix C in memory.
        /// @param matD_base_ Is the base address of matrix D.
        /// @param matD_ld_ Is the leading  dimension (pitch) size of the matrix D in memory.
        /// @param matatomic_sync_base_ Is the base address of matrix atomic_sync.
        /// @param matatomic_sync_ld_ Is the leading  dimension (pitch) size of the global sync space in memory.
        /// @param epilogue_args_ Is the epilogue arguments.
        inline arguments_t(uint32_t matrix_m_, uint32_t matrix_k_,
                uint32_t matrix_n_, matA_base_t matA_base_, uint32_t matA_ld_,
                matB_base_t matB_base_, uint32_t matB_ld_,
                matC_base_t matC_base_, uint32_t matC_ld_,
                matD_base_t matD_base_, uint32_t matD_ld_,
                matatomic_sync_base_t matatomic_sync_base_,
                uint32_t matatomic_sync_ld_, dispatch_stream_k &stream_k_args_,
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
            , matD_base(matD_base_)
            , matD_ld(matD_ld_)
            , matatomic_sync_base(matatomic_sync_base_)
            , matatomic_sync_ld(matatomic_sync_ld_)
            , stream_k_args(stream_k_args_)
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
            , matD_base(args.matD_base)
            , matD_ld(args.matD_ld)
            , matatomic_sync_base(args.matatomic_sync_base)
            , matatomic_sync_ld(args.matatomic_sync_ld)
            , stream_k_args(args.stream_k_args)
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
            this->matD_base = args.matD_base;
            this->matD_ld = args.matD_ld;
            this->matatomic_sync_base = args.matatomic_sync_base;
            this->matatomic_sync_ld = args.matatomic_sync_ld;
            this->stream_k_args = args.stream_k_args;
            this->epilogue_args = args.epilogue_args;
            return *this;
        }
    };

    /// @brief Gets named_barrier id consumption count.
    /// Users query and get a named_barrier id consumption count in compile time.
    /// @return The count of named barriers required.
    __XETLA_API static constexpr uint32_t get_barrier_count() {
        constexpr uint32_t count = gemm_t::barrier_count
                + epilogue_t::barrier_count
                + epilogue_stream_k_t::barrier_count;
        static_assert(
                count <= 32, "The named_barrier count should be less than 32!");
        return count;
    }

    /// @brief Gets local memory size consumption.
    /// Users query and get a local memory consumption size in compile time.
    /// @return The size of local memory required.
    __XETLA_API static constexpr uint32_t get_slm_size() {
        constexpr uint32_t size = gemm_t::slm_size + epilogue_t::slm_size;
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
        //Linearize for stream_k algorithm
        return cl::sycl::range<3> {1, 1, local_range_m * local_range_n};
    };

    /// @brief Host helper function to get the expected nd_range under the current GEMM config.
    /// @return Expected nd_range.
    static cl::sycl::nd_range<3> get_nd_range(arguments_t &args) {
        cl::sycl::range<3> local_range = get_local_range();
        cl::sycl::range<3> group_range = args.stream_k_args.get_group_range();
        return cl::sycl::nd_range<3> {group_range * local_range, local_range};
    };

    /// @brief Host helper function to get the expected accumulation buffer size of the current STREAMK_GEMM_UNIVERSAL config.
    /// @param matrix_m Is the size of the m dimension of the matrix multiplication (m x k x n).
    /// @param matrix_n Is the size of the n dimension of the matrix multiplication (m x k x n).
    /// @return Expected accumulation buffer size in unit of elements.
    static size_t get_acc_buf_size(dispatch_stream_k &stream_k_args) {
        return stream_k_args.matrix_m * stream_k_args.matrix_n;
    };

    /// @brief Host helper function to get the expected counter buffer size of the current STREAMK_GEMM_UNIVERSAL config.
    /// @return Expected counter buffer size in unit of elements.
    static size_t get_cnt_buf_size(dispatch_stream_k &stream_k_args) {

        //For atomic reduction each SK group needs a synchronization flag
        uint32_t num_flags
                = stream_k_args.sk_regions * stream_k_args.sk_groups_per_region;
        const int barrier_size = sizeof(uint32_t);
        uint32_t atomic_space_bytes
                = cacheline_align_up(num_flags * barrier_size);
        uint32_t atomic_space_elements = atomic_space_bytes / barrier_size;

        return atomic_space_elements;
    };

    /// @brief Check if the arguments can be implemented.
    /// @param args Is the GEMM_UNIVERSAL arguments for application-related runtime variables.
    /// @return Check result.
    static bool can_implement(arguments_t &args) {
        bool implementable = true;
        if (gemm_t::msg_type_a != msg_type::unaligned_2d) {
            if (gemm_t::msg_type_a == msg_type::block_2d) {
                implementable
                        &= kernel::block_2d<arch_tag, dtype_a>::check_tensor(
                                (uint64_t)(args.matA_base.base),
                                gemm_t::is_col_major_a ? args.matrix_m
                                                       : args.matrix_k,
                                gemm_t::is_col_major_a ? args.matrix_k
                                                       : args.matrix_m,
                                args.matA_ld);
            } else {
                implementable &= kernel::general_1d<arch_tag,
                        dtype_a>::check_alignment(args.matA_base.base,
                        args.matA_ld);
            }
        }
        if (gemm_t::msg_type_b != msg_type::unaligned_2d) {
            if (gemm_t::msg_type_b == msg_type::block_2d) {
                implementable
                        &= kernel::block_2d<arch_tag, dtype_b>::check_tensor(
                                (uint64_t)(args.matB_base.base),
                                gemm_t::is_col_major_b ? args.matrix_k
                                                       : args.matrix_n,
                                gemm_t::is_col_major_b ? args.matrix_n
                                                       : args.matrix_k,
                                args.matB_ld);
            } else {
                implementable &= kernel::general_1d<arch_tag,
                        dtype_b>::check_alignment(args.matB_base.base,
                        args.matB_ld);
            }
        }
        if (epilogue_t::msg_type_c != msg_type::unaligned_2d) {
            if (epilogue_t::msg_type_c == msg_type::block_2d) {
                implementable
                        &= kernel::block_2d<arch_tag, dtype_c>::check_tensor(
                                (uint64_t)(args.matC_base.base), args.matrix_n,
                                args.matrix_m, args.matC_ld);
            } else {
                implementable &= kernel::general_1d<arch_tag,
                        dtype_c>::check_alignment(args.matC_base.base,
                        args.matC_ld);
            }
        }

        return implementable;
    }

protected:
    ///Tile work descriptor
    struct TileWorkDesc {

        ///location of this tile in group-tile coordinates in output matrix
        int tile_offset_m;
        int tile_offset_n;

        //The first global-scoped MAC-iteration this group will perform for this tile
        int iter_begin;

        //The starting index in the k-domain for MAC-iterations this group will perform for this tile
        int k_begin;

        //The end index in the k-domain for MAC-iterations this group will perform for this tile
        int k_end;

        //The number of remaining MAC-iterations this group will perform for this tile
        int k_iters_remaining;
    };

    __XETLA_API static void init_dp_tile_work(TileWorkDesc &tile_work,
            int tile_idx, int matrix_k, int iters_per_tile) {

        //The first global-scoped MAC-iteration this workgroup will perform for this tile
        tile_work.iter_begin = tile_idx * iters_per_tile;

        //The number of MAC-iterations this workgroup will perform
        tile_work.k_iters_remaining = iters_per_tile;

        //The starting index in the k-domain for MAC iterations this workgroup will perform for this tile
        tile_work.k_begin = 0;

        //The ending index (one-past) in the k-domain for MAC-iterations this workgroup will perform for this tile
        tile_work.k_end = matrix_k;
    }

    __XETLA_API static void init_sk_tile_work(TileWorkDesc &tile_work,
            int tile_idx, int group_iter_begin, int group_iter_end,
            int matrix_k, int iters_per_tile, uint32_t stride_k) {

        //The first global-scoped MAC iteration for this tile
        int tile_iter_begin = tile_idx * iters_per_tile;

        //The first global-scoped MAC-iteration this workgroup will perform for this tile
        tile_work.iter_begin = xetla_max(group_iter_begin, tile_iter_begin);

        //The first tile-scoped MAC-iteration this workgroup will perform for this tile
        int k_iter_begin = tile_work.iter_begin - tile_iter_begin;

        //The last(one past) tile-scoped MAC-iteration this workgroup will perform for this tile
        int k_iter_end = group_iter_end - tile_iter_begin;

        //The number of MAC-iterations this workgroup will perform for this tile
        tile_work.k_iters_remaining = k_iter_end - k_iter_begin;

        //Starting index in the k-domain for MAC-iterations this workgroup will perform for this tile
        tile_work.k_begin = k_iter_begin * k_stride;

        //Ending index (one past) in the k-domain for MAC-iterations this workgroup will perform for this tile
        tile_work.k_end = xetla_min(matrix_k, int(k_iter_end * k_stride));
    }

public:
    /// @brief Main execution function for stream_k GEMM.
    /// The processing order is 1) set group-level base and boundary -> 2) gemm -> 3) epilogue.
    /// @param item Is the sycl::nd_item, returns execution related information, such as workgroup id, subgroup id...
    /// @param args Is the GEMM arguments for application-related runtime variables.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    __XETLA_API KERNEL_FUNC void operator()(sycl::nd_item<3> &item,
            const arguments_t &args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        const dispatch_stream_k &workgroup_mapping = args.stream_k_args;
        int group_idx = item.get_group(2);

        int tile_idx = 0;
        int group_iter_begin = 0;
        int group_iters_remaining = 0;

        uint32_t gemm_slm_base = slm_base;
        uint32_t gemm_nbarr_base = nbarrier_base;
        uint32_t epilogue_slm_base = gemm_slm_base + gemm_t::slm_size;
        uint32_t epilogue_nbarr_base = gemm_nbarr_base + gemm_t::barrier_count;

        int iters_per_tile = workgroup_mapping.get_iters_per_tile();
        int dp_start_group_idx
                = workgroup_mapping.sk_waves * workgroup_mapping.avail_xecores;

        bool dp_group = (group_idx >= dp_start_group_idx);

        //setup for matatomic_sync / flag space
        mem_desc_atomic_sync_t mem_desc_atomic_sync;
        mem_desc_atomic_sync.init(args.matatomic_sync_base,
                {args.matatomic_sync_ld, 1, args.matatomic_sync_ld}, {0, 0});

        //Initialize tile-work descriptor
        TileWorkDesc tile_work;

        if (dp_group) {

            int dp_group_idx = group_idx - dp_start_group_idx;
            int first_dp_tile = workgroup_mapping.sk_tiles;

            tile_idx = first_dp_tile + dp_group_idx;

            group_iters_remaining = iters_per_tile;

            init_dp_tile_work(
                    tile_work, tile_idx, args.matrix_k, group_iters_remaining);

        } else {

            //This is a SK group
            int group_iter_end;
            workgroup_mapping.get_iter_extents(
                    group_idx, group_iter_begin, group_iter_end);
            group_iters_remaining = group_iter_end - group_iter_begin;

            tile_idx = workgroup_mapping.get_sk_tile_idx(group_iter_end - 1);
            init_sk_tile_work(tile_work, tile_idx, group_iter_begin,
                    group_iter_begin + group_iters_remaining, args.matrix_k,
                    iters_per_tile, k_stride);
        }

        //Tile offset in M and N
        workgroup_mapping.get_tile_offsets(
                tile_idx, tile_work.tile_offset_m, tile_work.tile_offset_n);

        epilogue_stream_k_t epilogue_stream_k;

        //StreamK processing loop body
        while (true) {
            ///Process current Tile
            {
                // set up workgroup level coordinates and boundaries
                int start_n = tile_work.tile_offset_n * wg_tile_n;
                int start_m = tile_work.tile_offset_m * wg_tile_m;
                int start_k = tile_work.k_begin;
                uint32_t boundary_n = (start_n + wg_tile_n) > args.matrix_n
                        ? args.matrix_n
                        : (start_n + wg_tile_n);
                uint32_t boundary_m = (start_m + wg_tile_m) > args.matrix_m
                        ? args.matrix_m
                        : (start_m + wg_tile_m);
                uint32_t boundary_k = tile_work.k_end;

                int first_group_idx = workgroup_mapping.get_first_group_idx(
                        tile_idx, group_idx);
                bool tile_finished = (tile_work.k_end == args.matrix_k);
                bool tile_started = (tile_work.k_begin == 0);

                // set up arguments
                work_group_t g;
                g.init(item.get_local_linear_id());
                mem_desc_a_t mem_desc_a;
                mem_desc_b_t mem_desc_b;
                mem_desc_c_t mem_desc_c;
                mem_desc_d_t mem_desc_d;

                //setup for matA
                if constexpr (mem_desc_a_t::is_local) {
                    mem_desc_a.init(args.matA_base,
                            {args.matrix_k, real_wg_tile_m, args.matrix_k},
                            {0, 0});
                } else {
                    mem_desc_a.init(args.matA_base,
                            {boundary_k, boundary_m, args.matA_ld},
                            {start_k, start_m});
                }

                //setup for matB
                if constexpr (mem_desc_b_t::is_local) {
                    mem_desc_b.init(args.matB_base,
                            {real_wg_tile_n, args.matrix_k, real_wg_tile_n},
                            {0, 0});
                } else {
                    mem_desc_b.init(args.matB_base,
                            {boundary_n, boundary_k, args.matB_ld},
                            {start_n, start_k});
                }
                //setup for matC
                if constexpr (mem_desc_c_t::is_local) {
                    mem_desc_c.init(args.matC_base,
                            {real_wg_tile_n, real_wg_tile_m, real_wg_tile_n},
                            {0, 0});
                } else {
                    mem_desc_c.init(args.matC_base,
                            {boundary_n, boundary_m, args.matC_ld},
                            {start_n, start_m});
                }

                //setup for scratchspace matD
                mem_desc_d.init(args.matD_base,
                        {boundary_n, boundary_m, args.matD_ld},
                        {start_n, start_m});

                matAcc_t matAcc(0);

                uint32_t inner_loop_count = tile_work.k_iters_remaining;

                gemm_args_t gemm_args(mem_desc_a, mem_desc_b, inner_loop_count);
                gemm_t gemm;

                gemm(g, matAcc, gemm_args, gemm_slm_base, gemm_nbarr_base);

                epilogue_stream_k(g, matAcc, mem_desc_c, mem_desc_d,
                        mem_desc_atomic_sync, group_idx, first_group_idx,
                        tile_finished, tile_started, args.epilogue_args,
                        epilogue_slm_base, epilogue_nbarr_base);
            }

            group_iters_remaining -= tile_work.k_iters_remaining;
            if (group_iters_remaining == 0) { break; }

            //Continue to next tile
            if (dp_group) {
                //DP groups consume their tiles at stride
                tile_idx += workgroup_mapping.avail_xecores;
                init_dp_tile_work(tile_work, tile_idx, args.matrix_k,
                        group_iters_remaining);
            } else {
                //SK groups consume their tiles in backwards order
                tile_idx--;
                init_sk_tile_work(tile_work, tile_idx, group_iter_begin,
                        group_iter_begin + group_iters_remaining, args.matrix_k,
                        iters_per_tile, k_stride);
            }
            //Tile offset in M and N
            workgroup_mapping.get_tile_offsets(
                    tile_idx, tile_work.tile_offset_m, tile_work.tile_offset_n);
        }
    }
};

/// @} xetla_gemm_universal

} // namespace gpu::xetla::kernel
