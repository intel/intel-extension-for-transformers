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

#include "group/brgemm/common.hpp"
#include "group/brgemm/compute_policy.hpp"

namespace gpu::xetla::group {
namespace detail {

template <typename dtype_a, typename dtype_b, typename dtype_mma_a,
        typename dtype_mma_b>
struct check_dtype_default_xmx_xe {
    static_assert(std::is_same<remove_const_t<dtype_mma_a>,
                          remove_const_t<dtype_mma_b>>::value,
            "dtype_mma_a should be the same as dtype_mma_b in xe arch ");
    static_assert((sizeof(dtype_mma_a) == sizeof(dtype_a))
                    || (sizeof(dtype_mma_a) == 2 * sizeof(dtype_a))
                    || (2 * sizeof(dtype_mma_a) == sizeof(dtype_a)),
            "Current we cannot support fp32 <->fp8, since it will meet a lot "
            "of HW limitations. ");
    static_assert((sizeof(dtype_mma_b) == sizeof(dtype_b))
                    || (sizeof(dtype_mma_b) == 2 * sizeof(dtype_b))
                    || (2 * sizeof(dtype_mma_b) == sizeof(dtype_b)),
            "Current we cannot support fp32 <->fp8, since it will meet a lot "
            "of HW limitations. ");
};

template <mem_layout mem_layout_a, mem_layout mem_layout_b,
        mem_space mem_space_a, mem_space mem_space_b>
struct check_memory_default_xmx_xe {
    static constexpr bool is_col_major_a
            = mem_layout_a == mem_layout::col_major;
    static constexpr bool is_col_major_b
            = mem_layout_b == mem_layout::col_major;
    static constexpr bool is_local_a = mem_space_a == mem_space::local;
    static constexpr bool is_local_b = mem_space_b == mem_space::local;
    static_assert(
            !is_local_b, "current don't support matB load from local memory");
    static_assert(!is_local_a || !is_col_major_a,
            "if matA load from local memory, then matA should be row-major");
};

template <typename arch_attr, typename dtype_mma, int tile_size_x_a,
        int tile_size_y_a, int block_size_x_a, int block_size_y_a,
        int tile_size_x_b, int tile_size_y_b, int block_size_x_b,
        int block_size_y_b>
struct check_tile_size_default_xmx_xe {
    using mma_attr = typename arch_attr::mma_attr;
    static constexpr int32_t mma_m = mma_attr::mma_m_in_elem;
    static constexpr int32_t mma_n = mma_attr::mma_n_in_elem;
    static constexpr int32_t mma_k
            = mma_attr::mma_k_in_bytes / sizeof(dtype_mma);

    static_assert(tile_size_x_a % mma_k == 0,
            "tile_size_x_a should be a multiple of mma_k");
    static_assert(
            block_size_x_a == mma_k, "block_size_x_a should be equal to mma_k");
    static_assert(tile_size_y_a % mma_m == 0,
            "tile_size_y_a should be a multiple of mma_m");
    static_assert(block_size_y_a % mma_m == 0,
            "block_size_y_a should be a multiple of mma_m");

    static_assert(tile_size_x_b % mma_n == 0,
            "tile_size_x_b should be a multiple of mma_n");
    static_assert(
            block_size_x_b == mma_n, "block_size_x_b should be equal to mma_n");
    static_assert(tile_size_y_b % mma_k == 0,
            "tile_size_y_b should be a multiple of mma_k");
    static_assert(block_size_y_b % mma_k == 0,
            "block_size_y_b should be a multiple of mma_k");
};

} // namespace detail

/// @addtogroup xetla_brgemm
/// @{

/// @brief Is the brgemm functor for Xe architecture and matrix engine.
template <typename compute_attr_, typename perf_tuning_knob_,
        typename tile_shape_, typename mem_desc_a_t_, typename mem_desc_b_t_,
        typename pre_processing_t_>
class brgemm_t<compute_policy_default_xmx<compute_attr_, perf_tuning_knob_,
                       gpu_arch::Xe>,
        tile_shape_, // tile shape of workgroup-level gemm
        mem_desc_a_t_, // memory attribute of matA
        mem_desc_b_t_, // memory attribute of matB
        pre_processing_t_ // pre_processing functor
        > {
public:
    using mem_desc_a_t = mem_desc_a_t_;
    using mem_desc_b_t = mem_desc_b_t_;
    using tile_shape = tile_shape_;
    using pre_processing_t = pre_processing_t_;
    using compute_policy = compute_policy_default_xmx<compute_attr_,
            perf_tuning_knob_, gpu_arch::Xe>;
    static constexpr uint32_t accum_step = compute_policy::accum_step;
    static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
    static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    using work_group_t = typename tile_shape::work_group_t;

    constexpr static gpu_arch arch_tag = compute_policy::arch_tag;

private:
    /******** set data type **********/
    using dtype_a = typename mem_desc_a_t::dtype;
    using dtype_b = typename mem_desc_b_t::dtype;
    using dtype_mma_acc = typename compute_policy::dtype_mma_acc;
    using dtype_mma_a = typename compute_policy::dtype_mma_a;
    using dtype_mma_b = typename compute_policy::dtype_mma_b;

    using check_dtype = detail::check_dtype_default_xmx_xe<dtype_a, dtype_b,
            dtype_mma_a, dtype_mma_b>;

    /******** set memory attribute **********/
    static constexpr mem_layout mem_layout_a = mem_desc_a_t::layout;
    static constexpr mem_layout mem_layout_b = mem_desc_b_t::layout;
    static constexpr mem_space mem_space_a = mem_desc_a_t::space;
    static constexpr mem_space mem_space_b = mem_desc_b_t::space;
    static constexpr bool is_col_major_a
            = mem_layout_a == mem_layout::col_major;
    static constexpr bool is_col_major_b
            = mem_layout_b == mem_layout::col_major;
    static constexpr bool is_local_a = mem_space_a == mem_space::local;
    static constexpr bool is_local_b = mem_space_b == mem_space::local;
    static constexpr tdesc_update_dir update_dir_a = is_col_major_a
            ? tdesc_update_dir::y_dir
            : tdesc_update_dir::x_dir;
    static constexpr tdesc_update_dir update_dir_b = is_col_major_b
            ? tdesc_update_dir::x_dir
            : tdesc_update_dir::y_dir;

    using check_memory = detail::check_memory_default_xmx_xe<mem_layout_a,
            mem_layout_b, mem_space_a, mem_space_b>;

    static constexpr uint32_t stages = compute_policy::stages;
    static constexpr uint32_t sync_freq = compute_policy::sync_freq;

    /******** set tile layout && worker scope **********/
    static constexpr uint32_t tile_size_x_a = accum_step;
    static constexpr uint32_t tile_size_y_a = sg_tile_m;
    static constexpr uint32_t tile_size_x_b = sg_tile_n;
    static constexpr uint32_t tile_size_y_b = accum_step;
    static constexpr uint32_t tile_size_x_c = sg_tile_n;
    static constexpr uint32_t tile_size_y_c = sg_tile_m;
    static constexpr uint32_t block_size_x_a
            = compute_policy::block_bytes_x_a / sizeof(dtype_mma_a);
    static constexpr uint32_t block_size_y_a
            = (compute_policy::block_size_y_a > tile_size_y_a)
            ? tile_size_y_a
            : compute_policy::block_size_y_a;
    static constexpr uint32_t block_size_x_b = compute_policy::block_size_x_b;
    static constexpr uint32_t block_size_y_b
            = compute_policy::block_bytes_y_b / sizeof(dtype_mma_b);

    using arch_attr = arch_attr_t<arch_tag>;
    using check_tile_size = detail::check_tile_size_default_xmx_xe<arch_attr,
            dtype_mma_a, tile_size_x_a, tile_size_y_a, block_size_x_a,
            block_size_y_a, tile_size_x_b, tile_size_y_b, block_size_x_b,
            block_size_y_b>;

    /******** set tile  **********/
    static constexpr bool is_vnni_tiled_a
            = (sizeof(dtype_a) < sizeof(uint32_t)) && is_col_major_a;
    static constexpr reg_layout reg_layout_a
            = is_vnni_tiled_a ? reg_layout::vnni_tiled : reg_layout::tiled;
    using matA_tile_desc_t = subgroup::tile_desc_t<tile_size_x_a, tile_size_y_a,
            block_size_x_a, block_size_y_a, reg_layout_a>;
    using matA_t = subgroup::tile_t<dtype_a, matA_tile_desc_t>;
    using matA_payload_t = subgroup::mem_payload_t<dtype_a, matA_tile_desc_t,
            subgroup::msg_type_v<matA_tile_desc_t, mem_space_a>, mem_layout_a,
            mem_space_a, arch_tag>;
    using matA_acc_t = subgroup::tile_t<dtype_mma_a, matA_tile_desc_t>;
    using matA_prefetch_payload_t = subgroup::prefetch_payload_t<dtype_a,
            subgroup::tile_desc_t<tile_size_x_a, tile_size_y_a, 1, 1>,
            mem_layout_a, mem_space_a, wg_size_x, arch_tag>;
    static constexpr reg_layout reg_layout_b
            = sizeof(dtype_b) < sizeof(uint32_t) ? reg_layout::vnni_tiled
                                                 : reg_layout::tiled;
    using matB_tile_desc_t = subgroup::tile_desc_t<tile_size_x_b, tile_size_y_b,
            block_size_x_b, block_size_y_b, reg_layout_b>;
    using matB_t = subgroup::tile_t<dtype_b, matB_tile_desc_t>;
    using matB_payload_t = subgroup::mem_payload_t<dtype_b, matB_tile_desc_t,
            subgroup::msg_type_v<matB_tile_desc_t, mem_space_b>, mem_layout_b,
            mem_space_b, arch_tag>;
    using matB_acc_t = subgroup::tile_t<dtype_mma_b, matB_tile_desc_t>;
    using matB_prefetch_payload_t = subgroup::prefetch_payload_t<dtype_b,
            subgroup::tile_desc_t<tile_size_x_b, tile_size_y_b, 1, 1>,
            mem_layout_b, mem_space_b, wg_size_y, arch_tag>;

public:
    using matAcc_tile_desc_t = subgroup::tile_desc_t<tile_size_x_c,
            tile_size_y_c, block_size_x_b, block_size_y_a, reg_layout::tiled>;
    using matAcc_t = subgroup::tile_t<dtype_mma_acc, matAcc_tile_desc_t>;

private:
    using tile_mma = subgroup::tile_mma_t<matA_acc_t, matB_acc_t, matAcc_t,
            matAcc_t, mma_engine::xmx, arch_tag>;
    static constexpr bool enable_periodic_sync = (sync_freq != 0);
    static constexpr uint32_t barrier_count_x = wg_size_y > 1 ? wg_size_x : 0;
    static constexpr uint32_t barrier_count_y = wg_size_x > 1 ? wg_size_y : 0;

public:
    static constexpr uint32_t barrier_count
            = enable_periodic_sync ? barrier_count_x + barrier_count_y : 0;
    // current only support matA from slm
    static constexpr uint32_t slm_size
            = is_local_a ? wg_tile_m * accum_step * sizeof(dtype_a) : 0;
    using pre_processing_arg_t = typename pre_processing_t::arguments_t;

    /// @brief Arguments for brgemm.
    /// User should prepare matA_base_desc, matB_base_desc, inner_loop_count...
    struct arguments_t {
        /// @brief Is the memory description of matA, including base, shape and coordinate.
        mem_desc_a_t matA_base_desc;
        /// @brief Is the memory description of matB, including base, shape and coordinate.
        mem_desc_b_t matB_base_desc;
        /// @brief Is the total inner loop count required to compute the entire K-dim.
        uint32_t inner_loop_count;
        /// @brief Is the arguments for pre-processing functor.
        pre_processing_arg_t pre_processing_args;

        /// @brief Default construct.
        inline arguments_t() = default;

        /// @brief Constructs a new arguments t object.
        /// @param matA_desc Is the memory description of matA, including base, shape and coordinate.
        /// @param matB_desc Is the memory description of matB, including base, shape and coordinate.
        /// @param loop_count Is the total inner loop count required to compute the entire K-dim.
        /// @param args Is the arguments for pre-processing functor.
        inline arguments_t(mem_desc_a_t matA_desc, mem_desc_b_t matB_desc,
                uint32_t loop_count, pre_processing_arg_t args = {})
            : matA_base_desc(matA_desc)
            , matB_base_desc(matB_desc)
            , inner_loop_count(loop_count)
            , pre_processing_args(args) {}
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // inline ~arguments_t(){}
        inline arguments_t(const arguments_t &args)
            : matA_base_desc(args.matA_base_desc)
            , matB_base_desc(args.matB_base_desc)
            , inner_loop_count(args.inner_loop_count)
            , pre_processing_args(args.pre_processing_args) {}
        inline arguments_t &operator=(const arguments_t &args) {
            this->matA_base_desc = args.matA_base_desc;
            this->matB_base_desc = args.matB_base_desc;
            this->inner_loop_count = args.inner_loop_count;
            this->pre_processing_args = args.pre_processing_args;
            return *this;
        }

        /// @brief Explicit initialization function.
        /// @param matA_desc Is the memory description of matA, including base, shape and coordinate.
        /// @param matB_desc Is the memory description of matB, including base, shape and coordinate.
        /// @param loop_count Is the total inner loop count required to compute the entire K-dim.
        /// @param args Is the arguments for pre-processing functor.
        inline void init(mem_desc_a_t matA_desc, mem_desc_b_t matB_desc,
                uint32_t loop_count, pre_processing_arg_t args = {}) {
            matA_base_desc = matA_desc;
            matB_base_desc = matB_desc;
            inner_loop_count = loop_count;
            pre_processing_args = args;
        }
    };

    /// @brief Gets the subgroup-level tile offset x.
    /// @param g Is the workgroup of the current tile.
    /// @return Subgroup-level tile offset x.
    __XETLA_API static int get_matC_offset_x(work_group_t &g) {
        int32_t sg_idx = g.get_id() % wg_size_x;
        return sg_idx * sg_tile_n;
    }

    /// @brief Gets the subgroup-level tile offset y.
    /// @param g Is the workgroup of the current tile.
    /// @return Subgroup-level tile offset y.
    __XETLA_API static int get_matC_offset_y(work_group_t &g) {
        int32_t sg_idy = g.get_id() / wg_size_x;
        return sg_idy * sg_tile_m;
    }

    XETLA_MARKER(
            "This release function will wait until all the  r/w and nbarrier "
            "id used in this brgemm have been committed. By default, it will "
            "use barrier_id 0 to do the entire workgroup sync if wg_size > 1. "
            "If you call this function, please set a free barrier id or make "
            "sure barrier_id 0 is not being occupied and you need to allocate "
            "one more barrier count in addition to the brgemm barrier counts.")
    __XETLA_API static void release(uint8_t nbarrier_id = 0) {
        static constexpr bool need_local_fence
                = (mem_space_a == mem_space::local)
                || (mem_space_b == mem_space::local);
        if constexpr (need_local_fence) {
            xetla_fence<memory_kind::shared_local>();
        }
        xetla_fence<memory_kind::untyped_global>();
        static constexpr uint32_t wg_size = wg_size_x * wg_size_y;
        if constexpr (wg_size > 1) {
            xetla_nbarrier_t<wg_size, wg_size> nbarrier;
            nbarrier.init_nbarrier(
                    nbarrier_id, nbarrier_role::producer_consumer);
            nbarrier.arrive_wait();
        }
    }

    /// @brief Main execution function for brgemm.
    /// The basic process is load data -> matrix multiply.
    /// @param g Is the workgroup of the current tile.
    /// @param matAcc Is the reference of the accumulation buffer.
    /// @param args Is the brgemm::arguments_t.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g, matAcc_t &matAcc,
            arguments_t args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        int32_t sg_idx = g.get_id() % wg_size_x;
        int32_t sg_idy = g.get_id() / wg_size_x;
        update_sg_tile_tdesc(args, sg_idx, sg_idy);
        pre_processing_t pre_processing;
        matA_t matA;
        matB_t matB;
        //  >>>>>>>>>>>>>>>>>> pre_processing init
        pre_processing.init(g, args.pre_processing_args);
        matA_payload_t matA_payload(args.matA_base_desc);
        matB_payload_t matB_payload(args.matB_base_desc);
        matA_prefetch_payload_t matA_prefetch_payload(
                args.matA_base_desc, sg_idx);
        matB_prefetch_payload_t matB_prefetch_payload(
                args.matB_base_desc, sg_idy);
        xetla_nbarrier_t<wg_size_x, wg_size_x> nbarrier_a;
        nbarrier_a.init_nbarrier(
                sg_idy + nbarrier_base, nbarrier_role::producer_consumer);
        xetla_nbarrier_t<wg_size_y, wg_size_y> nbarrier_b;
        nbarrier_b.init_nbarrier(sg_idx + barrier_count_y + nbarrier_base,
                nbarrier_role::producer_consumer);

#pragma unroll
        for (int i = 0; i < stages; i++) {
            subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
                    matA_prefetch_payload);
            subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
                    matB_prefetch_payload);
            matA_prefetch_payload.template update_tdesc<update_dir_a>(
                    matA_t::tile_size_x);
            matB_prefetch_payload.template update_tdesc<update_dir_b>(
                    matB_t::tile_size_y);
        }

        for (int i = 0; i < args.inner_loop_count; i++) {
            if constexpr (enable_periodic_sync) {
                if ((i % sync_freq) == 0) {
                    if constexpr (wg_size_x > 1) { nbarrier_a.arrive(); }
                    if constexpr (wg_size_y > 1) { nbarrier_b.arrive(); }
                }
            }
            subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                    matA, matA_payload);
            subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                    matB, matB_payload);
            SW_BARRIER();
            if constexpr (stages != 0) {
                subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
                        matA_prefetch_payload);
                subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
                        matB_prefetch_payload);
            }
            SW_BARRIER();
            matA_payload.template update_tdesc<update_dir_a>(
                    matA_t::tile_size_x);
            matB_payload.template update_tdesc<update_dir_b>(
                    matB_t::tile_size_y);
            if constexpr (stages != 0) {
                matA_prefetch_payload.template update_tdesc<update_dir_a>(
                        matA_t::tile_size_x);
                matB_prefetch_payload.template update_tdesc<update_dir_b>(
                        matB_t::tile_size_y);
            }
            SW_BARRIER();
            matA_acc_t matA_acc;
            matB_acc_t matB_acc;
            if constexpr (is_vnni_tiled_a) { subgroup::vnni_reverse(matA); }
            pre_processing(matA_acc, matB_acc, matA, matB);
            SW_BARRIER();
            tile_mma::mma(matA_acc, matB_acc, matAcc, matAcc);
            SW_BARRIER();
            if constexpr (enable_periodic_sync) {
                if ((i % sync_freq) == 0) {
                    if constexpr (wg_size_x > 1) { nbarrier_a.wait(); }
                    if constexpr (wg_size_y > 1) { nbarrier_b.wait(); }
                }
            }
        }
        SW_BARRIER();
    }

private:
    /// @brief Updates tile base descriptor based on the tid.
    __XETLA_API static void update_sg_tile_tdesc(
            arguments_t &args, int32_t sg_idx, int32_t sg_idy) {
        int32_t tile_offset_n = sg_idx * sg_tile_n;
        int32_t tile_offset_m = sg_idy * sg_tile_m;

        args.matA_base_desc.update_coord_y(tile_offset_m);
        args.matB_base_desc.update_coord_x(tile_offset_n);
    }
};

/// @} xetla_brgemm

} // namespace gpu::xetla::group