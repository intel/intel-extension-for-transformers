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

#include "experimental/group/fused_op/layer_norm_fused_op_bwd_xe.hpp"
#include "experimental/kernel/layer_norm/api.hpp"
#include "experimental/kernel/layer_norm/common.hpp"
#include "experimental/kernel/layer_norm/config.hpp"

namespace gpu::xetla::kernel {

/// @brief
///
/// @tparam dtype_x_
/// @tparam dtype_y_
/// @tparam dtype_weight_
/// @tparam dtype_acc_
/// @tparam layer_norm_attr_
/// @tparam ln_bwd_fused_op_
template <typename dtype_x_, typename dtype_y_, typename dtype_weight_,
        typename dtype_acc_, typename layer_norm_attr_,
        typename ln_bwd_fused_op_>
struct layer_norm_bwd_t<dtype_x_, dtype_y_, dtype_weight_, dtype_acc_,
        layer_norm_attr_, gpu_arch::Xe, ln_bwd_fused_op_> {
    using dtype_x = dtype_x_;
    using dtype_y = dtype_y_;
    using dtype_weight = dtype_weight_;
    using dtype_acc = dtype_acc_;
    using layer_norm_attr = layer_norm_attr_;
    using ln_bwd_fused_op = ln_bwd_fused_op_;
    using ln_fused_op_arguments_t = typename ln_bwd_fused_op::arguments_t;
    static constexpr uint32_t wg_tile_m = layer_norm_attr::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr::wg_num_n;

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
    using work_group_t = work_group_t<wg_size_x * wg_size_y>;
    static_assert((wg_size_x <= 32) && ((wg_size_x & (wg_size_x - 1)) == 0),
            "Current only support wg_size_x <=32");
    static constexpr uint32_t count_col_reduce
            = (wg_size_x > 1) ? wg_size_y : 0;
    static constexpr uint32_t count_row_reduce
            = (wg_size_y > 1) ? wg_size_x : 0;
    struct get_barrier_count {
        static constexpr uint32_t count = count_col_reduce + count_row_reduce;
    };
    // 4 = (grad0 + grad1) * double buffering
    static constexpr uint32_t size_col_reduce = (wg_size_x > 1)
            ? wg_size_x * wg_size_y * 4 * sizeof(dtype_acc)
            : 0;
    // wg_size_y * rows
    static constexpr uint32_t size_row_reduce = (wg_size_y > 1)
            ? wg_size_y * wg_size_x * sg_tile_n * sizeof(dtype_acc)
            : 0;
    struct get_slm_size {
        static constexpr uint32_t size = size_col_reduce + size_row_reduce;
    };

    using ln_bwd_tile_desc_t = subgroup::tile_desc_t<sg_tile_n, 1, sg_tile_n, 1,
            reg_layout::tiled>;
    using dy_in_t = subgroup::tile_t<dtype_y, ln_bwd_tile_desc_t>;
    using x_in_t = subgroup::tile_t<dtype_x, ln_bwd_tile_desc_t>;
    using gamma_in_t = subgroup::tile_t<dtype_weight, ln_bwd_tile_desc_t>;
    using dx_out_t = subgroup::tile_t<dtype_x, ln_bwd_tile_desc_t>;

    using dy_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_y, mem_layout::row_major, mem_space::global>,
            ln_bwd_tile_desc_t,
            subgroup::msg_type_v<ln_bwd_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;
    using x_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_x, mem_layout::row_major, mem_space::global>,
            ln_bwd_tile_desc_t,
            subgroup::msg_type_v<ln_bwd_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;
    using gamma_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_weight, mem_layout::row_major, mem_space::global>,
            ln_bwd_tile_desc_t,
            subgroup::msg_type_v<ln_bwd_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;
    using dx_out_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_x, mem_layout::row_major, mem_space::global>,
            ln_bwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;

    using ln_group_row_reduce_store_t
            = group::group_row_reduce_store_t<dtype_acc, dtype_acc, sg_tile_n,
                    wg_size_x, wg_size_y, 32, gpu_arch::Xe>;

    /// @brief
    ///
    struct arguments_t {
        dtype_y *dy_in_ptr;
        dtype_x *x_in_ptr;
        dtype_weight *gamma_in_ptr;
        dtype_acc *rs_ptr;
        dtype_acc *mu_ptr;

        dtype_x *dx_out_ptr;
        dtype_acc *dgamma_acc_ptr;
        dtype_acc *dbeta_acc_ptr;

        uint32_t matrix_m;
        uint32_t matrix_n;
        uint32_t mat_ld;
        dtype_acc epsilon = 1e-5;
    };

private:
    /// @brief This is the group all reduce for layer norm bwd. Use slm double buffering to exchange the data.
    ///
    /// @tparam T Is the data type to do the reduction
    /// @tparam SZ Is the vector size per item
    /// @tparam N Is the number of independent items for one subgroup to do the parallel all-reduction
    /// @tparam Op Is the reduction op
    /// @tparam wg_size_x Is the wg size in x direction, i.e. is the number of threads that participate in this reduction.
    /// @tparam wg_size_y Is the wg size in y direction, is the number of parallel reductions in the wg(use to calculated the slm address).
    /// @tparam arch_
    template <typename T, uint32_t SZ, uint32_t N, reduce_op Op,
            uint32_t wg_size_x, uint32_t wg_size_y,
            gpu_arch arch_ = gpu_arch::Xe>
    struct ln_group_all_reduce_t {
        uint32_t itr_count;
        uint32_t slm_base_0;
        uint32_t slm_base_1;

        group::group_reduce_t<T, SZ, N, Op, wg_size_x, true, arch_>
                group_reduce;

        /// @brief Construct a new ln group all reduce t object
        ///
        /// @param sg_idx
        /// @param sg_idy
        /// @param slm_base
        /// @param nbarrier_base
        inline ln_group_all_reduce_t(uint32_t sg_idx = 0, uint32_t sg_idy = 0,
                uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
            slm_base_0 = slm_base + sg_idy * wg_size_x * N * sizeof(T);
            slm_base_1 = slm_base_0 + wg_size_x * wg_size_y * N * sizeof(T);
            itr_count = 0;
            group_reduce.init(sg_idx, sg_idy + nbarrier_base, slm_base_0);
        }

        /// @brief
        ///
        /// @param buffer
        /// @return KERNEL_FUNC
        inline KERNEL_FUNC xetla_vector<T, N> operator()(
                xetla_vector<T, N * SZ> buffer) {
            uint32_t slm_base = (itr_count & 1) ? slm_base_1 : slm_base_0;
            group_reduce.set_slm_base(slm_base);
            xetla_vector<T, N> ret = group_reduce(buffer);
            itr_count += 1;
            return ret;
        }
    };

    /// @brief Get the x temp object
    ///
    /// @tparam sizeof(dtype_acc)
    /// @param x
    /// @param rs
    /// @param mu
    /// @return
    template <uint32_t SIMD = 64 / sizeof(dtype_acc)>
    __XETLA_API static xetla_vector<dtype_acc, sg_tile_n> get_x_temp(
            xetla_vector<dtype_x, sg_tile_n> x, dtype_acc rs, dtype_acc mu) {
        xetla_vector<dtype_acc, sg_tile_n> x_temp;
        xetla_vector<dtype_acc, sg_tile_n> x_acc
                = xetla_cvt<dtype_acc, dtype_x>(x);
        /// to generate mixed instruction
#pragma unroll
        for (int i = 0; i < sg_tile_n / SIMD; i++) {
            x_temp.xetla_select<SIMD, 1>(i * SIMD)
                    = rs * (x_acc.xetla_select<SIMD, 1>(i * SIMD) - mu);
        }
        if constexpr ((sg_tile_n % SIMD) != 0) {
            constexpr uint32_t start = sg_tile_n / SIMD * SIMD;
            constexpr uint32_t SIMD_tail = sg_tile_n % SIMD;
            x_temp.xetla_select<SIMD_tail, 1>(start)
                    = rs * (x_acc.xetla_select<SIMD_tail, 1>(start) - mu);
        }
        return x_temp;
    }

    /// @brief Get the dy temp object
    ///
    /// @tparam sizeof(dtype_acc)
    /// @param gamma
    /// @param dy
    /// @return
    template <uint32_t SIMD = 64 / sizeof(dtype_acc)>
    __XETLA_API static xetla_vector<dtype_acc, sg_tile_n> get_dy_temp(
            xetla_vector<dtype_weight, sg_tile_n> gamma,
            xetla_vector<dtype_acc, sg_tile_n> dy) {
        xetla_vector<dtype_acc, sg_tile_n> dy_temp;
        xetla_vector<dtype_acc, sg_tile_n> gamma_acc
                = xetla_cvt<dtype_acc, dtype_weight>(gamma);
        /// to generate mixed instruction
#pragma unroll
        for (int i = 0; i < sg_tile_n / SIMD; i++) {
            dy_temp.xetla_select<SIMD, 1>(i * SIMD)
                    = gamma_acc.xetla_select<SIMD, 1>(i * SIMD)
                    * dy.xetla_select<SIMD, 1>(i * SIMD);
        }
        if constexpr ((sg_tile_n % SIMD) != 0) {
            constexpr uint32_t start = sg_tile_n / SIMD * SIMD;
            constexpr uint32_t SIMD_tail = sg_tile_n % SIMD;
            dy_temp.xetla_select<SIMD_tail, 1>(start)
                    = gamma_acc.xetla_select<SIMD_tail, 1>(start)
                    * dy.xetla_select<SIMD_tail, 1>(start);
        }
        return dy_temp;
    }
    using wg_col_reduce_t = ln_group_all_reduce_t<dtype_acc, sg_tile_n, 2,
            reduce_op::sum, wg_size_x, wg_size_y, gpu_arch::Xe>;

public:
    __XETLA_API static void call(sycl::nd_item<3> &item, arguments_t *args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0,
            ln_fused_op_arguments_t *fused_op_args = nullptr) {
        work_group_t g;
        g.init(item.get_local_linear_id());
        uint32_t sg_idx = g.get_id() % wg_size_x;
        uint32_t sg_idy = g.get_id() / wg_size_x;
        uint32_t wg_idx = item.get_group(2);
        uint32_t wg_idy = item.get_group(1);
        int start_n = wg_idx * wg_tile_n + sg_idx * sg_tile_n;
        int start_m = wg_idy * wg_tile_m + sg_idy * sg_tile_m;

        x_in_t x_in;
        x_in_payload_t x_in_payload;
        dy_in_t dy_in;
        dy_in_payload_t dy_in_payload;
        gamma_in_t gamma_in;
        gamma_in_payload_t gamma_in_payload;
        dx_out_t dx_out;
        dx_out_payload_t dx_out_payload;
        ln_bwd_fused_op fused_op;
        x_in_payload.init(args->x_in_ptr, args->matrix_n, args->matrix_m,
                args->mat_ld, start_n, start_m);
        dy_in_payload.init(args->dy_in_ptr, args->matrix_n, args->matrix_m,
                args->mat_ld, start_n, start_m);
        gamma_in_payload.init(args->gamma_in_ptr, args->matrix_n, 1,
                args->mat_ld, start_n, 0);
        dx_out_payload.init(args->dx_out_ptr, args->matrix_n, args->matrix_m,
                args->mat_ld, start_n, start_m);
        fused_op.init(fused_op_args, wg_idx, wg_idy, sg_idx, sg_idy);
        subgroup::tile_load(gamma_in, gamma_in_payload);

        const dtype_acc wg_rn = 1.0f / wg_tile_n;

        wg_col_reduce_t wg_col_reduce(sg_idx, sg_idy, slm_base, nbarrier_base);

        xetla_vector<dtype_acc, sg_tile_n> dgamma = 0;
        xetla_vector<dtype_acc, sg_tile_n> dbeta = 0;

        for (int row = start_m; row < args->matrix_m;
                row += wg_num_m * wg_tile_m) {
            subgroup::tile_load(dy_in, dy_in_payload);
            subgroup::tile_load(x_in, x_in_payload);
            xetla_vector<dtype_acc, 1> mu_v
                    = xetla_load_global<dtype_acc, 1, data_size::default_size,
                            cache_hint::cached, cache_hint::cached>(
                            args->mu_ptr, row * sizeof(dtype_acc));
            xetla_vector<dtype_acc, 1> rs_v
                    = xetla_load_global<dtype_acc, 1, data_size::default_size,
                            cache_hint::cached, cache_hint::cached>(
                            args->rs_ptr, row * sizeof(dtype_acc));
            dtype_acc mu = mu_v[0];
            dtype_acc rs = rs_v[0];
            dy_in_payload.update_tdesc(wg_num_m * wg_tile_m * args->mat_ld);
            x_in_payload.update_tdesc(wg_num_m * wg_tile_m * args->mat_ld);

            xetla_vector<dtype_acc, sg_tile_n> dy
                    = xetla_cvt<dtype_acc, dtype_y>(dy_in.reg);
            dy = fused_op.pre_op(dy);
            xetla_vector<dtype_acc, sg_tile_n> x_temp
                    = get_x_temp(x_in.reg, rs, mu);
            xetla_vector<dtype_acc, sg_tile_n> dy_temp
                    = get_dy_temp(gamma_in.reg, dy);
            dgamma += dy * x_temp;
            dbeta += dy;
            xetla_vector<dtype_acc, sg_tile_n * 2> buffer;
            auto buffer_2d = buffer.xetla_format<dtype_acc, 2, sg_tile_n>();
            buffer_2d.row(0) = dy_temp;
            buffer_2d.row(1) = x_temp * dy_temp;
            xetla_vector<dtype_acc, 2> grad_0_1 = wg_col_reduce(buffer);
            dtype_acc grad_0 = grad_0_1[0] * wg_rn;
            dtype_acc grad_1 = grad_0_1[1] * wg_rn;
            xetla_vector<dtype_acc, sg_tile_n> dx
                    = rs * (dy_temp - (grad_1 * x_temp + grad_0));
            dx = fused_op.post_op(dx);
            dx_out.reg = xetla_cvt<dtype_x, dtype_acc>(dx);
            subgroup::tile_store<cache_hint::uncached>(dx_out, dx_out_payload);
            dx_out_payload.update_tdesc(wg_num_m * wg_tile_m * args->mat_ld);
        }

        ln_group_row_reduce_store_t ln_group_row_reduce;
        uint32_t slm_row_reduce_base = slm_base + size_col_reduce;
        uint32_t nbarrier_row_reduce_base = nbarrier_base + count_col_reduce;
        ln_group_row_reduce.init(
                sg_idx, sg_idy, slm_row_reduce_base, nbarrier_row_reduce_base);
        /// temp buffers don't need to set different width and pitch
        ln_group_row_reduce(args->dgamma_acc_ptr, args->matrix_n, wg_num_m,
                args->matrix_n, start_n, wg_idy, dgamma);
        ln_group_row_reduce(args->dbeta_acc_ptr, args->matrix_n, wg_num_m,
                args->matrix_n, start_n, wg_idy, dbeta);
        fused_op.final_op(ln_group_row_reduce);
    }
};

} // namespace gpu::xetla::kernel
