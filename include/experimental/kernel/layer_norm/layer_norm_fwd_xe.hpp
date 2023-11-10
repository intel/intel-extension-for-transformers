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

#include "experimental/group/fused_op/layer_norm_fused_op_fwd_xe.hpp"
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
/// @tparam store_for_bwd_
/// @tparam ln_fwd_fused_op_
template <typename dtype_x_, typename dtype_y_, typename dtype_weight_,
        typename dtype_acc_, typename layer_norm_attr_, bool store_for_bwd_,
        typename ln_fwd_fused_op_>
struct layer_norm_fwd_t<dtype_x_, dtype_y_, dtype_weight_, dtype_acc_,
        layer_norm_attr_, store_for_bwd_, gpu_arch::Xe, ln_fwd_fused_op_> {
    using dtype_x = dtype_x_;
    using dtype_y = dtype_y_;
    using dtype_weight = dtype_weight_;
    using dtype_acc = dtype_acc_;
    using layer_norm_attr = layer_norm_attr_;
    using ln_fwd_fused_op = ln_fwd_fused_op_;
    using ln_fused_op_arguments_t = typename ln_fwd_fused_op::arguments_t;
    static constexpr bool store_for_bwd = store_for_bwd_;

    static constexpr uint32_t wg_tile_m = layer_norm_attr::wg_tile_m;
    static constexpr uint32_t wg_tile_n = layer_norm_attr::wg_tile_n;
    static constexpr uint32_t sg_tile_m = layer_norm_attr::sg_tile_m;
    static constexpr uint32_t sg_tile_n = layer_norm_attr::sg_tile_n;
    static constexpr uint32_t wg_num_m = layer_norm_attr::wg_num_m;
    static constexpr uint32_t wg_num_n = layer_norm_attr::wg_num_n;
    static constexpr uint32_t chunk_size = layer_norm_attr::chunk_size;
    static constexpr uint32_t n_chunks = sg_tile_n / chunk_size;
    static_assert(sg_tile_n % chunk_size == 0,
            "Current impl does not support tailing mechanism");

    static constexpr uint32_t wg_size_x
            = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    static constexpr uint32_t wg_size_y
            = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
    using work_group_t = work_group_t<wg_size_x * wg_size_y>;
    static_assert((wg_size_x <= 32) && ((wg_size_x & (wg_size_x - 1)) == 0),
            "Current only support wg_size_x <=32");

    /// @brief
    ///
    struct get_barrier_count {
        static constexpr uint32_t count = (wg_size_x > 1) ? wg_size_y : 0;
    };

    /// @brief
    ///
    struct get_slm_size {
        // 4 = (mu + m2) * double buffering
        static constexpr uint32_t size = (wg_size_x > 1)
                ? wg_size_x * wg_size_y * 4 * sizeof(dtype_acc)
                : 0;
    };

    using ln_fwd_tile_desc_t = subgroup::tile_desc_t<chunk_size, 1, chunk_size,
            1, reg_layout::tiled>;
    using x_in_t = subgroup::tile_t<dtype_x, ln_fwd_tile_desc_t>;
    using gamma_in_t = subgroup::tile_t<dtype_weight, ln_fwd_tile_desc_t>;
    using beta_in_t = subgroup::tile_t<dtype_weight, ln_fwd_tile_desc_t>;
    using y_out_t = subgroup::tile_t<dtype_y, ln_fwd_tile_desc_t>;

    using x_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_x, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t,
            subgroup::msg_type_v<ln_fwd_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;
    using gamma_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_weight, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t,
            subgroup::msg_type_v<ln_fwd_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;
    using beta_in_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_weight, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t,
            subgroup::msg_type_v<ln_fwd_tile_desc_t, mem_space::global>,
            gpu_arch::Xe>;
    using y_out_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_y, mem_layout::row_major, mem_space::global>,
            ln_fwd_tile_desc_t, msg_type::block_1d, gpu_arch::Xe>;

    /// @brief
    ///
    struct arguments_t {
        dtype_x *x_in_ptr;
        dtype_weight *gamma_ptr;
        dtype_weight *beta_ptr;
        dtype_y *y_out_ptr;
        dtype_acc *rs_ptr;
        dtype_acc *mu_ptr;
        uint32_t matrix_m;
        uint32_t matrix_n;
        uint32_t mat_ld;
        dtype_acc epsilon = 1e-5;
    };

    /// @brief
    ///
    /// @tparam T
    /// @tparam SZ
    /// @tparam N
    template <typename T, uint32_t SZ, uint32_t N>
    struct parallel_mu_m2_t {
        static inline xetla_vector<T, 2> call(
                xetla_vector<T, SZ> mu_vec, xetla_vector<T, SZ> m2_vec) {
            auto mu_vec_a = mu_vec.xetla_select<SZ / 2, 1>(0);
            auto mu_vec_b = mu_vec.xetla_select<SZ / 2, 1>(SZ / 2);
            auto m2_vec_a = m2_vec.xetla_select<SZ / 2, 1>(0);
            auto m2_vec_b = m2_vec.xetla_select<SZ / 2, 1>(SZ / 2);
            xetla_vector<T, SZ / 2> mu_vec_new = (mu_vec_a + mu_vec_b) / (T)2;
            xetla_vector<T, SZ / 2> m2_vec_new = m2_vec_a + m2_vec_b
                    + (mu_vec_a - mu_vec_b) * (mu_vec_a - mu_vec_b) * (T)N
                            / (T)2;
            return parallel_mu_m2_t<T, SZ / 2, N * 2>::call(
                    mu_vec_new, m2_vec_new);
        }
    };

    /// @brief
    ///
    /// @tparam T
    /// @tparam N
    template <typename T, uint32_t N>
    struct parallel_mu_m2_t<T, 1, N> {

        /// @brief
        ///
        /// @param mu_vec
        /// @param m2_vec
        /// @return xetla_vector<T, 2>
        static inline xetla_vector<T, 2> call(
                xetla_vector<T, 1> mu_vec, xetla_vector<T, 1> m2_vec) {
            xetla_vector<T, 2> ret;
            ret[0] = mu_vec[0];
            ret[1] = m2_vec[0];
            return ret;
        }
    };

    /// @brief
    ///
    /// @param ei
    /// @param args
    /// @param slm_base
    /// @param nbarrier_base
    /// @param fused_op_args
    /// @return
    __XETLA_API static void call(sycl::nd_item<3> &item, arguments_t *args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0,
            ln_fused_op_arguments_t *fused_op_args = nullptr) {
        work_group_t g;
        g.init(item.get_local_linear_id());
        int sg_idx = g.get_id() % wg_size_x;
        int sg_idy = g.get_id() / wg_size_x;
        int wg_idx = item.get_group(2);
        int wg_idy = item.get_group(1);
        int start_n = wg_idx * wg_tile_n + sg_idx * sg_tile_n;
        int start_m = wg_idy * wg_tile_m + sg_idy * sg_tile_m;

        xetla_nbarrier_t<wg_size_x, wg_size_x, gpu_arch::Xe> nbarrier;
        nbarrier.init_nbarrier(
                sg_idy + nbarrier_base, nbarrier_role::producer_consumer);

        x_in_t x_in;
        x_in_payload_t x_in_payload;
        gamma_in_t gamma_in;
        gamma_in_payload_t gamma_in_payload;
        beta_in_t beta_in;
        beta_in_payload_t beta_in_payload;
        y_out_t y_out;
        y_out_payload_t y_out_payload;
        ln_fwd_fused_op fused_op;
        x_in_payload.init(args->x_in_ptr, args->matrix_n, args->matrix_m,
                args->mat_ld, start_n, start_m);
        // >>>>>>>>>> fused op fwd init

        if constexpr (n_chunks == 1) {
            fused_op.init(
                    fused_op_args, wg_idx, wg_idy, sg_idx, sg_idy, start_m);
            gamma_in_payload.init(args->gamma_ptr, args->matrix_n, 1,
                    args->mat_ld, start_n, 0);
            beta_in_payload.init(args->beta_ptr, args->matrix_n, 1,
                    args->mat_ld, start_n, 0);
            subgroup::tile_load(gamma_in, gamma_in_payload);
            subgroup::tile_load(beta_in, beta_in_payload);
        }
        y_out_payload.init(args->y_out_ptr, args->matrix_n, args->matrix_m,
                args->mat_ld, start_n, start_m);
        const dtype_acc sg_rn = 1.0f / sg_tile_n;
        const dtype_acc wg_rn = 1.0f / wg_tile_n;
        uint32_t slm_store_base_0 = sg_idx * 2 * sizeof(dtype_acc)
                + sg_idy * wg_size_x * 2 * sizeof(dtype_acc) + slm_base;
        uint32_t slm_load_base_0
                = sg_idy * wg_size_x * 2 * sizeof(dtype_acc) + slm_base;
        uint32_t slm_store_base_1 = slm_store_base_0
                + wg_size_x * wg_size_y * 2 * sizeof(dtype_acc);
        uint32_t slm_load_base_1 = slm_load_base_0
                + wg_size_x * wg_size_y * 2 * sizeof(dtype_acc);
        uint32_t itr_count = 0;

        for (int row = start_m; row < args->matrix_m;
                row += wg_num_m * wg_tile_m) {
            if constexpr (n_chunks > 1) {
                fused_op.init(
                        fused_op_args, wg_idx, wg_idy, sg_idx, sg_idy, row);
            }
            xetla_vector<dtype_acc, chunk_size> input;
            xetla_vector<dtype_acc, 2> mu_m2;
            mu_m2[0] = 0;
            mu_m2[1] = 0;
            if constexpr (n_chunks > 1) {
                x_in_payload.init(args->x_in_ptr, args->matrix_n,
                        args->matrix_m, args->mat_ld, start_n, row);
            }
#pragma unroll
            for (int i = 0; i < n_chunks; i++) {
                subgroup::tile_load(x_in, x_in_payload);
                x_in_payload.update_tdesc(chunk_size);
                input = xetla_cvt<dtype_acc, dtype_x>(x_in.reg);
                // >>>>>>>>>> fused op pre-processing
                input = fused_op.pre_op(input);
                // >>>>>>>>>> first do sg_level reduction
                mu_m2[0] += xetla_reduce<dtype_acc, dtype_acc, chunk_size,
                        reduce_op::sum>(input);
            }
            mu_m2[0] *= sg_rn;
            if constexpr (n_chunks > 1) {
                fused_op.init(
                        fused_op_args, wg_idx, wg_idy, sg_idx, sg_idy, row);
                x_in_payload.init(args->x_in_ptr, args->matrix_n,
                        args->matrix_m, args->mat_ld, start_n, row);
            }
#pragma unroll
            for (int i = 0; i < n_chunks; i++) {
                if constexpr (n_chunks > 1) {
                    subgroup::tile_load(x_in, x_in_payload);
                    x_in_payload.update_tdesc(chunk_size);
                    input = xetla_cvt<dtype_acc, dtype_x>(x_in.reg);
                    // >>>>>>>>>> fused op pre-processing
                    input = fused_op.pre_op(input);
                }

                xetla_vector<dtype_acc, chunk_size> diff
                        = input - dtype_acc(mu_m2[0]);
                mu_m2[1] += xetla_reduce<dtype_acc, dtype_acc, chunk_size,
                        reduce_op::sum>(diff * diff);
            }
            // >>>>>>>>>> then do wg_level reduction
            if constexpr (wg_size_x > 1) {
                uint32_t slm_store_base = (itr_count & 1) == 0
                        ? slm_store_base_0
                        : slm_store_base_1;
                xetla_store_local<dtype_acc, 2>(slm_store_base, mu_m2);
                xetla_fence<memory_kind::shared_local>();
                nbarrier.arrive();
                uint32_t slm_load_base = (itr_count & 1) == 0 ? slm_load_base_0
                                                              : slm_load_base_1;
                itr_count += 1;
                nbarrier.wait();

                xetla_vector<dtype_acc, wg_size_x * 2> mu_m2_vec
                        = xetla_load_local<dtype_acc, wg_size_x * 2>(
                                slm_load_base);
                xetla_vector<dtype_acc, wg_size_x> mu_vec
                        = mu_m2_vec.xetla_select<wg_size_x, 2>(0);
                xetla_vector<dtype_acc, wg_size_x> m2_vec
                        = mu_m2_vec.xetla_select<wg_size_x, 2>(1);
                mu_m2 = parallel_mu_m2_t<dtype_acc, wg_size_x, sg_tile_n>::call(
                        mu_vec, m2_vec);
            }
            dtype_acc mu = mu_m2[0];
            dtype_acc m2 = mu_m2[1];
            dtype_acc rs = xetla_rsqrt(m2 * wg_rn + args->epsilon);

            if constexpr (store_for_bwd) {
                if (sg_idx == 0) {
                    xetla_store_global<dtype_acc, 1, data_size::default_size,
                            cache_hint::write_back, cache_hint::write_back>(
                            args->mu_ptr, row * sizeof(dtype_acc),
                            xetla_vector<dtype_acc, 1>(mu));
                    xetla_store_global<dtype_acc, 1, data_size::default_size,
                            cache_hint::write_back, cache_hint::write_back>(
                            args->rs_ptr, row * sizeof(dtype_acc),
                            xetla_vector<dtype_acc, 1>(rs));
                }
            }
            // to generate mixed instruction
            constexpr uint32_t SIMD = 64 / sizeof(dtype_acc);
            if constexpr (chunk_size > 1) {
                gamma_in_payload.init(args->gamma_ptr, args->matrix_n, 1,
                        args->mat_ld, start_n, 0);
                beta_in_payload.init(args->beta_ptr, args->matrix_n, 1,
                        args->mat_ld, start_n, 0);
            }

            xetla_vector<dtype_acc, chunk_size> output;

            if constexpr (n_chunks > 1) {
                fused_op.init(
                        fused_op_args, wg_idx, wg_idy, sg_idx, sg_idy, row);
                x_in_payload.init(args->x_in_ptr, args->matrix_n,
                        args->matrix_m, args->mat_ld, start_n, row);
            }
            xetla_vector<dtype_acc, chunk_size> beta;
            xetla_vector<dtype_acc, chunk_size> gamma;
#pragma unroll
            for (int i = 0; i < n_chunks; i++) {
                if constexpr (n_chunks > 1) {
                    subgroup::tile_load(gamma_in, gamma_in_payload);
                    gamma_in_payload.update_tdesc(chunk_size);

                    subgroup::tile_load(beta_in, beta_in_payload);
                    beta_in_payload.update_tdesc(chunk_size);

                    subgroup::tile_load(x_in, x_in_payload);
                    x_in_payload.update_tdesc(chunk_size);
                    input = xetla_cvt<dtype_acc, dtype_x>(x_in.reg);
                    // >>>>>>>>>> fused op pre-processing
                    input = fused_op.pre_op(input);
                }
                xetla_vector<dtype_acc, chunk_size> beta
                        = xetla_cvt<dtype_acc, dtype_weight, chunk_size>(
                                beta_in.reg);
                xetla_vector<dtype_acc, chunk_size> gamma
                        = xetla_cvt<dtype_acc, dtype_weight>(gamma_in.reg);

                output = beta + (rs * (input - mu)) * gamma;
                // >>>>>>>>>> fused op post-processing
                output = fused_op.post_op(output);
                y_out.reg = xetla_cvt<dtype_y, dtype_acc, chunk_size>(output);
                subgroup::tile_store<cache_hint::uncached,
                        cache_hint::write_back>(y_out, y_out_payload);
                y_out_payload.update_tdesc(chunk_size);
            }
            x_in_payload.update_tdesc(
                    wg_num_m * wg_tile_m * args->mat_ld - sg_tile_n);
            y_out_payload.update_tdesc(
                    wg_num_m * wg_tile_m * args->mat_ld - sg_tile_n);
        }
    }
};

} // namespace gpu::xetla::kernel
