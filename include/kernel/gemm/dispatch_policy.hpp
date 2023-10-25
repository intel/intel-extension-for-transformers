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

#include "kernel/gemm/common.hpp"

namespace gpu::xetla::kernel {
/// @addtogroup xetla_gemm_universal
/// @{

/// @brief Default GROUP_SWIZZLE implementation.
/// A general GROUP_SWIZZLE implementation to get an workgroup id .
/// @tparam arch_tag_ Is the HW architecture.
template <gpu_arch arch_tag_ = gpu_arch::Xe>
struct group_swizzle_default {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;

    template <int>
    int get_tile_idx(sycl::nd_item<3> &item);
    // get dim0 group id
    template <>
    __XETLA_API int get_tile_idx<0>(sycl::nd_item<3> &item) {
        return item.get_group(0);
    }
    // get dim1 group id
    template <>
    __XETLA_API int get_tile_idx<1>(sycl::nd_item<3> &item) {
        return item.get_group(1);
    }
    // get dim2 group id
    template <>
    __XETLA_API int get_tile_idx<2>(sycl::nd_item<3> &item) {
        return item.get_group(2);
    }
    // correct group range, nothing will be done under this swizzle policy
    static __XETLA_API void update_group_range(
            uint32_t &group_range_m, uint32_t &group_range_n) {}
};

/// @brief GROUP_SWIZZLE implementation of snake curve.
/// A GROUP_SWIZZLE implementation to remap linear workgroup id to a 2d coordination in snake order.
/// @tparam wg_num_n_ Is the number of workgroup in horizontal direction, given by users.
/// @tparam arch_tag_ Is the HW architecture.
template <int wg_num_n_, gpu_arch arch_tag_ = gpu_arch::Xe>
struct group_swizzle_snake {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;

    template <int>
    int get_tile_idx(sycl::nd_item<3> &item);
    // get dim0 group id
    template <>
    __XETLA_API int get_tile_idx<0>(sycl::nd_item<3> &item) {
        return item.get_group(0);
    }
    // get transformed dim1 group id
    template <>
    __XETLA_API int get_tile_idx<1>(sycl::nd_item<3> &item) {
        uint32_t group_range_n = item.get_group_range(2);
        uint32_t wg_repeat_n = group_range_n / wg_num_n;
        uint32_t repeat_id = get_2d_group_linear_id(item) / max_wg_num;
        uint32_t repeat_id_m = repeat_id / wg_repeat_n;
        uint32_t repeat_start_m = repeat_id_m * wg_num_m;
        uint32_t wg_inner_id = get_2d_group_linear_id(item) % max_wg_num;
        uint32_t wg_coord_m = wg_inner_id / wg_num_n;
        int start_m_id = repeat_start_m + wg_coord_m;
        return start_m_id;
    }
    // get transformed dim2 group id
    template <>
    __XETLA_API int get_tile_idx<2>(sycl::nd_item<3> &item) {
        uint32_t group_range_n = item.get_group_range(2);
        uint32_t wg_repeat_n = group_range_n / wg_num_n;
        uint32_t repeat_id = get_2d_group_linear_id(item) / max_wg_num;
        uint32_t repeat_id_n = repeat_id % wg_repeat_n;
        uint32_t repeat_id_m = repeat_id / wg_repeat_n;
        uint32_t repeat_start_n_0 = repeat_id_n * wg_num_n;
        uint32_t repeat_start_n_1 = (wg_repeat_n - repeat_id_n - 1) * wg_num_n;
        uint32_t repeat_start_n
                = (repeat_id_m & 1) == 0 ? repeat_start_n_0 : repeat_start_n_1;
        uint32_t wg_inner_id = get_2d_group_linear_id(item) % max_wg_num;
        uint32_t wg_coord_n = wg_inner_id % wg_num_n;
        int start_n_id = repeat_start_n + wg_coord_n;
        return start_n_id;
    }
    // correct group range, workgroup will be padded to fit the given wg_num_n
    // under this swizzle policy
    static __XETLA_API void update_group_range(
            uint32_t &group_range_m, uint32_t &group_range_n) {
        group_range_m = (group_range_m + wg_num_m - 1) / wg_num_m * wg_num_m;
        group_range_n = (group_range_n + wg_num_n - 1) / wg_num_n * wg_num_n;
    }

private:
    static constexpr uint32_t max_wg_num = arch_attr_t<arch_tag>::max_wg_num;
    static constexpr uint32_t wg_num_n = wg_num_n_;
    static_assert(!(max_wg_num % wg_num_n),
            "max_wg_num cannot be divisible by given wg_num_n!");
    static constexpr uint32_t wg_num_m = max_wg_num / wg_num_n;
};

/// @brief Default GEMM_UNIVERSAL implementation.
/// A general GEMM_UNIVERSAL implementation to provide a composition point of gemm_universal and epilogue.
/// @tparam arch_tag_ Is the HW architecture.
template <typename group_swizzle_policy_>
struct dispatch_policy_default {
    using group_swizzle_policy = group_swizzle_policy_;
    static constexpr gpu_arch arch_tag = group_swizzle_policy::arch_tag;
};

/// @brief Kslicing GEMM_UNIVERSAL implementation.
/// A special GEMM_UNIVERSAL implementation to increase the hardware occupancy by splitting the GEMM_UNIVERSAL task along k dimension.
/// It includes inter-group reduction (by using global atomic) and intra-group reduction (by using local memory for data exchange).
/// @tparam num_global_kslicing_ Is the k dim split ratio between groups.
/// @tparam num_local_kslicing_ Is the k dim split ratio within a group.
/// @tparam arch_tag_ Is the HW architecture.
template <typename group_swizzle_policy_, int global_ratio_ = 1,
        int local_ratio_ = 1>
struct dispatch_policy_kslicing {
    using group_swizzle_policy = group_swizzle_policy_;
    static constexpr int global_ratio = global_ratio_;
    static constexpr int local_ratio = local_ratio_;
    static constexpr gpu_arch arch_tag = group_swizzle_policy::arch_tag;
};

/// @brief StreamK GEMM implementation.
/// A special GEMM implementation to avoid tail effects when GEMM shape does not fit the machine
/// Implements variable K-slicing for effective load-balancing and performs inter-group reduction.
/// Implementation loosely based on this paper - https://arxiv.org/pdf/2301.03598.pdf
/// @tparam arch_tag_ Is the HW architecture.
template <gpu_arch arch_tag_ = gpu_arch::Xe>
struct dispatch_policy_stream_k {

    static constexpr gpu_arch arch_tag = arch_tag_;

    uint32_t matrix_m;
    uint32_t matrix_k;
    uint32_t matrix_n;

    uint32_t wg_tile_m;
    uint32_t wg_tile_k;
    uint32_t wg_tile_n;

    uint32_t sg_tile_m;
    uint32_t sg_tile_n;

    ///Number of xecores available for stream_k load balancing
    uint32_t avail_xecores;

    uint32_t num_workgroups;
    uint32_t dp_groups; /// Number of data-parallel workgroups

    uint32_t sk_tiles;
    uint32_t sk_waves;
    uint32_t sk_big_groups_per_region;
    uint32_t sk_iters_per_region;
    uint32_t sk_regions;
    uint32_t sk_groups_per_region;

    //FastDivMod counters initialized in host to use multiply and shift operations in kernel code for modulus and division
    FastDivMod div_mod_tiles_m;
    FastDivMod div_mod_tiles_n;
    FastDivMod div_mod_iters_per_tile;
    FastDivMod div_mod_sk_regions;
    FastDivMod div_mod_sk_groups_per_region;
    FastDivMod div_mod_sk_iters_per_normal_group;
    FastDivMod div_mod_sk_iters_per_region;
    FastDivMod div_mod_sk_iters_per_big_group;

    /// Minimum number  of MAC-iterations per streamk group
    static int const kMinItersPerSkGroup = 2;

    //Host+Device interface functions

    /// @brief Host helper function to get the expected nd_range under the current GEMM config.
    /// @return Expected nd_range.
    cl::sycl::range<3> get_group_range() const {
        cl::sycl::range<3> group_range
                = cl::sycl::range<3> {1, 1, num_workgroups};
        return group_range;
    };

    /// @brief Host helper function to compute sk_groups to dispatch for a given number of sk_tiles
    void get_sk_workgroups(int &sk_groups, /// [out]
            int &savings_iters, /// [out]
            int sk_tiles, int iters_per_tile, int avail_xecores,
            bool allow_partial_wave) const {

        savings_iters = INT_MIN;
        sk_groups = 0;

        if (sk_tiles == 0) { return; }

        int sk_iters = sk_tiles * iters_per_tile;

        int dp_equiv_waves = (sk_tiles + avail_xecores - 1) / avail_xecores;
        int dp_equiv_iters = iters_per_tile * dp_equiv_waves;

        int min_sk_groups = (allow_partial_wave)
                ? std::min(avail_xecores, sk_tiles + 1)
                : avail_xecores;
        int max_sk_groups
                = std::min(avail_xecores, sk_iters / kMinItersPerSkGroup);

        for (int trial_sk_groups = min_sk_groups;
                trial_sk_groups <= max_sk_groups; trial_sk_groups++) {

            int sk_waves
                    = (trial_sk_groups + avail_xecores - 1) / avail_xecores;
            int max_sk_iters_per_group
                    = (sk_iters + trial_sk_groups - 1) / trial_sk_groups;
            int sk_iter_equiv = max_sk_iters_per_group * sk_waves;

            int num_peers = ((trial_sk_groups + sk_tiles - 1) / sk_tiles) + 1;
            float iter_cost = 0.02f * float(num_peers) * float(sk_iter_equiv);

            if (trial_sk_groups % sk_tiles == 0) {

                //aligned
                num_peers = (trial_sk_groups / sk_tiles);
                iter_cost = 0.0f;
            }

            float peer_cost = 2.0f * float(num_peers);
            float base_cost = 2.0f * float(sk_waves);

            int fixup_iter_equiv = int(base_cost + iter_cost + peer_cost);

            int trial_savings_iter
                    = dp_equiv_iters - sk_iter_equiv - fixup_iter_equiv;

            if (trial_savings_iter >= savings_iters) {

                savings_iters = trial_savings_iter;
                sk_groups = trial_sk_groups;
            }
        }
    }

    /// @brief Determine the populations of DP and SK groups to invoke for the given number of output tiles
    void get_groups(int &dp_tiles, int &sk_groups, int output_tiles,
            int iters_per_tile, int avail_xecores) {

        int full_waves = output_tiles / avail_xecores;
        int full_wave_tiles = full_waves * avail_xecores;
        int partial_wave_tiles = output_tiles - full_wave_tiles;

        if (partial_wave_tiles == 0) {
            //No tails
            return;
        }
        int score = -1;
        dp_tiles = output_tiles;
        sk_groups = 0;

        if (full_waves < 1) {

            dp_tiles = full_wave_tiles;

            get_sk_workgroups(sk_groups, score, partial_wave_tiles,
                    iters_per_tile, avail_xecores, true);

            if (score < 0) {
                //Not profitable
                dp_tiles = output_tiles;
                sk_groups = 0;
            }

            return;
        }

        //Form the SK wave by combining the last full wave and the partial wave
        dp_tiles = full_wave_tiles - avail_xecores;

        get_sk_workgroups(sk_groups, score, partial_wave_tiles + avail_xecores,
                iters_per_tile, avail_xecores,
                false); // cannot run with less than a full wave of SK-groups

        std::cout << "SK Score: " << score << "\n\n";

        if (score < 0) { //Not profitable for stream_k split

            sk_groups = 0;
            dp_tiles = output_tiles;
        }
    }

    ///Constructor
    inline dispatch_policy_stream_k() = default;

    /// @brief Set for device copyable
    //static constexpr bool host_callable = true;

    inline dispatch_policy_stream_k(uint32_t matrix_m_, uint32_t matrix_k_,
            uint32_t matrix_n_, uint32_t wg_tile_m_, uint32_t wg_tile_k_,
            uint32_t wg_tile_n_, uint32_t sg_tile_m_, uint32_t sg_tile_n_,
            uint32_t avail_xecores_ = arch_attr_t<arch_tag>::max_wg_num)
        : matrix_m(matrix_m_)
        , matrix_k(matrix_k_)
        , matrix_n(matrix_n_)
        , wg_tile_m(wg_tile_m_)
        , wg_tile_k(wg_tile_k_)
        , wg_tile_n(wg_tile_n_)
        , sg_tile_m(sg_tile_m_)
        , sg_tile_n(sg_tile_n_)
        , avail_xecores(avail_xecores_) {

        int iters_per_tile = (matrix_k + wg_tile_k - 1) / wg_tile_k;

        //Default values for sk parameters
        int sk_iters_per_normal_group = 0;
        int sk_iters_per_big_group = 0;

        // Default : a single region of iteration space across all SK tiles
        sk_regions = 1;

        sk_groups_per_region = 1;
        sk_waves = 0;

        int num_tiles_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
        int num_tiles_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;

        int output_tiles = num_tiles_m * num_tiles_n;
        int waves = (output_tiles + avail_xecores - 1) / avail_xecores;
        float dp_efficiency
                = float(output_tiles) / float(waves * avail_xecores);

        int dp_tiles = output_tiles;
        int sk_groups = 0;

        //Use heuristics to get stream_k split
        get_groups(dp_tiles, sk_groups, output_tiles, iters_per_tile,
                avail_xecores);

        sk_tiles = output_tiles - dp_tiles;

        // Compute SK group iteration details
        if (sk_groups > 0) {

            sk_waves = (sk_groups + avail_xecores - 1) / avail_xecores;
            //Compute global iteration space - tiles_m*tiles_n*k_iters
            int sk_iters = sk_tiles * iters_per_tile;
            sk_groups = std::min(sk_groups, sk_iters);

            //sk_iters may not divide sk_groups evenly; some groups perform one additional iteration
            sk_iters_per_normal_group = sk_iters / sk_groups;
            int extra_sk_iters
                    = sk_iters - (sk_iters_per_normal_group * sk_groups);
            int sk_big_groups = extra_sk_iters;
            sk_iters_per_big_group = sk_iters_per_normal_group + 1;

            //KSlicing to fill up multiple regions within groups
            if ((sk_groups > sk_tiles) && (sk_groups % sk_tiles == 0)) {

                sk_regions = sk_tiles;
            }

            sk_groups_per_region = sk_groups / sk_regions;
            sk_big_groups_per_region = sk_big_groups / sk_regions;
            sk_iters_per_region = sk_iters / sk_regions;

            //Initialize fast divmod counters related to SK
            div_mod_sk_regions = FastDivMod(sk_regions);
            div_mod_sk_groups_per_region = FastDivMod(sk_groups_per_region);
            div_mod_sk_iters_per_normal_group
                    = FastDivMod(sk_iters_per_normal_group);
            div_mod_sk_iters_per_big_group = FastDivMod(sk_iters_per_big_group);
        }

        div_mod_tiles_m = FastDivMod(num_tiles_m);
        div_mod_tiles_n = FastDivMod(num_tiles_n);
        div_mod_iters_per_tile = FastDivMod(iters_per_tile);

        dp_groups = dp_tiles;
        num_workgroups = get_num_active_groups();

        //Print the stats
        uint32_t total_tiles = num_tiles_m * num_tiles_n;
        std::cout << " problem size: (" << matrix_m << "," << matrix_n << ")"
                  << ", tiled_shape: (" << num_tiles_m << "," << num_tiles_n
                  << ")"
                  << ", tiles: " << total_tiles
                  << ", dp_tiles: " << total_tiles - sk_tiles
                  << ", sk_tiles: " << sk_tiles
                  << ", iters_per_tile: " << iters_per_tile
                  << ", num_workgroups: " << num_workgroups
                  << ", dp_workgroups: " << dp_groups
                  << ", dp_waves: " << dp_groups / avail_xecores
                  << ", sk_groups_per_region: " << sk_groups_per_region
                  << ", sk_regions: " << sk_regions
                  << ", sk_waves: " << sk_waves
                  << ", sk_iters_per_normal_group: "
                  << sk_iters_per_normal_group
                  << ", sk_big_groups_per_region: " << sk_big_groups_per_region
                  << ", avail_xecores: " << avail_xecores << "\n\n";
    }

    ///@brief Host helper function to return number of groups after stream_k split
    int get_num_active_groups() const {
        return (sk_waves * avail_xecores) + dp_groups;
    }

    ///@brief Kernel helper function to return number of K-iters per output tile
    __XETLA_API KERNEL_FUNC int get_iters_per_tile() const {

        return static_cast<int>(div_mod_iters_per_tile);
    }

    ///@brief Kernel helper function to return number of K-iters for normal sk groups
    __XETLA_API KERNEL_FUNC int get_sk_iters_per_normal_group() const {

        return static_cast<int>(div_mod_sk_iters_per_normal_group);
    }

    ///@brief Kernel helper function to return number of SK regions
    __XETLA_API KERNEL_FUNC int get_sk_regions() const {

        return static_cast<int>(div_mod_sk_regions);
    }

    ///@brief Kernel helper function to return number of SK groups per region
    __XETLA_API KERNEL_FUNC int get_sk_groups_per_region() const {

        return static_cast<int>(div_mod_sk_groups_per_region);
    }

    ///@brief Kernel function to get tile offset for m and n
    __XETLA_API KERNEL_FUNC void get_tile_offsets(
            int tile_idx, int &tile_offset_m, int &tile_offset_n) const {

        int tiles_m = static_cast<int>(div_mod_tiles_m);
        int tiles_n = static_cast<int>(div_mod_tiles_n);
        if (tiles_m > tiles_n) {
            div_mod_tiles_n.fast_divmod(tile_offset_m, tile_offset_n, tile_idx);
        } else {
            div_mod_tiles_m.fast_divmod(tile_offset_n, tile_offset_m, tile_idx);
        }
    }

    ///@brief Kernel function to return tile idx for current sk iteration
    __XETLA_API KERNEL_FUNC int get_sk_tile_idx(int iter) const {

        int tile_idx = div_mod_iters_per_tile.div(iter);
        return tile_idx;
    }

    ///@brief Kernel function to get iteration extends for stream_k split
    __XETLA_API KERNEL_FUNC void get_iter_extents(int sk_group_idx,
            int &group_iter_begin, int &group_iter_end) const {
        int region_idx;
        int group_idx_in_region;
        div_mod_sk_groups_per_region.fast_divmod(
                region_idx, group_idx_in_region, sk_group_idx);

        group_iter_begin = (region_idx * sk_iters_per_region)
                + (group_idx_in_region * get_sk_iters_per_normal_group());

        //Adjust extents for the first num_big_group groups that get one extra iteration
        int group_iters = get_sk_iters_per_normal_group();
        if (group_idx_in_region < sk_big_groups_per_region) {

            group_iter_begin += group_idx_in_region;
            group_iters += 1;
        } else {

            //This is a regular group
            group_iter_begin += sk_big_groups_per_region;
        }

        group_iter_end = group_iter_begin + group_iters;
    }

    ///@brief kernel function to get the first sk group index writing the sliced output tile;
    __XETLA_API KERNEL_FUNC int get_first_group_idx(
            int tile_idx, int group_idx) const {

        if (tile_idx >= sk_tiles) {
            //DP group
            return group_idx;
        }

        int iter = tile_idx * get_iters_per_tile();

        int region_idx, iter_in_region;

        div_mod_sk_iters_per_region.fast_divmod(
                region_idx, iter_in_region, iter);

        //Number of iterations in the big group region
        int big_group_iters
                = sk_big_groups_per_region * get_sk_iters_per_normal_group()
                + sk_big_groups_per_region;

        //Number of iterations in the normal group region
        int normal_group_iters = iter_in_region - big_group_iters;

        int big_group_idx_in_region
                = div_mod_sk_iters_per_big_group.div(iter_in_region);

        int normal_group_idx_in_region = sk_big_groups_per_region
                + div_mod_sk_iters_per_normal_group.div(normal_group_iters);

        int group_idx_in_region
                = (big_group_idx_in_region < sk_big_groups_per_region)
                ? big_group_idx_in_region
                : normal_group_idx_in_region;

        int owning_group_idx = (get_sk_groups_per_region() * region_idx)
                + group_idx_in_region;

        return owning_group_idx;
    }
};

/// @} xetla_gemm_universal

} // namespace gpu::xetla::kernel
