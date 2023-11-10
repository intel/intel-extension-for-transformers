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

#include "common/common.hpp"
#include "group/group.hpp"
#include "subgroup/subgroup.hpp"

namespace gpu::xetla {

// enum

enum class tune_key {
    DATA_TYPE_A,
    MEMORY_LAYOUT_A,
    MEMORY_ALIGNMENT_A,
    MEMORY_SPACE_A,
    DATA_TYPE_B,
    MEMORY_LAYOUT_B,
    MEMORY_ALIGNMENT_B,
    MEMORY_SPACE_B,
    DATA_TYPE_C,
    MEMORY_LAYOUT_C,
    MEMORY_ALIGNMENT_C,
    MEMORY_SPACE_C,
    DATA_TYPE_ACC,
    GLOBAL_KSLICING_RATIO,
    LOCAL_KSLICING_RATIO,
    WG_TILE_SHAPE,
    WG_TILE_K,
    SG_TILE_SHAPE,
    PRE_PROCESSING,
    PREFETCH_DISTANCE,
    PERIODIC_SYNC_INTERVAL,
    MMA_ENGINE,
    GPU_ARCH,
    EPILOGUE_POLICY,
    DISPATCH_POLICY,
    GROUP_SWIZZLE_POLICY,
    PARAM_OPTIMZER_TYPE,
    SOURCE_LOCATION
};

enum class tune_key_value {
    PRE_PROCESSING_DEFAULT,
    PRE_PROCESSING_MATA_NEG_FILTER,
    DISPATCH_POLICY_DEFAULT,
    DISPATCH_POLICY_KSLICING,
    DISPATCH_POLICY_STREAM_K,
    PARAM_OPTIMZER_DUMMY,
    PARAM_OPTIMZER_DECISION_TREE
};

// parameter optimizer

enum class param_optimizer_tag { KERNEL, WORKGROUP };

template <param_optimizer_tag tag_, typename dict_t_>
struct param_optimizer;

struct param_optimizer_base {
    template <typename T, typename U>
    struct validate_attribute {
        static constexpr bool value = []() constexpr {
            bool valid = true;
            valid &= std::is_same<typename T::template find_elem_t<
                                          tune_key::DATA_TYPE_A>::type,
                    typename U::template find_elem_t<
                            tune_key::DATA_TYPE_A>::type>::value;
            valid &= T::template find_elem_v<tune_key::
                                     MEMORY_LAYOUT_A> == U::template find_elem_v<tune_key::MEMORY_LAYOUT_A>;
            valid &= T::template find_elem_v<tune_key::
                                     MEMORY_ALIGNMENT_A> == U::template find_elem_v<tune_key::MEMORY_ALIGNMENT_A>;
            valid &= std::is_same<typename T::template find_elem_t<
                                          tune_key::DATA_TYPE_B>::type,
                    typename U::template find_elem_t<
                            tune_key::DATA_TYPE_B>::type>::value;
            valid &= T::template find_elem_v<tune_key::
                                     MEMORY_LAYOUT_B> == U::template find_elem_v<tune_key::MEMORY_LAYOUT_B>;
            valid &= T::template find_elem_v<tune_key::
                                     MEMORY_ALIGNMENT_B> == U::template find_elem_v<tune_key::MEMORY_ALIGNMENT_B>;
            valid &= std::is_same<typename T::template find_elem_t<
                                          tune_key::DATA_TYPE_C>::type,
                    typename U::template find_elem_t<
                            tune_key::DATA_TYPE_C>::type>::value;
            valid &= T::template find_elem_v<tune_key::
                                     MEMORY_LAYOUT_C> == U::template find_elem_v<tune_key::MEMORY_LAYOUT_C>;
            valid &= T::template find_elem_v<tune_key::
                                     MEMORY_ALIGNMENT_C> == U::template find_elem_v<tune_key::MEMORY_ALIGNMENT_C>;
            valid &= T::template find_elem_v<tune_key::
                                     GPU_ARCH> == U::template find_elem_v<tune_key::GPU_ARCH>;
            return valid;
        }
        ();
    };
};

// parameter adaptor

enum class param_adaptor_tag { KERNEL, WORKGROUP_GEMM, WORKGROUP_EPILOGUE };

template <param_adaptor_tag tag_, typename dict_t_>
struct param_adaptor;

template <typename dict_t_>
struct param_adaptor_base {
    using dtype_acc = typename dict_t_::template find_elem_t<
            tune_key::DATA_TYPE_ACC>::type;
    using wg_tile_shape = typename dict_t_::template find_elem_t<
            tune_key::WG_TILE_SHAPE>::type;
    static constexpr uint32_t wg_tile_n = wg_tile_shape::template dim<0>();
    static constexpr uint32_t wg_tile_m = wg_tile_shape::template dim<1>();
    static constexpr uint32_t wg_tile_k
            = dict_t_::template find_elem_v<tune_key::WG_TILE_K>;
    using sg_tile_shape = typename dict_t_::template find_elem_t<
            tune_key::SG_TILE_SHAPE>::type;
    static constexpr uint32_t sg_tile_n = sg_tile_shape::template dim<0>();
    static constexpr uint32_t sg_tile_m = sg_tile_shape::template dim<1>();
    static constexpr uint32_t prefetch_distance
            = dict_t_::template find_elem_v<tune_key::PREFETCH_DISTANCE>;
    static constexpr uint32_t periodic_sync_interval
            = dict_t_::template find_elem_v<tune_key::PERIODIC_SYNC_INTERVAL>;
    static constexpr auto mma_engine_tag
            = dict_t_::template find_elem_v<tune_key::MMA_ENGINE>;
    static constexpr auto gpu_arch_tag
            = dict_t_::template find_elem_v<tune_key::GPU_ARCH>;

    // Org the compute shape for sub-matrix
    using tile_shape = group::tile_shape_t<wg_tile_n, // workgroup size in dim0
            wg_tile_m, //	workgroup size in dim1
            sg_tile_n, //	subgroup size in dim0
            sg_tile_m>; //	subgroup size in dim1
};

} // namespace gpu::xetla

#include "kernel/default_config/decision_tree_policy.hpp"
#include "kernel/default_config/dummy_policy.hpp"
