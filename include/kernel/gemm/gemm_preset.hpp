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

#include "kernel/default_config/common.hpp"
#include "kernel/gemm/common.hpp"

namespace gpu::xetla {
namespace detail {
using param_dtype_bf16_bf16_bf16 = dict_t<elem_t_t<tune_key::DATA_TYPE_A, bf16>,
        elem_t_t<tune_key::DATA_TYPE_B, bf16>,
        elem_t_t<tune_key::DATA_TYPE_C, bf16>>;

using param_memalignment_8_8_8
        = dict_t<elem_v_t<tune_key::MEMORY_ALIGNMENT_A, 8UL, uint32_t>,
                elem_v_t<tune_key::MEMORY_ALIGNMENT_B, 8UL, uint32_t>,
                elem_v_t<tune_key::MEMORY_ALIGNMENT_C, 8UL, uint32_t>>;

using param_memlayout_rrr
        = dict_t<elem_v_t<tune_key::MEMORY_LAYOUT_A, mem_layout::row_major>,
                elem_v_t<tune_key::MEMORY_LAYOUT_B, mem_layout::row_major>,
                elem_v_t<tune_key::MEMORY_LAYOUT_C, mem_layout::row_major>>;

using param_memspace_ggg
        = dict_t<elem_v_t<tune_key::MEMORY_SPACE_A, mem_space::global>,
                elem_v_t<tune_key::MEMORY_SPACE_B, mem_space::global>,
                elem_v_t<tune_key::MEMORY_SPACE_C, mem_space::global>>;

using param_performance_default
        = dict_t<elem_v_t<tune_key::WG_TILE_K, 32UL, uint32_t>,
                elem_v_t<tune_key::PREFETCH_DISTANCE, 3UL, uint32_t>,
                elem_v_t<tune_key::PERIODIC_SYNC_INTERVAL, 8UL, uint32_t>>;

using param_runtime_default
        = dict_t<elem_v_t<tune_key::PRE_PROCESSING,
                         tune_key_value::PRE_PROCESSING_DEFAULT>,
                elem_v_t<tune_key::MMA_ENGINE, mma_engine::xmx>,
                elem_v_t<tune_key::GPU_ARCH, gpu_arch::Xe>,
                elem_t_t<tune_key::EPILOGUE_POLICY,
                        group::epilogue_policy_default<gpu_arch::Xe>>,
                elem_v_t<tune_key::DISPATCH_POLICY,
                        tune_key_value::DISPATCH_POLICY_DEFAULT>,
                elem_t_t<tune_key::GROUP_SWIZZLE_POLICY,
                        kernel::group_swizzle_default<gpu_arch::Xe>>>;
} // namespace detail

using default_param_t = dict_t<>::template update_dict_t<
        detail::param_dtype_bf16_bf16_bf16>::template update_dict_t<detail::
                param_memlayout_rrr>::template update_dict_t<detail::
                param_memalignment_8_8_8>::template update_dict_t<detail::
                param_memspace_ggg>::template update_dict_t<detail::
                param_performance_default>::template update_dict_t<detail::
                param_runtime_default>::
        template update_t<elem_t_t<tune_key::DATA_TYPE_ACC, float>,
                elem_v_t<tune_key::GLOBAL_KSLICING_RATIO, 1UL, uint32_t>,
                elem_v_t<tune_key::LOCAL_KSLICING_RATIO, 1UL, uint32_t>,
                elem_t_t<tune_key::WG_TILE_SHAPE, shape<256, 256>>,
                elem_t_t<tune_key::SG_TILE_SHAPE, shape<64, 32>>,
                elem_v_t<tune_key::PARAM_OPTIMZER_TYPE,
                        tune_key_value::PARAM_OPTIMZER_DUMMY>>;

namespace kernel {
using param_kslicing_g1l1_t = default_param_t::template update_t<
        elem_v_t<tune_key::GLOBAL_KSLICING_RATIO, 1UL, uint32_t>,
        elem_v_t<tune_key::LOCAL_KSLICING_RATIO, 1UL, uint32_t>,
        elem_t_t<tune_key::WG_TILE_SHAPE, shape<256, 256>>,
        elem_v_t<tune_key::WG_TILE_K, 32UL, uint32_t>,
        elem_t_t<tune_key::SG_TILE_SHAPE, shape<64, 32>>,
        elem_v_t<tune_key::DISPATCH_POLICY,
                tune_key_value::DISPATCH_POLICY_KSLICING>>;

using param_kslicing_g2l1_t = default_param_t::template update_t<
        elem_v_t<tune_key::GLOBAL_KSLICING_RATIO, 2UL, uint32_t>,
        elem_v_t<tune_key::LOCAL_KSLICING_RATIO, 1UL, uint32_t>,
        elem_t_t<tune_key::WG_TILE_SHAPE, shape<256, 256>>,
        elem_v_t<tune_key::WG_TILE_K, 32UL, uint32_t>,
        elem_t_t<tune_key::SG_TILE_SHAPE, shape<64, 32>>,
        elem_v_t<tune_key::DISPATCH_POLICY,
                tune_key_value::DISPATCH_POLICY_KSLICING>>;

using param_kslicing_g1l2_t = default_param_t::template update_t<
        elem_v_t<tune_key::GLOBAL_KSLICING_RATIO, 1UL, uint32_t>,
        elem_v_t<tune_key::LOCAL_KSLICING_RATIO, 2UL, uint32_t>,
        elem_t_t<tune_key::WG_TILE_SHAPE, shape<128, 64>>,
        elem_v_t<tune_key::WG_TILE_K, 32UL, uint32_t>,
        elem_t_t<tune_key::SG_TILE_SHAPE, shape<32, 16>>,
        elem_v_t<tune_key::DISPATCH_POLICY,
                tune_key_value::DISPATCH_POLICY_KSLICING>>;

} // namespace kernel

namespace group {
using param_dict1_wg_t = default_param_t::template update_t<
        elem_t_t<tune_key::DATA_TYPE_ACC, float>,
        elem_t_t<tune_key::WG_TILE_SHAPE, shape<256, 256>>,
        elem_v_t<tune_key::WG_TILE_K, 32UL, uint32_t>,
        elem_t_t<tune_key::SG_TILE_SHAPE, shape<64, 32>>,
        elem_v_t<tune_key::PREFETCH_DISTANCE, 3UL, uint32_t>,
        elem_v_t<tune_key::PERIODIC_SYNC_INTERVAL, 8UL, uint32_t>,
        elem_t_t<tune_key::EPILOGUE_POLICY,
                group::epilogue_policy_default<gpu_arch::Xe>>>;
}
} // namespace gpu::xetla
