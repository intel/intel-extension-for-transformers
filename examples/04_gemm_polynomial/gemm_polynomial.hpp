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

#include "xetla.hpp"

namespace gpu::xetla::subgroup {

template <typename dtype_, uint32_t N>
struct polynomial_op_t {
    using dtype = dtype_;
    using coeff_t = xetla_vector<dtype, N>;

    struct arguments_t {
        coeff_t coeff;
        inline arguments_t() = default;
        inline arguments_t(coeff_t coeff_) : coeff(coeff_) {}
    };
    template <typename matAcc_t, typename coord_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc, coord_t coord,
            arguments_t args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        using dtype_acc = typename matAcc_t::dtype;
        // total flag register
        constexpr uint32_t elems = 8 * 16;
        constexpr uint32_t rounds = matAcc_t::tile_elems / elems;
#pragma unroll
        for (uint32_t r = 0; r < rounds; ++r) {
            xetla_vector<dtype_acc, elems> res(0.0f);
            auto x = matAcc.reg.xetla_select<elems, 1>(elems * r);
#pragma unroll
            for (uint32_t i = 0; i < N; ++i) {
                res = x * res;
                res += static_cast<dtype_acc>(args.coeff[i]);
            }
            x = res;
        }
        constexpr uint32_t remained_elems = matAcc_t::tile_elems % elems;
        if constexpr (remained_elems != 0) {
            xetla_vector<dtype_acc, remained_elems> res(0.0f);
            auto x = matAcc.reg.xetla_select<remained_elems, 1>(
                    elems * (matAcc_t::tile_elems / elems));
#pragma unroll
            for (uint32_t i = 0; i < N; ++i) {
                res = x * res;
                res += static_cast<dtype_acc>(args.coeff[i]);
            }
            x = res;
        }
    }
};

} // namespace gpu::xetla::subgroup
