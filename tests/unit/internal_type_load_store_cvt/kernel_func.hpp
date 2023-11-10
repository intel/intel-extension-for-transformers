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

using namespace gpu::xetla;

template <typename dtype, int swidth, int sheight, int spitch, int bwidth,
        int bheight, bool transform = false, bool transpose = false,
        gpu_arch arch_tag = gpu_arch::Xe>
struct load_store_cvt_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {

        xetla_tdescriptor a_src_tdesc;
        xetla_tdescriptor b_src_tdesc;
        xetla_fill_tdesc<dtype, bwidth, bheight, 1>(
                a_src_tdesc.xetla_format<uint32_t>(), a, swidth, sheight,
                spitch, 0, 0);
        xetla_fill_tdesc<dtype, bwidth, bheight, 1>(
                b_src_tdesc.xetla_format<uint32_t>(), b, swidth, sheight,
                spitch, 0, 0);

        xetla_tprefetch_global<dtype, cache_hint::cached, cache_hint::cached,
                arch_tag>(a_src_tdesc);
        xetla_tprefetch_global<dtype, cache_hint::cached, cache_hint::cached,
                arch_tag>(b_src_tdesc);
        xetla_vector<dtype, bwidth *bheight> A_load_vec
                = xetla_tload_global<dtype, bwidth * bheight,
                        cache_hint::cached, cache_hint::cached, transpose,
                        transform, arch_tag>(a_src_tdesc);
        xetla_vector<dtype, bwidth *bheight> B_load_vec
                = xetla_tload_global<dtype, bwidth * bheight,
                        cache_hint::cached, cache_hint::cached, transpose,
                        transform, arch_tag>(b_src_tdesc);

        xetla_vector<float, bwidth *bheight> add_a
                = xetla_cvt<float, dtype, bwidth * bheight>(A_load_vec);
        xetla_vector<float, bwidth *bheight> add_b
                = xetla_cvt<float, dtype, bwidth * bheight>(B_load_vec);

        xetla_vector<float, bwidth *bheight> result_c = add_a + add_b;
        xetla_vector<dtype, bwidth *bheight> out
                = xetla_cvt<dtype, float, bwidth * bheight>(result_c);

        constexpr int store_height = 8;
        static_assert((bheight % store_height) == 0,
                "to simplify the test, please set bheight be a multiple of "
                "store_height(8)");
        xetla_tdescriptor dst_tdesc;
        xetla_fill_tdesc<dtype, bwidth, store_height, 1>(
                dst_tdesc.xetla_format<uint32_t>(), c, swidth, sheight, spitch,
                0, 0);
#pragma unroll
        for (int unroll_i = 0; unroll_i < bheight; unroll_i += store_height) {
            xetla_tstore_global<dtype, bwidth * store_height,
                    cache_hint::write_back, cache_hint::write_back, arch_tag>(
                    dst_tdesc,
                    out.xetla_select<bwidth * store_height, 1>(
                            bwidth * unroll_i));
            xetla_update_tdesc_offsety(
                    dst_tdesc.xetla_format<uint32_t>(), store_height);
        }
    }
};
