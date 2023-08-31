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

#include "common/core/common.hpp"
#include "xetla.hpp"
#include <gtest/gtest.h>

using namespace gpu::xetla;
using namespace cl::sycl;

TEST(slm, check_aligment) {
    EXPECT_TRUE(limitation<gpu_arch::Xe>::slm::check_alignment("test", 0));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::slm::check_alignment("test", 1));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::slm::check_alignment("test", 2));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::slm::check_alignment("test", 3));
    EXPECT_TRUE(limitation<gpu_arch::Xe>::slm::check_alignment("test", 4));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::slm::check_alignment("test", 5));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::slm::check_alignment("test", 6));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::slm::check_alignment("test", 7));
    EXPECT_TRUE(limitation<gpu_arch::Xe>::slm::check_alignment("test", 8));
}

TEST(block_1d, check_alignment) {
    EXPECT_FALSE(limitation<gpu_arch::Xe>::block_1d<fp16>::check_alignment(
            (fp16 *)0x40000, 0));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::block_1d<fp16>::check_alignment(
            (fp16 *)0x40000, 1));
    EXPECT_TRUE(limitation<gpu_arch::Xe>::block_1d<fp16>::check_alignment(
            (fp16 *)0x40000, 2));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::block_1d<fp16>::check_alignment(
            (fp16 *)0x40000, 3));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::block_1d<uint8_t>::check_alignment(
            (uint8_t *)0x40000, 6));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::block_1d<uint8_t>::check_alignment(
            (uint8_t *)0x40000, 2));
    EXPECT_TRUE(limitation<gpu_arch::Xe>::block_1d<float>::check_alignment(
            (float *)0x40000, 1));

    EXPECT_FALSE(limitation<gpu_arch::Xe>::block_1d<float>::check_alignment(
            (float *)0x40001, 1));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::block_1d<float>::check_alignment(
            (float *)0x40002, 1));
    EXPECT_FALSE(limitation<gpu_arch::Xe>::block_1d<float>::check_alignment(
            (float *)0x40003, 1));
    EXPECT_TRUE(limitation<gpu_arch::Xe>::block_1d<float>::check_alignment(
            (float *)0x40004, 1));
}

TEST(block_2d, check_load) {
    xetla_tdescriptor td;

    xetla_fill_tdesc<fp16, 65, 31, 1>(
            td.xetla_format<uint32_t>(), (fp16 *)4096, 32, 64, 32, 0, 0);
    EXPECT_FALSE((
            limitation<gpu_arch::Xe>::block_2d<fp16>::template check_load<false,
                    false>(td)));

    xetla_fill_tdesc<fp16, 64, 31, 1>(
            td.xetla_format<uint32_t>(), (fp16 *)4096, 32, 64, 32, 0, 0);
    EXPECT_FALSE((
            limitation<gpu_arch::Xe>::block_2d<fp16>::template check_load<false,
                    false>(td)));

    xetla_fill_tdesc<fp16, 32, 31, 1>(
            td.xetla_format<uint32_t>(), (fp16 *)4096, 32, 64, 32, 0, 0);
    EXPECT_TRUE((
            limitation<gpu_arch::Xe>::block_2d<fp16>::template check_load<false,
                    false>(td)));

    xetla_fill_tdesc<fp16, 32, 31, 1>(
            td.xetla_format<uint32_t>(), (fp16 *)4096, 32, 64, 32, -1, 0);
    EXPECT_FALSE((
            limitation<gpu_arch::Xe>::block_2d<fp16>::template check_load<false,
                    false>(td)));

    xetla_fill_tdesc<fp16, 32, 31, 1>(
            td.xetla_format<uint32_t>(), (fp16 *)4096, 32, 64, 32, -4, 0);
    EXPECT_TRUE((
            limitation<gpu_arch::Xe>::block_2d<fp16>::template check_load<false,
                    false>(td)));

    xetla_fill_tdesc<fp16, 31, 31, 1>(
            td.xetla_format<uint32_t>(), (fp16 *)4096, 32, 64, 32, 0, 0);
    EXPECT_FALSE((
            limitation<gpu_arch::Xe>::block_2d<fp16>::template check_load<false,
                    false>(td)));
}

TEST(block_2d, check_store) {
    xetla_tdescriptor td;
    xetla_fill_tdesc<fp16, 100, 200, 1>(
            td.xetla_format<uint32_t>(), (fp16 *)4096, 32, 64, 32, 0, 0);
    EXPECT_FALSE((limitation<gpu_arch::Xe>::block_2d<fp16>::check_store(td)));

    xetla_fill_tdesc<fp16, 64, 200, 1>(
            td.xetla_format<uint32_t>(), (fp16 *)4096, 32, 64, 32, 0, 0);
    EXPECT_FALSE((limitation<gpu_arch::Xe>::block_2d<fp16>::check_store(td)));

    xetla_fill_tdesc<fp16, 64, 8, 1>(
            td.xetla_format<uint32_t>(), (fp16 *)4096, 32, 64, 32, 0, 0);
    EXPECT_FALSE((limitation<gpu_arch::Xe>::block_2d<fp16>::check_store(td)));

    xetla_fill_tdesc<fp16, 32, 8, 1>(
            td.xetla_format<uint32_t>(), (fp16 *)4096, 32, 64, 32, 0, 0);
    EXPECT_TRUE((limitation<gpu_arch::Xe>::block_2d<fp16>::check_store(td)));
}

TEST(block_2d, check_tensor) {
    using check_2d = limitation<gpu_arch::Xe>::block_2d<float>;
    EXPECT_FALSE((check_2d::check_tensor(8192, 32, 64, 32)));
    EXPECT_FALSE((check_2d::check_tensor(8192, 65, 64, 32)));
    EXPECT_FALSE((check_2d::check_tensor(8192, 64, 64, 32)));
    EXPECT_FALSE((check_2d::check_tensor(8192, 64, 64, 65)));
    EXPECT_TRUE((check_2d::check_tensor(8192, 64, 64, 72)));
    EXPECT_FALSE((check_2d::check_tensor(8224, 64, 64, 72)));
}
