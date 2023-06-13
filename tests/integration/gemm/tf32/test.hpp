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

#include "../tests/utils/utils.hpp"
#include <gtest/gtest.h>

class TF32_test_mat_m_9158_mat_k_4926_mat_n_466_wg_m_16_wg_n_64_sg_m_8_sg_n_16_sg_k_16_l3_kslicing_1_slm_kslicing_4_layout_a_row_major_layout_b_row_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32 {
public:
    static constexpr size_t mat_m = 9158;
    static constexpr size_t mat_k = 4926;
    static constexpr size_t mat_n = 466;
    static constexpr size_t wg_m = 16;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 4;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = tf32;
    using data_type_b = tf32;
    using data_type_c = tf32;
};

class TF32_test_mat_m_5376_mat_k_8008_mat_n_290_wg_m_16_wg_n_128_sg_m_8_sg_n_32_sg_k_8_l3_kslicing_1_slm_kslicing_4_layout_a_col_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32 {
public:
    static constexpr size_t mat_m = 5376;
    static constexpr size_t mat_k = 8008;
    static constexpr size_t mat_n = 290;
    static constexpr size_t wg_m = 16;
    static constexpr size_t wg_n = 128;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 32;
    static constexpr size_t sg_k = 8;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 4;
    static constexpr mem_layout layout_a = mem_layout::col_major;
    static constexpr mem_layout layout_b = mem_layout::col_major;
    using data_type_a = tf32;
    using data_type_b = tf32;
    using data_type_c = tf32;
};

class TF32_test_mat_m_8104_mat_k_4870_mat_n_732_wg_m_16_wg_n_128_sg_m_8_sg_n_32_sg_k_8_l3_kslicing_1_slm_kslicing_4_layout_a_row_major_layout_b_row_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32 {
public:
    static constexpr size_t mat_m = 8104;
    static constexpr size_t mat_k = 4870;
    static constexpr size_t mat_n = 732;
    static constexpr size_t wg_m = 16;
    static constexpr size_t wg_n = 128;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 32;
    static constexpr size_t sg_k = 8;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 4;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = tf32;
    using data_type_b = tf32;
    using data_type_c = tf32;
};

class TF32_test_mat_m_6502_mat_k_7092_mat_n_990_wg_m_16_wg_n_16_sg_m_8_sg_n_16_sg_k_16_l3_kslicing_1_slm_kslicing_2_layout_a_row_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32 {
public:
    static constexpr size_t mat_m = 6502;
    static constexpr size_t mat_k = 7092;
    static constexpr size_t mat_n = 990;
    static constexpr size_t wg_m = 16;
    static constexpr size_t wg_n = 16;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 2;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::col_major;
    using data_type_a = tf32;
    using data_type_b = tf32;
    using data_type_c = tf32;
};

class TF32_test_mat_m_2750_mat_k_6686_mat_n_520_wg_m_64_wg_n_32_sg_m_32_sg_n_32_sg_k_16_l3_kslicing_1_slm_kslicing_2_layout_a_row_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32 {
public:
    static constexpr size_t mat_m = 2750;
    static constexpr size_t mat_k = 6686;
    static constexpr size_t mat_n = 520;
    static constexpr size_t wg_m = 64;
    static constexpr size_t wg_n = 32;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 32;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 2;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::col_major;
    using data_type_a = tf32;
    using data_type_b = tf32;
    using data_type_c = tf32;
};

class TF32_test_mat_m_7000_mat_k_5568_mat_n_248_wg_m_32_wg_n_32_sg_m_16_sg_n_32_sg_k_16_l3_kslicing_1_slm_kslicing_2_layout_a_row_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32 {
public:
    static constexpr size_t mat_m = 7000;
    static constexpr size_t mat_k = 5568;
    static constexpr size_t mat_n = 248;
    static constexpr size_t wg_m = 32;
    static constexpr size_t wg_n = 32;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 32;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 2;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::col_major;
    using data_type_a = tf32;
    using data_type_b = tf32;
    using data_type_c = tf32;
};

class TF32_test_mat_m_3844_mat_k_6020_mat_n_838_wg_m_32_wg_n_64_sg_m_16_sg_n_16_sg_k_8_l3_kslicing_1_slm_kslicing_4_layout_a_row_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32 {
public:
    static constexpr size_t mat_m = 3844;
    static constexpr size_t mat_k = 6020;
    static constexpr size_t mat_n = 838;
    static constexpr size_t wg_m = 32;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 8;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 4;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::col_major;
    using data_type_a = tf32;
    using data_type_b = tf32;
    using data_type_c = tf32;
};

class TF32_test_mat_m_3312_mat_k_5044_mat_n_750_wg_m_32_wg_n_64_sg_m_16_sg_n_16_sg_k_16_l3_kslicing_1_slm_kslicing_4_layout_a_row_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32 {
public:
    static constexpr size_t mat_m = 3312;
    static constexpr size_t mat_k = 5044;
    static constexpr size_t mat_n = 750;
    static constexpr size_t wg_m = 32;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 4;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::col_major;
    using data_type_a = tf32;
    using data_type_b = tf32;
    using data_type_c = tf32;
};

class TF32_test_mat_m_7682_mat_k_6370_mat_n_456_wg_m_32_wg_n_256_sg_m_16_sg_n_64_sg_k_16_l3_kslicing_1_slm_kslicing_4_layout_a_row_major_layout_b_row_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32 {
public:
    static constexpr size_t mat_m = 7682;
    static constexpr size_t mat_k = 6370;
    static constexpr size_t mat_n = 456;
    static constexpr size_t wg_m = 32;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t l3_kslicing = 1;
    static constexpr uint32_t slm_kslicing = 4;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = tf32;
    using data_type_b = tf32;
    using data_type_c = tf32;
};

using tests = ::testing::Types<
        TF32_test_mat_m_9158_mat_k_4926_mat_n_466_wg_m_16_wg_n_64_sg_m_8_sg_n_16_sg_k_16_l3_kslicing_1_slm_kslicing_4_layout_a_row_major_layout_b_row_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32,
        TF32_test_mat_m_5376_mat_k_8008_mat_n_290_wg_m_16_wg_n_128_sg_m_8_sg_n_32_sg_k_8_l3_kslicing_1_slm_kslicing_4_layout_a_col_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32,
        TF32_test_mat_m_8104_mat_k_4870_mat_n_732_wg_m_16_wg_n_128_sg_m_8_sg_n_32_sg_k_8_l3_kslicing_1_slm_kslicing_4_layout_a_row_major_layout_b_row_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32,
        TF32_test_mat_m_6502_mat_k_7092_mat_n_990_wg_m_16_wg_n_16_sg_m_8_sg_n_16_sg_k_16_l3_kslicing_1_slm_kslicing_2_layout_a_row_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32,
        TF32_test_mat_m_2750_mat_k_6686_mat_n_520_wg_m_64_wg_n_32_sg_m_32_sg_n_32_sg_k_16_l3_kslicing_1_slm_kslicing_2_layout_a_row_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32,
        TF32_test_mat_m_7000_mat_k_5568_mat_n_248_wg_m_32_wg_n_32_sg_m_16_sg_n_32_sg_k_16_l3_kslicing_1_slm_kslicing_2_layout_a_row_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32,
        TF32_test_mat_m_3844_mat_k_6020_mat_n_838_wg_m_32_wg_n_64_sg_m_16_sg_n_16_sg_k_8_l3_kslicing_1_slm_kslicing_4_layout_a_row_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32,
        TF32_test_mat_m_3312_mat_k_5044_mat_n_750_wg_m_32_wg_n_64_sg_m_16_sg_n_16_sg_k_16_l3_kslicing_1_slm_kslicing_4_layout_a_row_major_layout_b_col_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32,
        TF32_test_mat_m_7682_mat_k_6370_mat_n_456_wg_m_32_wg_n_256_sg_m_16_sg_n_64_sg_k_16_l3_kslicing_1_slm_kslicing_4_layout_a_row_major_layout_b_row_major_data_type_a_tf32_data_type_b_tf32_data_type_c_tf32>;
