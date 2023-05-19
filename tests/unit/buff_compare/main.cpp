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

#include <cstring>
#include <limits>
#include <random>
#include "utils/utils.hpp"

using namespace buff_cmp;

template <typename dtype1>
bool validate_buff_cmp_same(std::string name) {
    std::default_random_engine gen;
    std::uniform_int_distribution<dtype1> dist1(
            std::numeric_limits<dtype1>::min(),
            std::numeric_limits<dtype1>::max());
    dtype1 data[10000], other[10000];
    for (int i = 0; i < 10000; ++i) {
        data[i] = other[i] = dist1(gen);
    }
    buff_vals<dtype1> data_buf(data, 10000);
    buff_vals<dtype1> other_buf(other, 10000);
    bool result = xetla_buff_cmp(data_buf, other_buf, name);
    return result;
}

template <typename dtype1>
bool validate_buff_cmp_diff(std::string name) {
    std::default_random_engine gen;
    std::uniform_int_distribution<dtype1> dist1(
            std::numeric_limits<dtype1>::min(),
            std::numeric_limits<dtype1>::max() / 2);
    std::uniform_int_distribution<dtype1> dist2(
            0, std::numeric_limits<dtype1>::max() / 2);
    dtype1 data[10000], other[10000];
    for (int i = 0; i < 10000; ++i) {
        data[i] = dist1(gen);
        other[i] = data[i] + dist2(gen);
    }
    buff_vals<dtype1> data_buf(data, 10000);
    buff_vals<dtype1> other_buf(other, 10000);
    bool result = xetla_buff_cmp(data_buf, other_buf, name);
    return result;
}

template <typename dtype1>
bool validate_buff_cmp_same_fp(std::string name) {
    std::default_random_engine gen;
    std::uniform_real_distribution<dtype1> dist1(
            std::numeric_limits<dtype1>::min(),
            std::numeric_limits<dtype1>::max());
    dtype1 data[10000], other[10000];
    for (int i = 0; i < 10000; ++i) {
        data[i] = other[i] = dist1(gen);
    }
    buff_vals<dtype1> data_buf(data, 10000);
    buff_vals<dtype1> other_buf(other, 10000);
    bool result = xetla_buff_cmp(data_buf, other_buf, name);
    return result;
}

template <typename dtype1>
bool validate_buff_cmp_diff_fp(std::string name) {
    std::default_random_engine gen;
    std::uniform_real_distribution<dtype1> dist1(
            std::numeric_limits<dtype1>::min() / 2,
            std::numeric_limits<dtype1>::max() / 2);
    std::uniform_real_distribution<dtype1> dist2(0, 1);
    dtype1 data[10000], other[10000];
    for (int i = 0; i < 10000; ++i) {
        data[i] = dist1(gen);
        other[i] = data[i] + dist1(gen) * (1 + dist2(gen));
    }
    buff_vals<dtype1> data_buf(data, 10000);
    buff_vals<dtype1> other_buf(other, 10000);
    bool result = xetla_buff_cmp(data_buf, other_buf, name);
    return result;
}

/* ============== INT TESTS =============== */

// int8
TEST(buff_compare, matching_int8) {
    bool result = validate_buff_cmp_same<int8_t>("matching_int8_test");
    ASSERT_EQ(true, result);
}

TEST(buff_compare, diff_int8) {
    bool result = validate_buff_cmp_diff<int8_t>("diff_int8_test");
    ASSERT_EQ(false, result);
}

// int16
TEST(buff_compare, matching_int16) {
    bool result = validate_buff_cmp_same<int16_t>("matching_int16_test");
    ASSERT_EQ(true, result);
}

TEST(buff_compare, diff_int16) {
    bool result = validate_buff_cmp_diff<int16_t>("diff_int16_test");
    ASSERT_EQ(false, result);
}

// int32
TEST(buff_compare, matching_int32) {
    bool result = validate_buff_cmp_same<int32_t>("matching_int32_test");
    ASSERT_EQ(true, result);
}

TEST(buff_compare, diff_int32) {
    bool result = validate_buff_cmp_diff<int32_t>("diff_int32_test");
    ASSERT_EQ(false, result);
}

// int64
TEST(buff_compare, matching_int64) {
    bool result = validate_buff_cmp_same<int64_t>("matching_int64_test");
    ASSERT_EQ(true, result);
}

TEST(buff_compare, diff_int64) {
    bool result = validate_buff_cmp_diff<int64_t>("diff_int64_test");
    ASSERT_EQ(false, result);
}

/* ============== UINT TESTS =============== */

// uint8
TEST(buff_compare, matching_uint8) {
    bool result = validate_buff_cmp_same<uint8_t>("matching_uint8_test");
    ASSERT_EQ(true, result);
}

TEST(buff_compare, diff_uint8) {
    bool result = validate_buff_cmp_diff<uint8_t>("diff_uint8_test");
    ASSERT_EQ(false, result);
}

// uint16
TEST(buff_compare, matching_uint16) {
    bool result = validate_buff_cmp_same<uint16_t>("matching_uint16_test");
    ASSERT_EQ(true, result);
}

TEST(buff_compare, diff_uint16) {
    bool result = validate_buff_cmp_diff<uint16_t>("diff_uint16_test");
    ASSERT_EQ(false, result);
}

// uint32
TEST(buff_compare, matching_uint32) {
    bool result = validate_buff_cmp_same<uint32_t>("matching_uint32_test");
    ASSERT_EQ(true, result);
}

TEST(buff_compare, diff_uint32) {
    bool result = validate_buff_cmp_diff<uint32_t>("diff_uint32_test");
    ASSERT_EQ(false, result);
}

// uint64
TEST(buff_compare, matching_uint64) {
    bool result = validate_buff_cmp_same<uint64_t>("matching_uint64_test");
    ASSERT_EQ(true, result);
}

TEST(buff_compare, diff_uint64) {
    bool result = validate_buff_cmp_diff<uint64_t>("diff_uint64_test");
    ASSERT_EQ(false, result);
}

/* ============ float tests ============ */

TEST(buff_compare, matching_float32) {
    bool result = validate_buff_cmp_same_fp<float>("matching_fp32_test");
    ASSERT_EQ(true, result);
}

TEST(buff_compare, diff_float32) {
    bool result = validate_buff_cmp_diff_fp<float>("diff_fp32_test");
    ASSERT_EQ(false, result);
}
