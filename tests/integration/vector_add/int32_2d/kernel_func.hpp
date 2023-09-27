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

using namespace gpu::xetla;

template <typename T1, int n1, typename T2, int n2>
KERNEL_FUNC inline void load2D_D32_A64_CA_CA(
        xetla_vector_ref<T1, n1> __REF__ data, xetla_vector<T2, n2> payload) {
    // uses 2D block read
    const uint32_t exDesc = 0;
    constexpr uint8_t numDst = (n1 * sizeof(T1)) / 64;

    uint32_t desc = 3; // load 2D block
    desc |= 0x0 << 7; // disable vnni transform
    desc += 0x2 << 9; // 32bit data
    desc += 0x0 << 15; // transpose off
    desc += 0x4 << 17; // L1 cached L2 cached
    desc += numDst
            << 20; // Response length. 31 represents 32 with hidden -1. The rest doesn't use -1.
    desc += 0x1 << 25; // Msg length
    desc += 0x0 << 29; // flat address

    constexpr uint8_t sfid = 0xf; // ugm
    constexpr uint8_t numSrc0 = 1;
    constexpr uint8_t execSize = 0; // 0=simd1
    xetla_raw_send<T1, n1, T2, n2, execSize, sfid, numSrc0, numDst>(
            data, payload, exDesc, desc);
}

template <typename T1, int n1>
KERNEL_FUNC inline void prefetch2D_D32_A64_CA_CA(xetla_vector<T1, n1> payload) {
    // uses 2D block read
    const uint32_t exDesc = 0;

    uint32_t desc = 3; // load 2D block
    desc |= 0x0 << 7; // disable vnni transform
    desc += 0x2 << 9; // 32bit data
    desc += 0x0 << 15; // transpose off
    desc += 0x4 << 17; // L1 cached L2 cached
    desc += 0
            << 20; // Response length, prefetch=0. 31 represents 32 with hidden -1. The rest doesn't use -1.
    desc += 0x1 << 25; // Msg length
    desc += 0x0 << 29; // flat address

    constexpr uint8_t sfid = 0xf; // ugm
    constexpr uint8_t numSrc0 = 1;
    constexpr uint8_t execSize = 0; // 0=simd1
    xetla_raw_send<T1, n1, execSize, sfid, numSrc0>(payload, exDesc, desc);
}

template <typename T1, int n1, typename T2, int n2>
KERNEL_FUNC inline void store2D_D32_A64_WB_WB(
        xetla_vector<T1, n1> payload, xetla_vector<T2, n2> result) {
    // uses 2D block write
    const uint32_t exDesc = 0;

    uint32_t desc = 7; // store 2D block
    desc += 0x2 << 9; // 32bit data
    desc += 0x7 << 17; // L1 writeback L2 writeback
    desc += 0
            << 20; // Response length. 31 represents 32 with hidden -1. The rest doesn't use -1.
    desc += 0x1 << 25; // Msg length
    desc += 0x0 << 29; // flat address

    constexpr uint8_t sfid = 0xf; // ugm
    constexpr uint8_t numSrc0 = 1;
    constexpr uint8_t numSrc1 = (n2 * sizeof(T2)) / 64;
    constexpr uint8_t execSize = 0; // 0=simd1
    xetla_raw_send<T1, n1, T2, n2, execSize, sfid, numSrc0, numSrc1>(
            payload, result, exDesc, desc);
}

#define TWOD_BUF_ADDR0 0
#define TWOD_BUF_ADDR1 1
#define TWOD_BUF_WIDTH_IN_BYTE 2
#define TWOD_BUF_HEIGHT_IN_ELEM 3
#define TWOD_BUF_STRIDE_IN_BYTE 4
#define TWOD_BUF_X_COORD_IN_ELEM 5
#define TWOD_BUF_Y_COORD_IN_ELEM 6
#define TWOD_BUF_BLOCK_W_H_A_IN_ELEM 7

template <typename dtype, int SIMD, int BLOCK_SIZE>
KERNEL_FUNC inline void vector_add_func(
        xetla_exec_item<1> *ei, dtype *a, dtype *b, dtype *c) {

    //    xetla_vector<uint32_t, 16> A_prefetch_msg;
    //    xetla_vector<uint32_t, 16> B_prefetch_msg;
    xetla_vector<uint32_t, 16> A_load_msg;
    xetla_vector<uint32_t, 16> B_load_msg;
    xetla_vector<uint32_t, 16> C_store_msg;

    xetla_vector<dtype, SIMD * SIMD> A_buffer;
    xetla_vector<dtype, SIMD * SIMD> B_buffer;
    xetla_vector<dtype, SIMD * SIMD> C_buffer;

    xetla_matrix_ref<dtype, SIMD, SIMD> __REF__ A_buffer_2d
            = A_buffer.xetla_format<dtype, SIMD, SIMD>();
    xetla_matrix_ref<dtype, SIMD, SIMD> __REF__ B_buffer_2d
            = B_buffer.xetla_format<dtype, SIMD, SIMD>();
    xetla_matrix_ref<dtype, SIMD, SIMD> __REF__ C_buffer_2d
            = C_buffer.xetla_format<dtype, SIMD, SIMD>();

    xetla_vector<uint32_t, 2> xy_C;
    xetla_vector<uint32_t, 2> xy_B;
    xetla_vector<uint32_t, 2> xy_A;

    // construct matA read and prefetch message
    xy_A[0] = ei->get_group(0) * BLOCK_SIZE;
    xy_A[1] = 0;
    A_load_msg[TWOD_BUF_BLOCK_W_H_A_IN_ELEM] = 0x00707;
    A_load_msg.xetla_select<2, 1>(TWOD_BUF_ADDR0).xetla_format<uint64_t>()
            = (uint64_t)a; // base address
    //hardcode dim for simple test purpose
    A_load_msg[TWOD_BUF_WIDTH_IN_BYTE]
            = BLOCK_SIZE * 8 * sizeof(dtype) - 1; // surface width in byte
    A_load_msg[TWOD_BUF_HEIGHT_IN_ELEM] = BLOCK_SIZE - 1;
    A_load_msg[TWOD_BUF_STRIDE_IN_BYTE]
            = BLOCK_SIZE * 8 * sizeof(dtype) - 1; // surface pitch in byte
    A_load_msg.xetla_select<2, 1>(TWOD_BUF_X_COORD_IN_ELEM)
            = xy_A; // block x start. points to the right most column for more efficiency prefetch of two cache lines per line

    B_load_msg = A_load_msg;
    B_load_msg.xetla_select<2, 1>(TWOD_BUF_ADDR0).xetla_format<uint64_t>()
            = (uint64_t)b; // base address

    C_store_msg = A_load_msg;
    C_store_msg.xetla_select<2, 1>(TWOD_BUF_ADDR0).xetla_format<uint64_t>()
            = (uint64_t)c; // base address

    prefetch2D_D32_A64_CA_CA<uint32_t, 16>(A_load_msg.xetla_format<uint32_t>());
    prefetch2D_D32_A64_CA_CA<uint32_t, 16>(B_load_msg.xetla_format<uint32_t>());

    for (int iy = 0; iy < BLOCK_SIZE; iy += SIMD) {

        A_load_msg[TWOD_BUF_Y_COORD_IN_ELEM] = iy;
        B_load_msg[TWOD_BUF_Y_COORD_IN_ELEM] = iy;
        C_store_msg[TWOD_BUF_Y_COORD_IN_ELEM] = iy;
        for (int ix = 0; ix < BLOCK_SIZE; ix += SIMD) {
            A_load_msg[TWOD_BUF_X_COORD_IN_ELEM]
                    = ei->get_group(0) * BLOCK_SIZE + ix;
            B_load_msg[TWOD_BUF_X_COORD_IN_ELEM]
                    = ei->get_group(0) * BLOCK_SIZE + ix;
            C_store_msg[TWOD_BUF_X_COORD_IN_ELEM]
                    = ei->get_group(0) * BLOCK_SIZE + ix;
            load2D_D32_A64_CA_CA<dtype, SIMD * SIMD, uint32_t, 16>(
                    A_buffer.xetla_format<dtype>(),
                    A_load_msg.xetla_format<uint32_t>());

            load2D_D32_A64_CA_CA<dtype, SIMD * SIMD, uint32_t, 16>(
                    B_buffer.xetla_format<dtype>(),
                    B_load_msg.xetla_format<uint32_t>());

            for (int irow = 0; irow < SIMD; irow++) {
                C_buffer_2d.row(irow)
                        = A_buffer_2d.row(irow) + B_buffer_2d.row(irow);
            }

            store2D_D32_A64_WB_WB<uint32_t, 16, dtype, SIMD * SIMD>(
                    C_store_msg, C_buffer);
        }
    }
}
