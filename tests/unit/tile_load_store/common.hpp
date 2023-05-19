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

#include "kernel_func.hpp"

template <typename dtype, bool transform = false, bool transpose = false>
int tile_load_store_result_validate(dtype *A, dtype *B, dtype *C,
        unsigned Sizex, unsigned Sizey, unsigned Blockx, unsigned Blocky) {
    int err_cnt = 0;
    int a_index, c_index;
    int ele_per_dw = transform ? sizeof(uint32_t) / sizeof(dtype) : 1;
    for (unsigned i = 0; i < Blocky; ++i) {
        for (unsigned j = 0; j < Blockx; ++j) {
            a_index = transpose ? j * Sizey + i : i * Sizex + j;
            c_index = i / ele_per_dw * Sizex * ele_per_dw + j * ele_per_dw
                    + i % ele_per_dw;

            if (A[a_index] != C[c_index]) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index " << i * Blockx + j << ", "
                              << C[c_index] << " != " << A[a_index] << "\n";
                }
            }
        }
    }

    unsigned Size = Blockx * Blocky;
    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
                  << (Size - err_cnt) << "/" << Size << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}

template <typename dtype>
int tile_load_broadcase_store_result_validate(dtype *A, dtype *B, dtype *C,
        unsigned Sizex, unsigned Blockx, unsigned Blocky) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Blocky; ++i) {
        for (unsigned j = 0; j < Blockx; ++j) {
            if (A[j] != C[i * Sizex + j]) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index " << i * Blockx + j << ", "
                              << C[i * Sizex + j] << " != " << A[i * Sizex + j]
                              << "\n";
                }
            }
        }
    }

    unsigned Size = Blockx * Blocky;
    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
                  << (Size - err_cnt) << "/" << Size << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}

template <typename dtype>
int tile_padding_load_store_result_validate(dtype *A, dtype *B, dtype *C,
        unsigned Sizex, unsigned Sizey, unsigned Startx, unsigned Starty) {
    int err_cnt = 0;
    int a_index, c_index;
    int pwidth;
    for (unsigned i = 0; i < Sizey; ++i) {
        for (unsigned j = 0; j < Sizex; ++j) {
            c_index = i * Sizex + j;
            int a_index_x = j + Startx;
            int a_index_y = i + Starty;

            dtype c_temp = C[c_index];
            dtype a_temp;
            if ((a_index_x < 0) || (a_index_x >= Sizex) || (a_index_y < 0)
                    || (a_index_y >= Sizey)) {
                a_temp = 0;
            } else {
                a_temp = A[a_index_y * Sizex + a_index_x];
            }

            if (a_temp != c_temp) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index [" << i << ", " << j
                              << "]: " << c_temp << " != " << a_temp << "\n";
                }
            }
        }
    }
    int Size = Sizex * Sizey;
    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
                  << (Size - err_cnt) << "/" << Size << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
