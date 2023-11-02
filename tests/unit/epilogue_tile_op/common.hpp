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

template <typename dtype>
dtype tanh_cpu(dtype x) {
    dtype exp2x = std::exp(x * 2.f);
    dtype ret = (exp2x - 1.f) / (exp2x + 1.f);
    return (x >= 10) ? 1 : ret;
}
template <typename dtype>
struct relu_op {
    static dtype run(dtype x) { return (x > 0) ? x : 0; }
};

template <typename dtype>
struct gelu_op {
    static_assert((std::is_same<remove_const_t<dtype>, float>::value),
            "only use float! ");
    static dtype run(dtype x) {
        constexpr dtype C0 = 0.044715f;
        constexpr dtype sqrt_two_over_pi = 0.79788458347320556640625f;
        dtype input_x = sqrt_two_over_pi * x * (1.f + C0 * x * x);
        dtype tanh_value = tanh_cpu(input_x);
        dtype result = (0.5f * x * (1.f + tanh_value));
        return result;
    }
};

template <typename dtype>
struct gelu_fwd_w_op {
    static_assert((std::is_same<remove_const_t<dtype>, float>::value),
            "only use float! ");
    static dtype run(dtype x) {
        constexpr float C0 = 0.044715f;
        constexpr float D0 = 0.134145f;
        constexpr float sqrt_two_over_pi = 0.79788458347320556640625f;
        float input_x = sqrt_two_over_pi * x * (1.f + C0 * x * x);
        float z = tanh_cpu(input_x);
        float result = 0.5f * (1 + z)
                + 0.5f * x * (1.f - z * z)
                        * (sqrt_two_over_pi * (1.f + D0 * x * x));
        return result;
    }
};

template <typename dtype>
struct gelu_bwd_op {
    static_assert((std::is_same<remove_const_t<dtype>, float>::value),
            "only use float! ");
    static dtype run(dtype x, dtype y) {
        constexpr float C0 = 0.044715f;
        constexpr float D0 = 0.134145f;
        constexpr float sqrt_two_over_pi = 0.79788458347320556640625f;
        float input_x = sqrt_two_over_pi * x * (1.f + C0 * x * x);
        float z = tanh_cpu(input_x);
        float result = 0.5f * (1 + z)
                + 0.5f * x * (1.f - z * z)
                        * (sqrt_two_over_pi * (1.f + D0 * x * x));
        return result * y;
    }
};

template <typename dtype, typename op>
int tile_elemwise_op_validate(dtype *A, dtype *B, dtype *C, unsigned Sizex,
        unsigned Blockx, unsigned Blocky) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Blocky; ++i) {
        for (unsigned j = 0; j < Blockx; ++j) {
            dtype golden = op::run((A[i * Sizex + j] - 50.f) / 100.f);
            if ((golden - C[i * Sizex + j]) / (golden + 0.000001) > 0.001) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index " << i * Blockx + j << ", "
                              << C[i * Sizex + j] << " != " << golden << "\n";
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
int tile_elemwise_bias_add_validate(dtype *A, dtype *B, dtype *C,
        unsigned Sizex, unsigned Blockx, unsigned Blocky) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Blocky; ++i) {
        for (unsigned j = 0; j < Blockx; ++j) {
            dtype golden = A[j];
            if ((golden - C[i * Sizex + j]) / (golden + 0.000001) > 0.001) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index " << i * Blockx + j << ", "
                              << C[i * Sizex + j] << " != " << golden << "\n";
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
int tile_elemwise_gelu_bwd_validate(dtype *A, dtype *B, dtype *C,
        unsigned Sizex, unsigned Blockx, unsigned Blocky) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Blocky; ++i) {
        for (unsigned j = 0; j < Blockx; ++j) {
            dtype golden = gelu_bwd_op<dtype>::run(
                    B[i * Sizex + j], (A[i * Sizex + j] - 50.f) / 100.f);
            if ((golden - C[i * Sizex + j]) / (golden + 0.000001) > 0.001) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index " << i * Blockx + j << ", "
                              << C[i * Sizex + j] << " != " << golden << "\n";
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
int tile_elemwise_res_add_validate(dtype *A, dtype *B, dtype *C, unsigned Sizex,
        unsigned Blockx, unsigned Blocky) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Blocky; ++i) {
        for (unsigned j = 0; j < Blockx; ++j) {
            dtype golden = A[i * Sizex + j];
            if ((golden - C[i * Sizex + j]) / (golden + 0.000001) > 0.001) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index " << i * Blockx + j << ", "
                              << C[i * Sizex + j] << " != " << golden << "\n";
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
int tile_elemwise_linear_op_validate(dtype *A, dtype *B, dtype *C,
        unsigned Sizex, unsigned Blockx, unsigned Blocky) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Blocky; ++i) {
        for (unsigned j = 0; j < Blockx; ++j) {
            dtype golden = A[i * Sizex + j] * 4;
            if (abs(golden - C[i * Sizex + j]) / (golden + 0.000001) > 0.001) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index " << i * Blockx + j << ", "
                              << C[i * Sizex + j] << " != " << golden << "\n";
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