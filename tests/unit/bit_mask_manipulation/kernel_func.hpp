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

template <typename T, int SIMD, typename T_op>
static inline void run_bit_shift_op_common(
        xetla_exec_item<1> *ei, T *a, T *b, T *c, T_op op) {
    uint64_t offset = sizeof(T) * SIMD * ei->get_group(0);
    xetla_vector<uint32_t, SIMD> offsets = xetla_vector_gen<T, SIMD>(0, 1);
    offsets *= sizeof(T);
    offsets += offset;

    xetla_vector<T, SIMD> A_load_vec = xetla_load_global<T, SIMD>(a, offset);
    xetla_vector<T, SIMD> result = op(A_load_vec, 1);

    xetla_store_global(c, offsets, result);
}

template <typename T, int SIMD>
struct shl_with_vector_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            xetla_saturation_off_tag tag;
            return xetla_shl<T, T>(x, bit, tag);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct shl_with_scalar_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            xetla_saturation_off_tag tag;
            T x0 = x[0];
            T result = xetla_shl<T>(x0, bit, tag);
            return xetla_vector_gen<T, SIMD>(result, 0);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct shr_with_vector_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            xetla_saturation_off_tag tag;
            return xetla_shr<T, T>(x, bit, tag);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct shr_with_scalar_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            xetla_saturation_off_tag tag;
            T x0 = x[0];
            T result = xetla_shr<T>(x0, bit, tag);
            return xetla_vector_gen<T, SIMD>(result, 0);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct rol_with_2_vector_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            return xetla_rol<T, T, SIMD>(x, xetla_vector<T, SIMD>(bit));
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct rol_with_a_vector_and_a_scalar_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            return xetla_rol<T, T>(x, bit);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct rol_with_2_scalar_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            T x0 = x[0];
            return xetla_rol<T>(x0, bit);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct ror_with_2_vector_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            return xetla_ror<T, T, SIMD>(x, xetla_vector<T, SIMD>(bit));
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct ror_with_a_vector_and_a_scalar_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            return xetla_ror<T, T>(x, bit);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct ror_with_2_scalar_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            T x0 = x[0];
            return xetla_ror<T>(x0, bit);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct lsr_with_vector_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            xetla_saturation_off_tag tag;
            return xetla_lsr<T, T>(x, bit, tag);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct lsr_with_scalar_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            xetla_saturation_off_tag tag;
            T x0 = x[0];
            T result = xetla_lsr<T>(x0, bit, tag);
            return xetla_vector_gen<T, SIMD>(result, 0);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct asr_with_vector_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            xetla_saturation_off_tag tag;
            return xetla_asr<T, T>(x, bit, tag);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};

template <typename T, int SIMD>
struct asr_with_scalar_input {
    static KERNEL_FUNC inline void run(
            xetla_exec_item<1> *ei, T *a, T *b, T *c) {
        auto op = [](xetla_vector<T, SIMD> x, int bit) {
            xetla_saturation_off_tag tag;
            T x0 = x[0];
            T result = xetla_asr<T>(x0, bit, tag);
            return xetla_vector_gen<T, SIMD>(result, 0);
        };
        run_bit_shift_op_common<T, SIMD>(ei, a, b, c, op);
    }
};
