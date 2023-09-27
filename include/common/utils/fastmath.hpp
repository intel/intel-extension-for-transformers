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

/// @file
/// C++ API

#pragma once

#include "common/utils/common.hpp"

namespace gpu::xetla {

///@brief Host side utility function to compute number of leading zeros in the binary representation
inline int clz(int x) {

    for (int i = 31; i >= 0; i--) {

        if ((1 << i) & x) { return (31 - i); }
    }
    return 32;
}

///@brief Host side utility function to compute log2 function
inline int find_log2(int x) {

    int a = int(31 - clz(x));
    a = a + ((x & (x - 1)) != 0);
    return a;
}

///Fast division + modulus operation
///Host code pre-computes values to avoid expensive operations in kernel code
struct FastDivMod {

    int divisor;
    unsigned int multiplier;
    unsigned int shift_right;

    ///Constructor, called in Hostcode to pre-compute multiplier and shift_right;
    inline FastDivMod() : divisor(0), multiplier(0), shift_right(0) {}

    inline FastDivMod(int divisor_) : divisor(divisor_) {

        if (divisor != 1) {
            unsigned int p = 31 + find_log2(divisor);
            unsigned m = unsigned(
                    ((1ull << p) + unsigned(divisor) - 1) / unsigned(divisor));

            multiplier = m;
            shift_right = p - 32;
        } else {

            multiplier = 0;
            shift_right = 0;
        }
    }

    operator int() const { return divisor; }

    ///@brief Kernel side function to find quotient and remainder
    __XETLA_API KERNEL_FUNC void fast_divmod(
            int &quotient, int &remainder, int dividend) const {

        int lo_part;
        quotient = (divisor != 1)
                ? __ESIMD_ENS::imul(lo_part, dividend, multiplier)
                        >> shift_right
                : dividend;

        remainder = dividend - (quotient * divisor);
    }

    ///@brief kernel side utility functions for query of quotient
    __XETLA_API KERNEL_FUNC int div(int dividend) const {

        int quotient, remainder;
        fast_divmod(quotient, remainder, dividend);
        return quotient;
    }
};

} // namespace gpu::xetla