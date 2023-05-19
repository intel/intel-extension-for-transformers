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

using namespace std::placeholders;
using namespace gpu;
using namespace gpu::xetla;

template <typename datatype>
int load_store_result_validate(
        datatype *A, datatype *B, datatype *C, unsigned Size) {
    int err_cnt = 0;
    for (unsigned i = 0; i < Size; ++i) {
        if (A[i] != B[i]) {
            if (++err_cnt < 10) {
                std::cout << "failed at index " << i << ", " << B[i]
                          << " != " << A[i] << "\n";
            }
        }
    }
    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
                  << (Size - err_cnt) << "/" << Size << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}

struct None {
    static constexpr std::string_view name = "cache_hint::none";
    static constexpr cache_hint value = cache_hint::none;
};
struct CA {
    static constexpr std::string_view name = "cache_hint::cached";
    static constexpr cache_hint value = cache_hint::cached;
};
struct UC {
    static constexpr std::string_view name = "cache_hint::uncached";
    static constexpr cache_hint value = cache_hint::uncached;
};
struct WB {
    static constexpr std::string_view name = "cache_hint::write_back";
    static constexpr cache_hint value = cache_hint::write_back;
};
struct WT {
    static constexpr std::string_view name = "cache_hint::write_through";
    static constexpr cache_hint value = cache_hint::write_through;
};
struct ST {
    static constexpr std::string_view name = "cache_hint::streaming";
    static constexpr cache_hint value = cache_hint::streaming;
};
struct IAR {
    static constexpr std::string_view name = "cache_hint::read_invalidate";
    static constexpr cache_hint value = cache_hint::read_invalidate;
};
