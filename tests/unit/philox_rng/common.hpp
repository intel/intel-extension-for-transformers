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

struct val_64b_t {
    uint32_t u32[2];
    val_64b_t(const val_64b_t &val) {
        this->u32[0] = val.u32[0];
        this->u32[1] = val.u32[1];
    }
    val_64b_t() = default;
    ~val_64b_t() {}
    val_64b_t &operator=(const val_64b_t &val) {
        if (this != &val) {
            this->u32[0] = val.u32[0];
            this->u32[1] = val.u32[1];
        }
        return *this;
    }
};
struct val_128b_t {
    uint32_t u32[4];
    val_128b_t(const val_128b_t &val) {
        this->u32[0] = val.u32[0];
        this->u32[1] = val.u32[1];
        this->u32[2] = val.u32[2];
        this->u32[3] = val.u32[3];
    }
    val_128b_t() = default;
    ~val_128b_t() {}
    val_128b_t &operator=(const val_128b_t &val) {
        if (this != &val) {
            this->u32[0] = val.u32[0];
            this->u32[1] = val.u32[1];
            this->u32[2] = val.u32[2];
            this->u32[3] = val.u32[3];
        }
        return *this;
    }
};

template <uint32_t round = 7>
class philox_t {
public:
    philox_t(uint64_t seed, uint64_t subseq, uint64_t offset) {
        key.u32[0] = static_cast<uint32_t>(seed);
        key.u32[1] = static_cast<uint32_t>(seed >> 32);

        counter.u32[0] = static_cast<uint32_t>(offset);
        counter.u32[1] = static_cast<uint32_t>(offset >> 32);
        counter.u32[2] = static_cast<uint32_t>(subseq);
        counter.u32[3] = static_cast<uint32_t>(subseq >> 32);
    }
    val_128b_t operator()() {
        val_64b_t key_ = key;
        val_128b_t counter_ = counter;
        for (int i = 0; i < round; i++) {
            counter_ = single_round(counter_, key_);
            key_.u32[0] += kPhilox10A;
            key_.u32[1] += kPhilox10B;
        }
        val_128b_t out = single_round(counter_, key_);
        incr();
        return out;
    }

private:
    val_64b_t key;
    val_128b_t counter;
    static constexpr uint32_t kPhilox10A = 0x9E3779B9;
    static constexpr uint32_t kPhilox10B = 0xBB67AE85;
    static constexpr uint32_t kPhiloxSA = 0xD2511F53;
    static constexpr uint32_t kPhiloxSB = 0xCD9E8D57;

    val_128b_t single_round(val_128b_t counter_, val_64b_t key_) {
        val_64b_t res0;
        val_64b_t res1;
        uint64_t res0_tmp = (uint64_t)counter_.u32[0] * kPhiloxSA;
        uint64_t res1_tmp = (uint64_t)counter_.u32[2] * kPhiloxSB;

        res0.u32[0] = static_cast<uint32_t>(res0_tmp);
        res0.u32[1] = static_cast<uint32_t>(res0_tmp >> 32);
        res1.u32[0] = static_cast<uint32_t>(res1_tmp);
        res1.u32[1] = static_cast<uint32_t>(res1_tmp >> 32);

        val_128b_t ret;
        ret.u32[0] = res1.u32[1] ^ counter_.u32[1] ^ key_.u32[0];
        ret.u32[1] = res1.u32[0];
        ret.u32[2] = res0.u32[1] ^ counter_.u32[3] ^ key_.u32[1];
        ret.u32[3] = res0.u32[0];
        return ret;
    }
    void incr() {
        if (++counter.u32[0]) { return; }
        if (++counter.u32[1]) { return; }
        if (++counter.u32[2]) { return; }
        ++counter.u32[3];
    }
};

int rand_result_validate(uint32_t *A, uint32_t *B, uint32_t *C, unsigned Size) {
    std::vector<uint32_t> ret0(Size * 4, 0);
    std::vector<uint32_t> ret1(Size * 4, 0);
    for (int i = 0; i < Size; i++) {
        philox_t rand(A[3], i, A[5]);
        val_128b_t out0 = rand();
        val_128b_t out1 = rand();
        for (int j = 0; j < 4; j++) {
            ret0[i + j * Size] = out0.u32[j];
            ret1[i + j * Size] = out1.u32[j];
        }
    }
    int err_cnt = 0;
    for (unsigned i = 0; i < Size * 4; ++i) {
        if (ret0[i] != B[i]) {
            if (++err_cnt < 128) {
                std::cout << "ret0 failed at index " << i << ", " << ret0[i]
                          << " != " << B[i] << "\n";
            }
        }
        if (ret1[i] != C[i]) {
            if (++err_cnt < 128) {
                std::cout << "ret1 failed at index " << i << ", " << ret1[i]
                          << " != " << C[i] << "\n";
            }
        }
    }
    if (err_cnt > 0) {
        std::cout << "pass rate: "
                  << ((float)(Size * 8 - err_cnt) / ((float)Size * 8)) * 100.0f
                  << "% (" << (Size * 8 - err_cnt) << "/" << Size * 8 << ")\n";
    }
    std::cout << (err_cnt > 0 ? "FAILED\n" : "PASSED\n");
    return err_cnt;
}
