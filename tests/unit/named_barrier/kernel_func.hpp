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

template <typename dtype, int SIMD, gpu_arch arch_tag = gpu_arch::Xe>
struct named_barrier_func {
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(dtype);

        xetla_nbarrier_t<16, 16, arch_tag> nbarrier;
        nbarrier.init_nbarrier(0, nbarrier_role::producer_consumer);
        nbarrier.arrive();
        nbarrier.wait();
#pragma unroll
        for (int i = 0; i < 16; i++) {
            if (item->get_local_id(0) == i) {
                xetla_vector<dtype, SIMD> A_load_vec
                        = xetla_load_global(a, offsets);
                xetla_vector<dtype, SIMD> Dst = A_load_vec * 2;
                xetla_store_global(c, offsets, A_load_vec);
                xetla_store_global(a, offsets, Dst);
            }
            nbarrier.arrive();
            nbarrier.wait();
        }
    }
};
template <typename dtype, int SIMD, gpu_arch arch_tag = gpu_arch::Xe>
struct named_barrier_producer_consumer_1_func {
    // 2 producer and 2 consumer threads
    // only one named barrier used
    // tidX=2,3 are producers, reads original data , multiplies by 2 and writes to SLM
    // tidX=0,1 are consumers, reads multiplied data from SLM and writes to output buffer
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {
        xetla_nbarrier_t<2, 2, arch_tag> nbarrier;
        auto nelem_per_thread = SIMD;
        auto tidX = item->get_local_id(0);

        if (tidX >= 2) {
            auto read_offset = (tidX - 2) * nelem_per_thread * sizeof(dtype);
            xetla_vector<dtype, SIMD> A_load_vec
                    = xetla_load_global<dtype, SIMD>(a, read_offset);
            A_load_vec = A_load_vec * 2;
            xetla_store_local<dtype, SIMD>(read_offset, A_load_vec);
            // Producer only signal
            nbarrier.init_nbarrier(0, nbarrier_role::producer);
            nbarrier.arrive();
        } else {
            auto write_offset = tidX * nelem_per_thread * sizeof(dtype);
            // Consumer only signal
            nbarrier.init_nbarrier(0, nbarrier_role::consumer);
            nbarrier.arrive_wait();
            xetla_vector<dtype, SIMD> slm_vec
                    = xetla_load_local<dtype, SIMD>(write_offset);
            xetla_store_global<dtype, SIMD>(c, write_offset, slm_vec);
        }
    }
};

template <typename dtype, int SIMD, gpu_arch arch_tag = gpu_arch::Xe>
struct named_barrier_producer_consumer_2_func {
    // 32 threads in workgroup
    // 16 producer threads, 16 consumer threads
    // 16 named barriers used
    // tidX=16..31 are producers, reads original data , multiplies by 2 and writes to SLM
    // tidX=0..15 are consumers, reads multiplied data from SLM and writes to output buffer
    // only 1 producer , 1 consumer per named barrier
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {

        xetla_nbarrier_t<1, 1, arch_tag> nbarrier;
        auto nelem_per_thread = SIMD;
        auto tidX = item->get_local_id(0);
        auto barrier_id = tidX % 16;

        if (tidX >= 16) {
            auto read_offset = (tidX - 16) * nelem_per_thread * sizeof(dtype);
            xetla_vector<dtype, SIMD> A_load_vec
                    = xetla_load_global<dtype, SIMD>(a, read_offset);
            A_load_vec = A_load_vec * 2;
            xetla_store_local<dtype, SIMD>(read_offset, A_load_vec);
            // Producer only signal
            nbarrier.init_nbarrier(barrier_id, nbarrier_role::producer);
            nbarrier.arrive();
        } else {

            auto write_offset = tidX * nelem_per_thread * sizeof(dtype);
            // Consumer only signal
            nbarrier.init_nbarrier(barrier_id, nbarrier_role::consumer);
            nbarrier.arrive_wait();
            xetla_vector<dtype, SIMD> slm_vec
                    = xetla_load_local<dtype, SIMD>(write_offset);
            xetla_store_global<dtype, SIMD>(c, write_offset, slm_vec);
        }
    }
};

template <typename dtype, int SIMD, gpu_arch arch_tag = gpu_arch::Xe>
struct named_barrier_producer_consumer_3_func {
    // 16 threads in workgroup
    // 8 producer threads, 8 consumer threads
    // 16 named barriers used, only 1 producer , 1 consumer per named barrier, each named barrier is used multiple times
    // tidX=0..7  reads original data , multiplies by 2 writes to SLM, do a barrier wait, then multiplies by 2 again, writes to SLM
    // tidX=8..15 wait for (tidX%8) thread finish the first write, reads multiplied data from SLM ,wait again and do another read, then add the two vector, and writes to output buffer
    static KERNEL_FUNC inline void run(
            sycl::nd_item<1> *item, dtype *a, dtype *b, dtype *c) {

        xetla_nbarrier_t<1, 1, arch_tag> nbarrier1;
        xetla_nbarrier_t<1, 1, arch_tag> nbarrier2;

        auto nelem_per_thread = SIMD;
        auto tidX = item->get_local_id(0);
        auto barrier_id_1 = tidX % 8;
        auto barrier_id_2 = tidX % 8 + 8;

        if (tidX < 8) {
            nbarrier1.init_nbarrier(barrier_id_1, nbarrier_role::producer);
            nbarrier2.init_nbarrier(barrier_id_2, nbarrier_role::consumer);
            auto read_offset = tidX * nelem_per_thread * sizeof(dtype);
            xetla_vector<dtype, SIMD> A_load_vec
                    = xetla_load_global<dtype, SIMD>(a, read_offset);
            A_load_vec = A_load_vec * 2;
            xetla_store_local<dtype, SIMD>(read_offset, A_load_vec);
            // signal to th#tid+8
            nbarrier1.arrive();
            // Wait signal from th#tid+8
            nbarrier2.arrive_wait();

            A_load_vec = A_load_vec * 2;
            xetla_store_local<dtype, SIMD>(read_offset, A_load_vec);
            // signal to th#tid+8
            nbarrier1.arrive();
        } else {
            nbarrier1.init_nbarrier(barrier_id_1, nbarrier_role::consumer);
            nbarrier2.init_nbarrier(barrier_id_2, nbarrier_role::producer);

            // Wait signal from th#tid-8
            nbarrier1.arrive_wait();

            auto write_offset = (tidX % 8) * nelem_per_thread * sizeof(dtype);
            xetla_vector<dtype, SIMD> slm_vec
                    = xetla_load_local<dtype, SIMD>(write_offset);

            // signal to th#tid-8
            nbarrier2.arrive();
            // Wait signal from th#tid-8
            nbarrier1.arrive_wait();

            xetla_vector<dtype, SIMD> slm_vec2
                    = xetla_load_local<dtype, SIMD>(write_offset);

            slm_vec = slm_vec + slm_vec2;

            xetla_store_global<dtype, SIMD>(c, write_offset, slm_vec);
        }
    }
};
