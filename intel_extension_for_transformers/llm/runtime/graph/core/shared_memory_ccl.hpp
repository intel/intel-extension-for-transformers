//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#include <assert.h>
#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include "oneapi/ccl.hpp"

// states for collectives
enum ccl_state {
  ccl_begin = 0,
  copy_in_done,
  reduce_done,
  copy_out_done,
};

void* shared_open(const char* name, size_t nbytes) {
  int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    return mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
  } else {
    printf("shared_open %s failed\n", name);
    return nullptr;
  }
}

void* shared_create(const char* name, void* bytes, size_t nbytes) {
  int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if ((d != -1) && (nbytes = write(d, bytes, nbytes))) {
    return mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
  } else {
    printf("shared_create %s failed\n", name);
    return nullptr;
  }
}

void shared_close(const char* name, void* bytes, size_t nbytes) {
  int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    munmap(bytes, nbytes);
    shm_unlink(name);
  }
}

#define MAX_BUF_SIZE 1048576 * 4
struct ccl_buffer {
  enum ccl_state state;
  char data[MAX_BUF_SIZE];
};
struct ccl_buffer* cbuffer;

void wait_state_equal(int index, enum ccl_state state) {
  volatile enum ccl_state* state_ptr = &(cbuffer[index].state);
  while (*state_ptr != state)
    ;
}

void wait_state_change(int index, enum ccl_state state) {
  volatile enum ccl_state* state_ptr = &(cbuffer[index].state);
  while (*state_ptr == state)
    ;
}

void reduce_2_fp32_buffers(int num_elements, void* rank_0, void* rank_1) __attribute__((target("avx512bw")));

void reduce_fp32_buffers(int num_elements, int num_buffers, struct ccl_buffer* cbuffer)
    __attribute__((target("avx512bw")));

// N_REDUCE_LIMIT is the number of buffers that can be reduced together in one shot.
// Compared with do N-1 2-reduces which needs 2*(N-1) read and N-1 write,
// N-reduce only needs N read and 1 write, this saves 2/3 memory bandwidth.
// When increase N_REDUCE_LIMIT to a bigger number, do the following steps
// 1. Extend REPEAT_<X> macros list down below
// 2. Extend switch cases which call "REPEAT(X, ...)" down below
#define N_REDUCE_LIMIT 8

void reduce_buffers(struct ccl_buffer* cbuffer, int num_elements, int num_buffers) {
  if (num_buffers == 2) {
    reduce_2_fp32_buffers(num_elements, cbuffer[0].data, cbuffer[1].data);
  } else if (num_buffers > 2 && num_buffers <= N_REDUCE_LIMIT) {
    reduce_fp32_buffers(num_elements, num_buffers, cbuffer);
  } else {
    assert(!"Not supported buffer number.");
  }
}

#define REPEAT(N, x) REPEAT_##N(x)
#define REPEAT_1(x) x(1)
#define REPEAT_2(x) \
  REPEAT_1(x);      \
  x(2)
#define REPEAT_3(x) \
  REPEAT_2(x);      \
  x(3)
#define REPEAT_4(x) \
  REPEAT_3(x);      \
  x(4)
#define REPEAT_5(x) \
  REPEAT_4(x);      \
  x(5)
#define REPEAT_6(x) \
  REPEAT_5(x);      \
  x(6)
#define REPEAT_7(x) \
  REPEAT_6(x);      \
  x(7)

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length.  256bit vector ==> 32 bytes, 512bit vector ==> 64 bytes
#define VECTOR_LENGTH_IN_BYTES 32

#define REDUCE_ADD_F32(x)                                              \
  do {                                                                 \
    auto in##x##_val = _mm256_loadu_ps((float*)(cbuffer[x].data + i)); \
    inout_val = _mm256_add_ps(inout_val, in##x##_val);                 \
  } while (0)

void reduce_fp32_buffers(int num_elements, int num_buffers, struct ccl_buffer* cbuffer) {
  // For vector reduce add
  assert(num_elements % 16 == 0);
#pragma omp parallel for
  for (int i = 0; i < num_elements * sizeof(float); i += VECTOR_LENGTH_IN_BYTES) {
    auto inout_val = _mm256_loadu_ps((float*)(cbuffer[0].data + i));
    switch (num_buffers) {
      case 8:
        REPEAT(7, REDUCE_ADD_F32);
        break;
      case 7:
        REPEAT(6, REDUCE_ADD_F32);
        break;
      case 6:
        REPEAT(5, REDUCE_ADD_F32);
        break;
      case 5:
        REPEAT(4, REDUCE_ADD_F32);
        break;
      case 4:
        REPEAT(3, REDUCE_ADD_F32);
        break;
      case 3:
        REPEAT(2, REDUCE_ADD_F32);
        break;
      default:
        assert(!"Should not get here.");
    }
    _mm256_storeu_ps((float*)(cbuffer[0].data + i), inout_val);
  }
}

void reduce_2_fp32_buffers(int num_elements, void* rank_0, void* rank_1) {
#pragma omp parallel for
  for (int i = 0; i < num_elements * sizeof(float); i += VECTOR_LENGTH_IN_BYTES) {
    auto rank_0_val = _mm256_loadu_ps((float*)((char*)rank_0 + i));
    auto rank_1_val = _mm256_loadu_ps((float*)((char*)rank_1 + i));
    rank_0_val = _mm256_add_ps(rank_0_val, rank_1_val);
    _mm256_storeu_ps((float*)((char*)rank_0 + i), rank_0_val);
  }
}

static void parallel_memcpy(void* to, void* from, size_t n_bytes) __attribute__((target("avx512bw")));
static void parallel_memcpy(void* to, void* from, size_t n_bytes) {
#pragma omp parallel for
  for (int i = 0; i < n_bytes; i += VECTOR_LENGTH_IN_BYTES) {
    auto val = _mm256_loadu_si256((__m256i*)((char*)from + i));
    _mm256_storeu_si256((__m256i*)((char*)to + i), val);
  }
}

void shm_all_reduce(float* sendBuf, float* recvBuf, size_t count, size_t rank, size_t world_size) {
  for (int offset = 0; offset < count * sizeof(float); offset += MAX_BUF_SIZE) {
    auto send_ptr = (char*)sendBuf + offset;
    auto recv_ptr = (char*)recvBuf + offset;
    size_t chunk_size = count * sizeof(float) - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : count * sizeof(float) - offset;
    size_t chunk_count = chunk_size / sizeof(float);

    parallel_memcpy(cbuffer[rank].data, send_ptr, chunk_size);
    std::atomic_thread_fence(std::memory_order_release);
    cbuffer[rank].state = copy_in_done;

    if (rank == 0) {
      // compute allreduce result on rank 0
      for (int i = 1; i < world_size; i++) {
        // wait until the other rank copy the buffer
        wait_state_equal(i, copy_in_done);
      }
      reduce_buffers(cbuffer, chunk_count, world_size);
      std::atomic_thread_fence(std::memory_order_release);
      cbuffer[rank].state = reduce_done;
      parallel_memcpy(recv_ptr, cbuffer[0].data, chunk_size);
    }
    if (rank != 0) {
      wait_state_equal(0, reduce_done);
      parallel_memcpy(recv_ptr, cbuffer[0].data, chunk_size);
      std::atomic_thread_fence(std::memory_order_release);
      cbuffer[rank].state = copy_out_done;
    }
    if (rank == 0) {
      for (int i = 1; i < world_size; i++) {
        wait_state_equal(i, copy_out_done);
      }
      std::atomic_thread_fence(std::memory_order_release);
      cbuffer[rank].state = ccl_begin;
    }
    if (rank != 0) {
      // if rank 0 spin too fast it could be in state 1 of next allreduce
      // in this case wait_state_change(0, 0) may cause deadlock
      // what we are certain is when rank 0 finishes the state won't be 2
      wait_state_change(0, reduce_done);
      cbuffer[rank].state = ccl_begin;
    }
  }
}
