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
#pragma once
#ifdef NE_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef NE_BUILD
#define NE_API __declspec(dllexport)
#else
#define NE_API __declspec(dllimport)
#endif
#else
#define NE_API __attribute__((visibility("default")))
#endif
#else
#define NE_API
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "core/data_types.h"
#include "layers/Ops.h"

#define NE_FILE_MAGIC 0x67676d6c  // "ne"
#define NE_FILE_VERSION 1

#define NE_MAX_DIMS 4
#define NE_MAX_NODES 4096
#define NE_MAX_PARAMS 256
#define NE_MAX_CONTEXTS 64
#define NE_MAX_OPT 4
#define NE_DEFAULT_N_THREADS 4

#define NE_SIZE_CALC -1

#ifdef __cplusplus
extern "C" {
#endif

struct ne_object;
struct ne_context;

enum ne_backend {
  NE_BACKEND_CPU = 0,
  NE_BACKEND_CUDA = 1,
};

// ne object
struct ne_object {
  size_t offs;
  size_t size;

  struct ne_object* next;

  char padding[8];
};

static const size_t NE_OBJECT_SIZE = sizeof(struct ne_object);

// scratch buffer
struct ne_scratch {
  size_t offs;
  size_t size;
  void* data;
};

//
// ne context
//

struct ne_context {
  size_t mem_size;
  void* mem_buffer;
  bool mem_buffer_owned;
  bool no_alloc;

  int n_objects;

  struct ne_object* objects_begin;
  struct ne_object* objects_end;

  struct ne_scratch scratch;
  struct ne_scratch scratch_save;
};

struct ne_context_container {
  bool used;

  struct ne_context context;
};

// n-dimensional tensor
struct ne_tensor {
  enum ne_type type;
  enum ne_backend backend;

  int n_dims;
  int64_t ne[NE_MAX_DIMS];  // number of elements
  size_t nb[NE_MAX_DIMS];   // stride in bytes:
                            // nb[0] = sizeof(type)
                            // nb[1] = nb[0]   * ne[0] + padding
                            // nb[i] = nb[i-1] * ne[i-1]

  // compute data
  enum ne_op op;

  bool is_param;

  struct ne_tensor* grad;
  struct ne_tensor* src0;
  struct ne_tensor* src1;
  struct ne_tensor* opt[NE_MAX_OPT];

  // thread scheduling
  int n_tasks;

  // performance
  int perf_runs;
  int64_t perf_cycles;
  int64_t perf_time_us;

  void* data;
  size_t size;

  char name[32];

  char padding[8];
};

// computation graph
struct ne_cgraph {
  int n_nodes;
  int n_leafs;
  int n_threads;

  size_t work_size;
  struct ne_tensor* work;

  struct ne_tensor* nodes[NE_MAX_NODES];
  struct ne_tensor* grads[NE_MAX_NODES];
  struct ne_tensor* leafs[NE_MAX_NODES];

  // performance
  int perf_runs;
  int64_t perf_cycles;
  int64_t perf_time_us;
};

struct ne_init_params {
  // memory pool
  size_t mem_size;   // bytes
  void* mem_buffer;  // if NULL, memory will be allocated internally
  bool no_alloc;     // don't allocate memory for the tensor data
};

#ifdef __cplusplus
}
#endif
