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
// Defines CLOCK_MONOTONIC on Linux
#define _GNU_SOURCE

#include "ne_layers.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>  // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#include <immintrin.h>
#include <mm_malloc.h>
#include <sched.h>
#endif

// layers
#include "layers/vec_dot.h"
#include "vectors/cpu/quantize.h"
#include "data_types.h"
#include "layers/Ops.h"
#include "layers/ele_reduce.h"
#include "layers/ele_wise.h"
#include "layers/mha_dense.h"
#include "ne.h"
#include "ne_jblas.h"

// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef static_assert
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif

#if defined(_WIN32)

#include <windows.h>

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;

static void atomic_store(atomic_int* ptr, LONG val) { InterlockedExchange(ptr, val); }
static LONG atomic_load(atomic_int* ptr) { return InterlockedCompareExchange(ptr, 0, 0); }
static LONG atomic_fetch_add(atomic_int* ptr, LONG inc) { return InterlockedExchangeAdd(ptr, inc); }
static LONG atomic_fetch_sub(atomic_int* ptr, LONG dec) { return atomic_fetch_add(ptr, -(dec)); }

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t* out, void* unused, thread_ret_t (*func)(void*), void* arg) {
  (void)unused;
  HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, arg, 0, NULL);
  if (handle == NULL) {
    return EAGAIN;
  }

  *out = handle;
  return 0;
}

static int pthread_join(pthread_t thread, void* unused) {
  (void)unused;
  return (int)WaitForSingleObject(thread, INFINITE);
}

static int sched_yield(void) {
  Sleep(0);
  return 0;
}
#else
#include <pthread.h>
#include <stdatomic.h>

typedef void* thread_ret_t;
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static_assert(sizeof(block_q4_0) == sizeof(ne_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");
static_assert(sizeof(block_q4_1) == 2 * sizeof(ne_fp16_t) + QK4_1 / 2, "wrong q4_1 block size/padding");
static_assert(sizeof(block_q5_0) == sizeof(ne_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");
static_assert(sizeof(block_q5_1) == 2 * sizeof(ne_fp16_t) + sizeof(uint32_t) + QK5_1 / 2,
              "wrong q5_1 block size/padding");
static_assert(sizeof(block_q8_0) == sizeof(ne_fp16_t) + QK8_0, "wrong q8_0 block size/padding");
static_assert(sizeof(block_q8_1) == 2 * sizeof(float) + QK8_1, "wrong q8_1 block size/padding");

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#ifndef __SSE3__
#define __SSE3__
#endif
#endif

/*#define NE_PERF*/
#define NE_DEBUG 0
#define NE_GELU_FP16
#define NE_SILU_FP16

#define NE_SOFT_MAX_UNROLL 4

#if UINTPTR_MAX == 0xFFFFFFFF
#define NE_MEM_ALIGN 4
#else
#define NE_MEM_ALIGN 16
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#define NE_ALIGNED_MALLOC(size) _aligned_malloc(size, NE_MEM_ALIGN)
#define NE_ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
inline static void* ne_aligned_malloc(size_t size) {
  void* aligned_memory = NULL;
  int result = posix_memalign(&aligned_memory, NE_MEM_ALIGN, size);
  if (result != 0) {
    // Handle allocation failure
    return NULL;
  }
  return aligned_memory;
}
#define NE_ALIGNED_MALLOC(size) ne_aligned_malloc(size)
#define NE_ALIGNED_FREE(ptr) free(ptr)
#endif

#define UNUSED(x) (void)(x)
#define SWAP(x, y, T) \
  do {                \
    T SWAP = x;       \
    x = y;            \
    y = SWAP;         \
  } while (0)

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// note: do not use these inside ne.c
// these are meant to be used via the ne.h API
float ne_fp16_to_fp32(ne_fp16_t x) { return (float)NE_FP16_TO_FP32(x); }

ne_fp16_t ne_fp32_to_fp16(float x) { return NE_FP32_TO_FP16(x); }

void ne_fp16_to_fp32_row(const ne_fp16_t* x, float* y, size_t n) {
  for (size_t i = 0; i < n; i++) {
    y[i] = NE_FP16_TO_FP32(x[i]);
  }
}

void ne_fp32_to_fp16_row(const float* x, ne_fp16_t* y, size_t n) {
  size_t i = 0;
#if defined(__F16C__)
  for (; i + 7 < n; i += 8) {
    __m256 x_vec = _mm256_loadu_ps(x + i);
    __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i*)(y + i), y_vec);
  }
  for (; i + 3 < n; i += 4) {
    __m128 x_vec = _mm_loadu_ps(x + i);
    __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
    _mm_storel_epi64((__m128i*)(y + i), y_vec);
  }
#endif
  for (; i < n; i++) {
    y[i] = NE_FP32_TO_FP16(x[i]);
  }
}

//
// timing
//

#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq;
void ne_time_init(void) {
  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);
  timer_freq = frequency.QuadPart;
}
int64_t ne_time_ms(void) {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return (t.QuadPart * 1000) / timer_freq;
}
int64_t ne_time_us(void) {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return (t.QuadPart * 1000000) / timer_freq;
}
#else
void ne_time_init(void) {}
int64_t ne_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

int64_t ne_time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}
#endif

int64_t ne_cycles(void) { return clock(); }

int64_t ne_cycles_per_ms(void) { return CLOCKS_PER_SEC / 1000; }

#ifdef NE_PERF
#define ne_perf_time_ms() ne_time_ms()
#define ne_perf_time_us() ne_time_us()
#define ne_perf_cycles() ne_cycles()
#define ne_perf_cycles_per_ms() ne_cycles_per_ms()
#else
#define ne_perf_time_ms() 0
#define ne_perf_time_us() 0
#define ne_perf_cycles() 0
#define ne_perf_cycles_per_ms() 0
#endif

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE hardware_destructive_interference_size
#else
#define CACHE_LINE_SIZE 64
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE / sizeof(float);

static const quantize_fns_t quantize_fns[NE_TYPE_COUNT] = {
    [NE_TYPE_Q4_0] =
        {
            .dequantize_row_q = (dequantize_row_q_t)dequantize_row_q4_0,
            .quantize_row_q = quantize_row_q4_0,
            .quantize_row_q_reference = (quantize_row_q_t)quantize_row_q4_0_reference,
            .quantize_row_q_dot = quantize_row_q8_0,
            .vec_dot_q = ne_vec_dot_q4_0_q8_0,
            .vec_dot_type = NE_TYPE_Q8_0,
        },
    [NE_TYPE_Q4_1] =
        {
            .dequantize_row_q = (dequantize_row_q_t)dequantize_row_q4_1,
            .quantize_row_q = quantize_row_q4_1,
            .quantize_row_q_reference = (quantize_row_q_t)quantize_row_q4_1_reference,
            .quantize_row_q_dot = quantize_row_q8_1,
            .vec_dot_q = ne_vec_dot_q4_1_q8_1,
            .vec_dot_type = NE_TYPE_Q8_1,
        },
    [NE_TYPE_Q5_0] =
        {
            .dequantize_row_q = (dequantize_row_q_t)dequantize_row_q5_0,
            .quantize_row_q = quantize_row_q5_0,
            .quantize_row_q_reference = (quantize_row_q_t)quantize_row_q5_0_reference,
            .quantize_row_q_dot = quantize_row_q8_0,
            .vec_dot_q = ne_vec_dot_q5_0_q8_0,
            .vec_dot_type = NE_TYPE_Q8_0,
        },
    [NE_TYPE_Q5_1] =
        {
            .dequantize_row_q = (dequantize_row_q_t)dequantize_row_q5_1,
            .quantize_row_q = quantize_row_q5_1,
            .quantize_row_q_reference = (quantize_row_q_t)quantize_row_q5_1_reference,
            .quantize_row_q_dot = quantize_row_q8_1,
            .vec_dot_q = ne_vec_dot_q5_1_q8_1,
            .vec_dot_type = NE_TYPE_Q8_1,
        },
    [NE_TYPE_Q8_0] =
        {
            .dequantize_row_q = dequantize_row_q8_0,
            .quantize_row_q = quantize_row_q8_0,
            .quantize_row_q_reference = (quantize_row_q_t)quantize_row_q8_0_reference,
            .quantize_row_q_dot = quantize_row_q8_0,
            .vec_dot_q = ne_vec_dot_q8_0_q8_0,
            .vec_dot_type = NE_TYPE_Q8_0,
        },
    [NE_TYPE_Q8_1] =
        {
            .dequantize_row_q = NULL,  // TODO
            .quantize_row_q = quantize_row_q8_1,
            .quantize_row_q_reference = (quantize_row_q_t)quantize_row_q8_1_reference,
            .quantize_row_q_dot = quantize_row_q8_1,
            .vec_dot_q = NULL,  // TODO
            .vec_dot_type = NE_TYPE_Q8_1,
        },
};

// For internal test use
quantize_fns_t ne_internal_get_quantize_fn(size_t i) {
  NE_ASSERT(i < NE_TYPE_COUNT);
  return quantize_fns[i];
}

//
// logging
//

#if (NE_DEBUG >= 1)
#define NE_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define NE_PRINT_DEBUG(...)
#endif

#if (NE_DEBUG >= 5)
#define NE_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define NE_PRINT_DEBUG_5(...)
#endif

#if (NE_DEBUG >= 10)
#define NE_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define NE_PRINT_DEBUG_10(...)
#endif

#define NE_PRINT(...) printf(__VA_ARGS__)

//
// data types
//

static const int NE_BLCK_SIZE[NE_TYPE_COUNT] = {
    [NE_TYPE_F32] = 1,      [NE_TYPE_F16] = 1,      [NE_TYPE_Q4_0] = QK4_0, [NE_TYPE_Q4_1] = QK4_1,
    [NE_TYPE_Q5_0] = QK5_0, [NE_TYPE_Q5_1] = QK5_1, [NE_TYPE_Q8_0] = QK8_0, [NE_TYPE_Q8_1] = QK8_1,
    [NE_TYPE_I8] = 1,       [NE_TYPE_I16] = 1,      [NE_TYPE_I32] = 1,
};
static_assert(NE_TYPE_COUNT == 14, "NE_BLCK_SIZE is outdated");

static const size_t NE_TYPE_SIZE[NE_TYPE_COUNT] = {
    [NE_TYPE_F32] = sizeof(float),       [NE_TYPE_F16] = sizeof(ne_fp16_t),   [NE_TYPE_Q4_0] = sizeof(block_q4_0),
    [NE_TYPE_Q4_1] = sizeof(block_q4_1), [NE_TYPE_Q5_0] = sizeof(block_q5_0), [NE_TYPE_Q5_1] = sizeof(block_q5_1),
    [NE_TYPE_Q8_0] = sizeof(block_q8_0), [NE_TYPE_Q8_1] = sizeof(block_q8_1), [NE_TYPE_I8] = sizeof(int8_t),
    [NE_TYPE_I16] = sizeof(int16_t),     [NE_TYPE_I32] = sizeof(int32_t),
};
static_assert(NE_TYPE_COUNT == 14, "NE_TYPE_SIZE is outdated");

static const char* NE_TYPE_NAME[NE_TYPE_COUNT] = {
    [NE_TYPE_F32] = "f32",   [NE_TYPE_F16] = "f16",   [NE_TYPE_Q4_0] = "q4_0", [NE_TYPE_Q4_1] = "q4_1",
    [NE_TYPE_Q5_0] = "q5_0", [NE_TYPE_Q5_1] = "q5_1", [NE_TYPE_Q8_0] = "q8_0", [NE_TYPE_Q8_1] = "q8_1",
    [NE_TYPE_I8] = "i8",     [NE_TYPE_I16] = "i16",   [NE_TYPE_I32] = "i32",
};
static_assert(NE_TYPE_COUNT == 14, "NE_TYPE_NAME is outdated");

static bool NE_IS_QUANTIZED[NE_TYPE_COUNT] = {
    [NE_TYPE_F32] = false, [NE_TYPE_F16] = false, [NE_TYPE_Q4_0] = true, [NE_TYPE_Q4_1] = true,
    [NE_TYPE_Q5_0] = true, [NE_TYPE_Q5_1] = true, [NE_TYPE_Q8_0] = true, [NE_TYPE_Q8_1] = true,
    [NE_TYPE_I8] = false,  [NE_TYPE_I16] = false, [NE_TYPE_I32] = false, [NE_TYPE_JBLAS] = true,
};
static_assert(NE_TYPE_COUNT == 14, "NE_IS_QUANTIZED is outdated");

static const char* NE_OP_LABEL[NE_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "LOG",
    "SUM",
    "SUM_ROWS",
    "MEAN",
    "REPEAT",
    "ABS",
    "SGN",
    "NEG",
    "STEP",
    "RELU",
    "GELU",
    "SILU",
    "SILU_BACK",
    "NORM",
    "RMS_NORM",
    "RMS_NORM_BACK",

    "MUL_MAT",
    "MUL_MAT_WITH_BIAS",
    "SCALE",
    "SET",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "GET_ROWS_BACK",
    "DIAG",
    "DIAG_MASK_INF",
    "DIAG_MASK_ZERO",
    "SOFT_MAX",
    "ROPE",
    "ROPE_BACK",
    "ALIBI",
    "CLAMP",
    "CONV_1D_1S",
    "CONV_1D_2S",

    "MUL_QKV",
    "FFN_SILU",
    "FFN_GeLU",
    "FFN_ADD_GeLU",
    "FLASH_ATTN",
    "FLASH_FF",

    "MAP_UNARY",
    "MAP_BINARY",
};

static_assert(NE_OP_COUNT == 56, "NE_OP_COUNT != 56");

static const char* NE_OP_SYMBOL[NE_OP_COUNT] = {
    "none",

    "x",
    "x+y",
    "x+y",
    "view(x,nb,offset)+=y->x",
    "x-y",
    "x*y",
    "x/y",
    "x^2",
    "√x",
    "log(x)",
    "Σx",
    "Σx_k",
    "Σx/n",
    "repeat(x)",
    "abs(x)",
    "sgn(x)",
    "-x",
    "step(x)",
    "relu(x)",
    "gelu(x)",
    "silu(x)",
    "silu_back(x)",
    "norm(x)",
    "rms_norm(x)",
    "rms_norm_back(x)",

    "X*Y",
    "X*Y+Z",
    "x*v",
    "y-\\>view(x)",
    "x-\\>y",
    "cont(x)",
    "reshape(x)",
    "view(x)",
    "permute(x)",
    "transpose(x)",
    "get_rows(x)",
    "get_rows_back(x)",
    "diag(x)",
    "diag_mask_inf(x)",
    "diag_mask_zero(x)",
    "soft_max(x)",
    "rope(x)",
    "rope_back(x)",
    "alibi(x)",
    "clamp(x)",
    "conv_1d_1s(x)",
    "conv_1d_2s(x)",

    "QKV(x)",
    "ffn_silu(x)",
    "ffn_gelu(x)",
    "ffn_gelu_with_bias(x)",
    "flash_attn(x)",
    "flash_ff(x)",

    "f(x)",
    "f(x,y)",
};

static_assert(sizeof(struct ne_object) % NE_MEM_ALIGN == 0, "ne_object size must be a multiple of NE_MEM_ALIGN");
static_assert(sizeof(struct ne_tensor) % NE_MEM_ALIGN == 0, "ne_tensor size must be a multiple of NE_MEM_ALIGN");

//
// compute types
//

enum ne_task_type {
  NE_TASK_INIT = 0,
  NE_TASK_COMPUTE,
  NE_TASK_FINALIZE,
};

struct ne_compute_params {
  enum ne_task_type type;

  int ith, nth;

  // work buffer for all threads
  size_t wsize;
  void* wdata;
};

//
// ne state
//

struct ne_state {
  struct ne_context_container contexts[NE_MAX_CONTEXTS];
};

// global state
static struct ne_state g_state;
static atomic_int g_state_barrier = 0;

// barrier via spin lock
inline static void ne_critical_section_start(void) {
  int processing = atomic_fetch_add(&g_state_barrier, 1);

  while (processing > 0) {
    // wait for other threads to finish
    atomic_fetch_sub(&g_state_barrier, 1);
    sched_yield();  // TODO: reconsider this
    processing = atomic_fetch_add(&g_state_barrier, 1);
  }
}

// TODO: make this somehow automatically executed
//       some sort of "sentry" mechanism
inline static void ne_critical_section_end(void) { atomic_fetch_sub(&g_state_barrier, 1); }

////////////////////////////////////////////////////////////////////////////////

void ne_print_object(const struct ne_object* obj) {
  NE_PRINT(" - ne_object: offset = %zu, size = %zu, next = %p\n", obj->offs, obj->size, (const void*)obj->next);
}

void ne_print_objects(const struct ne_context* ctx) {
  struct ne_object* obj = ctx->objects_begin;

  NE_PRINT("%s: objects in context %p:\n", __func__, (const void*)ctx);

  while (obj != NULL) {
    ne_print_object(obj);
    obj = obj->next;
  }

  NE_PRINT("%s: --- end ---\n", __func__);
}

int64_t ne_nelements(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

int ne_nrows(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

size_t ne_nbytes(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return (ne_nelements(tensor) * NE_TYPE_SIZE[tensor->type]) / NE_BLCK_SIZE[tensor->type];
}

int ne_blck_size(enum ne_type type) { return NE_BLCK_SIZE[type]; }

size_t ne_type_size(enum ne_type type) { return NE_TYPE_SIZE[type]; }

float ne_type_sizef(enum ne_type type) { return ((float)(NE_TYPE_SIZE[type])) / NE_BLCK_SIZE[type]; }

const char* ne_type_name(enum ne_type type) { return NE_TYPE_NAME[type]; }

size_t ne_element_size(const struct ne_tensor* tensor) { return NE_TYPE_SIZE[tensor->type]; }

static inline bool ne_is_scalar(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool ne_is_vector(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool ne_is_matrix(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool ne_can_mul_mat(const struct ne_tensor* t0, const struct ne_tensor* t1) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  // verify t0 is broadcastable
  return (t0->ne[0] == t1->ne[0]) && (t1->ne[2] % t0->ne[2] == 0) && (t1->ne[3] % t0->ne[3] == 0);
}

bool ne_is_quantized(enum ne_type type) { return NE_IS_QUANTIZED[type]; }

enum ne_type ne_ftype_to_ne_type(enum ne_ftype ftype) {
  enum ne_type wtype = NE_TYPE_COUNT;

  switch (ftype) {
    case NE_FTYPE_ALL_F32:
      wtype = NE_TYPE_F32;
      break;
    case NE_FTYPE_MOSTLY_F16:
      wtype = NE_TYPE_F16;
      break;
    case NE_FTYPE_MOSTLY_Q4_0:
      wtype = NE_TYPE_Q4_0;
      break;
    case NE_FTYPE_MOSTLY_Q4_1:
      wtype = NE_TYPE_Q4_1;
      break;
    case NE_FTYPE_MOSTLY_Q5_0:
      wtype = NE_TYPE_Q5_0;
      break;
    case NE_FTYPE_MOSTLY_Q5_1:
      wtype = NE_TYPE_Q5_1;
      break;
    case NE_FTYPE_MOSTLY_Q8_0:
      wtype = NE_TYPE_Q8_0;
      break;
    case NE_FTYPE_UNKNOWN:
      wtype = NE_TYPE_COUNT;
      break;
    case NE_FTYPE_MOSTLY_Q4_1_SOME_F16:
      wtype = NE_TYPE_COUNT;
      break;
  }

  NE_ASSERT(wtype != NE_TYPE_COUNT);

  return wtype;
}

static inline bool ne_is_transposed(const struct ne_tensor* tensor) { return tensor->nb[0] > tensor->nb[1]; }

static inline bool ne_is_contiguous(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return tensor->nb[0] == NE_TYPE_SIZE[tensor->type] &&
         tensor->nb[1] == (tensor->nb[0] * tensor->ne[0]) / NE_BLCK_SIZE[tensor->type] &&
         tensor->nb[2] == tensor->nb[1] * tensor->ne[1] && tensor->nb[3] == tensor->nb[2] * tensor->ne[2];
}

static inline bool ne_is_padded_1d(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return tensor->nb[0] == NE_TYPE_SIZE[tensor->type] && tensor->nb[2] == tensor->nb[1] * tensor->ne[1] &&
         tensor->nb[3] == tensor->nb[2] * tensor->ne[2];
}

static inline bool ne_are_same_shape(const struct ne_tensor* t0, const struct ne_tensor* t1) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return (t0->ne[0] == t1->ne[0]) && (t0->ne[1] == t1->ne[1]) && (t0->ne[2] == t1->ne[2]) && (t0->ne[3] == t1->ne[3]);
}

// check if t1 can be represented as a repeatition of t0
static inline bool ne_can_repeat(const struct ne_tensor* t0, const struct ne_tensor* t1) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return (t1->ne[0] % t0->ne[0] == 0) && (t1->ne[1] % t0->ne[1] == 0) && (t1->ne[2] % t0->ne[2] == 0) &&
         (t1->ne[3] % t0->ne[3] == 0);
}

static inline bool ne_can_repeat_rows(const struct ne_tensor* t0, const struct ne_tensor* t1) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return (t0->ne[0] == t1->ne[0]) && ne_can_repeat(t0, t1);
}

static inline int ne_up32(int n) { return (n + 31) & ~31; }

// static inline int ne_up64(int n) {
//     return (n + 63) & ~63;
// }

static inline int ne_up(int n, int m) {
  // assert m is a power of 2
  NE_ASSERT((m & (m - 1)) == 0);
  return (n + m - 1) & ~(m - 1);
}

// assert that pointer is aligned to NE_MEM_ALIGN
#define ne_assert_aligned(ptr) NE_ASSERT(((uintptr_t)(ptr)) % NE_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

struct ne_context* ne_init(struct ne_init_params params) {
  // make this function thread safe
  ne_critical_section_start();

  static bool is_first_call = true;

  if (is_first_call) {
    // initialize time system (required on Windows)
    ne_time_init();
    // initialize jblas's amx instruction.
    jblas_init();
    // initialize GELU, SILU and EXP F32 tables
    {
      const uint64_t t_start = ne_time_us();
      UNUSED(t_start);

      ne_fp16_t ii;
      for (int i = 0; i < (1 << 16); ++i) {
        uint16_t ui = i;
        memcpy(&ii, &ui, sizeof(ii));
        const float f = table_f32_f16[i] = NE_COMPUTE_FP16_TO_FP32(ii);
        table_gelu_f16[i] = NE_FP32_TO_FP16(ne_gelu_f32(f));
        table_silu_f16[i] = NE_FP32_TO_FP16(ne_silu_f32(f));
        table_exp_f16[i] = NE_FP32_TO_FP16(expf(f));
      }

      const uint64_t t_end = ne_time_us();
      UNUSED(t_end);

      NE_PRINT_DEBUG("%s: GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start) / 1000.0f);
    }

    // initialize g_state
    {
      const uint64_t t_start = ne_time_us();
      UNUSED(t_start);

      g_state = (struct ne_state){
          /*.contexts =*/{{0}},
      };

      for (int i = 0; i < NE_MAX_CONTEXTS; ++i) {
        g_state.contexts[i].used = false;
      }

      const uint64_t t_end = ne_time_us();
      UNUSED(t_end);

      NE_PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start) / 1000.0f);
    }

    is_first_call = false;
  }

  // find non-used context in g_state
  struct ne_context* ctx = NULL;

  for (int i = 0; i < NE_MAX_CONTEXTS; i++) {
    if (!g_state.contexts[i].used) {
      g_state.contexts[i].used = true;
      ctx = &g_state.contexts[i].context;

      NE_PRINT_DEBUG("%s: found unused context %d\n", __func__, i);
      break;
    }
  }

  if (ctx == NULL) {
    NE_PRINT_DEBUG("%s: no unused context found\n", __func__);

    ne_critical_section_end();

    return NULL;
  }

  const size_t mem_size = (params.mem_size + NE_MEM_ALIGN - 1) & ~(NE_MEM_ALIGN - 1);

  *ctx = (struct ne_context){
      /*.mem_size           =*/mem_size,
      /*.mem_buffer         =*/params.mem_buffer ? params.mem_buffer : NE_ALIGNED_MALLOC(mem_size),
      /*.mem_buffer_owned   =*/params.mem_buffer ? false : true,
      /*.no_alloc           =*/params.no_alloc,
      /*.n_objects          =*/0,
      /*.objects_begin      =*/NULL,
      /*.objects_end        =*/NULL,
      /*.scratch            =*/
      {
          0,
          0,
          NULL,
      },
      /*.scratch_save       =*/
      {
          0,
          0,
          NULL,
      },
  };

  NE_ASSERT(ctx->mem_buffer != NULL);

  ne_assert_aligned(ctx->mem_buffer);

  NE_PRINT_DEBUG("%s: context initialized\n", __func__);

  ne_critical_section_end();

  return ctx;
}

void ne_free(struct ne_context* ctx) {
  // make this function thread safe
  ne_critical_section_start();

  bool found = false;

  for (int i = 0; i < NE_MAX_CONTEXTS; i++) {
    if (&g_state.contexts[i].context == ctx) {
      g_state.contexts[i].used = false;

      NE_PRINT_DEBUG("%s: context %d with %d objects has been freed. memory used = %zu\n", __func__, i, ctx->n_objects,
                     ctx->objects_end->offs + ctx->objects_end->size);

      if (ctx->mem_buffer_owned) {
        NE_ALIGNED_FREE(ctx->mem_buffer);
      }

      found = true;
      break;
    }
  }

  if (!found) {
    NE_PRINT_DEBUG("%s: context not found\n", __func__);
  }

  ne_critical_section_end();
}

size_t ne_used_mem(const struct ne_context* ctx) {
  return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

size_t ne_set_scratch(struct ne_context* ctx, struct ne_scratch scratch) {
  const size_t result = ctx->scratch.data ? ctx->scratch.offs : 0;

  ctx->scratch = scratch;

  return result;
}

// IMPORTANT:
// when creating "opt" tensors, always save and load the scratch buffer
// this is an error prone process, but it is necessary to support inplace
// operators when using scratch buffers
// TODO: implement a better way
void ne_scratch_save(struct ne_context* ctx) {
  ctx->scratch_save = ctx->scratch;
  ctx->scratch.data = NULL;
}

void ne_scratch_load(struct ne_context* ctx) { ctx->scratch = ctx->scratch_save; }

////////////////////////////////////////////////////////////////////////////////

struct ne_tensor* ne_new_tensor_impl(struct ne_context* ctx, enum ne_type type, int n_dims, const int64_t* ne,
                                     void* data, size_t size) {
  // always insert objects at the end of the context's memory pool
  struct ne_object* obj_cur = ctx->objects_end;

  const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
  const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
  const size_t cur_end = cur_offs + cur_size;

  size_t size_needed = 0;

  if (data == NULL && !ctx->no_alloc) {
    if (type == NE_TYPE_JBLAS) {
      size_needed = size;
    } else {
      size_needed += NE_TYPE_SIZE[type] * (ne[0] / NE_BLCK_SIZE[type]);
      for (int i = 1; i < n_dims; i++) {
        size_needed *= ne[i];
      }
      size_needed = ((size_needed + NE_MEM_ALIGN - 1) / NE_MEM_ALIGN) * NE_MEM_ALIGN;
    }
  }

  char* const mem_buffer = ctx->mem_buffer;
  struct ne_object* const obj_new = (struct ne_object*)(mem_buffer + cur_end);

  if (ctx->scratch.data == NULL || data != NULL) {
    size_needed += sizeof(struct ne_tensor);

    if (cur_end + size_needed + NE_OBJECT_SIZE > ctx->mem_size) {
      NE_PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n", __func__,
               cur_end + size_needed + NE_OBJECT_SIZE, ctx->mem_size);
      assert(false);
      return NULL;
    }

    *obj_new = (struct ne_object){
        .offs = cur_end + NE_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
    };
  } else {
    if (ctx->scratch.offs + size_needed > ctx->scratch.size) {
      NE_PRINT("%s: not enough space in the scratch memory\n", __func__);
      assert(false);
      return NULL;
    }

    if (cur_end + sizeof(struct ne_tensor) + NE_OBJECT_SIZE > ctx->mem_size) {
      NE_PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n", __func__,
               cur_end + sizeof(struct ne_tensor) + NE_OBJECT_SIZE, ctx->mem_size);
      assert(false);
      return NULL;
    }

    data = (char* const)ctx->scratch.data + ctx->scratch.offs;

    *obj_new = (struct ne_object){
        .offs = cur_end + NE_OBJECT_SIZE,
        .size = sizeof(struct ne_tensor),
        .next = NULL,
    };

    ctx->scratch.offs += size_needed;
  }

  if (obj_cur != NULL) {
    obj_cur->next = obj_new;
  } else {
    // this is the first object in this context
    ctx->objects_begin = obj_new;
  }

  ctx->objects_end = obj_new;

  struct ne_tensor* const result = (struct ne_tensor*)(mem_buffer + obj_new->offs);

  *result = (struct ne_tensor){
      /*.type         =*/type,
      /*.backend      =*/NE_BACKEND_CPU,
      /*.n_dims       =*/n_dims,
      /*.ne           =*/{1, 1, 1, 1},
      /*.nb           =*/{0, 0, 0, 0},
      /*.op           =*/NE_OP_NONE,
      /*.is_param     =*/false,
      /*.grad         =*/NULL,
      /*.src0         =*/NULL,
      /*.src1         =*/NULL,
      /*.opt          =*/{NULL},
      /*.n_tasks      =*/0,
      /*.perf_runs    =*/0,
      /*.perf_cycles  =*/0,
      /*.perf_time_us =*/0,
      /*.data         =*/(data == NULL && !ctx->no_alloc) ? (void*)(result + 1) : data,
      /*.size         =*/size_needed,
      /*.name         =*/{0},
      /*.pad          =*/{0},
  };

  for (int i = 0; i < n_dims; i++) {
    result->ne[i] = ne[i];
  }
  result->nb[0] = NE_TYPE_SIZE[type];
  if (type != NE_TYPE_JBLAS) {
    result->nb[1] = result->nb[0] * (result->ne[0] / NE_BLCK_SIZE[type]);
  }

  for (int i = 2; i < NE_MAX_DIMS; i++) {
    result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
  }

  ctx->n_objects++;

  return result;
}

struct ne_tensor* ne_new_tensor(struct ne_context* ctx, enum ne_type type, int n_dims, const int64_t* ne, size_t size) {
  return ne_new_tensor_impl(ctx, type, n_dims, ne, NULL, size);
}

struct ne_tensor* ne_new_tensor_1d(struct ne_context* ctx, enum ne_type type, int64_t ne0, size_t size) {
  return ne_new_tensor(ctx, type, 1, &ne0, size);
}

struct ne_tensor* ne_new_tensor_2d(struct ne_context* ctx, enum ne_type type, int64_t ne0, int64_t ne1, size_t size) {
  const int64_t ne[2] = {ne0, ne1};
  return ne_new_tensor(ctx, type, 2, ne, size);
}

struct ne_tensor* ne_new_tensor_3d(struct ne_context* ctx, enum ne_type type, int64_t ne0, int64_t ne1, int64_t ne2,
                                   size_t size) {
  const int64_t ne[3] = {ne0, ne1, ne2};
  return ne_new_tensor(ctx, type, 3, ne, size);
}

struct ne_tensor* ne_new_tensor_4d(struct ne_context* ctx, enum ne_type type, int64_t ne0, int64_t ne1, int64_t ne2,
                                   int64_t ne3, size_t size) {
  const int64_t ne[4] = {ne0, ne1, ne2, ne3};
  return ne_new_tensor(ctx, type, 4, ne, size);
}

struct ne_tensor* ne_new_i32(struct ne_context* ctx, int32_t value) {
  ne_scratch_save(ctx);

  struct ne_tensor* result = ne_new_tensor_1d(ctx, NE_TYPE_I32, 1, NE_SIZE_CALC);

  ne_scratch_load(ctx);

  ne_set_i32(result, value);

  return result;
}

struct ne_tensor* ne_new_f32(struct ne_context* ctx, float value) {
  ne_scratch_save(ctx);

  struct ne_tensor* result = ne_new_tensor_1d(ctx, NE_TYPE_F32, 1, NE_SIZE_CALC);

  ne_scratch_load(ctx);

  ne_set_f32(result, value);

  return result;
}

struct ne_tensor* ne_dup_tensor(struct ne_context* ctx, const struct ne_tensor* src) {
  return ne_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, NULL, src->size);
}

struct ne_tensor* ne_set_zero(struct ne_tensor* tensor) {
  memset(tensor->data, 0, ne_nbytes(tensor));
  return tensor;
}

struct ne_tensor* ne_set_i32(struct ne_tensor* tensor, int32_t value) {
  const int n = ne_nrows(tensor);
  const int nc = tensor->ne[0];
  const size_t n1 = tensor->nb[1];

  char* const data = tensor->data;

  switch (tensor->type) {
    case NE_TYPE_I8: {
      assert(tensor->nb[0] == sizeof(int8_t));
      for (int i = 0; i < n; i++) {
        ne_vec_set_i8(nc, (int8_t*)(data + i * n1), value);
      }
    } break;
    case NE_TYPE_I16: {
      assert(tensor->nb[0] == sizeof(int16_t));
      for (int i = 0; i < n; i++) {
        ne_vec_set_i16(nc, (int16_t*)(data + i * n1), value);
      }
    } break;
    case NE_TYPE_I32: {
      assert(tensor->nb[0] == sizeof(int32_t));
      for (int i = 0; i < n; i++) {
        ne_vec_set_i32(nc, (int32_t*)(data + i * n1), value);
      }
    } break;
    case NE_TYPE_F16: {
      assert(tensor->nb[0] == sizeof(ne_fp16_t));
      for (int i = 0; i < n; i++) {
        ne_vec_set_f16(nc, (ne_fp16_t*)(data + i * n1), value);
      }
    } break;
    case NE_TYPE_F32: {
      assert(tensor->nb[0] == sizeof(float));
      for (int i = 0; i < n; i++) {
        ne_vec_set_f32(nc, (float*)(data + i * n1), value);
      }
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }

  return tensor;
}

struct ne_tensor* ne_set_f32(struct ne_tensor* tensor, float value) {
  const int n = ne_nrows(tensor);
  const int nc = tensor->ne[0];
  const size_t n1 = tensor->nb[1];

  char* const data = tensor->data;

  switch (tensor->type) {
    case NE_TYPE_I8: {
      assert(tensor->nb[0] == sizeof(int8_t));
      for (int i = 0; i < n; i++) {
        ne_vec_set_i8(nc, (int8_t*)(data + i * n1), value);
      }
    } break;
    case NE_TYPE_I16: {
      assert(tensor->nb[0] == sizeof(int16_t));
      for (int i = 0; i < n; i++) {
        ne_vec_set_i16(nc, (int16_t*)(data + i * n1), value);
      }
    } break;
    case NE_TYPE_I32: {
      assert(tensor->nb[0] == sizeof(int32_t));
      for (int i = 0; i < n; i++) {
        ne_vec_set_i32(nc, (int32_t*)(data + i * n1), value);
      }
    } break;
    case NE_TYPE_F16: {
      assert(tensor->nb[0] == sizeof(ne_fp16_t));
      for (int i = 0; i < n; i++) {
        ne_vec_set_f16(nc, (ne_fp16_t*)(data + i * n1), value);
      }
    } break;
    case NE_TYPE_F32: {
      assert(tensor->nb[0] == sizeof(float));
      for (int i = 0; i < n; i++) {
        ne_vec_set_f32(nc, (float*)(data + i * n1), value);
      }
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }

  return tensor;
}

int32_t ne_get_i32_1d(const struct ne_tensor* tensor, int i) {
  switch (tensor->type) {
    case NE_TYPE_I8: {
      NE_ASSERT(tensor->nb[0] == sizeof(int8_t));
      return ((int8_t*)(tensor->data))[i];
    } break;
    case NE_TYPE_I16: {
      NE_ASSERT(tensor->nb[0] == sizeof(int16_t));
      return ((int16_t*)(tensor->data))[i];
    } break;
    case NE_TYPE_I32: {
      NE_ASSERT(tensor->nb[0] == sizeof(int32_t));
      return ((int32_t*)(tensor->data))[i];
    } break;
    case NE_TYPE_F16: {
      NE_ASSERT(tensor->nb[0] == sizeof(ne_fp16_t));
      return NE_FP16_TO_FP32(((ne_fp16_t*)(tensor->data))[i]);
    } break;
    case NE_TYPE_F32: {
      NE_ASSERT(tensor->nb[0] == sizeof(float));
      return ((float*)(tensor->data))[i];
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }

  return 0.0f;
}

void ne_set_i32_1d(const struct ne_tensor* tensor, int i, int32_t value) {
  switch (tensor->type) {
    case NE_TYPE_I8: {
      NE_ASSERT(tensor->nb[0] == sizeof(int8_t));
      ((int8_t*)(tensor->data))[i] = value;
    } break;
    case NE_TYPE_I16: {
      NE_ASSERT(tensor->nb[0] == sizeof(int16_t));
      ((int16_t*)(tensor->data))[i] = value;
    } break;
    case NE_TYPE_I32: {
      NE_ASSERT(tensor->nb[0] == sizeof(int32_t));
      ((int32_t*)(tensor->data))[i] = value;
    } break;
    case NE_TYPE_F16: {
      NE_ASSERT(tensor->nb[0] == sizeof(ne_fp16_t));
      ((ne_fp16_t*)(tensor->data))[i] = NE_FP32_TO_FP16(value);
    } break;
    case NE_TYPE_F32: {
      NE_ASSERT(tensor->nb[0] == sizeof(float));
      ((float*)(tensor->data))[i] = value;
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

float ne_get_f32_1d(const struct ne_tensor* tensor, int i) {
  switch (tensor->type) {
    case NE_TYPE_I8: {
      NE_ASSERT(tensor->nb[0] == sizeof(int8_t));
      return ((int8_t*)(tensor->data))[i];
    } break;
    case NE_TYPE_I16: {
      NE_ASSERT(tensor->nb[0] == sizeof(int16_t));
      return ((int16_t*)(tensor->data))[i];
    } break;
    case NE_TYPE_I32: {
      NE_ASSERT(tensor->nb[0] == sizeof(int32_t));
      return ((int32_t*)(tensor->data))[i];
    } break;
    case NE_TYPE_F16: {
      NE_ASSERT(tensor->nb[0] == sizeof(ne_fp16_t));
      return NE_FP16_TO_FP32(((ne_fp16_t*)(tensor->data))[i]);
    } break;
    case NE_TYPE_F32: {
      NE_ASSERT(tensor->nb[0] == sizeof(float));
      return ((float*)(tensor->data))[i];
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }

  return 0.0f;
}

void ne_set_f32_1d(const struct ne_tensor* tensor, int i, float value) {
  switch (tensor->type) {
    case NE_TYPE_I8: {
      NE_ASSERT(tensor->nb[0] == sizeof(int8_t));
      ((int8_t*)(tensor->data))[i] = value;
    } break;
    case NE_TYPE_I16: {
      NE_ASSERT(tensor->nb[0] == sizeof(int16_t));
      ((int16_t*)(tensor->data))[i] = value;
    } break;
    case NE_TYPE_I32: {
      NE_ASSERT(tensor->nb[0] == sizeof(int32_t));
      ((int32_t*)(tensor->data))[i] = value;
    } break;
    case NE_TYPE_F16: {
      NE_ASSERT(tensor->nb[0] == sizeof(ne_fp16_t));
      ((ne_fp16_t*)(tensor->data))[i] = NE_FP32_TO_FP16(value);
    } break;
    case NE_TYPE_F32: {
      NE_ASSERT(tensor->nb[0] == sizeof(float));
      ((float*)(tensor->data))[i] = value;
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

void* ne_get_data(const struct ne_tensor* tensor) { return tensor->data; }

float* ne_get_data_f32(const struct ne_tensor* tensor) {
  assert(tensor->type == NE_TYPE_F32);
  return (float*)(tensor->data);
}

const char* ne_get_name(const struct ne_tensor* tensor) { return tensor->name; }

void ne_set_name(struct ne_tensor* tensor, const char* name) {
  strncpy(tensor->name, name, sizeof(tensor->name));
  tensor->name[sizeof(tensor->name) - 1] = '\0';
}

struct ne_tensor* ne_view_tensor(struct ne_context* ctx, const struct ne_tensor* src) {
  struct ne_tensor* result = ne_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, src->data, src->size);

  result->nb[0] = src->nb[0];
  result->nb[1] = src->nb[1];
  result->nb[2] = src->nb[2];
  result->nb[3] = src->nb[3];

  return result;
}

////////////////////////////////////////////////////////////////////////////////

// ne_dup

struct ne_tensor* ne_dup_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_DUP;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_dup(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_dup_impl(ctx, a, false);
}

struct ne_tensor* ne_dup_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_dup_impl(ctx, a, true);
}

// ne_add

struct ne_tensor* ne_add_impl(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, bool inplace) {
  NE_ASSERT(ne_can_repeat_rows(b, a));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_ADD;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct ne_tensor* ne_add(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_add_impl(ctx, a, b, false);
}

struct ne_tensor* ne_add_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_add_impl(ctx, a, b, true);
}

// ne_add1

struct ne_tensor* ne_add1_impl(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, bool inplace) {
  NE_ASSERT(ne_is_scalar(b));
  NE_ASSERT(ne_is_padded_1d(a));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_ADD1;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct ne_tensor* ne_add1(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_add1_impl(ctx, a, b, false);
}

struct ne_tensor* ne_add1_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_add1_impl(ctx, a, b, true);
}

// ne_acc

struct ne_tensor* ne_acc_impl(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1, size_t nb2,
                              size_t nb3, size_t offset, bool inplace) {
  NE_ASSERT(ne_nelements(b) <= ne_nelements(a));
  NE_ASSERT(ne_is_contiguous(a));
  NE_ASSERT(a->type == NE_TYPE_F32);
  NE_ASSERT(b->type == NE_TYPE_F32);

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  ne_scratch_save(ctx);

  struct ne_tensor* c = ne_new_tensor_1d(ctx, NE_TYPE_I32, 5, NE_SIZE_CALC);

  ((int32_t*)c->data)[0] = nb1;
  ((int32_t*)c->data)[1] = nb2;
  ((int32_t*)c->data)[2] = nb3;
  ((int32_t*)c->data)[3] = offset;
  ((int32_t*)c->data)[4] = inplace ? 1 : 0;

  ne_scratch_load(ctx);

  result->op = NE_OP_ACC;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;
  result->opt[0] = c;

  return result;
}

struct ne_tensor* ne_acc(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1, size_t nb2,
                         size_t nb3, size_t offset) {
  return ne_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

struct ne_tensor* ne_acc_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1,
                                 size_t nb2, size_t nb3, size_t offset) {
  return ne_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ne_sub

struct ne_tensor* ne_sub_impl(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, bool inplace) {
  NE_ASSERT(ne_are_same_shape(a, b));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_SUB;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct ne_tensor* ne_sub(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_sub_impl(ctx, a, b, false);
}

struct ne_tensor* ne_sub_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_sub_impl(ctx, a, b, true);
}

// ne_mul

struct ne_tensor* ne_mul_impl(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, bool inplace) {
  // TODO: support less-strict constraint
  //       NE_ASSERT(ne_can_repeat(b, a));
  NE_ASSERT(ne_can_repeat_rows(b, a));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    // TODO: support backward pass for broadcasting
    NE_ASSERT(ne_are_same_shape(a, b));
    is_node = true;
  }

  if (inplace) {
    NE_ASSERT(is_node == false);
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_MUL;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct ne_tensor* ne_mul(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_mul_impl(ctx, a, b, false);
}

struct ne_tensor* ne_mul_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_mul_impl(ctx, a, b, true);
}

// ne_div

struct ne_tensor* ne_div_impl(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, bool inplace) {
  NE_ASSERT(ne_are_same_shape(a, b));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  if (inplace) {
    NE_ASSERT(is_node == false);
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_DIV;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct ne_tensor* ne_div(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_div_impl(ctx, a, b, false);
}

struct ne_tensor* ne_div_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_div_impl(ctx, a, b, true);
}

// ne_sqr

struct ne_tensor* ne_sqr_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_SQR;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_sqr(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_sqr_impl(ctx, a, false);
}

struct ne_tensor* ne_sqr_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_sqr_impl(ctx, a, true);
}

// ne_sqrt

struct ne_tensor* ne_sqrt_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_SQRT;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_sqrt(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_sqrt_impl(ctx, a, false);
}

struct ne_tensor* ne_sqrt_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_sqrt_impl(ctx, a, true);
}

// ne_log

struct ne_tensor* ne_log_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_LOG;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_log(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_log_impl(ctx, a, false);
}

struct ne_tensor* ne_log_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_log_impl(ctx, a, true);
}

// ne_sum

struct ne_tensor* ne_sum(struct ne_context* ctx, struct ne_tensor* a) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ne_tensor* result = ne_new_tensor_1d(ctx, a->type, 1, NE_SIZE_CALC);

  result->op = NE_OP_SUM;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

// ne_sum_rows

struct ne_tensor* ne_sum_rows(struct ne_context* ctx, struct ne_tensor* a) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  int64_t ne[4] = {1, 1, 1, 1};
  for (int i = 1; i < a->n_dims; ++i) {
    ne[i] = a->ne[i];
  }

  struct ne_tensor* result = ne_new_tensor(ctx, a->type, a->n_dims, ne, a->size);

  result->op = NE_OP_SUM_ROWS;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

// ne_mean

struct ne_tensor* ne_mean(struct ne_context* ctx, struct ne_tensor* a) {
  bool is_node = false;

  if (a->grad) {
    NE_ASSERT(false);  // TODO: implement
    is_node = true;
  }

  int64_t ne[NE_MAX_DIMS] = {1, a->ne[1], a->ne[2], a->ne[3]};
  struct ne_tensor* result = ne_new_tensor(ctx, NE_TYPE_F32, a->n_dims, ne, NE_SIZE_CALC);

  result->op = NE_OP_MEAN;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

// ne_repeat

struct ne_tensor* ne_repeat(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  NE_ASSERT(ne_can_repeat(a, b));

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  if (ne_are_same_shape(a, b) && !is_node) {
    return a;
  }

  struct ne_tensor* result = ne_new_tensor(ctx, a->type, b->n_dims, b->ne, NE_SIZE_CALC);

  result->op = NE_OP_REPEAT;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

// ne_abs

struct ne_tensor* ne_abs_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_ABS;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_abs(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_abs_impl(ctx, a, false);
}

struct ne_tensor* ne_abs_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_abs_impl(ctx, a, true);
}

// ne_sgn

struct ne_tensor* ne_sgn_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_SGN;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_sgn(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_sgn_impl(ctx, a, false);
}

struct ne_tensor* ne_sgn_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_sgn_impl(ctx, a, true);
}

// ne_neg

struct ne_tensor* ne_neg_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_NEG;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_neg(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_neg_impl(ctx, a, false);
}

struct ne_tensor* ne_neg_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_neg_impl(ctx, a, true);
}

// ne_step

struct ne_tensor* ne_step_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_STEP;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_step(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_step_impl(ctx, a, false);
}

struct ne_tensor* ne_step_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_step_impl(ctx, a, true);
}

// ne_relu

struct ne_tensor* ne_relu_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_RELU;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_relu(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_relu_impl(ctx, a, false);
}

struct ne_tensor* ne_relu_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_relu_impl(ctx, a, true);
}

// ne_gelu

struct ne_tensor* ne_gelu_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_GELU;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_gelu(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_gelu_impl(ctx, a, false);
}

struct ne_tensor* ne_gelu_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_gelu_impl(ctx, a, true);
}

// ne_silu

struct ne_tensor* ne_silu_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_SILU;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_silu(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_silu_impl(ctx, a, false);
}

struct ne_tensor* ne_silu_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_silu_impl(ctx, a, true);
}

// ne_silu_back

struct ne_tensor* ne_silu_back(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  bool is_node = false;

  if (a->grad || b->grad) {
    // TODO: implement backward
    is_node = true;
  }

  struct ne_tensor* result = ne_dup_tensor(ctx, a);

  result->op = NE_OP_SILU_BACK;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

// ne_norm

struct ne_tensor* ne_norm_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    NE_ASSERT(false);  // TODO: implement backward
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_NORM;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;  // TODO: maybe store epsilon here?

  return result;
}

struct ne_tensor* ne_norm(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_norm_impl(ctx, a, false);
}

struct ne_tensor* ne_norm_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_norm_impl(ctx, a, true);
}

struct ne_tensor* ne_rms_norm_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_RMS_NORM;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;  // TODO: maybe store epsilon here?

  return result;
}

struct ne_tensor* ne_rms_norm(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_rms_norm_impl(ctx, a, false);
}

struct ne_tensor* ne_rms_norm_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_rms_norm_impl(ctx, a, true);
}

struct ne_tensor* ne_rms_norm_back(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  bool is_node = false;

  if (a->grad) {
    // TODO: implement backward
    is_node = true;
  }

  struct ne_tensor* result = ne_dup_tensor(ctx, a);

  result->op = NE_OP_RMS_NORM_BACK;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

// ne_mul_mat

struct ne_tensor* ne_mul_mat(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  NE_ASSERT(ne_can_mul_mat(a, b));
  NE_ASSERT(!ne_is_transposed(a));

  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {a->ne[1], b->ne[1], b->ne[2], b->ne[3]};
  struct ne_tensor* result = ne_new_tensor(ctx, NE_TYPE_F32, MAX(a->n_dims, b->n_dims), ne, NE_SIZE_CALC);

  result->op = NE_OP_MUL_MAT;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

// ne_mul_mat_with_bias

struct ne_tensor* ne_mul_mat_with_bias(struct ne_context* ctx, struct ne_tensor* w, struct ne_tensor* b,
                                       struct ne_tensor* a) {
  NE_ASSERT(ne_can_mul_mat(w, a));
  NE_ASSERT(!ne_is_transposed(w));

  bool is_node = false;

  if (w->grad || b->grad || a->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {w->ne[1], a->ne[1], w->ne[2], a->ne[3]};
  struct ne_tensor* result = ne_new_tensor(ctx, a->type, MIN(w->n_dims, a->n_dims), ne, NE_SIZE_CALC);

  result->op = NE_OP_MUL_MAT_BIAS;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = w;
  result->src1 = a;
  result->opt[0] = b;
  return result;
}

// ne_mul_qkv

struct ne_tensor* ne_mul_qkv(struct ne_context* ctx, struct ne_tensor* qw, struct ne_tensor* kw, struct ne_tensor* vw,
                             struct ne_tensor* src) {
  NE_ASSERT(ne_can_mul_mat(src, qw));
  NE_ASSERT(ne_can_mul_mat(src, kw));
  NE_ASSERT(ne_can_mul_mat(src, vw));
  NE_ASSERT(ne_are_same_shape(qw, kw));
  NE_ASSERT(ne_are_same_shape(qw, vw));
  NE_ASSERT(!ne_is_transposed(src));

  bool is_node = false;

  if (src->grad || qw->grad || vw->grad || kw->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {qw->ne[1], src->ne[1], src->ne[2] * 3, src->ne[3]};
  struct ne_tensor* result = ne_new_tensor(ctx, NE_TYPE_F32, MIN(src->n_dims, qw->n_dims), ne, NE_SIZE_CALC);

  result->op = NE_OP_MUL_QKV;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = src;
  result->src1 = qw;
  result->opt[0] = kw;
  result->opt[1] = vw;

  return result;
}

// src -w1-> tmp -> silu -> tmp
//     -w3-> tmp1               -mul->tmp -w2-> dst
struct ne_tensor* ne_ffn_silu(struct ne_context* ctx, struct ne_tensor* w1, struct ne_tensor* w2, struct ne_tensor* w3,
                              struct ne_tensor* src) {
  NE_ASSERT(ne_are_same_shape(w1, w3));
  NE_ASSERT(w2->ne[0] == w1->ne[1]);

  bool is_node = false;

  if (src->grad || w1->grad || w2->grad || w3->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {w2->ne[1], src->ne[1], src->ne[2], src->ne[3]};
  struct ne_tensor* result = ne_new_tensor(ctx, NE_TYPE_F32, src->n_dims, ne, NE_SIZE_CALC);
  const int64_t tne[4] = {w1->ne[1], src->ne[1], src->ne[2], src->ne[3]};
  struct ne_tensor* tmp = ne_new_tensor(ctx, NE_TYPE_F32, src->n_dims, tne, NE_SIZE_CALC);
  struct ne_tensor* tmp1 = ne_new_tensor(ctx, NE_TYPE_F32, src->n_dims, tne, NE_SIZE_CALC);

  result->op = NE_OP_MUL_FFN_SILU;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = src;
  result->src1 = w1;
  result->opt[0] = w2;
  result->opt[1] = w3;
  result->opt[2] = tmp;
  result->opt[3] = tmp1;
  return result;
}

struct ne_tensor* ne_ffn_add_gelu(struct ne_context* ctx, struct ne_tensor* w1, struct ne_tensor* w2,
                                  struct ne_tensor* b1, struct ne_tensor* b2, struct ne_tensor* src) {
  NE_ASSERT(w2->ne[0] == w1->ne[1]);

  bool is_node = false;

  if (src->grad || w1->grad || w2->grad || b1->grad || b2->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {w2->ne[1], src->ne[1], src->ne[2], src->ne[3]};
  struct ne_tensor* result = ne_new_tensor(ctx, NE_TYPE_F32, src->n_dims, ne, NE_SIZE_CALC);
  const int64_t tne[4] = {w1->ne[1], src->ne[1], src->ne[2], src->ne[3]};
  struct ne_tensor* tmp = ne_new_tensor(ctx, NE_TYPE_F32, src->n_dims, tne, NE_SIZE_CALC);

  result->op = NE_OP_MUL_FFN_ADD_GELU;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = src;
  result->src1 = w1;
  result->opt[0] = w2;
  result->opt[1] = b1;
  result->opt[2] = b2;
  result->opt[3] = tmp;
  return result;
}

struct ne_tensor* ne_ffn_gelu(struct ne_context* ctx, struct ne_tensor* w1, struct ne_tensor* w2,
                              struct ne_tensor* src) {
  NE_ASSERT(w2->ne[0] == w1->ne[1]);

  bool is_node = false;

  if (src->grad || w1->grad || w2->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {w2->ne[1], src->ne[1], src->ne[2], src->ne[3]};
  struct ne_tensor* result = ne_new_tensor(ctx, NE_TYPE_F32, src->n_dims, ne, NE_SIZE_CALC);
  const int64_t tne[4] = {w1->ne[1], src->ne[1], src->ne[2], src->ne[3]};
  struct ne_tensor* tmp = ne_new_tensor(ctx, NE_TYPE_F32, src->n_dims, tne, NE_SIZE_CALC);

  result->op = NE_OP_MUL_FFN_GELU;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = src;
  result->src1 = w1;
  result->opt[0] = w2;
  result->opt[1] = tmp;
  return result;
}

// ne_scale

struct ne_tensor* ne_scale_impl(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, bool inplace) {
  NE_ASSERT(ne_is_scalar(b));
  NE_ASSERT(ne_is_padded_1d(a));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_SCALE;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct ne_tensor* ne_scale(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_scale_impl(ctx, a, b, false);
}

struct ne_tensor* ne_scale_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_scale_impl(ctx, a, b, true);
}

// ne_set

struct ne_tensor* ne_set_impl(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1, size_t nb2,
                              size_t nb3, size_t offset, bool inplace) {
  NE_ASSERT(ne_nelements(a) >= ne_nelements(b));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  // make a view of the destination
  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  ne_scratch_save(ctx);

  struct ne_tensor* c = ne_new_tensor_1d(ctx, NE_TYPE_I32, 5, NE_SIZE_CALC);

  ((int32_t*)c->data)[0] = nb1;
  ((int32_t*)c->data)[1] = nb2;
  ((int32_t*)c->data)[2] = nb3;
  ((int32_t*)c->data)[3] = offset;
  ((int32_t*)c->data)[4] = inplace ? 1 : 0;

  ne_scratch_load(ctx);

  result->op = NE_OP_SET;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;
  result->opt[0] = c;

  return result;
}

struct ne_tensor* ne_set(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1, size_t nb2,
                         size_t nb3, size_t offset) {
  return ne_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

struct ne_tensor* ne_set_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1,
                                 size_t nb2, size_t nb3, size_t offset) {
  return ne_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

struct ne_tensor* ne_set_1d(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t offset) {
  return ne_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

struct ne_tensor* ne_set_1d_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t offset) {
  return ne_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

struct ne_tensor* ne_set_2d(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1,
                            size_t offset) {
  return ne_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

struct ne_tensor* ne_set_2d_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1,
                                    size_t offset) {
  return ne_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

// ne_cpy

struct ne_tensor* ne_cpy_impl(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, bool inplace) {
  NE_ASSERT(ne_nelements(a) == ne_nelements(b));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  // make a view of the destination
  struct ne_tensor* result = ne_view_tensor(ctx, b);

  result->op = NE_OP_CPY;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct ne_tensor* ne_cpy(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_cpy_impl(ctx, a, b, false);
}

struct ne_tensor* ne_cpy_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  return ne_cpy_impl(ctx, a, b, true);
}

// ne_cont

struct ne_tensor* ne_cont_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (!inplace && a->grad) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_CONT;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_cont(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_cont_impl(ctx, a, false);
}

struct ne_tensor* ne_cont_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_cont_impl(ctx, a, true);
}

// ne_reshape

struct ne_tensor* ne_reshape(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  NE_ASSERT(ne_is_contiguous(a));
  NE_ASSERT(ne_is_contiguous(b));
  NE_ASSERT(ne_nelements(a) == ne_nelements(b));

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  if (b->grad) {
    // gradient propagation is not supported
    // NE_ASSERT(false);
  }

  struct ne_tensor* result = ne_new_tensor_impl(ctx, a->type, b->n_dims, b->ne, a->data, NE_SIZE_CALC);

  result->op = NE_OP_RESHAPE;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_reshape_1d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0) {
  NE_ASSERT(ne_is_contiguous(a));
  NE_ASSERT(ne_nelements(a) == ne0);

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[1] = {ne0};
  struct ne_tensor* result = ne_new_tensor_impl(ctx, a->type, 1, ne, a->data, NE_SIZE_CALC);

  result->op = NE_OP_RESHAPE;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_reshape_2d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1) {
  NE_ASSERT(ne_is_contiguous(a));
  NE_ASSERT(ne_nelements(a) == ne0 * ne1);

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[2] = {ne0, ne1};
  struct ne_tensor* result = ne_new_tensor_impl(ctx, a->type, 2, ne, a->data, NE_SIZE_CALC);

  result->op = NE_OP_RESHAPE;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_reshape_3d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2) {
  NE_ASSERT(ne_is_contiguous(a));
  NE_ASSERT(ne_nelements(a) == ne0 * ne1 * ne2);

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[3] = {ne0, ne1, ne2};
  struct ne_tensor* result = ne_new_tensor_impl(ctx, a->type, 3, ne, a->data, NE_SIZE_CALC);

  result->op = NE_OP_RESHAPE;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_reshape_4d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2,
                                int64_t ne3) {
  NE_ASSERT(ne_is_contiguous(a));
  NE_ASSERT(ne_nelements(a) == ne0 * ne1 * ne2 * ne3);

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {ne0, ne1, ne2, ne3};
  struct ne_tensor* result = ne_new_tensor_impl(ctx, a->type, 4, ne, a->data, NE_SIZE_CALC);

  result->op = NE_OP_RESHAPE;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

// ne_view_1d

struct ne_tensor* ne_view_1d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, size_t offset) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ne_tensor* result = ne_new_tensor_impl(ctx, a->type, 1, &ne0, (char*)a->data + offset, NE_SIZE_CALC);

  result->op = NE_OP_VIEW;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  if (is_node) {
    memcpy(result->padding, &offset, sizeof(offset));
  }

  return result;
}

// ne_view_2d

struct ne_tensor* ne_view_2d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1, size_t nb1,
                             size_t offset) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[NE_MAX_DIMS] = {ne0, ne1, 1, 1};

  struct ne_tensor* result = ne_new_tensor_impl(ctx, a->type, 2, ne, (char*)a->data + offset, NE_SIZE_CALC);

  result->nb[1] = nb1;
  result->nb[2] = result->nb[1] * ne1;
  result->nb[3] = result->nb[2];

  result->op = NE_OP_VIEW;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  if (is_node) {
    memcpy(result->padding, &offset, sizeof(offset));
  }

  return result;
}

// ne_view_3d

struct ne_tensor* ne_view_3d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2,
                             size_t nb1, size_t nb2, size_t offset) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[NE_MAX_DIMS] = {ne0, ne1, ne2, 1};

  struct ne_tensor* result = ne_new_tensor_impl(ctx, a->type, 3, ne, (char*)a->data + offset, NE_SIZE_CALC);

  result->nb[1] = nb1;
  result->nb[2] = nb2;
  result->nb[3] = result->nb[2] * ne2;

  result->op = NE_OP_VIEW;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  if (is_node) {
    memcpy(result->padding, &offset, sizeof(offset));
  }

  return result;
}

// ne_view_4d

struct ne_tensor* ne_view_4d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2,
                             int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[NE_MAX_DIMS] = {ne0, ne1, ne2, ne3};

  struct ne_tensor* result = ne_new_tensor_impl(ctx, a->type, 4, ne, (char*)a->data + offset, NE_SIZE_CALC);

  result->nb[1] = nb1;
  result->nb[2] = nb2;
  result->nb[3] = nb3;

  result->op = NE_OP_VIEW;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  if (is_node) {
    memcpy(result->padding, &offset, sizeof(offset));
  }

  return result;
}

// ne_permute

struct ne_tensor* ne_permute(struct ne_context* ctx, struct ne_tensor* a, int axis0, int axis1, int axis2, int axis3) {
  NE_ASSERT(axis0 >= 0 && axis0 < NE_MAX_DIMS);
  NE_ASSERT(axis1 >= 0 && axis1 < NE_MAX_DIMS);
  NE_ASSERT(axis2 >= 0 && axis2 < NE_MAX_DIMS);
  NE_ASSERT(axis3 >= 0 && axis3 < NE_MAX_DIMS);

  NE_ASSERT(axis0 != axis1);
  NE_ASSERT(axis0 != axis2);
  NE_ASSERT(axis0 != axis3);
  NE_ASSERT(axis1 != axis2);
  NE_ASSERT(axis1 != axis3);
  NE_ASSERT(axis2 != axis3);

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ne_tensor* result = ne_view_tensor(ctx, a);

  int ne[NE_MAX_DIMS];
  int nb[NE_MAX_DIMS];

  ne[axis0] = a->ne[0];
  ne[axis1] = a->ne[1];
  ne[axis2] = a->ne[2];
  ne[axis3] = a->ne[3];

  nb[axis0] = a->nb[0];
  nb[axis1] = a->nb[1];
  nb[axis2] = a->nb[2];
  nb[axis3] = a->nb[3];

  result->ne[0] = ne[0];
  result->ne[1] = ne[1];
  result->ne[2] = ne[2];
  result->ne[3] = ne[3];

  result->nb[0] = nb[0];
  result->nb[1] = nb[1];
  result->nb[2] = nb[2];
  result->nb[3] = nb[3];

  result->op = NE_OP_PERMUTE;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  if (is_node) {
    result->padding[0] = axis0;
    result->padding[1] = axis1;
    result->padding[2] = axis2;
    result->padding[3] = axis3;
  }

  return result;
}

// ne_transpose

struct ne_tensor* ne_transpose(struct ne_context* ctx, struct ne_tensor* a) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ne_tensor* result = ne_view_tensor(ctx, a);

  result->ne[0] = a->ne[1];
  result->ne[1] = a->ne[0];

  result->nb[0] = a->nb[1];
  result->nb[1] = a->nb[0];

  result->op = NE_OP_TRANSPOSE;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

// ne_get_rows

struct ne_tensor* ne_get_rows(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  NE_ASSERT(ne_is_matrix(a) && ne_is_vector(b) && b->type == NE_TYPE_I32);

  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  // TODO: implement non F32 return
  // struct ne_tensor * result = ne_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
  struct ne_tensor* result = ne_new_tensor_2d(ctx, NE_TYPE_F32, a->ne[0], b->ne[0], NE_SIZE_CALC);

  result->op = NE_OP_GET_ROWS;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

// ne_get_rows_back

struct ne_tensor* ne_get_rows_back(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b,
                                   struct ne_tensor* c) {
  NE_ASSERT(ne_is_matrix(a) && ne_is_vector(b) && b->type == NE_TYPE_I32);
  NE_ASSERT(ne_is_matrix(c) && (a->ne[0] == c->ne[0]));

  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  // TODO: implement non F32 return
  // struct ne_tensor * result = ne_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
  struct ne_tensor* result = ne_new_tensor_2d(ctx, NE_TYPE_F32, c->ne[0], c->ne[1], NE_SIZE_CALC);

  result->op = NE_OP_GET_ROWS_BACK;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;
  result->opt[0] = c;

  return result;
}

// ne_diag

struct ne_tensor* ne_diag(struct ne_context* ctx, struct ne_tensor* a) {
  NE_ASSERT(a->ne[1] == 1);
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {a->ne[0], a->ne[0], a->ne[2], a->ne[3]};
  struct ne_tensor* result = ne_new_tensor(ctx, a->type, MAX(a->n_dims, 2), ne, NE_SIZE_CALC);

  result->op = NE_OP_DIAG;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

// ne_diag_mask_inf

struct ne_tensor* ne_diag_mask_inf_impl(struct ne_context* ctx, struct ne_tensor* a, int n_past, bool inplace) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  ne_scratch_save(ctx);

  struct ne_tensor* b = ne_new_tensor_1d(ctx, NE_TYPE_I32, 2, NE_SIZE_CALC);

  ((int32_t*)b->data)[0] = n_past;
  ((int32_t*)b->data)[1] = inplace ? 1 : 0;

  ne_scratch_load(ctx);

  result->op = NE_OP_DIAG_MASK_INF;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct ne_tensor* ne_diag_mask_inf(struct ne_context* ctx, struct ne_tensor* a, int n_past) {
  return ne_diag_mask_inf_impl(ctx, a, n_past, false);
}

struct ne_tensor* ne_diag_mask_inf_inplace(struct ne_context* ctx, struct ne_tensor* a, int n_past) {
  return ne_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ne_diag_mask_zero

struct ne_tensor* ne_diag_mask_zero_impl(struct ne_context* ctx, struct ne_tensor* a, int n_past, bool inplace) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  ne_scratch_save(ctx);

  struct ne_tensor* b = ne_new_tensor_1d(ctx, NE_TYPE_I32, 2, NE_SIZE_CALC);
  ne_set_name(b, "n_past, inplace");

  ((int32_t*)b->data)[0] = n_past;
  ((int32_t*)b->data)[1] = inplace ? 1 : 0;

  ne_scratch_load(ctx);

  result->op = NE_OP_DIAG_MASK_ZERO;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct ne_tensor* ne_diag_mask_zero(struct ne_context* ctx, struct ne_tensor* a, int n_past) {
  return ne_diag_mask_zero_impl(ctx, a, n_past, false);
}

struct ne_tensor* ne_diag_mask_zero_inplace(struct ne_context* ctx, struct ne_tensor* a, int n_past) {
  return ne_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ne_soft_max

struct ne_tensor* ne_soft_max_impl(struct ne_context* ctx, struct ne_tensor* a, bool inplace) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_SOFT_MAX;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = NULL;

  return result;
}

struct ne_tensor* ne_soft_max(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_soft_max_impl(ctx, a, false);
}

struct ne_tensor* ne_soft_max_inplace(struct ne_context* ctx, struct ne_tensor* a) {
  return ne_soft_max_impl(ctx, a, true);
}

// ne_rope

struct ne_tensor* ne_rope_impl(struct ne_context* ctx, struct ne_tensor* a, int n_past, int n_dims, int mode,
                               bool inplace) {
  NE_ASSERT(n_past >= 0);
  bool is_node = false;

  if (!inplace && a->grad) {
    is_node = true;
  }

  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  ne_scratch_save(ctx);

  struct ne_tensor* b = ne_new_tensor_1d(ctx, NE_TYPE_I32, 3, NE_SIZE_CALC);

  ((int32_t*)b->data)[0] = n_past;
  ((int32_t*)b->data)[1] = n_dims;
  ((int32_t*)b->data)[2] = mode;

  ne_scratch_load(ctx);

  result->op = NE_OP_ROPE;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct ne_tensor* ne_rope(struct ne_context* ctx, struct ne_tensor* a, int n_past, int n_dims, int mode) {
  return ne_rope_impl(ctx, a, n_past, n_dims, mode, false);
}

struct ne_tensor* ne_rope_inplace(struct ne_context* ctx, struct ne_tensor* a, int n_past, int n_dims, int mode) {
  return ne_rope_impl(ctx, a, n_past, n_dims, mode, true);
}

// ne_rope_back

struct ne_tensor* ne_rope_back(struct ne_context* ctx, struct ne_tensor* a, int n_past, int n_dims, int mode) {
  NE_ASSERT(n_past >= 0);
  bool is_node = false;

  if (a->grad) {
    NE_ASSERT(false);  // TODO: implement backward
    is_node = true;
  }

  struct ne_tensor* result = ne_dup_tensor(ctx, a);

  ne_scratch_save(ctx);

  struct ne_tensor* b = ne_new_tensor_1d(ctx, NE_TYPE_I32, 3, NE_SIZE_CALC);
  ne_set_name(b, "n_past, n_dims, mode");

  ((int32_t*)b->data)[0] = n_past;
  ((int32_t*)b->data)[1] = n_dims;
  ((int32_t*)b->data)[2] = mode;

  ne_scratch_load(ctx);

  result->op = NE_OP_ROPE_BACK;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

// ne_alibi

struct ne_tensor* ne_alibi(struct ne_context* ctx, struct ne_tensor* a, int n_past, int n_head, float bias_max) {
  NE_ASSERT(n_past >= 0);
  bool is_node = false;

  if (a->grad) {
    NE_ASSERT(false);  // TODO: implement backward
    is_node = true;
  }

  // TODO: when implement backward, fix this:
  // struct ne_tensor * result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);
  struct ne_tensor* result = ne_view_tensor(ctx, a);

  ne_scratch_save(ctx);

  struct ne_tensor* b = ne_new_tensor_1d(ctx, NE_TYPE_I32, 3, NE_SIZE_CALC);

  ((int32_t*)b->data)[0] = n_past;
  ((int32_t*)b->data)[1] = n_head;
  NE_ASSERT(sizeof(float) == sizeof(int32_t));
  (((float*)b->data)[2]) = bias_max;

  ne_scratch_load(ctx);

  result->op = NE_OP_ALIBI;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

// ne_clamp

struct ne_tensor* ne_clamp(struct ne_context* ctx, struct ne_tensor* a, float min, float max) {
  bool is_node = false;

  if (a->grad) {
    NE_ASSERT(false);  // TODO: implement backward
    is_node = true;
  }

  // TODO: when implement backward, fix this:
  struct ne_tensor* result = ne_view_tensor(ctx, a);

  ne_scratch_save(ctx);

  struct ne_tensor* b = ne_new_tensor_1d(ctx, NE_TYPE_I32, 3, NE_SIZE_CALC);

  ((float*)b->data)[0] = min;
  ((float*)b->data)[1] = max;

  ne_scratch_load(ctx);

  result->op = NE_OP_CLAMP;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

// ne_conv_1d_1s

struct ne_tensor* ne_conv_1d_1s(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  NE_ASSERT(ne_is_matrix(b));
  NE_ASSERT(a->ne[1] == b->ne[1]);
  NE_ASSERT(a->ne[3] == 1);
  bool is_node = false;

  if (a->grad || b->grad) {
    NE_ASSERT(false);  // TODO: implement backward
    is_node = true;
  }

  const int64_t ne[4] = {
      b->ne[0],
      a->ne[2],
      1,
      1,
  };
  struct ne_tensor* result = ne_new_tensor(ctx, NE_TYPE_F32, 2, ne, NE_SIZE_CALC);

  result->op = NE_OP_CONV_1D_1S;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

// ne_conv_1d_2s

struct ne_tensor* ne_conv_1d_2s(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b) {
  NE_ASSERT(ne_is_matrix(b));
  NE_ASSERT(a->ne[1] == b->ne[1]);
  NE_ASSERT(a->ne[3] == 1);
  bool is_node = false;

  if (a->grad || b->grad) {
    NE_ASSERT(false);  // TODO: implement backward
    is_node = true;
  }

  const int64_t ne[4] = {
      b->ne[0] / 2,
      a->ne[2],
      1,
      1,
  };
  struct ne_tensor* result = ne_new_tensor(ctx, NE_TYPE_F32, 2, ne, NE_SIZE_CALC);

  result->op = NE_OP_CONV_1D_2S;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

// ne_flash_attn

struct ne_tensor* ne_flash_attn(struct ne_context* ctx, struct ne_tensor* q, struct ne_tensor* k, struct ne_tensor* v,
                                float scale, bool masked) {
  NE_ASSERT(ne_can_mul_mat(k, q));
  int batch = q->ne[3];
  int headsize = q->ne[0];
  int headnum = q->ne[2];
  int seq_cur = q->ne[1];
  int seq_all = k->ne[1];
  int seq_past = seq_all - seq_cur;
  NE_ASSERT(headsize == v->ne[1]);
  NE_ASSERT(seq_all == v->ne[0]);
  NE_ASSERT(headnum == v->ne[2]);
  NE_ASSERT(batch == v->ne[3]);
  bool is_node = true;
  struct ne_tensor* result = ne_new_tensor_4d(ctx, NE_TYPE_F32, headsize, headnum, seq_cur, batch, NE_SIZE_CALC);
  attn_shape_t atte_shape = {batch, headnum, headsize, seq_cur, seq_all};
  size_t tmpsize = jblas_fusion_attn_bf16_workspace_size(&atte_shape);
  struct ne_tensor* tmp_t = ne_new_tensor_1d(ctx, NE_TYPE_I8, tmpsize, NE_SIZE_CALC);
  result->op = NE_OP_FLASH_ATTN;
  result->grad = NULL;
  result->src0 = q;
  result->src1 = k;
  result->opt[0] = v;
  result->opt[1] = tmp_t;
  *(float*)result->padding = scale;
  *(bool*)&result->padding[sizeof(scale)] = masked;

  return result;
}

// ne_flash_ff

struct ne_tensor* ne_flash_ff(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b0, struct ne_tensor* b1,
                              struct ne_tensor* c0, struct ne_tensor* c1) {
  NE_ASSERT(ne_can_mul_mat(b0, a));
  // TODO: more checks

  bool is_node = false;

  if (a->grad || b0->grad || b1->grad || c0->grad || c1->grad) {
    NE_ASSERT(false);  // TODO: implement backward
    is_node = true;
  }

  // struct ne_tensor * result = ne_dup_tensor(ctx, a);
  struct ne_tensor* result = ne_new_tensor(ctx, NE_TYPE_F32, 4, a->ne, NE_SIZE_CALC);

  result->op = NE_OP_FLASH_FF;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b0;
  result->opt[0] = b1;
  result->opt[1] = c0;
  result->opt[2] = c1;

  return result;
}

// ne_map_unary

struct ne_tensor* ne_map_unary_impl_f32(struct ne_context* ctx, struct ne_tensor* a, const ne_unary_op_f32_t fun,
                                        bool inplace) {
  bool is_node = false;

  if (!inplace && a->grad) {
    is_node = true;
  }

  struct ne_tensor* addr_tensor = ne_new_tensor_1d(ctx, NE_TYPE_I32, sizeof(void*) / sizeof(int32_t), NE_SIZE_CALC);
  *((void (**)(void))addr_tensor->data) = (void (*)(void))fun;
  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_MAP_UNARY;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->opt[0] = addr_tensor;

  return result;
}

struct ne_tensor* ne_map_unary_f32(struct ne_context* ctx, struct ne_tensor* a, const ne_unary_op_f32_t fun) {
  return ne_map_unary_impl_f32(ctx, a, fun, false);
}

struct ne_tensor* ne_map_unary_inplace_f32(struct ne_context* ctx, struct ne_tensor* a, const ne_unary_op_f32_t fun) {
  return ne_map_unary_impl_f32(ctx, a, fun, true);
}

// ne_map_binary

struct ne_tensor* ne_map_binary_impl_f32(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b,
                                         const ne_binary_op_f32_t fun, bool inplace) {
  NE_ASSERT(ne_are_same_shape(a, b));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  struct ne_tensor* addr_tensor = ne_new_tensor_1d(ctx, NE_TYPE_I32, sizeof(void*) / sizeof(int32_t), NE_SIZE_CALC);
  *((void (**)(void))addr_tensor->data) = (void (*)(void))fun;
  struct ne_tensor* result = inplace ? ne_view_tensor(ctx, a) : ne_dup_tensor(ctx, a);

  result->op = NE_OP_MAP_BINARY;
  result->grad = is_node ? ne_dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;
  result->opt[0] = addr_tensor;

  return result;
}

struct ne_tensor* ne_map_binary_f32(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b,
                                    const ne_binary_op_f32_t fun) {
  return ne_map_binary_impl_f32(ctx, a, b, fun, false);
}

struct ne_tensor* ne_map_binary_inplace_f32(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b,
                                            const ne_binary_op_f32_t fun) {
  return ne_map_binary_impl_f32(ctx, a, b, fun, true);
}

////////////////////////////////////////////////////////////////////////////////

void ne_set_param(struct ne_context* ctx, struct ne_tensor* tensor) {
  tensor->is_param = true;

  NE_ASSERT(tensor->grad == NULL);
  tensor->grad = ne_dup_tensor(ctx, tensor);
}

// ne_compute_forward_dup

static void ne_compute_forward_dup_same_cont(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                             struct ne_tensor* dst) {
  NE_ASSERT(ne_nelements(dst) == ne_nelements(src0));
  NE_ASSERT(ne_is_contiguous(dst) && ne_is_contiguous(src0));
  NE_ASSERT(src0->type == dst->type);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const size_t nb00 = src0->nb[0];
  const size_t nb0 = dst->nb[0];

  const int ith = params->ith;  // thread index
  const int nth = params->nth;  // number of threads

  // parallelize by elements
  const int ne = ne_nelements(dst);
  const int dr = (ne + nth - 1) / nth;
  const int ie0 = dr * ith;
  const int ie1 = MIN(ie0 + dr, ne);

  if (ie0 < ie1) {
    memcpy(((char*)dst->data + ie0 * nb0), ((char*)src0->data + ie0 * nb00), (ie1 - ie0) * NE_TYPE_SIZE[src0->type]);
  }
}
static void ne_compute_forward_dup_f16(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       struct ne_tensor* dst) {
  NE_ASSERT(ne_nelements(dst) == ne_nelements(src0));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int ith = params->ith;  // thread index
  const int nth = params->nth;  // number of threads

  if (ne_is_contiguous(src0) && ne_is_contiguous(dst) && src0->type == dst->type) {
    ne_compute_forward_dup_same_cont(params, src0, dst);
    return;
  }

  // parallelize by rows
  const int nr = ne01;
  // number of rows per thread
  const int dr = (nr + nth - 1) / nth;
  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (src0->type == dst->type && ne00 == ne0 && nb00 == NE_TYPE_SIZE[src0->type] && nb0 == NE_TYPE_SIZE[dst->type]) {
    // copy by rows
    const size_t rs = ne00 * nb00;
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          memcpy(((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                 ((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03), rs);
        }
      }
    }
    return;
  }

  // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

  if (ne_is_contiguous(dst)) {
    if (nb00 == sizeof(ne_fp16_t)) {
      if (dst->type == NE_TYPE_F16) {
        size_t id = 0;
        const size_t rs = ne00 * nb00;
        char* dst_ptr = (char*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const char* src0_ptr = (char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
              memcpy(dst_ptr + id, src0_ptr, rs);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else if (dst->type == NE_TYPE_F32) {
        size_t id = 0;
        float* dst_ptr = (float*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const ne_fp16_t* src0_ptr = (ne_fp16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
              for (int i00 = 0; i00 < ne00; i00++) {
                dst_ptr[id] = NE_FP16_TO_FP32(src0_ptr[i00]);
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else if (ne_is_quantized(dst->type)) {
        quantize_row_q_t const quantize_row_q = quantize_fns[dst->type].quantize_row_q;
        float* src0_f32 = (float*)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / NE_BLCK_SIZE[dst->type]);
        char* dst_ptr = (char*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const ne_fp16_t* src0_ptr = (ne_fp16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

              for (int i00 = 0; i00 < ne00; i00++) {
                src0_f32[i00] = NE_FP16_TO_FP32(src0_ptr[i00]);
              }

              quantize_row_q(src0_f32, dst_ptr + id, ne00);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else {
        NE_ASSERT(false);  // TODO: implement
      }
    } else {
      // printf("%s: this is not optimal - fix me\n", __func__);

      if (dst->type == NE_TYPE_F32) {
        size_t id = 0;
        float* dst_ptr = (float*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const ne_fp16_t* src0_ptr =
                    (ne_fp16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = NE_FP16_TO_FP32(*src0_ptr);
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else if (dst->type == NE_TYPE_F16) {
        size_t id = 0;
        ne_fp16_t* dst_ptr = (ne_fp16_t*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const ne_fp16_t* src0_ptr =
                    (ne_fp16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = *src0_ptr;
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else {
        NE_ASSERT(false);  // TODO: implement
      }
    }
    return;
  }

  // dst counters
  int64_t i10 = 0;
  int64_t i11 = 0;
  int64_t i12 = 0;
  int64_t i13 = 0;

  if (dst->type == NE_TYPE_F16) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            NE_ASSERT(dst_ptr != NULL);
            memcpy(dst_ptr, src0_ptr, sizeof(ne_fp16_t));

            if (++i10 == ne00) {
              i10 = 0;
              if (++i11 == ne01) {
                i11 = 0;
                if (++i12 == ne02) {
                  i12 = 0;
                  if (++i13 == ne03) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else if (dst->type == NE_TYPE_F32) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            *(float*)dst_ptr = NE_FP16_TO_FP32(*(const ne_fp16_t*)src0_ptr);

            if (++i10 == ne0) {
              i10 = 0;
              if (++i11 == ne1) {
                i11 = 0;
                if (++i12 == ne2) {
                  i12 = 0;
                  if (++i13 == ne3) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else {
    NE_ASSERT(false);  // TODO: implement
  }
}

static void ne_compute_forward_dup_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       struct ne_tensor* dst) {
  NE_ASSERT(ne_nelements(dst) == ne_nelements(src0));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int ith = params->ith;  // thread index
  const int nth = params->nth;  // number of threads

  if (ne_is_contiguous(src0) && ne_is_contiguous(dst) && src0->type == dst->type) {
    ne_compute_forward_dup_same_cont(params, src0, dst);
    return;
  }

  // parallelize by rows
  const int nr = ne01;
  // number of rows per thread
  const int dr = (nr + nth - 1) / nth;
  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (src0->type == dst->type && ne00 == ne0 && nb00 == NE_TYPE_SIZE[src0->type] && nb0 == NE_TYPE_SIZE[dst->type]) {
    // copy by rows
    const size_t rs = ne00 * nb00;
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          memcpy(((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                 ((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03), rs);
        }
      }
    }
    return;
  }

  if (ne_is_contiguous(dst)) {
    // TODO: simplify
    if (nb00 == sizeof(float)) {
      if (dst->type == NE_TYPE_F32) {
        size_t id = 0;
        const size_t rs = ne00 * nb00;
        char* dst_ptr = (char*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const char* src0_ptr = (char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
              memcpy(dst_ptr + id, src0_ptr, rs);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else if (dst->type == NE_TYPE_F16) {
        if (ne_is_contiguous(src0)) {  // fp32->fp16 conversion
          int nele = ne_nelements(src0);
          // number of rows per thread
          int dn = (nele + nth - 1) / nth;
          // row range for this thread
          int in0 = dn * ith;
          int in1 = MIN(in0 + dn, nele);
          float* srcptr = (float*)src0->data;
          ne_fp16_t* dstptr = (ne_fp16_t*)dst->data;
          for (int i = in0; i < in1; i++) {
            dstptr[i] = NE_FP32_TO_FP16(srcptr[i]);
          }
        } else {
          size_t id = 0;
          ne_fp16_t* dst_ptr = (ne_fp16_t*)dst->data;
          for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
              id += ne00 * ir0;
              for (int i01 = ir0; i01 < ir1; i01++) {
                for (int i00 = 0; i00 < ne00; i00++) {
                  const float* src0_ptr =
                      (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                  dst_ptr[id] = NE_FP32_TO_FP16(*src0_ptr);
                  id++;
                }
              }
              id += ne00 * (ne01 - ir1);
            }
          }
        }

      } else if (ne_is_quantized(dst->type)) {
        quantize_row_q_t const quantize_row_q = quantize_fns[dst->type].quantize_row_q;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / NE_BLCK_SIZE[dst->type]);
        char* dst_ptr = (char*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const float* src0_ptr = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
              quantize_row_q(src0_ptr, dst_ptr + id, ne00);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else {
        NE_ASSERT(false);  // TODO: implement
      }
    } else {
      // printf("%s: this is not optimal - fix me\n", __func__);

      if (dst->type == NE_TYPE_F32) {
        size_t id = 0;
        float* dst_ptr = (float*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const float* src0_ptr = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = *src0_ptr;
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else if (dst->type == NE_TYPE_F16) {
        size_t id = 0;
        ne_fp16_t* dst_ptr = (ne_fp16_t*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const float* src0_ptr = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = NE_FP32_TO_FP16(*src0_ptr);
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else {
        NE_ASSERT(false);  // TODO: implement
      }
    }

    return;
  }

  // dst counters

  int64_t i10 = 0;
  int64_t i11 = 0;
  int64_t i12 = 0;
  int64_t i13 = 0;

  if (dst->type == NE_TYPE_F32) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            NE_ASSERT(dst_ptr != NULL);
            memcpy(dst_ptr, src0_ptr, sizeof(float));

            if (++i10 == ne0) {
              i10 = 0;
              if (++i11 == ne1) {
                i11 = 0;
                if (++i12 == ne2) {
                  i12 = 0;
                  if (++i13 == ne3) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else if (dst->type == NE_TYPE_F16) {
    bool dst_contiguous = nb0 < nb1 && nb1 < nb2 && nb2 < nb3;
    bool src_perm1203 = nb01 < nb02 && nb02 < nb00 && nb00 < nb03;
    if (dst_contiguous && src_perm1203) {  // number of rows per thread
      int nele = ne1 * ne2;
      int dn = (nele + nth - 1) / nth;
      // row range for this thread
      int in0 = dn * ith;
      int in1 = MIN(in0 + dn, nele);

      for (int ib = 0; ib < ne3; ib++) {
        float* srcptr = (float*)((char*)src0->data + ib * nb03);
        ne_fp16_t* dstptr = (ne_fp16_t*)((char*)dst->data + ib * nb3);
        for (int j = 0; j < ne0; j++) {
          for (int i = in0; i < in1; i++) {
            dstptr[i * nb1 / sizeof(*dstptr) + j] = NE_FP32_TO_FP16(srcptr[i + j * nb00 / sizeof(*srcptr)]);
          }
        }
      }
    } else {
      for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
          i10 += ne00 * ir0;
          while (i10 >= ne0) {
            i10 -= ne0;
            if (++i11 == ne1) {
              i11 = 0;
              if (++i12 == ne2) {
                i12 = 0;
                if (++i13 == ne3) {
                  i13 = 0;
                }
              }
            }
          }
          for (int64_t i01 = ir0; i01 < ir1; i01++) {
            for (int64_t i00 = 0; i00 < ne00; i00++) {
              const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
              char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

              *(ne_fp16_t*)dst_ptr = NE_FP32_TO_FP16(*(const float*)src0_ptr);

              if (++i10 == ne0) {
                i10 = 0;
                if (++i11 == ne1) {
                  i11 = 0;
                  if (++i12 == ne2) {
                    i12 = 0;
                    if (++i13 == ne3) {
                      i13 = 0;
                    }
                  }
                }
              }
            }
          }
          i10 += ne00 * (ne01 - ir1);
          while (i10 >= ne0) {
            i10 -= ne0;
            if (++i11 == ne1) {
              i11 = 0;
              if (++i12 == ne2) {
                i12 = 0;
                if (++i13 == ne3) {
                  i13 = 0;
                }
              }
            }
          }
        }
      }
    }

  } else {
    NE_ASSERT(false);  // TODO: implement
  }
}

static void ne_compute_forward_dup(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   struct ne_tensor* dst) {
  if (ne_is_contiguous(src0) && ne_is_contiguous(dst) && src0->type == dst->type) {
    ne_compute_forward_dup_same_cont(params, src0, dst);
    return;
  }
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_dup_f16(params, src0, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_dup_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_add

static void ne_compute_forward_add_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_can_repeat_rows(src1, src0) && ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t nr = ne_nrows(src0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = ne11 == 1 ? 0 : src1->nb[1];
  const size_t nb12 = ne12 == 1 ? 0 : src1->nb[2];
  const size_t nb13 = ne13 == 1 ? 0 : src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb00 == sizeof(float));
  NE_ASSERT(ne00 == ne10);

  if (nb10 == sizeof(float)) {
    for (int64_t ir = ith; ir < nr; ir += nth) {
      // src0 and dst are same shape => same indices
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
      float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

      ne_vec_add_f32(ne00, dst_ptr, src0_ptr, src1_ptr);
    }
  } else {
    // src1 is not contiguous
    for (int64_t ir = ith; ir < nr; ir += nth) {
      // src0 and dst are same shape => same indices
      // src1 is broadcastable across src0 and dst in i1, i2, i3
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);

      for (int64_t i0 = 0; i0 < ne00; i0++) {
        float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
      }
    }
  }
}

static void ne_compute_forward_add_f16_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                           const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, src1) && ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(src0);
  const int64_t ne0 = src0->ne[0];
  const int64_t ne1 = src0->ne[1];
  const int64_t ne2 = src0->ne[2];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  NE_ASSERT(src0->type == NE_TYPE_F16);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F16);

  NE_ASSERT(nb0 == sizeof(ne_fp16_t));
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (nb10 == sizeof(float)) {
    for (int ir = ir0; ir < ir1; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      ne_fp16_t* dst_ptr = (ne_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      ne_fp16_t* src0_ptr = (ne_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

      for (int i = 0; i < ne0; i++) {
        dst_ptr[i] = NE_FP32_TO_FP16(NE_FP16_TO_FP32(src0_ptr[i]) + src1_ptr[i]);
      }
    }
  } else {
    // src1 is not contiguous
    NE_ASSERT(false);
  }
}

static void ne_compute_forward_add_f16_f16(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                           const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, src1) && ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(src0);
  const int64_t ne0 = src0->ne[0];
  const int64_t ne1 = src0->ne[1];
  const int64_t ne2 = src0->ne[2];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  NE_ASSERT(src0->type == NE_TYPE_F16);
  NE_ASSERT(src1->type == NE_TYPE_F16);
  NE_ASSERT(dst->type == NE_TYPE_F16);

  NE_ASSERT(nb0 == sizeof(ne_fp16_t));
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (nb10 == sizeof(ne_fp16_t)) {
    for (int ir = ir0; ir < ir1; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      ne_fp16_t* dst_ptr = (ne_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      ne_fp16_t* src0_ptr = (ne_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      ne_fp16_t* src1_ptr = (ne_fp16_t*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

      for (int i = 0; i < ne0; i++) {
        dst_ptr[i] = NE_FP32_TO_FP16(NE_FP16_TO_FP32(src0_ptr[i]) + NE_FP16_TO_FP32(src1_ptr[i]));
      }
    }
  } else {
    // src1 is not contiguous
    NE_ASSERT(false);
  }
}

static void ne_compute_forward_add_q_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                         const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, src1) && ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int nr = ne_nrows(src0);
  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  // const int64_t ne03 = src0->ne[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const enum ne_type type = src0->type;
  dequantize_row_q_t const dequantize_row_q = quantize_fns[type].dequantize_row_q;
  quantize_row_q_t const quantize_row_q = quantize_fns[type].quantize_row_q;

  // we don't support permuted src0 or src1
  NE_ASSERT(nb00 == NE_TYPE_SIZE[type]);
  NE_ASSERT(nb10 == sizeof(float));

  // dst cannot be transposed or permuted
  NE_ASSERT(nb0 <= nb1);
  NE_ASSERT(nb1 <= nb2);
  NE_ASSERT(nb2 <= nb3);

  NE_ASSERT(ne_is_quantized(src0->type));
  NE_ASSERT(dst->type == src0->type);
  NE_ASSERT(src1->type == NE_TYPE_F32);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  float* wdata = (float*)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 indices
    const int i03 = ir / (ne02 * ne01);
    const int i02 = (ir - i03 * ne02 * ne01) / ne01;
    const int i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

    // src1 and dst are same shape as src0 => same indices
    const int i13 = i03;
    const int i12 = i02;
    const int i11 = i01;

    const int i3 = i03;
    const int i2 = i02;
    const int i1 = i01;

    void* src0_row = (void*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
    float* src1_row = (float*)((char*)src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13));
    void* dst_row = (void*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb0));

    assert(ne00 % 32 == 0);

    // unquantize row from src0 to temp buffer
    dequantize_row_q(src0_row, wdata, ne00);
    // add src1
    ne_vec_acc_f32(ne00, wdata, src1_row);
    // quantize row to dst
    quantize_row_q(wdata, dst_row, ne00);
  }
}

static void ne_compute_forward_add(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_add_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_F16: {
      if (src1->type == NE_TYPE_F16) {
        ne_compute_forward_add_f16_f16(params, src0, src1, dst);
      } else if (src1->type == NE_TYPE_F32) {
        ne_compute_forward_add_f16_f32(params, src0, src1, dst);
      } else {
        NE_ASSERT(false);
      }
    } break;
    case NE_TYPE_Q4_0:
    case NE_TYPE_Q4_1:
    case NE_TYPE_Q5_0:
    case NE_TYPE_Q5_1:
    case NE_TYPE_Q8_0: {
      ne_compute_forward_add_q_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_add1

static void ne_compute_forward_add1_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, dst));
  NE_ASSERT(ne_is_scalar(src1));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(src0);
  const int64_t ne0 = src0->ne[0];
  const int64_t ne1 = src0->ne[1];
  const int64_t ne2 = src0->ne[2];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb00 == sizeof(float));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    ne_vec_add1_f32(ne0, (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1),
                    (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01), *(float*)src1->data);
  }
}

static void ne_compute_forward_add1_f16_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                            const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, dst));
  NE_ASSERT(ne_is_scalar(src1));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  // scalar to add
  const float v = *(float*)src1->data;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(src0);
  const int64_t ne0 = src0->ne[0];
  const int64_t ne1 = src0->ne[1];
  const int64_t ne2 = src0->ne[2];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  NE_ASSERT(src0->type == NE_TYPE_F16);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F16);

  NE_ASSERT(nb0 == sizeof(ne_fp16_t));
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    ne_fp16_t* dst_ptr = (ne_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
    ne_fp16_t* src0_ptr = (ne_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
    for (int i = 0; i < ne0; i++) {
      dst_ptr[i] = NE_FP32_TO_FP16(NE_FP16_TO_FP32(src0_ptr[i]) + v);
    }
  }
}

static void ne_compute_forward_add1_f16_f16(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                            const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, dst));
  NE_ASSERT(ne_is_scalar(src1));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  // scalar to add
  const float v = NE_FP16_TO_FP32(*(ne_fp16_t*)src1->data);

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(src0);
  const int64_t ne0 = src0->ne[0];
  const int64_t ne1 = src0->ne[1];
  const int64_t ne2 = src0->ne[2];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  NE_ASSERT(src0->type == NE_TYPE_F16);
  NE_ASSERT(src1->type == NE_TYPE_F16);
  NE_ASSERT(dst->type == NE_TYPE_F16);

  NE_ASSERT(nb0 == sizeof(ne_fp16_t));
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    ne_fp16_t* dst_ptr = (ne_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
    ne_fp16_t* src0_ptr = (ne_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
    for (int i = 0; i < ne0; i++) {
      dst_ptr[i] = NE_FP32_TO_FP16(NE_FP16_TO_FP32(src0_ptr[i]) + v);
    }
  }
}

static void ne_compute_forward_add1_q_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                          const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, dst));
  NE_ASSERT(ne_is_scalar(src1));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  // scalar to add
  const float v = *(float*)src1->data;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(src0);
  const int64_t ne0 = src0->ne[0];
  const int64_t ne1 = src0->ne[1];
  const int64_t ne2 = src0->ne[2];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const enum ne_type type = src0->type;
  dequantize_row_q_t const dequantize_row_q = quantize_fns[type].dequantize_row_q;
  quantize_row_q_t const quantize_row_q = quantize_fns[type].quantize_row_q;

  // we don't support permuted src0
  NE_ASSERT(nb00 == NE_TYPE_SIZE[type]);

  // dst cannot be transposed or permuted
  NE_ASSERT(nb0 <= nb1);
  NE_ASSERT(nb1 <= nb2);
  NE_ASSERT(nb2 <= nb3);

  NE_ASSERT(ne_is_quantized(src0->type));
  NE_ASSERT(dst->type == src0->type);
  NE_ASSERT(src1->type == NE_TYPE_F32);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  float* wdata = (float*)params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    void* src0_row = (void*)((char*)src0->data + (i1 * nb01 + i2 * nb02 + i3 * nb03));
    void* dst_row = (void*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb0));

    assert(ne0 % 32 == 0);

    // unquantize row from src0 to temp buffer
    dequantize_row_q(src0_row, wdata, ne0);
    // add src1
    ne_vec_acc1_f32(ne0, wdata, v);
    // quantize row to dst
    quantize_row_q(wdata, dst_row, ne0);
  }
}

static void ne_compute_forward_add1(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_add1_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_F16: {
      if (src1->type == NE_TYPE_F16) {
        ne_compute_forward_add1_f16_f16(params, src0, src1, dst);
      } else if (src1->type == NE_TYPE_F32) {
        ne_compute_forward_add1_f16_f32(params, src0, src1, dst);
      } else {
        NE_ASSERT(false);
      }
    } break;
    case NE_TYPE_Q4_0:
    case NE_TYPE_Q4_1:
    case NE_TYPE_Q5_0:
    case NE_TYPE_Q5_1:
    case NE_TYPE_Q8_0:
    case NE_TYPE_Q8_1: {
      ne_compute_forward_add1_q_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_acc

static void ne_compute_forward_acc_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       const struct ne_tensor* src1, const struct ne_tensor* opt0,
                                       struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, dst));
  NE_ASSERT(ne_is_contiguous(dst) && ne_is_contiguous(src0));

  NE_ASSERT(opt0->type == NE_TYPE_I32);
  NE_ASSERT(ne_nelements(opt0) == 5);

  // view src0 and dst with these strides and data offset inbytes during acc
  // nb0 is implicitely element_size because src0 and dst are contiguous
  size_t nb1 = ((int32_t*)opt0->data)[0];
  size_t nb2 = ((int32_t*)opt0->data)[1];
  size_t nb3 = ((int32_t*)opt0->data)[2];
  size_t offset = ((int32_t*)opt0->data)[3];
  bool inplace = (bool)((int32_t*)opt0->data)[4];

  if (!inplace && (params->type == NE_TASK_INIT)) {
    // memcpy needs to be synchronized across threads to avoid race conditions.
    // => do it in INIT phase
    memcpy(((char*)dst->data), ((char*)src0->data), ne_nbytes(dst));
  }

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(src1);
  const int nc = src1->ne[0];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  // src0 and dst as viewed during acc
  const size_t nb0 = ne_element_size(src0);

  const size_t nb00 = nb0;
  const size_t nb01 = nb1;
  const size_t nb02 = nb2;
  const size_t nb03 = nb3;

  NE_ASSERT(offset + (ne10 == 0 ? 0 : ne10 - 1) * nb0 + (ne11 == 0 ? 0 : ne11 - 1) * nb1 +
                (ne12 == 0 ? 0 : ne12 - 1) * nb2 + (ne13 == 0 ? 0 : ne13 - 1) * nb3 <
            ne_nbytes(dst));
  NE_ASSERT(offset + (ne10 == 0 ? 0 : ne10 - 1) * nb00 + (ne11 == 0 ? 0 : ne11 - 1) * nb01 +
                (ne12 == 0 ? 0 : ne12 - 1) * nb02 + (ne13 == 0 ? 0 : ne13 - 1) * nb03 <
            ne_nbytes(src0));

  NE_ASSERT(nb10 == sizeof(float));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are viewed with shape of src1 and offset
    // => same indices
    const int i3 = ir / (ne12 * ne11);
    const int i2 = (ir - i3 * ne12 * ne11) / ne11;
    const int i1 = (ir - i3 * ne12 * ne11 - i2 * ne11);

    ne_vec_add_f32(nc, (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + offset),
                   (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + offset),
                   (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
  }
}

static void ne_compute_forward_acc(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, const struct ne_tensor* opt0, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_acc_f32(params, src0, src1, opt0, dst);
    } break;
    case NE_TYPE_F16:
    case NE_TYPE_Q4_0:
    case NE_TYPE_Q4_1:
    case NE_TYPE_Q5_0:
    case NE_TYPE_Q5_1:
    case NE_TYPE_Q8_0:
    case NE_TYPE_Q8_1:
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_sub

static void ne_compute_forward_sub_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       const struct ne_tensor* src1, struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(ne_are_same_shape(src0, src1) && ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int nr = ne_nrows(src0);
  const int64_t ne0 = src0->ne[0];
  const int64_t ne1 = src0->ne[1];
  const int64_t ne2 = src0->ne[2];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb00 == sizeof(float));

  if (nb10 == sizeof(float)) {
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      ne_vec_sub_f32(ne0, (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1),
                     (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01),
                     (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
      // }
      // }
    }
  } else {
    // src1 is not contiguous
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      float* dst_ptr = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      for (int i0 = 0; i0 < ne0; i0++) {
        float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] - *src1_ptr;
      }
    }
  }
}

static void ne_compute_forward_sub(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_sub_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_mul

static void ne_compute_forward_mul_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_can_repeat_rows(src1, src0) && ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }
  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t nr = ne_nrows(src0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = ne11 == 1 ? 0 : src1->nb[1];
  const size_t nb12 = ne12 == 1 ? 0 : src1->nb[2];
  const size_t nb13 = ne13 == 1 ? 0 : src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb00 == sizeof(float));
  NE_ASSERT(ne00 == ne10);

  if (nb10 == sizeof(float)) {
    for (int64_t ir = ith; ir < nr; ir += nth) {
      // src0 and dst are same shape => same indices
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
      float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

      ne_vec_mul_f32(ne00, dst_ptr, src0_ptr, src1_ptr);
    }
  } else {
    // src1 is not contiguous
    for (int64_t ir = ith; ir < nr; ir += nth) {
      // src0 and dst are same shape => same indices
      // src1 is broadcastable across src0 and dst in i1, i2, i3
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);

      for (int64_t i0 = 0; i0 < ne00; i0++) {
        float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] * (*src1_ptr);
      }
    }
  }
}

static void ne_compute_forward_mul(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_mul_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_div

static void ne_compute_forward_div_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       const struct ne_tensor* src1, struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(ne_are_same_shape(src0, src1) && ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int nr = ne_nrows(src0);
  const int64_t ne0 = src0->ne[0];
  const int64_t ne1 = src0->ne[1];
  const int64_t ne2 = src0->ne[2];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb00 == sizeof(float));

  if (nb10 == sizeof(float)) {
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      ne_vec_div_f32(ne0, (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1),
                     (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01),
                     (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
      // }
      // }
    }
  } else {
    // src1 is not contiguous
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      float* dst_ptr = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      for (int i0 = 0; i0 < ne0; i0++) {
        float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] / (*src1_ptr);
      }
    }
  }
}

static void ne_compute_forward_div(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_div_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_sqr

static void ne_compute_forward_sqr_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ne_vec_sqr_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ne_compute_forward_sqr(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_sqr_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_sqrt

static void ne_compute_forward_sqrt_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ne_vec_sqrt_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ne_compute_forward_sqrt(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_sqrt_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_log

static void ne_compute_forward_log_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       struct ne_tensor* dst) {
  NE_ASSERT(params->ith == 0);
  NE_ASSERT(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  NE_ASSERT(dst->nb[0] == sizeof(float));
  NE_ASSERT(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ne_vec_log_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ne_compute_forward_log(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_log_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_sum

static void ne_compute_forward_sum_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(ne_is_scalar(dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  assert(ne_is_scalar(dst));
  assert(src0->nb[0] == sizeof(float));

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  ne_float sum = 0;
  ne_float row_sum = 0;

  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = 0; i01 < ne01; i01++) {
        ne_vec_sum_ggf(ne00, &row_sum, (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
        sum += row_sum;
      }
    }
  }
  ((float*)dst->data)[0] = sum;
}

static void ne_compute_forward_sum(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_sum_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_sum_rows

static void ne_compute_forward_sum_rows_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                            struct ne_tensor* dst) {
  NE_ASSERT(params->ith == 0);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  NE_ASSERT(src0->nb[0] == sizeof(float));
  NE_ASSERT(dst->nb[0] == sizeof(float));

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  NE_ASSERT(ne0 == 1);
  NE_ASSERT(ne1 == ne01);
  NE_ASSERT(ne2 == ne02);
  NE_ASSERT(ne3 == ne03);

  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  for (int64_t i3 = 0; i3 < ne03; i3++) {
    for (int64_t i2 = 0; i2 < ne02; i2++) {
      for (int64_t i1 = 0; i1 < ne01; i1++) {
        float* src_row = (float*)((char*)src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        float* dst_row = (float*)((char*)dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        float row_sum = 0;
        ne_vec_sum_f32(ne00, &row_sum, src_row);
        dst_row[0] = row_sum;
      }
    }
  }
}

static void ne_compute_forward_sum_rows(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_sum_rows_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_mean

static void ne_compute_forward_mean_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  assert(src0->nb[0] == sizeof(float));

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  assert(ne0 == 1);
  assert(ne1 == ne01);
  assert(ne2 == ne02);
  assert(ne3 == ne03);

  UNUSED(ne0);
  UNUSED(ne1);
  UNUSED(ne2);
  UNUSED(ne3);

  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = 0; i01 < ne01; i01++) {
        ne_vec_sum_f32(ne00, (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                       (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));

        *(float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3) /= (float)ne00;
      }
    }
  }
}

static void ne_compute_forward_mean(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_mean_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_repeat

static void ne_compute_forward_repeat_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                          struct ne_tensor* dst) {
  NE_ASSERT(params->ith == 0);
  NE_ASSERT(ne_can_repeat(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  // guaranteed to be an integer due to the check in ne_can_repeat
  const int nr0 = (int)(ne0 / ne00);
  const int nr1 = (int)(ne1 / ne01);
  const int nr2 = (int)(ne2 / ne02);
  const int nr3 = (int)(ne3 / ne03);

  // TODO: support for transposed / permuted tensors
  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb00 == sizeof(float));

  // TODO: maybe this is not optimal?
  for (int i3 = 0; i3 < nr3; i3++) {
    for (int k3 = 0; k3 < ne03; k3++) {
      for (int i2 = 0; i2 < nr2; i2++) {
        for (int k2 = 0; k2 < ne02; k2++) {
          for (int i1 = 0; i1 < nr1; i1++) {
            for (int k1 = 0; k1 < ne01; k1++) {
              for (int i0 = 0; i0 < nr0; i0++) {
                ne_vec_cpy_f32(ne00,
                               (float*)((char*)dst->data + (i3 * ne03 + k3) * nb3 + (i2 * ne02 + k2) * nb2 +
                                        (i1 * ne01 + k1) * nb1 + (i0 * ne00) * nb0),
                               (float*)((char*)src0->data + (k3)*nb03 + (k2)*nb02 + (k1)*nb01));
              }
            }
          }
        }
      }
    }
  }
}

static void ne_compute_forward_repeat(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                      struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_repeat_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_abs

static void ne_compute_forward_abs_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ne_vec_abs_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ne_compute_forward_abs(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_abs_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_sgn

static void ne_compute_forward_sgn_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ne_vec_sgn_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ne_compute_forward_sgn(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_sgn_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_neg

static void ne_compute_forward_neg_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ne_vec_neg_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ne_compute_forward_neg(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_neg_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_step

static void ne_compute_forward_step_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ne_vec_step_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ne_compute_forward_step(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_step_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_relu

static void ne_compute_forward_relu_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ne_vec_relu_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ne_compute_forward_relu(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_relu_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_gelu

static void ne_compute_forward_gelu_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  NE_ASSERT(ne_is_contiguous(src0));
  NE_ASSERT(ne_is_contiguous(dst));
  NE_ASSERT(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ne_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ne_vec_gelu_f32(nc, (float*)((char*)dst->data + i1 * (dst->nb[1])),
                    (float*)((char*)src0->data + i1 * (src0->nb[1])));

#ifndef NDEBUG
    for (int k = 0; k < nc; k++) {
      const float x = ((float*)((char*)dst->data + i1 * (dst->nb[1])))[k];
      UNUSED(x);
      assert(!isnan(x));
      assert(!isinf(x));
    }
#endif
  }
}

static void ne_compute_forward_gelu(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_gelu_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }

  // printf("XXXXXXXX gelu\n");
}

// ne_compute_forward_silu

static void ne_compute_forward_silu_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  NE_ASSERT(ne_is_contiguous(src0));
  NE_ASSERT(ne_is_contiguous(dst));
  NE_ASSERT(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ne_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ne_vec_silu_f32(nc, (float*)((char*)dst->data + i1 * (dst->nb[1])),
                    (float*)((char*)src0->data + i1 * (src0->nb[1])));

#ifndef NDEBUG
    for (int k = 0; k < nc; k++) {
      const float x = ((float*)((char*)dst->data + i1 * (dst->nb[1])))[k];
      UNUSED(x);
      assert(!isnan(x));
      assert(!isinf(x));
    }
#endif
  }
}

static void ne_compute_forward_silu(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_silu_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_silu_back

static void ne_compute_forward_silu_back_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                             const struct ne_tensor* grad, struct ne_tensor* dst) {
  NE_ASSERT(ne_is_contiguous(grad));
  NE_ASSERT(ne_is_contiguous(src0));
  NE_ASSERT(ne_is_contiguous(dst));
  NE_ASSERT(ne_are_same_shape(src0, dst));
  NE_ASSERT(ne_are_same_shape(src0, grad));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ne_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ne_vec_silu_backward_f32(nc, (float*)((char*)dst->data + i1 * (dst->nb[1])),
                             (float*)((char*)src0->data + i1 * (src0->nb[1])),
                             (float*)((char*)grad->data + i1 * (grad->nb[1])));

#ifndef NDEBUG
    for (int k = 0; k < nc; k++) {
      const float x = ((float*)((char*)dst->data + i1 * (dst->nb[1])))[k];
      UNUSED(x);
      assert(!isnan(x));
      assert(!isinf(x));
    }
#endif
  }
}

static void ne_compute_forward_silu_back(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                         const struct ne_tensor* grad, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_silu_back_f32(params, src0, grad, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_norm

static void ne_compute_forward_norm_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  NE_ASSERT(src0->nb[0] == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const float eps = 1e-5f;  // TODO: make this a parameter

  // TODO: optimize
  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
        const float* x = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

        ne_float sum = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
          sum += (ne_float)x[i00];
        }

        float mean = sum / ne00;

        float* y = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        ne_float sum2 = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
          float v = x[i00] - mean;
          y[i00] = v;
          sum2 += (ne_float)(v * v);
        }

        float variance = sum2 / ne00;
        const float scale = 1.0f / sqrtf(variance + eps);

        ne_vec_scale_f32(ne00, y, scale);
      }
    }
  }
}

static void ne_compute_forward_norm(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_norm_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

static void ne_compute_forward_rms_norm_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                            struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  NE_ASSERT(src0->nb[0] == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const float eps = 1e-6f;  // TODO: make this a parameter

  // TODO: optimize
  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
        const float* x = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

        ne_float sum = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
          sum += (ne_float)(x[i00] * x[i00]);
        }

        float mean = sum / ne00;

        float* y = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        memcpy(y, x, ne00 * sizeof(float));
        // for (int i00 = 0; i00 < ne00; i00++) {
        //     y[i00] = x[i00];
        // }

        const float scale = 1.0f / sqrtf(mean + eps);

        ne_vec_scale_f32(ne00, y, scale);
      }
    }
  }
}

static void ne_compute_forward_rms_norm(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_rms_norm_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

static void ne_compute_forward_rms_norm_back_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                                 const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, dst) && ne_are_same_shape(src0, src1));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  NE_ASSERT(src0->nb[0] == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const float eps = 1e-6f;  // TODO: make this a parameter

  // TODO: optimize
  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
        // src1 is same shape as src0 => same indices
        const int64_t i11 = i01;
        const int64_t i12 = i02;
        const int64_t i13 = i03;

        const float* x = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        const float* dz = (float*)((char*)src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);

        ne_float sum_xx = 0.0;
        ne_float sum_xdz = 0.0;

        for (int64_t i00 = 0; i00 < ne00; i00++) {
          sum_xx += (ne_float)(x[i00] * x[i00]);
          sum_xdz += (ne_float)(x[i00] * dz[i00]);
        }

        // const float mean     = (float)(sum_xx)/ne00;
        const float mean_eps = (float)(sum_xx) / ne00 + eps;
        const float sum_eps = (float)(sum_xx) + eps * ne00;
        // const float mean_xdz = (float)(sum_xdz)/ne00;
        //  we could cache rms from forward pass to improve performance.
        //  to do this implement ne_rms and compose ne_rms_norm using ne_rms.
        // const float rms      = sqrtf(mean_eps);
        const float rrms = 1.0f / sqrtf(mean_eps);
        // const float scale    = -rrms/(ne00 * mean_eps); // -1/(n*rms**3)

        {
          // z = rms_norm(x)
          //
          // rms_norm(src0) =
          //     scale(
          //         src0,
          //         div(
          //             1,
          //             sqrt(
          //                 add(
          //                     scale(
          //                         sum(
          //                             sqr(
          //                                 src0)),
          //                         (1.0/N)),
          //                     eps))));

          // postorder:
          // ## op    args         grad
          // 00 param src0         grad[#00]
          // 01 const 1
          // 02 sqr   (#00)        grad[#02]
          // 03 sum   (#02)        grad[#03]
          // 04 const 1/N
          // 05 scale (#03, #04)   grad[#05]
          // 06 const eps
          // 07 add   (#05, #06)   grad[#07]
          // 08 sqrt  (#07)        grad[#08]
          // 09 div   (#01,#08)    grad[#09]
          // 10 scale (#00,#09)    grad[#10]
          //
          // backward pass, given grad[#10]
          // #10: scale
          // grad[#00] += scale(grad[#10],#09)
          // grad[#09] += sum(mul(grad[#10],#00))
          // #09: div
          // grad[#08] += neg(mul(grad[#09], div(#09,#08)))
          // #08: sqrt
          // grad[#07] += mul(grad[#08], div(0.5, #08))
          // #07: add
          // grad[#05] += grad[#07]
          // #05: scale
          // grad[#03] += scale(grad[#05],#04)
          // #03: sum
          // grad[#02] += repeat(grad[#03], #02)
          // #02:
          // grad[#00] += scale(mul(#00, grad[#02]), 2.0)
          //
          // substitute and simplify:
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
          // grad[#02] = repeat(grad[#03], #02)
          // grad[#02] = repeat(scale(grad[#05],#04), #02)
          // grad[#02] = repeat(scale(grad[#07],#04), #02)
          // grad[#02] = repeat(scale(mul(grad[#08], div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(scale(mul(neg(mul(grad[#09], div(#09,#08))), div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(scale(mul(neg(mul(sum(mul(grad[#10],#00)), div(#09,#08))), div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(#09,#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(div(#01,#08),#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#08*#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5,
          // #08) * (1/N)), #02)), 2.0) grad[#00] = scale(grad(#10), #09) + scale(scale(#00, -(sum(mul(grad[#10],#00)) *
          // div(1,#07) * div(0.5, #08) * (1/N))), 2.0) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // -(sum(mul(grad[#10],#00)) * div(1,#07) * div(1,#08) * (1/N))) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,mean_eps*rms) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*mean_eps)) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // sum(mul(grad[#10],#00)) * div(-1,rms*N*(sum_xx/N+eps))) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // sum(mul(grad[#10],#00)) * div(-1,rms*N*sum_xx+rms*N*eps)) grad[#00] = scale(dz, rrms) + scale(x,
          // sum(mul(dz,x)) * div(-1,rms*N*mean_eps)) grad[#00] = scale(dz, rrms) + scale(x, sum_xdz *
          // div(-1,rms*N*mean_eps)) a = b*c + d*e a = b*c*f/f + d*e*f/f a = (b*c*f + d*e*f)*(1/f) a = (b*c*(1/c) +
          // d*e*(1/c))*(1/(1/c)) a = (b + d*e/c)*c b = dz, c = rrms, d = x, e = sum_xdz * div(-1,rms*N*mean_eps) a =
          // (dz + x*sum_xdz * div(-1,rms*N*mean_eps)/rrms)*rrms a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)*rms)*rrms
          // a = (dz + x*sum_xdz * div(-rms,rms*N*mean_eps))*rrms
          // a = (dz + x*sum_xdz * div(-1,N*mean_eps))*rrms
          // a = (dz + x*div(-sum_xdz,N*mean_eps))*rrms
          // a = (dz + x*div(-mean_xdz,mean_eps))*rrms
          // grad[#00] = scale(dz + scale(x, div(-mean_xdz,mean_eps)),rrms)
          // grad[#00] = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
          // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
        }
        // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
        // post-order:
        // dx := x
        // dx := scale(dx,-mean_xdz/mean_eps)
        // dx := add(dx, dz)
        // dx := scale(dx, rrms)
        float* dx = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        ne_vec_cpy_f32(ne00, dx, x);
        // ne_vec_scale_f32(ne00, dx, -mean_xdz/mean_eps);
        ne_vec_scale_f32(ne00, dx, (float)(-sum_xdz) / sum_eps);
        ne_vec_acc_f32(ne00, dx, dz);
        ne_vec_scale_f32(ne00, dx, rrms);
      }
    }
  }
}

static void ne_compute_forward_rms_norm_back(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                             const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_rms_norm_back_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

static void ne_compute_forward_mul_mat_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                           const struct ne_tensor* src1, struct ne_tensor* dst) {
  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne11 = src1->ne[1];

  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const size_t nb00 = src0->nb[0];

  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const int nb10 = src1->nb[0];

  const int nb11 = src1->nb[1];
  UNUSED(nb11);
  const int nb12 = src1->nb[2];
  UNUSED(nb12);
  const int nb13 = src1->nb[3];
  UNUSED(nb13);

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  NE_ASSERT(ne0 == ne01);
  NE_ASSERT(ne1 == ne11);
  NE_ASSERT(ne2 == ne12);
  NE_ASSERT(ne3 == ne13);

  // we don't support permuted src0 or src1
  NE_ASSERT(nb00 == sizeof(float));
  NE_ASSERT(nb10 == sizeof(float));

  // dst cannot be transposed or permuted
  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb0 <= nb1);
  NE_ASSERT(nb1 <= nb2);
  NE_ASSERT(nb2 <= nb3);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // parallelize by src0 rows
  const int64_t dr = (ne01 + nth - 1) / nth;

  const int64_t ir10 = dr * ith;
  const int64_t ir11 = MIN(ir10 + dr, ne01);

  // src1 rows
  const int64_t nr1 = ne11 * ne12 * ne13;

  for (int64_t ir1 = 0; ir1 < nr1; ++ir1) {
    const int64_t i13 = (ir1 / (ne12 * ne11));
    const int64_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
    const int64_t i11 = (ir1 - i13 * ne12 * ne11 - i12 * ne11);

    const int64_t ir0 = (ir1 / ne11) % (ne02 * ne03);
    const int64_t i03 = (ir0 / (ne02));
    // Hack for "Falcon multi-query-attention key stutter" / alternative to ggml_repeat2.
    // See https://github.com/ggerganov/llama.cpp/issues/1602#issuecomment-1606087470:
    const int64_t i02 = (i12 / (ne12 / ne02));
    // Original from PR/224 (and also essential/correct for non-broadcast matmuls in Falcon)
    // const int64_t i02 = (ir0 - i03*ne02);

    const int64_t i1 = i11;
    const int64_t i2 = i12;
    const int64_t i3 = i13;

    char* src0_row = (char*)src0->data + (0 + i02 * nb02 + i03 * nb03);
    char* src1_col = (char*)src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13);

    float* dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

    for (int64_t ir = ir10; ir < ir11; ++ir) {
      ne_vec_dot_f32(ne00, &dst_col[ir], (float*)(src0_row + ir * nb01), (float*)src1_col);
    }
  }
}

static void ne_compute_forward_mul_mat_f16_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                               const struct ne_tensor* src1, struct ne_tensor* dst) {
  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);
  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];
  // const int64_t ne   = ne0*ne1*ne2*ne3;

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  NE_ASSERT(ne0 == ne01);
  NE_ASSERT(ne1 == ne11);
  NE_ASSERT(ne2 == ne12);
  NE_ASSERT(ne3 == ne13);

  // TODO: we don't support permuted src0
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));

  // dst cannot be transposed or permuted
  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb0 <= nb1);
  NE_ASSERT(nb1 <= nb2);
  NE_ASSERT(nb2 <= nb3);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  if (params->type == NE_TASK_INIT) {
    ne_fp16_t* const wdata = params->wdata;

    size_t id = 0;
    for (int64_t i13 = 0; i13 < ne13; ++i13) {
      for (int64_t i12 = 0; i12 < ne12; ++i12) {
        for (int64_t i11 = 0; i11 < ne11; ++i11) {
          for (int64_t i10 = 0; i10 < ne10; ++i10) {
            wdata[id++] =
                NE_FP32_TO_FP16(*(float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i10 * nb10));
          }
        }
      }
    }

    NE_ASSERT(id * sizeof(ne_fp16_t) <= params->wsize);

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // fp16 -> half the size, so divide by 2
  // TODO: do not support transposed src1
  assert(nb10 / 2 == sizeof(ne_fp16_t));

  // parallelize by src0 rows
  const int64_t dr = (ne01 + nth - 1) / nth;

  const int64_t ir10 = dr * ith;
  const int64_t ir11 = MIN(ir10 + dr, ne01);

  // src1 rows
  const int64_t nr1 = ne11 * ne12 * ne13;

  void* wdata = params->wdata;
  const size_t row_size = ne10 * NE_TYPE_SIZE[NE_TYPE_F16];

  for (int64_t ir1 = 0; ir1 < nr1; ++ir1) {
    const int64_t i13 = (ir1 / (ne12 * ne11));
    const int64_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
    const int64_t i11 = (ir1 - i13 * ne12 * ne11 - i12 * ne11);

    const int64_t ir0 = (ir1 / ne11) % (ne02 * ne03);
    const int64_t i03 = (ir0 / (ne02));
    // Hack for "Falcon multi-query-attention key stutter" / alternative to ggml_repeat2.
    // See https://github.com/ggerganov/llama.cpp/issues/1602#issuecomment-1606087470:
    const int64_t i02 = (i12 / (ne12 / ne02));
    // Original from PR/224 (and also essential/correct for non-broadcast matmuls in Falcon)
    // const int64_t i02 = (ir0 - i03*ne02);

    const int64_t i1 = i11;
    const int64_t i2 = i12;
    const int64_t i3 = i13;

    char* src0_row = (char*)src0->data + (0 + i02 * nb02 + i03 * nb03);
    char* src1_col = (char*)wdata + (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size;

    float* dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

    for (int64_t ir = ir10; ir < ir11; ++ir) {
      ne_vec_dot_f16(ne00, &dst_col[ir], (ne_fp16_t*)(src0_row + ir * nb01), (ne_fp16_t*)src1_col);
    }
  }

  // int64_t t1 = ne_time_us();
  // static int64_t acc = 0;
  // acc += t1 - t0;
  // if (t1 - t0 > 10) {
  //     printf("\n");
  //     printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
  //     printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
  //     printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

  //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1
  //    - t0), (int) acc);
  //}
}

static void ne_compute_forward_mul_mat_q_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                             const struct ne_tensor* src1, struct ne_tensor* dst) {
  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const enum ne_type type = src0->type;
  quantize_row_q_t const quantize_row_q_dot = quantize_fns[type].quantize_row_q_dot;
  vec_dot_q_t const vec_dot_q = quantize_fns[type].vec_dot_q;
  enum ne_type const vec_dot_type = quantize_fns[type].vec_dot_type;

  NE_ASSERT(ne0 == ne01);
  NE_ASSERT(ne1 == ne11);
  NE_ASSERT(ne2 == ne12);
  NE_ASSERT(ne3 == ne13);

  // we don't support permuted src0 or src1
  NE_ASSERT(nb00 == (int)NE_TYPE_SIZE[type]);
  NE_ASSERT(nb10 == sizeof(float));

  // dst cannot be transposed or permuted
  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb0 <= nb1);
  NE_ASSERT(nb1 <= nb2);
  NE_ASSERT(nb2 <= nb3);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  if (params->type == NE_TASK_INIT) {
    char* wdata = params->wdata;
    const size_t row_size = ne10 * NE_TYPE_SIZE[vec_dot_type] / NE_BLCK_SIZE[vec_dot_type];

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
      for (int64_t i12 = 0; i12 < ne12; ++i12) {
        for (int64_t i11 = 0; i11 < ne11; ++i11) {
          quantize_row_q_dot((float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11), (void*)wdata, ne10);
          wdata += row_size;
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // parallelize by src0 rows
  const int64_t dr = (ne01 + nth - 1) / nth;

  const int64_t ir10 = dr * ith;
  const int64_t ir11 = MIN(ir10 + dr, ne01);

  // src1 rows
  const int64_t nr1 = ne11 * ne12 * ne13;

  const void* wdata = params->wdata;
  const size_t row_size = ne10 * NE_TYPE_SIZE[vec_dot_type] / NE_BLCK_SIZE[vec_dot_type];

  for (int64_t ir1 = 0; ir1 < nr1; ++ir1) {
    const int64_t i13 = (ir1 / (ne12 * ne11));
    const int64_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
    const int64_t i11 = (ir1 - i13 * ne12 * ne11 - i12 * ne11);

    const int64_t ir0 = (ir1 / ne11) % (ne02 * ne03);
    const int64_t i03 = (ir0 / (ne02));
    // Hack for "Falcon multi-query-attention key stutter" / alternative to ggml_repeat2.
    // See https://github.com/ggerganov/llama.cpp/issues/1602#issuecomment-1606087470:
    const int64_t i02 = (i12 / (ne12 / ne02));
    // Original from PR/224 (and also essential/correct for non-broadcast matmuls in Falcon)
    // const int64_t i02 = (ir0 - i03*ne02);

    const int64_t i1 = i11;
    const int64_t i2 = i12;
    const int64_t i3 = i13;

    const char* src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);
    const char* src1_col = (const char*)wdata + (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size;

    float* dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

    for (int64_t ir = ir10; ir < ir11; ++ir) {
      vec_dot_q(ne00, &dst_col[ir], src0_row + ir * nb01, src1_col);
    }
  }

  // int64_t t1 = ne_time_us();
  // static int64_t acc = 0;
  // acc += t1 - t0;
  // if (t1 - t0 > 10) {
  //     printf("\n");
  //     printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
  //     printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
  //     printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

  //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1
  //    - t0), (int) acc);
  //}
}

static void ne_compute_forward_mul_mat_q_f32_jblas(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                                   const struct ne_tensor* src1, struct ne_tensor* dst) {
  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  NE_ASSERT(ne02 == ne12);
  NE_ASSERT(ne03 == ne13);
  NE_ASSERT(ne2 == ne12);
  NE_ASSERT(ne3 == ne13);

  const enum ne_type type = src0->type;
  quantize_row_q_t const quantize_row_q_dot = quantize_fns[type].quantize_row_q_dot;
  vec_dot_q_t const vec_dot_q = quantize_fns[type].vec_dot_q;
  enum ne_type const vec_dot_type = quantize_fns[type].vec_dot_type;

  // we don't support permuted src0 or src1
  NE_ASSERT(nb00 == (int)NE_TYPE_SIZE[type]);
  NE_ASSERT(nb10 == sizeof(float));

  // dst cannot be transposed or permuted
  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb0 <= nb1);
  NE_ASSERT(nb1 <= nb2);
  NE_ASSERT(nb2 <= nb3);

  NE_ASSERT(ne0 == ne01);
  NE_ASSERT(ne1 == ne11);
  NE_ASSERT(ne2 == ne02);
  NE_ASSERT(ne3 == ne03);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }
  jblas_f32f32_forward((float*)src1->data, src0->data, (float*)dst->data, ne1, ne0, ne10, ne10, ne0, params->wdata);
}

static void ne_compute_forward_mul_mat(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_Q4_0:
    case NE_TYPE_Q4_1:
    case NE_TYPE_Q5_0:
    case NE_TYPE_Q5_1:
    case NE_TYPE_Q8_0:
    case NE_TYPE_Q8_1: {
      ne_compute_forward_mul_mat_q_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_JBLAS: {
      ne_compute_forward_mul_mat_q_f32_jblas(params, src0, src1, dst);
    } break;
    case NE_TYPE_F16: {
      ne_compute_forward_mul_mat_f16_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_mul_mat_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

static void ne_compute_forward_mul_mat_bias_q_f32_jblas(const struct ne_compute_params* params,
                                                        const struct ne_tensor* src0, const struct ne_tensor* src1,
                                                        const struct ne_tensor* bias, struct ne_tensor* dst) {
  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const int nb00 = src0->nb[0];
  const int nb01 = src0->nb[1];
  const int nb02 = src0->nb[2];
  const int nb03 = src0->nb[3];

  const int nb10 = src1->nb[0];
  const int nb11 = src1->nb[1];
  const int nb12 = src1->nb[2];
  const int nb13 = src1->nb[3];

  const int nb0 = dst->nb[0];
  const int nb1 = dst->nb[1];
  const int nb2 = dst->nb[2];
  const int nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  NE_ASSERT(ne02 == ne12);
  NE_ASSERT(ne03 == ne13);
  NE_ASSERT(ne2 == ne12);
  NE_ASSERT(ne3 == ne13);

  const enum ne_type type = src0->type;
  quantize_row_q_t const quantize_row_q_dot = quantize_fns[type].quantize_row_q_dot;
  vec_dot_q_t const vec_dot_q = quantize_fns[type].vec_dot_q;
  enum ne_type const vec_dot_type = quantize_fns[type].vec_dot_type;

  // we don't support permuted src0 or src1
  NE_ASSERT(nb00 == (int)NE_TYPE_SIZE[type]);
  NE_ASSERT(nb10 == sizeof(float));

  // dst cannot be transposed or permuted
  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb0 <= nb1);
  NE_ASSERT(nb1 <= nb2);
  NE_ASSERT(nb2 <= nb3);

  NE_ASSERT(ne0 == ne01);
  NE_ASSERT(ne1 == ne11);
  NE_ASSERT(ne2 == ne02);
  NE_ASSERT(ne3 == ne03);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }
  const bool boardcast_bias = bias->ne[1] == 1;
  jblas_fusion_add_f32f32_forward((float*)src1->data, src0->data, (float*)bias->data, (float*)dst->data, ne1, ne0, ne10,
                                  ne10, ne0, boardcast_bias, params->wdata);
}

static void ne_compute_forward_mul_mat_bias(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                            const struct ne_tensor* src1, const struct ne_tensor* bias,
                                            struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_JBLAS: {
      ne_compute_forward_mul_mat_bias_q_f32_jblas(params, src0, src1, bias, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

static void ne_compute_forward_mul_qkv(const struct ne_compute_params* params, const struct ne_tensor* src,
                                       const struct ne_tensor* qw, const struct ne_tensor* kw, struct ne_tensor* vw,
                                       struct ne_tensor* dst) {
  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }
  const int n = dst->ne[0];
  const int m = dst->ne[1];
  const int k = src->ne[0];
  jblas_fusion_QKV_f32f32_forward((float*)src->data, qw->data, kw->data, vw->data, (float*)dst->data, m, n, k, k, n,
                                  params->wdata);
}

static void ne_compute_forward_ffn_silu(const struct ne_compute_params* params, const struct ne_tensor* src,
                                        const struct ne_tensor* w1, const struct ne_tensor* w2, struct ne_tensor* w3,
                                        const struct ne_tensor* tmp, struct ne_tensor* tmp1, struct ne_tensor* dst) {
  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }
  const int fin = src->ne[0];
  const int fout = dst->ne[0];
  const int fmid = w1->ne[1];
  const int seq = dst->ne[1];
  jblas_fusion_FFN_SiLu_f32f32_forward((float*)src->data, w1->data, w2->data, w3->data, (float*)tmp->data,
                                       (float*)tmp1->data, (float*)dst->data, seq, fin, fmid, fout, params->wdata);
}

static void ne_compute_forward_ffn_add_gelu(const struct ne_compute_params* params, const struct ne_tensor* src,
                                            const struct ne_tensor* w1, const struct ne_tensor* w2,
                                            const struct ne_tensor* b1, const struct ne_tensor* b2,
                                            const struct ne_tensor* tmp, struct ne_tensor* dst) {
  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }
  const int fin = src->ne[0];
  const int fout = dst->ne[0];
  const int fmid = w1->ne[1];
  const int seq = dst->ne[1];
  const bool boardcast_bias = b1->ne[1] == 1 || b2->ne[1] == 1;
  jblas_fusion_FFN_Add_GeLu_f32f32_forward((float*)src->data, w1->data, w2->data, (float*)b1->data, (float*)b2->data,
                                           (float*)tmp->data, (float*)dst->data, seq, fin, fmid, fout, boardcast_bias,
                                           params->wdata);
}

static void ne_compute_forward_ffn_gelu(const struct ne_compute_params* params, const struct ne_tensor* src,
                                        const struct ne_tensor* w1, const struct ne_tensor* w2,
                                        const struct ne_tensor* tmp, struct ne_tensor* dst) {
  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }
  const int fin = src->ne[0];
  const int fout = dst->ne[0];
  const int fmid = w1->ne[1];
  const int seq = dst->ne[1];
  jblas_fusion_FFN_GeLu_f32f32_forward((float*)src->data, w1->data, w2->data, (float*)tmp->data, (float*)dst->data, seq,
                                       fin, fmid, fout, params->wdata);
}

// ne_compute_forward_scale

static void ne_compute_forward_scale_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                         const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(ne_is_contiguous(src0));
  NE_ASSERT(ne_is_contiguous(dst));
  NE_ASSERT(ne_are_same_shape(src0, dst));
  NE_ASSERT(ne_is_scalar(src1));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  // scale factor
  const float v = *(float*)src1->data;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ne_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const size_t nb01 = src0->nb[1];

  const size_t nb1 = dst->nb[1];

  for (int i1 = ir0; i1 < ir1; i1++) {
    if (dst->data != src0->data) {
      // src0 is same shape as dst => same indices
      memcpy((char*)dst->data + i1 * nb1, (char*)src0->data + i1 * nb01, nc * sizeof(float));
    }
    ne_vec_scale_f32(nc, (float*)((char*)dst->data + i1 * nb1), v);
  }
}

static void ne_compute_forward_scale(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                     const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_scale_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_set

static void ne_compute_forward_set_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       const struct ne_tensor* src1, const struct ne_tensor* opt0,
                                       struct ne_tensor* dst) {
  NE_ASSERT(ne_are_same_shape(src0, dst));
  NE_ASSERT(ne_is_contiguous(dst) && ne_is_contiguous(src0));

  NE_ASSERT(opt0->type == NE_TYPE_I32);
  NE_ASSERT(ne_nelements(opt0) == 5);

  // view src0 and dst with these strides and data offset inbytes during set
  // nb0 is implicitely element_size because src0 and dst are contiguous
  size_t nb1 = ((int32_t*)opt0->data)[0];
  size_t nb2 = ((int32_t*)opt0->data)[1];
  size_t nb3 = ((int32_t*)opt0->data)[2];
  size_t offset = ((int32_t*)opt0->data)[3];
  bool inplace = (bool)((int32_t*)opt0->data)[4];

  if (!inplace && (params->type == NE_TASK_INIT)) {
    // memcpy needs to be synchronized across threads to avoid race conditions.
    // => do it in INIT phase
    memcpy(((char*)dst->data), ((char*)src0->data), ne_nbytes(dst));
  }

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(src1);
  const int nc = src1->ne[0];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  const size_t nb12 = src1->nb[2];
  const size_t nb13 = src1->nb[3];

  // src0 and dst as viewed during set
  const size_t nb0 = ne_element_size(src0);

  const int im0 = (ne10 == 0 ? 0 : ne10 - 1);
  const int im1 = (ne11 == 0 ? 0 : ne11 - 1);
  const int im2 = (ne12 == 0 ? 0 : ne12 - 1);
  const int im3 = (ne13 == 0 ? 0 : ne13 - 1);

  NE_ASSERT(offset + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 < ne_nbytes(dst));

  NE_ASSERT(nb10 == sizeof(float));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are viewed with shape of src1 and offset
    // => same indices
    const int i3 = ir / (ne12 * ne11);
    const int i2 = (ir - i3 * ne12 * ne11) / ne11;
    const int i1 = (ir - i3 * ne12 * ne11 - i2 * ne11);

    ne_vec_cpy_f32(nc, (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + offset),
                   (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
  }
}

static void ne_compute_forward_set(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, const struct ne_tensor* opt0, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_set_f32(params, src0, src1, opt0, dst);
    } break;
    case NE_TYPE_F16:
    case NE_TYPE_Q4_0:
    case NE_TYPE_Q4_1:
    case NE_TYPE_Q5_0:
    case NE_TYPE_Q5_1:
    case NE_TYPE_Q8_0:
    case NE_TYPE_Q8_1:
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_cpy

static void ne_compute_forward_cpy(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   struct ne_tensor* dst) {
  ne_compute_forward_dup(params, src0, dst);
}

// ne_compute_forward_cont

static void ne_compute_forward_cont(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    struct ne_tensor* dst) {
  ne_compute_forward_dup(params, src0, dst);
}

// ne_compute_forward_reshape

static void ne_compute_forward_reshape(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                       struct ne_tensor* dst) {
  // NOP
  UNUSED(params);
  UNUSED(src0);
  UNUSED(dst);
}

// ne_compute_forward_view

static void ne_compute_forward_view(const struct ne_compute_params* params, const struct ne_tensor* src0) {
  // NOP
  UNUSED(params);
  UNUSED(src0);
}

// ne_compute_forward_permute

static void ne_compute_forward_permute(const struct ne_compute_params* params, const struct ne_tensor* src0) {
  // NOP
  UNUSED(params);
  UNUSED(src0);
}

// ne_compute_forward_transpose

static void ne_compute_forward_transpose(const struct ne_compute_params* params, const struct ne_tensor* src0) {
  // NOP
  UNUSED(params);
  UNUSED(src0);
}

// ne_compute_forward_get_rows

static void ne_compute_forward_get_rows_q(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                          const struct ne_tensor* src1, struct ne_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ne_nelements(src1);
  const enum ne_type type = src0->type;
  dequantize_row_q_t const dequantize_row_q = quantize_fns[type].dequantize_row_q;

  assert(dst->ne[0] == nc);
  assert(dst->ne[1] == nr);
  assert(src0->nb[0] == NE_TYPE_SIZE[type]);

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t*)src1->data)[i];

    dequantize_row_q((const void*)((char*)src0->data + r * src0->nb[1]), (float*)((char*)dst->data + i * dst->nb[1]),
                     nc);
  }
}

static void ne_compute_forward_get_rows_f16(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                            const struct ne_tensor* src1, struct ne_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ne_nelements(src1);

  assert(dst->ne[0] == nc);
  assert(dst->ne[1] == nr);
  assert(src0->nb[0] == sizeof(ne_fp16_t));

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t*)src1->data)[i];

    for (int j = 0; j < nc; ++j) {
      ne_fp16_t v = ((ne_fp16_t*)((char*)src0->data + r * src0->nb[1]))[j];
      ((float*)((char*)dst->data + i * dst->nb[1]))[j] = NE_FP16_TO_FP32(v);
    }
  }
}

static void ne_compute_forward_get_rows_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                            const struct ne_tensor* src1, struct ne_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ne_nelements(src1);

  assert(dst->ne[0] == nc);
  assert(dst->ne[1] == nr);
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t*)src1->data)[i];

    ne_vec_cpy_f32(nc, (float*)((char*)dst->data + i * dst->nb[1]), (float*)((char*)src0->data + r * src0->nb[1]));
  }
}

static void ne_compute_forward_get_rows(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_Q4_0:
    case NE_TYPE_Q4_1:
    case NE_TYPE_Q5_0:
    case NE_TYPE_Q5_1:
    case NE_TYPE_Q8_0:
    case NE_TYPE_Q8_1: {
      ne_compute_forward_get_rows_q(params, src0, src1, dst);
    } break;
    case NE_TYPE_F16: {
      ne_compute_forward_get_rows_f16(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_get_rows_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }

  // static bool first = true;
  // printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
  // if (first) {
  //     first = false;
  // } else {
  //     for (int k = 0; k < dst->ne[1]; ++k) {
  //         for (int j = 0; j < dst->ne[0]/16; ++j) {
  //             for (int i = 0; i < 16; ++i) {
  //                 printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
  //             }
  //             printf("\n");
  //         }
  //         printf("\n");
  //     }
  //     printf("\n");
  //     exit(0);
  // }
}

// ne_compute_forward_get_rows_back

static void ne_compute_forward_get_rows_back_f32_f16(const struct ne_compute_params* params,
                                                     const struct ne_tensor* src0, const struct ne_tensor* src1,
                                                     const struct ne_tensor* opt0, struct ne_tensor* dst) {
  NE_ASSERT(params->ith == 0);
  NE_ASSERT(ne_are_same_shape(opt0, dst));
  NE_ASSERT(ne_is_contiguous(opt0));
  NE_ASSERT(ne_is_contiguous(dst));

  ne_compute_forward_dup_same_cont(params, opt0, dst);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ne_nelements(src1);

  NE_ASSERT(dst->ne[0] == nc);
  NE_ASSERT(src0->nb[0] == sizeof(ne_fp16_t));

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t*)src1->data)[i];

    for (int j = 0; j < nc; ++j) {
      ne_fp16_t v = ((ne_fp16_t*)((char*)src0->data + i * src0->nb[1]))[j];
      ((float*)((char*)dst->data + r * dst->nb[1]))[j] += NE_FP16_TO_FP32(v);
    }
  }
}

static void ne_compute_forward_get_rows_back_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                                 const struct ne_tensor* src1, const struct ne_tensor* opt0,
                                                 struct ne_tensor* dst) {
  NE_ASSERT(params->ith == 0);
  NE_ASSERT(ne_are_same_shape(opt0, dst));
  NE_ASSERT(ne_is_contiguous(opt0));
  NE_ASSERT(ne_is_contiguous(dst));

  ne_compute_forward_dup_same_cont(params, opt0, dst);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ne_nelements(src1);

  NE_ASSERT(dst->ne[0] == nc);
  NE_ASSERT(src0->nb[0] == sizeof(float));

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t*)src1->data)[i];

    ne_vec_add_f32(nc, (float*)((char*)dst->data + r * dst->nb[1]), (float*)((char*)dst->data + r * dst->nb[1]),
                   (float*)((char*)src0->data + i * src0->nb[1]));
  }
}

static void ne_compute_forward_get_rows_back(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                             const struct ne_tensor* src1, const struct ne_tensor* opt0,
                                             struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_get_rows_back_f32_f16(params, src0, src1, opt0, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_get_rows_back_f32(params, src0, src1, opt0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }

  // static bool first = true;
  // printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
  // if (first) {
  //     first = false;
  // } else {
  //     for (int k = 0; k < dst->ne[1]; ++k) {
  //         for (int j = 0; j < dst->ne[0]/16; ++j) {
  //             for (int i = 0; i < 16; ++i) {
  //                 printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
  //             }
  //             printf("\n");
  //         }
  //         printf("\n");
  //     }
  //     printf("\n");
  //     exit(0);
  // }
}

// ne_compute_forward_diag

static void ne_compute_forward_diag_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  NE_ASSERT(params->ith == 0);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  const int ne00 = src0->ne[0];
  const int ne01 = src0->ne[1];
  const int ne02 = src0->ne[2];
  const int ne03 = src0->ne[3];
  const int ne0 = dst->ne[0];
  const int ne1 = dst->ne[1];
  const int ne2 = dst->ne[2];
  const int ne3 = dst->ne[3];
  NE_ASSERT(ne00 == ne0);
  NE_ASSERT(ne00 == ne1);
  NE_ASSERT(ne01 == 1);
  NE_ASSERT(ne02 == ne2);
  NE_ASSERT(ne03 == ne3);

  const size_t nb00 = src0->nb[0];
  // const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];
  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  NE_ASSERT(nb00 == sizeof(float));
  NE_ASSERT(nb0 == sizeof(float));

  for (int i3 = 0; i3 < ne3; i3++) {
    for (int i2 = 0; i2 < ne2; i2++) {
      for (int i1 = 0; i1 < ne1; i1++) {
        float* d = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
        float* s = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02);
        for (int i0 = 0; i0 < i1; i0++) {
          d[i0] = 0;
        }
        d[i1] = s[i1];
        for (int i0 = i1 + 1; i0 < ne0; i0++) {
          d[i0] = 0;
        }
      }
    }
  }
}

static void ne_compute_forward_diag(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_diag_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_diag_mask_inf

static void ne_compute_forward_diag_mask_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                             const struct ne_tensor* src1, struct ne_tensor* dst, const float value) {
  assert(src1->type == NE_TYPE_I32);
  assert(ne_nelements(src1) == 2);

  const int ith = params->ith;
  const int nth = params->nth;

  const int n_past = ((int32_t*)src1->data)[0];
  const bool inplace = (bool)((int32_t*)src1->data)[1];

  assert(n_past >= 0);

  if (!inplace && (params->type == NE_TASK_INIT)) {
    // memcpy needs to be synchronized across threads to avoid race conditions.
    // => do it in INIT phase
    NE_ASSERT(ne_nelements(dst) == ne_nelements(src0));
    NE_ASSERT(ne_is_contiguous(dst) && ne_is_contiguous(src0));
    memcpy(((char*)dst->data), ((char*)src0->data), ne_nbytes(dst));
  }

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];
  const int nr = src0->ne[1];
  const int nz = n / nr;

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int k = 0; k < nz; k++) {
    for (int j = ith; j < nr; j += nth) {
      for (int i = n_past; i < nc; i++) {
        if (i > n_past + j) {
          *(float*)((char*)dst->data + k * dst->nb[2] + j * dst->nb[1] + i * dst->nb[0]) = value;
        }
      }
    }
  }
}

static void ne_compute_forward_diag_mask_inf(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                             const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_diag_mask_f32(params, src0, src1, dst, -INFINITY);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

static void ne_compute_forward_diag_mask_zero(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                              const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_diag_mask_f32(params, src0, src1, dst, 0);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_soft_max

static void ne_compute_forward_soft_max_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                            struct ne_tensor* dst) {
  NE_ASSERT(ne_is_contiguous(src0));
  NE_ASSERT(ne_is_contiguous(dst));
  NE_ASSERT(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ne_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* sp = (float*)((char*)src0->data + i1 * src0->nb[1]);
    float* dp = (float*)((char*)dst->data + i1 * dst->nb[1]);

#ifndef NDEBUG
    for (int i = 0; i < nc; ++i) {
      // printf("p[%d] = %f\n", i, p[i]);
      assert(!isnan(sp[i]));
    }
#endif

    float max = -INFINITY;
    ne_vec_max_f32(nc, &max, sp);

    ne_float sum = 0.0;

    uint16_t scvt;
    for (int i = 0; i < nc; i++) {
      if (sp[i] == -INFINITY) {
        dp[i] = 0.0f;
      } else {
        // const float val = (sp[i] == -INFINITY) ? 0.0 : exp(sp[i] - max);
        ne_fp16_t s = NE_FP32_TO_FP16(sp[i] - max);
        memcpy(&scvt, &s, sizeof(scvt));
        const float val = NE_FP16_TO_FP32(table_exp_f16[scvt]);
        sum += (ne_float)val;
        dp[i] = val;
      }
    }

    assert(sum > 0.0);

    sum = 1.0 / sum;
    ne_vec_scale_f32(nc, dp, sum);

#ifndef NDEBUG
    for (int i = 0; i < nc; ++i) {
      assert(!isnan(dp[i]));
      assert(!isinf(dp[i]));
    }
#endif
  }
}

static void ne_compute_forward_soft_max(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_soft_max_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_alibi

static void ne_compute_forward_alibi_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                         const struct ne_tensor* src1, struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(src1->type == NE_TYPE_I32);
  assert(ne_nelements(src1) == 3);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n_past = ((int32_t*)src1->data)[0];
  const int n_head = ((int32_t*)src1->data)[1];
  const float max_bias = ((float*)src1->data)[2];

  assert(n_past >= 0);

  const int ne0 = src0->ne[0];  // all_seq_len = n_past + ne1
  const int ne1 = src0->ne[1];  // seq_len_without_past
  // const int ne2 = src0->ne[2]; // n_head -> this is k
  // const int ne3 = src0->ne[3]; // 1 -> bsz

  const int n = ne_nrows(src0);
  const int ne2_ne3 = n / ne1;  // ne2*ne3

  const size_t nb0 = src0->nb[0];
  const size_t nb1 = src0->nb[1];
  const size_t nb2 = src0->nb[2];
  // const size_t nb3 = src0->nb[3];

  assert(nb0 == sizeof(float));
  assert(ne1 + n_past == ne0);
  (void)n_past;

  // add alibi to src0 (KQ_scaled)
  const int n_heads_log2_floor = 1 << (int)floor(log2(n_head));

  const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
  const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

  for (int i = 0; i < ne0; i++) {
    for (int j = 0; j < ne1; j++) {
      for (int k = 0; k < ne2_ne3; k++) {
        float* const src = (float*)((char*)src0->data + i * nb0 + j * nb1 + k * nb2);
        float* pdst = (float*)((char*)dst->data + i * nb0 + j * nb1 + k * nb2);

        // TODO: k*nb2 or k*nb3

        float m_k;

        if (k < n_heads_log2_floor) {
          m_k = powf(m0, k + 1);
        } else {
          m_k = powf(m1, 2 * (k - n_heads_log2_floor) + 1);
        }

        pdst[0] = (i - ne0 + 1) * m_k + src[0];
      }
    }
  }
}

static void ne_compute_forward_alibi_f16(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                         const struct ne_tensor* src1, struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(src1->type == NE_TYPE_I32);
  assert(ne_nelements(src1) == 3);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n_past = ((int32_t*)src1->data)[0];
  const int n_head = ((int32_t*)src1->data)[1];
  const float max_bias = ((float*)src1->data)[2];

  assert(n_past >= 0);

  const int ne0 = src0->ne[0];  // all_seq_len = n_past + ne1
  const int ne1 = src0->ne[1];  // seq_len_without_past
  // const int ne2 = src0->ne[2]; // n_head -> this is k
  // const int ne3 = src0->ne[3]; // 1 -> bsz

  const int n = ne_nrows(src0);
  const int ne2_ne3 = n / ne1;  // ne2*ne3

  const size_t nb0 = src0->nb[0];
  const size_t nb1 = src0->nb[1];
  const size_t nb2 = src0->nb[2];
  // const size_t nb3 = src0->nb[3];

  assert(nb0 == sizeof(ne_fp16_t));
  assert(ne1 + n_past == ne0);
  (void)n_past;

  // add alibi to src0 (KQ_scaled)
  const int n_heads_log2_floor = 1 << (int)floor(log2(n_head));

  const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
  const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

  for (int i = 0; i < ne0; i++) {
    for (int j = 0; j < ne1; j++) {
      for (int k = 0; k < ne2_ne3; k++) {
        ne_fp16_t* const src = (ne_fp16_t*)((char*)src0->data + i * nb0 + j * nb1 + k * nb2);
        float* pdst = (float*)((char*)dst->data + i * nb0 + j * nb1 + k * nb2);

        // TODO: k*nb2 or k*nb3

        float m_k;

        if (k < n_heads_log2_floor) {
          m_k = powf(m0, k + 1);
        } else {
          m_k = powf(m1, 2 * (k - n_heads_log2_floor) + 1);
        }

        // we return F32
        pdst[0] = (i - ne0 + 1) * m_k + NE_FP16_TO_FP32(src[0]);
      }
    }
  }
}

static void ne_compute_forward_alibi(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                     const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_alibi_f16(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_alibi_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_Q4_0:
    case NE_TYPE_Q4_1:
    case NE_TYPE_Q5_0:
    case NE_TYPE_Q5_1:
    case NE_TYPE_Q8_0:
    case NE_TYPE_Q8_1:
    case NE_TYPE_I8:
    case NE_TYPE_I16:
    case NE_TYPE_I32:
    case NE_TYPE_COUNT: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_clamp

static void ne_compute_forward_clamp_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                         const struct ne_tensor* src1, struct ne_tensor* dst) {
  assert(params->ith == 0);
  assert(src1->type == NE_TYPE_I32);
  assert(ne_nelements(src1) == 2);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int min = ((float*)src1->data)[0];
  const int max = ((float*)src1->data)[1];

  const int ith = params->ith;
  const int nth = params->nth;

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];

  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb00 == sizeof(float));

  for (int j = ith; j < n; j += nth) {
    float* dst_ptr = (float*)((char*)dst->data + j * nb1);
    float* src0_ptr = (float*)((char*)src0->data + j * nb01);

    for (int i = 0; i < nc; i++) {
      dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
    }
  }
}

static void ne_compute_forward_clamp(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                     const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_clamp_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_F16:
    case NE_TYPE_Q4_0:
    case NE_TYPE_Q4_1:
    case NE_TYPE_Q5_0:
    case NE_TYPE_Q5_1:
    case NE_TYPE_Q8_0:
    case NE_TYPE_Q8_1:
    case NE_TYPE_I8:
    case NE_TYPE_I16:
    case NE_TYPE_I32:
    case NE_TYPE_COUNT: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_rope

static void ne_compute_forward_rope_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src1->type == NE_TYPE_I32);
  NE_ASSERT(ne_nelements(src1) == 3);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n_past = ((int32_t*)src1->data)[0];
  const int n_dims = ((int32_t*)src1->data)[1];
  const int mode = ((int32_t*)src1->data)[2];

  assert(n_past >= 0);

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  // printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
  // printf("n_past = %d, ne2 = %d\n", n_past, ne2);

  NE_ASSERT(nb00 == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(dst);

  NE_ASSERT(n_dims <= ne0);
  NE_ASSERT(n_dims % 2 == 0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(10000.0, -2.0f / n_dims);

  const bool is_neox = mode & 2;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = ((mode & 1) == 0 ? 0 : n_past); i2 < ne2; i2++) {
      const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) continue;
        if (ir > ir1) break;

        float theta = (float)p;

        if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            theta *= theta_scale;

            const float* const src = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0 * cos_theta - x1 * sin_theta;
            dst_data[1] = x0 * sin_theta + x1 * cos_theta;
          }
        } else {
          // TODO: this is probably wrong, but I can't figure it out ..
          // ref:
          // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#LL251C1-L294C28
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const float* const src = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float x0 = src[0];
              const float x1 = src[n_dims / 2];

              dst_data[0] = x0 * cos_theta - x1 * sin_theta;
              dst_data[n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
            }
          }
        }
      }
    }
  }
}

static void ne_compute_forward_rope_f16(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                        const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src1->type == NE_TYPE_I32);
  NE_ASSERT(ne_nelements(src1) == 3);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n_past = ((int32_t*)src1->data)[0];
  const int n_dims = ((int32_t*)src1->data)[1];
  const int mode = ((int32_t*)src1->data)[2];

  assert(n_past >= 0);

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  // printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
  // printf("n_past = %d, ne2 = %d\n", n_past, ne2);

  NE_ASSERT(nb0 == sizeof(ne_fp16_t));

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(dst);

  NE_ASSERT(n_dims <= ne0);
  NE_ASSERT(n_dims % 2 == 0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(10000.0, -2.0f / n_dims);

  const bool is_neox = mode & 2;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = ((mode & 1) == 0 ? 0 : n_past); i2 < ne2; i2++) {
      const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) continue;
        if (ir > ir1) break;

        float theta = (float)p;

        if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            theta *= theta_scale;

            const ne_fp16_t* const src =
                (ne_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            ne_fp16_t* dst_data = (ne_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float x0 = NE_FP16_TO_FP32(src[0]);
            const float x1 = NE_FP16_TO_FP32(src[1]);

            dst_data[0] = NE_FP32_TO_FP16(x0 * cos_theta - x1 * sin_theta);
            dst_data[1] = NE_FP32_TO_FP16(x0 * sin_theta + x1 * cos_theta);
          }
        } else {
          // TODO: this is probably wrong, but I can't figure it out ..
          // ref:
          // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#LL251C1-L294C28
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const ne_fp16_t* const src =
                  (ne_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              ne_fp16_t* dst_data = (ne_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float x0 = NE_FP16_TO_FP32(src[0]);
              const float x1 = NE_FP16_TO_FP32(src[n_dims / 2]);

              dst_data[0] = NE_FP32_TO_FP16(x0 * cos_theta - x1 * sin_theta);
              dst_data[n_dims / 2] = NE_FP32_TO_FP16(x0 * sin_theta + x1 * cos_theta);
            }
          }
        }
      }
    }
  }
}

static void ne_compute_forward_rope(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                    const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_rope_f16(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_rope_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_rope_back

static void ne_compute_forward_rope_back_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                             const struct ne_tensor* src1, struct ne_tensor* dst) {
  assert(src1->type == NE_TYPE_I32);
  assert(ne_nelements(src1) == 3);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  // y = rope(x, src1)
  // dx = rope_back(dy, src1)
  // src0 is dy, src1 contains options

  const int n_past = ((int32_t*)src1->data)[0];
  const int n_dims = ((int32_t*)src1->data)[1];
  const int mode = ((int32_t*)src1->data)[2];

  assert(n_past >= 0);

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  // printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
  // printf("n_past = %d, ne2 = %d\n", n_past, ne2);

  assert(nb0 == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(dst);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(10000.0, -2.0f / n_dims);

  const bool is_neox = mode & 2;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = ((mode & 1) == 0 ? 0 : n_past); i2 < ne2; i2++) {
      const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) continue;
        if (ir > ir1) break;

        float theta = (float)p;

        if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            theta *= theta_scale;

            const float* const dy = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            float* dx = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float dy0 = dy[0];
            const float dy1 = dy[1];

            dx[0] = dy0 * cos_theta + dy1 * sin_theta;
            dx[1] = -dy0 * sin_theta + dy1 * cos_theta;
          }
        } else {
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const float* const dy = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              float* dx = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float dy0 = dy[0];
              const float dy1 = dy[n_dims / 2];

              dx[0] = dy0 * cos_theta + dy1 * sin_theta;
              dx[n_dims / 2] = -dy0 * sin_theta + dy1 * cos_theta;
            }
          }
        }
      }
    }
  }
}

static void ne_compute_forward_rope_back_f16(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                             const struct ne_tensor* src1, struct ne_tensor* dst) {
  assert(src1->type == NE_TYPE_I32);
  assert(ne_nelements(src1) == 3);

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  // y = rope(x, src1)
  // dx = rope_back(dy, src1)
  // src0 is dy, src1 contains options

  const int n_past = ((int32_t*)src1->data)[0];
  const int n_dims = ((int32_t*)src1->data)[1];
  const int mode = ((int32_t*)src1->data)[2];

  assert(n_past >= 0);

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  // printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
  // printf("n_past = %d, ne2 = %d\n", n_past, ne2);

  assert(nb0 == sizeof(ne_fp16_t));

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ne_nrows(dst);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(10000.0, -2.0f / n_dims);

  const bool is_neox = mode & 2;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = ((mode & 1) == 0 ? 0 : n_past); i2 < ne2; i2++) {
      const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) continue;
        if (ir > ir1) break;

        float theta = (float)p;

        if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            theta *= theta_scale;

            const ne_fp16_t* const dy = (ne_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            ne_fp16_t* dx = (ne_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float dy0 = NE_FP16_TO_FP32(dy[0]);
            const float dy1 = NE_FP16_TO_FP32(dy[1]);

            dx[0] = NE_FP32_TO_FP16(dy0 * cos_theta + dy1 * sin_theta);
            dx[1] = NE_FP32_TO_FP16(-dy0 * sin_theta + dy1 * cos_theta);
          }
        } else {
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const ne_fp16_t* const dy =
                  (ne_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              ne_fp16_t* dx = (ne_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float dy0 = NE_FP16_TO_FP32(dy[0]);
              const float dy1 = NE_FP16_TO_FP32(dy[n_dims / 2]);

              dx[0] = NE_FP32_TO_FP16(dy0 * cos_theta + dy1 * sin_theta);
              dx[n_dims / 2] = NE_FP32_TO_FP16(-dy0 * sin_theta + dy1 * cos_theta);
            }
          }
        }
      }
    }
  }
}

static void ne_compute_forward_rope_back(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                         const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_rope_back_f16(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_rope_back_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_conv_1d_1s

static void ne_compute_forward_conv_1d_1s_f16_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                                  const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F16);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  // const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  // const int64_t ne12 = src1->ne[2];
  // const int64_t ne13 = src1->ne[3];

  // const int64_t ne0  = dst->ne[0];
  // const int64_t ne1  = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];
  // const int64_t ne   = ne0*ne1*ne2*ne3;

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  // const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  // const size_t nb12 = src1->nb[2];
  // const size_t nb13 = src1->nb[3];

  // const size_t nb0  = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  // const size_t nb2  = dst->nb[2];
  // const size_t nb3  = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO: support even kernel sizes
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      ne_fp16_t* const wdata = (ne_fp16_t*)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const ne_fp16_t* const src = (ne_fp16_t*)((char*)src0->data + i02 * nb02 + i01 * nb01);
          ne_fp16_t* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      ne_fp16_t* const wdata = (ne_fp16_t*)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = (float*)((char*)src1->data + i11 * nb11);
        ne_fp16_t* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = NE_FP32_TO_FP16(src[i10]);
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = (float*)((char*)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; ++i0) {
      dst_data[i0] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f16(ew0, &v, (ne_fp16_t*)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                       (ne_fp16_t*)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0] += v;
      }
    }
  }
}

static void ne_compute_forward_conv_1d_1s_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                              const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F32);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  // const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  // const int64_t ne12 = src1->ne[2];
  // const int64_t ne13 = src1->ne[3];

  // const int64_t ne0  = dst->ne[0];
  // const int64_t ne1  = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];
  // const int64_t ne   = ne0*ne1*ne2*ne3;

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  // const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  // const size_t nb12 = src1->nb[2];
  // const size_t nb13 = src1->nb[3];

  // const size_t nb0  = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  // const size_t nb2  = dst->nb[2];
  // const size_t nb3  = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO: support even kernel sizes
  NE_ASSERT(nb00 == sizeof(float));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      float* const wdata = (float*)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float* const src = (float*)((char*)src0->data + i02 * nb02 + i01 * nb01);
          float* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float* const wdata = (float*)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = (float*)((char*)src1->data + i11 * nb11);
        float* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = (float*)((char*)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; ++i0) {
      dst_data[i0] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f32(ew0, &v, (float*)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                       (float*)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0] += v;
      }
    }
  }
}

static void ne_compute_forward_conv_1d_1s(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                          const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_conv_1d_1s_f16_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_conv_1d_1s_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_conv_1d_2s

static void ne_compute_forward_conv_1d_2s_f16_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                                  const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F16);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  // const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  // const int64_t ne12 = src1->ne[2];
  // const int64_t ne13 = src1->ne[3];

  // const int64_t ne0  = dst->ne[0];
  // const int64_t ne1  = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];
  // const int64_t ne   = ne0*ne1*ne2*ne3;

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  // const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  // const size_t nb12 = src1->nb[2];
  // const size_t nb13 = src1->nb[3];

  // const size_t nb0  = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  // const size_t nb2  = dst->nb[2];
  // const size_t nb3  = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO: support even kernel sizes
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      ne_fp16_t* const wdata = (ne_fp16_t*)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const ne_fp16_t* const src = (ne_fp16_t*)((char*)src0->data + i02 * nb02 + i01 * nb01);
          ne_fp16_t* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      ne_fp16_t* const wdata = (ne_fp16_t*)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = (float*)((char*)src1->data + i11 * nb11);
        ne_fp16_t* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = NE_FP32_TO_FP16(src[i10]);
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = (float*)((char*)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
      dst_data[i0 / 2] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f16(ew0, &v, (ne_fp16_t*)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                       (ne_fp16_t*)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0 / 2] += v;
      }
    }
  }
}

static void ne_compute_forward_conv_1d_2s_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                              const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F32);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  // const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  // const int64_t ne12 = src1->ne[2];
  // const int64_t ne13 = src1->ne[3];

  // const int64_t ne0  = dst->ne[0];
  // const int64_t ne1  = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];
  // const int64_t ne   = ne0*ne1*ne2*ne3;

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  // const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  // const size_t nb12 = src1->nb[2];
  // const size_t nb13 = src1->nb[3];

  // const size_t nb0  = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  // const size_t nb2  = dst->nb[2];
  // const size_t nb3  = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO: support even kernel sizes
  NE_ASSERT(nb00 == sizeof(float));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      float* const wdata = (float*)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float* const src = (float*)((char*)src0->data + i02 * nb02 + i01 * nb01);
          float* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float* const wdata = (float*)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = (float*)((char*)src1->data + i11 * nb11);
        float* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = (float*)((char*)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
      dst_data[i0 / 2] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f32(ew0, &v, (float*)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                       (float*)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0 / 2] += v;
      }
    }
  }
}

static void ne_compute_forward_conv_1d_2s(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                          const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_conv_1d_2s_f16_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_conv_1d_2s_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_flash_attn

static void ne_compute_forward_flash_attn_f32(const struct ne_compute_params* params, const struct ne_tensor* q,
                                              const struct ne_tensor* k, const struct ne_tensor* v, const bool masked,
                                              struct ne_tensor* dst) {
  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t neq0 = q->ne[0];
  const int64_t neq1 = q->ne[1];
  const int64_t neq2 = q->ne[2];
  const int64_t neq3 = q->ne[3];

  const int64_t nek0 = k->ne[0];
  const int64_t nek1 = k->ne[1];
  // const int64_t nek2 = k->ne[2];
  // const int64_t nek3 = k->ne[3];

  // const int64_t nev0 = v->ne[0];
  const int64_t nev1 = v->ne[1];
  // const int64_t nev2 = v->ne[2];
  // const int64_t nev3 = v->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];

  const size_t nbk0 = k->nb[0];
  const size_t nbk1 = k->nb[1];
  const size_t nbk2 = k->nb[2];
  const size_t nbk3 = k->nb[3];

  const size_t nbq0 = q->nb[0];
  const size_t nbq1 = q->nb[1];
  const size_t nbq2 = q->nb[2];
  const size_t nbq3 = q->nb[3];

  const size_t nbv0 = v->nb[0];
  const size_t nbv1 = v->nb[1];
  const size_t nbv2 = v->nb[2];
  const size_t nbv3 = v->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t D = neq0;
  const int64_t N = neq1;
  const int64_t P = nek1 - N;
  const int64_t M = P + N;

  const int Mup = ne_up(M, NE_SOFT_MAX_UNROLL);

  NE_ASSERT(ne0 == D);
  NE_ASSERT(ne1 == N);
  NE_ASSERT(P >= 0);

  NE_ASSERT(nbq0 == sizeof(float));
  NE_ASSERT(nbk0 == sizeof(float));
  NE_ASSERT(nbv0 == sizeof(float));

  NE_ASSERT(neq0 == D);
  NE_ASSERT(nek0 == D);
  NE_ASSERT(nev1 == D);

  NE_ASSERT(neq1 == N);
  NE_ASSERT(nek1 == N + P);
  NE_ASSERT(nev1 == D);

  // dst cannot be transposed or permuted
  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb0 <= nb1);
  NE_ASSERT(nb1 <= nb2);
  NE_ASSERT(nb2 <= nb3);

  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // parallelize by q rows using ne_vec_dot_f32

  // total rows in q
  const int nr = neq1 * neq2 * neq3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const float scale = 1.0f / sqrtf(D);

  // printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

  for (int ir = ir0; ir < ir1; ++ir) {
    // q indices
    const int iq3 = ir / (neq2 * neq1);
    const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
    const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

    float* S = (float*)params->wdata + ith * (Mup + CACHE_LINE_SIZE_F32);

    for (int i = M; i < Mup; ++i) {
      S[i] = -INFINITY;
    }

    for (int64_t ic = 0; ic < nek1; ++ic) {
      // k indices
      const int ik3 = iq3;
      const int ik2 = iq2;
      const int ik1 = ic;

      // S indices
      const int i1 = ik1;

      ne_vec_dot_f32(neq0, S + i1, (float*)((char*)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                     (float*)((char*)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
    }

    // scale
    ne_vec_scale_f32(nek1, S, scale);

    if (masked) {
      for (int64_t i = P; i < M; i++) {
        if (i > P + iq1) {
          S[i] = -INFINITY;
        }
      }
    }

    // softmax
    {
      float max = -INFINITY;
      ne_vec_max_f32(M, &max, S);

      ne_float sum = 0.0;
      {
        uint16_t scvt[NE_SOFT_MAX_UNROLL];
        ne_float sump[NE_SOFT_MAX_UNROLL] = {0.0};

        for (int i = 0; i < Mup; i += NE_SOFT_MAX_UNROLL) {
          float* SS = S + i;

          for (int j = 0; j < NE_SOFT_MAX_UNROLL; ++j) {
            if (SS[j] == -INFINITY) {
              SS[j] = 0.0f;
            } else {
              ne_fp16_t s = NE_FP32_TO_FP16(SS[j] - max);
              memcpy(&scvt[j], &s, sizeof(uint16_t));
              const float val = NE_FP16_TO_FP32(table_exp_f16[scvt[j]]);
              sump[j] += (ne_float)val;
              SS[j] = val;
            }
          }
        }

        for (int i = 0; i < NE_SOFT_MAX_UNROLL; i++) {
          sum += sump[i];
        }
      }

      assert(sum > 0.0);

      sum = 1.0 / sum;
      ne_vec_scale_f32(M, S, sum);

#ifndef NDEBUG
      for (int i = 0; i < M; ++i) {
        assert(!isnan(S[i]));
        assert(!isinf(S[i]));
      }
#endif
    }

    for (int64_t ic = 0; ic < nev1; ++ic) {
      // dst indices
      const int i1 = iq1;
      const int i2 = iq2;
      const int i3 = iq3;

      ne_vec_dot_f32(nek1, (float*)((char*)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                     (float*)((char*)v->data + (ic * nbv1 + i2 * nbv2 + i3 * nbv3)), S);
    }
  }
}

static void ne_compute_forward_flash_attn_f32_f16_f16(const struct ne_compute_params* params, const struct ne_tensor* q,
                                                      const struct ne_tensor* k, const struct ne_tensor* v,
                                                      const struct ne_tensor* tmp, struct ne_tensor* dst) {
  const int64_t neq0 = q->ne[0];
  const int64_t neq1 = q->ne[1];
  const int64_t neq2 = q->ne[2];
  const int64_t neq3 = q->ne[3];

  const int64_t nek0 = k->ne[0];
  const int64_t nek1 = k->ne[1];
  // const int64_t nek2 = k->ne[2];
  // const int64_t nek3 = k->ne[3];

  // const int64_t nev0 = v->ne[0];
  const int64_t nev1 = v->ne[1];
  // const int64_t nev2 = v->ne[2];
  // const int64_t nev3 = v->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];

  const int nbk0 = k->nb[0];
  const int nbk1 = k->nb[1];
  const int nbk2 = k->nb[2];
  const int nbk3 = k->nb[3];

  const int nbq0 = q->nb[0];
  const int nbq1 = q->nb[1];
  const int nbq2 = q->nb[2];
  const int nbq3 = q->nb[3];

  const int nbv0 = v->nb[0];
  const int nbv1 = v->nb[1];
  const int nbv2 = v->nb[2];
  const int nbv3 = v->nb[3];

  const int nb0 = dst->nb[0];
  const int nb1 = dst->nb[1];
  const int nb2 = dst->nb[2];
  const int nb3 = dst->nb[3];

  const int64_t headsize = neq0;
  const int64_t headnum = neq2;
  const int64_t embedsize = headnum * headsize;
  const int64_t seq_cur = neq1;
  const int64_t seq_all = nek1;
  const int64_t seq_past = seq_all - seq_cur;
  const int64_t batch = neq3;

  const int step_k_bs = k->nb[3] / sizeof(float);
  const int step_v_bs = v->nb[3] / sizeof(float);

  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }
  float scale = *(float*)dst->padding;
  bool mask = *(bool*)&dst->padding[sizeof(scale)];
  attn_fp32_fp16_fp16_fp32_fwd_args_t args = {
      .Q = (float*)q->data,
      .K = (ne_fp16_t*)k->data,
      .V = (ne_fp16_t*)v->data,
      .batch_size = batch,
      .head_size = headsize,
      .head_num = headnum,
      .dst = (float*)dst->data,
      .is_causal = mask,
      .QK_scale = scale,
      .sl_q = seq_cur,
      .sl_kv = seq_all,
      .step_q_bs = seq_cur * embedsize,
      .step_q_head_num = headsize,
      .step_q_sl = embedsize,
      .step_k_bs = step_k_bs,
      .step_k_head_num = headsize,
      .step_k_sl = embedsize,
      .step_k_head_size = 1,  // TODO
      .step_v_bs = step_v_bs,
      .step_v_head_num = headsize,
      .step_v_sl = embedsize,
      .step_dst_bs = seq_cur * embedsize,
      .step_dst_head_num = headsize,
      .step_dst_sl = embedsize,
      .tmp = tmp->data,
  };
  jblas_fusion_attn_fp32_fp16_fp16_fp32_forward(&args);
}

static void ne_compute_forward_flash_attn_f16(const struct ne_compute_params* params, const struct ne_tensor* q,
                                              const struct ne_tensor* k, const struct ne_tensor* v, const bool masked,
                                              struct ne_tensor* dst) {
  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t neq0 = q->ne[0];
  const int64_t neq1 = q->ne[1];
  const int64_t neq2 = q->ne[2];
  const int64_t neq3 = q->ne[3];

  const int64_t nek0 = k->ne[0];
  const int64_t nek1 = k->ne[1];
  // const int64_t nek2 = k->ne[2];
  // const int64_t nek3 = k->ne[3];

  // const int64_t nev0 = v->ne[0];
  const int64_t nev1 = v->ne[1];
  // const int64_t nev2 = v->ne[2];
  // const int64_t nev3 = v->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];

  const size_t nbk0 = k->nb[0];
  const size_t nbk1 = k->nb[1];
  const size_t nbk2 = k->nb[2];
  const size_t nbk3 = k->nb[3];

  const size_t nbq0 = q->nb[0];
  const size_t nbq1 = q->nb[1];
  const size_t nbq2 = q->nb[2];
  const size_t nbq3 = q->nb[3];

  const size_t nbv0 = v->nb[0];
  const size_t nbv1 = v->nb[1];
  const size_t nbv2 = v->nb[2];
  const size_t nbv3 = v->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t D = neq0;
  const int64_t N = neq1;
  const int64_t P = nek1 - N;
  const int64_t M = P + N;

  const int Mup = ne_up(M, NE_SOFT_MAX_UNROLL);

  NE_ASSERT(ne0 == D);
  NE_ASSERT(ne1 == N);
  NE_ASSERT(P >= 0);

  NE_ASSERT(nbq0 == sizeof(ne_fp16_t));
  NE_ASSERT(nbk0 == sizeof(ne_fp16_t));
  NE_ASSERT(nbv0 == sizeof(ne_fp16_t));

  NE_ASSERT(neq0 == D);
  NE_ASSERT(nek0 == D);
  NE_ASSERT(nev1 == D);

  NE_ASSERT(neq1 == N);
  NE_ASSERT(nek1 == N + P);
  NE_ASSERT(nev1 == D);

  // dst cannot be transposed or permuted
  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb0 <= nb1);
  NE_ASSERT(nb1 <= nb2);
  NE_ASSERT(nb2 <= nb3);

  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // parallelize by q rows using ne_vec_dot_f32

  // total rows in q
  const int nr = neq1 * neq2 * neq3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const float scale = 1.0f / sqrtf(D);

  // printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

  for (int ir = ir0; ir < ir1; ++ir) {
    // q indices
    const int iq3 = ir / (neq2 * neq1);
    const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
    const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

    float* S = (float*)params->wdata + ith * (2 * Mup + CACHE_LINE_SIZE_F32);

    for (int i = M; i < Mup; ++i) {
      S[i] = -INFINITY;
    }

    if (NE_VEC_DOT_UNROLL > 2 || nek1 % NE_VEC_DOT_UNROLL != 0) {
      for (int64_t ic = 0; ic < nek1; ++ic) {
        // k indices
        const int ik3 = iq3;
        const int ik2 = iq2;
        const int ik1 = ic;

        // S indices
        const int i1 = ik1;

        ne_vec_dot_f16(neq0, S + i1, (ne_fp16_t*)((char*)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                       (ne_fp16_t*)((char*)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
      }
    } else {
      for (int64_t ic = 0; ic < nek1; ic += NE_VEC_DOT_UNROLL) {
        // k indices
        const int ik3 = iq3;
        const int ik2 = iq2;
        const int ik1 = ic;

        // S indices
        const int i1 = ik1;

        ne_vec_dot_f16_unroll(neq0, nbk1, S + i1, ((char*)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                              (ne_fp16_t*)((char*)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
      }
    }

    // scale
    ne_vec_scale_f32(nek1, S, scale);

    if (masked) {
      for (int64_t i = P; i < M; i++) {
        if (i > P + iq1) {
          S[i] = -INFINITY;
        }
      }
    }

    // softmax
    {
      float max = -INFINITY;
      ne_vec_max_f32(M, &max, S);

      ne_float sum = 0.0;
      {
        uint16_t scvt[NE_SOFT_MAX_UNROLL];
        ne_float sump[NE_SOFT_MAX_UNROLL] = {0.0};

        for (int i = 0; i < Mup; i += NE_SOFT_MAX_UNROLL) {
          float* SS = S + i;

          for (int j = 0; j < NE_SOFT_MAX_UNROLL; ++j) {
            if (SS[j] == -INFINITY) {
              SS[j] = 0.0f;
            } else {
              ne_fp16_t s = NE_FP32_TO_FP16(SS[j] - max);
              memcpy(&scvt[j], &s, sizeof(uint16_t));
              const float val = NE_FP16_TO_FP32(table_exp_f16[scvt[j]]);
              sump[j] += (ne_float)val;
              SS[j] = val;
            }
          }
        }

        for (int i = 0; i < NE_SOFT_MAX_UNROLL; i++) {
          sum += sump[i];
        }
      }

      assert(sum > 0.0);

      sum = 1.0 / sum;
      ne_vec_scale_f32(M, S, sum);

#ifndef NDEBUG
      for (int i = 0; i < M; ++i) {
        assert(!isnan(S[i]));
        assert(!isinf(S[i]));
      }
#endif
    }

    ne_fp16_t* S16 = (ne_fp16_t*)((float*)params->wdata + ith * (2 * Mup + CACHE_LINE_SIZE_F32) + Mup);

    for (int64_t i = 0; i < M; i++) {
      S16[i] = NE_FP32_TO_FP16(S[i]);
    }

    if (NE_VEC_DOT_UNROLL == 1 || (nev1 % NE_VEC_DOT_UNROLL != 0)) {
      for (int64_t ic = 0; ic < nev1; ++ic) {
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        ne_vec_dot_f16(nek1, (float*)((char*)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                       (ne_fp16_t*)((char*)v->data + (ic * nbv1 + i2 * nbv2 + i3 * nbv3)), S16);
      }
    } else {
      for (int64_t ic = 0; ic < nev1; ic += NE_VEC_DOT_UNROLL) {
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        ne_vec_dot_f16_unroll(nek1, nbv1, (float*)((char*)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                              ((char*)v->data + (ic * nbv1 + i2 * nbv2 + i3 * nbv3)), S16);
      }
    }
  }
}

static void ne_compute_forward_flash_attn(const struct ne_compute_params* params, const struct ne_tensor* q,
                                          const struct ne_tensor* k, const struct ne_tensor* v,
                                          const struct ne_tensor* tmp, struct ne_tensor* dst) {
  switch (q->type) {
    case NE_TYPE_F32: {
      if (k->type == NE_TYPE_F16) {
        ne_compute_forward_flash_attn_f32_f16_f16(params, q, k, v, tmp, dst);
      } else {
        NE_ASSERT(false);
      }
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_flash_ff

static void ne_compute_forward_flash_ff_f16(const struct ne_compute_params* params,
                                            const struct ne_tensor* a,   // F16
                                            const struct ne_tensor* b0,  // F16 fc_w
                                            const struct ne_tensor* b1,  // F32 fc_b
                                            const struct ne_tensor* c0,  // F16 proj_w
                                            const struct ne_tensor* c1,  // F32 proj_b
                                            struct ne_tensor* dst) {
  int64_t t0 = ne_perf_time_us();
  UNUSED(t0);

  const int64_t nea0 = a->ne[0];
  const int64_t nea1 = a->ne[1];
  const int64_t nea2 = a->ne[2];
  const int64_t nea3 = a->ne[3];

  const int64_t neb00 = b0->ne[0];
  const int64_t neb01 = b0->ne[1];
  // const int64_t neb02 = b0->ne[2];
  // const int64_t neb03 = b0->ne[3];

  const int64_t neb10 = b1->ne[0];
  const int64_t neb11 = b1->ne[1];
  // const int64_t neb12 = b1->ne[2];
  // const int64_t neb13 = b1->ne[3];

  const int64_t nec00 = c0->ne[0];
  const int64_t nec01 = c0->ne[1];
  // const int64_t nec02 = c0->ne[2];
  // const int64_t nec03 = c0->ne[3];

  const int64_t nec10 = c1->ne[0];
  const int64_t nec11 = c1->ne[1];
  // const int64_t nec12 = c1->ne[2];
  // const int64_t nec13 = c1->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  // const int64_t ne3 = dst->ne[3];

  const size_t nba0 = a->nb[0];
  const size_t nba1 = a->nb[1];
  const size_t nba2 = a->nb[2];
  const size_t nba3 = a->nb[3];

  const size_t nbb00 = b0->nb[0];
  const size_t nbb01 = b0->nb[1];
  const size_t nbb02 = b0->nb[2];
  const size_t nbb03 = b0->nb[3];

  const size_t nbb10 = b1->nb[0];
  // const size_t nbb11 = b1->nb[1];
  // const size_t nbb12 = b1->nb[2];
  // const size_t nbb13 = b1->nb[3];

  const size_t nbc00 = c0->nb[0];
  const size_t nbc01 = c0->nb[1];
  const size_t nbc02 = c0->nb[2];
  const size_t nbc03 = c0->nb[3];

  const size_t nbc10 = c1->nb[0];
  // const size_t nbc11 = c1->nb[1];
  // const size_t nbc12 = c1->nb[2];
  // const size_t nbc13 = c1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t D = nea0;
  // const int64_t N = nea1;
  const int64_t M = neb01;

  NE_ASSERT(ne0 == nea0);
  NE_ASSERT(ne1 == nea1);
  NE_ASSERT(ne2 == nea2);

  NE_ASSERT(nba0 == sizeof(ne_fp16_t));
  NE_ASSERT(nbb00 == sizeof(ne_fp16_t));
  NE_ASSERT(nbb10 == sizeof(float));
  NE_ASSERT(nbc00 == sizeof(ne_fp16_t));
  NE_ASSERT(nbc10 == sizeof(float));

  NE_ASSERT(neb00 == D);
  NE_ASSERT(neb01 == M);
  NE_ASSERT(neb10 == M);
  NE_ASSERT(neb11 == 1);

  NE_ASSERT(nec00 == M);
  NE_ASSERT(nec01 == D);
  NE_ASSERT(nec10 == D);
  NE_ASSERT(nec11 == 1);

  // dst cannot be transposed or permuted
  NE_ASSERT(nb0 == sizeof(float));
  NE_ASSERT(nb0 <= nb1);
  NE_ASSERT(nb1 <= nb2);
  NE_ASSERT(nb2 <= nb3);

  if (params->type == NE_TASK_INIT) {
    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // parallelize by a rows using ne_vec_dot_f32

  // total rows in a
  const int nr = nea1 * nea2 * nea3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // a indices
    const int ia3 = ir / (nea2 * nea1);
    const int ia2 = (ir - ia3 * nea2 * nea1) / nea1;
    const int ia1 = (ir - ia3 * nea2 * nea1 - ia2 * nea1);

    float* S = (float*)params->wdata + ith * (2 * M + CACHE_LINE_SIZE_F32);

    for (int64_t ic = 0; ic < neb01; ++ic) {
      // b0 indices
      const int ib03 = ia3;
      const int ib02 = ia2;
      const int ib01 = ic;

      // S indices
      const int i1 = ib01;

      ne_vec_dot_f16(nea0, S + i1, (ne_fp16_t*)((char*)b0->data + (ib01 * nbb01 + ib02 * nbb02 + ib03 * nbb03)),
                     (ne_fp16_t*)((char*)a->data + (ia1 * nba1 + ia2 * nba2 + ia3 * nba3)));
    }

    ne_vec_add_f32(neb01, S, S, (float*)b1->data);
    // ne_vec_gelu_f32(neb01, S, S);

    ne_fp16_t* S16 = (ne_fp16_t*)((float*)params->wdata + ith * (2 * M + CACHE_LINE_SIZE_F32) + M);

    for (int64_t i = 0; i < M; i++) {
      S16[i] = NE_FP32_TO_FP16(S[i]);
    }

    ne_vec_gelu_f16(neb01, S16, S16);

    {
      // dst indices
      const int i1 = ia1;
      const int i2 = ia2;
      const int i3 = ia3;

      for (int64_t ic = 0; ic < nec01; ++ic) {
        ne_vec_dot_f16(neb01, (float*)((char*)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                       (ne_fp16_t*)((char*)c0->data + (ic * nbc01 + i2 * nbc02 + i3 * nbc03)), S16);
      }

      ne_vec_add_f32(nec01, (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3)),
                     (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3)), (float*)c1->data);
    }
  }
}

static void ne_compute_forward_flash_ff(const struct ne_compute_params* params, const struct ne_tensor* a,
                                        const struct ne_tensor* b0, const struct ne_tensor* b1,
                                        const struct ne_tensor* c0, const struct ne_tensor* c1, struct ne_tensor* dst) {
  switch (b0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_flash_ff_f16(params, a, b0, b1, c0, c1, dst);
    } break;
    case NE_TYPE_F32: {
      NE_ASSERT(false);  // TODO
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_map_unary

static void ne_compute_forward_map_unary_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                             struct ne_tensor* dst, const ne_unary_op_f32_t fun) {
  NE_ASSERT(ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    fun(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ne_compute_forward_map_unary(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                         struct ne_tensor* dst, const ne_unary_op_f32_t fun) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_map_unary_f32(params, src0, dst, fun);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_map_binary

static void ne_compute_forward_map_binary_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                              const struct ne_tensor* src1, struct ne_tensor* dst,
                                              const ne_binary_op_f32_t fun) {
  assert(params->ith == 0);
  assert(ne_are_same_shape(src0, src1) && ne_are_same_shape(src0, dst));

  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  const int n = ne_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));
  assert(src1->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    fun(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])),
        (float*)((char*)src1->data + i * (src1->nb[1])));
  }
}

static void ne_compute_forward_map_binary(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                          const struct ne_tensor* src1, struct ne_tensor* dst,
                                          const ne_binary_op_f32_t fun) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_map_binary_f32(params, src0, src1, dst, fun);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

/////////////////////////////////

static void ne_compute_forward(struct ne_compute_params* params, struct ne_tensor* tensor) {
  NE_ASSERT(params);

  switch (tensor->op) {
    case NE_OP_DUP: {
      ne_compute_forward_dup(params, tensor->src0, tensor);
    } break;
    case NE_OP_ADD: {
      ne_compute_forward_add(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_ADD1: {
      ne_compute_forward_add1(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_ACC: {
      ne_compute_forward_acc(params, tensor->src0, tensor->src1, tensor->opt[0], tensor);
    } break;
    case NE_OP_SUB: {
      ne_compute_forward_sub(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_MUL: {
      ne_compute_forward_mul(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_DIV: {
      ne_compute_forward_div(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_SQR: {
      ne_compute_forward_sqr(params, tensor->src0, tensor);
    } break;
    case NE_OP_SQRT: {
      ne_compute_forward_sqrt(params, tensor->src0, tensor);
    } break;
    case NE_OP_LOG: {
      ne_compute_forward_log(params, tensor->src0, tensor);
    } break;
    case NE_OP_SUM: {
      ne_compute_forward_sum(params, tensor->src0, tensor);
    } break;
    case NE_OP_SUM_ROWS: {
      ne_compute_forward_sum_rows(params, tensor->src0, tensor);
    } break;
    case NE_OP_MEAN: {
      ne_compute_forward_mean(params, tensor->src0, tensor);
    } break;
    case NE_OP_REPEAT: {
      ne_compute_forward_repeat(params, tensor->src0, tensor);
    } break;
    case NE_OP_ABS: {
      ne_compute_forward_abs(params, tensor->src0, tensor);
    } break;
    case NE_OP_SGN: {
      ne_compute_forward_sgn(params, tensor->src0, tensor);
    } break;
    case NE_OP_NEG: {
      ne_compute_forward_neg(params, tensor->src0, tensor);
    } break;
    case NE_OP_STEP: {
      ne_compute_forward_step(params, tensor->src0, tensor);
    } break;
    case NE_OP_RELU: {
      ne_compute_forward_relu(params, tensor->src0, tensor);
    } break;
    case NE_OP_GELU: {
      ne_compute_forward_gelu(params, tensor->src0, tensor);
    } break;
    case NE_OP_SILU: {
      ne_compute_forward_silu(params, tensor->src0, tensor);
    } break;
    case NE_OP_SILU_BACK: {
      ne_compute_forward_silu_back(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_NORM: {
      ne_compute_forward_norm(params, tensor->src0, tensor);
    } break;
    case NE_OP_RMS_NORM: {
      ne_compute_forward_rms_norm(params, tensor->src0, tensor);
    } break;
    case NE_OP_RMS_NORM_BACK: {
      ne_compute_forward_rms_norm_back(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_MUL_MAT_BIAS: {
      ne_compute_forward_mul_mat_bias(params, tensor->src0, tensor->src1, tensor->opt[0], tensor);
    } break;
    case NE_OP_MUL_MAT: {
      ne_compute_forward_mul_mat(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_MUL_QKV: {
      ne_compute_forward_mul_qkv(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor);
    } break;
    case NE_OP_MUL_FFN_SILU: {
      ne_compute_forward_ffn_silu(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor->opt[2],
                                  tensor->opt[3], tensor);
    } break;
    case NE_OP_MUL_FFN_ADD_GELU: {
      ne_compute_forward_ffn_add_gelu(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1],
                                      tensor->opt[2], tensor->opt[3], tensor);
    } break;
    case NE_OP_MUL_FFN_GELU: {
      ne_compute_forward_ffn_gelu(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor);
    } break;
    case NE_OP_SCALE: {
      ne_compute_forward_scale(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_SET: {
      ne_compute_forward_set(params, tensor->src0, tensor->src1, tensor->opt[0], tensor);
    } break;
    case NE_OP_CPY: {
      ne_compute_forward_cpy(params, tensor->src0, tensor);
    } break;
    case NE_OP_CONT: {
      ne_compute_forward_cont(params, tensor->src0, tensor);
    } break;
    case NE_OP_RESHAPE: {
      ne_compute_forward_reshape(params, tensor->src0, tensor);
    } break;
    case NE_OP_VIEW: {
      ne_compute_forward_view(params, tensor->src0);
    } break;
    case NE_OP_PERMUTE: {
      ne_compute_forward_permute(params, tensor->src0);
    } break;
    case NE_OP_TRANSPOSE: {
      ne_compute_forward_transpose(params, tensor->src0);
    } break;
    case NE_OP_GET_ROWS: {
      ne_compute_forward_get_rows(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_GET_ROWS_BACK: {
      ne_compute_forward_get_rows_back(params, tensor->src0, tensor->src1, tensor->opt[0], tensor);
    } break;
    case NE_OP_DIAG: {
      ne_compute_forward_diag(params, tensor->src0, tensor);
    } break;
    case NE_OP_DIAG_MASK_INF: {
      ne_compute_forward_diag_mask_inf(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_DIAG_MASK_ZERO: {
      ne_compute_forward_diag_mask_zero(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_SOFT_MAX: {
      ne_compute_forward_soft_max(params, tensor->src0, tensor);
    } break;
    case NE_OP_ROPE: {
      ne_compute_forward_rope(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_ROPE_BACK: {
      ne_compute_forward_rope_back(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_ALIBI: {
      ne_compute_forward_alibi(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_CLAMP: {
      ne_compute_forward_clamp(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_CONV_1D_1S: {
      ne_compute_forward_conv_1d_1s(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_CONV_1D_2S: {
      ne_compute_forward_conv_1d_2s(params, tensor->src0, tensor->src1, tensor);
    } break;
    case NE_OP_FLASH_ATTN: {
      ne_compute_forward_flash_attn(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor);
    } break;
    case NE_OP_FLASH_FF: {
      ne_compute_forward_flash_ff(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor->opt[2],
                                  tensor);
    } break;
    case NE_OP_MAP_UNARY: {
      const ne_unary_op_f32_t fun = *((ne_unary_op_f32_t*)tensor->opt[0]->data);
      ne_compute_forward_map_unary(params, tensor->src0, tensor, fun);
    } break;
    case NE_OP_MAP_BINARY: {
      const ne_binary_op_f32_t fun = *((ne_binary_op_f32_t*)tensor->opt[0]->data);
      ne_compute_forward_map_binary(params, tensor->src0, tensor->src1, tensor, fun);
    } break;
    case NE_OP_NONE: {
      // nop
    } break;
    case NE_OP_COUNT: {
      NE_ASSERT(false);
    } break;
  }
}

////////////////////////////////////////////////////////////////////////////////

static void ne_compute_backward(struct ne_context* ctx, struct ne_tensor* tensor, bool inplace) {
  struct ne_tensor* src0 = tensor->src0;
  struct ne_tensor* src1 = tensor->src1;

  switch (tensor->op) {
    case NE_OP_DUP: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
    } break;
    case NE_OP_ADD: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
      if (src1->grad) {
        src1->grad = ne_add_impl(ctx, src1->grad, tensor->grad, inplace);
      }
    } break;
    case NE_OP_ADD1: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
      if (src1->grad) {
        src1->grad =
            ne_add_impl(ctx, src1->grad, ne_mean(ctx, tensor->grad),  // TODO: should probably be sum instead of mean
                        inplace);
      }
    } break;
    case NE_OP_ACC: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
      if (src1->grad) {
        NE_ASSERT(ne_nelements(tensor->opt[0]) == 5);
        NE_ASSERT(tensor->opt[0]->type == NE_TYPE_I32);
        const size_t nb1 = ((int32_t*)tensor->opt[0]->data)[0];
        const size_t nb2 = ((int32_t*)tensor->opt[0]->data)[1];
        const size_t nb3 = ((int32_t*)tensor->opt[0]->data)[2];
        const size_t offset = ((int32_t*)tensor->opt[0]->data)[3];

        struct ne_tensor* tensor_grad_view = ne_view_4d(ctx, tensor->grad, src1->grad->ne[0], src1->grad->ne[1],
                                                        src1->grad->ne[2], src1->grad->ne[3], nb1, nb2, nb3, offset);

        src1->grad = ne_add_impl(ctx, src1->grad, ne_reshape(ctx, ne_cont(ctx, tensor_grad_view), src1->grad), inplace);
      }
    } break;
    case NE_OP_SUB: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
      if (src1->grad) {
        src1->grad = ne_sub_impl(ctx, src1->grad, tensor->grad, inplace);
      }
    } break;
    case NE_OP_MUL: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_mul(ctx, src1, tensor->grad), inplace);
      }
      if (src1->grad) {
        src1->grad = ne_add_impl(ctx, src1->grad, ne_mul(ctx, src0, tensor->grad), inplace);
      }
    } break;
    case NE_OP_DIV: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_div(ctx, tensor->grad, src1), inplace);
      }
      if (src1->grad) {
        src1->grad = ne_sub_impl(ctx, src1->grad, ne_mul(ctx, tensor->grad, ne_div(ctx, tensor, src1)), inplace);
      }
    } break;
    case NE_OP_SQR: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_scale(ctx, ne_mul(ctx, src0, tensor->grad), ne_new_f32(ctx, 2.0f)),
                                 inplace);
      }
    } break;
    case NE_OP_SQRT: {
      if (src0->grad) {
        src0->grad = ne_add_impl(
            ctx, src0->grad,
            ne_mul(ctx,
                   tensor->grad,  // this was not catched by test_grad because in test_grad tensor->grad is 1
                   ne_div(ctx, ne_repeat(ctx, ne_new_f32(ctx, 0.5f), tensor), tensor)),
            inplace);
      }
    } break;
    case NE_OP_LOG: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_div(ctx, tensor->grad, src0), inplace);
      }
    } break;
    case NE_OP_SUM: {
      if (src0->grad) {
        src0->grad = ne_add1_impl(ctx, src0->grad, tensor->grad, inplace);
      }
    } break;
    case NE_OP_SUM_ROWS: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_repeat(ctx, tensor->grad, src0->grad), inplace);
      }
    } break;
    case NE_OP_MEAN: {
      NE_ASSERT(false);  // TODO: implement
    } break;
    case NE_OP_REPEAT: {
      // necessary for llama
      if (src0->grad) {
        NE_ASSERT(src0->n_dims == 1 || src0->n_dims == 2);
        const int nc = tensor->ne[0];
        const int nr = tensor->ne[1];
        const int nc0 = src0->ne[0];
        const int nr0 = src0->ne[1];
        const int ncr = nc / nc0;  // guaranteed to be an integer due to the check in ne_can_repeat
        const int nrr = nr / nr0;  // guaranteed to be an integer due to the check in ne_can_repeat
        // tensor->grad [nc,nr,1,1]
        // reshape      [nc0,nc/nc0,nr0,nr/nr0]
        // permute      [nc0,nr0,nc/nc0,nr/nr0]
        // substitute   [nc0,nr0,ncr,nrr]
        // reshape      [nc0*nr0,ncr*nrr,1,1]
        // transpose    [ncr*nrr,nc0*nr0,1,1]
        // sum rows     [1,nc0*nr0,1,1]
        // transpose    [nc0*nr0,1,1]
        // reshape      [nc0,nr0,1,1] reshape_1d or reshape_2d
        // add to src0->grad

        int64_t ne[4] = {nc0, ncr, nr0, nrr};

        struct ne_tensor* F00 = tensor->grad;
        struct ne_tensor* F01 = ne_reshape(ctx, F00, ne_new_tensor(ctx, tensor->grad->type, 4, ne, NE_SIZE_CALC));
        struct ne_tensor* F02 = ne_permute(ctx, F01, 0, 2, 1, 3);
        struct ne_tensor* F03 = ne_cont(ctx, F02);
        struct ne_tensor* F04 = ne_reshape_2d(ctx, F03, nc0 * nr0, ncr * nrr);
        struct ne_tensor* F05 = ne_transpose(ctx, F04);
        struct ne_tensor* F06 = ne_cont(ctx, F05);
        struct ne_tensor* F07 = ne_sum_rows(ctx, F06);
        struct ne_tensor* F08 = ne_transpose(ctx, F07);
        struct ne_tensor* F09 = ne_cont(ctx, F08);
        struct ne_tensor* F10 = ne_reshape(ctx, F09, src0->grad);

        src0->grad = ne_add_impl(ctx, src0->grad, F10, inplace);
      }
    } break;
    case NE_OP_ABS: {
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_mul(ctx, ne_sgn(ctx, src0), tensor->grad), inplace);
      }
    } break;
    case NE_OP_SGN: {
      if (src0->grad) {
        // noop
      }
    } break;
    case NE_OP_NEG: {
      if (src0->grad) {
        src0->grad = ne_sub_impl(ctx, src0->grad, tensor->grad, inplace);
      }
    } break;
    case NE_OP_STEP: {
      if (src0->grad) {
        // noop
      }
    } break;
    case NE_OP_RELU: {
      if (src0->grad) {
        src0->grad = ne_sub_impl(ctx, src0->grad, ne_mul(ctx, ne_step(ctx, src0), tensor->grad), inplace);
      }
    } break;
    case NE_OP_GELU: {
      NE_ASSERT(false);  // TODO: not implemented
    } break;
    case NE_OP_ALIBI: {
      NE_ASSERT(false);  // TODO: not implemented
    } break;
    case NE_OP_CLAMP: {
      NE_ASSERT(false);  // TODO: not implemented
    } break;
    case NE_OP_SILU: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_silu_back(ctx, src0, tensor->grad), inplace);
      }
    } break;
    case NE_OP_SILU_BACK: {
      NE_ASSERT(false);  // TODO: not implemented
    } break;
    case NE_OP_NORM: {
      NE_ASSERT(false);  // TODO: not implemented
    } break;
    case NE_OP_RMS_NORM: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_rms_norm_back(ctx, src0, tensor->grad), inplace);
      }
    } break;
    case NE_OP_RMS_NORM_BACK: {
      NE_ASSERT(false);  // TODO: not implemented
    } break;
    case NE_OP_MUL_MAT: {
      // https://cs231n.github.io/optimization-2/#staged
      // # forward pass
      // s0 = np.random.randn(5, 10)
      // s1 = np.random.randn(10, 3)
      // t = s0.dot(s1)

      // # now suppose we had the gradient on t from above in the circuit
      // dt = np.random.randn(*t.shape) # same shape as t
      // ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
      // ds1 = t.T.dot(dt)

      // tensor.shape [m,p]
      // src0.shape   [n,m]
      // src1.shape   [n,p]

      // necessary for llama
      if (src0->grad) {
        // TODO: this requires outer product - ne_out_prod(ctx, src1, tensor->grad);
        src0->grad = ne_add_impl(ctx, src0->grad,
                                 // ds0 = dt.dot(s1.T)
                                 // ne_out_prod(ctx, // [n,m]
                                 //     src1,          // [n,p]
                                 //     tensor->grad), // [m,p]
                                 // for now just using A*B==(B.T*A.T).T
                                 ne_cont(ctx,                                                          // [n,m]
                                         ne_transpose(ctx,                                             // [n,m]
                                                      ne_mul_mat(ctx,                                  // [m,n]
                                                                 ne_cont(ctx,                          // [p,m]
                                                                         ne_transpose(ctx,             // [p,m]
                                                                                      tensor->grad)),  // [m,p]
                                                                 ne_cont(ctx,                          // [p,n]
                                                                         ne_transpose(ctx,             // [p,n]
                                                                                      src1))))),       // [n,p]
                                 inplace);
      }
      if (src1->grad) {
        src1->grad = ne_add_impl(ctx, src1->grad,
                                 // ds1 = s0.T.dot(dt):
                                 ne_mul_mat(ctx,                               // [n,p]
                                            ne_cont(ctx,                       // [m,n]
                                                    ne_transpose(ctx, src0)),  // [m,n]
                                            tensor->grad),                     // [m,p]
                                 inplace);
      }
    } break;
    case NE_OP_SCALE: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_scale_impl(ctx, tensor->grad, src1, false), inplace);
      }
      if (src1->grad) {
        src1->grad = ne_add_impl(ctx, src1->grad, ne_sum(ctx, ne_mul_impl(ctx, tensor->grad, src0, false)), inplace);
      }
    } break;
    case NE_OP_SET: {
      NE_ASSERT(ne_nelements(tensor->opt[0]) == 5);
      NE_ASSERT(tensor->opt[0]->type == NE_TYPE_I32);
      const size_t nb1 = ((int32_t*)tensor->opt[0]->data)[0];
      const size_t nb2 = ((int32_t*)tensor->opt[0]->data)[1];
      const size_t nb3 = ((int32_t*)tensor->opt[0]->data)[2];
      const size_t offset = ((int32_t*)tensor->opt[0]->data)[3];

      struct ne_tensor* tensor_grad_view = NULL;

      if (src0->grad || src1->grad) {
        NE_ASSERT(src0->type == tensor->type);
        NE_ASSERT(tensor->grad->type == tensor->type);
        NE_ASSERT(tensor->grad->type == src1->grad->type);

        tensor_grad_view = ne_view_4d(ctx, tensor->grad, src1->grad->ne[0], src1->grad->ne[1], src1->grad->ne[2],
                                      src1->grad->ne[3], nb1, nb2, nb3, offset);
      }

      if (src0->grad) {
        src0->grad = ne_add_impl(
            ctx, src0->grad,
            ne_acc_impl(ctx, tensor->grad, ne_neg(ctx, tensor_grad_view), nb1, nb2, nb3, offset, false), inplace);
      }

      if (src1->grad) {
        src1->grad = ne_add_impl(ctx, src1->grad, ne_reshape(ctx, ne_cont(ctx, tensor_grad_view), src1->grad), inplace);
      }
    } break;
    case NE_OP_CPY: {
      // necessary for llama
      // cpy overwrites value of src1 by src0 and returns view(src1)
      // the overwriting is mathematically equivalent to:
      // tensor = src0 * 1 + src1 * 0
      if (src0->grad) {
        // dsrc0 = dtensor * 1
        src0->grad = ne_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
      if (src1->grad) {
        // dsrc1 = dtensor * 0 -> noop
      }
    } break;
    case NE_OP_CONT: {
      // same as cpy
      if (src0->grad) {
        NE_ASSERT(ne_is_contiguous(src0->grad));
        NE_ASSERT(ne_is_contiguous(tensor->grad));
        src0->grad = ne_add_impl(ctx, src0->grad, tensor->grad, inplace);
      }
    } break;
    case NE_OP_RESHAPE: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_reshape(ctx, tensor->grad, src0->grad), inplace);
      }
    } break;
    case NE_OP_VIEW: {
      // necessary for llama
      if (src0->grad) {
        size_t offset;
        memcpy(&offset, tensor->padding, sizeof(offset));

        size_t nb1 = tensor->nb[1];
        size_t nb2 = tensor->nb[2];
        size_t nb3 = tensor->nb[3];

        if (src0->type != src0->grad->type) {
          // gradient is typically F32, but src0 could be other type
          size_t ng = ne_element_size(src0->grad);
          size_t n0 = ne_element_size(src0);
          NE_ASSERT(offset % n0 == 0);
          NE_ASSERT(nb1 % n0 == 0);
          NE_ASSERT(nb2 % n0 == 0);
          NE_ASSERT(nb3 % n0 == 0);
          offset = (offset / n0) * ng;
          nb1 = (nb1 / n0) * ng;
          nb2 = (nb2 / n0) * ng;
          nb3 = (nb3 / n0) * ng;
        }

        src0->grad = ne_acc_impl(ctx, src0->grad, tensor->grad, nb1, nb2, nb3, offset, inplace);
      }
    } break;
    case NE_OP_PERMUTE: {
      // necessary for llama
      if (src0->grad) {
        int axis0 = tensor->padding[0] & 0x3;
        int axis1 = tensor->padding[1] & 0x3;
        int axis2 = tensor->padding[2] & 0x3;
        int axis3 = tensor->padding[3] & 0x3;
        int axes_backward[4] = {0, 0, 0, 0};
        axes_backward[axis0] = 0;
        axes_backward[axis1] = 1;
        axes_backward[axis2] = 2;
        axes_backward[axis3] = 3;
        src0->grad = ne_add_impl(
            ctx, src0->grad,
            ne_permute(ctx, tensor->grad, axes_backward[0], axes_backward[1], axes_backward[2], axes_backward[3]),
            inplace);
      }
    } break;
    case NE_OP_TRANSPOSE: {
      // necessary for llama
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_transpose(ctx, tensor->grad), inplace);
      }
    } break;
    case NE_OP_GET_ROWS: {
      // necessary for llama (only for tokenizer)
      if (src0->grad) {
        src0->grad = ne_add_impl(ctx, src0->grad, ne_get_rows_back(ctx, tensor->grad, src1, src0->grad), inplace);
      }
      if (src1->grad) {
        // noop
      }
    } break;
    case NE_OP_GET_ROWS_BACK: {
      NE_ASSERT(false);  // TODO: not implemented
    } break;
    case NE_OP_DIAG: {
      NE_ASSERT(false);  // TODO: not implemented
    } break;
    case NE_OP_DIAG_MASK_INF: {
      // necessary for llama
      if (src0->grad) {
        assert(src1->type == NE_TYPE_I32);
        assert(ne_nelements(src1) == 2);
        const int n_past = ((int32_t*)src1->data)[0];
        src0->grad = ne_add_impl(ctx, src0->grad, ne_diag_mask_zero_impl(ctx, tensor->grad, n_past, false), inplace);
      }
      if (src1->grad) {
        // noop
      }
    } break;
    case NE_OP_DIAG_MASK_ZERO: {
      // necessary for llama
      if (src0->grad) {
        assert(src1->type == NE_TYPE_I32);
        assert(ne_nelements(src1) == 2);
        const int n_past = ((int32_t*)src1->data)[0];
        src0->grad = ne_add_impl(ctx, src0->grad, ne_diag_mask_zero_impl(ctx, tensor->grad, n_past, false), inplace);
      }
      if (src1->grad) {
        // noop
      }
    } break;
    case NE_OP_SOFT_MAX: {
      // necessary for llama
      if (src0->grad) {
        // y = softmax(x)
        //
        // Jii = yi - yi*yi
        // Jij = -yi*yj
        // J = diag(y)-y.*y
        // dx = J * dy
        // dxk = sum(Jkj * dyk)

        int64_t ne2[4] = {tensor->ne[0], 1, tensor->ne[1] * tensor->ne[2], tensor->ne[3]};
        struct ne_tensor* tensor2 =
            ne_cont(ctx, ne_reshape_4d(ctx, ne_cont(ctx, tensor), ne2[0], ne2[1], ne2[2], ne2[3]));

        struct ne_tensor* grad2 =
            ne_cont(ctx, ne_reshape_4d(ctx, ne_cont(ctx, tensor->grad), ne2[0], ne2[1], ne2[2], ne2[3]));

        struct ne_tensor* tensor2_t = ne_cont(ctx,                 // [1,ne0,ne1*ne2,ne3]
                                              ne_permute(ctx,      // [1,ne0,ne1*ne2,ne3]
                                                         tensor2,  // [ne0,1,ne1*ne2,ne3]
                                                         1, 0, 2, 3));

        src0->grad = ne_add_impl(ctx,
                                 src0->grad,                                           // [ne0,ne1,ne2,ne3]
                                 ne_reshape(ctx,                                       // [ne0,ne1,ne2,ne3]
                                            ne_mul_mat(ctx,                            // [ne0,1,ne1*ne2,ne3]
                                                       ne_sub(ctx,                     // [ne0,ne0,ne1*ne2,ne3]
                                                              ne_diag(ctx,             // [ne0,ne0,ne1*ne2,ne3]
                                                                      tensor2),        // [ne0,1,ne1*ne2,ne3]
                                                              ne_mul_mat(ctx,          // [ne0,ne0,ne1*ne2,ne3]
                                                                         tensor2_t,    // [1,ne0,ne1*ne2,ne3]
                                                                         tensor2_t)),  // [1,ne0,ne1*ne2,ne3]
                                                       grad2),                         // [ne0,1,ne1*ne2,ne3]
                                            src0->grad),
                                 inplace);
      }
    } break;
    case NE_OP_ROPE: {
      // necessary for llama
      if (src0->grad) {
        assert(src1->type == NE_TYPE_I32);
        assert(ne_nelements(src1) == 3);
        const int n_past = ((int32_t*)src1->data)[0];
        const int n_dims = ((int32_t*)src1->data)[1];
        const int mode = ((int32_t*)src1->data)[2];
        src0->grad = ne_add_impl(ctx, src0->grad, ne_rope_back(ctx, tensor->grad, n_past, n_dims, mode), inplace);
      }
      if (src1->grad) {
        // noop
      }
    } break;
    case NE_OP_ROPE_BACK: {
      if (src0->grad) {
        assert(src1->type == NE_TYPE_I32);
        assert(ne_nelements(src1) == 3);
        const int n_past = ((int32_t*)src1->data)[0];
        const int n_dims = ((int32_t*)src1->data)[1];
        const int mode = ((int32_t*)src1->data)[2];
        src0->grad = ne_add_impl(ctx, src0->grad, ne_rope(ctx, tensor->grad, n_past, n_dims, mode), inplace);
      }
      if (src1->grad) {
        // noop
      }
    } break;
    case NE_OP_CONV_1D_1S: {
      NE_ASSERT(false);  // TODO: not implemented
    } break;
    case NE_OP_CONV_1D_2S: {
      NE_ASSERT(false);  // TODO: not implemented
    } break;
    case NE_OP_FLASH_ATTN: {
      NE_ASSERT(false);  // not supported
    } break;
    case NE_OP_FLASH_FF: {
      NE_ASSERT(false);  // not supported
    } break;
    case NE_OP_MAP_UNARY:
    case NE_OP_MAP_BINARY: {
      NE_ASSERT(false);  // not supported
    } break;
    case NE_OP_NONE: {
      // nop
    } break;
    case NE_OP_COUNT: {
      NE_ASSERT(false);
    } break;
  }
}

static void ne_visit_parents(struct ne_cgraph* cgraph, struct ne_tensor* node) {
  if (node->grad == NULL) {
    // this usually happens when we generate intermediate nodes from constants in the backward pass
    // it can also happen during forward pass, if the user performs computations with constants
    if (node->op != NE_OP_NONE) {
      // NE_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
    }
  }

  // check if already visited
  for (int i = 0; i < cgraph->n_nodes; i++) {
    if (cgraph->nodes[i] == node) {
      return;
    }
  }

  for (int i = 0; i < cgraph->n_leafs; i++) {
    if (cgraph->leafs[i] == node) {
      return;
    }
  }

  if (node->src0) {
    ne_visit_parents(cgraph, node->src0);
  }

  if (node->src1) {
    ne_visit_parents(cgraph, node->src1);
  }

  for (int i = 0; i < NE_MAX_OPT; ++i) {
    if (node->opt[i]) {
      ne_visit_parents(cgraph, node->opt[i]);
    }
  }

  if (node->op == NE_OP_NONE && node->grad == NULL) {
    // reached a leaf node, not part of the gradient graph (e.g. a constant)
    NE_ASSERT(cgraph->n_leafs < NE_MAX_NODES);

    cgraph->leafs[cgraph->n_leafs] = node;
    cgraph->n_leafs++;
  } else {
    NE_ASSERT(cgraph->n_nodes < NE_MAX_NODES);

    cgraph->nodes[cgraph->n_nodes] = node;
    cgraph->grads[cgraph->n_nodes] = node->grad;
    cgraph->n_nodes++;
  }
}

static void ne_build_forward_impl(struct ne_cgraph* cgraph, struct ne_tensor* tensor, bool expand) {
  if (!expand) {
    cgraph->n_nodes = 0;
    cgraph->n_leafs = 0;
  }

  const int n0 = cgraph->n_nodes;
  UNUSED(n0);

  ne_visit_parents(cgraph, tensor);

  const int n_new = cgraph->n_nodes - n0;
  NE_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

  if (n_new > 0) {
    // the last added node should always be starting point
    NE_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
  }
}

void ne_build_forward_expand(struct ne_cgraph* cgraph, struct ne_tensor* tensor) {
  ne_build_forward_impl(cgraph, tensor, true);
}

struct ne_cgraph ne_build_forward(struct ne_tensor* tensor) {
  struct ne_cgraph result = {
      /*.n_nodes      =*/0,
      /*.n_leafs      =*/0,
      /*.n_threads    =*/NE_DEFAULT_N_THREADS,
      /*.work_size    =*/0,
      /*.work         =*/NULL,
      /*.nodes        =*/{NULL},
      /*.grads        =*/{NULL},
      /*.leafs        =*/{NULL},
      /*.perf_runs    =*/0,
      /*.perf_cycles  =*/0,
      /*.perf_time_us =*/0,
  };

  ne_build_forward_impl(&result, tensor, false);

  return result;
}

struct ne_cgraph ne_build_backward(struct ne_context* ctx, struct ne_cgraph* gf, bool keep) {
  struct ne_cgraph result = *gf;

  NE_ASSERT(gf->n_nodes > 0);

  // if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
  if (keep) {
    for (int i = 0; i < gf->n_nodes; i++) {
      struct ne_tensor* node = gf->nodes[i];

      if (node->grad) {
        node->grad = ne_dup_tensor(ctx, node);
        gf->grads[i] = node->grad;
      }
    }
  }

  for (int i = gf->n_nodes - 1; i >= 0; i--) {
    struct ne_tensor* node = gf->nodes[i];

    // because we detached the grad nodes from the original graph, we can afford inplace operations
    if (node->grad) {
      ne_compute_backward(ctx, node, keep);
    }
  }

  for (int i = gf->n_nodes - 1; i >= 0; i--) {
    struct ne_tensor* node = gf->nodes[i];

    if (node->is_param) {
      NE_PRINT_DEBUG("%s: found root node %p\n", __func__, (void*)node);
      ne_build_forward_impl(&result, node->grad, true);
    }
  }

  return result;
}

//
// thread data
//
// synchronization is done via busy loops
// I tried using spin locks, but not sure how to use them correctly - the things I tried were slower than busy loops
//

#ifdef __APPLE__

// #include <os/lock.h>
//
// typedef os_unfair_lock ne_lock_t;
//
// #define ne_lock_init(x)    UNUSED(x)
// #define ne_lock_destroy(x) UNUSED(x)
// #define ne_lock_lock       os_unfair_lock_lock
// #define ne_lock_unlock     os_unfair_lock_unlock
//
// #define NE_LOCK_INITIALIZER OS_UNFAIR_LOCK_INIT

typedef int ne_lock_t;

#define ne_lock_init(x) UNUSED(x)
#define ne_lock_destroy(x) UNUSED(x)
#define ne_lock_lock(x) UNUSED(x)
#define ne_lock_unlock(x) UNUSED(x)

#define NE_LOCK_INITIALIZER 0

typedef pthread_t ne_thread_t;

#define ne_thread_create pthread_create
#define ne_thread_join pthread_join

#else

// typedef pthread_spinlock_t ne_lock_t;

// #define ne_lock_init(x) pthread_spin_init(x, PTHREAD_PROCESS_PRIVATE)
// #define ne_lock_destroy pthread_spin_destroy
// #define ne_lock_lock    pthread_spin_lock
// #define ne_lock_unlock  pthread_spin_unlock

typedef int ne_lock_t;

#define ne_lock_init(x) UNUSED(x)
#define ne_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ne_lock_lock(x) _mm_pause()
#else
#define ne_lock_lock(x) UNUSED(x)
#endif
#define ne_lock_unlock(x) UNUSED(x)

#define NE_LOCK_INITIALIZER 0

typedef pthread_t ne_thread_t;

#define ne_thread_create pthread_create
#define ne_thread_join pthread_join

#endif

struct ne_compute_state_shared {
  ne_lock_t spin;

  int n_threads;

  // synchronization primitives
  atomic_int n_ready;
  atomic_bool has_work;
  atomic_bool stop;  // stop all threads
};

struct ne_compute_state {
  ne_thread_t thrd;

  struct ne_compute_params params;
  struct ne_tensor* node;

  struct ne_compute_state_shared* shared;
};

static thread_ret_t ne_graph_compute_thread(void* data) {
  struct ne_compute_state* state = (struct ne_compute_state*)data;

  const int n_threads = state->shared->n_threads;

  while (true) {
    if (atomic_fetch_add(&state->shared->n_ready, 1) == n_threads - 1) {
      atomic_store(&state->shared->has_work, false);
    } else {
      while (atomic_load(&state->shared->has_work)) {
        if (atomic_load(&state->shared->stop)) {
          return 0;
        }
        ne_lock_lock(&state->shared->spin);
        ne_lock_unlock(&state->shared->spin);
      }
    }

    atomic_fetch_sub(&state->shared->n_ready, 1);

    // wait for work
    while (!atomic_load(&state->shared->has_work)) {
      if (atomic_load(&state->shared->stop)) {
        return 0;
      }
      ne_lock_lock(&state->shared->spin);
      ne_lock_unlock(&state->shared->spin);
    }

    // check if we should stop
    if (atomic_load(&state->shared->stop)) {
      break;
    }

    if (state->node) {
      if (state->params.ith < state->params.nth) {
        ne_compute_forward(&state->params, state->node);
      }

      state->node = NULL;
    } else {
      break;
    }
  }

  return 0;
}

void ne_graph_compute(struct ne_context* ctx, struct ne_cgraph* cgraph) {
  int n_threads = cgraph->n_threads;

  struct ne_compute_state_shared state_shared = {
      /*.spin      =*/NE_LOCK_INITIALIZER,
      /*.n_threads =*/n_threads,
      /*.n_ready   =*/0,
      /*.has_work  =*/false,
      /*.stop      =*/false,
  };
  struct ne_compute_state* workers = n_threads > 1 ? alloca(sizeof(struct ne_compute_state) * (n_threads - 1)) : NULL;
#ifndef _OPENMP
  // create thread pool
  if (n_threads > 1) {
    ne_lock_init(&state_shared.spin);

    atomic_store(&state_shared.has_work, true);

    for (int j = 0; j < n_threads - 1; j++) {
      workers[j] = (struct ne_compute_state){
          .thrd = 0,
          .params =
              {
                  .type = NE_TASK_COMPUTE,
                  .ith = j + 1,
                  .nth = n_threads,
                  .wsize = cgraph->work ? ne_nbytes(cgraph->work) : 0,
                  .wdata = cgraph->work ? cgraph->work->data : NULL,
              },
          .node = NULL,
          .shared = &state_shared,
      };

      int rc = ne_thread_create(&workers[j].thrd, NULL, ne_graph_compute_thread, &workers[j]);
      NE_ASSERT(rc == 0);
      UNUSED(rc);
    }
  }
#else
  n_threads = jblas_set_threads(n_threads);  // prevent from using two sockets
  omp_set_num_threads(n_threads);
#endif
  // initialize tasks + work buffer
  {
    size_t work_size = 0;

    // thread scheduling for the different operations
    for (int i = 0; i < cgraph->n_nodes; i++) {
      struct ne_tensor* node = cgraph->nodes[i];

      switch (node->op) {
        case NE_OP_CPY: {
          node->n_tasks = n_threads;  // node->ne[0] == 1 ? n_threads : 1;
          size_t cur = 0;
          if (ne_is_quantized(node->type)) {
            cur = NE_TYPE_SIZE[NE_TYPE_F32] * node->ne[0] * n_threads;
          }
          work_size = MAX(work_size, cur);
        } break;
        case NE_OP_DUP: {
          node->n_tasks = n_threads;

          size_t cur = 0;
          if (ne_is_quantized(node->type)) {
            cur = NE_TYPE_SIZE[NE_TYPE_F32] * node->ne[0] * n_threads;
          }

          work_size = MAX(work_size, cur);
        } break;
        case NE_OP_ADD:
        case NE_OP_ADD1: {
          if (node->src0->ne[1] > 4) {
            node->n_tasks = n_threads;
          } else {
            node->n_tasks = 1;
          }

          size_t cur = 0;

          if (ne_is_quantized(node->src0->type)) {
            cur = NE_TYPE_SIZE[NE_TYPE_F32] * node->src0->ne[0] * n_threads;
          }

          work_size = MAX(work_size, cur);
        } break;
        case NE_OP_ACC: {
          node->n_tasks = n_threads;

          size_t cur = 0;

          if (ne_is_quantized(node->src0->type)) {
            cur = NE_TYPE_SIZE[NE_TYPE_F32] * node->src1->ne[0] * n_threads;
          }

          work_size = MAX(work_size, cur);
        } break;
        case NE_OP_SUB:
        case NE_OP_DIV:
        case NE_OP_SQR:
        case NE_OP_SQRT:
        case NE_OP_LOG:
        case NE_OP_SUM:
        case NE_OP_SUM_ROWS:
        case NE_OP_MEAN:
        case NE_OP_ABS:
        case NE_OP_REPEAT:
        case NE_OP_SGN:
        case NE_OP_NEG:
        case NE_OP_STEP:
        case NE_OP_MUL:
        case NE_OP_RMS_NORM:
        case NE_OP_RELU: {
          if (node->src0->ne[1] > 4) {
            node->n_tasks = n_threads;
          } else {
            node->n_tasks = 1;
          }
        } break;
        case NE_OP_GELU:
        case NE_OP_SILU:
        case NE_OP_SILU_BACK:
        case NE_OP_NORM:
        case NE_OP_RMS_NORM_BACK: {
          node->n_tasks = n_threads;
        } break;
        case NE_OP_MUL_MAT_BIAS:
        case NE_OP_MUL_MAT: {
          node->n_tasks = n_threads;

          // TODO: use different scheduling for different matrix sizes
          // const int nr0 = ne_nrows(node->src0);
          // const int nr1 = ne_nrows(node->src1);

          // node->n_tasks = MIN(n_threads, MAX(1, nr0/128));
          // printf("nr0 = %8d, nr1 = %8d, nr0*nr1 = %8d, n_tasks = %d\n", nr0, nr1, nr0*nr1, node->n_tasks);

          size_t cur = 0;
          if (node->src0->type == NE_TYPE_JBLAS) {
            cur = jblas_f32f32_get_workspace_size(node->src1->ne[1], node->src0->ne[1], node->src1->ne[0],
                                                  node->src0->data);
            node->n_tasks = 1;
          } else if (node->src0->type == NE_TYPE_F16 && node->src1->type == NE_TYPE_F32) {
            cur = NE_TYPE_SIZE[NE_TYPE_F16] * ne_nelements(node->src1);
          } else if (node->src0->type == NE_TYPE_F32 && node->src1->type == NE_TYPE_F32) {
            cur = 0;
          } else if (ne_is_quantized(node->src0->type) && node->src1->type == NE_TYPE_F32) {
            {
              const enum ne_type type_q = quantize_fns[node->src0->type].vec_dot_type;
              cur = NE_TYPE_SIZE[type_q] * ne_nelements(node->src1) / NE_BLCK_SIZE[type_q];
            }
          } else {
            NE_ASSERT(false);
          }

          work_size = MAX(work_size, cur);
        } break;
        case NE_OP_MUL_FFN_SILU:
        case NE_OP_MUL_FFN_GELU:
        case NE_OP_MUL_FFN_ADD_GELU: {
          size_t cur = 0;
          cur = jblas_fusion_FFN_f32f32_get_workspace_size(node->src0->ne[1], node->src0->ne[0], node->src1->ne[1],
                                                           node->opt[0]->ne[1], node->src1->data, node->opt[0]->data);
          work_size = MAX(work_size, cur);
          node->n_tasks = 1;
        } break;
        case NE_OP_MUL_QKV: {
          size_t cur = 0;
          cur = jblas_fusion_QKV_f32f32_get_workspace_size(node->src0->ne[1], node->src1->ne[1], node->src1->ne[0],
                                                           node->src1->data);
          work_size = MAX(work_size, cur);
          node->n_tasks = 1;
        } break;
        case NE_OP_SCALE: {
          node->n_tasks = 1;
        } break;
        case NE_OP_SET:
        case NE_OP_CONT:
        case NE_OP_RESHAPE:
        case NE_OP_VIEW:
        case NE_OP_PERMUTE:
        case NE_OP_TRANSPOSE:
        case NE_OP_GET_ROWS:
        case NE_OP_GET_ROWS_BACK:
        case NE_OP_DIAG:
        case NE_OP_DIAG_MASK_ZERO: {
          node->n_tasks = 1;
        } break;
        case NE_OP_DIAG_MASK_INF:
        case NE_OP_ROPE:
          if (node->src0->ne[1] > 4) {
            node->n_tasks = n_threads;
          } else {
            node->n_tasks = 1;
          }
          break;
        case NE_OP_SOFT_MAX: {
          size_t rows = ne_nrows(node->src0);
          node->n_tasks = rows > 1 ? n_threads : 1;
        } break;
        case NE_OP_ROPE_BACK: {
          node->n_tasks = n_threads;
        } break;
        case NE_OP_ALIBI: {
          node->n_tasks = 1;  // TODO
        } break;
        case NE_OP_CLAMP: {
          node->n_tasks = 1;  // TODO
        } break;
        case NE_OP_CONV_1D_1S:
        case NE_OP_CONV_1D_2S: {
          node->n_tasks = n_threads;

          NE_ASSERT(node->src0->ne[3] == 1);
          NE_ASSERT(node->src1->ne[2] == 1);
          NE_ASSERT(node->src1->ne[3] == 1);

          size_t cur = 0;
          const int nk = node->src0->ne[0];

          if (node->src0->type == NE_TYPE_F16 && node->src1->type == NE_TYPE_F32) {
            cur = sizeof(ne_fp16_t) * (nk * ne_up32(node->src0->ne[1]) * node->src0->ne[2] +
                                       (2 * (nk / 2) + node->src1->ne[0]) * node->src1->ne[1]);
          } else if (node->src0->type == NE_TYPE_F32 && node->src1->type == NE_TYPE_F32) {
            cur = sizeof(float) * (nk * ne_up32(node->src0->ne[1]) * node->src0->ne[2] +
                                   (2 * (nk / 2) + node->src1->ne[0]) * node->src1->ne[1]);
          } else {
            NE_ASSERT(false);
          }

          work_size = MAX(work_size, cur);
        } break;
        case NE_OP_FLASH_ATTN: {
          node->n_tasks = 1;
          work_size = 0LL;
        } break;
        case NE_OP_FLASH_FF: {
          node->n_tasks = n_threads;

          size_t cur = 0;

          if (node->src1->type == NE_TYPE_F32) {
            cur = sizeof(float) * node->src1->ne[1] * node->n_tasks;   // TODO: this can become (n_tasks-1)
            cur += sizeof(float) * node->src1->ne[1] * node->n_tasks;  // this is overestimated by x2
          }

          if (node->src1->type == NE_TYPE_F16) {
            cur = sizeof(float) * node->src1->ne[1] * node->n_tasks;   // TODO: this can become (n_tasks-1)
            cur += sizeof(float) * node->src1->ne[1] * node->n_tasks;  // this is overestimated by x2
          }

          work_size = MAX(work_size, cur);
        } break;
        case NE_OP_MAP_UNARY:
        case NE_OP_MAP_BINARY: {
          node->n_tasks = 1;
        } break;
        case NE_OP_NONE: {
          node->n_tasks = 1;
        } break;
        case NE_OP_COUNT: {
          NE_ASSERT(false);
        } break;
      }
    }

    if (cgraph->work != NULL && work_size > cgraph->work_size) {
      NE_ASSERT(false);  // TODO: better handling
    }

    if (work_size > 0 && cgraph->work == NULL) {
      cgraph->work_size = work_size + CACHE_LINE_SIZE * (n_threads - 1);

      NE_PRINT_DEBUG("%s: allocating work buffer for graph (%zu bytes)\n", __func__, cgraph->work_size);
      cgraph->work = ne_new_tensor_1d(ctx, NE_TYPE_I8, cgraph->work_size, NE_SIZE_CALC);
    }
  }

  const int64_t perf_start_cycles = ne_perf_cycles();
  const int64_t perf_start_time_us = ne_perf_time_us();

  for (int i = 0; i < cgraph->n_nodes; i++) {
    NE_PRINT_DEBUG_5("%s: %d/%d\n", __func__, i, cgraph->n_nodes);

    struct ne_tensor* node = cgraph->nodes[i];

    // TODO: this could be used to avoid unnecessary computations, but it needs to be improved
    // if (node->grad == NULL && node->perf_runs > 0) {
    //    continue;
    //}

    const int64_t perf_node_start_cycles = ne_perf_cycles();
    const int64_t perf_node_start_time_us = ne_perf_time_us();
#if NE_DEBUG
    jblas_timer(true);
#endif
#ifndef _OPENMP
    // INIT
    struct ne_compute_params params = {
        /*.type  =*/NE_TASK_INIT,
        /*.ith   =*/0,
        /*.nth   =*/node->n_tasks,
        /*.wsize =*/cgraph->work ? ne_nbytes(cgraph->work) : 0,
        /*.wdata =*/cgraph->work ? cgraph->work->data : NULL,
    };

    ne_compute_forward(&params, node);

    // COMPUTE
    if (node->n_tasks > 1) {
      if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
        atomic_store(&state_shared.has_work, false);
      }

      while (atomic_load(&state_shared.has_work)) {
        ne_lock_lock(&state_shared.spin);
        ne_lock_unlock(&state_shared.spin);
      }

      // launch thread pool
      for (int j = 0; j < n_threads - 1; j++) {
        workers[j].params = (struct ne_compute_params){
            .type = NE_TASK_COMPUTE,
            .ith = j + 1,
            .nth = node->n_tasks,
            .wsize = cgraph->work ? ne_nbytes(cgraph->work) : 0,
            .wdata = cgraph->work ? cgraph->work->data : NULL,
        };
        workers[j].node = node;
      }

      atomic_fetch_sub(&state_shared.n_ready, 1);

      while (atomic_load(&state_shared.n_ready) > 0) {
        ne_lock_lock(&state_shared.spin);
        ne_lock_unlock(&state_shared.spin);
      }

      atomic_store(&state_shared.has_work, true);
    }

    params.type = NE_TASK_COMPUTE;
    ne_compute_forward(&params, node);

    // wait for thread pool
    if (node->n_tasks > 1) {
      if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
        atomic_store(&state_shared.has_work, false);
      }

      while (atomic_load(&state_shared.has_work)) {
        ne_lock_lock(&state_shared.spin);
        ne_lock_unlock(&state_shared.spin);
      }

      atomic_fetch_sub(&state_shared.n_ready, 1);

      while (atomic_load(&state_shared.n_ready) != 0) {
        ne_lock_lock(&state_shared.spin);
        ne_lock_unlock(&state_shared.spin);
      }
    }
    // FINALIZE
    if (node->n_tasks > 1) {
      if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
        atomic_store(&state_shared.has_work, false);
      }

      while (atomic_load(&state_shared.has_work)) {
        ne_lock_lock(&state_shared.spin);
        ne_lock_unlock(&state_shared.spin);
      }

      // launch thread pool
      for (int j = 0; j < n_threads - 1; j++) {
        workers[j].params = (struct ne_compute_params){
            .type = NE_TASK_FINALIZE,
            .ith = j + 1,
            .nth = node->n_tasks,
            .wsize = cgraph->work ? ne_nbytes(cgraph->work) : 0,
            .wdata = cgraph->work ? cgraph->work->data : NULL,
        };
        workers[j].node = node;
      }

      atomic_fetch_sub(&state_shared.n_ready, 1);

      while (atomic_load(&state_shared.n_ready) > 0) {
        ne_lock_lock(&state_shared.spin);
        ne_lock_unlock(&state_shared.spin);
      }

      atomic_store(&state_shared.has_work, true);
    }

    params.type = NE_TASK_FINALIZE;
    ne_compute_forward(&params, node);

    // wait for thread pool
    if (node->n_tasks > 1) {
      if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
        atomic_store(&state_shared.has_work, false);
      }

      while (atomic_load(&state_shared.has_work)) {
        ne_lock_lock(&state_shared.spin);
        ne_lock_unlock(&state_shared.spin);
      }

      atomic_fetch_sub(&state_shared.n_ready, 1);

      while (atomic_load(&state_shared.n_ready) != 0) {
        ne_lock_lock(&state_shared.spin);
        ne_lock_unlock(&state_shared.spin);
      }
    }
#else
    // INIT
    struct ne_compute_params params = {
        /*.type  =*/NE_TASK_INIT,
        /*.ith   =*/0,
        /*.nth   =*/node->n_tasks,
        /*.wsize =*/cgraph->work ? ne_nbytes(cgraph->work) : 0,
        /*.wdata =*/cgraph->work ? cgraph->work->data : NULL,
    };
    ne_compute_forward(&params, node);
    if (node->n_tasks == 1) {
      params.type = NE_TASK_COMPUTE;
      ne_compute_forward(&params, node);
      params.type = NE_TASK_FINALIZE;
      ne_compute_forward(&params, node);

    } else {
#pragma omp parallel
      {
        struct ne_compute_params params = {
            /*.type  =*/NE_TASK_COMPUTE,
            /*.ith   =*/omp_get_thread_num(),
            /*.nth   =*/node->n_tasks,
            /*.wsize =*/cgraph->work ? ne_nbytes(cgraph->work) : 0,
            /*.wdata =*/cgraph->work ? cgraph->work->data : NULL,
        };
        if (params.ith < node->n_tasks) {
          ne_compute_forward(&params, node);
        }
#pragma omp barrier
        params.type = NE_TASK_FINALIZE;
        if (params.ith < node->n_tasks) {
          ne_compute_forward(&params, node);
        }
      }
    }

#endif
#if NE_DEBUG
    printf("Node %d ", node->op);
    jblas_timer(false);
#endif
    // performance stats (node)
    {
      int64_t perf_cycles_cur = ne_perf_cycles() - perf_node_start_cycles;
      int64_t perf_time_us_cur = ne_perf_time_us() - perf_node_start_time_us;

      node->perf_runs++;
      node->perf_cycles += perf_cycles_cur;
      node->perf_time_us += perf_time_us_cur;
    }
  }

  // join thread pool
#ifndef _OPENMP
  if (n_threads > 1) {
    atomic_store(&state_shared.stop, true);
    atomic_store(&state_shared.has_work, true);

    for (int j = 0; j < n_threads - 1; j++) {
      int rc = ne_thread_join(workers[j].thrd, NULL);
      NE_ASSERT(rc == 0);
      UNUSED(rc);
    }

    ne_lock_destroy(&state_shared.spin);
  }
#endif

  // performance stats (graph)
  {
    int64_t perf_cycles_cur = ne_perf_cycles() - perf_start_cycles;
    int64_t perf_time_us_cur = ne_perf_time_us() - perf_start_time_us;

    cgraph->perf_runs++;
    cgraph->perf_cycles += perf_cycles_cur;
    cgraph->perf_time_us += perf_time_us_cur;

    NE_PRINT_DEBUG("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n", __func__, cgraph->perf_runs,
                   (double)perf_cycles_cur / (double)ne_cycles_per_ms(),
                   (double)cgraph->perf_cycles / (double)ne_cycles_per_ms() / (double)cgraph->perf_runs,
                   (double)perf_time_us_cur / 1000.0, (double)cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
  }
}

void ne_graph_profiling(const struct ne_cgraph* cgraph) {
  int64_t perf_total_per_op_us[NE_OP_COUNT] = {0};

  NE_PRINT("=== GRAPH Profiling ===\n");

  int64_t ip_duration = 0;
  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ne_tensor* node = cgraph->nodes[i];
    if (node->op == NE_OP_MUL_MAT && node->ne[1] == node->ne[2]) {
      ip_duration += node->perf_time_us;
    } else {
      perf_total_per_op_us[node->op] += node->perf_time_us;
    }
  }

  for (int i = 0; i < NE_OP_COUNT; i++) {
    if (perf_total_per_op_us[i] == 0) {
      continue;
    }
    NE_PRINT("perf_total_per_op_us[%16s] = %7.3f ms\n", NE_OP_LABEL[i], (double)perf_total_per_op_us[i] / 1000.0);
  }
  NE_PRINT("perf_total_per_op_us[%16s] = %7.3f ms\n", "INNER PRODUCT", (double)ip_duration / 1000.0);
  NE_PRINT("========================================\n");
}

void ne_graph_reset(struct ne_cgraph* cgraph) {
  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ne_tensor* grad = cgraph->grads[i];

    if (grad) {
      ne_set_zero(grad);
    }
  }
}

void ne_graph_print(const struct ne_cgraph* cgraph) {
  int64_t perf_total_per_op_us[NE_OP_COUNT] = {0};

  NE_PRINT("=== GRAPH ===\n");

  NE_PRINT_DEBUG("n_threads       = %d\n", cgraph->n_threads);
  NE_PRINT_DEBUG("total work size = %zu bytes\n", cgraph->work_size);

  NE_PRINT("n_nodes = %d\n", cgraph->n_nodes);
  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ne_tensor* node = cgraph->nodes[i];

    perf_total_per_op_us[node->op] += MAX(1, node->perf_time_us);

    NE_PRINT(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64
             "] %16s %s (%3d) cpu = %7.3f / %7.3f ms, wall = %7.3f / %7.3f ms\n",
             i, node->ne[0], node->ne[1], node->ne[2], NE_OP_LABEL[node->op],
             node->is_param ? "x" : node->grad ? "g" : " ", node->perf_runs,
             (double)node->perf_cycles / (double)ne_cycles_per_ms(),
             (double)node->perf_cycles / (double)ne_cycles_per_ms() / (double)node->perf_runs,
             (double)node->perf_time_us / 1000.0, (double)node->perf_time_us / 1000.0 / node->perf_runs);
  }

  NE_PRINT("n_leafs = %d\n", cgraph->n_leafs);
  for (int i = 0; i < cgraph->n_leafs; i++) {
    struct ne_tensor* node = cgraph->leafs[i];

    NE_PRINT(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s\n", i, node->ne[0], node->ne[1], NE_OP_LABEL[node->op]);
  }

  for (int i = 0; i < NE_OP_COUNT; i++) {
    if (perf_total_per_op_us[i] == 0) {
      continue;
    }

    NE_PRINT("perf_total_per_op_us[%16s] = %7.3f ms\n", NE_OP_LABEL[i], (double)perf_total_per_op_us[i] / 1000.0);
  }

  NE_PRINT("========================================\n");
}

// check if node is part of the graph
static bool ne_graph_find(const struct ne_cgraph* cgraph, const struct ne_tensor* node) {
  if (cgraph == NULL) {
    return true;
  }

  for (int i = 0; i < cgraph->n_nodes; i++) {
    if (cgraph->nodes[i] == node) {
      return true;
    }
  }

  return false;
}

static struct ne_tensor* ne_graph_get_parent(const struct ne_cgraph* cgraph, const struct ne_tensor* node) {
  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ne_tensor* parent = cgraph->nodes[i];

    if (parent->grad == node) {
      return parent;
    }
  }

  return NULL;
}

void ne_graph_dump_dot(const struct ne_cgraph* gb, const struct ne_cgraph* gf, const char* filename) {
  char color[16];

  FILE* fp = fopen(filename, "w");
  NE_ASSERT(fp);

  fprintf(fp, "digraph G {\n");
  fprintf(fp, "  newrank = true;\n");
  fprintf(fp, "  rankdir = LR;\n");

  for (int i = 0; i < gb->n_nodes; i++) {
    struct ne_tensor* node = gb->nodes[i];

    if (ne_graph_get_parent(gb, node) != NULL) {
      continue;
    }

    if (node->is_param) {
      snprintf(color, sizeof(color), "yellow");
    } else if (node->grad) {
      if (ne_graph_find(gf, node)) {
        snprintf(color, sizeof(color), "green");
      } else {
        snprintf(color, sizeof(color), "lightblue");
      }
    } else {
      snprintf(color, sizeof(color), "white");
    }

    fprintf(fp,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = record; "
            "label=\"",
            (void*)node, color);

    if (strlen(node->name) > 0) {
      fprintf(fp, "%s |", node->name);
    }

    if (node->n_dims == 2) {
      fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], NE_OP_SYMBOL[node->op]);
    } else {
      fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], node->ne[2],
              NE_OP_SYMBOL[node->op]);
    }

    if (node->grad) {
      fprintf(fp, " | <g>%s\"; ]\n", NE_OP_SYMBOL[node->grad->op]);
    } else {
      fprintf(fp, "\"; ]\n");
    }
  }

  for (int i = 0; i < gb->n_leafs; i++) {
    struct ne_tensor* node = gb->leafs[i];

    snprintf(color, sizeof(color), "pink");

    fprintf(fp,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = record; "
            "label=\"<x>",
            (void*)node, color);

    if (strlen(node->name) > 0) {
      fprintf(fp, "%s | ", node->name);
    }
    if (ne_nelements(node) == 1) {
      if (node->type == NE_TYPE_I8 || node->type == NE_TYPE_I16 || node->type == NE_TYPE_I32) {
        fprintf(fp, "%d", ne_get_i32_1d(node, 0));
      } else {
        fprintf(fp, "%.1e", (double)ne_get_f32_1d(node, 0));
      }
    } else {
      fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, node->ne[0], node->ne[1]);
    }
    fprintf(fp, "\"; ]\n");
  }

  for (int i = 0; i < gb->n_nodes; i++) {
    struct ne_tensor* node = gb->nodes[i];

    struct ne_tensor* parent = ne_graph_get_parent(gb, node);

    if (node->src0) {
      struct ne_tensor* parent0 = ne_graph_get_parent(gb, node->src0);

      fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"x\"; ]\n",
              parent0 ? (void*)parent0 : (void*)node->src0, parent0 ? "g" : "x", parent ? (void*)parent : (void*)node,
              parent ? "g" : "x", parent ? "empty" : "vee", parent ? "dashed" : "solid");
    }

    if (node->src1) {
      struct ne_tensor* parent1 = ne_graph_get_parent(gb, node->src1);

      fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"y\"; ]\n",
              parent1 ? (void*)parent1 : (void*)node->src1, parent1 ? "g" : "x", parent ? (void*)parent : (void*)node,
              parent ? "g" : "x", parent ? "empty" : "vee", parent ? "dashed" : "solid");
    }
  }

  for (int i = 0; i < gb->n_leafs; i++) {
    struct ne_tensor* node = gb->leafs[i];

    if (node->src0) {
      fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"x\"; ]\n", (void*)node->src0, "x", (void*)node, "x");
    }

    if (node->src1) {
      fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"y\"; ]\n", (void*)node->src1, "x", (void*)node, "x");
    }
  }

  fprintf(fp, "}\n");

  fclose(fp);

  NE_PRINT("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

static void ne_opt_set_params(int np, struct ne_tensor* const ps[], const float* x) {
  int i = 0;
  for (int p = 0; p < np; ++p) {
    const int64_t ne = ne_nelements(ps[p]);
    // TODO: add function to set tensor from array
    for (int64_t j = 0; j < ne; ++j) {
      ne_set_f32_1d(ps[p], j, x[i++]);
    }
  }
}

static void ne_opt_get_params(int np, struct ne_tensor* const ps[], float* x) {
  int i = 0;
  for (int p = 0; p < np; ++p) {
    const int64_t ne = ne_nelements(ps[p]);
    // TODO: add function to get all elements at once
    for (int64_t j = 0; j < ne; ++j) {
      x[i++] = ne_get_f32_1d(ps[p], j);
    }
  }
}

static void ne_opt_get_grad(int np, struct ne_tensor* const ps[], float* g) {
  int i = 0;
  for (int p = 0; p < np; ++p) {
    const int64_t ne = ne_nelements(ps[p]);
    // TODO: add function to get all elements at once
    for (int64_t j = 0; j < ne; ++j) {
      g[i++] = ne_get_f32_1d(ps[p]->grad, j);
    }
  }
}

//
// ADAM
//
//   ref: https://arxiv.org/pdf/1412.6980.pdf
//

static enum ne_opt_result ne_opt_adam(struct ne_context* ctx, struct ne_opt_params params, struct ne_tensor* f,
                                      struct ne_cgraph* gf, struct ne_cgraph* gb) {
  NE_ASSERT(ne_is_scalar(f));

  gf->n_threads = params.n_threads;
  gb->n_threads = params.n_threads;

  // these will store the parameters we want to optimize
  struct ne_tensor* ps[NE_MAX_PARAMS];

  int np = 0;
  int nx = 0;
  for (int i = 0; i < gf->n_nodes; ++i) {
    if (gf->nodes[i]->is_param) {
      NE_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

      NE_ASSERT(np < NE_MAX_PARAMS);

      ps[np++] = gf->nodes[i];
      nx += ne_nelements(gf->nodes[i]);
    }
  }

  // constants
  const float alpha = params.adam.alpha;
  const float beta1 = params.adam.beta1;
  const float beta2 = params.adam.beta2;
  const float eps = params.adam.eps;

  float* x = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;   // view of the parameters
  float* g1 = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;  // gradient
  float* g2 = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;  // gradient squared
  float* m = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;   // first moment
  float* v = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;   // second moment
  float* mh = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;  // first moment hat
  float* vh = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;  // second moment hat

  float* pf = params.past > 0 ? ne_new_tensor_1d(ctx, NE_TYPE_F32, params.past, NE_SIZE_CALC)->data
                              : NULL;  // past function values

  // initialize
  ne_vec_set_f32(nx, m, 0.0f);
  ne_vec_set_f32(nx, v, 0.0f);

  // update view
  ne_opt_get_params(np, ps, x);

  // compute the function value
  ne_graph_reset(gf);
  ne_set_f32(f->grad, 1.0f);
  ne_graph_compute(ctx, gb);

  float fx_prev = ne_get_f32_1d(f, 0);
  if (pf) {
    pf[0] = fx_prev;
  }

  int n_no_improvement = 0;
  float fx_best = fx_prev;

  // run the optimizer
  for (int t = 0; t < params.adam.n_iter; ++t) {
    NE_PRINT_DEBUG("=== iter %d ===\n", t);

    NE_PRINT_DEBUG("f      = %10.6f\n", ne_get_f32_1d(f, 0));
    NE_PRINT_DEBUG_5("df/dx0 = %10.6f\n", ne_get_f32_1d(ps[0]->grad, 0));
    NE_PRINT_DEBUG_5("df/dx1 = %10.6f\n", ne_get_f32_1d(ps[1]->grad, 0));

    for (int i = 0; i < np; ++i) {
      NE_PRINT_DEBUG("param %d: %10.6f, g = %10.6f\n", i, ne_get_f32_1d(ps[i], 0), ne_get_f32_1d(ps[i]->grad, 0));
    }

    const int64_t t_start_wall = ne_time_us();
    const int64_t t_start_cpu = ne_cycles();
    UNUSED(t_start_wall);
    UNUSED(t_start_cpu);

    {
      // update the gradient
      ne_opt_get_grad(np, ps, g1);

      // m_t = beta1*m_t-1 + (1 - beta1)*g_t
      ne_vec_scale_f32(nx, m, beta1);
      ne_vec_mad_f32(nx, m, g1, 1.0f - beta1);

      // g2 = g1^2
      ne_vec_sqr_f32(nx, g2, g1);

      // v_t = beta2*v_t-1 + (1 - beta2)*g_t^2
      ne_vec_scale_f32(nx, v, beta2);
      ne_vec_mad_f32(nx, v, g2, 1.0f - beta2);

      // m^hat = m_t / (1 - beta1^t)
      // v^hat = v_t / (1 - beta2^t)
      // x_t = x_t-1 - alpha*m^hat/(sqrt(v^hat) + eps)
      ne_vec_cpy_f32(nx, mh, m);
      ne_vec_cpy_f32(nx, vh, v);

      ne_vec_scale_f32(nx, mh, alpha / (1.0f - powf(beta1, t + 1)));
      ne_vec_scale_f32(nx, vh, 1.0f / (1.0f - powf(beta2, t + 1)));

      ne_vec_sqrt_f32(nx, vh, vh);
      ne_vec_acc1_f32(nx, vh, eps);

      ne_vec_div_f32(nx, mh, mh, vh);
      ne_vec_sub_f32(nx, x, x, mh);

      // update the parameters
      ne_opt_set_params(np, ps, x);
    }

    ne_graph_reset(gf);
    ne_set_f32(f->grad, 1.0f);
    ne_graph_compute(ctx, gb);

    const float fx = ne_get_f32_1d(f, 0);

    // check convergence
    if (fabsf(fx - fx_prev) / fx < params.adam.eps_f) {
      NE_PRINT_DEBUG("converged\n");

      return NE_OPT_OK;
    }

    // delta-based convergence test
    if (pf != NULL) {
      // need at least params.past iterations to start checking for convergence
      if (params.past <= t) {
        const float rate = (pf[t % params.past] - fx) / fx;

        if (fabsf(rate) < params.delta) {
          return NE_OPT_OK;
        }
      }

      pf[t % params.past] = fx;
    }

    // check for improvement
    if (params.max_no_improvement > 0) {
      if (fx_best > fx) {
        fx_best = fx;
        n_no_improvement = 0;
      } else {
        ++n_no_improvement;

        if (n_no_improvement >= params.max_no_improvement) {
          return NE_OPT_OK;
        }
      }
    }

    fx_prev = fx;

    {
      const int64_t t_end_cpu = ne_cycles();
      NE_PRINT_DEBUG("time iter:      %5.3f s\n", ((float)(t_end_cpu - t_start_cpu)) / CLOCKS_PER_SEC);
      UNUSED(t_end_cpu);

      const int64_t t_end_wall = ne_time_us();
      NE_PRINT_DEBUG("wall time iter: %5.3f s\n", (t_end_wall - t_start_wall) / 1e6);
      UNUSED(t_end_wall);
    }
  }

  return NE_OPT_DID_NOT_CONVERGE;
}

//
// L-BFGS
//
// the L-BFGS implementation below is based on the following implementation:
//
//   https://github.com/chokkan/liblbfgs
//

struct ne_lbfgs_iteration_data {
  float alpha;
  float ys;
  float* s;
  float* y;
};

static enum ne_opt_result linesearch_backtracking(struct ne_context* ctx, const struct ne_opt_params* params, int nx,
                                                  float* x, float* fx, float* g, float* d, float* step, const float* xp,
                                                  struct ne_tensor* f, struct ne_cgraph* gf, struct ne_cgraph* gb,
                                                  const int np, struct ne_tensor* ps[]) {
  int count = 0;

  float width = 0.0f;
  float dg = 0.0f;
  float finit = 0.0f;
  float dginit = 0.0f;
  float dgtest = 0.0f;

  const float dec = 0.5f;
  const float inc = 2.1f;

  if (*step <= 0.f) {
    return NE_LINESEARCH_INVALID_PARAMETERS;
  }

  // compute the initial gradient in the search direction
  ne_vec_dot_f32(nx, &dginit, g, d);

  // make sure that d points to a descent direction
  if (0 < dginit) {
    return NE_LINESEARCH_FAIL;
  }

  // initialize local variables
  finit = *fx;
  dgtest = params->lbfgs.ftol * dginit;

  while (true) {
    ne_vec_cpy_f32(nx, x, xp);
    ne_vec_mad_f32(nx, x, d, *step);

    // evaluate the function and gradient values
    {
      ne_opt_set_params(np, ps, x);

      ne_graph_reset(gf);
      ne_set_f32(f->grad, 1.0f);
      ne_graph_compute(ctx, gb);

      ne_opt_get_grad(np, ps, g);

      *fx = ne_get_f32_1d(f, 0);
    }

    ++count;

    if (*fx > finit + (*step) * dgtest) {
      width = dec;
    } else {
      // Armijo condition is satisfied
      if (params->lbfgs.linesearch == NE_LINESEARCH_BACKTRACKING_ARMIJO) {
        return count;
      }

      ne_vec_dot_f32(nx, &dg, g, d);

      // check the Wolfe condition
      if (dg < params->lbfgs.wolfe * dginit) {
        width = inc;
      } else {
        if (params->lbfgs.linesearch == NE_LINESEARCH_BACKTRACKING_WOLFE) {
          // regular Wolfe conditions
          return count;
        }

        if (dg > -params->lbfgs.wolfe * dginit) {
          width = dec;
        } else {
          // strong Wolfe condition (NE_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
          return count;
        }
        return count;
      }
    }

    if (*step < params->lbfgs.min_step) {
      return NE_LINESEARCH_MINIMUM_STEP;
    }
    if (*step > params->lbfgs.max_step) {
      return NE_LINESEARCH_MAXIMUM_STEP;
    }
    if (params->lbfgs.max_linesearch <= count) {
      return NE_LINESEARCH_MAXIMUM_ITERATIONS;
    }

    (*step) *= width;
  }

  return NE_LINESEARCH_FAIL;
}

static enum ne_opt_result ne_opt_lbfgs(struct ne_context* ctx, struct ne_opt_params params, struct ne_tensor* f,
                                       struct ne_cgraph* gf, struct ne_cgraph* gb) {
  if (params.lbfgs.linesearch == NE_LINESEARCH_BACKTRACKING_WOLFE ||
      params.lbfgs.linesearch == NE_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
    if (params.lbfgs.wolfe <= params.lbfgs.ftol || 1.f <= params.lbfgs.wolfe) {
      return NE_OPT_INVALID_WOLFE;
    }
  }

  gf->n_threads = params.n_threads;
  gb->n_threads = params.n_threads;

  const int m = params.lbfgs.m;

  // these will store the parameters we want to optimize
  struct ne_tensor* ps[NE_MAX_PARAMS];

  int np = 0;
  int nx = 0;
  for (int i = 0; i < gf->n_nodes; ++i) {
    if (gf->nodes[i]->is_param) {
      NE_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

      NE_ASSERT(np < NE_MAX_PARAMS);

      ps[np++] = gf->nodes[i];
      nx += ne_nelements(gf->nodes[i]);
    }
  }

  float* x = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;   // current parameters
  float* xp = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;  // previous parameters
  float* g = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;   // current gradient
  float* gp = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;  // previous gradient
  float* d = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;   // search direction

  float* pf = params.past > 0 ? ne_new_tensor_1d(ctx, NE_TYPE_F32, params.past, NE_SIZE_CALC)->data
                              : NULL;  // past function values

  float fx = 0.0f;     // cost function value
  float xnorm = 0.0f;  // ||x||
  float gnorm = 0.0f;  // ||g||
  float step = 0.0f;

  // initialize x from the graph nodes
  ne_opt_get_params(np, ps, x);

  // the L-BFGS memory
  struct ne_lbfgs_iteration_data* lm = alloca(sizeof(struct ne_lbfgs_iteration_data) * m);

  for (int i = 0; i < m; ++i) {
    lm[i].alpha = 0.0f;
    lm[i].ys = 0.0f;
    lm[i].s = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;
    lm[i].y = ne_new_tensor_1d(ctx, NE_TYPE_F32, nx, NE_SIZE_CALC)->data;
  }

  // evaluate the function value and its gradient
  {
    ne_opt_set_params(np, ps, x);

    ne_graph_reset(gf);
    ne_set_f32(f->grad, 1.0f);
    ne_graph_compute(ctx, gb);

    ne_opt_get_grad(np, ps, g);

    fx = ne_get_f32_1d(f, 0);
  }

  if (pf) {
    pf[0] = fx;
  }

  float fx_best = fx;

  // search direction = -gradient
  ne_vec_neg_f32(nx, d, g);

  // ||x||, ||g||
  ne_vec_norm_f32(nx, &xnorm, x);
  ne_vec_norm_f32(nx, &gnorm, g);

  if (xnorm < 1.0f) {
    xnorm = 1.0f;
  }

  // already optimized
  if (gnorm / xnorm <= params.lbfgs.eps) {
    return NE_OPT_OK;
  }

  // initial step
  ne_vec_norm_inv_f32(nx, &step, d);

  int j = 0;
  int k = 1;
  int ls = 0;
  int end = 0;
  int bound = 0;
  int n_no_improvement = 0;

  float ys = 0.0f;
  float yy = 0.0f;
  float beta = 0.0f;

  while (true) {
    // store the current position and gradient vectors
    ne_vec_cpy_f32(nx, xp, x);
    ne_vec_cpy_f32(nx, gp, g);

    ls = linesearch_backtracking(ctx, &params, nx, x, &fx, g, d, &step, xp, f, gf, gb, np, ps);

    if (ls < 0) {
      // linesearch failed - go back to the previous point and return
      ne_vec_cpy_f32(nx, x, xp);
      ne_vec_cpy_f32(nx, g, gp);

      return ls;
    }

    ne_vec_norm_f32(nx, &xnorm, x);
    ne_vec_norm_f32(nx, &gnorm, g);

    NE_PRINT_DEBUG("f = %10.6f\n", ne_get_f32_1d(f, 0));

    if (xnorm < 1.0f) {
      xnorm = 1.0f;
    }
    if (gnorm / xnorm <= params.lbfgs.eps) {
      // converged
      return NE_OPT_OK;
    }

    // delta-based convergence test
    if (pf != NULL) {
      // need at least params.past iterations to start checking for convergence
      if (params.past <= k) {
        const float rate = (pf[k % params.past] - fx) / fx;

        if (fabsf(rate) < params.delta) {
          return NE_OPT_OK;
        }
      }

      pf[k % params.past] = fx;
    }

    // check for improvement
    if (params.max_no_improvement > 0) {
      if (fx < fx_best) {
        fx_best = fx;
        n_no_improvement = 0;
      } else {
        n_no_improvement++;

        if (n_no_improvement >= params.max_no_improvement) {
          return NE_OPT_OK;
        }
      }
    }

    if (params.lbfgs.n_iter != 0 && params.lbfgs.n_iter < k + 1) {
      // reached the maximum number of iterations
      return NE_OPT_DID_NOT_CONVERGE;
    }

    // update vectors s and y:
    //   s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
    //   y_{k+1} = g_{k+1} - g_{k}.
    //
    ne_vec_sub_f32(nx, lm[end].s, x, xp);
    ne_vec_sub_f32(nx, lm[end].y, g, gp);

    // compute scalars ys and yy:
    //     ys = y^t \cdot s    -> 1 / \rho.
    //     yy = y^t \cdot y.
    //
    ne_vec_dot_f32(nx, &ys, lm[end].y, lm[end].s);
    ne_vec_dot_f32(nx, &yy, lm[end].y, lm[end].y);

    lm[end].ys = ys;

    // find new search direction
    //   ref: https://en.wikipedia.org/wiki/Limited-memory_BFGS

    bound = (m <= k) ? m : k;
    k++;
    end = (end + 1) % m;

    // initialize search direction with -g
    ne_vec_neg_f32(nx, d, g);

    j = end;
    for (int i = 0; i < bound; ++i) {
      j = (j + m - 1) % m;
      // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
      ne_vec_dot_f32(nx, &lm[j].alpha, lm[j].s, d);
      lm[j].alpha /= lm[j].ys;
      // q_{i} = q_{i+1} - \alpha_{i} y_{i}
      ne_vec_mad_f32(nx, d, lm[j].y, -lm[j].alpha);
    }

    ne_vec_scale_f32(nx, d, ys / yy);

    for (int i = 0; i < bound; ++i) {
      // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}
      ne_vec_dot_f32(nx, &beta, lm[j].y, d);
      beta /= lm[j].ys;
      // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}
      ne_vec_mad_f32(nx, d, lm[j].s, lm[j].alpha - beta);
      j = (j + 1) % m;
    }

    step = 1.0;
  }

  return NE_OPT_DID_NOT_CONVERGE;
}

struct ne_opt_params ne_opt_default_params(enum ne_opt_type type) {
  struct ne_opt_params result;

  switch (type) {
    case NE_OPT_ADAM: {
      result = (struct ne_opt_params){
          .type = NE_OPT_ADAM,
          .n_threads = 1,
          .past = 0,
          .delta = 1e-5f,

          .max_no_improvement = 100,

          .print_forward_graph = true,
          .print_backward_graph = true,

          .adam =
              {
                  .n_iter = 10000,
                  .alpha = 0.001f,
                  .beta1 = 0.9f,
                  .beta2 = 0.999f,
                  .eps = 1e-8f,
                  .eps_f = 1e-5f,
                  .eps_g = 1e-3f,
              },
      };
    } break;
    case NE_OPT_LBFGS: {
      result = (struct ne_opt_params){
          .type = NE_OPT_LBFGS,
          .n_threads = 1,
          .past = 0,
          .delta = 1e-5f,

          .max_no_improvement = 0,

          .print_forward_graph = true,
          .print_backward_graph = true,

          .lbfgs =
              {
                  .m = 6,
                  .n_iter = 100,
                  .max_linesearch = 20,

                  .eps = 1e-5f,
                  .ftol = 1e-4f,
                  .wolfe = 0.9f,
                  .min_step = 1e-20f,
                  .max_step = 1e+20f,

                  .linesearch = NE_LINESEARCH_DEFAULT,
              },
      };
    } break;
  }

  return result;
}

enum ne_opt_result ne_opt(struct ne_context* ctx, struct ne_opt_params params, struct ne_tensor* f) {
  bool free_ctx = false;
  if (ctx == NULL) {
    struct ne_init_params params_ctx = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    ctx = ne_init(params_ctx);
    if (ctx == NULL) {
      return NE_OPT_NO_CONTEXT;
    }

    free_ctx = true;
  }

  enum ne_opt_result result = NE_OPT_OK;

  // build forward + backward compute graphs
  struct ne_cgraph gf = ne_build_forward(f);
  struct ne_cgraph gb = ne_build_backward(ctx, &gf, true);

  switch (params.type) {
    case NE_OPT_ADAM: {
      result = ne_opt_adam(ctx, params, f, &gf, &gb);
    } break;
    case NE_OPT_LBFGS: {
      result = ne_opt_lbfgs(ctx, params, f, &gf, &gb);
    } break;
  }

  if (params.print_forward_graph) {
    ne_graph_print(&gf);
    ne_graph_dump_dot(&gf, NULL, "opt-forward.dot");
  }

  if (params.print_backward_graph) {
    ne_graph_print(&gb);
    ne_graph_dump_dot(&gb, &gf, "opt-backward.dot");
  }

  if (free_ctx) {
    ne_free(ctx);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////

size_t ne_quantize_q4_0(const float* src, void* dst, int n, int k, int64_t* hist) {
  assert(k % QK4_0 == 0);
  const size_t nb = k / QK4_0;

  for (int b = 0; b < n; b += k) {
    block_q4_0* restrict y = (block_q4_0*)dst + b / QK4_0;

    quantize_row_q4_0_reference(src + b, y, k);

    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < QK4_0; j += 2) {
        const uint8_t vi0 = y[i].qs[j / 2] & 0x0F;
        const uint8_t vi1 = y[i].qs[j / 2] >> 4;

        hist[vi0]++;
        hist[vi1]++;
      }
    }
  }

  return (n / QK4_0 * sizeof(block_q4_0));
}

size_t ne_quantize_q4_1(const float* src, void* dst, int n, int k, int64_t* hist) {
  assert(k % QK4_1 == 0);
  const size_t nb = k / QK4_1;

  for (int b = 0; b < n; b += k) {
    block_q4_1* restrict y = (block_q4_1*)dst + b / QK4_1;

    quantize_row_q4_1_reference(src + b, y, k);

    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < QK4_1; j += 2) {
        const uint8_t vi0 = y[i].qs[j / 2] & 0x0F;
        const uint8_t vi1 = y[i].qs[j / 2] >> 4;

        hist[vi0]++;
        hist[vi1]++;
      }
    }
  }

  return (n / QK4_1 * sizeof(block_q4_1));
}

size_t ne_quantize_q5_0(const float* src, void* dst, int n, int k, int64_t* hist) {
  assert(k % QK5_0 == 0);
  const size_t nb = k / QK5_0;

  for (int b = 0; b < n; b += k) {
    block_q5_0* restrict y = (block_q5_0*)dst + b / QK5_0;

    quantize_row_q5_0_reference(src + b, y, k);

    for (int i = 0; i < nb; i++) {
      uint32_t qh;
      memcpy(&qh, &y[i].qh, sizeof(qh));

      for (int j = 0; j < QK5_0; j += 2) {
        const uint8_t vh0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
        const uint8_t vh1 = ((qh & (1u << (j + 16))) >> (j + 12));

        // cast to 16 bins
        const uint8_t vi0 = ((y[i].qs[j / 2] & 0x0F) | vh0) / 2;
        const uint8_t vi1 = ((y[i].qs[j / 2] >> 4) | vh1) / 2;

        hist[vi0]++;
        hist[vi1]++;
      }
    }
  }

  return (n / QK5_0 * sizeof(block_q5_0));
}

size_t ne_quantize_q5_1(const float* src, void* dst, int n, int k, int64_t* hist) {
  assert(k % QK5_1 == 0);
  const size_t nb = k / QK5_1;

  for (int b = 0; b < n; b += k) {
    block_q5_1* restrict y = (block_q5_1*)dst + b / QK5_1;

    quantize_row_q5_1_reference(src + b, y, k);

    for (int i = 0; i < nb; i++) {
      uint32_t qh;
      memcpy(&qh, &y[i].qh, sizeof(qh));

      for (int j = 0; j < QK5_1; j += 2) {
        const uint8_t vh0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
        const uint8_t vh1 = ((qh & (1u << (j + 16))) >> (j + 12));

        // cast to 16 bins
        const uint8_t vi0 = ((y[i].qs[j / 2] & 0x0F) | vh0) / 2;
        const uint8_t vi1 = ((y[i].qs[j / 2] >> 4) | vh1) / 2;

        hist[vi0]++;
        hist[vi1]++;
      }
    }
  }

  return (n / QK5_1 * sizeof(block_q5_1));
}

size_t ne_quantize_q8_0(const float* src, void* dst, int n, int k, int64_t* hist) {
  assert(k % QK8_0 == 0);
  const size_t nb = k / QK8_0;

  for (int b = 0; b < n; b += k) {
    block_q8_0* restrict y = (block_q8_0*)dst + b / QK8_0;

    quantize_row_q8_0_reference(src + b, y, k);

    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < QK8_0; ++j) {
        const int8_t vi = y[i].qs[j];

        hist[vi / 16 + 8]++;
      }
    }
  }

  return (n / QK8_0 * sizeof(block_q8_0));
}

size_t ne_quantize_chunk(enum ne_type type, const float* src, void* dst, int start, int n, int64_t* hist) {
  size_t result = 0;
  switch (type) {
    case NE_TYPE_Q4_0: {
      NE_ASSERT(start % QK4_0 == 0);
      block_q4_0* block = (block_q4_0*)dst + start / QK4_0;
      result = ne_quantize_q4_0(src + start, block, n, n, hist);
    } break;
    case NE_TYPE_Q4_1: {
      NE_ASSERT(start % QK4_1 == 0);
      block_q4_1* block = (block_q4_1*)dst + start / QK4_1;
      result = ne_quantize_q4_1(src + start, block, n, n, hist);
    } break;
    case NE_TYPE_Q5_0: {
      NE_ASSERT(start % QK5_0 == 0);
      block_q5_0* block = (block_q5_0*)dst + start / QK5_0;
      result = ne_quantize_q5_0(src + start, block, n, n, hist);
    } break;
    case NE_TYPE_Q5_1: {
      NE_ASSERT(start % QK5_1 == 0);
      block_q5_1* block = (block_q5_1*)dst + start / QK5_1;
      result = ne_quantize_q5_1(src + start, block, n, n, hist);
    } break;
    case NE_TYPE_Q8_0: {
      NE_ASSERT(start % QK8_0 == 0);
      block_q8_0* block = (block_q8_0*)dst + start / QK8_0;
      result = ne_quantize_q8_0(src + start, block, n, n, hist);
    } break;
    default:
      assert(false);
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////

int ne_cpu_has_avx(void) {
#if defined(__AVX__)
  return 1;
#else
  return 0;
#endif
}

int ne_cpu_has_avx2(void) {
#if defined(__AVX2__)
  return 1;
#else
  return 0;
#endif
}

int ne_cpu_has_avx512(void) {
#if defined(__AVX512F__)
  return 1;
#else
  return 0;
#endif
}

int ne_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
  return 1;
#else
  return 0;
#endif
}

int ne_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
  return 1;
#else
  return 0;
#endif
}

int ne_cpu_has_fma(void) {
#if defined(__FMA__)
  return 1;
#else
  return 0;
#endif
}

int ne_cpu_has_f16c(void) {
#if defined(__F16C__)
  return 1;
#else
  return 0;
#endif
}

int ne_cpu_has_blas(void) { return 0; }

int ne_cpu_has_sse3(void) {
#if defined(__SSE3__)
  return 1;
#else
  return 0;
#endif
}

int ne_cpu_has_vsx(void) { return 0; }

////////////////////////////////////////////////////////////////////////////////
