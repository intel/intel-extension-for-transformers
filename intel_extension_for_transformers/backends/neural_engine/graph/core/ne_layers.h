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
#include <stdio.h>
#include <stdlib.h>

#include "core/ne.h"
#include "core/data_types.h"
#include "layers/layers.h"

#define NE_FILE_MAGIC 0x67676d6c  // "ne"
#define NE_FILE_VERSION 1

#define NE_QNT_VERSION 2            // bump this on quantization format changes
#define NE_QNT_VERSION_FACTOR 1000  // do not change this

#define NE_MAX_DIMS 4
#define NE_MAX_NODES 4096
#define NE_MAX_PARAMS 256
#define NE_MAX_CONTEXTS 64
#define NE_MAX_OPT 4
#define NE_DEFAULT_N_THREADS 4

#define NE_ASSERT(x)                                                     \
  do {                                                                   \
    if (!(x)) {                                                          \
      fprintf(stderr, "NE_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
      abort();                                                           \
    }                                                                    \
  } while (0)

#ifdef __cplusplus
extern "C" {
#endif

// convert FP16 <-> FP32
NE_API float ne_fp16_to_fp32(ne_fp16_t x);
NE_API ne_fp16_t ne_fp32_to_fp16(float x);

NE_API void ne_fp16_to_fp32_row(const ne_fp16_t* x, float* y, size_t n);
NE_API void ne_fp32_to_fp16_row(const float* x, ne_fp16_t* y, size_t n);

// misc

NE_API void ne_time_init(void);  // call this once at the beginning of the program
NE_API int64_t ne_time_ms(void);
NE_API int64_t ne_time_us(void);
NE_API int64_t ne_cycles(void);
NE_API int64_t ne_cycles_per_ms(void);

NE_API void ne_print_object(const struct ne_object* obj);
NE_API void ne_print_objects(const struct ne_context* ctx);

NE_API int64_t ne_nelements(const struct ne_tensor* tensor);
NE_API size_t ne_nbytes(const struct ne_tensor* tensor);

NE_API int ne_blck_size(enum ne_type type);
NE_API size_t ne_type_size(enum ne_type type);  // size in bytes for all elements in a block
NE_API float ne_type_sizef(enum ne_type type);  // ne_type_size()/ne_blck_size() as float

NE_API const char* ne_type_name(enum ne_type type);

NE_API size_t ne_element_size(const struct ne_tensor* tensor);

NE_API bool ne_is_quantized(enum ne_type type);

// TODO: temporary until model loading of ne examples is refactored
NE_API enum ne_type ne_ftype_to_ne_type(enum ne_ftype ftype);

// main

NE_API struct ne_context* ne_init(struct ne_init_params params);
NE_API void ne_free(struct ne_context* ctx);

NE_API size_t ne_used_mem(const struct ne_context* ctx);

NE_API size_t ne_set_scratch(struct ne_context* ctx, struct ne_scratch scratch);

NE_API struct ne_tensor* ne_new_tensor(struct ne_context* ctx, enum ne_type type, int n_dims, const int64_t* ne,
                                       size_t size);

NE_API struct ne_tensor* ne_new_tensor_1d(struct ne_context* ctx, enum ne_type type, int64_t ne0, size_t size);

NE_API struct ne_tensor* ne_new_tensor_2d(struct ne_context* ctx, enum ne_type type, int64_t ne0, int64_t ne1,
                                          size_t size);

NE_API struct ne_tensor* ne_new_tensor_3d(struct ne_context* ctx, enum ne_type type, int64_t ne0, int64_t ne1,
                                          int64_t ne2, size_t size);

NE_API struct ne_tensor* ne_new_tensor_4d(struct ne_context* ctx, enum ne_type type, int64_t ne0, int64_t ne1,
                                          int64_t ne2, int64_t ne3, size_t size);

#define d_ne_new_tensor(...) ne_new_tensor(__VA_ARGS__,NE_SIZE_CALC)
#define d_ne_new_tensor_1d(...) ne_new_tensor_1d(__VA_ARGS__, NE_SIZE_CALC)
#define d_ne_new_tensor_2d(...) ne_new_tensor_2d(__VA_ARGS__, NE_SIZE_CALC)
#define d_ne_new_tensor_3d(...) ne_new_tensor_3d(__VA_ARGS__, NE_SIZE_CALC)
#define d_ne_new_tensor_4d(...) ne_new_tensor_4d(__VA_ARGS__, NE_SIZE_CALC)

NE_API struct ne_tensor* ne_new_i32(struct ne_context* ctx, int32_t value);
NE_API struct ne_tensor* ne_new_f32(struct ne_context* ctx, float value);

NE_API struct ne_tensor* ne_dup_tensor(struct ne_context* ctx, const struct ne_tensor* src);
NE_API struct ne_tensor* ne_view_tensor(struct ne_context* ctx, const struct ne_tensor* src);

NE_API struct ne_tensor* ne_set_zero(struct ne_tensor* tensor);
NE_API struct ne_tensor* ne_set_i32(struct ne_tensor* tensor, int32_t value);
NE_API struct ne_tensor* ne_set_f32(struct ne_tensor* tensor, float value);

NE_API int32_t ne_get_i32_1d(const struct ne_tensor* tensor, int i);
NE_API void ne_set_i32_1d(const struct ne_tensor* tensor, int i, int32_t value);

NE_API float ne_get_f32_1d(const struct ne_tensor* tensor, int i);
NE_API void ne_set_f32_1d(const struct ne_tensor* tensor, int i, float value);

NE_API void* ne_get_data(const struct ne_tensor* tensor);
NE_API float* ne_get_data_f32(const struct ne_tensor* tensor);

NE_API const char* ne_get_name(const struct ne_tensor* tensor);
NE_API void ne_set_name(struct ne_tensor* tensor, const char* name);

//
// operations on tensors with backpropagation
//

NE_API struct ne_tensor* ne_dup(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_add(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

NE_API struct ne_tensor* ne_add_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

NE_API struct ne_tensor* ne_add1(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

NE_API struct ne_tensor* ne_acc(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1,
                                size_t nb2, size_t nb3, size_t offset);

NE_API struct ne_tensor* ne_acc_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1,
                                        size_t nb2, size_t nb3, size_t offset);

NE_API struct ne_tensor* ne_sub(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

NE_API struct ne_tensor* ne_mul(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

NE_API struct ne_tensor* ne_div(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

NE_API struct ne_tensor* ne_sqr(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_sqrt(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_log(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_log_inplace(struct ne_context* ctx, struct ne_tensor* a);

// return scalar
NE_API struct ne_tensor* ne_sum(struct ne_context* ctx, struct ne_tensor* a);

// sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
NE_API struct ne_tensor* ne_sum_rows(struct ne_context* ctx, struct ne_tensor* a);

// mean along rows
NE_API struct ne_tensor* ne_mean(struct ne_context* ctx, struct ne_tensor* a);

// if a is the same shape as b, and a is not parameter, return a
// otherwise, return a new tensor: repeat(a) to fit in b
NE_API struct ne_tensor* ne_repeat(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

NE_API struct ne_tensor* ne_abs(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_sgn(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_neg(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_step(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_relu(struct ne_context* ctx, struct ne_tensor* a);

// TODO: double-check this computation is correct
NE_API struct ne_tensor* ne_gelu(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_silu(struct ne_context* ctx, struct ne_tensor* a);

// a - x
// b - dy
NE_API struct ne_tensor* ne_silu_back(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

// normalize along rows
// TODO: eps is hardcoded to 1e-5 for now
NE_API struct ne_tensor* ne_norm(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_rms_norm(struct ne_context* ctx, struct ne_tensor* a);

// a - x
// b - dy
NE_API struct ne_tensor* ne_rms_norm_back(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

// A: m rows, n columns
// B: p rows, n columns (i.e. we transpose it internally)
// result is m columns, p rows
NE_API struct ne_tensor* ne_mul_mat(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

// merged Q K V  ne_mul_mat
NE_API struct ne_tensor* ne_mul_qkv(struct ne_context* ctx, struct ne_tensor* qw, struct ne_tensor* kw,
                                    struct ne_tensor* vw, struct ne_tensor* src);

// merged Q K V  ne_mul_mat
NE_API struct ne_tensor* ne_ffn_silu(struct ne_context* ctx, struct ne_tensor* w1, struct ne_tensor* w2,
                                     struct ne_tensor* w3, struct ne_tensor* src);

//
// operations on tensors without backpropagation
//

NE_API struct ne_tensor* ne_scale(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

// in-place, returns view(a)
NE_API struct ne_tensor* ne_scale_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

// b -> view(a,offset,nb1,nb2,3), return modified a
NE_API struct ne_tensor* ne_set(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1,
                                size_t nb2, size_t nb3, size_t offset);

// b -> view(a,offset,nb1,nb2,3), return view(a)
NE_API struct ne_tensor* ne_set_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1,
                                        size_t nb2, size_t nb3, size_t offset);

NE_API struct ne_tensor* ne_set_1d(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t offset);

NE_API struct ne_tensor* ne_set_1d_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b,
                                           size_t offset);

// b -> view(a,offset,nb1,nb2,3), return modified a
NE_API struct ne_tensor* ne_set_2d(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1,
                                   size_t offset);

// b -> view(a,offset,nb1,nb2,3), return view(a)
NE_API struct ne_tensor* ne_set_2d_inplace(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b, size_t nb1,
                                           size_t offset);

// a -> b, return view(b)
NE_API struct ne_tensor* ne_cpy(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

// make contiguous
NE_API struct ne_tensor* ne_cont(struct ne_context* ctx, struct ne_tensor* a);

// return view(a), b specifies the new shape
// TODO: when we start computing gradient, make a copy instead of view
NE_API struct ne_tensor* ne_reshape(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
NE_API struct ne_tensor* ne_reshape_1d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0);

NE_API struct ne_tensor* ne_reshape_2d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
NE_API struct ne_tensor* ne_reshape_3d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1,
                                       int64_t ne2);

NE_API struct ne_tensor* ne_reshape_4d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1,
                                       int64_t ne2, int64_t ne3);

// offset in bytes
NE_API struct ne_tensor* ne_view_1d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, size_t offset);

NE_API struct ne_tensor* ne_view_2d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1,
                                    size_t nb1,  // row stride in bytes
                                    size_t offset);

NE_API struct ne_tensor* ne_view_3d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2,
                                    size_t nb1,  // row   stride in bytes
                                    size_t nb2,  // slice stride in bytes
                                    size_t offset);

NE_API struct ne_tensor* ne_view_4d(struct ne_context* ctx, struct ne_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2,
                                    int64_t ne3,
                                    size_t nb1,  // row   stride in bytes
                                    size_t nb2,  // slice stride in bytes
                                    size_t nb3, size_t offset);

NE_API struct ne_tensor* ne_permute(struct ne_context* ctx, struct ne_tensor* a, int axis0, int axis1, int axis2,
                                    int axis3);

// alias for ne_permute(ctx, a, 1, 0, 2, 3)
NE_API struct ne_tensor* ne_transpose(struct ne_context* ctx, struct ne_tensor* a);

NE_API struct ne_tensor* ne_get_rows(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

NE_API struct ne_tensor* ne_get_rows_back(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b,
                                          struct ne_tensor* c);

NE_API struct ne_tensor* ne_diag(struct ne_context* ctx, struct ne_tensor* a);

// set elements above the diagonal to -INF
NE_API struct ne_tensor* ne_diag_mask_inf(struct ne_context* ctx, struct ne_tensor* a, int n_past);

// in-place, returns view(a)
NE_API struct ne_tensor* ne_diag_mask_inf_inplace(struct ne_context* ctx, struct ne_tensor* a, int n_past);

// set elements above the diagonal to 0
NE_API struct ne_tensor* ne_diag_mask_zero(struct ne_context* ctx, struct ne_tensor* a, int n_past);

// in-place, returns view(a)
NE_API struct ne_tensor* ne_diag_mask_zero_inplace(struct ne_context* ctx, struct ne_tensor* a, int n_past);

NE_API struct ne_tensor* ne_soft_max(struct ne_context* ctx, struct ne_tensor* a);

// in-place, returns view(a)
NE_API struct ne_tensor* ne_soft_max_inplace(struct ne_context* ctx, struct ne_tensor* a);

// rotary position embedding
// if mode & 1 == 1, skip n_past elements
// if mode & 2 == 1, GPT-NeoX style
// TODO: avoid creating a new tensor every time
NE_API struct ne_tensor* ne_rope(struct ne_context* ctx, struct ne_tensor* a, int n_past, int n_dims, int mode);

// in-place, returns view(a)
NE_API struct ne_tensor* ne_rope_inplace(struct ne_context* ctx, struct ne_tensor* a, int n_past, int n_dims, int mode);

// rotary position embedding backward, i.e compute dx from dy
// a - dy
NE_API struct ne_tensor* ne_rope_back(struct ne_context* ctx, struct ne_tensor* a, int n_past, int n_dims, int mode);

// alibi position embedding
// in-place, returns view(a)
struct ne_tensor* ne_alibi(struct ne_context* ctx, struct ne_tensor* a, int n_past, int n_head, float bias_max);

// clamp
// in-place, returns view(a)
struct ne_tensor* ne_clamp(struct ne_context* ctx, struct ne_tensor* a, float min, float max);

// padding = 1
// TODO: we don't support extra parameters for now
//       that's why we are hard-coding the stride, padding, and dilation
//       not great ..
NE_API struct ne_tensor* ne_conv_1d_1s(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

NE_API struct ne_tensor* ne_conv_1d_2s(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b);

NE_API struct ne_tensor* ne_flash_attn(struct ne_context* ctx, struct ne_tensor* q, struct ne_tensor* k,
                                       struct ne_tensor* v, bool masked);

NE_API struct ne_tensor* ne_flash_ff(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b0,
                                     struct ne_tensor* b1, struct ne_tensor* c0, struct ne_tensor* c1);

// Mapping operations
typedef void (*ne_unary_op_f32_t)(const int, float*, const float*);
typedef void (*ne_binary_op_f32_t)(const int, float*, const float*, const float*);

NE_API struct ne_tensor* ne_map_unary_f32(struct ne_context* ctx, struct ne_tensor* a, ne_unary_op_f32_t fun);

NE_API struct ne_tensor* ne_map_binary_f32(struct ne_context* ctx, struct ne_tensor* a, struct ne_tensor* b,
                                           ne_binary_op_f32_t fun);

//
// automatic differentiation
//

NE_API void ne_set_param(struct ne_context* ctx, struct ne_tensor* tensor);

NE_API void ne_build_forward_expand(struct ne_cgraph* cgraph, struct ne_tensor* tensor);

NE_API struct ne_cgraph ne_build_forward(struct ne_tensor* tensor);
NE_API struct ne_cgraph ne_build_backward(struct ne_context* ctx, struct ne_cgraph* gf, bool keep);

NE_API void ne_graph_compute(struct ne_context* ctx, struct ne_cgraph* cgraph);
NE_API void ne_graph_reset(struct ne_cgraph* cgraph);

// print info and performance information for the graph
NE_API void ne_graph_print(const struct ne_cgraph* cgraph);

// profiling the performance information for each kernel in graph, enable by set env ENGINE_PROFILING = 1
NE_API void ne_graph_profiling(const struct ne_cgraph* cgraph);

// dump the graph into a file using the dot format
NE_API void ne_graph_dump_dot(const struct ne_cgraph* gb, const struct ne_cgraph* gf, const char* filename);

//
// optimization
//

// optimization methods
enum ne_opt_type {
  NE_OPT_ADAM,
  NE_OPT_LBFGS,
};

// linesearch methods
enum ne_linesearch {
  NE_LINESEARCH_DEFAULT = 1,

  NE_LINESEARCH_BACKTRACKING_ARMIJO = 0,
  NE_LINESEARCH_BACKTRACKING_WOLFE = 1,
  NE_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
};

// optimization return values
enum ne_opt_result {
  NE_OPT_OK = 0,
  NE_OPT_DID_NOT_CONVERGE,
  NE_OPT_NO_CONTEXT,
  NE_OPT_INVALID_WOLFE,
  NE_OPT_FAIL,

  NE_LINESEARCH_FAIL = -128,
  NE_LINESEARCH_MINIMUM_STEP,
  NE_LINESEARCH_MAXIMUM_STEP,
  NE_LINESEARCH_MAXIMUM_ITERATIONS,
  NE_LINESEARCH_INVALID_PARAMETERS,
};

// optimization parameters
//
//   see ne.c (ne_opt_default_params) for default values
//
struct ne_opt_params {
  enum ne_opt_type type;

  int n_threads;

  // delta-based convergence test
  //
  //   if past == 0 - disabled
  //   if past > 0:
  //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
  //
  int past;
  float delta;

  // maximum number of iterations without improvement
  //
  //   if 0 - disabled
  //   if > 0:
  //     assume convergence if no cost improvement in this number of iterations
  //
  int max_no_improvement;

  bool print_forward_graph;
  bool print_backward_graph;

  // ADAM parameters
  struct {
    int n_iter;

    float alpha;  // learning rate
    float beta1;
    float beta2;
    float eps;    // epsilon for numerical stability
    float eps_f;  // epsilon for convergence test
    float eps_g;  // epsilon for convergence test
  } adam;

  // LBFGS parameters
  struct {
    int m;  // number of corrections to approximate the inv. Hessian
    int n_iter;
    int max_linesearch;

    float eps;   // convergence tolerance
    float ftol;  // line search tolerance
    float wolfe;
    float min_step;
    float max_step;

    enum ne_linesearch linesearch;
  } lbfgs;
};

NE_API struct ne_opt_params ne_opt_default_params(enum ne_opt_type type);

// optimize the function defined by the tensor f
NE_API enum ne_opt_result ne_opt(struct ne_context* ctx, struct ne_opt_params params, struct ne_tensor* f);

//
// quantization
//

NE_API size_t ne_quantize_q4_0(const float* src, void* dst, int n, int k, int64_t* hist);
NE_API size_t ne_quantize_q4_1(const float* src, void* dst, int n, int k, int64_t* hist);
NE_API size_t ne_quantize_q5_0(const float* src, void* dst, int n, int k, int64_t* hist);
NE_API size_t ne_quantize_q5_1(const float* src, void* dst, int n, int k, int64_t* hist);
NE_API size_t ne_quantize_q8_0(const float* src, void* dst, int n, int k, int64_t* hist);

NE_API size_t ne_quantize_chunk(enum ne_type type, const float* src, void* dst, int start, int n, int64_t* hist);

//
// system info
//

NE_API int ne_cpu_has_avx(void);
NE_API int ne_cpu_has_avx2(void);
NE_API int ne_cpu_has_avx512(void);
NE_API int ne_cpu_has_avx512_vbmi(void);
NE_API int ne_cpu_has_avx512_vnni(void);
NE_API int ne_cpu_has_fma(void);
NE_API int ne_cpu_has_f16c(void);
NE_API int ne_cpu_has_blas(void);
NE_API int ne_cpu_has_sse3(void);
NE_API int ne_cpu_has_vsx(void);

//
// Internal types and functions exposed for tests and benchmarks
//

#ifdef __cplusplus
// restrict not standard in C++
#define NE_RESTRICT
#else
#define NE_RESTRICT restrict
#endif
typedef void (*dequantize_row_q_t)(const void* NE_RESTRICT x, float* NE_RESTRICT y, int k);
typedef void (*quantize_row_q_t)(const float* NE_RESTRICT x, void* NE_RESTRICT y, int k);
typedef void (*vec_dot_q_t)(const int n, float* NE_RESTRICT s, const void* NE_RESTRICT x, const void* NE_RESTRICT y);

typedef struct {
  dequantize_row_q_t dequantize_row_q;
  quantize_row_q_t quantize_row_q;
  quantize_row_q_t quantize_row_q_reference;
  quantize_row_q_t quantize_row_q_dot;
  vec_dot_q_t vec_dot_q;
  enum ne_type vec_dot_type;
} quantize_fns_t;

quantize_fns_t ne_internal_get_quantize_fn(size_t i);

#ifdef __cplusplus
}
#endif
