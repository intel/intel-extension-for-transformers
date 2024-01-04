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
// Defines fileno on msys:

#ifndef GGUF_H
#define GGUF_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#include "core/layers/jblas_common.hpp"
#include "core/ne_layers.h"
#include "models/model_utils/util.h"

#define GGML_MAX_DIMS 4
#define GGUF_MAGIC "GGUF"

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_API __declspec(dllexport)
#        else
#            define GGML_API __declspec(dllimport)
#        endif
#    else
#        define GGML_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GGML_API
#endif

    // typedef void (*ggml_to_float_t)  (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
    // typedef void (*ggml_from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int k);
    // typedef void (*ggml_vec_dot_t)   (const int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT x, const void * GGML_RESTRICT y);

    // typedef struct {
    //     const char      * type_name;
    //     int               blck_size;
    //     size_t            type_size;
    //     bool              is_quantized;
    //     ggml_to_float_t   to_float;
    //     ggml_from_float_t from_float;
    //     ggml_from_float_t from_float_reference;
    //     ggml_vec_dot_t    vec_dot;
    //     enum ggml_type    vec_dot_type;
    // } ggml_type_traits_t;

    // GGML_API ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);

// static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
//     [GGML_TYPE_I8] = {
//         .type_name                = "i8",
//         .blck_size                = 1,
//         .type_size                = sizeof(int8_t),
//         .is_quantized             = false,
//     },
//     [GGML_TYPE_I16] = {
//         .type_name                = "i16",
//         .blck_size                = 1,
//         .type_size                = sizeof(int16_t),
//         .is_quantized             = false,
//     },
//     [GGML_TYPE_I32] = {
//         .type_name                = "i32",
//         .blck_size                = 1,
//         .type_size                = sizeof(int32_t),
//         .is_quantized             = false,
//     },
//     [GGML_TYPE_F32] = {
//         .type_name                = "f32",
//         .blck_size                = 1,
//         .type_size                = sizeof(float),
//         .is_quantized             = false,
//         .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
//         .vec_dot_type             = GGML_TYPE_F32,
//     },
//     [GGML_TYPE_F16] = {
//         .type_name                = "f16",
//         .blck_size                = 1,
//         .type_size                = sizeof(ggml_fp16_t),
//         .is_quantized             = false,
//         .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
//         .from_float               = (ggml_from_float_t) ggml_fp32_to_fp16_row,
//         .from_float_reference     = (ggml_from_float_t) ggml_fp32_to_fp16_row,
//         .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
//         .vec_dot_type             = GGML_TYPE_F16,
//     },
//     [GGML_TYPE_Q4_0] = {
//         .type_name                = "q4_0",
//         .blck_size                = QK4_0,
//         .type_size                = sizeof(block_q4_0),
//         .is_quantized             = true,
//         .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
//         .from_float               = quantize_row_q4_0,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q4_0_reference,
//         .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
//         .vec_dot_type             = GGML_TYPE_Q8_0,
//     },
//     [GGML_TYPE_Q4_1] = {
//         .type_name                = "q4_1",
//         .blck_size                = QK4_1,
//         .type_size                = sizeof(block_q4_1),
//         .is_quantized             = true,
//         .to_float                 = (ggml_to_float_t) dequantize_row_q4_1,
//         .from_float               = quantize_row_q4_1,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q4_1_reference,
//         .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
//         .vec_dot_type             = GGML_TYPE_Q8_1,
//     },
//     [4] = { // GGML_TYPE_Q4_2
//         .type_name                = "DEPRECATED",
//         .blck_size                = 0,
//         .type_size                = 0,
//         .is_quantized             = false,
//         .to_float                 = NULL,
//         .from_float               = NULL,
//         .from_float_reference     = NULL,
//         .vec_dot                  = NULL,
//         .vec_dot_type             = GGML_TYPE_COUNT,
//     },
//     [5] = { // GGML_TYPE_Q4_3
//         .type_name                = "DEPRECATED",
//         .blck_size                = 0,
//         .type_size                = 0,
//         .is_quantized             = false,
//         .to_float                 = NULL,
//         .from_float               = NULL,
//         .from_float_reference     = NULL,
//         .vec_dot                  = NULL,
//         .vec_dot_type             = GGML_TYPE_COUNT,
//     },
//     [GGML_TYPE_Q5_0] = {
//         .type_name                = "q5_0",
//         .blck_size                = QK5_0,
//         .type_size                = sizeof(block_q5_0),
//         .is_quantized             = true,
//         .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
//         .from_float               = quantize_row_q5_0,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q5_0_reference,
//         .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
//         .vec_dot_type             = GGML_TYPE_Q8_0,
//     },
//     [GGML_TYPE_Q5_1] = {
//         .type_name                = "q5_1",
//         .blck_size                = QK5_1,
//         .type_size                = sizeof(block_q5_1),
//         .is_quantized             = true,
//         .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
//         .from_float               = quantize_row_q5_1,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q5_1_reference,
//         .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
//         .vec_dot_type             = GGML_TYPE_Q8_1,
//     },
//     [GGML_TYPE_Q8_0] = {
//         .type_name                = "q8_0",
//         .blck_size                = QK8_0,
//         .type_size                = sizeof(block_q8_0),
//         .is_quantized             = true,
//         .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
//         .from_float               = quantize_row_q8_0,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q8_0_reference,
//         .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
//         .vec_dot_type             = GGML_TYPE_Q8_0,
//     },
//     [GGML_TYPE_Q8_1] = {
//         .type_name                = "q8_1",
//         .blck_size                = QK8_1,
//         .type_size                = sizeof(block_q8_1),
//         .is_quantized             = true,
//         .from_float               = quantize_row_q8_1,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q8_1_reference,
//         .vec_dot_type             = GGML_TYPE_Q8_1,
//     },
//     [GGML_TYPE_Q2_K] = {
//         .type_name                = "q2_K",
//         .blck_size                = QK_K,
//         .type_size                = sizeof(block_q2_K),
//         .is_quantized             = true,
//         .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
//         .from_float               = quantize_row_q2_K,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q2_K_reference,
//         .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
//         .vec_dot_type             = GGML_TYPE_Q8_K,
//     },
//     [GGML_TYPE_Q3_K] = {
//         .type_name                = "q3_K",
//         .blck_size                = QK_K,
//         .type_size                = sizeof(block_q3_K),
//         .is_quantized             = true,
//         .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
//         .from_float               = quantize_row_q3_K,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q3_K_reference,
//         .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
//         .vec_dot_type             = GGML_TYPE_Q8_K,
//     },
//     [GGML_TYPE_Q4_K] = {
//         .type_name                = "q4_K",
//         .blck_size                = QK_K,
//         .type_size                = sizeof(block_q4_K),
//         .is_quantized             = true,
//         .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
//         .from_float               = quantize_row_q4_K,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q4_K_reference,
//         .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
//         .vec_dot_type             = GGML_TYPE_Q8_K,
//     },
//     [GGML_TYPE_Q5_K] = {
//         .type_name                = "q5_K",
//         .blck_size                = QK_K,
//         .type_size                = sizeof(block_q5_K),
//         .is_quantized             = true,
//         .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
//         .from_float               = quantize_row_q5_K,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q5_K_reference,
//         .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
//         .vec_dot_type             = GGML_TYPE_Q8_K,
//     },
//     [GGML_TYPE_Q6_K] = {
//         .type_name                = "q6_K",
//         .blck_size                = QK_K,
//         .type_size                = sizeof(block_q6_K),
//         .is_quantized             = true,
//         .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
//         .from_float               = quantize_row_q6_K,
//         .from_float_reference     = (ggml_from_float_t) quantize_row_q6_K_reference,
//         .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
//         .vec_dot_type             = GGML_TYPE_Q8_K,
//     },
//     [GGML_TYPE_Q8_K] = {
//         .type_name                = "q8_K",
//         .blck_size                = QK_K,
//         .type_size                = sizeof(block_q8_K),
//         .is_quantized             = true,
//         .from_float               = quantize_row_q8_K,
//     }
// };



enum ggml_log_level { GGML_LOG_LEVEL_ERROR = 2, GGML_LOG_LEVEL_WARN = 3, GGML_LOG_LEVEL_INFO = 4 };

typedef void (*ggml_log_callback)(enum ggml_log_level level, const char* text, void* user_data);
static void llama_log_callback_default(ggml_log_level level, const char* text, void* user_data) {
  (void)level;
  (void)user_data;
  fputs(text, stderr);
  fflush(stderr);
}

struct llama_state {
  llama_state() {}

  // We save the log callback globally
  ggml_log_callback log_callback = llama_log_callback_default;
  void* log_callback_user_data = nullptr;
};

static llama_state g_state;

static void llama_log_internal(ggml_log_level level, const char* format, ...);

#define LLAMA_LOG_INFO(...) llama_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define LLAMA_LOG_WARN(...) llama_log_internal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

static void llama_log_internal_v(ggml_log_level level, const char* format, va_list args) {
  va_list args_copy;
  va_copy(args_copy, args);
  char buffer[128];
  int len = vsnprintf(buffer, 128, format, args);
  if (len < 128) {
    g_state.log_callback(level, buffer, g_state.log_callback_user_data);
  } else {
    char* buffer2 = new char[len + 1];
    vsnprintf(buffer2, len + 1, format, args_copy);
    buffer2[len] = 0;
    g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
    delete[] buffer2;
  }
  va_end(args_copy);
}

static void llama_log_internal(ggml_log_level level, const char* format, ...) {
  va_list args;
  va_start(args, format);
  llama_log_internal_v(level, format, args);
  va_end(args);
}

struct gguf_str {
  uint64_t n;  // GGUFv2
  char* data;
};

enum llama_fver {
  GGUF_FILE_VERSION_V1 = 1,
  GGUF_FILE_VERSION_V2 = 2,
  GGUF_FILE_VERSION_V3 = 3,
};

enum ggml_type {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  // GGML_TYPE_Q4_2 = 4, support has been removed
  // GGML_TYPE_Q4_3 (5) support has been removed
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  // k-quantizations
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_I8,
  GGML_TYPE_I16,
  GGML_TYPE_I32,
  GGML_TYPE_COUNT,
};

enum gguf_type {
  GGUF_TYPE_UINT8 = 0,
  GGUF_TYPE_INT8 = 1,
  GGUF_TYPE_UINT16 = 2,
  GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4,
  GGUF_TYPE_INT32 = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL = 7,
  GGUF_TYPE_STRING = 8,
  GGUF_TYPE_ARRAY = 9,
  GGUF_TYPE_UINT64 = 10,
  GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12,
  GGUF_TYPE_COUNT,  // marks the end of the enum
};

static const char* GGUF_TYPE_NAME[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8] = "u8",    [GGUF_TYPE_INT8] = "i8",   [GGUF_TYPE_UINT16] = "u16",  [GGUF_TYPE_INT16] = "i16",
    [GGUF_TYPE_UINT32] = "u32",  [GGUF_TYPE_INT32] = "i32", [GGUF_TYPE_FLOAT32] = "f32", [GGUF_TYPE_BOOL] = "bool",
    [GGUF_TYPE_STRING] = "str",  [GGUF_TYPE_ARRAY] = "arr", [GGUF_TYPE_UINT64] = "u64",  [GGUF_TYPE_INT64] = "i64",
    [GGUF_TYPE_FLOAT64] = "f64",
};

union gguf_value {
  uint8_t uint8;
  int8_t int8;
  uint16_t uint16;
  int16_t int16;
  uint32_t uint32;
  int32_t int32;
  float float32;
  uint64_t uint64;
  int64_t int64;
  double float64;
  bool bool_;

  struct gguf_str str;

  struct {
    enum gguf_type type;

    uint64_t n;  // GGUFv2
    void* data;
  } arr;
};

struct gguf_kv {
  struct gguf_str key;

  enum gguf_type type;
  union gguf_value value;
};

struct gguf_header {
  char magic[4];
  uint32_t version;
  uint64_t n_tensors;  // GGUFv2
  uint64_t n_kv;       // GGUFv2
};

struct gguf_context {
  struct gguf_header header;

  struct gguf_kv* kv;
  struct gguf_tensor_info* infos;

  size_t alignment;
  size_t offset;  // offset of `data` from beginning of file
  size_t size;    // size of `data` in bytes

  // uint8_t * padding;
  void* data;
};

#if UINTPTR_MAX == 0xFFFFFFFF
#define GGML_MEM_ALIGN 4
#else
#define GGML_MEM_ALIGN 16
#endif

#define GGUF_DEFAULT_ALIGNMENT 32

static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8] = sizeof(uint8_t),
    [GGUF_TYPE_INT8] = sizeof(int8_t),
    [GGUF_TYPE_UINT16] = sizeof(uint16_t),
    [GGUF_TYPE_INT16] = sizeof(int16_t),
    [GGUF_TYPE_UINT32] = sizeof(uint32_t),
    [GGUF_TYPE_INT32] = sizeof(int32_t),
    [GGUF_TYPE_FLOAT32] = sizeof(float),
    [GGUF_TYPE_BOOL] = sizeof(bool),
    [GGUF_TYPE_STRING] = sizeof(struct gguf_str),
    [GGUF_TYPE_ARRAY] = 0,  // undefined
    [GGUF_TYPE_UINT64] = sizeof(uint64_t),
    [GGUF_TYPE_INT64] = sizeof(int64_t),
    [GGUF_TYPE_FLOAT64] = sizeof(double),
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

enum llm_arch {
  LLM_ARCH_LLAMA,
  LLM_ARCH_FALCON,
  LLM_ARCH_BAICHUAN,
  LLM_ARCH_GPT2,
  LLM_ARCH_GPTJ,
  LLM_ARCH_GPTNEOX,
  LLM_ARCH_MPT,
  LLM_ARCH_STARCODER,
  LLM_ARCH_PERSIMMON,
  LLM_ARCH_REFACT,
  LLM_ARCH_BLOOM,
  LLM_ARCH_STABLELM,
  LLM_ARCH_QWEN,
  LLM_ARCH_CHATGLM2,
  LLM_ARCH_UNKNOWN,
};

static std::map<llm_arch, std::string> LLM_ARCH_NAMES = {
    {LLM_ARCH_LLAMA, "llama"},       {LLM_ARCH_FALCON, "falcon"},       {LLM_ARCH_GPT2, "gpt2"},
    {LLM_ARCH_GPTJ, "gptj"},         {LLM_ARCH_GPTNEOX, "gptneox"},     {LLM_ARCH_MPT, "mpt"},
    {LLM_ARCH_BAICHUAN, "baichuan"}, {LLM_ARCH_STARCODER, "starcoder"}, {LLM_ARCH_PERSIMMON, "persimmon"},
    {LLM_ARCH_REFACT, "refact"},     {LLM_ARCH_BLOOM, "bloom"},         {LLM_ARCH_STABLELM, "stablelm"},
    {LLM_ARCH_QWEN, "qwen"},         {LLM_ARCH_CHATGLM2, "chatglm2"},
};

struct gguf_tensor_info {
  struct gguf_str name;

  uint32_t n_dims;
  uint64_t ne[GGML_MAX_DIMS];

  enum ggml_type type;

  uint64_t offset;  // offset from start of `data`, must be a multiple of `ALIGNMENT`

  // for writing API
  const void* data;
  size_t size;
};

static bool gguf_fread_el(FILE* file, void* dst, size_t size, size_t* offset) {
  const size_t n = fread(dst, 1, size, file);
  *offset += n;
  return n == size;
}

static bool gguf_fread_str(FILE* file, struct gguf_str* p, size_t* offset) {
  p->n = 0;
  p->data = NULL;

  bool ok = true;

  ok = ok && gguf_fread_el(file, &p->n, sizeof(p->n), offset);
  p->data = reinterpret_cast<char*>(calloc(p->n + 1, 1));
  ok = ok && gguf_fread_el(file, p->data, p->n, offset);

  return ok;
}

static const char* llama_file_version_name(llama_fver version) {
  switch (version) {
    case GGUF_FILE_VERSION_V1:
      return "GGUF V1 (support until nov 2023)";
    case GGUF_FILE_VERSION_V2:
      return "GGUF V2";
    case GGUF_FILE_VERSION_V3:
      return "GGUF V3 (latest)";
  }

  return "unknown";
}

inline static void* ggml_aligned_malloc(size_t size) {
  if (size == 0) {
    printf("WARNING: Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
    return NULL;
  }
  void* aligned_memory = NULL;
#ifdef GGML_USE_CPU_HBM
  int result = hbw_posix_memalign(&aligned_memory, 16, size);
#elif GGML_USE_METAL
  int result = posix_memalign(&aligned_memory, sysconf(_SC_PAGESIZE), size);
#else
  int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);
#endif
  if (result != 0) {
    // Handle allocation failure
    const char* error_desc = "unknown allocation error";
    switch (result) {
      case EINVAL:
        error_desc = "invalid alignment value";
        break;
      case ENOMEM:
        error_desc = "insufficient memory";
        break;
    }
    printf("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size / (1024.0 * 1024.0));
    return NULL;
  }
  return aligned_memory;
}
#define GGML_ALIGNED_MALLOC(size) ggml_aligned_malloc(size)

#define GGUF_GET_KEY(ctx, dst, func, type, req, key)                                                        \
  do {                                                                                                      \
    const std::string skey(key);                                                                            \
    const int kid = gguf_find_key(ctx, skey.c_str());                                                       \
    if (kid >= 0) {                                                                                         \
      enum gguf_type ktype = gguf_get_kv_type(ctx, kid);                                                    \
      if (ktype != (type)) {                                                                                \
        throw std::runtime_error(format("key %s has wrong type: %s", skey.c_str(), gguf_type_name(ktype))); \
      }                                                                                                     \
      (dst) = func(ctx, kid);                                                                               \
    } else if (req) {                                                                                       \
      throw std::runtime_error(format("key not found in model: %s", skey.c_str()));                         \
    }                                                                                                       \
  } while (0)

static void replace_all(std::string& s, const std::string& search, const std::string& replace) {
  std::string result;
  for (size_t pos = 0;; pos += search.length()) {
    auto new_pos = s.find(search, pos);
    if (new_pos == std::string::npos) {
      result += s.substr(pos, s.size() - pos);
      break;
    }
    result += s.substr(pos, new_pos - pos) + replace;
    pos = new_pos;
  }
  s = std::move(result);
}

static uint32_t codepoint_from_utf8(const std::string& utf8, size_t& offset) {
  assert(offset < utf8.size());
  if (!(utf8[offset + 0] & 0x80)) {
    auto result = utf8[offset + 0];
    offset += 1;
    return result;
  } else if (!(utf8[offset + 0] & 0x40)) {
    throw std::invalid_argument("invalid character");
  } else if (!(utf8[offset + 0] & 0x20)) {
    if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80))
      throw std::invalid_argument("invalid character");
    auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
    offset += 2;
    return result;
  } else if (!(utf8[offset + 0] & 0x10)) {
    if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80))
      throw std::invalid_argument("invalid character");
    auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
    offset += 3;
    return result;
  } else if (!(utf8[offset + 0] & 0x08)) {
    if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80) ||
        !((utf8[offset + 3] & 0xc0) == 0x80))
      throw std::invalid_argument("invalid character");
    auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) |
                  ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
    offset += 4;
    return result;
  }
  throw std::invalid_argument("invalid string");
}

static std::vector<uint32_t> codepoints_from_utf8(const std::string& utf8) {
  std::vector<uint32_t> result;
  size_t offset = 0;
  while (offset < utf8.size()) {
    result.push_back(codepoint_from_utf8(utf8, offset));
  }
  return result;
}

enum llm_kv {
  LLM_KV_GENERAL_ARCHITECTURE,
  LLM_KV_GENERAL_QUANTIZATION_VERSION,
  LLM_KV_GENERAL_ALIGNMENT,
  LLM_KV_GENERAL_NAME,
  LLM_KV_GENERAL_AUTHOR,
  LLM_KV_GENERAL_URL,
  LLM_KV_GENERAL_DESCRIPTION,
  LLM_KV_GENERAL_LICENSE,
  LLM_KV_GENERAL_SOURCE_URL,
  LLM_KV_GENERAL_SOURCE_HF_REPO,

  LLM_KV_CONTEXT_LENGTH,
  LLM_KV_EMBEDDING_LENGTH,
  LLM_KV_BLOCK_COUNT,
  LLM_KV_FEED_FORWARD_LENGTH,
  LLM_KV_USE_PARALLEL_RESIDUAL,
  LLM_KV_TENSOR_DATA_LAYOUT,

  LLM_KV_ATTENTION_HEAD_COUNT,
  LLM_KV_ATTENTION_HEAD_COUNT_KV,
  LLM_KV_ATTENTION_MAX_ALIBI_BIAS,
  LLM_KV_ATTENTION_CLAMP_KQV,
  LLM_KV_ATTENTION_LAYERNORM_EPS,
  LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,

  LLM_KV_ROPE_DIMENSION_COUNT,
  LLM_KV_ROPE_FREQ_BASE,
  LLM_KV_ROPE_SCALE_LINEAR,
  LLM_KV_ROPE_SCALING_TYPE,
  LLM_KV_ROPE_SCALING_FACTOR,
  LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,
  LLM_KV_ROPE_SCALING_FINETUNED,

  LLM_KV_TOKENIZER_MODEL,
  LLM_KV_TOKENIZER_LIST,
  LLM_KV_TOKENIZER_TOKEN_TYPE,
  LLM_KV_TOKENIZER_SCORES,
  LLM_KV_TOKENIZER_MERGES,
  LLM_KV_TOKENIZER_BOS_ID,
  LLM_KV_TOKENIZER_EOS_ID,
  LLM_KV_TOKENIZER_UNK_ID,
  LLM_KV_TOKENIZER_SEP_ID,
  LLM_KV_TOKENIZER_PAD_ID,
  LLM_KV_TOKENIZER_ADD_BOS,
  LLM_KV_TOKENIZER_ADD_EOS,
  LLM_KV_TOKENIZER_HF_JSON,
  LLM_KV_TOKENIZER_RWKV,
};

static std::map<llm_kv, std::string> LLM_KV_NAMES = {
    {LLM_KV_GENERAL_ARCHITECTURE, "general.architecture"},
    {LLM_KV_GENERAL_QUANTIZATION_VERSION, "general.quantization_version"},
    {LLM_KV_GENERAL_ALIGNMENT, "general.alignment"},
    {LLM_KV_GENERAL_NAME, "general.name"},
    {LLM_KV_GENERAL_AUTHOR, "general.author"},
    {LLM_KV_GENERAL_URL, "general.url"},
    {LLM_KV_GENERAL_DESCRIPTION, "general.description"},
    {LLM_KV_GENERAL_LICENSE, "general.license"},
    {LLM_KV_GENERAL_SOURCE_URL, "general.source.url"},
    {LLM_KV_GENERAL_SOURCE_HF_REPO, "general.source.huggingface.repository"},

    {LLM_KV_CONTEXT_LENGTH, "%s.context_length"},
    {LLM_KV_EMBEDDING_LENGTH, "%s.embedding_length"},
    {LLM_KV_BLOCK_COUNT, "%s.block_count"},
    {LLM_KV_FEED_FORWARD_LENGTH, "%s.feed_forward_length"},
    {LLM_KV_USE_PARALLEL_RESIDUAL, "%s.use_parallel_residual"},
    {LLM_KV_TENSOR_DATA_LAYOUT, "%s.tensor_data_layout"},

    {LLM_KV_ATTENTION_HEAD_COUNT, "%s.attention.head_count"},
    {LLM_KV_ATTENTION_HEAD_COUNT_KV, "%s.attention.head_count_kv"},
    {LLM_KV_ATTENTION_MAX_ALIBI_BIAS, "%s.attention.max_alibi_bias"},
    {LLM_KV_ATTENTION_CLAMP_KQV, "%s.attention.clamp_kqv"},
    {LLM_KV_ATTENTION_LAYERNORM_EPS, "%s.attention.layer_norm_epsilon"},
    {LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, "%s.attention.layer_norm_rms_epsilon"},

    {LLM_KV_ROPE_DIMENSION_COUNT, "%s.rope.dimension_count"},
    {LLM_KV_ROPE_FREQ_BASE, "%s.rope.freq_base"},
    {LLM_KV_ROPE_SCALE_LINEAR, "%s.rope.scale_linear"},
    {LLM_KV_ROPE_SCALING_TYPE, "%s.rope.scaling.type"},
    {LLM_KV_ROPE_SCALING_FACTOR, "%s.rope.scaling.factor"},
    {LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, "%s.rope.scaling.original_context_length"},
    {LLM_KV_ROPE_SCALING_FINETUNED, "%s.rope.scaling.finetuned"},

    {LLM_KV_TOKENIZER_MODEL, "tokenizer.ggml.model"},
    {LLM_KV_TOKENIZER_LIST, "tokenizer.ggml.tokens"},
    {LLM_KV_TOKENIZER_TOKEN_TYPE, "tokenizer.ggml.token_type"},
    {LLM_KV_TOKENIZER_SCORES, "tokenizer.ggml.scores"},
    {LLM_KV_TOKENIZER_MERGES, "tokenizer.ggml.merges"},
    {LLM_KV_TOKENIZER_BOS_ID, "tokenizer.ggml.bos_token_id"},
    {LLM_KV_TOKENIZER_EOS_ID, "tokenizer.ggml.eos_token_id"},
    {LLM_KV_TOKENIZER_UNK_ID, "tokenizer.ggml.unknown_token_id"},
    {LLM_KV_TOKENIZER_SEP_ID, "tokenizer.ggml.seperator_token_id"},
    {LLM_KV_TOKENIZER_PAD_ID, "tokenizer.ggml.padding_token_id"},
    {LLM_KV_TOKENIZER_ADD_BOS, "tokenizer.ggml.add_bos_token"},
    {LLM_KV_TOKENIZER_ADD_EOS, "tokenizer.ggml.add_eos_token"},
    {LLM_KV_TOKENIZER_HF_JSON, "tokenizer.huggingface.json"},
    {LLM_KV_TOKENIZER_RWKV, "tokenizer.rwkv.world"},
};

struct LLM_KV {
  LLM_KV(llm_arch arch) : arch(arch) {}

  llm_arch arch;

  std::string operator()(llm_kv kv) const { return ::format(LLM_KV_NAMES[kv].c_str(), LLM_ARCH_NAMES[arch].c_str()); }
};

static std::string gguf_data_to_str(enum gguf_type type, const void* data, int i) {
  switch (type) {
    case GGUF_TYPE_UINT8:
      return std::to_string(((const uint8_t*)data)[i]);
    case GGUF_TYPE_INT8:
      return std::to_string(((const int8_t*)data)[i]);
    case GGUF_TYPE_UINT16:
      return std::to_string(((const uint16_t*)data)[i]);
    case GGUF_TYPE_INT16:
      return std::to_string(((const int16_t*)data)[i]);
    case GGUF_TYPE_UINT32:
      return std::to_string(((const uint32_t*)data)[i]);
    case GGUF_TYPE_INT32:
      return std::to_string(((const int32_t*)data)[i]);
    case GGUF_TYPE_UINT64:
      return std::to_string(((const uint64_t*)data)[i]);
    case GGUF_TYPE_INT64:
      return std::to_string(((const int64_t*)data)[i]);
    case GGUF_TYPE_FLOAT32:
      return std::to_string(((const float*)data)[i]);
    case GGUF_TYPE_FLOAT64:
      return std::to_string(((const double*)data)[i]);
    case GGUF_TYPE_BOOL:
      return ((const bool*)data)[i] ? "true" : "false";
    default:
      return format("unknown type %d", type);
  }
}

#endif  // GGUF_H