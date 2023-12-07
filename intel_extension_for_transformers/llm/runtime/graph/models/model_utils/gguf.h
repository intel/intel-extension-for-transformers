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

        struct gguf_str {
        uint64_t n;  // GGUFv2
        char * data;
    };
    
    enum gguf_type {
        GGUF_TYPE_UINT8   = 0,
        GGUF_TYPE_INT8    = 1,
        GGUF_TYPE_UINT16  = 2,
        GGUF_TYPE_INT16   = 3,
        GGUF_TYPE_UINT32  = 4,
        GGUF_TYPE_INT32   = 5,
        GGUF_TYPE_FLOAT32 = 6,
        GGUF_TYPE_BOOL    = 7,
        GGUF_TYPE_STRING  = 8,
        GGUF_TYPE_ARRAY   = 9,
        GGUF_TYPE_UINT64  = 10,
        GGUF_TYPE_INT64   = 11,
        GGUF_TYPE_FLOAT64 = 12,
        GGUF_TYPE_COUNT,       // marks the end of the enum
    };


        union gguf_value {
        uint8_t  uint8;
        int8_t   int8;
        uint16_t uint16;
        int16_t  int16;
        uint32_t uint32;
        int32_t  int32;
        float    float32;
        uint64_t uint64;
        int64_t  int64;
        double   float64;
        bool     bool_;

        struct gguf_str str;

        struct {
            enum gguf_type type;

            uint64_t n;  // GGUFv2
            void * data;
        } arr;
    };

    struct gguf_kv {
      struct gguf_str key;

      enum  gguf_type  type;
      union gguf_value value;
    };

struct gguf_header {
    char magic[4];
    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct gguf_context {
    struct gguf_header header;

    struct gguf_kv          * kv;
    struct gguf_tensor_info * infos;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};



#if UINTPTR_MAX == 0xFFFFFFFF
    #define GGML_MEM_ALIGN 4
#else
    #define GGML_MEM_ALIGN 16
#endif

// static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
//     [GGUF_TYPE_UINT8]   = sizeof(uint8_t),
//     [GGUF_TYPE_INT8]    = sizeof(int8_t),
//     [GGUF_TYPE_UINT16]  = sizeof(uint16_t),
//     [GGUF_TYPE_INT16]   = sizeof(int16_t),
//     [GGUF_TYPE_UINT32]  = sizeof(uint32_t),
//     [GGUF_TYPE_INT32]   = sizeof(int32_t),
//     [GGUF_TYPE_FLOAT32] = sizeof(float),
//     [GGUF_TYPE_BOOL]    = sizeof(bool),
//     [GGUF_TYPE_STRING]  = sizeof(struct gguf_str),
//     [GGUF_TYPE_UINT64]  = sizeof(uint64_t),
//     [GGUF_TYPE_INT64]   = sizeof(int64_t),
//     [GGUF_TYPE_FLOAT64] = sizeof(double),
//     [GGUF_TYPE_ARRAY]   = 0, // undefined
// };

  static bool gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset) {
      const size_t n = fread(dst, 1, size, file);
      *offset += n;
      return n == size;
  }

static bool gguf_fread_str(FILE * file, struct gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && gguf_fread_el(file, &p->n,    sizeof(p->n), offset); p->data = reinterpret_cast<char *>(calloc(p->n + 1, 1));
    ok = ok && gguf_fread_el(file,  p->data, p->n,         offset);

    return ok;
}







inline static void * ggml_aligned_malloc(size_t size) {
    if (size == 0) {
        printf("WARNING: Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        return NULL;
    }
    void * aligned_memory = NULL;
#ifdef GGML_USE_CPU_HBM
    int result = hbw_posix_memalign(&aligned_memory, 16, size);
#elif GGML_USE_METAL
    int result = posix_memalign(&aligned_memory, sysconf(_SC_PAGESIZE), size);
#else
    int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);
#endif
    if (result != 0) {
        // Handle allocation failure
        const char *error_desc = "unknown allocation error";
        switch (result) {
            case EINVAL:
                error_desc = "invalid alignment value";
                break;
            case ENOMEM:
                error_desc = "insufficient memory";
                break;
        }
        printf("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        return NULL;
    }
    return aligned_memory;
}
#define GGML_ALIGNED_MALLOC(size) ggml_aligned_malloc(size)

#define GGUF_GET_KEY(ctx, dst, func, type, req, key) \
do { \
    const std::string skey(key); \
    const int kid = gguf_find_key(ctx, skey.c_str()); \
    if (kid >= 0) { \
        enum gguf_type ktype = gguf_get_kv_type(ctx, kid); \
        if (ktype != (type)) { \
            throw std::runtime_error(format("key %s has wrong type: %s", skey.c_str(), gguf_type_name(ktype))); \
        } \
        (dst) = func(ctx, kid); \
    } else if (req) { \
        throw std::runtime_error(format("key not found in model: %s", skey.c_str())); \
    } \
} while (0)



#endif  // GGUF_H