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
#ifndef MODEL_TYPES_H
#define MODEL_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <array>
#include <ctime>
#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cstring>
#include <climits>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <numeric>

#include "models/util.h"
#include "core/ne_layers.h"

#ifdef MODEL_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef MODEL_BUILD
#define MODEL_API __declspec(dllexport)
#else
#define MODEL_API __declspec(dllimport)
#endif
#else
#define MODEL_API __attribute__((visibility("default")))
#endif
#else
#define MODEL_API
#endif

#define MODEL_USE_SCRATCH
#define MODEL_MAX_SCRATCH_BUFFERS 16

#define MODEL_FILE_MAGIC_GGJT 0x67676a74u  // 'ggjt'
#define MODEL_FILE_MAGIC_GGLA 0x67676c61u  // 'ggla'
#define MODEL_FILE_MAGIC_GGMF 0x67676d66u  // 'ggmf'
#define MODEL_FILE_MAGIC_NE 0x67676d6cu    // 'ne'
#define MODEL_FILE_MAGIC_GGSN 0x6767736eu  // 'ggsn'

#define MODEL_FILE_VERSION 3
#define MODEL_FILE_MAGIC MODEL_FILE_MAGIC_GGJT
#define MODEL_FILE_MAGIC_UNVERSIONED MODEL_FILE_MAGIC_NE
#define MODEL_SESSION_MAGIC MODEL_FILE_MAGIC_GGSN
#define MODEL_SESSION_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif
// available model models
enum e_model {
  MODEL_UNKNOWN,
  MODEL_7B,
  MODEL_13B,
  MODEL_30B,
  MODEL_65B,
};

enum name_model { MODEL_LLAMA };

static const size_t MB = 1024 * 1024;

// computed for n_ctx == 2048
// TODO: dynamically determine these sizes
//       needs modifications in ne

static const std::map<e_model, size_t>& MEM_REQ_SCRATCH0() {
  static std::map<e_model, size_t> k_sizes = {
      {MODEL_7B, 512ull * MB},
      {MODEL_13B, 512ull * MB},
      {MODEL_30B, 512ull * MB},
      {MODEL_65B, 1024ull * MB},
  };
  return k_sizes;
}

static const std::map<e_model, size_t>& MEM_REQ_SCRATCH1() {
  static std::map<e_model, size_t> k_sizes = {
      {MODEL_7B, 512ull * MB},
      {MODEL_13B, 512ull * MB},
      {MODEL_30B, 512ull * MB},
      {MODEL_65B, 1024ull * MB},
  };
  return k_sizes;
}

// 2*n_embd*n_ctx*n_layer*sizeof(float16)
static const std::map<e_model, size_t>& MEM_REQ_KV_SELF() {
  static std::map<e_model, size_t> k_sizes = {
      {MODEL_7B, 1026ull * MB},
      {MODEL_13B, 1608ull * MB},
      {MODEL_30B, 3124ull * MB},
      {MODEL_65B, 5120ull * MB},
  };
  return k_sizes;
}

// this is mostly needed for temporary mul_mat buffers to dequantize the data
// not actually needed if BLAS is disabled
static const std::map<e_model, size_t>& MEM_REQ_EVAL() {
  static std::map<e_model, size_t> k_sizes = {
      {MODEL_7B, 768ull * MB},
      {MODEL_13B, 1024ull * MB},
      {MODEL_30B, 1280ull * MB},
      {MODEL_65B, 1536ull * MB},
  };
  return k_sizes;
}

enum model_file_version {
  MODEL_FILE_VERSION_NE,
  MODEL_FILE_VERSION_GGMF_V1,  // added version field and scores in vocab
  MODEL_FILE_VERSION_GGJT_V1,  // added padding
  MODEL_FILE_VERSION_GGJT_V2,  // changed quantization format
  MODEL_FILE_VERSION_GGJT_V3,  // changed Q4 and Q8 quantization format
};

//
// C interface
//
// TODO: show sample usage
//

// default hparams (LLaMA 7B)
struct model_hparams {
  uint32_t n_vocab = 32000;
  uint32_t n_ctx = 512;  // this is provided as user input?
  uint32_t n_embd = 4096;
  uint32_t n_mult = 256;
  uint32_t n_head = 32;
  uint32_t n_layer = 32;
  uint32_t n_rot = 64;
  enum ne_ftype ftype = NE_FTYPE_MOSTLY_F16;

  bool operator!=(const model_hparams& other) const {
    return static_cast<bool>(memcmp(this, &other, sizeof(model_hparams)));
  }
};

struct model_layer {
  // normalization
  struct ne_tensor* attention_norm;

  // attention
  struct ne_tensor* wq;
  struct ne_tensor* wk;
  struct ne_tensor* wv;
  struct ne_tensor* wo;

  // normalization
  struct ne_tensor* ffn_norm;

  // ff
  struct ne_tensor* w1;
  struct ne_tensor* w2;
  struct ne_tensor* w3;
};

struct model_kv_cache {
  struct ne_tensor* k;
  struct ne_tensor* v;

  struct ne_context* ctx = NULL;

  model_ctx_buffer buf;

  int n;  // number of tokens currently in the cache

  ~model_kv_cache() {
    if (ctx) {
      ne_free(ctx);
    }
  }
};

struct model_model {
  e_model type = MODEL_UNKNOWN;

  model_hparams hparams;

  struct ne_tensor* tok_embeddings;

  struct ne_tensor* norm;
  struct ne_tensor* output;

  std::vector<model_layer> layers;

  // context
  struct ne_context* ctx = NULL;

  // key + value cache for the self attention
  // TODO: move to model_state
  struct model_kv_cache kv_self;

  // the model memory buffer
  model_ctx_buffer buf;

  // model memory mapped file
  std::unique_ptr<model_mmap> mapping;

  // objects representing data potentially being locked in memory
  model_mlock mlock_buf;
  model_mlock mlock_mmap;

  // for quantize-stats only
  std::vector<std::pair<std::string, struct ne_tensor*>> tensors_by_name;

  ~model_model() {
    if (ctx) {
      ne_free(ctx);
    }
  }
};

struct model_vocab {
  using id = int32_t;
  using token = std::string;

  struct token_score {
    token tok;
    float score;
  };

  std::unordered_map<token, id> token_to_id;
  std::vector<token_score> id_to_token;
};

struct model_context {
  std::mt19937 rng;

  int64_t t_load_us = 0;
  int64_t t_start_us = 0;
  bool has_evaluated_once = false;

  int64_t t_sample_us = 0;
  int64_t t_eval_us = 0;
  int64_t t_p_eval_us = 0;
  std::vector<int64_t> eval_times;

  int32_t n_sample = 0;  // number of tokens sampled
  int32_t n_eval = 0;    // number of eval calls
  int32_t n_p_eval = 0;  // number of tokens in eval calls for the prompt (with batch size > 1)

  model_model model;
  model_vocab vocab;

  size_t mem_per_token = 0;

  // decode output (2-dimensional array: [n_tokens][n_vocab])
  std::vector<float> logits;
  bool logits_all = false;

  // input embedding (1-dimensional array: [n_embd])
  std::vector<float> embedding;

  // memory buffers used to evaluate the model
  // TODO: move in model_state
  model_ctx_buffer buf_compute;
  model_ctx_buffer buf_scratch[MODEL_MAX_SCRATCH_BUFFERS];

  int buf_last = 0;
  size_t buf_max_size[MODEL_MAX_SCRATCH_BUFFERS] = {0};

  void use_buf(struct ne_context* ctx, int i) {
#if defined(MODEL_USE_SCRATCH)
    size_t last_size = 0;

    if (i == -1) {
      last_size = ne_set_scratch(ctx, {
                                          0,
                                          0,
                                          nullptr,
                                      });
    } else {
      auto& buf = buf_scratch[i];
      last_size = ne_set_scratch(ctx, {
                                          0,
                                          buf.size,
                                          buf.addr,
                                      });
    }

    if (buf_last >= 0) {
      buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
    }

    buf_last = i;
#else
    (void)i;
    (void)ctx;
#endif
  }

  size_t get_buf_max_mem(int i) const {
#if defined(MODEL_USE_SCRATCH)
    return buf_max_size[i];
#else
    (void)i;
    return 0;
#endif
  }
};

typedef int model_token;

typedef struct model_token_data {
  model_token id;  // token id
  float logit;     // log-odds of the token
  float p;         // probability of the token
} model_token_data;

typedef struct model_token_data_array {
  model_token_data* data;
  size_t size;
  bool sorted;
} model_token_data_array;

typedef void (*model_progress_callback)(float progress, void* ctx);

struct model_context_params {
  int n_ctx;         // text context
  int n_gpu_layers;  // number of layers to store in VRAM
  int seed;          // RNG seed, -1 for random

  bool f16_kv;      // use fp16 for KV cache
  bool logits_all;  // the model_eval() call computes all logits, not just the last one
  bool vocab_only;  // only load the vocabulary, no weights
  bool use_mmap;    // use mmap if possible
  bool use_mlock;   // force system to keep model in RAM
  bool embedding;   // embedding mode only

  // called with a progress value between 0 and 1, pass NULL to disable
  model_progress_callback progress_callback;
  // context pointer passed to the progress callback
  void* progress_callback_user_data;
};

#ifdef __cplusplus
}
#endif

// Internal API to be implemented by model.cpp and used by tests/benchmarks only
#ifdef MODEL_API_INTERNAL

#include <vector>
#include <string>
struct ne_tensor;

std::vector<std::pair<std::string, struct ne_tensor*>>& model_internal_get_tensor_map(struct model_context* ctx);

#endif

#endif  // MODEL_TYPES_H
