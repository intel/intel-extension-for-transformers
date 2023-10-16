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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstring>
#include <ctime>
#include <fstream>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "core/ne_layers.h"
#include "models/model_utils/util.h"

#define MODEL_MAX_NORM 4
#define MODEL_MAX_ATTN 8
#define MODEL_MAX_FFN 6
#define MODEL_MAX_OTHERS 7

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

enum model_archs {
  MODEL_UNKNOWN,
  MODEL_LLAMA,
  MODEL_GPTJ,
  MODEL_MPT,
  MODEL_GPTNEOX,
  MODEL_STARCODER,
  MODEL_FALCON,
  MODEL_OPT,
  MODEL_BLOOM,
  MODEL_BAICHUAN,
  MODEL_CHATGLM2,
  MODEL_CHATGLM
};

static const size_t MB = 1024 * 1024;

typedef enum KV_MEM_TYPE {  // Memory kv data type
  KV_MEM_TYPE_AUTO,         // Try with jblas flash attn managed format; fall back to fp16 if failed
  KV_MEM_TYPE_F16,          // Use F16 for memory kv
  KV_MEM_TYPE_F32,          // Use F32 for memory kv
} KV_MEM_TYPE;

struct model_scratch {
  size_t scratch0;
  size_t scratch1;
  size_t eval;
};

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
  uint32_t n_embd = 4096;
  uint32_t n_mult = 256;
  uint32_t n_head = 32;
  uint32_t n_head_kv = 0;  //  MQA, multi-query attention (default =0 means no MQA)
  uint32_t n_layer = 32;
  uint32_t n_rot = 64;
  enum ne_ftype ftype = NE_FTYPE_MOSTLY_F16;
  int32_t max_seq_len = 0;            // for mpt
  float alibi_bias_max = 0;           // for mpt
  float clip_qkv = 0;                 // for mpt
  int32_t par_res = 1;                // for neox 1 = true, 0 = false
  uint32_t word_embed_proj_dim = 0;   // for opt
  bool do_layer_norm_before = false;  // for opt

  // ChatGLM-2
  int32_t multi_query_group_num = 0;
  int32_t ffn_hidden_size = 0;

  // ChatGLM-1
  int32_t inner_hidden_size = 0;

  bool operator!=(const model_hparams& other) const {
    return static_cast<bool>(memcmp(this, &other, sizeof(model_hparams)));
  }
};

struct model_layer {
  // normalization
  struct ne_tensor* norm[MODEL_MAX_NORM];

  // attention
  struct ne_tensor* attn[MODEL_MAX_ATTN];

  // ff
  struct ne_tensor* ffn[MODEL_MAX_FFN];

  struct ne_tensor* k_cache;
  struct ne_tensor* v_cache;
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

struct model_struct {
  model_archs arch;

  model_hparams hparams;
  model_scratch scratchs;

  struct ne_tensor* others[MODEL_MAX_OTHERS];
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

  ~model_struct() {
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
  id bos_token_id = -1;
  id eos_token_id = -1;
  id pad_token_id = -1;
  id sep_token_id = -1;
};

// reference: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
struct generation_config {
  uint32_t max_new_tokens;  // n_predict there
  uint32_t min_new_tokens = 0;
  // Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
  // the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
  // likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
  // `length_penalty` < 0.0 encourages shorter sequences. (default = 1.0)
  float length_penalty = 1.0f;
  bool do_early_stopping = false;
};

class beam_search_kv_cache_reorder;  //  forward declaration
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

  int32_t n_ctx = 512;  // number of tokens to keep as context
  // start size to keep; n_ctx = n_keep + n_recent; refer the streaming-llm paper for details:
  // https://arxiv.org/abs/2309.17453
  int32_t n_keep = 0;

  model_struct model;
  model_vocab vocab;
  int batch_size = 1;
  bool beam_search = false;
  bool support_jblas_kv = false;  // whether the model graph supports jblas-kvcache
  int beam_size = 1;
  int kv_n_ctx_block = 1;
  generation_config generation_conf;
  std::shared_ptr<beam_search_kv_cache_reorder> bs_kv_reorder;
  std::vector<std::vector<std::string>> tensors_name;

  size_t mem_per_token = 0;

  // decode output (3-dimensional array: [batch_size] [n_tokens] [n_vocab])
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

typedef model_vocab::id model_token;

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
  model_archs arch;  // arch of models (GPT-J, LLAMA)
  int n_ctx;         // text context
  // start size to keep; n_ctx = n_keep + n_recent; refer the streaming-llm paper for details:
  // https://arxiv.org/abs/2309.17453
  int n_keep;
  int n_gpu_layers;     // number of layers to store in VRAM
  int seed;             // RNG seed, -1 for random
  KV_MEM_TYPE kv_type;  // KV cache type specification
  bool logits_all;      // the model_eval() call computes all logits, not just the last one
  bool vocab_only;      // only load the vocabulary, no weights
  bool use_mmap;        // use mmap if possible
  bool use_mlock;       // force system to keep model in RAM
  bool embedding;       // embedding mode only
  int batch_size;       // batch_size of prompt
  bool beam_search;     // beam search or not
  int beam_size;        // number of beams for beam search

  // called with a progress value between 0 and 1, pass NULL to disable
  model_progress_callback progress_callback;
  // context pointer passed to the progress callback
  void* progress_callback_user_data;
};

class model_name_to_arch {
 public:
  static model_name_to_arch& init() {
    static model_name_to_arch ins;
    return ins;
  }

  void valid_options() {
    for (auto pair : name2arch_) {
      printf("%s, ", pair.first.c_str());
    }
    printf("\n");
  }

  model_archs find(const std::string& name) {
    auto it = name2arch_.find(name);
    if (it == name2arch_.end()) {
      printf("%s is not a valid model name, supported model names are: ", name.c_str());
      valid_options();
      return MODEL_UNKNOWN;
    } else {
      return name2arch_.at(name);
    }
  }

 private:
  model_name_to_arch() {}
  // update this table if has new cpp model
  std::unordered_map<std::string, model_archs> name2arch_ = {
      {"unknown", MODEL_UNKNOWN}, {"llama", MODEL_LLAMA},      {"gptj", MODEL_GPTJ},
      {"mpt", MODEL_MPT},         {"opt", MODEL_OPT},          {"gptneox", MODEL_GPTNEOX},
      {"dolly", MODEL_GPTNEOX},   {"polyglot", MODEL_GPTNEOX}, {"starcoder", MODEL_STARCODER},
      {"falcon", MODEL_FALCON},   {"bloom", MODEL_BLOOM},      {"chatglm2", MODEL_CHATGLM2},
      {"chatglm", MODEL_CHATGLM}, {"baichuan", MODEL_BAICHUAN}};
};

#ifdef __cplusplus
}
#endif

// Internal API to be implemented by model.cpp and used by tests/benchmarks only
#ifdef MODEL_API_INTERNAL

#include <string>
#include <vector>
struct ne_tensor;

std::vector<std::pair<std::string, struct ne_tensor*>>& model_internal_get_tensor_map(struct model_context* ctx);

#endif

#endif  // MODEL_TYPES_H
