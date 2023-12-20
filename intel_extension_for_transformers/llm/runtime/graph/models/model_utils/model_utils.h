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
#ifndef MODEL_H
#define MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <limits>

#include "application/common.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/quant_config.h"

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

void model_load_internal(const std::string& fname, model_archs arch, model_context* ctx, int n_gpu_layers,
                         bool use_mmap, bool use_mlock, bool vocab_only, model_progress_callback progress_callback,
                         void* progress_callback_user_data);

#ifdef __cplusplus
extern "C" {
#endif

MODEL_API struct model_context_params model_context_default_params();

MODEL_API bool model_mmap_supported();
MODEL_API bool model_mlock_supported();

// TODO: not great API - very likely to change
// Initialize the model + ne backend
// Call once at the start of the program
MODEL_API void model_init_backend();

MODEL_API int64_t model_time_us();

// Various functions for loading a ne model model.
// Allocate (almost) all memory needed for the model.
// Return NULL on failure
MODEL_API struct model_context* model_init_from_file(const char* path_model, struct model_context_params params);

// Frees all allocated memory
MODEL_API void model_free(struct model_context* ctx);

// TODO: not great API - very likely to change
// Returns 0 on success
// param - from args
// quant_layer - depends on each model's config
MODEL_API int model_quantize(const quant_params& param, std::shared_ptr<quant_layer_base> quant_layer);
size_t jblas_qpack(const int8_t* src_w, const float* src_scales, const int8_t* src_zps, void* dstpr,
                   const quant_params_internal params, int nthread, int n, int k, int* g_idx);
size_t jblas_quantize(const float* f32ptr, void* dstpr, const quant_params_internal params, int nthread, size_t n,
                      size_t k);
// Apply a LoRA adapter to a loaded model
// path_base_model is the path to a higher quality model to use as a base for
// the layers modified by the adapter. Can be NULL to use the current loaded
// model. The model needs to be reloaded before applying a new adapter,
// otherwise the adapter will be applied on top of the previous one Returns 0 on
// success
MODEL_API int model_apply_lora_from_file(struct model_context* ctx, const char* path_lora, const char* path_base_model,
                                         int n_threads);

// Returns the number of tokens in the KV cache
MODEL_API int model_get_kv_cache_token_count(const struct model_context* ctx);

// Sets the current rng seed.
MODEL_API void model_set_rng_seed(struct model_context* ctx, int seed);

// Returns the maximum size in bytes of the state (rng, logits, embedding
// and kv_cache) - will often be smaller after compacting tokens
MODEL_API size_t model_get_state_size(const struct model_context* ctx);

// Copies the state to the specified destination address.
// Destination needs to have allocated enough memory.
// Returns the number of bytes copied
MODEL_API size_t model_copy_state_data(struct model_context* ctx, uint8_t* dst);

// Set the state reading from the specified address
// Returns the number of bytes read
MODEL_API size_t model_set_state_data(struct model_context* ctx, uint8_t* src);

// Save/load session file
MODEL_API bool model_load_session_file(struct model_context* ctx, const char* path_session, model_token* tokens_out,
                                       size_t n_token_capacity, size_t* n_token_total_out);
MODEL_API bool model_save_session_file(struct model_context* ctx, const char* path_session, const model_token* tokens,
                                       size_t n_token_total);

// Run the model inference to obtain the logits and probabilities for the next
// model_input has some necessary members for inference (more details please see model_types.h):
// token. tokens + n_tokens is the provided batch of new tokens to process
// n_past is the offset to which the kv is cached to
// n_total is the number of tokens evaluated in previous eval calls
// Returns 0 on success
MODEL_API int model_eval(struct model_context* ctx, const model_input* inputs, const int n_input, int n_threads);

// Convert the provided text into tokens.
// The tokens pointer must be large enough to hold the resulting tokens.
// Returns the number of tokens on success, no more than n_max_tokens
// Returns a negative number on failure - the number of tokens that would have
// been returned
// TODO: not sure if correct
MODEL_API int model_tokenize(struct model_context* ctx, const char* text, model_token* tokens, int n_max_tokens,
                             bool add_bos);

MODEL_API int model_n_vocab(const struct model_context* ctx);
MODEL_API int model_n_ctx(const struct model_context* ctx);
MODEL_API int model_n_embd(const struct model_context* ctx);

// Token logits obtained from the last call to model_eval()
// The logits for the last token are stored in the last row
// Can be mutated in order to change the probabilities of the next token
// Rows: n_tokens
// Cols: n_vocab
MODEL_API float* model_get_logits(struct model_context* ctx);

// Get the embeddings for the input
// shape: [n_embd] (1-dimensional)
MODEL_API float* model_get_embeddings(struct model_context* ctx);

// Token Id -> String. Uses the vocabulary in the provided context
MODEL_API const char* model_token_to_str(const struct model_context* ctx, model_token token);

// Special tokens
MODEL_API model_token model_token_nl();

// Sampling functions

/// @details Repetition penalty described in CTRL academic paper
/// https://arxiv.org/abs/1909.05858, with negative logit fix.
MODEL_API void model_sample_repetition_penalty(struct model_context* ctx, model_token_data_array* candidates,
                                               const model_token* last_tokens, size_t last_tokens_size, float penalty);

/// @details Frequency and presence penalties described in OpenAI API
/// https://platform.openai.com/docs/api-reference/parameter-details.
MODEL_API void model_sample_frequency_and_presence_penalties(struct model_context* ctx,
                                                             model_token_data_array* candidates,
                                                             const model_token* last_tokens, size_t last_tokens_size,
                                                             float alpha_frequency, float alpha_presence);

/// @details Sorts candidate tokens by their logits in descending order and
/// calculate probabilities based on logits.
MODEL_API void model_sample_softmax(struct model_context* ctx, model_token_data_array* candidates);

/// @details Top-K sampling described in academic paper "The Curious Case of
/// Neural Text Degeneration" https://arxiv.org/abs/1904.09751
MODEL_API void model_sample_top_k(struct model_context* ctx, model_token_data_array* candidates, int k,
                                  size_t min_keep);

/// @details Nucleus sampling described in academic paper "The Curious Case of
/// Neural Text Degeneration" https://arxiv.org/abs/1904.09751
MODEL_API void model_sample_top_p(struct model_context* ctx, model_token_data_array* candidates, float p,
                                  size_t min_keep);

MODEL_API model_token model_sample_top_k_top_p(struct model_context* ctx, const int n_logits, const float* logits,
                                               int top_k, double top_p, double temp);

/// @details Tail Free Sampling described in
/// https://www.trentonbricken.com/Tail-Free-Sampling/.
MODEL_API void model_sample_tail_free(struct model_context* ctx, model_token_data_array* candidates, float z,
                                      size_t min_keep);

/// @details Locally Typical Sampling implementation described in the paper
/// https://arxiv.org/abs/2202.00666.
MODEL_API void model_sample_typical(struct model_context* ctx, model_token_data_array* candidates, float p,
                                    size_t min_keep);
MODEL_API void model_sample_temperature(struct model_context* ctx, model_token_data_array* candidates, float temp);

/// @details Mirostat 1.0 algorithm described in the paper
/// https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
/// @param candidates A vector of `model_token_data` containing the candidate
/// tokens, their probabilities (p), and log-odds (logit) for the current
/// position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve
/// for the generated text. A higher value corresponds to more surprising or
/// less predictable text, while a lower value corresponds to less surprising or
/// more predictable text.
/// @param eta The learning rate used to update `mu` based on the error between
/// the target and observed surprisal of the sampled word. A larger learning
/// rate will cause `mu` to be updated more quickly, while a smaller learning
/// rate will result in slower updates.
/// @param m The number of tokens considered in the estimation of `s_hat`. This
/// is an arbitrary value that is used to calculate `s_hat`, which in turn helps
/// to calculate the value of `k`. In the paper, they use `m = 100`, but you can
/// experiment with different values to see how it affects the performance of
/// the algorithm.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the
/// target cross-entropy (`2 * tau`) and is updated in the algorithm based on
/// the error between the target and observed surprisal.
MODEL_API model_token model_sample_token_mirostat(struct model_context* ctx, model_token_data_array* candidates,
                                                  float tau, float eta, int m, float* mu);

/// @details Mirostat 2.0 algorithm described in the paper
/// https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
/// @param candidates A vector of `model_token_data` containing the candidate
/// tokens, their probabilities (p), and log-odds (logit) for the current
/// position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve
/// for the generated text. A higher value corresponds to more surprising or
/// less predictable text, while a lower value corresponds to less surprising or
/// more predictable text.
/// @param eta The learning rate used to update `mu` based on the error between
/// the target and observed surprisal of the sampled word. A larger learning
/// rate will cause `mu` to be updated more quickly, while a smaller learning
/// rate will result in slower updates.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the
/// target cross-entropy (`2 * tau`) and is updated in the algorithm based on
/// the error between the target and observed surprisal.
MODEL_API model_token model_sample_token_mirostat_v2(struct model_context* ctx, model_token_data_array* candidates,
                                                     float tau, float eta, float* mu);

/// @details Selects the token with the highest probability.
MODEL_API model_token model_sample_token_greedy(struct model_context* ctx, model_token_data_array* candidates);

/// @details Randomly selects a token from the candidates based on their
/// probabilities.
MODEL_API model_token model_sample_token(struct model_context* ctx, model_token_data_array* candidates);

// Performance information
MODEL_API void model_print_timings(struct model_context* ctx);
MODEL_API void model_reset_timings(struct model_context* ctx);

// Print system information
MODEL_API const char* model_print_system_info(void);

#ifdef __cplusplus
}
#endif

/* kv cache utils */
// kv cache both stores permuted tensor
// k shape is [head_dim, N, n_head]
// v shape is [N, head_dim, n_head] or [N, n_embd]
/* kv cache utils */

// copy consecutive tokens from one seq to another
MODEL_API void model_kv_cache_seq_cpy(struct model_context* ctx, const model_seq_id& seq_id_src,
                                      const model_seq_id& seq_id_dst, const model_pos& p0, const model_pos& p1);

// concat several seqs into a continuous batch from kv cache
MODEL_API ne_tensor* model_kv_cache_seq_concat(struct ne_cgraph* cgraph, struct model_context* moctx,
                                               struct ne_context* nectx, const int64_t& ne0, const int64_t& ne1,
                                               const int64_t& ne2, const int64_t& ne3,
                                               const std::vector<int>& block_ids, const int& layer_idx,
                                               const bool& concat_k = true);

/*  beam search utils  */
#define NEG_INF -std::numeric_limits<float>::max()

typedef struct beam_next_token {
  model_token id = -1;  // token id
  float score = 0.0f;   // score of the token
  int beam_idx = -1;    // token in which beam (-1 means unknown)
} beam_next_token;

struct beam {
  const model_context* ctx = nullptr;
  std::vector<model_token> token_ids;
  // Cumulative beam score (log-softmax here)
  float score = 0.0f;
  // record related indices
  // 0 - request_bs-1
  int request_idx = -1;
  // 0 - num_beams-1
  int beam_idx = -1;
  // if stop generation (append new token_id)
  bool done = false;

  // end-of-text
  const bool eos() const { return !token_ids.empty() && token_ids.back() == ctx->vocab.eos_token_id; }

  void print() const {
    printf("length: %ld, score: %12.6f, eos: %d, request_idx: %d, beam_idx: %d, done: %d, tokens:\n", token_ids.size(),
           score, eos(), request_idx, beam_idx, done);
    for (const auto& id : token_ids) {
      printf("%d: %s, ", id, model_token_to_str(ctx, id));
    }
    printf("\n");
  }

  void clear() {
    token_ids.clear();
    score = 0.0f;
    request_idx = -1;
    beam_idx = -1;
    done = false;
  }
};

struct beam_hypotheses {
  const model_context* const ctx = nullptr;
  const int num_beams;
  const float length_penalty = 1.0f;
  const bool early_stopping = false;
  std::vector<beam> beams;

  beam_hypotheses(model_context* lctx)
      : ctx(lctx),
        num_beams(lctx->beam_size),
        length_penalty(lctx->generation_conf.length_penalty),
        early_stopping(lctx->generation_conf.do_early_stopping) {
    beams.reserve(lctx->beam_size);
  }

  int len() { return beams.size(); }

  void add(beam b, const uint32_t& n_prompt_tokens) {
    auto comp = [](const beam& a, const beam& b) { return a.score > b.score; };
    uint32_t cur_len = b.eos() ? b.token_ids.size() - 1 : b.token_ids.size();
    float score = b.score / std::pow(cur_len + n_prompt_tokens, length_penalty);
#ifdef NE_BEAM_SEARCH_VERBOSE_ON
    printf("add beam hypos: \n");
    b.print();
    printf("origin_score: %12.6f, new_score: %12.6f, sentence_len: %d \n", b.score, score, cur_len + n_prompt_tokens);
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n");
#endif
    b.score = score;
    if (beams.size() < num_beams) {
      beams.push_back(std::move(b));
      if (beams.size() == num_beams) {
        std::make_heap(beams.begin(), beams.end(), comp);
      }
    } else {
      MODEL_ASSERT(beams.size() == num_beams);
      if (beams.front().score > b.score) {
        return;
      }
      std::pop_heap(beams.begin(), beams.end(), comp);
      beams.back() = b;
      std::push_heap(beams.begin(), beams.end(), comp);
    }
  }

  const bool is_done() const {
    if (beams.size() < num_beams) {
      return false;
    }
    // stop as soon as at least `num_beams` hypotheses are finished
    if (early_stopping) {
      return true;
    }
    return false;
  }

  const beam& top1() const {
    auto const by_score = [](beam const& a, beam const& b) { return a.score < b.score; };
    return *std::max_element(beams.begin(), beams.end(), by_score);
  }

  void clear() { beams.clear(); }
};

struct logits_info;

class logits_processor {
 public:
  explicit logits_processor(model_context* lctx) : ctx(lctx), min_new_tokens(lctx->generation_conf.min_new_tokens) {}
  ~logits_processor() {}

  void process(const std::vector<uint32_t>& cur_lens, const model_vocab::id& eos_token_id);
  void min_new_tokens_logits_process(const std::vector<uint32_t>& cur_lens, const model_vocab::id& eos_token_id);

 private:
  model_context* ctx = nullptr;
  const uint32_t min_new_tokens;
};

// base kv_cache reorder class for beam search
// assume k shape = [head_dim, head_k, seq_len], v shape = [seq_len, head_dim, head_v] in GPT-liked Decoder only model
// TODO: but k shape may be [head_dim, seq_len, head_k] optimized layouts for original k permute time reduction
// if a model use different kv cache layout in its own eval process, please inherit this class and override the
// `update` virtual function
class beam_search_kv_cache_reorder {
 public:
  explicit beam_search_kv_cache_reorder(model_context* lctx)
      : ctx(lctx), n_ctx(lctx->n_ctx), kv_n_ctx_block(lctx->kv_n_ctx_block) {}
  virtual ~beam_search_kv_cache_reorder() {}

  virtual void update(const std::vector<uint32_t>& n_past, const std::vector<uint32_t>& n_prompt_tokens,
                      const std::vector<int> request_running_indices,
                      const std::vector<std::tuple<int, int>>& kv_reorder_indices = {},
                      const std::vector<beam>& next_beams = {});

 protected:
  model_context* ctx = nullptr;
  const uint32_t n_ctx;
  const uint32_t kv_n_ctx_block;
};

class beam_search_flow {
 public:
  beam_search_flow(model_context* lctx, const int batch_size = 1)
      : ctx(lctx), beam_size(lctx->beam_size), request_bs(batch_size), lp(logits_processor(lctx)) {
    cur_beams.resize(batch_size * beam_size);
    next_beams.resize(batch_size * beam_size);
    for (int i = 0; i < batch_size; ++i) {
      beam_hypos.push_back(std::move(beam_hypotheses(lctx)));
    }
    response.resize(batch_size);
    requests_done.assign(batch_size, false);
    request_running_indices.reserve(batch_size);
    next_done_request_ids.reserve(batch_size);
    n_tokens.reserve(batch_size);
    n_past.reserve(batch_size);
    n_prompt_tokens.reserve(batch_size);
    n_total.reserve(batch_size);
    padding_side.reserve(batch_size);
    n_padding.reserve(batch_size);
  }
  ~beam_search_flow() {}

  // public interface
  // static batching (padding inputs or batch = 1)
  const std::vector<std::vector<model_token>>& loop(const std::vector<model_input>& inputs, const int& n_threads);
  // continuous batching (scheduling from the outside)
  const bool step_prefill(const model_input& input);
  const bool step_decoding();
  const std::vector<int> request_done_ids();
  const std::vector<std::vector<model_token>> request_done_reponse();

 private:
  std::vector<beam_next_token> beam_top_k_next_tokens(model_context* ctx, const std::vector<float>& beams_score,
                                                      const std::vector<int>& num_beams,
                                                      const std::vector<int> beam_indices, const int& sample_scale = 2,
                                                      const int& dim = -1);
  void fill_next_beams_by_top_scores();
  std::vector<std::tuple<int, int>> update_kv_cache_reorder_indices();
  void update_status();
  const beam& finalize(const int& request_idx);

  model_context* ctx = nullptr;
  const int beam_size;
  const int request_bs;  // could be the max bs in continuous batching mechanism
  std::vector<beam> cur_beams;
  std::vector<beam> next_beams;
  std::vector<beam_hypotheses> beam_hypos;
  std::vector<bool> requests_done;
  std::vector<int> request_running_indices;
  std::vector<int> next_done_request_ids;
  std::vector<uint32_t> n_tokens;
  std::vector<uint32_t> n_past;
  std::vector<uint32_t> n_prompt_tokens;
  std::vector<uint32_t> n_total;
  std::vector<int> padding_side;
  std::vector<uint32_t> n_padding;
  int num_threads = 4;  // default by 4
  logits_processor lp;
  std::shared_ptr<beam_search_kv_cache_reorder> kv_reorder;
  std::vector<std::vector<model_token>> response;
};

// static batching generation
MODEL_API std::vector<std::vector<model_token>> beam_search(model_context* lctx, const int& n_predict,
                                                            const std::vector<model_input>& inputs,
                                                            const int& n_threads);

// Internal API to be implemented by model.cpp and used by tests/benchmarks only
#ifdef MODEL_API_INTERNAL

#include <string>
#include <vector>
struct ne_tensor;

std::vector<std::pair<std::string, struct ne_tensor*>>& model_internal_get_tensor_map(struct model_context* ctx);

#endif

#endif  // MODEL_H
