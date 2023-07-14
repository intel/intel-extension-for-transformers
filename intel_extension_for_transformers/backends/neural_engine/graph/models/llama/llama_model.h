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

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "model_types.h"

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
// nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(), else the number given
MODEL_API int model_model_quantize(const char* fname_inp, const char* fname_out, ne_ftype ftype, int nthread);

// Apply a LoRA adapter to a loaded model
// path_base_model is the path to a higher quality model to use as a base for
// the layers modified by the adapter. Can be NULL to use the current loaded model.
// The model needs to be reloaded before applying a new adapter, otherwise the adapter
// will be applied on top of the previous one
// Returns 0 on success
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
                                       size_t n_token_capacity, size_t* n_token_count_out);
MODEL_API bool model_save_session_file(struct model_context* ctx, const char* path_session, const model_token* tokens,
                                       size_t n_token_count);

// Run the model inference to obtain the logits and probabilities for the next token.
// tokens + n_tokens is the provided batch of new tokens to process
// n_past is the number of tokens to use from previous eval calls
// Returns 0 on success
MODEL_API int model_eval(struct model_context* ctx, const model_token* tokens, int n_tokens, int n_past, int n_threads);

// Convert the provided text into tokens.
// The tokens pointer must be large enough to hold the resulting tokens.
// Returns the number of tokens on success, no more than n_max_tokens
// Returns a negative number on failure - the number of tokens that would have been returned
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
MODEL_API model_token model_token_bos();
MODEL_API model_token model_token_eos();
MODEL_API model_token model_token_nl();

// Sampling functions

/// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit
/// fix.
MODEL_API void model_sample_repetition_penalty(struct model_context* ctx, model_token_data_array* candidates,
                                               const model_token* last_tokens, size_t last_tokens_size, float penalty);

/// @details Frequency and presence penalties described in OpenAI API
/// https://platform.openai.com/docs/api-reference/parameter-details.
MODEL_API void model_sample_frequency_and_presence_penalties(struct model_context* ctx,
                                                             model_token_data_array* candidates,
                                                             const model_token* last_tokens, size_t last_tokens_size,
                                                             float alpha_frequency, float alpha_presence);

/// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
MODEL_API void model_sample_softmax(struct model_context* ctx, model_token_data_array* candidates);

/// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration"
/// https://arxiv.org/abs/1904.09751
MODEL_API void model_sample_top_k(struct model_context* ctx, model_token_data_array* candidates, int k,
                                  size_t min_keep);

/// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration"
/// https://arxiv.org/abs/1904.09751
MODEL_API void model_sample_top_p(struct model_context* ctx, model_token_data_array* candidates, float p,
                                  size_t min_keep);

/// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
MODEL_API void model_sample_tail_free(struct model_context* ctx, model_token_data_array* candidates, float z,
                                      size_t min_keep);

/// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
MODEL_API void model_sample_typical(struct model_context* ctx, model_token_data_array* candidates, float p,
                                    size_t min_keep);
MODEL_API void model_sample_temperature(struct model_context* ctx, model_token_data_array* candidates, float temp);

/// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of
/// words.
/// @param candidates A vector of `model_token_data` containing the candidate tokens, their probabilities (p), and
/// log-odds (logit) for the current position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value
/// corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more
/// predictable text.
/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the
/// sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will
/// result in slower updates.
/// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to
/// calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can
/// experiment with different values to see how it affects the performance of the algorithm.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is
/// updated in the algorithm based on the error between the target and observed surprisal.
MODEL_API model_token model_sample_token_mirostat(struct model_context* ctx, model_token_data_array* candidates,
                                                  float tau, float eta, int m, float* mu);

/// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of
/// words.
/// @param candidates A vector of `model_token_data` containing the candidate tokens, their probabilities (p), and
/// log-odds (logit) for the current position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value
/// corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more
/// predictable text.
/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the
/// sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will
/// result in slower updates.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is
/// updated in the algorithm based on the error between the target and observed surprisal.
MODEL_API model_token model_sample_token_mirostat_v2(struct model_context* ctx, model_token_data_array* candidates,
                                                     float tau, float eta, float* mu);

/// @details Selects the token with the highest probability.
MODEL_API model_token model_sample_token_greedy(struct model_context* ctx, model_token_data_array* candidates);

/// @details Randomly selects a token from the candidates based on their probabilities.
MODEL_API model_token model_sample_token(struct model_context* ctx, model_token_data_array* candidates);

// Performance information
MODEL_API void model_print_timings(struct model_context* ctx);
MODEL_API void model_reset_timings(struct model_context* ctx);

// Print system information
MODEL_API const char* model_print_system_info(void);

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

#endif  // MODEL_H
