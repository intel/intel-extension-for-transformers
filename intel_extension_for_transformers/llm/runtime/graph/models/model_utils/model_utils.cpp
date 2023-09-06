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
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

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

#include "application/common.h"
#include "core/layers/jblas_common.hpp"
#include "models/model_utils/model_files.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"
#include "models/models.h"

//
// kv cache
//

static bool kv_cache_init(const struct model_hparams& hparams, struct model_kv_cache& cache, ne_type wtype, int n_ctx) {
  const int n_embd = hparams.n_embd;
  const int head_dim = n_embd / hparams.n_head;
  const int n_head_kv = hparams.n_head_kv;
  const int n_layer = hparams.n_layer;

  const int64_t n_mem = n_layer * n_ctx;
  const int64_t n_elements = n_head_kv > 0 ? n_head_kv * head_dim * n_mem : n_embd * n_mem;

  cache.buf.resize(2u * n_elements * ne_type_size(wtype) + 2u * MB);

  struct ne_init_params params;
  params.mem_size = cache.buf.size;
  params.mem_buffer = cache.buf.addr;
  params.no_alloc = false;

  cache.ctx = ne_init(params);

  if (!cache.ctx) {
    fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
    return false;
  }

  cache.k = ne_new_tensor_1d(cache.ctx, wtype, n_elements, NE_SIZE_CALC);
  cache.v = ne_new_tensor_1d(cache.ctx, wtype, n_elements, NE_SIZE_CALC);
  ne_set_name(cache.k, "cache_k");
  ne_set_name(cache.v, "cache_v");

  return true;
}

struct model_context_params model_context_default_params() {
  struct model_context_params result = {
      /*arch                         =*/MODEL_LLAMA,
      /*.n_ctx                       =*/512,
      /*.gpu_layers                  =*/0,
      /*.seed                        =*/-1,
      /*.f16_kv                      =*/true,
      /*.logits_all                  =*/false,
      /*.vocab_only                  =*/false,
      /*.use_mmap                    =*/true,
      /*.use_mlock                   =*/false,
      /*.embedding                   =*/false,
      /*.batch_size                  =*/1,
      /*.beam_search                 =*/false,
      /*.beam_size                   =*/1,
      /*.progress_callback           =*/nullptr,
      /*.progress_callback_user_data =*/nullptr,
  };

  return result;
}

bool model_mmap_supported() { return model_mmap::SUPPORTED; }

bool model_mlock_supported() { return model_mlock::SUPPORTED; }

void model_init_backend() {
  ne_time_init();

  // needed to initialize f16 tables
  {
    struct ne_init_params params = {0, NULL, false};
    struct ne_context* ctx = ne_init(params);
    ne_free(ctx);
  }
}

int64_t model_time_us() { return ne_time_us(); }

//
// model loading
//

static bool model_load(const std::string& fname, model_archs arch, model_context& lctx, int n_ctx, int n_gpu_layers,
                       ne_type memory_type, bool use_mmap, bool use_mlock, bool vocab_only,
                       model_progress_callback progress_callback, void* progress_callback_user_data) {
  try {
    model_load_internal(fname, arch, lctx, n_ctx, n_gpu_layers, memory_type, use_mmap, use_mlock, vocab_only,
                        progress_callback, progress_callback_user_data);
    return true;
  } catch (const std::string& err) {
    fprintf(stderr, "error loading model: %s\n", err.c_str());
    return false;
  }
}

//
// tokenizer
//

static size_t utf8_len(char src) {
  const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t highbits = static_cast<uint8_t>(src) >> 4;
  return lookup[highbits];
}

struct model_sp_symbol {
  using index = int;
  index prev;
  index next;
  const char* text;
  size_t n;
};

static_assert(std::is_trivially_copyable<model_sp_symbol>::value, "model_sp_symbol is not trivially copyable");

struct model_sp_bigram {
  struct comparator {
    bool operator()(model_sp_bigram& l, model_sp_bigram& r) {
      return (l.score < r.score) || (l.score == r.score && l.left > r.left);
    }
  };
  using queue_storage = std::vector<model_sp_bigram>;
  using queue = std::priority_queue<model_sp_bigram, queue_storage, comparator>;
  model_sp_symbol::index left;
  model_sp_symbol::index right;
  float score;
  size_t size;
};

// original implementation:
// https://github.com/ggerganov/model.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct model_tokenizer {
  model_tokenizer(const model_vocab& vocab) : vocab_(vocab) {}

  void tokenize(const std::string& text, std::vector<model_vocab::id>& output) {
    // split string into utf8 chars
    int index = 0;
    size_t offs = 0;
    while (offs < text.size()) {
      model_sp_symbol sym;
      size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
      sym.text = text.c_str() + offs;
      sym.n = char_len;
      offs += char_len;
      sym.prev = index - 1;
      sym.next = offs == text.size() ? -1 : index + 1;
      index++;
      symbols_.emplace_back(sym);
    }

    // seed the work queue with all possible 2-character tokens.
    for (size_t i = 1; i < symbols_.size(); ++i) {
      try_add_bigram(i - 1, i);
    }

    // keep substituting the highest frequency pairs for as long as we can.
    while (!work_queue_.empty()) {
      auto bigram = work_queue_.top();
      work_queue_.pop();

      auto& left_sym = symbols_[bigram.left];
      auto& right_sym = symbols_[bigram.right];

      // if one of the symbols already got merged, skip it.
      if (left_sym.n == 0 || right_sym.n == 0 || left_sym.n + right_sym.n != bigram.size) {
        continue;
      }

      // merge the right sym into the left one
      left_sym.n += right_sym.n;
      right_sym.n = 0;

      // printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text,
      // bigram.size);

      // remove the right sym from the chain
      left_sym.next = right_sym.next;
      if (right_sym.next >= 0) {
        symbols_[right_sym.next].prev = bigram.left;
      }

      // find more substitutions
      try_add_bigram(left_sym.prev, bigram.left);
      try_add_bigram(bigram.left, left_sym.next);
    }

    for (int i = 0; i != -1; i = symbols_[i].next) {
      auto& symbol = symbols_[i];
      auto symbol_text = std::string(symbol.text, symbol.n);
      auto token = vocab_.token_to_id.find(symbol_text);

      if (token == vocab_.token_to_id.end()) {
        // output any symbols that did not form tokens as bytes.
        for (int j = 0; j < (int)symbol.n; ++j) {
          model_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
          output.push_back(token_id);
        }
      } else {
        output.push_back((*token).second);
      }
    }
  }

 private:
  void try_add_bigram(int left, int right) {
    if (left == -1 || right == -1) {
      return;
    }

    const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
    auto token = vocab_.token_to_id.find(text);

    if (token == vocab_.token_to_id.end()) {
      return;
    }

    if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
      return;
    }

    const auto& tok_score = vocab_.id_to_token[(*token).second];

    model_sp_bigram bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.score = tok_score.score;
    bigram.size = text.size();
    work_queue_.push(bigram);
  }

  const model_vocab& vocab_;
  std::vector<model_sp_symbol> symbols_;
  model_sp_bigram::queue work_queue_;
};

static std::vector<model_vocab::id> model_tokenize(const model_vocab& vocab, const std::string& text, bool bos) {
  model_tokenizer tokenizer(vocab);
  std::vector<model_vocab::id> output;

  if (text.empty()) {
    return output;
  }

  if (bos) {
    output.push_back(vocab.bos_token_id);
  }

  tokenizer.tokenize(text, output);
  return output;
}

//
// sampling
//

void model_sample_softmax(struct model_context* ctx, model_token_data_array* candidates) {
  assert(candidates->size > 0);

  const int64_t t_start_sample_us = ne_time_us();

  // Sort the logits in descending order
  if (!candidates->sorted) {
    std::sort(candidates->data, candidates->data + candidates->size,
              [](const model_token_data& a, const model_token_data& b) { return a.logit > b.logit; });
    candidates->sorted = true;
  }

  float max_l = candidates->data[0].logit;
  float cum_sum = 0.0f;
  for (size_t i = 0; i < candidates->size; ++i) {
    float p = expf(candidates->data[i].logit - max_l);
    candidates->data[i].p = p;
    cum_sum += p;
  }
  for (size_t i = 0; i < candidates->size; ++i) {
    candidates->data[i].p /= cum_sum;
  }

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_top_k(struct model_context* ctx, model_token_data_array* candidates, int k, size_t min_keep) {
  const int64_t t_start_sample_us = ne_time_us();

  k = std::max(k, (int)min_keep);
  k = std::min(k, (int)candidates->size);

  // Sort scores in descending order
  if (!candidates->sorted) {
    auto comp = [](const model_token_data& a, const model_token_data& b) { return a.logit > b.logit; };
    if (k == (int)candidates->size) {
      std::sort(candidates->data, candidates->data + candidates->size, comp);
    } else {
      std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
    }
    candidates->sorted = true;
  }
  candidates->size = k;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_top_p(struct model_context* ctx, model_token_data_array* candidates, float p, size_t min_keep) {
  if (p >= 1.0f) {
    return;
  }

  const int64_t t_start_sample_us = ne_time_us();

  model_sample_softmax(ctx, candidates);

  // Compute the cumulative probabilities
  float cum_sum = 0.0f;
  size_t last_idx = candidates->size;

  for (size_t i = 0; i < candidates->size; ++i) {
    cum_sum += candidates->data[i].p;

    // Check if the running sum is greater than p or if we have kept at least
    // min_keep tokens
    if (cum_sum > p && i >= min_keep) {
      last_idx = i;
      break;
    }
  }

  // Resize the output vector to keep only the top-p tokens
  candidates->size = last_idx;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_tail_free(struct model_context* ctx, model_token_data_array* candidates, float z, size_t min_keep) {
  if (z >= 1.0f || candidates->size <= 2) {
    return;
  }

  const int64_t t_start_sample_us = ne_time_us();

  model_sample_softmax(nullptr, candidates);

  // Compute the first and second derivatives
  std::vector<float> first_derivatives(candidates->size - 1);
  std::vector<float> second_derivatives(candidates->size - 2);

  for (size_t i = 0; i < first_derivatives.size(); ++i) {
    first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
  }
  for (size_t i = 0; i < second_derivatives.size(); ++i) {
    second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
  }

  // Calculate absolute value of second derivatives
  for (size_t i = 0; i < second_derivatives.size(); ++i) {
    second_derivatives[i] = abs(second_derivatives[i]);
  }

  // Normalize the second derivatives
  float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);
  for (float& value : second_derivatives) {
    value /= second_derivatives_sum;
  }

  float cum_sum = 0.0f;
  size_t last_idx = candidates->size;
  for (size_t i = 0; i < second_derivatives.size(); ++i) {
    cum_sum += second_derivatives[i];

    // Check if the running sum is greater than z or if we have kept at least
    // min_keep tokens
    if (cum_sum > z && i >= min_keep) {
      last_idx = i;
      break;
    }
  }

  // Resize the output vector to keep only the tokens above the tail location
  candidates->size = last_idx;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_typical(struct model_context* ctx, model_token_data_array* candidates, float p, size_t min_keep) {
  // Reference implementation:
  // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
  if (p >= 1.0f) {
    return;
  }

  const int64_t t_start_sample_us = ne_time_us();

  // Compute the softmax of logits and calculate entropy
  model_sample_softmax(nullptr, candidates);

  float entropy = 0.0f;
  for (size_t i = 0; i < candidates->size; ++i) {
    entropy += -candidates->data[i].p * logf(candidates->data[i].p);
  }

  // Compute the absolute difference between negative log probability and
  // entropy for each candidate
  std::vector<float> shifted_scores;
  for (size_t i = 0; i < candidates->size; ++i) {
    float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
    shifted_scores.push_back(shifted_score);
  }

  // Sort tokens based on the shifted_scores and their corresponding indices
  std::vector<size_t> indices(candidates->size);
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) { return shifted_scores[a] < shifted_scores[b]; });

  // Compute the cumulative probabilities
  float cum_sum = 0.0f;
  size_t last_idx = indices.size();

  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];
    cum_sum += candidates->data[idx].p;

    // Check if the running sum is greater than typical or if we have kept at
    // least min_keep tokens
    if (cum_sum > p && i >= min_keep - 1) {
      last_idx = i + 1;
      break;
    }
  }

  // Resize the output vector to keep only the locally typical tokens
  std::vector<model_token_data> new_candidates;
  for (size_t i = 0; i < last_idx; ++i) {
    size_t idx = indices[i];
    new_candidates.push_back(candidates->data[idx]);
  }

  // Replace the data in candidates with the new_candidates data
  std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
  candidates->size = new_candidates.size();

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_temperature(struct model_context* ctx, model_token_data_array* candidates_p, float temp) {
  const int64_t t_start_sample_us = ne_time_us();

  for (size_t i = 0; i < candidates_p->size; ++i) {
    candidates_p->data[i].logit /= temp;
  }

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

model_token model_sample_top_k_top_p(struct model_context* ctx, const int n_logits, const float* logits, int top_k,
                                     double top_p, double temp) {
  const int64_t t_start_sample_us = ne_time_us();
  std::vector<std::pair<double, model_token>> logits_id;
  logits_id.reserve(n_logits);

  {
    const double scale = 1.0 / temp;
    for (int i = 0; i < n_logits; ++i) {
      logits_id.push_back(std::make_pair(logits[i] * scale, i));
    }
  }

  // find the top K tokens
  std::partial_sort(logits_id.begin(), logits_id.begin() + top_k, logits_id.end(),
                    [](const std::pair<double, model_token>& a, const std::pair<double, model_token>& b) {
                      return a.first > b.first;
                    });

  logits_id.resize(top_k);

  double maxl = -INFINITY;
  for (const auto& kv : logits_id) {
    maxl = std::max(maxl, kv.first);
  }

  // compute probs for the top K tokens
  std::vector<double> probs;
  probs.reserve(logits_id.size());

  double sum = 0.0;
  for (const auto& kv : logits_id) {
    double p = exp(kv.first - maxl);
    probs.push_back(p);
    sum += p;
  }

  // normalize the probs
  for (auto& p : probs) {
    p /= sum;
  }

  if (top_p < 1.0f) {
    double cumsum = 0.0f;
    for (int i = 0; i < top_k; i++) {
      cumsum += probs[i];
      if (cumsum >= top_p) {
        top_k = i + 1;
        probs.resize(top_k);
        logits_id.resize(top_k);
        break;
      }
    }

    cumsum = 1.0 / cumsum;
    for (int i = 0; i < (int)probs.size(); i++) {
      probs[i] *= cumsum;
    }
  }

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
  return logits_id[dist(ctx->rng)].second;
}

void model_sample_repetition_penalty(struct model_context* ctx, model_token_data_array* candidates,
                                     const model_token* last_tokens, size_t last_tokens_size, float penalty) {
  if (last_tokens_size == 0 || penalty == 1.0f) {
    return;
  }

  const int64_t t_start_sample_us = ne_time_us();

  for (size_t i = 0; i < candidates->size; ++i) {
    const auto* token_iter = std::find(last_tokens, last_tokens + last_tokens_size, candidates->data[i].id);
    if (token_iter == last_tokens + last_tokens_size) {
      continue;
    }

    // The academic publication that described this technique actually just only
    // divided, but that would cause tokens with negative logits to become more
    // likely, which is obviously wrong. This is common fix for this problem,
    // which is to multiply by the penalty instead of dividing.
    if (candidates->data[i].logit <= 0) {
      candidates->data[i].logit *= penalty;
    } else {
      candidates->data[i].logit /= penalty;
    }
  }

  candidates->sorted = false;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_frequency_and_presence_penalties(struct model_context* ctx, model_token_data_array* candidates,
                                                   const model_token* last_tokens_p, size_t last_tokens_size,
                                                   float alpha_frequency, float alpha_presence) {
  if (last_tokens_size == 0 || (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
    return;
  }

  const int64_t t_start_sample_us = ne_time_us();

  // Create a frequency map to count occurrences of each token in last_tokens
  std::unordered_map<model_token, int> token_count;
  for (size_t i = 0; i < last_tokens_size; ++i) {
    token_count[last_tokens_p[i]]++;
  }

  // Apply frequency and presence penalties to the candidates
  for (size_t i = 0; i < candidates->size; ++i) {
    auto token_iter = token_count.find(candidates->data[i].id);
    if (token_iter == token_count.end()) {
      continue;
    }

    int count = token_iter->second;
    candidates->data[i].logit -= float(count) * alpha_frequency + float(count > 0) * alpha_presence;
  }

  candidates->sorted = false;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

model_token model_sample_token_mirostat(struct model_context* ctx, model_token_data_array* candidates, float tau,
                                        float eta, int m, float* mu) {
  assert(ctx);
  auto N = float(model_n_vocab(ctx));
  int64_t t_start_sample_us;
  t_start_sample_us = ne_time_us();

  model_sample_softmax(nullptr, candidates);

  // Estimate s_hat using the most probable m tokens
  float s_hat = 0.0;
  float sum_ti_bi = 0.0;
  float sum_ti_sq = 0.0;
  for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
    float t_i = logf(float(i + 2) / float(i + 1));
    float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
    sum_ti_bi += t_i * b_i;
    sum_ti_sq += t_i * t_i;
  }
  s_hat = sum_ti_bi / sum_ti_sq;

  // Compute k from the estimated s_hat and target surprise value
  float epsilon_hat = s_hat - 1;
  float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

  // Sample the next word X using top-k sampling
  model_sample_top_k(nullptr, candidates, int(k), 1);
  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
  model_token X = model_sample_token(ctx, candidates);
  t_start_sample_us = ne_time_us();

  // Compute error as the difference between observed surprise and target
  // surprise value
  size_t X_idx = std::distance(candidates->data,
                               std::find_if(candidates->data, candidates->data + candidates->size,
                                            [&](const model_token_data& candidate) { return candidate.id == X; }));
  float observed_surprise = -log2f(candidates->data[X_idx].p);
  float e = observed_surprise - tau;

  // Update mu using the learning rate and error
  *mu = *mu - eta * e;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
    ctx->n_sample++;
  }
  return X;
}

model_token model_sample_token_mirostat_v2(struct model_context* ctx, model_token_data_array* candidates, float tau,
                                           float eta, float* mu) {
  assert(ctx);
  int64_t t_start_sample_us;
  t_start_sample_us = ne_time_us();

  model_sample_softmax(ctx, candidates);

  // Truncate the words with surprise values greater than mu
  candidates->size = std::distance(
      candidates->data, std::find_if(candidates->data, candidates->data + candidates->size,
                                     [&](const model_token_data& candidate) { return -log2f(candidate.p) > *mu; }));

  // Normalize the probabilities of the remaining words
  model_sample_softmax(ctx, candidates);

  // Sample the next word X from the remaining words
  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
  model_token X = model_sample_token(ctx, candidates);
  t_start_sample_us = ne_time_us();

  // Compute error as the difference between observed surprise and target
  // surprise value
  size_t X_idx = std::distance(candidates->data,
                               std::find_if(candidates->data, candidates->data + candidates->size,
                                            [&](const model_token_data& candidate) { return candidate.id == X; }));
  float observed_surprise = -log2f(candidates->data[X_idx].p);
  float e = observed_surprise - tau;

  // Update mu using the learning rate and error
  *mu = *mu - eta * e;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
  return X;
}

model_token model_sample_token_greedy(struct model_context* ctx, model_token_data_array* candidates) {
  const int64_t t_start_sample_us = ne_time_us();

  // Find max element
  auto* max_iter =
      std::max_element(candidates->data, candidates->data + candidates->size,
                       [](const model_token_data& a, const model_token_data& b) { return a.logit < b.logit; });

  model_token result = max_iter->id;
  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
    ctx->n_sample++;
  }
  return result;
}

model_token model_sample_token(struct model_context* ctx, model_token_data_array* candidates) {
  assert(ctx);
  const int64_t t_start_sample_us = ne_time_us();
  model_sample_softmax(nullptr, candidates);

  std::vector<float> probs;
  probs.reserve(candidates->size);
  for (size_t i = 0; i < candidates->size; ++i) {
    probs.push_back(candidates->data[i].p);
  }

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  auto& rng = ctx->rng;
  int idx = dist(rng);

  model_token result = candidates->data[idx].id;

  ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  ctx->n_sample++;
  return result;
}

//
// quantization
//
quant_params_internal quant_params_to_internal(const quant_params& params) {
  return quant_params_internal{parse_bits(params.bits), parse_alg(params.alg), params.block_size,
                               parse_scale_dtype(params.scale_dtype), parse_compute_type(params.compute_type)};
}

size_t jblas_quantize(const float* f32ptr, void* dstpr, const quant_params_internal params, int nthread, int n, int k) {
  using CompType = jblas::prologue::weight_comp::gemm_kblcok::WeightCompType;
  using namespace ne_jblas;
  auto cd = jblas::utils::parallel::CpuDevice::getInstance();
  jblas::prologue::PackedWeight* packedw = NULL;

  cd->setThreads(nthread);
  if (params.bits == quant_bits::q4) {
    if (params.scale_dtype == quant_sdtype::fp32) {
      if (params.compute_type == quant_comp::int8) {
        using Kernel = WeiS4ClipFp32<GcCompInt8KBlock, JblasAVX512F>;
        using KernelRef = WeiS4ClipFp32<GcCompInt8KBlock, JblasNoSIMD>;
        static Kernel kernel;
        static KernelRef kernelref;
        packedw = kernel.createStorage(n, k, params.block_size);
        if (cd->AVX512F()) {
          kernel.packTransposeWeight(n, k, f32ptr, k, packedw);
        } else {
          kernelref.packTransposeWeight(n, k, f32ptr, k, packedw);
        }
      } else if (params.compute_type == quant_comp::fp32) {
        using Kernel = WeiS4ClipFp32<GcCompFp32, JblasAVX512_FP16>;
        using KernelRef = WeiS4ClipFp32<GcCompFp32, JblasNoSIMD>;
        static Kernel kernel;
        static Kernel kernelref;
        packedw = kernel.createStorage(n, k, params.block_size);
        if (cd->AVX512_FP16()) {
          kernel.packTransposeWeight(n, k, f32ptr, k, packedw);
        } else {
          kernelref.packTransposeWeight(n, k, f32ptr, k, packedw);
        }
      } else if (params.compute_type == quant_comp::bf16) {
        using Kernel = WeiS4ClipFp32<GcCompBf16, JblasAMX_BF16>;
        using KernelRef = WeiS4ClipFp32<GcCompBf16, JblasNoSIMD>;
        static Kernel kernel;
        static Kernel kernelref;
        packedw = kernel.createStorage(n, k, params.block_size);
        if (cd->AMX_BF16()) {
          kernel.packTransposeWeight(n, k, f32ptr, k, packedw);
        } else {
          kernelref.packTransposeWeight(n, k, f32ptr, k, packedw);
        }
      }
    }

  } else if (params.bits == quant_bits::q8) {
    // TODO add 8bit quantization
    if (params.scale_dtype == quant_sdtype::fp32) {
      if (params.compute_type == quant_comp::int8) {
        if (params.block_size == -1) {
          using Kernel = WeiS8Fp32PerN<GcCompInt8, JblasAVX512F>;
          using KernelRef = WeiS8Fp32PerN<GcCompInt8, JblasNoSIMD>;
          static Kernel kernel;
          static Kernel kernelref;
          packedw = kernel.createStorage(n, k);
          if (cd->AVX512F()) {
            kernel.packTransposeWeight(n, k, f32ptr, k, packedw);
          } else {
            kernelref.packTransposeWeight(n, k, f32ptr, k, packedw);
          }
        } else {
          using Kernel = WeiS8Fp32<GcCompInt8KBlock, JblasAVX512F>;
          using KernelRef = WeiS8Fp32<GcCompInt8KBlock, JblasNoSIMD>;
          static Kernel kernel;
          static Kernel kernelref;
          packedw = kernel.createStorage(n, k, params.block_size);
          if (cd->AVX512F()) {
            kernel.packTransposeWeight(n, k, f32ptr, k, packedw);
          } else {
            kernelref.packTransposeWeight(n, k, f32ptr, k, packedw);
          }
        }
      } else if (params.compute_type == quant_comp::fp32) {
        using Kernel = WeiS8Fp32<GcCompFp32, JblasAVX512_FP16>;
        using KernelRef = WeiS8Fp32<GcCompFp32, JblasNoSIMD>;
        static Kernel kernel;
        static Kernel kernelref;
        packedw = kernel.createStorage(n, k, params.block_size);
        if (cd->AVX512_FP16()) {
          kernel.packTransposeWeight(n, k, f32ptr, k, packedw);
        } else {
          kernelref.packTransposeWeight(n, k, f32ptr, k, packedw);
        }
      } else if (params.compute_type == quant_comp::bf16) {
        using Kernel = WeiS8Fp32<GcCompBf16, JblasAMX_BF16>;
        using KernelRef = WeiS8Fp32<GcCompBf16, JblasNoSIMD>;
        static Kernel kernel;
        static Kernel kernelref;
        packedw = kernel.createStorage(n, k, params.block_size);
        if (cd->AMX_BF16()) {
          kernel.packTransposeWeight(n, k, f32ptr, k, packedw);
        } else {
          kernelref.packTransposeWeight(n, k, f32ptr, k, packedw);
        }
      }
    }
  }
  assert(packedw != 0);
  auto size = packedw->getSerializedSize();
  packedw->serializeToBuffer(dstpr);
  delete packedw;
  return size;
}

size_t ggml_quantize(const float* f32ptr, void* dstpr, const ne_type new_type, int nthread, size_t nelements) {
  std::vector<int64_t> hist_cur(1 << 4, 0);
  std::vector<std::thread> workers;
  std::mutex mutex;
  int chunk_size = 32 * 512;
  const int nchunk = (nelements + chunk_size - 1) / chunk_size;
  const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
  size_t new_size = 0;
  if (nthread_use < 2) {
    new_size = ne_quantize_chunk(new_type, f32ptr, dstpr, 0, nelements, hist_cur.data());
  } else {
    size_t counter = 0;
    new_size = 0;
    auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32ptr, dstpr, nelements, chunk_size]() {
      std::vector<int64_t> local_hist;
      size_t local_size = 0;
      while (true) {
        std::unique_lock<std::mutex> lock(mutex);
        size_t first = counter;
        counter += chunk_size;
        if (first >= nelements) {
          if (!local_hist.empty()) {
            for (int j = 0; j < int(local_hist.size()); ++j) {
              hist_cur[j] += local_hist[j];
            }
            new_size += local_size;
          }
          break;
        }
        lock.unlock();
        size_t last = std::min(nelements, first + chunk_size);
        if (local_hist.empty()) {
          local_hist.resize(hist_cur.size(), 0);
        }
        local_size += ne_quantize_chunk(new_type, f32ptr, dstpr, first, last - first, local_hist.data());
      }
    };
    if ((int)workers.size() < nthread_use - 1) {
      workers.resize(nthread_use - 1);
    }
    for (int it = 0; it < nthread_use - 1; ++it) {
      workers[it] = std::thread(compute);
    }
    compute();
    for (int it = 0; it < nthread_use - 1; ++it) {
      workers[it].join();
    }
  }
  return new_size;
}

void ne_common_quantize(const int nthread, const quant_params_internal& params, model_load_tensor& tensor,
                        model_file_saver& saver, size_t& size_org, size_t& size_new) {
  size_t nelements = tensor.ne.at(0) * tensor.ne.at(1);
  enum ne_type new_type = quant_params_to_type(params);
  model_buffer work;
  work.resize(nelements * 4);  // upper bound on size
  void* new_data = work.addr;
  size_t new_size = 0;
  float* f32_data = NULL;
  model_buffer f32_conv_buf;
  if (tensor.type == NE_TYPE_F32) {
    f32_data = (float*)tensor.data;
  } else if (tensor.type == NE_TYPE_F16) {
    f32_conv_buf.resize(nelements * sizeof(float));
    f32_data = (float*)f32_conv_buf.addr;
    const auto* f16_data = (const ne_fp16_t*)tensor.data;
    for (size_t i = 0; i < nelements; i++) {
      f32_data[i] = ne_fp16_to_fp32(f16_data[i]);
    }
  } else {
    throw format("type %s unsupported for integer quantization", ne_type_name(tensor.type));
  }
  printf("quantizing .. ");
  fflush(stdout);
  if (new_type == NE_TYPE_JBLAS) {
    int k_ = tensor.ne.at(0);
    int n_ = tensor.ne.at(1);
    new_size = jblas_quantize(f32_data, work.addr, params, nthread, n_, k_);
    printf("JBLAS ");
  } else if (new_type >= NE_TYPE_Q4_0 && new_type < NE_TYPE_JBLAS) {
    new_size = ggml_quantize(f32_data, work.addr, new_type, nthread, nelements);
    printf("GGML ");
  }
  printf("size = %8.2f MB -> %8.2f MB\n", tensor.size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);

__WRITE_FILE:
  size_org += tensor.size;
  size_new += new_size;
  saver.write_tensor(tensor, new_type, new_data, new_size);
  printf("\n");
}

static void model_quantize_internal(const quant_params& params, std::shared_ptr<quant_layer_base> quant_layer) {
  auto ftype = quant_params_to_ftype(params);
  quant_layer->set_global_config(params.nthread, quant_params_to_internal(params));
  int nthread = params.nthread;
  if (nthread <= 0) {
    nthread = std::thread::hardware_concurrency();
  }
  std::unique_ptr<model_model_loader> model_loader(new model_model_loader(params.model_file, /*use_mmap*/ false,
                                                                          /*vocab_only*/ false));
  model_file_saver file_saver(params.out_file.c_str(), model_loader->file_loaders.at(0).get(), ftype);
  size_t total_size_org = 0;
  size_t total_size_new = 0;
  size_t idx = 0;
  for (model_load_tensor& tensor : model_loader->tensors_map.tensors) {
    model_buffer read_data;
    read_data.resize(tensor.size);
    tensor.data = read_data.addr;
    model_loader->load_data_for(tensor);
    printf("[%4zu/%4zu] %36s - %16s, type = %6s, ", ++idx, model_loader->tensors_map.tensors.size(),
           tensor.name.c_str(), model_format_tensor_shape(tensor.ne).c_str(), ne_type_name(tensor.type));
    std::vector<int64_t> tmpne(tensor.ne.size());
    for (size_t i = 0; i < tmpne.size(); i++) {
      tmpne[i] = static_cast<int64_t>(tensor.ne[i]);
    }
    auto lconfig = quant_layer->get_layer_config(tensor.name, tmpne, tensor.type);
    bool quantize = lconfig.valid();
    printf("%s,", lconfig.getstr().c_str());
    if (quantize) {
      ne_common_quantize(nthread, lconfig, tensor, file_saver, total_size_org, total_size_new);
    } else {
      printf("size = %8.3f MB\n", tensor.size / 1024.0 / 1024.0);
      total_size_org += tensor.size;
      total_size_new += tensor.size;
      file_saver.write_tensor(tensor, tensor.type, tensor.data, tensor.size);
      printf("\n");
    }
  }
  printf("%s: model size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
  printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);
}

//
// interface implementation
//

struct model_context* model_init_from_file(const char* path_model, struct model_context_params params) {
  ne_time_init();

  model_context* ctx = new model_context;

  if (params.seed < 0) {
    params.seed = time(NULL);
  }

  unsigned cur_percentage = 0;
  if (params.progress_callback == NULL) {
    params.progress_callback_user_data = &cur_percentage;
    params.progress_callback = [](float progress, void* ctx) {
      unsigned* cur_percentage_p = (unsigned*)ctx;
      unsigned percentage = (unsigned)(100 * progress);
      while (percentage > *cur_percentage_p) {
        *cur_percentage_p = percentage;
        fprintf(stderr, ".");
        fflush(stderr);
        if (percentage >= 100) {
          fprintf(stderr, "\n");
        }
      }
    };
  }

  ctx->rng = std::mt19937(params.seed);
  ctx->logits_all = params.logits_all;
  ctx->batch_size = params.batch_size;

  ne_type memory_type = params.f16_kv ? NE_TYPE_F16 : NE_TYPE_F32;
  model_archs arch = params.arch;

  if (!model_load(path_model, arch, *ctx, params.n_ctx, params.n_gpu_layers, memory_type, params.use_mmap,
                  params.use_mlock, params.vocab_only, params.progress_callback, params.progress_callback_user_data)) {
    fprintf(stderr, "%s: failed to load model\n", __func__);
    model_free(ctx);
    return nullptr;
  }

  // reserve memory for context buffers
  if (!params.vocab_only) {
    int kv_ctx = ctx->model.hparams.n_ctx;
    if (params.beam_search) {
      ctx->beam_search = true;
      ctx->beam_size = params.beam_size;
      ctx->kv_n_ctx_block = ctx->batch_size * ctx->beam_size;
      kv_ctx *= ctx->kv_n_ctx_block;
    }
    if (!kv_cache_init(ctx->model.hparams, ctx->model.kv_self, memory_type, kv_ctx)) {
      fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
      model_free(ctx);
      return nullptr;
    }

    {
      const size_t memory_size = ne_nbytes(ctx->model.kv_self.k) + ne_nbytes(ctx->model.kv_self.v);
      fprintf(stderr, "%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
    }

    const auto& hparams = ctx->model.hparams;

    // resized during inference
    if (params.logits_all) {
      ctx->logits.reserve(hparams.n_ctx * hparams.n_vocab);
    } else {
      ctx->logits.reserve(hparams.n_vocab);
    }

    if (params.embedding) {
      ctx->embedding.resize(hparams.n_embd);
    }

    ctx->buf_compute.resize(ctx->model.scratchs.eval);

    ctx->buf_scratch[0].resize(ctx->model.scratchs.scratch0);
    ctx->buf_scratch[1].resize(ctx->model.scratchs.scratch1);
  }

  return ctx;
}

void model_free(struct model_context* ctx) { delete ctx; }

int model_quantize(const quant_params& params, std::shared_ptr<quant_layer_base> quant_layer) {
  try {
    model_quantize_internal(params, quant_layer);
    return 0;
  } catch (const std::string& err) {
    fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.c_str());
    return 1;
  }
}

int model_apply_lora_from_file_internal(struct model_context* ctx, const char* path_lora, const char* path_base_model,
                                        int n_threads) {
  fprintf(stderr, "%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

  auto& model = ctx->model;

  const int64_t t_start_lora_us = ne_time_us();

  auto fin = std::ifstream(path_lora, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_lora);
    return 1;
  }

  // verify magic and version
  {
    uint32_t magic;
    fin.read((char*)&magic, sizeof(magic));
    if (magic != MODEL_FILE_MAGIC_GGLA) {
      fprintf(stderr, "%s: bad file magic\n", __func__);
      return 1;
    }
    uint32_t format_version;
    fin.read((char*)&format_version, sizeof(format_version));

    if (format_version != 1) {
      fprintf(stderr, "%s: unsupported file version\n", __func__);
      return 1;
    }
  }

  int32_t lora_r;
  int32_t lora_alpha;
  fin.read((char*)&lora_r, sizeof(lora_r));
  fin.read((char*)&lora_alpha, sizeof(lora_alpha));
  float scaling = (float)lora_alpha / (float)lora_r;

  fprintf(stderr, "%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);

  // create a temporary ne context to store the lora tensors
  // todo: calculate size from biggest possible tensor
  std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
  struct ne_init_params params;
  params.mem_size = lora_buf.size();
  params.mem_buffer = lora_buf.data();
  params.no_alloc = false;

  ne_context* lora_ctx = ne_init(params);
  std::unordered_map<std::string, struct ne_tensor*> lora_tensors;

  // create a name -> tensor map of the model to accelerate lookups
  std::unordered_map<std::string, struct ne_tensor*> model_tensors;
  for (auto& kv : model.tensors_by_name) {
    model_tensors.insert(kv);
  }

  // load base model
  std::unique_ptr<model_model_loader> model_loader;
  ne_context* base_ctx = NULL;
  model_buffer base_buf;
  if (path_base_model) {
    fprintf(stderr, "%s: loading base model from '%s'\n", __func__, path_base_model);
    model_loader.reset(new model_model_loader(path_base_model, /*use_mmap*/ true, /*vocab_only*/ false));

    size_t ctx_size;
    size_t mmapped_size;
    model_loader->calc_sizes(&ctx_size, &mmapped_size);
    base_buf.resize(ctx_size);

    ne_init_params base_params;
    base_params.mem_size = base_buf.size;
    base_params.mem_buffer = base_buf.addr;
    base_params.no_alloc = model_loader->use_mmap;

    base_ctx = ne_init(base_params);

    model_loader->ne_ctx = base_ctx;

    // maybe this should in model_model_loader
    if (model_loader->use_mmap) {
      model_loader->mapping.reset(new model_mmap(&model_loader->file_loaders.at(0)->file, /* prefetch */ 0));
    }
  }

  // read tensors and apply
  bool warned = false;
  int n_tensors = 0;
  while (true) {
    int32_t n_dims;
    int32_t length;
    int32_t ftype;

    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char*>(&length), sizeof(length));
    fin.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));
    if (fin.eof()) {
      break;
    }

    int32_t ne[2] = {1, 1};
    for (int i = 0; i < n_dims; ++i) {
      fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
    }

    std::string name;
    {
      char buf[1024];
      fin.read(buf, length);
      name = std::string(buf, length);
    }

    // check for lora suffix and get the type of tensor
    const std::string lora_suffix = ".lora";
    size_t pos = name.rfind(lora_suffix);
    if (pos == std::string::npos) {
      fprintf(stderr, "%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
      return 1;
    }

    std::string lora_type = name.substr(pos + lora_suffix.length());
    std::string base_name = name;
    base_name.erase(pos);
    // fprintf(stderr, "%s: %s => %s (lora type %s) ", __func__,
    // name.c_str(),base_name.c_str(), lora_type.c_str());

    if (model_tensors.find(base_name) == model_tensors.end()) {
      fprintf(stderr, "%s: unknown tensor '%s' in lora adapter\n", __func__, name.data());
      return 1;
    }

    // create ne tensor
    ne_type wtype;
    switch (ftype) {
      case 0:
        wtype = NE_TYPE_F32;
        break;
      case 1:
        wtype = NE_TYPE_F16;
        break;
      default: {
        fprintf(stderr, "%s: invalid tensor data type '%d'\n", __func__, ftype);
        return false;
      }
    }
    ne_tensor* lora_tensor;
    if (n_dims == 2) {
      lora_tensor = ne_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1], NE_SIZE_CALC);
    } else {
      fprintf(stderr, "%s: unsupported tensor dimension %d\n", __func__, n_dims);
      return 1;
    }

    // load tensor data
    size_t offset = fin.tellg();
    size_t tensor_data_size = ne_nbytes(lora_tensor);
    offset = (offset + 31) & -32;
    fin.seekg(offset);
    fin.read((char*)lora_tensor->data, tensor_data_size);

    lora_tensors[name] = lora_tensor;

    // check if we have both A and B tensors and apply
    if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
        lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {
      ne_tensor* dest_t = model_tensors[base_name];
      ne_tensor* base_t;
      if (model_loader) {
        // load from base model
        if (model_loader->tensors_map.name_to_idx.find(base_name) == model_loader->tensors_map.name_to_idx.end()) {
          fprintf(stderr, "%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
          return 1;
        }
        size_t idx = model_loader->tensors_map.name_to_idx[base_name];
        model_load_tensor& lt = model_loader->tensors_map.tensors[idx];
        base_t =
            model_loader->get_tensor(base_name, {(uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1]}, NE_BACKEND_CPU);
        lt.data = (uint8_t*)lt.ne_tensor->data;
        model_loader->load_data_for(lt);
        lt.ne_tensor->data = lt.data;
      } else {
        base_t = dest_t;
      }

      if (ne_is_quantized(base_t->type)) {
        if (!warned) {
          fprintf(stderr,
                  "%s: warning: using a lora adapter with a quantized model "
                  "may result in poor quality, "
                  "use a f16 or f32 base model with --lora-base\n",
                  __func__);
          warned = true;
        }
      }

      ne_tensor* loraA = lora_tensors[base_name + ".loraA"];
      ne_tensor* loraB = lora_tensors[base_name + ".loraB"];

      if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
        fprintf(stderr,
                "%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64
                ");"
                " are you sure that this adapter is for this model?\n",
                __func__, base_t->ne[0], loraA->ne[1]);
        return 1;
      }

      // w = w + BA*s
      ne_tensor* BA = ne_mul_mat(lora_ctx, loraA, loraB);

      if (scaling != 1.0f) {
        ne_tensor* scale_tensor = ne_new_f32(lora_ctx, scaling);
        BA = ne_scale_inplace(lora_ctx, BA, scale_tensor);
      }

      ne_tensor* r;
      if (base_t == dest_t) {
        r = ne_add_inplace(lora_ctx, dest_t, BA);
      } else {
        r = ne_add(lora_ctx, base_t, BA);
        r = ne_cpy(lora_ctx, r, dest_t);
      }

      struct ne_cgraph gf = ne_build_forward(r);
      gf.n_threads = n_threads;
      ne_graph_compute(lora_ctx, &gf);

      // we won't need these tensors again, reset the context to save memory
      ne_free(lora_ctx);
      lora_ctx = ne_init(params);
      lora_tensors.clear();

      n_tensors++;
      if (n_tensors % 4 == 0) {
        fprintf(stderr, ".");
      }
    }
  }

  // TODO: this should be in a destructor, it will leak on failure
  ne_free(lora_ctx);
  if (base_ctx) {
    ne_free(base_ctx);
  }

  const int64_t t_lora_us = ne_time_us() - t_start_lora_us;
  fprintf(stderr, " done (%.2f ms)\n", t_lora_us / 1000.0);

  return 0;
}

int model_apply_lora_from_file(struct model_context* ctx, const char* path_lora, const char* path_base_model,
                               int n_threads) {
  try {
    return model_apply_lora_from_file_internal(ctx, path_lora, path_base_model, n_threads);
  } catch (const std::string& err) {
    fprintf(stderr, "%s: failed to apply lora adapter: %s\n", __func__, err.c_str());
    return 1;
  }
}

struct model_context* model_init_from_gpt_params(const gpt_params& params) {
  if (params.model_arch == MODEL_UNKNOWN) {
    fprintf(stderr, "error, please set model_name \n");
    exit(0);
  }
  auto lparams = model_context_default_params();

  lparams.arch = params.model_arch;
  lparams.n_ctx = params.n_ctx;
  lparams.n_gpu_layers = params.n_gpu_layers;
  lparams.seed = params.seed;
  lparams.f16_kv = params.memory_f16;
  lparams.use_mmap = params.use_mmap;
  lparams.use_mlock = params.use_mlock;
  lparams.logits_all = params.perplexity;
  lparams.embedding = params.embedding;
  lparams.batch_size = params.batch_size;
  lparams.beam_search = params.beam_search;
  lparams.beam_size = params.beam_size;

  model_context* lctx = model_init_from_file(params.model.c_str(), lparams);

  if (lctx == NULL) {
    fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
    return NULL;
  }

  if (!params.lora_adapter.empty()) {
    int err = model_apply_lora_from_file(lctx, params.lora_adapter.c_str(),
                                         params.lora_base.empty() ? NULL : params.lora_base.c_str(), params.n_threads);
    if (err != 0) {
      fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
      return NULL;
    }
  }

  return lctx;
}

int model_get_kv_cache_token_count(const struct model_context* ctx) { return ctx->model.kv_self.n; }

#define MODEL_MAX_RNG_STATE (64 * 1024)

void model_set_rng_seed(struct model_context* ctx, int seed) {
  if (seed < 0) {
    seed = time(NULL);
  }
  ctx->rng.seed(seed);
}

// Returns the *maximum* size of the state
size_t model_get_state_size(const struct model_context* ctx) {
  // we don't know size of rng until we actually serialize it. so reserve more
  // than enough memory for its serialized state. for reference,
  // std::mt19937(1337) serializes to 6701 bytes.
  const size_t s_rng_size = sizeof(size_t);
  const size_t s_rng = MODEL_MAX_RNG_STATE;
  const size_t s_logits_capacity = sizeof(size_t);
  const size_t s_logits_size = sizeof(size_t);
  const size_t s_logits = ctx->logits.capacity() * sizeof(float);
  const size_t s_embedding_size = sizeof(size_t);
  const size_t s_embedding = ctx->embedding.size() * sizeof(float);
  const size_t s_kv_size = sizeof(size_t);
  const size_t s_kv_ntok = sizeof(int);
  const size_t s_kv = ctx->model.kv_self.buf.size;

  const size_t s_total = (+s_rng_size + s_rng + s_logits_capacity + s_logits_size + s_logits + s_embedding_size +
                          s_embedding + s_kv_size + s_kv_ntok + s_kv);

  return s_total;
}

// Copies the state to the specified destination address
size_t model_copy_state_data(struct model_context* ctx, uint8_t* dst) {
  uint8_t* out = dst;

  // copy rng
  {
    std::stringstream rng_ss;
    rng_ss << ctx->rng;

    const size_t rng_size = rng_ss.str().size();
    char rng_buf[MODEL_MAX_RNG_STATE];

    memset(&rng_buf[0], 0, MODEL_MAX_RNG_STATE);
    memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

    memcpy(out, &rng_size, sizeof(rng_size));
    out += sizeof(rng_size);
    memcpy(out, &rng_buf[0], MODEL_MAX_RNG_STATE);
    out += MODEL_MAX_RNG_STATE;
  }

  // copy logits
  {
    const size_t logits_cap = ctx->logits.capacity();
    const size_t logits_size = ctx->logits.size();

    memcpy(out, &logits_cap, sizeof(logits_cap));
    out += sizeof(logits_cap);
    memcpy(out, &logits_size, sizeof(logits_size));
    out += sizeof(logits_size);

    if (logits_size) {
      memcpy(out, ctx->logits.data(), logits_size * sizeof(float));
    }

    out += logits_cap * sizeof(float);
  }

  // copy embeddings
  {
    const size_t embedding_size = ctx->embedding.size();

    memcpy(out, &embedding_size, sizeof(embedding_size));
    out += sizeof(embedding_size);

    if (embedding_size) {
      memcpy(out, ctx->embedding.data(), embedding_size * sizeof(float));
      out += embedding_size * sizeof(float);
    }
  }

  // copy kv cache
  {
    const auto& kv_self = ctx->model.kv_self;
    const auto& hparams = ctx->model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = hparams.n_ctx;

    const size_t kv_size = kv_self.buf.size;
    const int kv_ntok = model_get_kv_cache_token_count(ctx);

    memcpy(out, &kv_size, sizeof(kv_size));
    out += sizeof(kv_size);
    memcpy(out, &kv_ntok, sizeof(kv_ntok));
    out += sizeof(kv_ntok);

    if (kv_size) {
      const size_t elt_size = ne_element_size(kv_self.k);

      char buffer[4096];

      ne_context* cpy_ctx = ne_init({sizeof(buffer), buffer, /* no_alloc */ true});
      ne_cgraph gf{};
      gf.n_threads = 1;

      ne_tensor* kout3d = ne_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer, NE_SIZE_CALC);
      kout3d->data = out;
      out += ne_nbytes(kout3d);

      ne_tensor* vout3d = ne_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer, NE_SIZE_CALC);
      vout3d->data = out;
      out += ne_nbytes(vout3d);

      ne_tensor* k3d =
          ne_view_3d(cpy_ctx, kv_self.k, n_embd, kv_ntok, n_layer, elt_size * n_embd, elt_size * n_embd * n_ctx, 0);

      ne_tensor* v3d =
          ne_view_3d(cpy_ctx, kv_self.v, kv_ntok, n_embd, n_layer, elt_size * n_ctx, elt_size * n_ctx * n_embd, 0);

      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, k3d, kout3d));
      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, v3d, vout3d));
      ne_graph_compute(cpy_ctx, &gf);

      ne_free(cpy_ctx);
    }
  }

  const size_t written = out - dst;
  const size_t max_size = model_get_state_size(ctx);

  MODEL_ASSERT(written <= max_size);

  return written;
}

// Sets the state reading from the specified source address
size_t model_set_state_data(struct model_context* ctx, uint8_t* src) {
  uint8_t* inp = src;

  // set rng
  {
    size_t rng_size;
    char rng_buf[MODEL_MAX_RNG_STATE];

    memcpy(&rng_size, inp, sizeof(rng_size));
    inp += sizeof(rng_size);
    memcpy(&rng_buf[0], inp, MODEL_MAX_RNG_STATE);
    inp += MODEL_MAX_RNG_STATE;

    std::stringstream rng_ss;
    rng_ss.str(std::string(&rng_buf[0], rng_size));
    rng_ss >> ctx->rng;

    MODEL_ASSERT(rng_ss.fail() == false);
  }

  // set logits
  {
    size_t logits_cap;
    size_t logits_size;

    memcpy(&logits_cap, inp, sizeof(logits_cap));
    inp += sizeof(logits_cap);
    memcpy(&logits_size, inp, sizeof(logits_size));
    inp += sizeof(logits_size);

    MODEL_ASSERT(ctx->logits.capacity() == logits_cap);

    if (logits_size) {
      ctx->logits.resize(logits_size);
      memcpy(ctx->logits.data(), inp, logits_size * sizeof(float));
    }

    inp += logits_cap * sizeof(float);
  }

  // set embeddings
  {
    size_t embedding_size;

    memcpy(&embedding_size, inp, sizeof(embedding_size));
    inp += sizeof(embedding_size);

    MODEL_ASSERT(ctx->embedding.capacity() == embedding_size);

    if (embedding_size) {
      memcpy(ctx->embedding.data(), inp, embedding_size * sizeof(float));
      inp += embedding_size * sizeof(float);
    }
  }

  // set kv cache
  {
    const auto& kv_self = ctx->model.kv_self;
    const auto& hparams = ctx->model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = hparams.n_ctx;

    size_t kv_size;
    int kv_ntok;

    memcpy(&kv_size, inp, sizeof(kv_size));
    inp += sizeof(kv_size);
    memcpy(&kv_ntok, inp, sizeof(kv_ntok));
    inp += sizeof(kv_ntok);

    if (kv_size) {
      MODEL_ASSERT(kv_self.buf.size == kv_size);

      const size_t elt_size = ne_element_size(kv_self.k);

      char buffer[4096];

      ne_context* cpy_ctx = ne_init({sizeof(buffer), buffer, /* no_alloc */ true});
      ne_cgraph gf{};
      gf.n_threads = 1;

      ne_tensor* kin3d = ne_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer, NE_SIZE_CALC);
      kin3d->data = (void*)inp;
      inp += ne_nbytes(kin3d);

      ne_tensor* vin3d = ne_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer, NE_SIZE_CALC);
      vin3d->data = (void*)inp;
      inp += ne_nbytes(vin3d);

      ne_tensor* k3d =
          ne_view_3d(cpy_ctx, kv_self.k, n_embd, kv_ntok, n_layer, elt_size * n_embd, elt_size * n_embd * n_ctx, 0);

      ne_tensor* v3d =
          ne_view_3d(cpy_ctx, kv_self.v, kv_ntok, n_embd, n_layer, elt_size * n_ctx, elt_size * n_ctx * n_embd, 0);

      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, kin3d, k3d));
      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, vin3d, v3d));
      ne_graph_compute(cpy_ctx, &gf);

      ne_free(cpy_ctx);
    }

    ctx->model.kv_self.n = kv_ntok;
  }

  const size_t nread = inp - src;
  const size_t max_size = model_get_state_size(ctx);

  MODEL_ASSERT(nread <= max_size);

  return nread;
}

bool model_load_session_file(struct model_context* ctx, const char* path_session, model_token* tokens_out,
                             size_t n_token_capacity, size_t* n_token_count_out) {
  model_file file(path_session, "rb");

  // sanity checks
  {
    const uint32_t magic = file.read_u32();
    const uint32_t version = file.read_u32();

    if (magic != MODEL_SESSION_MAGIC || version != MODEL_SESSION_VERSION) {
      fprintf(stderr, "%s : unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
      return false;
    }

    model_hparams session_hparams;
    file.read_raw(&session_hparams, sizeof(model_hparams));

    if (session_hparams != ctx->model.hparams) {
      fprintf(stderr, "%s : model hparams didn't match from session file!\n", __func__);
      return false;
    }
  }

  // load the prompt
  {
    const uint32_t n_token_count = file.read_u32();

    if (n_token_count > n_token_capacity) {
      fprintf(stderr, "%s : token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count,
              n_token_capacity);
      return false;
    }

    file.read_raw(tokens_out, sizeof(model_token) * n_token_count);
    *n_token_count_out = n_token_count;
  }

  // restore the context state
  {
    const size_t n_state_size_cur = file.size - file.tell();
    const size_t n_state_size_max = model_get_state_size(ctx);

    if (n_state_size_cur > n_state_size_max) {
      fprintf(stderr, "%s : the state size in session file is too big! max %zu, got %zu\n", __func__, n_state_size_max,
              n_state_size_cur);
      return false;
    }

    std::vector<uint8_t> state_data(n_state_size_max);
    file.read_raw(state_data.data(), n_state_size_cur);

    model_set_state_data(ctx, state_data.data());
  }

  return true;
}

bool model_save_session_file(struct model_context* ctx, const char* path_session, const model_token* tokens,
                             size_t n_token_count) {
  model_file file(path_session, "wb");

  file.write_u32(MODEL_SESSION_MAGIC);
  file.write_u32(MODEL_SESSION_VERSION);

  file.write_raw(&ctx->model.hparams, sizeof(model_hparams));

  // save the prompt
  file.write_u32((uint32_t)n_token_count);
  file.write_raw(tokens, sizeof(model_token) * n_token_count);

  // save the context state
  {
    const size_t n_state_size_max = model_get_state_size(ctx);

    std::vector<uint8_t> state_data(n_state_size_max);
    const size_t n_state_size_cur = model_copy_state_data(ctx, state_data.data());

    file.write_raw(state_data.data(), n_state_size_cur);
  }

  return true;
}

int model_tokenize(struct model_context* ctx, const char* text, model_token* tokens, int n_max_tokens, bool add_bos) {
  auto res = model_tokenize(ctx->vocab, text, add_bos);

  if (n_max_tokens < (int)res.size()) {
    fprintf(stderr, "%s: too many tokens\n", __func__);
    return -((int)res.size());
  }

  for (size_t i = 0; i < res.size(); i++) {
    tokens[i] = res[i];
  }

  return res.size();
}

std::vector<model_token> model_tokenize(struct model_context* ctx, const std::string& text, bool add_bos) {
  // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
  std::vector<model_token> res(text.size() + (int)add_bos);
  const int n = model_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
  assert(n >= 0);
  res.resize(n);

  return res;
}

int model_n_vocab(const struct model_context* ctx) { return ctx->vocab.id_to_token.size(); }

int model_n_ctx(const struct model_context* ctx) { return ctx->model.hparams.n_ctx; }

int model_n_embd(const struct model_context* ctx) { return ctx->model.hparams.n_embd; }

float* model_get_logits(struct model_context* ctx) { return ctx->logits.data(); }

float* model_get_embeddings(struct model_context* ctx) { return ctx->embedding.data(); }

const char* model_token_to_str(const struct model_context* ctx, model_token token) {
  if (token >= model_n_vocab(ctx)) {
    return nullptr;
  }

  return ctx->vocab.id_to_token[token].tok.c_str();
}

model_token model_token_bos() { return 1; }

model_token model_token_eos() { return 2; }

model_token model_token_nl() { return 13; }

void model_print_timings(struct model_context* ctx) {
  const int64_t t_end_us = ne_time_us();

  const int32_t n_sample = std::max(1, ctx->n_sample);
  const int32_t n_eval = std::max(1, ctx->n_eval);
  const int32_t n_p_eval = std::max(1, ctx->n_p_eval);

  fprintf(stderr, "\n");
  fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0);
  fprintf(stderr, "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per token)\n", __func__,
          1e-3 * ctx->t_sample_us, n_sample, 1e-3 * ctx->t_sample_us / n_sample);
  fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__,
          1e-3 * ctx->t_p_eval_us, n_p_eval, 1e-3 * ctx->t_p_eval_us / n_p_eval);
  fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per token)\n", __func__,
          1e-3 * ctx->t_eval_us, n_eval, 1e-3 * ctx->t_eval_us / n_eval);
  fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us) / 1000.0);
  printf("========== eval time log of each prediction ==========\n");
  for (int i = 0; i < ctx->eval_times.size(); ++i) {
    printf("prediction %3d, time: %.2fms\n", i, ctx->eval_times[i] / 1000.0f);
  }
}

void model_reset_timings(struct model_context* ctx) {
  ctx->t_start_us = ne_time_us();
  ctx->t_sample_us = ctx->n_sample = 0;
  ctx->t_eval_us = ctx->n_eval = 0;
  ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char* model_print_system_info(void) {
  static std::string s;

  s = "";
  s += "AVX = " + std::to_string(ne_cpu_has_avx()) + " | ";
  s += "AVX2 = " + std::to_string(ne_cpu_has_avx2()) + " | ";
  s += "AVX512 = " + std::to_string(ne_cpu_has_avx512()) + " | ";
  s += "AVX512_VBMI = " + std::to_string(ne_cpu_has_avx512_vbmi()) + " | ";
  s += "AVX512_VNNI = " + std::to_string(ne_cpu_has_avx512_vnni()) + " | ";
  s += "FMA = " + std::to_string(ne_cpu_has_fma()) + " | ";
  s += "F16C = " + std::to_string(ne_cpu_has_f16c()) + " | ";
  s += "BLAS = " + std::to_string(ne_cpu_has_blas()) + " | ";
  s += "SSE3 = " + std::to_string(ne_cpu_has_sse3()) + " | ";
  s += "VSX = " + std::to_string(ne_cpu_has_vsx()) + " | ";

  return s.c_str();
}

// For internal test use
std::vector<std::pair<std::string, struct ne_tensor*>>& model_internal_get_tensor_map(struct model_context* ctx) {
  return ctx->model.tensors_by_name;
}

// beam search
// A struct for calculating logits-related info.
struct logits_info {
  const model_context* const ctx = nullptr;
  // (batch, seq_len * vocab_size)
  const float* const logits = nullptr;
  const int batch_size;
  const int32_t n_vocab;
  // last seq_len indice
  // only handle last seq_len indice
  const size_t offset;
  const size_t bs_stride;
  // max logit (batch,)
  std::vector<float> max_ls;
  // 1 / exp sum (batch,)
  std::vector<float> normalizers;
  struct sum_exp {
    float max_l;
    float operator()(float sum, float l) const { return sum + std::exp(l - max_l); }
  };

  logits_info(struct model_context* lctx)
      : ctx(lctx),
        logits(model_get_logits(lctx)),
        batch_size(lctx->batch_size),
        n_vocab(lctx->model.hparams.n_vocab),
        offset(lctx->logits.size() / lctx->batch_size - n_vocab),
        bs_stride(lctx->logits.size() / lctx->batch_size) {
    max_ls.resize(lctx->batch_size);
    normalizers.resize(lctx->batch_size);
    MODEL_ASSERT(lctx->logits.size() % lctx->batch_size == 0);
    // batch
#pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
      max_ls[i] = *std::max_element(logits + i * bs_stride + offset, logits + i * bs_stride + offset + n_vocab);
      normalizers[i] = 1.0f / std::accumulate(logits + i * bs_stride + offset,
                                              logits + i * bs_stride + offset + n_vocab, 0.0f, sum_exp{max_ls[i]});
    }
  }

  model_token_data get_token_data(const int& batch_idx, const int32_t& token_idx) const {
    return {token_idx, *(logits + batch_idx * bs_stride + offset + token_idx), 0.0f};
  }

  // Return top k token_data by logit. (batch, top_k)
  std::vector<std::vector<model_token_data>> top_k(const int& k) {
    std::vector<std::vector<model_token_data>> min_heap(batch_size);  // min-heap by logit
    int tk = std::min(k, n_vocab);
    // min_heap.reserve(batch_size * tk);
    for (int idx = 0; idx < batch_size; ++idx) {
      for (int32_t token_idx = 0; token_idx < tk; ++token_idx) {
        min_heap[idx].push_back(get_token_data(idx, token_idx));
      }
    }
    auto comp = [](const model_token_data& a, const model_token_data& b) { return a.logit > b.logit; };
    for (int idx = 0; idx < batch_size; ++idx) {
      std::make_heap(min_heap[idx].begin(), min_heap[idx].end(), comp);
      for (int32_t token_idx = tk; token_idx < n_vocab; ++token_idx) {
        if (min_heap[idx].front().logit < get_token_data(idx, token_idx).logit) {
          std::pop_heap(min_heap[idx].begin(), min_heap[idx].end(), comp);
          min_heap[idx].back().id = token_idx;
          min_heap[idx].back().logit = get_token_data(idx, token_idx).logit;
          std::push_heap(min_heap[idx].begin(), min_heap[idx].end(), comp);
        }
      }
    }
    return min_heap;
  }

  float probability_from_logit(const int& batch_idx, const float& logit) {
    return normalizers[batch_idx] * std::exp(logit - max_ls[batch_idx]);
  }

  float log_probability_from_logit(const int& batch_idx, const float& logit) {
    return std::log(probability_from_logit(batch_idx, logit));
  }
};

void logits_processor::min_new_tokens_logits_process(const uint32_t& cur_len, const model_vocab::id& eos_token_id) {
  MODEL_ASSERT(ctx->generation_conf.min_new_tokens >= 0);
  if (ctx->generation_conf.min_new_tokens == 0 || ctx->generation_conf.min_new_tokens <= cur_len) {
    return;
  } else {
    int batch_size = ctx->batch_size;
    size_t offset = ctx->logits.size() / ctx->batch_size - ctx->model.hparams.n_vocab;
    size_t bs_stride = ctx->logits.size() / ctx->batch_size;
    for (int i = 0; i < batch_size; ++i) {
      // forbidden to choose eos_token if cur_len < min_new_tokens
      *(model_get_logits(ctx) + i * bs_stride + offset + eos_token_id) = 0.0f;
    }
  }
}

void logits_processor::process(const uint32_t& cur_len, const model_vocab::id& eos_token_id) {
  MODEL_ASSERT(model_get_logits(ctx) != nullptr);
  if (min_new_tokens > 0) {
    min_new_tokens_logits_process(cur_len, eos_token_id);
  }
}

//  TODO dispatch JBLAS kv cache manager
void beam_search_kv_cache_reorder::update(const uint32_t& n_past, const uint32_t& n_prompt_tokens,
                                          const std::unordered_map<int, int>& kv_reorder_indices,
                                          const std::vector<beam>& next_beams) {
  // first step
  if (n_past == n_prompt_tokens) {
    // cpy batch 1 to all batches
#pragma omp parallel for
    for (int i = 0; i < ctx->model.layers.size(); ++i) {
      for (int j = 1; j < kv_n_ctx_block; ++j) {
        // [n_embd, N]
        memcpy(static_cast<char*>(ctx->model.kv_self.k->data) +
                   (i * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd * kv_n_ctx_block +
                    j * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd),
               static_cast<char*>(ctx->model.kv_self.k->data) +
                   i * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd * kv_n_ctx_block,
               ne_element_size(ctx->model.kv_self.k) * n_embd * n_prompt_tokens);
        // [N, n_embd]
        for (int k = 0; k < n_embd; ++k) {
          memcpy(static_cast<char*>(ctx->model.kv_self.v->data) +
                     (i * n_ctx * ne_element_size(ctx->model.kv_self.v) * n_embd * kv_n_ctx_block +
                      j * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd +
                      n_ctx * k * ne_element_size(ctx->model.kv_self.v)),
                 static_cast<char*>(ctx->model.kv_self.v->data) +
                     (i * n_ctx * ne_element_size(ctx->model.kv_self.v) * n_embd * kv_n_ctx_block +
                      n_ctx * k * ne_element_size(ctx->model.kv_self.v)),
                 ne_element_size(ctx->model.kv_self.v) * n_prompt_tokens);
        }
      }
    }
  } else if (n_past > n_prompt_tokens) {
    // next setp
    for (auto it : kv_reorder_indices) {
      if (it.first != it.second) {
        uint32_t len = next_beams[it.first].token_ids.size() - 1;
        // last token in beam is for next step inference
        MODEL_ASSERT(len == n_past - n_prompt_tokens);
        size_t input_token_offset_k = n_prompt_tokens * ne_element_size(ctx->model.kv_self.k) * n_embd;
        size_t input_token_offset_v = n_prompt_tokens * ne_element_size(ctx->model.kv_self.v);
        if (len + n_prompt_tokens > n_ctx) {
          // all token hidden states cache should be updated
          input_token_offset_k = 0;
          input_token_offset_v = 0;
          len = n_ctx;
        }
#pragma omp parallel for
        for (int i = 0; i < ctx->model.layers.size(); ++i) {
          // [n_embd, N]
          memcpy(static_cast<char*>(ctx->model.kv_self.k->data) +
                     (i * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd * kv_n_ctx_block +
                      it.first * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd) +
                     input_token_offset_k,
                 static_cast<char*>(ctx->model.kv_self.k->data) +
                     i * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd * kv_n_ctx_block +
                     it.second * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd + input_token_offset_k,
                 ne_element_size(ctx->model.kv_self.k) * n_embd * len);
          // [N, n_embd]
          for (int k = 0; k < n_embd; ++k) {
            memcpy(static_cast<char*>(ctx->model.kv_self.v->data) +
                       (i * n_ctx * ne_element_size(ctx->model.kv_self.v) * n_embd * kv_n_ctx_block +
                        it.first * n_ctx * ne_element_size(ctx->model.kv_self.v) * n_embd +
                        n_ctx * ne_element_size(ctx->model.kv_self.v) * k + input_token_offset_v),
                   static_cast<char*>(ctx->model.kv_self.v->data) +
                       (i * n_ctx * ne_element_size(ctx->model.kv_self.v) * n_embd * kv_n_ctx_block +
                        it.second * n_ctx * ne_element_size(ctx->model.kv_self.v) * n_embd +
                        n_ctx * ne_element_size(ctx->model.kv_self.v) + input_token_offset_v),
                   ne_element_size(ctx->model.kv_self.v) * len);
          }
        }
      }
    }
  } else {
    return;
  }
}

// TODO debug info unify (function ptr?)
void beam_search_flow::fill_next_beams_by_top_probabilities() {
  auto const comp = [](const beam& a, const beam& b) { return a.score > b.score; };
  std::vector<model_token> embd_inp;
  std::vector<int> infer_beam_ids(beam_size);
  int record = 0;
  int batch_size = 0;
  uint32_t cur_len = 0;
  for (int i = 0; i < beam_size; ++i) {
    // is done or not
    if (!cur_beams[i].eos()) {
      if (cur_len != 0) {
        MODEL_ASSERT(cur_len == cur_beams[i].token_ids.size());
      } else {
        cur_len = cur_beams[i].token_ids.size();
      }
      // (batch, 1)
      // ordered by infer_bs_id
      embd_inp.push_back(cur_beams[i].token_ids.back());
      infer_beam_ids[i] = record++;
      batch_size++;
    }
  }
  // DEBUG
#if 0
  printf("====================== \n");
  for (auto kk : embd_inp) {
    printf("%s \n", (ctx->vocab.id_to_token.at(kk).tok).c_str());
  }
#endif
  ctx->batch_size = batch_size;
  int n_tokens = 1;

  model_eval(ctx, embd_inp.data(), n_tokens, n_past, num_threads);
  // DEBUG
#if 0
  size_t bs_stride = n_tokens * ctx->model.hparams.n_vocab;
  for (int k = 0; k < batch_size; ++k) {
    printf("====================== \n");
    for (int kk = 0; kk < 10; ++kk) {
      printf("%4.5f \n", model_get_logits(ctx) + k * bs_stride + kk);
    }
  }
#endif

  lp.process(cur_len, 50256);  //  TODO ctx->model.eos_id;
  logits_info li(ctx);
  //  sample 2
  const int sample_num = 2;
  std::vector<std::vector<model_token_data>> next_tokens = li.top_k(sample_num);
  // DEBUG
#if 0
  for (int k = 0; k < next_tokens.size(); ++k) {
    printf("====================== \n");
    for (auto kk : next_tokens[k]) {
      printf("%s, l: %3.6f, p: %0.6f \n", (ctx->vocab.id_to_token.at(kk.id).tok).c_str(), kk.logit,
             li.log_probability_from_logit(k, kk.logit));
    }
  }
#endif
  MODEL_ASSERT(next_tokens.size() == batch_size);
  for (int i = 0; i < beam_size; ++i) {
    beam b = cur_beams[i];
    if (b.eos()) {
      // b is at end-of-sentence, so just copy it to next_beams if its
      // probability is high enough.
      if (next_beams.size() < beam_size) {
        next_beams.push_back(b);
        if (next_beams.size() == beam_size) {
          std::make_heap(next_beams.begin(), next_beams.end(), comp);
        }
      } else if (next_beams.front().score < b.score) {
        std::pop_heap(next_beams.begin(), next_beams.end(), comp);
        next_beams.back() = b;
        std::push_heap(next_beams.begin(), next_beams.end(), comp);
      }
    } else {
      int j = 0;
      if (next_beams.size() < beam_size) {
        for (; next_beams.size() < beam_size && j < sample_num; ++j) {
          beam next_beam = b;
          next_beam.token_ids.push_back(next_tokens[infer_beam_ids[i]][j].id);
          next_beam.score += li.log_probability_from_logit(infer_beam_ids[i], next_tokens[infer_beam_ids[i]][j].logit);
          next_beams.push_back(std::move(next_beam));
        }
        std::make_heap(next_beams.begin(), next_beams.end(), comp);
      }
      for (; j < sample_num; ++j) {
        float const next_score =
            b.score + li.log_probability_from_logit(infer_beam_ids[i], next_tokens[infer_beam_ids[i]][j].logit);
        if (next_beams.front().score < next_score) {
          std::pop_heap(next_beams.begin(), next_beams.end(), comp);
          next_beams.back() = b;
          next_beams.back().token_ids.push_back(next_tokens[infer_beam_ids[i]][j].id);
          next_beams.back().score = next_score;
          std::push_heap(next_beams.begin(), next_beams.end(), comp);
        }
      }
    }
  }
  std::sort(next_beams.begin(), next_beams.end(), [](beam& a, beam& b) { return a.infer_bs_id < b.infer_bs_id; });
}

// get kv cache reorder indices,
// k: dst_beam batch idx, v: src_beam batch idx
// for copy predicted past token kv cache
// for example:
//     - c
// a -|    ---------->|
//     - d            |       - ac
//                    | ---> |      (beam_size = 2)
//     - f            |       - ad
// b -|    ---------->|
//     - g
// kv_cache_reorder_indices = {0:0, 1:0}
// if kv_cache_reorder_indices = {0:0, 1:1}, then do not need reorder (cpy)
std::unordered_map<int, int> beam_search_flow::update_kv_cache_reorder_indices() {
  MODEL_ASSERT(next_beams.size() == beam_size);
  MODEL_ASSERT(cur_beams.size() == beam_size);
  // DEBUG
#if 0
  printf("cur_beams: ");
  for (int i = 0; i < beam_size; ++i) {
    printf("%d, ", cur_beams[i].infer_bs_id);
  }
  printf("\n");
  printf("next_beams: ");
  for (int i = 0; i < beam_size; ++i) {
    printf("%d, ", next_beams[i].infer_bs_id);
  }
  printf("\n");
#endif
  std::unordered_map<int, int> kv_reorder_indices;
  kv_reorder_indices.reserve(beam_size);
  // shuffle beams which are early stopped (eos)
  // keep them behind beams which have non-eos
  // next_beams infer_bs_id: [0, 1(eos), 2(eos), 3] - > [0, 3, 1(eos), 2(eos)]
  std::vector<int> cpy_eos_bs_ids;
  std::vector<int> cpy_final_bs_ids;
  std::vector<int> nb_eos_ids;
  std::vector<int> nb_shuffle_ids;
  cpy_final_bs_ids.reserve(beam_size);
  for (int i = 0; i < beam_size; ++i) {
    MODEL_ASSERT(cur_beams[i].infer_bs_id == i);
    if (next_beams[i].eos()) {
      cpy_eos_bs_ids.push_back(next_beams[i].infer_bs_id);
      nb_eos_ids.push_back(i);
    } else {
      cpy_final_bs_ids.push_back(next_beams[i].infer_bs_id);
      nb_shuffle_ids.push_back(i);
    }
  }
  cpy_final_bs_ids.insert(cpy_final_bs_ids.end(), cpy_eos_bs_ids.begin(), cpy_eos_bs_ids.end());
  nb_shuffle_ids.insert(nb_shuffle_ids.end(), nb_eos_ids.begin(), nb_eos_ids.end());

  // update indices and batch ids
  for (int i = 0; i < beam_size; ++i) {
    kv_reorder_indices[i] = cpy_final_bs_ids[i];
    // update infer_bs_id before next beam generation
    next_beams[nb_shuffle_ids[i]].infer_bs_id = i;
  }
  // beams should be ordered by batch id
  std::sort(next_beams.begin(), next_beams.end(), [](beam& a, beam& b) { return a.infer_bs_id < b.infer_bs_id; });
#if 0  // DEBUG
  printf("cpy_final_bs_ids: ");
  for (int i = 0; i < beam_size; ++i) {
    printf("%d, ", cpy_final_bs_ids[i]);
  }
  printf("\n");
  printf("nb_shuffle_ids: ");
  for (int i = 0; i < beam_size; ++i) {
    printf("%d, ", nb_shuffle_ids[i]);
  }
  printf("\n");
  printf("next_beams after: ");
  for (int i = 0; i < beam_size; ++i) {
    printf("%d, ", next_beams[i].infer_bs_id);
  }
  printf("\n");
#endif
  return kv_reorder_indices;
}

void beam_search_flow::beam_score_length_penalize() {
  float length_penalty = ctx->generation_conf.length_penalty;
  std::for_each(cur_beams.begin(), cur_beams.end(),
                [&](beam& b) { b.score /= std::pow(b.token_ids.size(), length_penalty); });
}

// Return beam with highest probability.
const beam& beam_search_flow::top_beam() {
  auto const by_score = [](beam const& a, beam const& b) { return a.score < b.score; };
  return *std::max_element(cur_beams.begin(), cur_beams.end(), by_score);
}

// TODO batch_size = 1 only
// TODO batch prompt processing
std::vector<model_token> beam_search_flow::loop(const model_token* tokens_inp, const int& n_tokens,
                                                const int& n_threads) {
  if (n_tokens > model_n_ctx(ctx)) {
    fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, n_tokens, model_n_ctx(ctx) - 4);
    return std::vector<model_token>();
  }
  num_threads = n_threads;
  std::vector<model_token> beam_search_response;
  std::vector<model_token> embd(tokens_inp, tokens_inp + n_tokens);

  ctx->batch_size = 1;
  const uint32_t max_new_tokens = ctx->generation_conf.max_new_tokens;

  // Loop while there are any beams that have not yet reached end-of-sentence.
  // If the top beam is at end-of-sentence, then finish since all other
  // beam score can only decrease.
  auto const eos = [](const beam& b) { return b.eos(); };
  kv_reorder = ctx->bs_kv_reorder;
  if (kv_reorder == nullptr) {
    kv_reorder = std::make_shared<beam_search_kv_cache_reorder>(ctx);
  }
  for (int n = 0; n < max_new_tokens && !eos(top_beam()) && !std::all_of(cur_beams.begin(), cur_beams.end(), eos);
       ++n) {
    // first step
    if (n_past == 0) {
      model_eval(ctx, embd.data(), n_tokens, n_past, num_threads);
      n_past += n_tokens;
      kv_reorder->update(n_past, n_tokens);
      lp.process(0, 50256);  //  TODO ctx->model.eos_id;
      logits_info li(ctx);
      std::vector<std::vector<model_token_data>> next_tokens = li.top_k(beam_size);
      MODEL_ASSERT(next_tokens.size() == 1);
      cur_beams.clear();
      for (int i = 0; i < beam_size; ++i) {
        beam b;
        b.ctx = ctx;
        b.token_ids.push_back(next_tokens[0][i].id);
        b.score = li.log_probability_from_logit(0, next_tokens[0][i].logit);
        b.infer_bs_id = i;
        cur_beams.push_back(b);
      }
      beam_score_length_penalize();
    } else {
      fill_next_beams_by_top_probabilities();
      std::unordered_map<int, int> kv_reorder_indices = update_kv_cache_reorder_indices();
      n_past += 1;
      kv_reorder->update(n_past, n_tokens, kv_reorder_indices, next_beams);
      cur_beams.swap(next_beams);
      next_beams.clear();
      beam_score_length_penalize();
    }

#if 0  // DEBUG: print current beams for this iteration
    printf("\n\nCurrent beams:\n");
    for (size_t j = 0; j < beams.size(); ++j) {
      printf("beams[%d]: ", j);
      beams[j].print();
      fflush(stdout);
    }
#endif
  }

  const beam& top_b = top_beam();

#if 0  // DEBUG: print final beam result
    printf("\n\nFinal beam:\n");
    top_b.print();
#endif

  beam_search_response.clear();
  for (const auto& id : top_b.token_ids) {
    beam_search_response.push_back(id);
  }
  return beam_search_response;
}

std::vector<model_token> beam_search(model_context* lctx, const int& n_predict, const model_token* tokens_inp,
                                     const int& n_tokens, const int& n_threads) {
  lctx->generation_conf.max_new_tokens = n_predict;
  beam_search_flow bsf(lctx);
  return bsf.loop(tokens_inp, n_tokens, n_threads);
}
