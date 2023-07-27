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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/data_types.h"
#include "core/ne.h"
#include "core/ne_layers.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"

#define GPT_J_TOKEN_EOS 50256

// evaluate the transformer
//
//   - lctx:      model context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool gptj_model_eval_internal(model_context& lctx, const model_token* tokens, const int n_tokens,
                                     const int n_past, const int n_threads) {
  // // enforce that the first token is BOS
  // if (n_past == 0 && tokens[0] != model_token_bos()) {
  //   fprintf(stderr, "%s: first token must be BOS\n", __func__);
  //   return false;
  // }

  const int64_t t_start_us = ne_time_us();

  const int batch_size = lctx.batch_size;
  const int N = n_tokens;

  const auto& model = lctx.model;
  const auto& hparams = model.hparams;

  const auto& kv_self = model.kv_self;

  MODEL_ASSERT(!!kv_self.ctx);

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = hparams.n_ctx;
  const int n_head = hparams.n_head;
  const int n_vocab = hparams.n_vocab;
  const int n_rot = hparams.n_rot;

  auto& mem_per_token = lctx.mem_per_token;
  auto& buf_compute = lctx.buf_compute;

  struct ne_init_params params = {
      /*.mem_size   =*/buf_compute.size,
      /*.mem_buffer =*/buf_compute.addr,
      /*.no_alloc   =*/false,
  };

  struct ne_context* ctx0 = ne_init(params);

  // for big prompts, if BLAS is enabled, it is better to use only one thread
  // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
  ne_cgraph gf = {};
  gf.n_threads = N >= 32 && ne_cpu_has_blas() ? 1 : n_threads;

  struct ne_tensor* embd = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, N * batch_size);
  ne_set_name(embd, "embd");
  for (int i = 0; i < batch_size; ++i) {
    memcpy(static_cast<model_token*>(embd->data) + i * N * ne_element_size(embd),
           tokens + i * N * ne_element_size(embd), N * ne_element_size(embd));
  }

  struct ne_tensor* inpL = ne_get_rows(ctx0, model.others[0], embd);

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;

    lctx.use_buf(ctx0, 0);

    // norm
    cur = ne_norm(ctx0, inpL);

    // cur = ln_1_g*cur + ln_1_b
    cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[0], cur), cur),
                 ne_repeat(ctx0, model.layers[il].norm[1], cur));

    struct ne_tensor* inpSA = cur;

    ne_tensor *Qcur, *Kcur, *Vcur;
    if (model.layers[il].attn[0]->type == NE_TYPE_JBLAS) {  // fused execution of QKV
      struct ne_tensor* QKVcur =
          ne_mul_qkv(ctx0, model.layers[il].attn[0], model.layers[il].attn[1], model.layers[il].attn[2], cur);
      Qcur = ne_rope_inplace(
          ctx0,
          ne_reshape_4d(ctx0, ne_view_1d(ctx0, QKVcur, N * n_embd, 0 * N * n_embd * ne_element_size(QKVcur)),
                        n_embd / n_head, n_head, N, batch_size),
          n_past, n_rot, 0);
      Kcur = ne_rope_inplace(
          ctx0,
          ne_reshape_4d(ctx0, ne_view_1d(ctx0, QKVcur, N * n_embd, 1 * N * n_embd * ne_element_size(QKVcur)),
                        n_embd / n_head, n_head, N, batch_size),
          n_past, n_rot, 0);
      Vcur = ne_view_1d(ctx0, QKVcur, N * n_embd, 2 * N * n_embd * ne_element_size(QKVcur));

    } else {
      Qcur = ne_rope_inplace(
          ctx0,
          ne_reshape_4d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[0], cur), n_embd / n_head, n_head, N, batch_size),
          n_past, n_rot, 0);
      Kcur = ne_rope_inplace(
          ctx0,
          ne_reshape_4d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[1], cur), n_embd / n_head, n_head, N, batch_size),
          n_past, n_rot, 0);
      Vcur = ne_mul_mat(ctx0, model.layers[il].attn[2], cur);
    }
    ne_set_name(Qcur, "Qcur");
    ne_set_name(Kcur, "Kcur");
    ne_set_name(Vcur, "Vcur");
    // self-attention
    // store key and value to memory
    // important: storing RoPE-ed version of K in the KV cache!
    {
      std::vector<ne_tensor*> Kcur_bs(batch_size);
      std::vector<ne_tensor*> Vcur_bs(batch_size);
      std::vector<ne_tensor*> k_bs(batch_size);
      std::vector<ne_tensor*> v_bs(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        // batch K
        Kcur_bs[i] = ne_view_4d(ctx0, Kcur, n_embd / n_head, n_head, N, 1, ne_element_size(Kcur) * n_embd / n_head,
                                ne_element_size(Kcur) * n_embd, ne_element_size(Kcur) * n_embd * N,
                                i * ne_element_size(Kcur) * n_embd * N);
        k_bs[i] = ne_view_1d(ctx0, kv_self.k, n_embd * N * 1,
                             (ne_element_size(kv_self.k) * n_embd) * (il * n_ctx * batch_size + n_past) +
                                 i * n_ctx * n_embd * ne_element_size(kv_self.k));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur_bs[i], k_bs[i]));

        // batch V
        Vcur_bs[i] = ne_permute(ctx0,
                                ne_reshape_4d(ctx0,
                                              ne_view_2d(ctx0, Vcur, n_embd, N, ne_element_size(Vcur) * n_embd,
                                                         i * ne_element_size(Vcur) * n_embd * N),
                                              n_embd / n_head, n_head, N, 1),
                                1, 2, 0, 3);
        v_bs[i] = ne_view_4d(ctx0, kv_self.v, N, n_embd / n_head, n_head, 1, n_ctx * ne_element_size(kv_self.v),
                             n_ctx * ne_element_size(kv_self.v) * n_embd / n_head,
                             n_ctx * ne_element_size(kv_self.v) * n_embd,
                             ((il * n_ctx) * ne_element_size(kv_self.v) * n_embd * batch_size +
                              i * n_ctx * n_embd * ne_element_size(kv_self.v) + n_past * ne_element_size(kv_self.v)));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur_bs[i], v_bs[i]));
      }
    }

    struct ne_tensor* Q = ne_permute(ctx0, Qcur, 0, 2, 1, 3);
    ne_set_name(Q, "Q");

    struct ne_tensor* K =
        ne_permute(ctx0,
                   ne_view_4d(ctx0, kv_self.k, n_embd / n_head, n_head, (n_past + N), batch_size,
                              ne_element_size(kv_self.k) * n_embd / n_head, ne_element_size(kv_self.k) * n_embd,
                              ne_element_size(kv_self.k) * n_embd * n_ctx,
                              il * n_ctx * ne_element_size(kv_self.k) * n_embd * batch_size),
                   0, 2, 1, 3);
    ne_set_name(K, "K");

    // K * Q
    struct ne_tensor* KQ = ne_mul_mat(ctx0, K, Q);
    ne_set_name(KQ, "KQ");

    // KQ_scaled = KQ / sqrt(n_embd/n_head)
    struct ne_tensor* KQ_scale = ne_new_f32(ctx0, 1.0f / sqrtf(float(n_embd) / n_head));
    ne_set_name(KQ_scale, "1/sqrt(n_embd/n_head)");

    // KQ_scaled shape [n_past + N, N, n_head, 1]
    struct ne_tensor* KQ_scaled = ne_scale_inplace(ctx0, KQ, KQ_scale);
    ne_set_name(KQ_scaled, "KQ_scaled");

    // KQ_masked = mask_past(KQ_scaled)
    struct ne_tensor* KQ_masked = ne_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
    ne_set_name(KQ_masked, "KQ_masked");

    // KQ = soft_max(KQ_masked)
    struct ne_tensor* KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_masked);
    ne_set_name(KQ_soft_max, "KQ_soft_max");

    // split cached V into n_head heads
    struct ne_tensor* V = ne_view_4d(
        ctx0, kv_self.v, (n_past + N), n_embd / n_head, n_head, batch_size, n_ctx * ne_element_size(kv_self.v),
        n_ctx * ne_element_size(kv_self.v) * n_embd / n_head, n_ctx * ne_element_size(kv_self.v) * n_embd,
        il * n_ctx * ne_element_size(kv_self.v) * n_embd * batch_size);
    ne_set_name(V, "V");

    struct ne_tensor* KQV = ne_mul_mat(ctx0, V, KQ_soft_max);
    ne_set_name(KQV, "KQV");

    // KQV_merged = KQV.permute(0, 2, 1, 3)
    struct ne_tensor* KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);
    ne_set_name(KQV_merged, "KQV_merged");

    // cur = KQV_merged.contiguous().view(n_embd, N)
    struct ne_tensor* KQV_merged_contiguous =
        ne_cpy(ctx0, KQV_merged, ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N * batch_size, NE_SIZE_CALC));
    ne_set_name(KQV_merged_contiguous, "KQV_merged_contiguous");

    // projection (no bias)
    struct ne_tensor* KQV_out = ne_mul_mat(ctx0, model.layers[il].attn[3], KQV_merged_contiguous);
    ne_set_name(KQV_out, "KQV_out");

    lctx.use_buf(ctx0, 1);
    struct ne_tensor* inpFF = KQV_out;

    // feed-forward network
    if (model.layers[il].ffn[0]->type == NE_TYPE_JBLAS && model.layers[il].ffn[2]->type == NE_TYPE_JBLAS) {
      const int64_t in_ne[4] = {model.layers[il].ffn[0]->ne[1], inpSA->ne[1], model.layers[il].ffn[0]->ne[2],
                                inpSA->ne[3]};
      struct ne_tensor* FFN_in =
          ne_new_tensor(ctx0, NE_TYPE_F32, MIN(model.layers[il].ffn[0]->n_dims, inpSA->n_dims), in_ne, NE_SIZE_CALC);
      const int64_t out_ne[4] = {model.layers[il].ffn[2]->ne[1], FFN_in->ne[1], model.layers[il].ffn[2]->ne[2],
                                 FFN_in->ne[3]};
      struct ne_tensor* FFN_out =
          ne_new_tensor(ctx0, NE_TYPE_F32, MIN(model.layers[il].ffn[2]->n_dims, FFN_in->n_dims), out_ne, NE_SIZE_CALC);
      cur = ne_ffn_add_gelu(ctx0, model.layers[il].ffn[0], model.layers[il].ffn[2],
                            ne_repeat(ctx0, model.layers[il].ffn[1], FFN_in),
                            ne_repeat(ctx0, model.layers[il].ffn[3], FFN_out), inpSA);
    } else {
      struct ne_tensor* FFN_in = ne_mul_mat(ctx0, model.layers[il].ffn[0], inpSA);
      ne_set_name(FFN_in, "FFN_in");

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].ffn[1], FFN_in), FFN_in);

      // GELU activation
      cur = ne_gelu(ctx0, cur);

      struct ne_tensor* FFN_out = ne_mul_mat(ctx0, model.layers[il].ffn[2], cur);
      ne_set_name(FFN_out, "FFN_out");

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].ffn[3], FFN_out), FFN_out);
    }
    cur = ne_add(ctx0, cur, inpFF);

    // input for next layer
    inpL = ne_add(ctx0, cur, inpL);
  }
  lctx.use_buf(ctx0, 0);

  // used at the end to optionally extract the embeddings
  struct ne_tensor* embeddings = NULL;

  // norm
  {
    inpL = ne_norm(ctx0, inpL);

    // inpL = inpL*norm(broadcasted)
    inpL = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.others[1], inpL), inpL),
                  ne_repeat(ctx0, model.others[2], inpL));
  }

  // lm_head
  inpL = ne_mul_mat(ctx0, model.others[3], inpL);
  inpL = ne_add(ctx0, ne_repeat(ctx0, model.others[4], inpL), inpL);

  lctx.use_buf(ctx0, -1);

  // logits -> probs
  // inpL = ne_soft_max_inplace(ctx0, inpL);

  // run the computation
  ne_build_forward_expand(&gf, inpL);
  ne_graph_compute(ctx0, &gf);

#ifdef NE_PERF
  bool engine_profiling_ = (getenv("ENGINE_PROFILING") != NULL);
  if (engine_profiling_) {
    ne_graph_profiling(&gf);
  }
#endif

  // update kv token count
  lctx.model.kv_self.n = n_past + N;

  // extract logits
  {
    auto& logits_out = lctx.logits;

    size_t bs_stride = n_vocab * N;
    if (lctx.logits_all) {
      logits_out.resize(n_vocab * N * batch_size);

      for (int i = 0; i < batch_size; ++i) {
        memcpy(logits_out.data() + i * bs_stride, (float*)ne_get_data(inpL) + (i * bs_stride),
               sizeof(float) * n_vocab * N);
      }
    } else {
      // return result for just the last token
      logits_out.resize(n_vocab * batch_size);
      for (int i = 0; i < batch_size; ++i) {
        memcpy(logits_out.data() + (i * n_vocab), (float*)ne_get_data(inpL) + (i * bs_stride) + (n_vocab * (N - 1)),
               sizeof(float) * n_vocab);
      }
    }
  }

  // extract embeddings
  if (!lctx.embedding.empty()) {
    auto& embedding_out = lctx.embedding;

    embedding_out.resize(n_embd);
    memcpy(embedding_out.data(), (float*)ne_get_data(embeddings) + (n_embd * (N - 1)), sizeof(float) * n_embd);
  }

  if (mem_per_token == 0) {
    mem_per_token = ne_used_mem(ctx0) / N;
  }

  ne_free(ctx0);

  // measure the performance only for the single-token evals
  int64_t time_interval = ne_time_us() - t_start_us;
  if (N == 1) {
    lctx.t_eval_us += time_interval;
    lctx.n_eval++;
  } else if (N > 1) {
    lctx.t_p_eval_us += time_interval;
    lctx.n_p_eval += N;
  }
  lctx.eval_times.push_back(time_interval);

  return true;
}

int model_eval(struct model_context* ctx, const model_token* tokens, int n_tokens, int n_past, int n_threads) {
  if (!gptj_model_eval_internal(*ctx, tokens, n_tokens, n_past, n_threads)) {
    fprintf(stderr, "%s: failed to eval\n", __func__);
    return 1;
  }

  // get a more accurate load time, upon first eval
  // TODO: fix this
  if (!ctx->has_evaluated_once) {
    ctx->t_load_us = ne_time_us() - ctx->t_start_us;
    ctx->has_evaluated_once = true;
  }

  return 0;
}

// TODO: not great allocating this every time
std::vector<model_token> model_tokenize(struct model_context* ctx, const std::string& text, bool add_bos) {
  // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
  std::vector<model_token> res(text.size() + (int)add_bos);
  const int n = model_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
  assert(n >= 0);
  res.resize(n);

  return res;
}

struct model_context* model_init_from_gpt_params(const gpt_params& params) {
  auto lparams = model_context_default_params();

  lparams.name = params.name;
  lparams.n_ctx = params.n_ctx;
  lparams.n_gpu_layers = params.n_gpu_layers;
  lparams.seed = params.seed;
  lparams.f16_kv = params.memory_f16;
  lparams.use_mmap = params.use_mmap;
  lparams.use_mlock = params.use_mlock;
  lparams.logits_all = params.perplexity;
  lparams.embedding = params.embedding;

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

/*  beam search utils  */
// TODO move it to utils folder
struct beam {
  const model_context* ctx;
  std::vector<model_token> token_ids;
  // Cumulative beam probability (renormalized with each token)
  float p;
  // record inference batch indice
  int infer_bs_id;
  // end-of-sentence
  const bool eos() const { return !token_ids.empty() && token_ids.back() == GPT_J_TOKEN_EOS; }
  void print() {
    printf("p: %0.6f, eos: %d, tokens: ", p, eos());
    for (const auto& id : token_ids) {
      printf("%s", (ctx->vocab.id_to_token.at(id).tok).c_str());
    }
    printf("\n");
  }
};

// A struct for calculating logits-related info.
struct logits_info {
  // (batch, seq_len * vocab_size)
  const float* const logits;
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
      : logits(model_get_logits(lctx)),
        batch_size(lctx->batch_size),
        n_vocab(lctx->model.hparams.n_vocab),
        offset(lctx->logits.size() / lctx->batch_size - n_vocab),
        bs_stride(lctx->logits.size() / lctx->batch_size) {
    max_ls.resize(lctx->batch_size);
    normalizers.resize(lctx->batch_size);
    MODEL_ASSERT(lctx->logits.size() % lctx->batch_size == 0);
    // batch
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
};

void fill_next_beams_by_top_probabilities(std::vector<beam>& next_beams, const std::vector<beam>& cur_beams,
                                          const int& beam_size, model_context* lctx, const int& n_threads,
                                          const int& n_past) {
  auto const comp = [](const beam& a, const beam& b) { return a.p > b.p; };
  std::vector<model_token> embd_inp;
  std::vector<int> infer_beam_ids(beam_size);
  int record = 0;
  int batch_size = 0;
  for (int i = 0; i < beam_size; ++i) {
    // is done or not
    if (!cur_beams[i].eos()) {
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
  for (auto kk : embd_inp[k]) {
    printf("%s \n", (lctx->vocab.id_to_token.at(kk).tok).c_str());
  }
#endif
  lctx->batch_size = batch_size;
  int n_tokens = 1;
  model_eval(lctx, embd_inp.data(), n_tokens, n_past, n_threads);
  // DEBUG
#if 0
  size_t bs_stride = n_tokens * lctx->model.hparams.n_vocab;
  for (int k = 0; k < batch_size; ++k) {
    printf("====================== \n");
    for (int kk = 0; kk < 10; ++kk) {
      printf("%4.5f \n", model_get_logits(lctx) + k * bs_stride + kk);
    }
  }
#endif
  logits_info li(lctx);
  // top_k num = beam_size
  std::vector<std::vector<model_token_data>> next_tokens = li.top_k(beam_size);
  // DEBUG
#if 0
  for (int k = 0; k < next_tokens.size(); ++k) {
    printf("====================== \n");
    for (auto kk : next_tokens[k]) {
      printf("%s, l: %3.6f, p: %0.6f \n", (lctx->vocab.id_to_token.at(kk.id).tok).c_str(), kk.logit,
             li.probability_from_logit(k, kk.logit));
    }
  }
#endif
  MODEL_ASSERT(next_tokens.size() == batch_size);
  for (int i = 0; i < beam_size; ++i) {
    beam b = cur_beams[i];
    if (b.eos()) {
      // b is at end-of-sentence, so just copy it to next_beams if its probability is high enough.
      if (next_beams.size() < beam_size) {
        next_beams.push_back(b);
        if (next_beams.size() == beam_size) {
          std::make_heap(next_beams.begin(), next_beams.end(), comp);
        }
      } else if (next_beams.front().p < b.p) {
        std::pop_heap(next_beams.begin(), next_beams.end(), comp);
        next_beams.back() = b;
        std::push_heap(next_beams.begin(), next_beams.end(), comp);
      }
    } else {
      int j = 0;
      if (next_beams.size() < beam_size) {
        for (; next_beams.size() < beam_size; ++j) {
          beam next_beam = b;
          next_beam.token_ids.push_back(next_tokens[infer_beam_ids[i]][j].id);
          next_beam.p *= li.probability_from_logit(infer_beam_ids[i], next_tokens[infer_beam_ids[i]][j].logit);
          next_beams.push_back(std::move(next_beam));
        }
        std::make_heap(next_beams.begin(), next_beams.end(), comp);
      }
      for (; j < beam_size; ++j) {
        float const next_p =
            b.p * li.probability_from_logit(infer_beam_ids[i], next_tokens[infer_beam_ids[i]][j].logit);
        if (next_beams.front().p < next_p) {
          std::pop_heap(next_beams.begin(), next_beams.end(), comp);
          next_beams.back() = b;
          next_beams.back().token_ids.push_back(next_tokens[infer_beam_ids[i]][j].id);
          next_beams.back().p = next_p;
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
std::unordered_map<int, int> update_kv_cache_reorder_indices(std::vector<beam>& next_beams,
                                                             const std::vector<beam>& cur_beams, const int& beam_size) {
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
      MODEL_ASSERT(i = next_beams[i].infer_bs_id);
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
  return kv_reorder_indices;
}

// As beams grow, the cumulative probabilities decrease.
// Renormalize them to avoid floating point underflow.
void renormalize_beam_probabilities(std::vector<beam>& beams) {
  auto const sum_p = [](float sum, beam& b) { return sum + b.p; };
  float const inv_sum = 1.0f / std::accumulate(beams.begin(), beams.end(), 0.0f, sum_p);
  std::for_each(beams.begin(), beams.end(), [inv_sum](beam& b) { b.p *= inv_sum; });
}

// Return beam with highest probability.
const beam& top_beam(std::vector<beam> const& beams) {
  auto const by_p = [](beam const& a, beam const& b) { return a.p < b.p; };
  return *std::max_element(beams.begin(), beams.end(), by_p);
}

// This is deterministic, but can be made probabilistic in
// fill_next_beams_by_top_probabilities() by randomly selecting from all next_beams.
// Not thread-safe.
// TODO make the first step outside?
// TODO batch_size = 4 only
// TODO better way to return?
const model_token* beam_search(const int& beam_size, const int& n_predict, model_context* lctx,
                               const model_token* tokens_inp, const int& n_tokens, const int& n_threads) {
  static std::vector<model_token> beam_search_response(n_tokens);
  memcpy(beam_search_response.data(), tokens_inp, n_tokens * sizeof(model_token));
  printf("%s: beam search in progress ", __func__);
  const int64_t t_start_search_us = ne_time_us();

  // const int32_t n_vocab = lctx->model.hparams.n_vocab;
  size_t n_past = 0;
  std::vector<model_token> embd(n_tokens * beam_size);
  // TODO add params.n_batch?
  for (int i = 0; i < beam_size; ++i) {
    embd.insert(embd.begin(), beam_search_response.begin(), beam_search_response.end());
  }
  lctx->batch_size = beam_size;
  std::vector<beam> beams;
  beams.reserve(beam_size);
  beams.push_back({lctx, {}, 1.0});
  // Init next_beams with unique next token_id each.
  std::vector<beam> next_beams;
  next_beams.reserve(beam_size);
  // Loop while there are any beams that have not yet reached end-of-sentence.
  // If the top beam is at end-of-sentence, then finish since all other
  // beam probabilities can only decrease.
  auto const eos = [](const beam& b) { return b.eos(); };
  for (int i = 0; i < n_predict && !eos(top_beam(beams)) && !std::all_of(beams.begin(), beams.end(), eos); ++i) {
    // first step
    if (n_past == 0) {
      // TODO add -b param for long prompt (memory issue)
      model_eval(lctx, embd.data(), n_tokens, n_past, n_threads);
      n_past += n_tokens;
      // TODO batch_size = 1
      // for (int i = 0; i < lctx->model.layers.size(); ++i) {
      //   int n_ctx = lctx->model.hparams.n_ctx;
      //   int n_embd = lctx->model.hparams.n_embd;
      //   // cpy batch 1 to all batch
      //   for (int j = 1; j < beam_size; ++j) {
      //     memcpy(static_cast<char*>(lctx->model.kv_self.k->data) +
      //                (i * n_ctx * ne_element_size(lctx->model.kv_self.k) * n_embd * beam_size +
      //                 j * n_ctx * ne_element_size(lctx->model.kv_self.k) * n_embd),
      //            static_cast<char*>(lctx->model.kv_self.k->data) +
      //                i * n_ctx * ne_element_size(lctx->model.kv_self.k) * n_embd * beam_size,
      //            ne_element_size(lctx->model.kv_self.k) * n_embd * (n_past));
      //     memcpy(static_cast<char*>(lctx->model.kv_self.v->data) +
      //                (i * n_ctx * ne_element_size(lctx->model.kv_self.v) * n_embd * beam_size +
      //                 j * n_ctx * ne_element_size(lctx->model.kv_self.v) * n_embd),
      //            static_cast<char*>(lctx->model.kv_self.v->data) +
      //                i * n_ctx * ne_element_size(lctx->model.kv_self.v) * n_embd * beam_size,
      //            ne_element_size(lctx->model.kv_self.v) * n_embd * (n_past));
      //   }
      // }

      logits_info li(lctx);
      std::vector<std::vector<model_token_data>> next_tokens = li.top_k(beam_size);
      // MODEL_ASSERT(next_tokens.size() == 2);
      beams.clear();
      for (int i = 0; i < beam_size; ++i) {
        beam b;
        b.token_ids.push_back(next_tokens[0][i].id);
        b.p = li.probability_from_logit(0, next_tokens[0][i].logit);
        b.infer_bs_id = i;
        beams.push_back(b);
      }
      renormalize_beam_probabilities(beams);
    } else {
      fill_next_beams_by_top_probabilities(next_beams, beams, beam_size, lctx, n_threads, n_past);
      std::unordered_map<int, int> kv_reorder_indices = update_kv_cache_reorder_indices(next_beams, beams, beam_size);
      n_past += 1;
      for (int i = 0; i < lctx->model.layers.size(); ++i) {
        int n_ctx = lctx->model.hparams.n_ctx;
        int n_embd = lctx->model.hparams.n_embd;
        for (auto it : kv_reorder_indices) {
          if (it.first != it.second) {
            int input_token_offset = n_tokens * ne_element_size(lctx->model.kv_self.v) * n_embd;
            memcpy(static_cast<char*>(lctx->model.kv_self.k->data) +
                       (i * n_ctx * ne_element_size(lctx->model.kv_self.k) * n_embd * beam_size +
                        it.first * n_ctx * ne_element_size(lctx->model.kv_self.k) * n_embd) +
                       input_token_offset,
                   static_cast<char*>(lctx->model.kv_self.k->data) +
                       i * n_ctx * ne_element_size(lctx->model.kv_self.k) * n_embd * beam_size +
                       it.second * n_ctx * ne_element_size(lctx->model.kv_self.k) * n_embd + input_token_offset,
                   ne_element_size(lctx->model.kv_self.k) * n_embd * (n_past - n_tokens));
            memcpy(static_cast<char*>(lctx->model.kv_self.v->data) +
                       (i * n_ctx * ne_element_size(lctx->model.kv_self.v) * n_embd * beam_size +
                        it.first * n_ctx * ne_element_size(lctx->model.kv_self.v) * n_embd) +
                       input_token_offset,
                   static_cast<char*>(lctx->model.kv_self.v->data) +
                       i * n_ctx * ne_element_size(lctx->model.kv_self.v) * n_embd * beam_size +
                       it.second * n_ctx * ne_element_size(lctx->model.kv_self.k) * n_embd + input_token_offset,
                   ne_element_size(lctx->model.kv_self.v) * n_embd * (n_past - n_tokens));
          }
        }
      }

      beams.swap(next_beams);
      next_beams.clear();
      renormalize_beam_probabilities(beams);
    }

#if 0  // DEBUG: print current beams for this iteration
    printf("\n\nCurrent beams:\n");
    for (size_t j = 0; j < beams.size(); ++j) {
      printf("beams[%d]: ", j);
      fflush(stdout);
      beams[j].print();
    }
#else
    // Show progress
    if (i % 10 == 0) {
      printf(".");
      fflush(stdout);
    }
#endif
  }

  const beam& top_b = top_beam(beams);
  printf(" done \n");

#if 0  // DEBUG: print final beam result
    printf("\n\nFinal beam:\n");
    top_b.print();
#endif

  for (const auto& id : top_b.token_ids) {
    beam_search_response.push_back(id);
  }

  int64_t t_search_us = ne_time_us() - t_start_search_us;
  printf("%s: beam_search time   = %8.2f ms\n", __func__, t_search_us / 1000.0f);
  return beam_search_response.data();
}
