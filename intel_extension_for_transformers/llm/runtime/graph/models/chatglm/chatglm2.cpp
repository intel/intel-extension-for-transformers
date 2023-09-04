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

// evaluate the transformer
//
//   - lctx:      model context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//

static bool chatglm_model_eval_internal(model_context& lctx, const model_token* tokens, const int n_tokens,
                                     const int n_past, const int n_threads) {
  // // enforce that the first token is BOS
  // if (n_past == 0 && tokens[0] != model_token_bos()) {
  //   fprintf(stderr, "%s: first token must be BOS\n", __func__);
  //   return false;
  // }

  const int64_t t_start_us = ne_time_us();

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
  const int n_rot = n_embd / n_head / 2;
  const int head_size = n_embd / n_head;
  const int rope_dim = head_size / 2;
  const int mqa_scale = n_head / hparams.multi_query_group_num;
  const int num_kv_heads = hparams.multi_query_group_num;
  
  const int hidden_size = n_embd;
  const int num_attention_heads = n_head;

  auto& mem_per_token = lctx.mem_per_token;
  auto& buf_compute = lctx.buf_compute;

  struct ne_init_params params = {
      /*.mem_size   =*/buf_compute.size,
      /*.mem_buffer =*/buf_compute.addr,
      /*.no_alloc   =*/false,
  };

  struct ne_context* ctx0 = ne_init(params);

  // for big prochatglms, if BLAS is enabled, it is better to use only one thread
  // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
  ne_cgraph gf = {};
  gf.n_threads = N >= 32 && ne_cpu_has_blas() ? 1 : n_threads;

  struct ne_tensor* embd = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, N);
  ne_set_name(embd, "embd");
  memcpy(embd->data, tokens, N * ne_element_size(embd));
  // int qlen = embd->ne[1];
  struct ne_tensor* inpL = ne_get_rows(ctx0, model.others[0], embd);

  int qlen = inpL->ne[1];
  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;

    lctx.use_buf(ctx0, 0);

    // self-attention
    cur = ne_rms_norm(ctx0, inpL);
    cur = ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[0], cur), cur);
    //cur = ne_mul(ctx0, cur, model.layers[il].norm[0]);
    {
      // compute QKV
      //rintf(" model.layers[il].attn[0]->ne[0] = %d, model.layers[il].attn[0]->ne[1] = %d  ", model.layers[il].attn[0]->ne[0], model.layers[il].attn[0]->ne[1]);
      cur = ne_mul_mat(ctx0, model.layers[il].attn[0], cur);
      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].attn[1], cur), cur);
      //cur = ne_add(ctx0, cur, model.layers[il].attn[1]);

      struct ne_tensor *query_layer =
        ne_view_3d(ctx0, cur, head_size, n_head, N, head_size * ne_element_size(cur), cur->nb[1],
                     0); // [qlen, heads, head_size]
      ne_set_name(query_layer, "query_layer");
      query_layer = ne_rope_inplace(ctx0, query_layer, n_past, rope_dim, 0);
      //query_layer = ne_rope_inplace(ctx0, query_layer, n_past, rope_dim, 0, 0);
      query_layer = ne_cont(ctx0, ne_permute(ctx0, query_layer, 0, 2, 1, 3)); // [heads, qlen, head_size]
      query_layer = ne_reshape_3d(ctx0, query_layer, head_size, mqa_scale * qlen,
                                  num_kv_heads); // [kv_heads, mqa_scale * qlen, head_size]

      struct ne_tensor *key_layer =
        ne_view_3d(ctx0, cur, head_size, num_kv_heads, qlen, head_size * ne_element_size(cur), cur->nb[1],
                     hidden_size * ne_element_size(cur)); // [qlen, kv_heads, head_size]
      ne_set_name(key_layer, "key_layer");
      key_layer = ne_rope_inplace(ctx0, key_layer, n_past, rope_dim, 0);
      //key_layer = ne_rope_inplace(ctx0, key_layer, n_past, rope_dim, 0, 0);
      key_layer = ne_permute(ctx0, key_layer, 0, 2, 1, 3); // [kv_heads, qlen, head_size]

      struct ne_tensor *value_layer =
          ne_view_3d(ctx0, cur, head_size, num_kv_heads, qlen, head_size * ne_element_size(cur), cur->nb[1],
                      (hidden_size + head_size * num_kv_heads) * ne_element_size(cur)); // [qlen, kv_heads, head_size]
      ne_set_name(value_layer, "value_layer");
      value_layer = ne_permute(ctx0, value_layer, 1, 2, 0, 3);                           // [kv_heads, head_size, qlen]

      // store key and value to memory
      // printf("qlen: %d, head_size: %d, num_kv_heads: %d\n", qlen, head_size, num_kv_heads);
      {

        struct ne_tensor *k_cache_view =
            ne_view_3d(ctx0, model.layers[il].k_cache, head_size, qlen, num_kv_heads, model.layers[il].k_cache->nb[1], model.layers[il].k_cache->nb[2],
                        n_past * head_size * ne_element_size(model.layers[il].k_cache)); // [kv_heads, qlen, head_size]
        ne_set_name(k_cache_view, "k_cache_view");
        struct ne_tensor *v_cache_view =
            ne_view_3d(ctx0, model.layers[il].v_cache, qlen, head_size, num_kv_heads, model.layers[il].v_cache->nb[1], model.layers[il].v_cache->nb[2],
                        n_past * ne_element_size(model.layers[il].v_cache)); // [kv_heads, head_size, qlen]
        ne_set_name(v_cache_view, "v_cache_view");
        
        ne_build_forward_expand(&gf, ne_cpy(ctx0, key_layer, k_cache_view));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, value_layer, v_cache_view));
      }

      // concat key & value with past kv
      key_layer = ne_view_3d(ctx0, model.layers[il].k_cache, head_size, n_past + qlen, num_kv_heads, model.layers[il].k_cache->nb[1], model.layers[il].k_cache->nb[2],
                              0); // [kv_heads, klen, head_size]
      value_layer = ne_view_3d(ctx0, model.layers[il].v_cache, n_past + qlen, head_size, num_kv_heads, model.layers[il].v_cache->nb[1], model.layers[il].v_cache->nb[2],
                                0); // [kv_heads, head_size, klen]

      // attention
      struct ne_tensor *attn_scores = ne_mul_mat(ctx0, key_layer, query_layer); // [kv_heads, mqa_scale * qlen, klen]
      ne_set_name(attn_scores, "attn_scores");
      attn_scores = ne_scale_inplace(ctx0, attn_scores, ne_new_f32(ctx0, 1.f / std::sqrt(head_size)));

      if (n_past == 0) {
        // build attention mask for context input
        attn_scores = ne_reshape_3d(ctx0, attn_scores, n_past + qlen, qlen,
                                      num_attention_heads); // [heads, qlen, klen]
        attn_scores = ne_diag_mask_inf_inplace(ctx0, attn_scores, n_past);
        attn_scores = ne_reshape_3d(ctx0, attn_scores, n_past + qlen, mqa_scale * qlen,
                                      num_kv_heads); // [kv_heads, mqa_scale * qlen, klen]
      }

      struct ne_tensor *attn_probs = ne_soft_max_inplace(ctx0, attn_scores); // [kv_heads, mqa_scale * qlen, klen]

      struct ne_tensor *context_layer = ne_mul_mat(ctx0, value_layer, attn_probs); // [kv_heads, mqa_scale * qlen, head_size]
      context_layer = ne_reshape_3d(ctx0, context_layer, head_size, qlen,
                                      num_attention_heads);                           // [heads, qlen, head_size]
      context_layer = ne_cont(ctx0, ne_permute(ctx0, context_layer, 0, 2, 1, 3)); // [qlen, heads, head_size]
      context_layer = ne_reshape_2d(ctx0, context_layer, hidden_size, qlen); // [qlen, hidden]

      // struct ne_tensor *attn_output = dense.forward(ctx0, context_layer);
      cur = ne_mul_mat(ctx0, model.layers[il].attn[2], context_layer);
    }

    lctx.use_buf(ctx0, 1);
    
    struct ne_tensor *hidden_states = ne_add(ctx0, inpL, cur);
    
    // mlp.forward
    struct ne_tensor *mlp_output = ne_rms_norm(ctx0, hidden_states);
    ne_set_name(mlp_output, "mlp_output");
    //mlp_output = ne_mul(ctx0, mlp_output, model.layers[il].norm[1]);
    mlp_output = ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[1], mlp_output), mlp_output);
    
    mlp_output = ne_mul_mat(ctx0, model.layers[il].ffn[0], mlp_output);
    struct ne_tensor *x0 = ne_view_2d(ctx0, mlp_output, mlp_output->ne[0] / 2, mlp_output->ne[1], mlp_output->nb[1], 0);
    x0 = ne_silu(ctx0, x0);
    struct ne_tensor *x1 = ne_view_2d(ctx0, mlp_output, mlp_output->ne[0] / 2, mlp_output->ne[1], mlp_output->nb[1],
                                mlp_output->ne[0] / 2 * ne_element_size(mlp_output));
    ne_set_name(x0, "x0");
    ne_set_name(x1, "x1");
    mlp_output = ne_mul(ctx0, x0, x1);
    mlp_output = ne_mul_mat(ctx0, model.layers[il].ffn[1], mlp_output);

    inpL = ne_add(ctx0, hidden_states, mlp_output);
  }

  lctx.use_buf(ctx0, 0);
  // used at the end to optionally extract the embeddings
  struct ne_tensor* embeddings = NULL;
  // norm
  {
    inpL = ne_rms_norm(ctx0, inpL);
    ne_set_name(inpL, "inpL");
    //inpL = ne_mul(ctx0, inpL, model.others[1]);
    inpL = ne_mul(ctx0, ne_repeat(ctx0, model.others[1], inpL), inpL);
  }

  lctx.use_buf(ctx0, -1);
  if (embd->ne[0] > 1) {
    inpL = ne_view_1d(ctx0, inpL, hidden_size, (embd->ne[0] - 1) * hidden_size * ne_element_size(inpL));
  }
  // lm_head
  inpL = ne_mul_mat(ctx0, model.others[2], inpL);


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

    if (lctx.logits_all) {
      logits_out.resize(n_vocab * N);
      memcpy(logits_out.data(), (float*)ne_get_data(inpL), sizeof(float) * n_vocab * N);
    } else {
      // return result for just the last token
      logits_out.resize(n_vocab);
      memcpy(logits_out.data(), (float*)ne_get_data(inpL), sizeof(float) * n_vocab);
    }
    // printf("logits_out: ");
    // for (int i = 0; i < 20; i++) {
    //   printf("%f, ", logits_out[i]);
    // }
    // printf("\n");
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
  if (!chatglm_model_eval_internal(*ctx, tokens, n_tokens, n_past, n_threads)) {
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
// std::vector<model_token> model_tokenize(struct model_context* ctx, const std::string& text, bool add_bos) {
//   // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
//   std::vector<model_token> res(text.size() + (int)add_bos);
//   const int n = model_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
//   assert(n >= 0);
//   res.resize(n);

//   return res;
// }

// struct model_context* model_init_from_gpt_params(const gpt_params& params) {
//   auto lparams = model_context_default_params();

//   lparams.arch = params.model_arch;
//   lparams.n_ctx = params.n_ctx;
//   lparams.n_gpu_layers = params.n_gpu_layers;
//   lparams.seed = params.seed;
//   lparams.f16_kv = params.memory_f16;
//   lparams.use_mmap = params.use_mmap;
//   lparams.use_mlock = params.use_mlock;
//   lparams.logits_all = params.perplexity;
//   lparams.embedding = params.embedding;

//   model_context* lctx = model_init_from_file(params.model.c_str(), lparams);

//   if (lctx == NULL) {
//     fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
//     return NULL;
//   }

//   if (!params.lora_adapter.empty()) {
//     int err = model_apply_lora_from_file(lctx, params.lora_adapter.c_str(),
//                                          params.lora_base.empty() ? NULL : params.lora_base.c_str(), params.n_threads);
//     if (err != 0) {
//       fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
//       return NULL;
//     }
//   }

//   return lctx;
// }
