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
#include "chatglm2.h"

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
#include "core/layers/mha_dense.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"

// evaluate the transformer
//
//   - lctx:      model context
//   - tokens:    new batch of tokens to process
//   - n_past:    the offset to which the kv is cached to
//   - n_total:   the number of tokens evaluated so far (including evicted tokens if there is any)
//   - n_threads: number of threads to use
//

static bool chatglm_model_eval_internal(model_context& lctx, const model_token* tokens, const int n_tokens,
                                        const int n_past, const int n_total, const int n_threads) {
  const int64_t t_start_us = ne_time_us();

  const int N = n_tokens;

  const auto& model = lctx.model;
  const auto& hparams = model.hparams;

  const auto& kv_self = model.kv_self;

  MODEL_ASSERT(!!kv_self.ctx);

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = lctx.n_ctx;
  const int n_keep = lctx.n_keep;
  const bool shift_roped_k = lctx.shift_roped_k;
  const bool is_ring_full = shift_roped_k && n_total > n_past;
  const int n_cached = shift_roped_k ? std::min(n_total + N, n_ctx) : (n_past + N);  // #tokens cached after kv-append
  const int n_head = hparams.n_head;
  const int n_vocab = hparams.n_vocab;
  const int head_size = n_embd / n_head;
  const int n_rot = head_size / 2;
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

  const bool run_mha_reordered = model.layers[0].k_cache->type == NE_TYPE_JBLAS;
  kv_cache_info_t kv_cache_info = {};
  if (run_mha_reordered) {
    NE_ASSERT(("kv cache should be the same dtype", model.layers[0].v_cache->type == NE_TYPE_JBLAS));
    attn_shape_t attn_shape = {
        /* .batch_size = */ 1,
        /* .head_num = */ n_head,
        /* .heads_kv = */ num_kv_heads,
        /* .head_size = */ head_size,
        /* .sl_q = */ N,  // Note: make sure that jblas reordered attn supports next token inference
        /* .sl_kv = */ n_cached,
    };

    NE_ASSERT(("jblas managed kv-cache not supported; use `--memory-f16 / --memory-f32` instead",
               jblas_reordered_attn_fp32_support(&attn_shape)));
    kv_shape_t kv_shape{
        /* .heads_kv = */ static_cast<uint32_t>(num_kv_heads),
        /* .head_size = */ static_cast<uint32_t>(head_size),
        /* .sl_kv_max = */ static_cast<uint32_t>(n_ctx),
    };
    jblas_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
  }

  struct ne_tensor* embd = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, N);
  ne_set_name(embd, "embd");
  memcpy(embd->data, tokens, N * ne_element_size(embd));
  struct ne_tensor* inpL = ne_get_rows(ctx0, model.others[0], embd);

  NE_ASSERT(N == inpL->ne[1]);
  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;

    lctx.use_buf(ctx0, 0);

    // self-attention
    cur = ne_rms_norm(ctx0, inpL);
    cur = ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[0], cur), cur);
    {
      // compute QKV
      cur = ne_mul_mat(ctx0, model.layers[il].attn[0], cur);
      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].attn[1], cur), cur);

      struct ne_tensor* query_layer =
          ne_view_3d(ctx0, cur, head_size, n_head, N, head_size * ne_element_size(cur), cur->nb[1],
                     0);  // [N, heads, head_size]
      ne_set_name(query_layer, "query_layer");
      query_layer = ne_rope_inplace(ctx0, query_layer, std::max(n_cached - N, n_past), n_rot, 0, 0);

      struct ne_tensor* key_layer =
          ne_view_3d(ctx0, cur, head_size, num_kv_heads, N, head_size * ne_element_size(cur), cur->nb[1],
                     hidden_size * ne_element_size(cur));  // [N, kv_heads, head_size]
      ne_set_name(key_layer, "key_layer");
      key_layer = ne_rope_inplace(  // n_ctx exceeds but it will be shift-roped back with cached K
          ctx0, key_layer, (is_ring_full ? n_ctx : n_past), n_rot, 0, 0);

      struct ne_tensor* value_layer =
          ne_view_3d(ctx0, cur, head_size, num_kv_heads, N, head_size * ne_element_size(cur), cur->nb[1],
                     (hidden_size + head_size * num_kv_heads) * ne_element_size(cur));  // [N, kv_heads, head_size]
      ne_set_name(value_layer, "value_layer");

      const float attn_scale = 1.f / std::sqrt(head_size);
      if (!run_mha_reordered) {
        query_layer = ne_cont(ctx0, ne_permute(ctx0, query_layer, 0, 2, 1, 3));  // [heads, N, head_size]
        query_layer = ne_reshape_3d(ctx0, query_layer, head_size, mqa_scale * N,
                                    num_kv_heads);                // [kv_heads, mqa_scale * N, head_size]
        key_layer = ne_permute(ctx0, key_layer, 0, 2, 1, 3);      // [kv_heads, N, head_size]
        value_layer = ne_permute(ctx0, value_layer, 1, 2, 0, 3);  // [kv_heads, head_size, N]
        // store key and value to memory
        {
          struct ne_tensor* k_cache_view =
              ne_view_3d(ctx0, model.layers[il].k_cache, head_size, N, num_kv_heads, model.layers[il].k_cache->nb[1],
                         model.layers[il].k_cache->nb[2],
                         n_past * head_size * ne_element_size(model.layers[il].k_cache));  // [kv_heads, N, head_size]
          ne_set_name(k_cache_view, "k_cache_view");
          struct ne_tensor* v_cache_view =
              ne_view_3d(ctx0, model.layers[il].v_cache, N, head_size, num_kv_heads, model.layers[il].v_cache->nb[1],
                         model.layers[il].v_cache->nb[2],
                         n_past * ne_element_size(model.layers[il].v_cache));  // [kv_heads, head_size, N]
          ne_set_name(v_cache_view, "v_cache_view");

          ne_build_forward_expand(&gf, ne_cpy(ctx0, key_layer, k_cache_view));
          ne_build_forward_expand(&gf, ne_cpy(ctx0, value_layer, v_cache_view));
        }

        // concat key & value with past kv
        key_layer = ne_view_3d(ctx0, model.layers[il].k_cache, head_size, n_cached, num_kv_heads,
                               model.layers[il].k_cache->nb[1], model.layers[il].k_cache->nb[2],
                               0);  // [kv_heads, klen, head_size]
        value_layer = ne_view_3d(ctx0, model.layers[il].v_cache, n_cached, head_size, num_kv_heads,
                                 model.layers[il].v_cache->nb[1], model.layers[il].v_cache->nb[2],
                                 0);  // [kv_heads, head_size, klen]

        if (is_ring_full) {
          key_layer = ne_permute(ctx0, key_layer, 0, 2, 1, 3);  // perm for rope
          struct ne_tensor* cossin_cache = nullptr;
          // Currently we only cache cossin for N == 1 in model-wide; It may be worthwhile to cache cossin for other N
          // in a single eval execution
          if (N == 1) cossin_cache = kv_self.cossin;
          key_layer = ne_rope_shift_inplace(ctx0, key_layer, -N, n_rot, 0, 0, n_keep, cossin_cache);
          key_layer = ne_permute(ctx0, key_layer, 0, 2, 1, 3);  // perm back
        }

        // attention
        struct ne_tensor* attn_scores = ne_mul_mat(ctx0, key_layer, query_layer);  // [kv_heads, mqa_scale * N, klen]
        ne_set_name(attn_scores, "attn_scores");
        attn_scores = ne_scale_inplace(ctx0, attn_scores, ne_new_f32(ctx0, attn_scale));

        if (N > 1 || !shift_roped_k) {
          // build attention mask for context input
          attn_scores = ne_reshape_3d(ctx0, attn_scores, n_cached, N,
                                      num_attention_heads);  // [heads, N, klen]
          attn_scores = ne_diag_mask_inf_inplace(ctx0, attn_scores, n_past);
          attn_scores = ne_reshape_3d(ctx0, attn_scores, n_cached, mqa_scale * N,
                                      num_kv_heads);  // [kv_heads, mqa_scale * N, klen]
        }

        struct ne_tensor* attn_probs = ne_soft_max_inplace(ctx0, attn_scores);  // [kv_heads, mqa_scale * N, klen]

        cur = ne_mul_mat(ctx0, value_layer, attn_probs);  // [kv_heads, mqa_scale * N, head_size]
        cur = ne_reshape_3d(ctx0, cur, head_size, N,
                            num_attention_heads);                // [heads, N, head_size]
        cur = ne_cont(ctx0, ne_permute(ctx0, cur, 0, 2, 1, 3));  // [N, heads, head_size]
        cur = ne_reshape_2d(ctx0, cur, hidden_size, N);          // [N, hidden]
      } else {                                                   // Using MHA (GQA/MQA) managed kv-cache
        const auto k_size = kv_cache_info.k_bytes;
        const auto v_size = kv_cache_info.v_bytes;

        // store key and value to memory
        {
          const auto k_cache = ne_view_3d(ctx0, model.layers[il].k_cache,  // tensor
                                          head_size, n_ctx, num_kv_heads,  // ne
                                          0, 0,                            // nb (jblas managed)
                                          0);                              // offset
          ne_build_forward_expand(&gf, ne_flash_attn_update_k(ctx0, k_cache, key_layer, n_past, is_ring_full));
          const auto v_cache = ne_view_3d(ctx0, model.layers[il].v_cache,  // tensor
                                          head_size, n_ctx, num_kv_heads,  // ne
                                          0, 0,                            // nb (jblas managed)
                                          0);                              // offset
          ne_build_forward_expand(&gf, ne_flash_attn_update_v(ctx0, v_cache, value_layer, n_past, is_ring_full));
        }

        query_layer = ne_permute(ctx0, query_layer, 0, 2, 1, 3);                          // [heads, N, head_size]
        key_layer =                                                                       //
            ne_view_3d(ctx0, model.layers[il].k_cache,                                    // tensor
                       head_size, n_cached, num_kv_heads,                                 // ne
                       kv_cache_info.stride_k_sl, kv_cache_info.stride_k_head_num,        // nb (jblas managed)
                       0);                                                                // offset
        *reinterpret_cast<ATTN_FWD_LAYOUT*>(&key_layer->nb[0]) = kv_cache_info.k_layout;  // us nb0 for layout
        if (is_ring_full) {
          struct ne_tensor* cossin_cache = nullptr;
          // Currently we only cache cossin for N == 1 in model-wide; It may be worthwhile to cache cossin for other N
          // in a single eval execution
          if (N == 1) cossin_cache = kv_self.cossin;
          key_layer = ne_rope_shift_inplace(ctx0, key_layer, -N, n_rot, 0, 0, n_keep, cossin_cache);
        }
        value_layer =
            ne_view_3d(ctx0, model.layers[il].v_cache,                                      // tensor
                       n_cached, head_size, num_kv_heads,                                   // ne
                       kv_cache_info.stride_v_head_size, kv_cache_info.stride_v_head_num,   // nb (jblas managed)
                       0);                                                                  // offset
        *reinterpret_cast<ATTN_FWD_LAYOUT*>(&value_layer->nb[0]) = kv_cache_info.v_layout;  // us nb0 for layout

        ne_attn_flags_t attn_flags = NE_ATTN_FLAG_NONE;
        if (n_total == 0 || !shift_roped_k) attn_flags |= NE_ATTN_FLAG_IS_CAUSAL;  // no causal mask on next-token cases
        struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, query_layer, key_layer, value_layer, attn_scale, attn_flags);
        cur = ne_view_2d(ctx0, KQV_Out, n_embd, N, n_embd * ne_element_size(KQV_Out), 0);
      }
      cur = ne_mul_mat(ctx0, model.layers[il].attn[2], cur);
    }

    lctx.use_buf(ctx0, 1);

    struct ne_tensor* hidden_states = ne_add(ctx0, inpL, cur);

    // mlp.forward
    struct ne_tensor* mlp_output = ne_rms_norm(ctx0, hidden_states);
    ne_set_name(mlp_output, "mlp_output");
    // mlp_output = ne_mul(ctx0, mlp_output, model.layers[il].norm[1]);
    mlp_output = ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[1], mlp_output), mlp_output);

    mlp_output = ne_mul_mat(ctx0, model.layers[il].ffn[0], mlp_output);
    struct ne_tensor* x0 = ne_view_2d(ctx0, mlp_output, mlp_output->ne[0] / 2, mlp_output->ne[1], mlp_output->nb[1], 0);
    x0 = ne_silu(ctx0, x0);
    struct ne_tensor* x1 = ne_view_2d(ctx0, mlp_output, mlp_output->ne[0] / 2, mlp_output->ne[1], mlp_output->nb[1],
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
    // inpL = ne_mul(ctx0, inpL, model.others[1]);
    inpL = ne_mul(ctx0, ne_repeat(ctx0, model.others[1], inpL), inpL);
  }

  lctx.use_buf(ctx0, -1);
  if (embd->ne[0] > 1) {
    inpL = ne_view_1d(ctx0, inpL, hidden_size, (embd->ne[0] - 1) * hidden_size * ne_element_size(inpL));
  }
  // lm_head
  inpL = ne_mul_mat(ctx0, model.others[2], inpL);

  ne_build_forward_expand(&gf, inpL);
  ne_graph_compute(ctx0, &gf);

#ifdef NE_PERF
  bool engine_profiling_ = (getenv("ENGINE_PROFILING") != NULL);
  if (engine_profiling_) {
    ne_graph_profiling(&gf);
  }
#endif

  // update kv token count
  lctx.model.kv_self.n = n_cached;

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

int model_eval(struct model_context* ctx, const model_token* tokens, int n_tokens, int n_past, int n_total,
               int n_threads) {
  if (!chatglm_model_eval_internal(*ctx, tokens, n_tokens, n_past, n_total, n_threads)) {
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
