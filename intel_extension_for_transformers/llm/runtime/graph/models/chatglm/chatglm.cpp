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
#include "core/ne_jblas.h"
#include "core/layers/mha_dense.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"

// evaluate the transformer
//
//   - lctx:      model context
//   - inputs:    model_input array
//   - n_input    num of model_input
//   - n_threads: number of threads to use
//

static int flag = 0;
static int first_tokens_size = 0;
static bool chatglm_model_eval_internal(model_context& lctx, const model_input* inputs, const int n_input,
                                        const int n_threads) {
  const int64_t t_start_us = ne_time_us();

  // TODO static batching for now
  const int N = inputs->n_tokens;
  const int n_past = inputs->n_past;
  const int n_total = inputs->n_total;
  const int beam_size = lctx.beam_search ? lctx.beam_size : 1;
  const int batch_size = lctx.batch_size;
  MODEL_ASSERT(batch_size == n_input);
  std::vector<int> block_ids;
  std::vector<int> n_padding;
  bool no_padding = true;
  for (int i = 0; i < batch_size; ++i) {
    block_ids.push_back((inputs + i)->request_idx * beam_size + (inputs + i)->beam_idx);
    n_padding.push_back((inputs + i)->n_padding);
    if (no_padding && (inputs + i)->n_padding != 0) no_padding = false;
  }

  const auto& model = lctx.model;
  const auto& hparams = model.hparams;

  const auto& kv_self = model.kv_self;

  MODEL_ASSERT(!!kv_self.ctx);

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = lctx.n_ctx;
  const int n_keep = lctx.n_keep;

  if (flag == 0) {
    first_tokens_size = N;
    flag++;
  }

  const int n_head = hparams.n_head;
  const int n_vocab = hparams.n_vocab;
  const int n_rot = n_embd / n_head / 2;
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

  struct ne_tensor* embd = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, N * batch_size);
  ne_set_name(embd, "embd");
  for (int i = 0; i < batch_size; ++i) {
    memcpy(static_cast<model_token*>(embd->data) + i * N, (inputs + i)->tokens, N * ne_element_size(embd));
  }

  struct ne_tensor* inpL = ne_get_rows(ctx0, model.others[0], embd);

  int hidden_size = inpL->ne[0];
  int qlen = inpL->ne[1] / batch_size;
  int head_size = hidden_size / num_attention_heads;
  int rope_dim = head_size / 2;
  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;
    struct ne_tensor* alpha = ne_new_f32(ctx0, std::sqrt(2.f * n_layer));

    lctx.use_buf(ctx0, 0);

    cur = ne_norm(ctx0, inpL);

    ne_set_name(cur, "cur");
    cur = ne_mul(ctx0, cur, model.layers[il].norm[0]);
    cur = ne_add(ctx0, cur, model.layers[il].norm[1]);

    struct ne_tensor* attn_input = cur;
    // SelfAttention
    {
      // Linear::forward compute QKV
      cur = ne_mul_mat(ctx0, model.layers[il].attn[0], cur);
      cur = ne_add(ctx0, cur, model.layers[il].attn[1]);

      ne_tensor* query_layer =
          ne_view_4d(ctx0, cur, head_size, n_head, N, batch_size, 3 * head_size * ne_element_size(cur), cur->nb[1],
                     cur->nb[1] * N, 0);  // [qlen * bs, 3 * hidden]

      ne_set_name(query_layer, "query_layer");
      query_layer = ne_rope_inplace(ctx0, query_layer, n_past, rope_dim, 4, first_tokens_size);
      query_layer = ne_permute(ctx0, query_layer, 0, 2, 1, 3);  // [bs, heads, qlen, head_size]

      ne_tensor* key_layer =
          ne_view_4d(ctx0, cur, head_size, num_attention_heads, qlen, batch_size, 3 * head_size * ne_element_size(cur),
                     cur->nb[1], cur->nb[1] * qlen, head_size * ne_element_size(cur));
      key_layer = ne_rope_inplace(ctx0, key_layer, n_past, rope_dim, 4, first_tokens_size);  // [qlen, heads, head_size]

      ne_tensor* value_layer = ne_view_4d(ctx0, cur, head_size, num_attention_heads, qlen, batch_size,
                                          3 * head_size * ne_element_size(cur), cur->nb[1], cur->nb[1] * qlen,
                                          2 * head_size * ne_element_size(cur));  // [bs, qlen, heads, head_size]

      // store key and value to memory
      {
        std::vector<ne_tensor*> Kcur_bs(batch_size);
        std::vector<ne_tensor*> Vcur_bs(batch_size);
        std::vector<ne_tensor*> k_bs(batch_size);
        std::vector<ne_tensor*> v_bs(batch_size);
        for (int i = 0; i < batch_size; ++i) {
          const int block_idx = block_ids[i];
          // [bs, heads, qlen, head_size]
          Kcur_bs[i] = ne_permute(ctx0,
                                  ne_view_4d(ctx0, key_layer, head_size, num_attention_heads, qlen, 1, key_layer->nb[1],
                                             key_layer->nb[2], key_layer->nb[3], i * key_layer->nb[3]),
                                  0, 2, 1, 3);
          k_bs[i] = ne_view_4d(
              ctx0, model.layers[il].k_cache, head_size, qlen, num_attention_heads, 1, model.layers[il].k_cache->nb[1],
              model.layers[il].k_cache->nb[2], model.layers[il].k_cache->nb[3],
              block_idx * n_ctx * n_embd * ne_element_size(model.layers[il].k_cache) +
                  n_past * head_size * ne_element_size(model.layers[il].k_cache));  // [kv_heads, qlen, head_size]

          // [bs, heads, head_size, qlen]
          Vcur_bs[i] =
              ne_permute(ctx0,
                         ne_view_4d(ctx0, value_layer, head_size, num_attention_heads, qlen, 1, value_layer->nb[1],
                                    value_layer->nb[2], value_layer->nb[3], i * value_layer->nb[3]),
                         1, 2, 0, 3);
          v_bs[i] = ne_view_4d(ctx0, model.layers[il].v_cache, qlen, head_size, num_attention_heads, 1,
                               model.layers[il].v_cache->nb[1], model.layers[il].v_cache->nb[2],
                               model.layers[il].v_cache->nb[3],
                               block_idx * n_ctx * n_embd * ne_element_size(model.layers[il].v_cache) +
                                   n_past * ne_element_size(model.layers[il].v_cache));  // [kv_heads, head_size, qlen]

          ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur_bs[i], k_bs[i]));
          ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur_bs[i], v_bs[i]));
        }
      }
      // concat key & value with past kv
      key_layer =
          ne_view_4d(ctx0, model.layers[il].k_cache, head_size, n_past + qlen, num_attention_heads, batch_size,
                     model.layers[il].k_cache->nb[1], model.layers[il].k_cache->nb[2], model.layers[il].k_cache->nb[3],
                     0);  // [bs, kv_heads, klen, head_size]
      value_layer =
          ne_view_4d(ctx0, model.layers[il].v_cache, n_past + qlen, head_size, num_attention_heads, batch_size,
                     model.layers[il].v_cache->nb[1], model.layers[il].v_cache->nb[2], model.layers[il].v_cache->nb[3],
                     0);  // [bs, kv_heads, head_size, klen]

      // attention
      struct ne_tensor* attn_scores =
          ne_mul_mat(ctx0, key_layer, query_layer);  // [bs, kv_heads, mqa_scale * qlen, klen]
      ne_set_name(attn_scores, "attn_scores");

      if (n_past == 0) {
        // build attention mask for context input
        ne_tensor* inf =
            ne_new_tensor_4d(ctx0, attn_scores->type, 1, qlen - 1, num_attention_heads, batch_size, NE_SIZE_CALC);
        ne_set_f32(inf, -INFINITY);

        ne_tensor* masked_attn_scores =
            ne_view_4d(ctx0, attn_scores, 1, qlen - 1, num_attention_heads, batch_size,
                       qlen * ne_element_size(attn_scores), qlen * qlen * ne_element_size(attn_scores),
                       qlen * qlen * ne_element_size(attn_scores) * num_attention_heads,
                       (qlen - 1) * ne_element_size(attn_scores));

        ne_set_name(masked_attn_scores, "masked_attn_scores");
        ne_build_forward_expand(&gf, ne_cpy(ctx0, inf, masked_attn_scores));
      }
      // mask left pad token
      attn_scores = ne_padding_left_mask_inf_inplace(ctx0, attn_scores, n_padding.data());
      attn_scores = ne_scale_inplace(ctx0, attn_scores, ne_new_f32(ctx0, 1.f / std::sqrt(head_size)));
      ne_set_name(attn_scores, "attn_scores");

      ne_tensor* attn_probs = ne_soft_max_inplace(ctx0, attn_scores);  // [bs, heads, qlen, klen]

      ne_tensor* context_layer = ne_mul_mat(ctx0, value_layer, attn_probs);  // [bs, heads, qlen, head_size]

      context_layer = ne_cont(ctx0, ne_permute(ctx0, context_layer, 0, 2, 1, 3));

      context_layer = ne_reshape_2d(ctx0, context_layer, hidden_size, qlen * batch_size);

      cur = ne_mul_mat(ctx0, model.layers[il].attn[2], context_layer);
      cur = ne_add(ctx0, cur, model.layers[il].attn[3]);
    }

    lctx.use_buf(ctx0, 1);

    ne_build_forward_expand(&gf, cur);
    attn_input = ne_scale_inplace(ctx0, attn_input, alpha);
    inpL = ne_add_inplace(ctx0, attn_input, cur);

    struct ne_tensor* mlp_input = ne_norm(ctx0, inpL);

    ne_set_name(mlp_input, "mlp_input");
    mlp_input = ne_mul(ctx0, mlp_input, model.layers[il].norm[2]);
    mlp_input = ne_add(ctx0, mlp_input, model.layers[il].norm[3]);

    // mlp.forward
    struct ne_tensor* mlp_output;
    bool status = jblas_fusion_FFN_Add_GeLu_f32f32_support(
        model.layers[il].ffn[0]->data, model.layers[il].ffn[2]->data, N * batch_size, mlp_input->ne[0],
        model.layers[il].ffn[0]->ne[1], model.layers[il].ffn[2]->ne[1]);
    if (status) {
      mlp_output = ne_ffn_add_gelu(ctx0, model.layers[il].ffn[0], model.layers[il].ffn[2], model.layers[il].ffn[1],
                                   model.layers[il].ffn[3], mlp_input);
    } else {
      mlp_output = ne_mul_mat(ctx0, model.layers[il].ffn[0], mlp_input);
      mlp_output = ne_add(ctx0, mlp_output, model.layers[il].ffn[1]);
      mlp_output = ne_gelu(ctx0, mlp_output);
      mlp_output = ne_mul_mat(ctx0, model.layers[il].ffn[2], mlp_output);
      mlp_output = ne_add(ctx0, mlp_output, model.layers[il].ffn[3]);
    }

    ne_build_forward_expand(&gf, mlp_output);
    mlp_input = ne_scale_inplace(ctx0, mlp_input, alpha);
    inpL = ne_add_inplace(ctx0, mlp_input, mlp_output);
  }

  lctx.use_buf(ctx0, 0);
  // used at the end to optionally extract the embeddings
  struct ne_tensor* embeddings = NULL;
  // norm
  {
    inpL = ne_norm(ctx0, inpL);

    ne_set_name(inpL, "inpL");
    inpL = ne_mul(ctx0, inpL, model.others[1]);
    inpL = ne_add(ctx0, inpL, model.others[2]);
  }

  lctx.use_buf(ctx0, -1);
  if (!lctx.logits_all && qlen > 1) {
    inpL = ne_view_2d(ctx0, inpL, hidden_size, batch_size, ne_element_size(inpL) * hidden_size * N,
                      (N - 1) * hidden_size * ne_element_size(inpL));
  }
  // lm_head
  inpL = ne_mul_mat(ctx0, model.others[3], inpL);

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
      logits_out.resize(n_vocab * N * batch_size);
      memcpy(logits_out.data(), (float*)ne_get_data(inpL), sizeof(float) * n_vocab * N * batch_size);
    } else {
      // return result for just the last token
      logits_out.resize(n_vocab * batch_size);
      memcpy(logits_out.data(), (float*)ne_get_data(inpL), sizeof(float) * n_vocab * batch_size);
    }
  }

  // extract embeddings
  if (!lctx.embedding.empty()) {
    auto& embedding_out = lctx.embedding;

    embedding_out.resize(n_embd * batch_size);
#pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
      memcpy(embedding_out.data() + (i * n_embd),
             (float*)ne_get_data(embeddings) + (i * n_embd * N) + (n_embd * (N - 1)), sizeof(float) * n_embd);
    }
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

int model_eval(struct model_context* ctx, const model_input* inputs, const int n_input, int n_threads) {
  if (!chatglm_model_eval_internal(*ctx, inputs, n_input, n_threads)) {
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
