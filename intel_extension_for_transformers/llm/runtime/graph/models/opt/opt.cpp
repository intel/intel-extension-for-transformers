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
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"

// evaluate the transformer
//
//   - lctx:      model context
//   - inputs:    model_input array
//   - n_input    num of model_input
//   - n_threads: number of threads to use
//
#define OPT_POS_EMBD_OFFS 2

static bool opt_model_eval_internal(model_context* ctx, const model_input* inputs, const int n_input,
                                    const int n_threads) {
  const int64_t t_start_us = ne_time_us();
  model_context& lctx = *ctx;

  // static batching for now
  const int N = inputs->n_tokens;
  const int n_past = inputs->n_past;
  const int n_total = inputs->n_total;
  const int batch_size = lctx.batch_size;
  MODEL_ASSERT(batch_size == n_input);

  const auto& model = lctx.model;
  const auto& hparams = model.hparams;

  const auto& kv_self = model.kv_self;

  MODEL_ASSERT(!!kv_self.ctx);

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = lctx.n_ctx;
  const int n_keep = lctx.n_keep;
  const int n_head = hparams.n_head;
  const int n_vocab = hparams.n_vocab;
  const int word_embed_proj_dim = hparams.word_embed_proj_dim;
  const bool do_layer_norm_before = hparams.do_layer_norm_before;

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

  struct ne_tensor* embd = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, N);
  ne_set_name(embd, "embd");
  for (int i = 0; i < batch_size; ++i) {
    memcpy(static_cast<model_token*>(embd->data) + i * N, (inputs + i)->tokens, N * ne_element_size(embd));
  }

  /* class OPTLearnedPositionalEmbedding(nn.Embedding)
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)
  */
  struct ne_tensor* position = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, N);
  for (int i = 0; i < N; ++i) {
    (reinterpret_cast<int32_t*>(position->data))[i] = n_past + i + OPT_POS_EMBD_OFFS;
  }

  // wte + wpe
  struct ne_tensor* word_embd = ne_get_rows(ctx0, model.others[0], embd);
  struct ne_tensor* pos_embd = ne_get_rows(ctx0, model.others[1], position);
  struct ne_tensor* inpL;
  if (word_embed_proj_dim == n_embd) {
    inpL = ne_add(ctx0, word_embd, pos_embd);
  } else {
    struct ne_tensor* word_embd_proj = ne_mul_mat(ctx0, model.others[4], word_embd);
    inpL = ne_add(ctx0, word_embd_proj, pos_embd);
  }

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur = inpL;

    lctx.use_buf(ctx0, 0);

    // attn norm
    if (do_layer_norm_before) {
      cur = ne_norm(ctx0, inpL);
      cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[0], cur), cur),
                   ne_repeat(ctx0, model.layers[il].norm[1], cur));
    }

    // self-attention
    {
      // Q K V GEMM
      struct ne_tensor* Qg = ne_mul_mat(ctx0, model.layers[il].attn[0], cur);
      Qg = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].attn[1], Qg), Qg);

      struct ne_tensor* Kg = ne_mul_mat(ctx0, model.layers[il].attn[2], cur);
      Kg = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].attn[3], Kg), Kg);

      struct ne_tensor* Vg = ne_mul_mat(ctx0, model.layers[il].attn[4], cur);
      Vg = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].attn[5], Vg), Vg);

      // reshape and reorder
      size_t head_dim = n_embd / n_head;
      struct ne_tensor* Qcur =
          ne_view_3d(ctx0, Qg, head_dim, n_head, N, head_dim * ne_element_size(Qg), n_embd * ne_element_size(Qg), 0);
      // head_dim, n_head, N --> head_dim, N, n_head
      struct ne_tensor* Kcur = ne_permute(
          ctx0,
          ne_view_3d(ctx0, Kg, head_dim, n_head, N, head_dim * ne_element_size(Kg), n_embd * ne_element_size(Kg), 0), 0,
          2, 1, 3);
      // head_dim, n_head, N --> N, head_dim, n_head
      struct ne_tensor* Vcur = ne_permute(
          ctx0,
          ne_view_3d(ctx0, Vg, head_dim, n_head, N, head_dim * ne_element_size(Vg), n_embd * ne_element_size(Vg), 0), 1,
          2, 0, 3);

      // store transposed key and value to memory (k_v cache)
      if (N >= 1) {
        // head_dim as col
        struct ne_tensor* k = ne_view_3d(
            ctx0, kv_self.k, head_dim, N, n_head, ne_element_size(kv_self.k) * head_dim,
            ne_element_size(kv_self.k) * head_dim * n_ctx,
            il * n_ctx * ne_element_size(kv_self.k) * n_embd + n_past * ne_element_size(kv_self.k) * head_dim);
        // N as col, n_embd as row
        struct ne_tensor* v =
            ne_view_3d(ctx0, kv_self.v, N, head_dim, n_head, n_ctx * ne_element_size(kv_self.v),
                       n_ctx * ne_element_size(kv_self.v) * head_dim,
                       il * n_ctx * ne_element_size(kv_self.v) * n_embd + n_past * ne_element_size(kv_self.v));
        // concat
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur, k));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur, v));
      }

      // head_dim, n_head, N --> head_dim, N, n_head
      struct ne_tensor* Q = ne_permute(ctx0, Qcur, 0, 2, 1, 3);

      struct ne_tensor* K =
          ne_view_3d(ctx0, kv_self.k, head_dim, N + n_past, n_head, ne_element_size(kv_self.k) * head_dim,
                     ne_element_size(kv_self.k) * head_dim * n_ctx, il * n_ctx * ne_element_size(kv_self.k) * n_embd);

      // GG: flash attention
      // struct ne_tensor * V =
      //    ne_cpy(ctx0,
      //            ne_permute(ctx0,
      //                ne_reshape_3d(ctx0,
      //                    ne_view_1d(ctx0, kv_self.v, (n_past + N)*n_embd,
      //                    il*n_ctx*ne_element_size(kv_self.v)*n_embd), n_embd/n_head, n_head, n_past + N),
      //                1, 2, 0, 3),
      //            ne_new_tensor_3d(ctx0, NE_TYPE_F32, n_past + N, n_embd/n_head, n_head, NE_SIZE_CALC));

      // struct ne_tensor * KQV = ne_flash_attn(ctx0, Q, K, V, NE_ATTN_FLAG_IS_CAUSAL);

      // K * Q
      // QK^T
      struct ne_tensor* KQ = ne_mul_mat(ctx0, K, Q);

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      // [n_past + N, N, n_head]
      struct ne_tensor* KQ_scaled = ne_scale_inplace(ctx0, KQ, ne_new_f32(ctx0, 1.0f / sqrt(static_cast<float>((n_embd) / n_head))));

      // KQ_masked = mask_past(KQ_scaled)
      // [n_past + N, N, n_head]
      struct ne_tensor* KQ_masked = ne_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

      // KQ = soft_max(KQ_masked)
      // [n_past + N, N, n_head]
      struct ne_tensor* KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_masked);

      // [n_past + N, head_dim, n_head]
      struct ne_tensor* V_trans =
          ne_view_3d(ctx0, kv_self.v, N + n_past, head_dim, n_head, n_ctx * ne_element_size(kv_self.v),
                     n_ctx * ne_element_size(kv_self.v) * head_dim, il * n_ctx * ne_element_size(kv_self.v) * n_embd);

      // [head_dim, N, n_head]
      struct ne_tensor* KQV = ne_mul_mat(ctx0, V_trans, KQ_soft_max);

      // [head_dim, n_head, N]
      struct ne_tensor* KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);

      // [n_embd, N]
      cur = ne_cpy(ctx0, KQV_merged, ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N, NE_SIZE_CALC));
    }

    // attn out projection
    // cur = proj_w*cur + proj_b
    // [n_embd, N]
    {
      cur = ne_mul_mat(ctx0, model.layers[il].attn[6], cur);

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].attn[7], cur), cur);
    }

    // add the input (residual)
    cur = ne_add(ctx0, cur, inpL);

    // attn norm
    if (!do_layer_norm_before) {
      cur = ne_norm(ctx0, cur);
      cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[0], cur), cur),
                   ne_repeat(ctx0, model.layers[il].norm[1], cur));
    }

    struct ne_tensor* inpFF = cur;

    lctx.use_buf(ctx0, 1);

    // feed-forward network (FFN)
    {
      // final norm
      if (do_layer_norm_before) {
        cur = ne_norm(ctx0, inpFF);
        // cur = ln_2_g*cur + ln_2_b
        // [ n_embd, N]
        cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[2], cur), cur),
                     ne_repeat(ctx0, model.layers[il].norm[3], cur));
      }

      // fc1
      // [4 * n_embd, N]
      cur = ne_mul_mat(ctx0, model.layers[il].ffn[0], cur);

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].ffn[1], cur), cur);

      // RELU activation
      // [4 * n_embd, N]
      cur = ne_relu(ctx0, cur);

      // fc2
      // [n_embd, N]
      cur = ne_mul_mat(ctx0, model.layers[il].ffn[2], cur);

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].ffn[3], cur), cur);
    }

    // input for next layer
    inpL = ne_add(ctx0, cur, inpFF);

    // final norm
    if (!do_layer_norm_before) {
      inpL = ne_norm(ctx0, inpL);
      // cur = ln_2_g*cur + ln_2_b
      // [n_embd, N]
      inpL = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[2], inpL), inpL),
                    ne_repeat(ctx0, model.layers[il].norm[3], inpL));
    }
  }

  lctx.use_buf(ctx0, 0);
  // used at the end to optionally extract the embeddings
  struct ne_tensor* embeddings = NULL;
  // final norm
  if (do_layer_norm_before) {
    // [n_embd, N]
    inpL = ne_norm(ctx0, inpL);
    // inpL = ln_f_g*inpL + ln_f_b
    // [n_embd, N]
    inpL = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.others[2], inpL), inpL),
                  ne_repeat(ctx0, model.others[3], inpL));
  }

  // word_embd_proj GEMM
  if (word_embed_proj_dim != n_embd) {
    // [word_embed_proj_dim, N]
    inpL = ne_mul_mat(ctx0, model.others[5], inpL);
  }

  lctx.use_buf(ctx0, -1);
  // lm_head GEMM
  // [n_vocab, N]
  inpL = ne_mul_mat(ctx0, model.others[6], inpL);

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
      memcpy(logits_out.data(), reinterpret_cast<float*>(ne_get_data(inpL)), sizeof(float) * n_vocab * N);
    } else {
      // return result for just the last token
      logits_out.resize(n_vocab);
      memcpy(logits_out.data(), reinterpret_cast<float*>(ne_get_data(inpL)) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);
    }
  }

  // extract embeddings
  if (!lctx.embedding.empty()) {
    auto& embedding_out = lctx.embedding;

    embedding_out.resize(n_embd);
    memcpy(embedding_out.data(), reinterpret_cast<float*>(ne_get_data(embeddings)) + (n_embd * (N - 1)), sizeof(float) * n_embd);
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
  if (!opt_model_eval_internal(ctx, inputs, n_input, n_threads)) {
    fprintf(stderr, "%s: failed to eval\n", __func__);
    return 1;
  }

  // get a more accurate load time, upon first eval

  if (!ctx->has_evaluated_once) {
    ctx->t_load_us = ne_time_us() - ctx->t_start_us;
    ctx->has_evaluated_once = true;
  }

  return 0;
}
