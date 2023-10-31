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
//   - tokens:    new batch of tokens to process
//   - n_past:    the offset to which the kv is cached to
//   - n_total:   the number of tokens evaluated so far (including evicted tokens if there is any)
//   - n_threads: number of threads to use
//
static bool mpt_model_eval_internal(model_context& lctx, const model_token* tokens, const int n_tokens,
                                    const int n_past, const int n_total, const int n_threads) {
  const int64_t t_start_us = ne_time_us();

  const int N = n_tokens;

  const int batch_size = lctx.batch_size;

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
  const int head_dim = hparams.n_embd / hparams.n_head;

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

  const bool run_mha_reordered = kv_self.k->type == NE_TYPE_JBLAS;
  kv_cache_info_t kv_cache_info = {};
  if (run_mha_reordered) {
    NE_ASSERT(("kv cache should be the same dtype", kv_self.v->type == NE_TYPE_JBLAS));
    attn_shape_t attn_shape = {
        /* .batch_size = */ 1,
        /* .head_num = */ n_head,
        /* .heads_kv = */ n_head,
        /* .head_size = */ head_dim,
        /* .sl_q = */ N,  // Note: make sure that jblas reordered attn supports next token inference
        /* .sl_kv = */ n_past + N,
    };

    NE_ASSERT(("jblas managed kv-cache not supported; use `--memory-f16 / --memory-f32` instead",
               jblas_reordered_attn_fp32_support(&attn_shape)));
    kv_shape_t kv_shape{
        /* .heads_kv = */ static_cast<uint32_t>(n_head),
        /* .head_size = */ static_cast<uint32_t>(head_dim),
        /* .sl_kv_max = */ static_cast<uint32_t>(n_ctx),
    };
    jblas_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
  }

  struct ne_tensor* embd = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, N);
  ne_set_name(embd, "embd");
  memcpy(embd->data, tokens, N * ne_element_size(embd));

  struct ne_tensor* inpL = ne_get_rows(ctx0, model.others[0], embd);

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;

    lctx.use_buf(ctx0, 0);

    // norm
    {
      cur = ne_norm(ctx0, inpL);

      cur = ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[0], cur), cur);
    }

    cur = ne_mul_mat(ctx0, model.layers[il].attn[0], cur);

    if (model.hparams.clip_qkv > 0.0f) {
      cur = ne_clamp(ctx0, cur, -model.hparams.clip_qkv, model.hparams.clip_qkv);
    }

    struct ne_tensor* Qcur = ne_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0 * sizeof(float) * n_embd);
    struct ne_tensor* Kcur = ne_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1 * sizeof(float) * n_embd);
    struct ne_tensor* Vcur = ne_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2 * sizeof(float) * n_embd);

    // self-attention
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    if (!run_mha_reordered) {
      // store key and value to memory
      {
        Vcur = ne_transpose(ctx0, Vcur);
        struct ne_tensor* k =
            ne_view_1d(ctx0, kv_self.k, N * n_embd, (ne_element_size(kv_self.k) * n_embd) * (il * n_ctx + n_past));
        struct ne_tensor* v =
            ne_view_2d(ctx0, kv_self.v, N, n_embd, (n_ctx)*ne_element_size(kv_self.v),
                       (il * n_ctx) * ne_element_size(kv_self.v) * n_embd + n_past * ne_element_size(kv_self.v));
        // important: storing RoPE-ed version of K in the KV cache!
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur, k));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur, v));
      }

      // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0,
      // 2, 1, 3) [64, N, 12]
      struct ne_tensor* Q = ne_permute(
          ctx0, ne_cpy(ctx0, Qcur, ne_new_tensor_3d(ctx0, NE_TYPE_F32, n_embd / n_head, n_head, N, NE_SIZE_CALC)), 0, 2,
          1, 3);

      // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1,
      // 3) [64, n_past + N, 12]
      struct ne_tensor* K = ne_permute(ctx0,
                                       ne_reshape_3d(ctx0,
                                                     ne_view_1d(ctx0, kv_self.k, (n_past + N) * n_embd,
                                                                il * n_ctx * ne_element_size(kv_self.k) * n_embd),
                                                     n_embd / n_head, n_head, n_past + N),
                                       0, 2, 1, 3);
      // K * Q
      struct ne_tensor* KQ = ne_mul_mat(ctx0, K, Q);

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      struct ne_tensor* KQ_scaled = ne_scale(ctx0, KQ, ne_new_f32(ctx0, attn_scale));

      struct ne_tensor* KQ_scaled_alibi = ne_alibi(ctx0, KQ_scaled, n_past, n_head, model.hparams.alibi_bias_max);

      // KQ_masked = mask_past(KQ_scaled)
      struct ne_tensor* KQ_masked = ne_diag_mask_inf(ctx0, KQ_scaled_alibi, n_past);

      // KQ = soft_max(KQ_masked)
      struct ne_tensor* KQ_soft_max = ne_soft_max(ctx0, KQ_masked);

      // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1,
      // 2, 0, 3).contiguous() [n_past + N, 64, 12]
      struct ne_tensor* V_trans = ne_view_3d(
          ctx0, kv_self.v, n_past + N, n_embd / n_head, n_head, n_ctx * ne_element_size(kv_self.v),
          n_ctx * ne_element_size(kv_self.v) * n_embd / n_head, il * n_ctx * ne_element_size(kv_self.v) * n_embd);

      // KQV = transpose(V) * KQ_soft_max
      struct ne_tensor* KQV = ne_mul_mat(ctx0, V_trans, KQ_soft_max);

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct ne_tensor* KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);

      // cur = KQV_merged.contiguous().view(n_embd, N)
      cur = ne_cpy(ctx0, KQV_merged, ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N, NE_SIZE_CALC));
    } else {
      const auto seq_kv = n_past + N;
      const auto k_size = kv_cache_info.k_bytes;
      const auto v_size = kv_cache_info.v_bytes;
      // store key and value to memory
      {
        const auto k_cache = ne_view_3d(ctx0, kv_self.k,          // tensor
                                        head_dim, n_ctx, n_head,  // ne
                                        0, 0,                     // nb (jblas managed)
                                        il * k_size);             // offset
        Kcur = ne_view_3d(ctx0, Kcur, head_dim, n_head, N, Kcur->nb[0] * head_dim, Kcur->nb[1], 0);
        ne_build_forward_expand(&gf, ne_flash_attn_update_k(ctx0, k_cache, Kcur, n_past));
        const auto v_cache = ne_view_3d(ctx0, kv_self.v,          // tensor
                                        head_dim, n_ctx, n_head,  // ne
                                        0, 0,                     // nb (jblas managed)
                                        il * v_size);             // offset
        Vcur = ne_view_3d(ctx0, Vcur, head_dim, n_head, N, Vcur->nb[0] * head_dim, Vcur->nb[1], 0);
        ne_build_forward_expand(&gf, ne_flash_attn_update_v(ctx0, v_cache, Vcur, n_past));
      }

      struct ne_tensor* Q = ne_view_3d(ctx0, Qcur, head_dim, n_head, N, Qcur->nb[0] * head_dim, Qcur->nb[1], 0);
      Q = ne_permute(ctx0, Q, 0, 2, 1, 3);
      ne_set_name(Q, "Q");
      struct ne_tensor* K =
          ne_view_3d(ctx0, kv_self.k,                                             // tensor
                     head_dim, seq_kv, n_head,                                    // ne
                     kv_cache_info.stride_k_sl, kv_cache_info.stride_k_head_num,  // nb (jblas managed)
                     il * k_size);                                                // offset
      *reinterpret_cast<ATTN_FWD_LAYOUT*>(&K->nb[0]) = kv_cache_info.k_layout;    // us nb0 for layout
      ne_set_name(K, "K");
      struct ne_tensor* V =
          ne_view_3d(ctx0, kv_self.v,                                                    // tensor
                     seq_kv, head_dim, n_head,                                           // ne
                     kv_cache_info.stride_v_head_size, kv_cache_info.stride_v_head_num,  // nb (jblas managed)
                     il * v_size);                                                       // offset
      *reinterpret_cast<ATTN_FWD_LAYOUT*>(&V->nb[0]) = kv_cache_info.v_layout;           // us nb0 for layout
      ne_set_name(V, "V");

      ne_attn_flags_t attn_flags = NE_ATTN_FLAG_IS_ALIBI8;    // mpt uses alibi operation
      if (n_past == 0) attn_flags |= NE_ATTN_FLAG_IS_CAUSAL;  // no causal mask on next-token cases
      struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, Q, K, V, attn_scale, attn_flags);
      cur = ne_view_2d(ctx0, KQV_Out, n_embd, N, n_embd * ne_element_size(KQV_Out), 0);
    }

    // projection
    { cur = ne_mul_mat(ctx0, model.layers[il].attn[1], cur); }
    inpL = ne_add(ctx0, inpL, cur);

    lctx.use_buf(ctx0, 1);

    // m = self.ln_2(x)
    {
      cur = ne_norm(ctx0, inpL);

      cur = ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].norm[1], cur), cur);
    }

    // n = self.mlp(m)
    {
      // FFN FUSION
      if (jblas_fusion_FFN_GeLu_f32f32_support(model.layers[il].ffn[0]->data, model.layers[il].ffn[1]->data,
                                               N * batch_size, cur->ne[0], model.layers[il].ffn[0]->ne[1],
                                               model.layers[il].ffn[1]->ne[1])) {
        cur = ne_ffn_gelu(ctx0, model.layers[il].ffn[0], model.layers[il].ffn[1], cur);
      } else {
        cur = ne_mul_mat(ctx0, model.layers[il].ffn[0], cur);

        // GELU activation
        cur = ne_gelu(ctx0, cur);

        // projection
        // cur = proj_w*cur + proj_b
        cur = ne_mul_mat(ctx0, model.layers[il].ffn[1], cur);
      }
    }

    // x = x + n
    inpL = ne_add(ctx0, inpL, cur);
  }

  lctx.use_buf(ctx0, 0);
  // used at the end to optionally extract the embeddings
  struct ne_tensor* embeddings = NULL;

  // norm
  {
    inpL = ne_norm(ctx0, inpL);
    // inpL = ln_f_g*inpL
    inpL = ne_mul(ctx0, ne_repeat(ctx0, model.others[1], inpL), inpL);
  }

  lctx.use_buf(ctx0, -1);

  // output embedding weight tied to input embedding
  inpL = ne_mul_mat(ctx0, model.others[0], inpL);

  // logits -> probs
  // inpL = ne_soft_max(ctx0, inpL);

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
      memcpy(logits_out.data(), (float*)ne_get_data(inpL) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);
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
  if (!mpt_model_eval_internal(*ctx, tokens, n_tokens, n_past, n_total, n_threads)) {
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
