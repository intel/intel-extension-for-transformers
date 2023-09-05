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
#include "models/llama/llama.h"

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
#include "core/layers/mha_dense.h"
#include "core/ne.h"
#include "core/ne_jblas.h"
#include "core/ne_layers.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"
#include "models/models.h"

// evaluate the transformer
//
//   - lctx:      model context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool llama_model_eval_internal(model_context& lctx, const model_token* tokens, const int n_tokens,
                                      const int n_past, const int n_threads) {
  // enforce that the first token is BOS
  if (n_past == 0 && tokens[0] != model_token_bos()) {
    fprintf(stderr, "%s: first token must be BOS\n", __func__);
    return false;
  }

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
  const int n_rot = hparams.n_embd / hparams.n_head;

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

  const bool kv_mem_jblas = kv_self.k->type == NE_TYPE_JBLAS;
  kv_cache_info_t kv_cache_info = {0, 0};
  if (kv_mem_jblas) {
    NE_ASSERT(kv_self.v->type == NE_TYPE_JBLAS);  // kv type should be the same
    attn_shape_t attn_shape = {
        /* .batch_size = */ 1,
        /* .head_num = */ n_head,
        /* .head_size = */ n_embd / n_head,
        /* .sl_q = */ N,  // Note: make sure that jblas reordered attn supports next token inferencing
        /* .sl_kv = */ n_past + N,
    };

    NE_ASSERT(("jblas managed kv-cache not supported; use `--memory-f16 / --memory-f32` instead",
               jblas_reordered_attn_fp32_support(&attn_shape)));
    kv_shape_t kv_shape{
        /* .head_num = */ static_cast<uint32_t>(n_head),
        /* .head_size = */ static_cast<uint32_t>(n_embd / n_head),
        /* .sl_kv_max = */ static_cast<uint32_t>(n_ctx),
    };
    jblas_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
  }

  struct ne_tensor* embd = ne_new_tensor_1d(ctx0, NE_TYPE_I32, N, NE_SIZE_CALC);
  ne_set_name(embd, "embd");
  memcpy(embd->data, tokens, N * ne_element_size(embd));

  struct ne_tensor* inpL = ne_get_rows(ctx0, model.others[0], embd);

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* inpSA = inpL;

    struct ne_tensor* cur;

    lctx.use_buf(ctx0, 0);

    // norm
    {
      cur = ne_rms_norm(ctx0, inpL);

      // cur = cur*attention_norm(broadcasted)
      cur = ne_mul(ctx0, cur, model.layers[il].norm[0]);
    }
    ne_tensor *Qcur, *Kcur, *Vcur;
    if (jblas_fusion_QKV_f32f32_support(model.layers[il].attn[0]->data, model.layers[il].attn[1]->data,
                                        model.layers[il].attn[2]->data, N, model.layers[il].attn[0]->ne[1],
                                        model.layers[il].attn[0]->ne[0])) {  // fused execution of QKV
      struct ne_tensor* QKVcur =
          ne_mul_qkv(ctx0, model.layers[il].attn[0], model.layers[il].attn[1], model.layers[il].attn[2], cur);
      Qcur = ne_rope_inplace(
          ctx0,
          ne_reshape_3d(ctx0, ne_view_1d(ctx0, QKVcur, N * n_embd, 0 * N * n_embd * ne_element_size(QKVcur)),
                        n_embd / n_head, n_head, N),
          n_past, n_rot, 0);
      Kcur = ne_rope_inplace(
          ctx0,
          ne_reshape_3d(ctx0, ne_view_1d(ctx0, QKVcur, N * n_embd, 1 * N * n_embd * ne_element_size(QKVcur)),
                        n_embd / n_head, n_head, N),
          n_past, n_rot, 0);
      if (!kv_mem_jblas) {
        Vcur = ne_transpose(
            ctx0, ne_reshape_2d(ctx0, ne_view_1d(ctx0, QKVcur, N * n_embd, 2 * N * n_embd * ne_element_size(QKVcur)),
                                n_embd, N));
      } else {
        Vcur = ne_reshape_3d(ctx0, ne_view_1d(ctx0, QKVcur, N * n_embd, 2 * N * n_embd * ne_element_size(QKVcur)),
                             n_embd / n_head, n_head, N);
      }

    } else {
      Qcur = ne_rope_inplace(
          ctx0, ne_reshape_3d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[0], cur), n_embd / n_head, n_head, N),
          n_past, n_rot, 0);
      Kcur = ne_rope_inplace(
          ctx0, ne_reshape_3d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[1], cur), n_embd / n_head, n_head, N),
          n_past, n_rot, 0);
      if (!kv_mem_jblas) {
        Vcur = ne_transpose(ctx0, ne_reshape_2d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[2], cur), n_embd, N));
      } else {
        Vcur = ne_rope_inplace(
            ctx0, ne_reshape_3d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[2], cur), n_embd / n_head, n_head, N),
            n_past, n_rot, 0);
      }
    }
    ne_set_name(Qcur, "Qcur");
    ne_set_name(Kcur, "Kcur");
    ne_set_name(Vcur, "Vcur");
    // self-attention
    if (!kv_mem_jblas) {
      // store key and value to memory
      {
        struct ne_tensor* k =
            ne_view_1d(ctx0, kv_self.k, N * n_embd, (ne_element_size(kv_self.k) * n_embd) * (il * n_ctx + n_past));
        struct ne_tensor* v =
            ne_view_2d(ctx0, kv_self.v, N, n_embd, (n_ctx)*ne_element_size(kv_self.v),
                       (il * n_ctx) * ne_element_size(kv_self.v) * n_embd + n_past * ne_element_size(kv_self.v));

        // important: storing RoPE-ed version of K in the KV cache!
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur, k));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur, v));
      }

      struct ne_tensor* Q = ne_permute(ctx0, Qcur, 0, 2, 1, 3);
      ne_set_name(Q, "Q");

      struct ne_tensor* K = ne_permute(ctx0,
                                       ne_reshape_3d(ctx0,
                                                     ne_view_1d(ctx0, kv_self.k, (n_past + N) * n_embd,
                                                                il * n_ctx * ne_element_size(kv_self.k) * n_embd),
                                                     n_embd / n_head, n_head, n_past + N),
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
      struct ne_tensor* V = ne_view_3d(
          ctx0, kv_self.v, n_past + N, n_embd / n_head, n_head, n_ctx * ne_element_size(kv_self.v),
          n_ctx * ne_element_size(kv_self.v) * n_embd / n_head, il * n_ctx * ne_element_size(kv_self.v) * n_embd);
      ne_set_name(V, "V");

      struct ne_tensor* KQV = ne_mul_mat(ctx0, V, KQ_soft_max);
      ne_set_name(KQV, "KQV");

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct ne_tensor* KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);
      ne_set_name(KQV_merged, "KQV_merged");

      // cur = KQV_merged.contiguous().view(n_embd, N)
      cur = ne_cpy(ctx0, KQV_merged, ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N, NE_SIZE_CALC));
      ne_set_name(cur, "KQV_merged_contiguous");

      // projection (no bias)
      cur = ne_mul_mat(ctx0, model.layers[il].attn[3], cur);
    } else {
      const auto head_size = n_embd / n_head;
      const auto seq_kv = n_past + N;

      const auto k_size = kv_cache_info.k_bytes;
      const auto v_size = kv_cache_info.v_bytes;
      // store key and value to memory
      {
        const auto k_cache = ne_view_3d(ctx0, kv_self.k,           // tensor
                                        head_size, n_ctx, n_head,  // ne
                                        0, 0,                      // nb (jblas managed)
                                        il * k_size);              // offset
        ne_build_forward_expand(&gf, ne_flash_attn_update_k(ctx0, k_cache, Kcur, n_past));
        const auto v_cache = ne_view_3d(ctx0, kv_self.v,           // tensor
                                        head_size, n_ctx, n_head,  // ne
                                        0, 0,                      // nb (jblas managed)
                                        il * v_size);              // offset
        ne_build_forward_expand(&gf, ne_flash_attn_update_v(ctx0, v_cache, Vcur, n_past));
      }

      struct ne_tensor* Q = ne_permute(ctx0, Qcur, 0, 2, 1, 3);
      ne_set_name(Q, "Q");

      struct ne_tensor* K =
          ne_view_3d(ctx0, kv_self.k,                                             // tensor
                     head_size, seq_kv, n_head,                                   // ne
                     kv_cache_info.stride_k_sl, kv_cache_info.stride_k_head_num,  // nb (jblas managed)
                     il * k_size);                                                // offset
      *reinterpret_cast<ATTN_FWD_LAYOUT*>(&K->nb[0]) = kv_cache_info.k_layout;    // us nb0 for layout
      ne_set_name(K, "K");
      struct ne_tensor* V =
          ne_view_3d(ctx0, kv_self.v,                                                    // tensor
                     seq_kv, head_size, n_head,                                          // ne
                     kv_cache_info.stride_v_head_size, kv_cache_info.stride_v_head_num,  // nb (jblas managed)
                     il * v_size);                                                       // offset
      *reinterpret_cast<ATTN_FWD_LAYOUT*>(&V->nb[0]) = kv_cache_info.v_layout;           // us nb0 for layout
      ne_set_name(V, "V");

      struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, Q, K, V, 1.0f / sqrtf(float(n_embd) / n_head),
                                                n_past == 0);  // no causal mask on next-token cases
      struct ne_tensor* KQV_merged_contiguous =
          ne_view_2d(ctx0, KQV_Out, n_embd, N, n_embd * ne_element_size(KQV_Out), 0);
      ne_set_name(KQV_merged_contiguous, "KQV_merged_contiguous");

      // projection (no bias)
      cur = ne_mul_mat(ctx0, model.layers[il].attn[3], KQV_merged_contiguous);
    }

    lctx.use_buf(ctx0, 1);

    struct ne_tensor* inpFF = ne_add(ctx0, cur, inpSA);

    // feed-forward network
    {
      // norm
      {
        cur = ne_rms_norm(ctx0, inpFF);

        // cur = cur*ffn_norm(broadcasted)
        cur = ne_mul(ctx0, cur, model.layers[il].norm[1]);
      }

      if (jblas_fusion_FFN_SiLu_f32f32_support(model.layers[il].ffn[0]->data, model.layers[il].ffn[1]->data,
                                               model.layers[il].ffn[2]->data, N, cur->ne[0],
                                               model.layers[il].ffn[0]->ne[1], model.layers[il].ffn[1]->ne[1])) {
        cur = ne_ffn_silu(ctx0, model.layers[il].ffn[0], model.layers[il].ffn[1], model.layers[il].ffn[2], cur);
      } else {
        struct ne_tensor* tmp = ne_mul_mat(ctx0, model.layers[il].ffn[2], cur);

        cur = ne_mul_mat(ctx0, model.layers[il].ffn[0], cur);

        // SILU activation
        cur = ne_silu(ctx0, cur);

        cur = ne_mul(ctx0, cur, tmp);

        cur = ne_mul_mat(ctx0, model.layers[il].ffn[1], cur);
      }
    }

    cur = ne_add(ctx0, cur, inpFF);

    // input for next layer
    inpL = cur;
  }

  lctx.use_buf(ctx0, 0);

  // used at the end to optionally extract the embeddings
  struct ne_tensor* embeddings = NULL;

  // norm
  {
    inpL = ne_rms_norm(ctx0, inpL);

    // inpL = inpL*norm(broadcasted)
    inpL = ne_mul(ctx0, inpL, model.others[1]);

    embeddings = inpL;
  }

  // lm_head
  inpL = ne_mul_mat(ctx0, model.others[2], inpL);

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

int model_eval(struct model_context* ctx, const model_token* tokens, int n_tokens, int n_past, int n_threads) {
  if (!llama_model_eval_internal(*ctx, tokens, n_tokens, n_past, n_threads)) {
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
