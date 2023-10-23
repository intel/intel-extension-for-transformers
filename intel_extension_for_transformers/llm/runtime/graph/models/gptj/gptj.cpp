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
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/data_types.h"
#include "core/layers/mha_dense.h"
#include "core/ne.h"
#include "core/ne_layers.h"
#include "core/ne_jblas.h"
#include "core/layers/mha_dense.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"

#define MHA_FUSION 0  //  turn it off for naive beam_search kv cache reorder
#define MHA_FP16 (MHA_FUSION && 0)

// evaluate the transformer
//
//   - lctx:      model context
//   - tokens:    new batch of tokens to process
//   - n_past:    the offset to which the kv is cached to
//   - n_total:   the number of tokens evaluated so far (including evicted tokens if there is any)
//   - n_threads: number of threads to use
//
static bool gptj_model_eval_internal(model_context& lctx, const model_token* tokens, const int n_tokens,
                                     const int n_past, const int n_total, const int n_threads) {
  const int64_t t_start_us = ne_time_us();

  const int batch_size = lctx.batch_size;  // num of beams of all batches
  const int N = n_tokens;

  const auto& model = lctx.model;
  const auto& hparams = model.hparams;

  const auto& kv_self = model.kv_self;

  MODEL_ASSERT(!!kv_self.ctx);

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = lctx.n_ctx;  // max number fo tokens to keep in the kv-cache
  const int n_keep = lctx.n_keep;
  const bool shift_roped_k = lctx.shift_roped_k;
  const bool is_ring_full = shift_roped_k && n_total > n_past;
  const int n_cached = shift_roped_k ? std::min(n_total + N, n_ctx) : (n_past + N);  // #tokens cached after kv-append
  int n_head = hparams.n_head;
  const int head_size = n_embd / n_head;
  const int n_vocab = hparams.n_vocab;
  const int n_rot = hparams.n_rot;

  bool enable_tp = false;
#ifdef NE_TP_MODEL
  parallel_context* p_ctx = init_parallel_context();
  int32_t world_size = get_tp_size(p_ctx);
  int32_t rank = get_tp_rank(p_ctx);
  enable_tp = world_size > 1 ? true : false;
  // IMPORTANT, when TP, the n_head will 1 / world_size
  if (enable_tp) {
    n_head /= world_size;
  }
#endif

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
  gf.n_threads = n_threads;

  const bool run_mha_reordered = kv_self.k->type == NE_TYPE_JBLAS;
  const bool run_mha_fp16 = !run_mha_reordered && MHA_FP16 && jblas_fusion_attn_fp16_support(NULL);
  const bool run_mha_bf16_first =
      !run_mha_reordered && MHA_FUSION && !MHA_FP16 && jblas_fusion_attn_fp32_fp16_fp16_fp32_support(NULL);
  kv_cache_info_t kv_cache_info = {0, 0};
  if (run_mha_reordered) {
    NE_ASSERT(kv_self.v->type == NE_TYPE_JBLAS);  // kv type should be the same
    attn_shape_t attn_shape = {
        /* .batch_size = */ batch_size,
        /* .head_num = */ n_head,
        /* .heads_kv = */ n_head,  // GPT-J does not have MQA/GQA
        /* .head_size = */ head_size,
        /* .sl_q = */ N,  // Note: make sure that jblas reordered attn supports next token inference
        /* .sl_kv = */ n_cached,
    };
    NE_ASSERT(("jblas managed kv-cache not supported; use `--memory-f16 / --memory-f32` instead",
               jblas_reordered_attn_fp32_support(&attn_shape)));
    kv_shape_t kv_shape{
        /* .head_num = */ static_cast<uint32_t>(n_head),
        /* .head_size = */ static_cast<uint32_t>(head_size),
        /* .sl_kv_max = */ static_cast<uint32_t>(n_ctx),
    };
    jblas_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
  }

  struct ne_tensor* embd = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, N * batch_size);
  ne_set_name(embd, "embd");
  for (int i = 0; i < batch_size; ++i) {
    memcpy(static_cast<model_token*>(embd->data) + i * N, tokens + i * N, N * ne_element_size(embd));
  }

#ifdef NE_TP_MODEL
  if (enable_tp) {
    // need to broadcast the ids
    broadcast(p_ctx, (float*)embd->data, N * batch_size * ne_element_size(embd));
  }
#endif

  struct ne_tensor* inpL = ne_get_rows(ctx0, model.others[0], embd);

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;

    lctx.use_buf(ctx0, 0);

    // norm
    cur = ne_norm(ctx0, inpL);

    // cur = ln_1_g*cur + ln_1_b
    cur = ne_add(ctx0, ne_mul(ctx0, cur, model.layers[il].norm[0]), model.layers[il].norm[1]);

    struct ne_tensor* inpSA = cur;

    ne_tensor *Qcur, *Kcur, *Vcur;
    int kv_n_ctx_block = lctx.kv_n_ctx_block;
    if (jblas_fusion_QKV_f32f32_support(model.layers[il].attn[0]->data, model.layers[il].attn[1]->data,
                                        model.layers[il].attn[2]->data, N * batch_size, head_size * n_head,
                                        head_size * n_head)) {  // fused execution of QKV
                                                                // if (false) {
      struct ne_tensor* QKVcur =
          ne_mul_qkv(ctx0, model.layers[il].attn[0], model.layers[il].attn[1], model.layers[il].attn[2], cur);
      const size_t qkv_size = N * head_size * n_head * batch_size;
      const size_t qkv_bytes = qkv_size * ne_element_size(QKVcur);
      Qcur = ne_reshape_4d(ctx0, ne_view_1d(ctx0, QKVcur, qkv_size, 0 * qkv_bytes), head_size, n_head, N, batch_size);
      Kcur = ne_reshape_4d(ctx0, ne_view_1d(ctx0, QKVcur, qkv_size, 1 * qkv_bytes), head_size, n_head, N, batch_size);
      Vcur = ne_view_1d(ctx0, QKVcur, qkv_size, 2 * qkv_bytes);
    } else {
      Qcur = ne_reshape_4d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[0], cur), head_size, n_head, N, batch_size);
      Kcur = ne_reshape_4d(ctx0, ne_mul_mat(ctx0, model.layers[il].attn[1], cur), head_size, n_head, N, batch_size);
      Vcur = ne_mul_mat(ctx0, model.layers[il].attn[2], cur);
    }
    Qcur = ne_rope_inplace(ctx0, Qcur, std::max(n_cached - N, n_past), n_rot, 0, 0);
    Kcur = ne_rope_inplace(  // n_ctx exceeds but it will be shift-roped back with cached K
        ctx0, Kcur, (is_ring_full ? n_ctx : n_past), n_rot, 0, 0);
    ne_set_name(Qcur, "Qcur");
    ne_set_name(Kcur, "Kcur");
    ne_set_name(Vcur, "Vcur");
    // self-attention
    // store key and value to memory
    // important: storing RoPE-ed version of K in the KV cache!
    if (!run_mha_reordered) {
      std::vector<ne_tensor*> Kcur_bs(batch_size);
      std::vector<ne_tensor*> Vcur_bs(batch_size);
      std::vector<ne_tensor*> k_bs(batch_size);
      std::vector<ne_tensor*> v_bs(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        if (run_mha_fp16) {
          // batch V
          Vcur_bs[i] =
              ne_view_4d(ctx0, Vcur, head_size, n_head, N, 1, ne_element_size(Vcur) * head_size,
                         ne_element_size(Vcur) * head_size * n_head, ne_element_size(Vcur) * head_size * n_head * N,
                         i * ne_element_size(Vcur) * head_size * n_head * N);
          v_bs[i] =
              ne_view_1d(ctx0, kv_self.v, head_size * n_head * N * 1,
                         (ne_element_size(kv_self.v) * head_size * n_head) * (il * n_ctx * kv_n_ctx_block + n_past) +
                             i * n_ctx * head_size * n_head * ne_element_size(kv_self.v));
          // batch K
          Kcur_bs[i] = ne_permute(
              ctx0,
              ne_reshape_4d(ctx0,
                            ne_view_2d(ctx0, Kcur, head_size * n_head, N, ne_element_size(Kcur) * head_size * n_head,
                                       i * ne_element_size(Kcur) * head_size * n_head * N),
                            head_size, n_head, N, 1),
              1, 2, 0, 3);
          k_bs[i] = ne_view_4d(
              ctx0, kv_self.k, N, head_size, n_head, 1, n_ctx * ne_element_size(kv_self.k),
              n_ctx * ne_element_size(kv_self.k) * head_size, n_ctx * ne_element_size(kv_self.k) * head_size * n_head,
              ((il * n_ctx) * ne_element_size(kv_self.k) * head_size * n_head * kv_n_ctx_block +
               i * n_ctx * head_size * n_head * ne_element_size(kv_self.k) + n_past * ne_element_size(kv_self.k)));
        } else {
          // batch K
          Kcur_bs[i] =
              ne_permute(ctx0,
                         ne_view_4d(ctx0, Kcur, n_embd / n_head, n_head, N, 1, ne_element_size(Kcur) * n_embd / n_head,
                                    ne_element_size(Kcur) * n_embd, ne_element_size(Kcur) * n_embd * N,
                                    i * ne_element_size(Kcur) * n_embd * N),
                         0, 2, 1, 3);
          k_bs[i] = ne_view_4d(
              ctx0, kv_self.k, n_embd / n_head, N, n_head, 1, ne_element_size(kv_self.k) * n_embd / n_head,
              ne_element_size(kv_self.k) * n_embd / n_head * n_ctx, ne_element_size(kv_self.k) * n_embd * n_ctx,
              ((il * n_ctx) * ne_element_size(kv_self.k) * n_embd * kv_n_ctx_block +
               i * n_ctx * n_embd * ne_element_size(kv_self.k) +
               n_embd / n_head * n_past * ne_element_size(kv_self.k)));

          // batch V
          Vcur_bs[i] = ne_permute(
              ctx0,
              ne_reshape_4d(ctx0,
                            ne_view_2d(ctx0, Vcur, head_size * n_head, N, ne_element_size(Vcur) * head_size * n_head,
                                       i * ne_element_size(Vcur) * head_size * n_head * N),
                            head_size, n_head, N, 1),
              1, 2, 0, 3);
          v_bs[i] = ne_view_4d(
              ctx0, kv_self.v, N, head_size, n_head, 1, n_ctx * ne_element_size(kv_self.v),
              n_ctx * ne_element_size(kv_self.v) * head_size, n_ctx * ne_element_size(kv_self.v) * head_size * n_head,
              ((il * n_ctx) * ne_element_size(kv_self.v) * head_size * n_head * kv_n_ctx_block +
               i * n_ctx * head_size * n_head * ne_element_size(kv_self.v) + n_past * ne_element_size(kv_self.v)));
        }
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur_bs[i], k_bs[i]));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur_bs[i], v_bs[i]));
      }
    } else {
      const auto k_size = kv_cache_info.k_bytes;
      const auto v_size = kv_cache_info.v_bytes;
      const auto k_cache = ne_view_4d(ctx0, kv_self.k,                       // tensor
                                      head_size, n_ctx, n_head, batch_size,  // ne
                                      0, 0, k_size,                          // nb (jblas managed)
                                      il * kv_n_ctx_block * k_size);         // offset
      ne_build_forward_expand(&gf, ne_flash_attn_update_k(ctx0, k_cache, Kcur, n_past, is_ring_full));
      const auto v_cache = ne_view_4d(ctx0, kv_self.v,                       // tensor
                                      head_size, n_ctx, n_head, batch_size,  // ne
                                      0, 0, v_size,                          // nb (jblas managed)
                                      il * kv_n_ctx_block * v_size);         // offset
      // jblas alway view V as (D, n_head, seq, bs)
      const auto Vcur_plain = ne_reshape_4d(ctx0, Vcur, head_size, n_head, N, batch_size);
      ne_build_forward_expand(&gf, ne_flash_attn_update_v(ctx0, v_cache, Vcur_plain, n_past, is_ring_full));
    }

    struct ne_tensor* Q = ne_permute(ctx0, Qcur, 0, 2, 1, 3);
    ne_set_name(Q, "Q");
    struct ne_tensor *K, *V;
    if (run_mha_reordered) {
      const auto k_size = kv_cache_info.k_bytes;
      K = ne_view_4d(ctx0, kv_self.k,                                                     // tensor
                     head_size, n_cached, n_head, batch_size,                             // ne
                     kv_cache_info.stride_k_sl, kv_cache_info.stride_k_head_num, k_size,  // nb (jblas managed)
                     il * kv_n_ctx_block * k_size);                                       // offset
      *reinterpret_cast<ATTN_FWD_LAYOUT*>(&K->nb[0]) = kv_cache_info.k_layout;
      if (is_ring_full) {
        struct ne_tensor* cossin_cache = nullptr;
        // Currently we only cache cossin for N == 1 in model-wide; It may be worthwhile to cache cossin for other N
        // in a single eval execution
        if (N == 1) cossin_cache = kv_self.cossin;
        K = ne_rope_shift_inplace(ctx0, K, -N, n_rot, 0, 0, n_keep, cossin_cache);
      }
      const auto v_size = kv_cache_info.v_bytes;
      V = ne_view_4d(ctx0, kv_self.v,                                                            // tensor
                     n_cached, head_size, n_head, batch_size,                                    // ne
                     kv_cache_info.stride_v_head_size, kv_cache_info.stride_v_head_num, v_size,  // nb (jblas managed)
                     il * kv_n_ctx_block * v_size);                                              // offset
      *reinterpret_cast<ATTN_FWD_LAYOUT*>(&V->nb[0]) = kv_cache_info.v_layout;
    } else if (run_mha_fp16) {
      V = ne_permute(ctx0,
                     ne_view_4d(ctx0, kv_self.v, head_size, n_head, n_cached, batch_size,
                                ne_element_size(kv_self.v) * head_size, ne_element_size(kv_self.v) * head_size * n_head,
                                ne_element_size(kv_self.v) * head_size * n_head * n_ctx,
                                il * n_ctx * ne_element_size(kv_self.v) * head_size * n_head * kv_n_ctx_block),
                     1, 2, 0, 3);

      // split cached V into n_head heads
      K = ne_view_4d(ctx0, kv_self.k, n_cached, head_size, n_head, batch_size, n_ctx * ne_element_size(kv_self.k),
                     n_ctx * ne_element_size(kv_self.k) * head_size,
                     n_ctx * ne_element_size(kv_self.k) * head_size * n_head,
                     il * n_ctx * ne_element_size(kv_self.k) * head_size * n_head * kv_n_ctx_block);
      K = ne_permute(ctx0, K, 1, 0, 2, 3);  // head_size n_cached n_head batch_size
      if (is_ring_full) {
        K = ne_permute(ctx0, K, 0, 2, 1, 3);
        struct ne_tensor* cossin_cache = nullptr;
        // Currently we only cache cossin for N == 1 in model-wide; It may be worthwhile to cache cossin for other N in
        // a single eval execution
        if (N == 1) cossin_cache = kv_self.cossin;
        K = ne_rope_shift_inplace(ctx0, K, -N, n_rot, 0, 0, n_keep, cossin_cache);
        K = ne_permute(ctx0, K, 0, 2, 1, 3);
      }
    } else {
      K = ne_view_4d(ctx0, kv_self.k, head_size, n_cached, n_head, batch_size, ne_element_size(kv_self.k) * head_size,
                     ne_element_size(kv_self.k) * head_size * n_ctx,
                     ne_element_size(kv_self.k) * head_size * n_head * n_ctx,
                     il * n_ctx * ne_element_size(kv_self.k) * head_size * n_head * kv_n_ctx_block);
      K = ne_permute(ctx0, K, 0, 2, 1, 3);
      if (is_ring_full) {
        struct ne_tensor* cossin_cache = nullptr;
        // Currently we only cache cossin for N == 1 in model-wide; It may be worthwhile to cache cossin for other N in
        // a single eval execution
        if (N == 1) cossin_cache = kv_self.cossin;
        K = ne_rope_shift_inplace(ctx0, K, -N, n_rot, 0, 0, n_keep, cossin_cache);
      }
      K = ne_permute(ctx0, K, 0, 2, 1, 3);

      // split cached V into n_head heads
      V = ne_view_4d(ctx0, kv_self.v, n_cached, head_size, n_head, batch_size, n_ctx * ne_element_size(kv_self.v),
                     n_ctx * ne_element_size(kv_self.v) * head_size,
                     n_ctx * ne_element_size(kv_self.v) * head_size * n_head,
                     il * n_ctx * ne_element_size(kv_self.v) * head_size * n_head * kv_n_ctx_block);
    }
    ne_set_name(K, "K");
    ne_set_name(V, "V");

    struct ne_tensor* KQV_merged_contiguous;

    const float attn_scale = 1.0f / sqrtf(float(head_size));
    ne_attn_flags_t attn_flags = NE_ATTN_FLAG_NONE;
    if (n_total == 0 || !shift_roped_k) attn_flags |= NE_ATTN_FLAG_IS_CAUSAL;  // no causal mask on next-token cases
    if (run_mha_reordered) {  // reordered kv-cache bf16 mha must be used if run_mha_reordered
      struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, Q, K, V, attn_scale, attn_flags);
      KQV_merged_contiguous = ne_view_2d(ctx0, KQV_Out, head_size * n_head, N * batch_size,
                                         head_size * n_head * ne_element_size(KQV_Out), 0);
    } else if (run_mha_fp16) {  // non-reordered kv-cache fp16 mha
      struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, Q, K, V, attn_scale, attn_flags);
      KQV_merged_contiguous = ne_view_2d(ctx0, KQV_Out, head_size * n_head, N * batch_size,
                                         head_size * n_head * ne_element_size(KQV_Out), 0);
    } else if (n_total == 0 && run_mha_bf16_first) {
      // non-reordered kv-cache bf16 mha (first token only)
      auto vnele = ne_nelements(Vcur);
      struct ne_tensor* Vtmp = ne_new_tensor_1d(ctx0, NE_TYPE_F16, vnele, NE_SIZE_CALC);
      Vtmp = ne_cpy(ctx0, ne_view_1d(ctx0, Vcur, vnele, 0), Vtmp);
      Vtmp = ne_view_4d(ctx0, Vtmp, head_size, n_head, N, batch_size, ne_element_size(Vtmp) * head_size,
                        ne_element_size(Vtmp) * head_size * n_head, N * ne_element_size(Vtmp) * head_size * n_head, 0);
      Vtmp = ne_permute(ctx0, Vtmp, 1, 2, 0, 3);
      struct ne_tensor* KQV_Out = ne_flash_attn(ctx0, Q, K, Vtmp, attn_scale, attn_flags);
      KQV_merged_contiguous = ne_view_2d(ctx0, KQV_Out, head_size * n_head, N * batch_size,
                                         head_size * n_head * ne_element_size(KQV_Out), 0);
    } else {
      // K * Q
      struct ne_tensor* KQ = ne_mul_mat(ctx0, K, Q);
      ne_set_name(KQ, "KQ");

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      struct ne_tensor* KQ_scale = ne_new_f32(ctx0, attn_scale);
      ne_set_name(KQ_scale, "1/sqrt(n_embd/n_head)");

      // KQ_scaled shape [n_cached, N, n_head, 1]
      struct ne_tensor* KQ_scaled = ne_scale_inplace(ctx0, KQ, KQ_scale);
      ne_set_name(KQ_scaled, "KQ_scaled");

      // KQ_scaled = mask_past(KQ_scaled)
      if (n_total == 0 || !shift_roped_k) {
        KQ_scaled = ne_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
        ne_set_name(KQ_scaled, "KQ_masked");
      }

      // KQ = soft_max(KQ_masked)
      struct ne_tensor* KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_scaled);
      ne_set_name(KQ_soft_max, "KQ_soft_max");

      struct ne_tensor* KQV = ne_mul_mat(ctx0, V, KQ_soft_max);
      ne_set_name(KQV, "KQV");

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct ne_tensor* KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);
      ne_set_name(KQV_merged, "KQV_merged");

      // cur = KQV_merged.contiguous().view(n_embd, N)
      KQV_merged_contiguous = ne_cpy(
          ctx0, KQV_merged, ne_new_tensor_2d(ctx0, NE_TYPE_F32, head_size * n_head, N * batch_size, NE_SIZE_CALC));
    }
    ne_set_name(KQV_merged_contiguous, "KQV_merged_contiguous");

    // projection (no bias)
    struct ne_tensor* KQV_out = ne_mul_mat(ctx0, model.layers[il].attn[3], KQV_merged_contiguous);
    ne_set_name(KQV_out, "KQV_out");

#ifdef NE_TP_MODEL
    if (enable_tp) {
      KQV_out = ne_all_reduce(ctx0, KQV_out);
    }
#endif

    lctx.use_buf(ctx0, 1);
    struct ne_tensor* inpFF = KQV_out;

    // feed-forward network
    // disable ffn fusion because fp32 support not ready
    if (jblas_fusion_FFN_Add_GeLu_f32f32_support(model.layers[il].ffn[0]->data, model.layers[il].ffn[2]->data,
                                                 N * batch_size, inpSA->ne[0], model.layers[il].ffn[0]->ne[1],
                                                 model.layers[il].ffn[2]->ne[1])) {
      cur = ne_ffn_add_gelu(ctx0, model.layers[il].ffn[0], model.layers[il].ffn[2], model.layers[il].ffn[1],
                            model.layers[il].ffn[3], inpSA);
    } else {
      struct ne_tensor* FFN_in = ne_mul_mat(ctx0, model.layers[il].ffn[0], inpSA);
      ne_set_name(FFN_in, "FFN_in");

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].ffn[1], FFN_in), FFN_in);

      // GELU activation
      cur = ne_gelu(ctx0, cur);

      struct ne_tensor* FFN_out = ne_mul_mat(ctx0, model.layers[il].ffn[2], cur);
      ne_set_name(FFN_out, "FFN_out");

#ifdef NE_TP_MODEL
      // if tp model then all reduce as the weight has been split
      if (enable_tp) {
        FFN_out = ne_all_reduce(ctx0, FFN_out);
      }
#endif
      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].ffn[3], FFN_out), FFN_out);
    }
    cur = ne_add(ctx0, cur, inpFF);
    // if (il == 20) {
    //   cur = ne_dump_tensor(ctx0, cur);
    // }

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
    inpL = ne_add(ctx0, ne_mul(ctx0, inpL, model.others[1]), model.others[2]);
  }

  // lm_head
  if (jblas_fusion_add_f32f32_support(model.others[3]->data, N * batch_size, model.others[3]->ne[1],
                                      model.others[3]->ne[0])) {
    inpL = ne_mul_mat_with_bias(ctx0, model.others[3], model.others[4], inpL);
  } else {
    inpL = ne_mul_mat(ctx0, model.others[3], inpL);
    inpL = ne_add(ctx0, ne_repeat(ctx0, model.others[4], inpL), inpL);
  }

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
  lctx.model.kv_self.n = n_cached;

  // extract logits
  {
    auto& logits_out = lctx.logits;

    size_t bs_stride = n_vocab * N;
    if (lctx.logits_all) {
      logits_out.resize(n_vocab * N * batch_size);
      memcpy(logits_out.data(), (float*)ne_get_data(inpL), sizeof(float) * n_vocab * N * batch_size);
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

int model_eval(struct model_context* ctx, const model_token* tokens, int n_tokens, int n_past, int n_total,
               int n_threads) {
  if (!gptj_model_eval_internal(*ctx, tokens, n_tokens, n_past, n_total, n_threads)) {
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
