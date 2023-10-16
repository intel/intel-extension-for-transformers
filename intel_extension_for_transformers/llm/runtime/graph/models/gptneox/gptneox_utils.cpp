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
#include "models/gptneox/gptneox.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"
#include "models/models.h"

void GPTNEOX::init(const char* path_model, model_context& lctx, int n_gpu_layer_, bool use_mmap_, bool use_mlock_,
                   bool vocab_only_) {
  n_gpu_layer = n_gpu_layer_;
  use_mmap = use_mmap_;
  use_mlock = use_mlock_;
  vocab_only = vocab_only_;
  auto& model = lctx.model;
  ml.reset(new model_model_loader(path_model, use_mmap, vocab_only));
  lctx.vocab = std::move(ml->file_loaders.at(0)->vocab);
  model.hparams = ml->file_loaders.at(0)->hparams;
  model_file_version file_version = ml->file_loaders.at(0)->file_version;
  auto& hparams = model.hparams;
  n_ff = 4 * hparams.n_embd;
  fprintf(stderr, "%s: n_vocab    = %u\n", __func__, hparams.n_vocab);
  fprintf(stderr, "%s: n_embd     = %u\n", __func__, hparams.n_embd);
  fprintf(stderr, "%s: n_mult     = %u\n", __func__, hparams.n_mult);
  fprintf(stderr, "%s: n_head     = %u\n", __func__, hparams.n_head);
  fprintf(stderr, "%s: n_layer    = %u\n", __func__, hparams.n_layer);
  fprintf(stderr, "%s: n_rot      = %u\n", __func__, hparams.n_rot);
  fprintf(stderr, "%s: n_ff       = %u\n", __func__, n_ff);
  fprintf(stderr, "%s: n_parts    = %zu\n", __func__, ml->file_loaders.size());
  n_embd = hparams.n_embd;
  n_vocab = hparams.n_vocab;
  n_layer = hparams.n_layer;
  scratch = gptneox_mem_req(n_layer);
  model.scratchs = scratch;
}

#define MODEL_BACKEND_OFFLOAD NE_BACKEND_CPU
void GPTNEOX::load(model_context& lctx, model_progress_callback progress_callback, void* progress_callback_user_data) {
  auto& model = lctx.model;
  auto& ctx = model.ctx;

  size_t ctx_size;
  size_t mmapped_size;
  ml->calc_sizes(&ctx_size, &mmapped_size);
  fprintf(stderr, "%s: ne ctx size = %7.2f MB\n", __func__, ctx_size / 1024.0 / 1024.0);

  // create the ne context
  lctx.model.buf.resize(ctx_size);
  if (use_mlock) {
    lctx.model.mlock_buf.init(lctx.model.buf.addr);
    lctx.model.mlock_buf.grow_to(lctx.model.buf.size);
  }

  struct ne_init_params params = {
      /*.mem_size   =*/lctx.model.buf.size,
      /*.mem_buffer =*/lctx.model.buf.addr,
      /*.no_alloc   =*/ml->use_mmap,
  };

  model.ctx = ne_init(params);
  if (!model.ctx) {
    throw format("ne_init() failed");
  }

  ml->ne_ctx = ctx;

  model.others[0] = ml->get_tensor("gpt_neox.embed_in.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);
  model.others[1] = ml->get_tensor("gpt_neox.final_layer_norm.weight", {n_embd}, NE_BACKEND_CPU);
  model.others[2] = ml->get_tensor("gpt_neox.final_layer_norm.bias", {n_embd}, NE_BACKEND_CPU);
  model.others[3] = ml->get_tensor("embed_out.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);
  const int i_gpu_start = n_layer - n_gpu_layer;

  model.layers.resize(n_layer);
  size_t vram_total = 0;
  for (uint32_t i = 0; i < n_layer; ++i) {
    const ne_backend backend = int(i) < i_gpu_start ? NE_BACKEND_CPU : MODEL_BACKEND_OFFLOAD;
    auto& layer = model.layers[i];
    std::string layers_i = "gpt_neox.layers." + std::to_string(i);

    // norm: cur = ln_1_g*cur + ln_1_b
    layer.norm[0] = ml->get_tensor(layers_i + ".input_layernorm.weight", {n_embd}, backend);
    layer.norm[1] = ml->get_tensor(layers_i + ".input_layernorm.bias", {n_embd}, backend);
    layer.norm[2] = ml->get_tensor(layers_i + ".post_attention_layernorm.weight", {n_embd}, backend);
    layer.norm[3] = ml->get_tensor(layers_i + ".post_attention_layernorm.bias", {n_embd}, backend);

    // qkv GEMM
    layer.attn[0] = ml->get_tensor(layers_i + ".attention.query_key_value.weight", {n_embd, 3 * n_embd}, backend);
    layer.attn[1] = ml->get_tensor(layers_i + ".attention.query_key_value.bias", {3 * n_embd}, backend);
    layer.attn[2] = ml->get_tensor(layers_i + ".attention.dense.weight", {n_embd, n_embd}, backend);
    layer.attn[3] = ml->get_tensor(layers_i + ".attention.dense.bias", {n_embd}, backend);

    // ffn GEMM
    layer.ffn[0] = ml->get_tensor(layers_i + ".mlp.dense_h_to_4h.weight", {n_embd, n_ff}, backend);
    layer.ffn[1] = ml->get_tensor(layers_i + ".mlp.dense_h_to_4h.bias", {n_ff}, backend);
    layer.ffn[2] = ml->get_tensor(layers_i + ".mlp.dense_4h_to_h.weight", {n_ff, n_embd}, backend);
    layer.ffn[3] = ml->get_tensor(layers_i + ".mlp.dense_4h_to_h.bias", {n_embd}, backend);

    if (backend != NE_BACKEND_CPU) {
      vram_total += ne_nbytes(layer.norm[0]) + ne_nbytes(layer.norm[1]) + ne_nbytes(layer.norm[2]) +
                    ne_nbytes(layer.norm[3]) + ne_nbytes(layer.attn[0]) + ne_nbytes(layer.attn[1]) +
                    ne_nbytes(layer.attn[2]) + ne_nbytes(layer.attn[3]) + ne_nbytes(layer.ffn[0]) +
                    ne_nbytes(layer.ffn[1]) + ne_nbytes(layer.ffn[2]) + ne_nbytes(layer.ffn[3]);
    }
  }

  // print memory requirements
  // this is the total memory required to run the inference
  const size_t mem_required = ctx_size + mmapped_size - vram_total +  // weights in VRAM not in memory
                              scratch.scratch0 + scratch.scratch1 + scratch.eval;
  fprintf(stderr, "%s: mem required  = %7.2f MB (+ memory per state)\n", __func__, mem_required / 1024.0 / 1024.0);

  (void)n_gpu_layer;

  // populate `tensors_by_name`
  for (model_load_tensor& lt : ml->tensors_map.tensors) {
    model.tensors_by_name.emplace_back(lt.name, lt.ne_tensor);
  }

  ml->load_all_data(progress_callback, progress_callback_user_data, use_mlock ? &lctx.model.mlock_mmap : NULL);

  if (progress_callback) {
    progress_callback(1.0f, progress_callback_user_data);
  }

  model.mapping = std::move(ml->mapping);
}

#undef MODEL_BACKEND_OFFLOAD

class gptneox_quant_layer : public quant_layer_base {
 public:
  virtual quant_params_internal get_layer_config(std::string layername, std::vector<int64_t> ne,
                                                 ne_type type) override {
    bool quantize = layername.rfind("weight") == layername.size() - 6;  // ends with 'weight'?
    if (layername == "gpt_neox.embed_in.weight") {
      // special layer process, can be loaded by config file
      return quant_params_internal();  // return q4_0 to cover the usage of getrow
    }
    quantize &= (ne.size() == 2);
    if (quantize) {
      return mGCfg;  // use global quant config
    } else {
      return quant_params_internal{quant_bits::count};  // non-quant
    }
  }
};
REGISTER_QUANT_LAYER_CLASS(gptneox);

class gptneox_beam_search_kv_cache_reorder : public beam_search_kv_cache_reorder {
 public:
  explicit gptneox_beam_search_kv_cache_reorder(model_context* lctx) : beam_search_kv_cache_reorder(lctx) {}
  ~gptneox_beam_search_kv_cache_reorder() {}

  virtual void update(const uint32_t& n_past, const uint32_t& n_prompt_tokens,
                      const std::vector<std::tuple<int, int>>& kv_reorder_indices = {},
                      const std::vector<beam>& next_beams = {}) override {
    // TODO(Yi): use get_batch_kv_elements_from_gpt_params;
    NE_ASSERT(ctx->model.kv_self.k->type != NE_TYPE_JBLAS);
    // first step
    if (n_past == n_prompt_tokens) {
      // cpy batch 1 to all batches
#pragma omp parallel for collapse(3)
      for (int i = 0; i < ctx->model.layers.size(); ++i) {  // K
        for (int j = 1; j < kv_n_ctx_block; ++j) {
          // [head_dim, N, n_head]
          for (int nh = 0; nh < n_head; ++nh) {
            memcpy(static_cast<char*>(ctx->model.kv_self.k->data) +
                       (i * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd * kv_n_ctx_block +
                        j * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd) +
                       ne_element_size(ctx->model.kv_self.k) * nh * head_dim * n_ctx,
                   static_cast<char*>(ctx->model.kv_self.k->data) +
                       i * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd * kv_n_ctx_block +
                       ne_element_size(ctx->model.kv_self.k) * nh * head_dim * n_ctx,
                   ne_element_size(ctx->model.kv_self.k) * head_dim * n_prompt_tokens);
          }
        }
      }
#pragma omp parallel for collapse(3)
      for (int i = 0; i < ctx->model.layers.size(); ++i) {  // V
        for (int j = 1; j < kv_n_ctx_block; ++j) {
          // [N, head_dim, n_head] or [N, n_embd]
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
      for (auto t : kv_reorder_indices) {
        int cur_id = std::get<0>(t);
        int cpy_id = std::get<1>(t);
        if (cur_id != cpy_id) {
          uint32_t len = next_beams[cur_id].token_ids.size() - 1;
          // last token in beam is for next step inference
          MODEL_ASSERT(len == n_past - n_prompt_tokens);
          size_t input_token_offset_k = n_prompt_tokens * ne_element_size(ctx->model.kv_self.k) * head_dim;
          size_t input_token_offset_v = n_prompt_tokens * ne_element_size(ctx->model.kv_self.v);
          if (len + n_prompt_tokens > n_ctx) {
            // all token hidden states cache should be updated
            input_token_offset_k = 0;
            input_token_offset_v = 0;
            len = n_ctx;
          }
#pragma omp parallel for collapse(2)
          for (int i = 0; i < ctx->model.layers.size(); ++i) {  // K
            // [head_dim, N, n_head]
            for (int nh = 0; nh < n_head; ++nh) {
              memcpy(static_cast<char*>(ctx->model.kv_self.k->data) +
                         (i * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd * kv_n_ctx_block +
                          cur_id * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd) +
                         ne_element_size(ctx->model.kv_self.k) * nh * head_dim * n_ctx + input_token_offset_k,
                     static_cast<char*>(ctx->model.kv_self.k->data) +
                         i * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd * kv_n_ctx_block +
                         cpy_id * n_ctx * ne_element_size(ctx->model.kv_self.k) * n_embd +
                         ne_element_size(ctx->model.kv_self.k) * nh * head_dim * n_ctx + input_token_offset_k,
                     ne_element_size(ctx->model.kv_self.k) * head_dim * len);
            }
          }
#pragma omp parallel for collapse(2)
          for (int i = 0; i < ctx->model.layers.size(); ++i) {  // V
            // [N, head_dim, n_head] or [N, n_embd]
            for (int k = 0; k < n_embd; ++k) {
              memcpy(static_cast<char*>(ctx->model.kv_self.v->data) +
                         (i * n_ctx * ne_element_size(ctx->model.kv_self.v) * n_embd * kv_n_ctx_block +
                          cur_id * n_ctx * ne_element_size(ctx->model.kv_self.v) * n_embd +
                          n_ctx * ne_element_size(ctx->model.kv_self.v) * k + input_token_offset_v),
                     static_cast<char*>(ctx->model.kv_self.v->data) +
                         (i * n_ctx * ne_element_size(ctx->model.kv_self.v) * n_embd * kv_n_ctx_block +
                          cpy_id * n_ctx * ne_element_size(ctx->model.kv_self.v) * n_embd +
                          n_ctx * ne_element_size(ctx->model.kv_self.v) * k + input_token_offset_v),
                     ne_element_size(ctx->model.kv_self.v) * len);
            }
          }
        }
      }
    } else {
      return;
    }
  }
};

void model_load_internal(const std::string& fname, model_archs arch, model_context& lctx, int n_gpu_layers,
                         bool use_mmap, bool use_mlock, bool vocab_only, model_progress_callback progress_callback,
                         void* progress_callback_user_data) {
  lctx.t_start_us = ne_time_us();

  std::unique_ptr<IModel> ms(new GPTNEOX());
  ms->init(fname.c_str(), lctx, n_gpu_layers, use_mmap, use_mlock, vocab_only);
  ms->load(lctx, progress_callback, progress_callback_user_data);
  if (lctx.beam_search) {
    lctx.bs_kv_reorder = std::make_shared<gptneox_beam_search_kv_cache_reorder>(&lctx);
#ifdef NE_BEAM_SEARCH_VERBOSE_ON
    printf("get GPTNEOX beam search kv cache update function. \n");
#endif
  }

  lctx.t_load_us = ne_time_us() - lctx.t_start_us;
}
