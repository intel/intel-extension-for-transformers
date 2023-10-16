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
#include "models/opt/opt.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"
#include "models/models.h"
void model_load_internal(const std::string& fname, model_archs arch, model_context& lctx, int n_gpu_layers,
                         bool use_mmap, bool use_mlock, bool vocab_only, model_progress_callback progress_callback,
                         void* progress_callback_user_data) {
  lctx.t_start_us = ne_time_us();

  std::unique_ptr<IModel> ms(new OPT());
  ms->init(fname.c_str(), lctx, n_gpu_layers, use_mmap, use_mlock, vocab_only);
  ms->load(lctx, progress_callback, progress_callback_user_data);

  lctx.t_load_us = ne_time_us() - lctx.t_start_us;
}

void OPT::init(const char* path_model, model_context& lctx, int n_gpu_layer_, bool use_mmap_, bool use_mlock_,
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
  fprintf(stderr, "%s: n_vocab                  = %u\n", __func__, hparams.n_vocab);
  fprintf(stderr, "%s: n_embd                   = %u\n", __func__, hparams.n_embd);
  fprintf(stderr, "%s: n_head                   = %u\n", __func__, hparams.n_head);
  fprintf(stderr, "%s: n_layer                  = %u\n", __func__, hparams.n_layer);
  fprintf(stderr, "%s: n_ff                     = %u\n", __func__, n_ff);
  fprintf(stderr, "%s: n_parts                  = %zu\n", __func__, ml->file_loaders.size());
  fprintf(stderr, "%s: word_embed_proj_dim      = %u\n", __func__, hparams.word_embed_proj_dim);
  fprintf(stderr, "%s: max_seq_len      = %u\n", __func__, hparams.max_seq_len);
  n_embd = hparams.n_embd;
  n_vocab = hparams.n_vocab;
  n_layer = hparams.n_layer;
  word_embed_proj_dim = hparams.word_embed_proj_dim;
  max_seq_len = hparams.max_seq_len;
  do_layer_norm_before = hparams.do_layer_norm_before;
  scratch = opt_mem_req(n_layer);
  model.scratchs = scratch;
}

#define MODEL_BACKEND_OFFLOAD NE_BACKEND_CPU
void OPT::load(model_context& lctx, model_progress_callback progress_callback, void* progress_callback_user_data) {
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

  // OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
  // and adjust num_embeddings appropriately. Other models don't have this hack
  const uint32_t pos_offset = 2;
  model.others[0] = ml->get_tensor("model.decoder.embed_tokens.weight", {word_embed_proj_dim, n_vocab}, NE_BACKEND_CPU);
  model.others[1] =
      ml->get_tensor("model.decoder.embed_positions.weight", {n_embd, max_seq_len + pos_offset}, NE_BACKEND_CPU);
  if (do_layer_norm_before) {
    model.others[2] = ml->get_tensor("model.decoder.final_layer_norm.weight", {n_embd}, NE_BACKEND_CPU);
    model.others[3] = ml->get_tensor("model.decoder.final_layer_norm.bias", {n_embd}, NE_BACKEND_CPU);
  }
  if (word_embed_proj_dim != n_embd) {
    model.others[4] = ml->get_tensor("model.decoder.project_in.weight", {word_embed_proj_dim, n_embd}, NE_BACKEND_CPU);
    model.others[5] = ml->get_tensor("model.decoder.project_out.weight", {n_embd, word_embed_proj_dim}, NE_BACKEND_CPU);
  }
  model.others[6] = ml->get_tensor("lm_head.weight", {word_embed_proj_dim, n_vocab}, NE_BACKEND_CPU);
  const int i_gpu_start = n_layer - n_gpu_layer;

  model.layers.resize(n_layer);
  size_t vram_total = 0;
  for (uint32_t i = 0; i < n_layer; ++i) {
    const ne_backend backend = int(i) < i_gpu_start ? NE_BACKEND_CPU : MODEL_BACKEND_OFFLOAD;
    auto& layer = model.layers[i];
    std::string layers_i = "model.decoder.layers." + std::to_string(i);

    // norm: cur = ln_1_g*cur + ln_1_b
    layer.norm[0] = ml->get_tensor(layers_i + ".self_attn_layer_norm.weight", {n_embd}, backend);
    layer.norm[1] = ml->get_tensor(layers_i + ".self_attn_layer_norm.bias", {n_embd}, backend);
    layer.norm[2] = ml->get_tensor(layers_i + ".final_layer_norm.weight", {n_embd}, backend);
    layer.norm[3] = ml->get_tensor(layers_i + ".final_layer_norm.bias", {n_embd}, backend);

    // qkv GEMM + out proj GEMM
    layer.attn[0] = ml->get_tensor(layers_i + ".self_attn.q_proj.weight", {n_embd, n_embd}, backend);
    layer.attn[1] = ml->get_tensor(layers_i + ".self_attn.q_proj.bias", {n_embd}, backend);
    layer.attn[2] = ml->get_tensor(layers_i + ".self_attn.k_proj.weight", {n_embd, n_embd}, backend);
    layer.attn[3] = ml->get_tensor(layers_i + ".self_attn.k_proj.bias", {n_embd}, backend);
    layer.attn[4] = ml->get_tensor(layers_i + ".self_attn.v_proj.weight", {n_embd, n_embd}, backend);
    layer.attn[5] = ml->get_tensor(layers_i + ".self_attn.v_proj.bias", {n_embd}, backend);
    layer.attn[6] = ml->get_tensor(layers_i + ".self_attn.out_proj.weight", {n_embd, n_embd}, backend);
    layer.attn[7] = ml->get_tensor(layers_i + ".self_attn.out_proj.bias", {n_embd}, backend);

    // ffn GEMM
    layer.ffn[0] = ml->get_tensor(layers_i + ".fc1.weight", {n_embd, n_ff}, backend);
    layer.ffn[1] = ml->get_tensor(layers_i + ".fc1.bias", {n_ff}, backend);
    layer.ffn[2] = ml->get_tensor(layers_i + ".fc2.weight", {n_ff, n_embd}, backend);
    layer.ffn[3] = ml->get_tensor(layers_i + ".fc2.bias", {n_embd}, backend);

    if (backend != NE_BACKEND_CPU) {
      vram_total +=
          ne_nbytes(layer.norm[0]) + ne_nbytes(layer.norm[1]) + ne_nbytes(layer.norm[2]) + ne_nbytes(layer.norm[3]) +
          ne_nbytes(layer.attn[0]) + ne_nbytes(layer.attn[1]) + ne_nbytes(layer.attn[2]) + ne_nbytes(layer.attn[3]) +
          ne_nbytes(layer.attn[4]) + ne_nbytes(layer.attn[5]) + ne_nbytes(layer.attn[6]) + ne_nbytes(layer.attn[7]) +
          ne_nbytes(layer.ffn[0]) + ne_nbytes(layer.ffn[1]) + ne_nbytes(layer.ffn[2]) + ne_nbytes(layer.ffn[3]);
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
class opt_quant_layer : public quant_layer_base {
 public:
  virtual quant_params_internal get_layer_config(std::string layername, std::vector<int64_t> ne,
                                                 ne_type type) override {
    bool quantize = layername.rfind("weight") == layername.size() - 6;  // ends with 'weight'?
    if (layername == "model.decoder.embed_tokens.weight" || layername == "model.decoder.embed_positions.weight") {
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
REGISTER_QUANT_LAYER_CLASS(opt);
