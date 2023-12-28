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
// Defines fileno on msys:

#ifndef MODEL_FILES_H
#define MODEL_FILES_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#if UINTPTR_MAX == 0xFFFFFFFF
#define NE_MEM_ALIGN 4
#else
#define NE_MEM_ALIGN 16
#endif

#include "core/layers/jblas_common.hpp"
#include "core/ne_layers.h"
#include "models/model_utils/util.h"
#include "models/models.h"

template <typename T>
static T checked_mul(T a, T b) {
  T ret = a * b;
  if (a != 0 && ret / a != b) {
    throw format("overflow multiplying %llu * %llu", (unsigned long long)a, (unsigned long long)b);
  }
  return ret;
}

static size_t checked_div(size_t a, size_t b) {
  if (b == 0 || a % b != 0) {
    throw format("error dividing %zu / %zu", a, b);
  }
  return a / b;
}

static std::string model_format_tensor_shape(const std::vector<uint32_t>& ne) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%5u", ne.at(0));
  for (size_t i = 1; i < ne.size(); i++) {
    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " x %5u", ne.at(i));
  }
  return buf;
}

static size_t model_calc_tensor_size(const std::vector<uint32_t>& ne, enum ne_type type) {
  size_t size = ne_type_size(type);
  for (uint32_t dim : ne) {
    size = checked_mul<size_t>(size, dim);
  }
  return size / ne_blck_size(type);
}

struct model_load_tensor_shard {
  std::vector<uint32_t> ne;
  size_t size;
  enum ne_type type;
  size_t file_idx;
  size_t file_off;

  void calc_size() { size = model_calc_tensor_size(ne, type); }
};

enum model_split_type { SPLIT_NONE, SPLIT_BY_COLUMNS, SPLIT_BY_ROWS, TP_1D_ROW, TP_1D_COLUMN, TP_1D_ONLY_MASTER };

struct model_load_tensor {
  std::vector<model_load_tensor_shard> shards;

#ifdef NE_TP_MODEL
  parallel_context* p_ctx = init_parallel_context();
  int32_t world_size = get_tp_size(p_ctx);
  int32_t rank = get_tp_rank(p_ctx);
  bool enable_tp = world_size > 1 ? true : false;

#endif
  std::string name;
  enum ne_type type = NE_TYPE_F32;
  model_split_type split_type = SPLIT_NONE;
  std::vector<uint32_t> ne;
  size_t size;
  struct ne_tensor* ne_tensor = NULL;
  uint8_t* data;

  model_load_tensor(const std::string& name) : name(name) {}

  void calc_all() {
    calc_type();
    calc_split_type();
    calc_ne();
    if (type == NE_TYPE_JBLAS) {
      size = shards[0].size;
    } else {
      calc_size();
    }
  }

  void calc_type() {
    const auto& first_shard = shards.at(0);
    for (const auto& shard : shards) {
      if (shard.type != first_shard.type) {
        throw format("inconsistent tensor shard type in '%s'", name.c_str());
      }
    }
    type = first_shard.type;
  }

  void calc_split_type() {
    if (shards.at(0).ne.size() == 1 ||  // 1D tensors are just duplicated in every file
        shards.size() == 1) {           // only one file?
      split_type = SPLIT_NONE;
    } else if (name.find("tok_embeddings.") == 0 || name.find(".attention.wo.weight") != std::string::npos ||
               name.find(".feed_forward.w2.weight") != std::string::npos) {
      split_type = SPLIT_BY_COLUMNS;
    } else {
      split_type = SPLIT_BY_ROWS;
    }

#ifdef NE_TP_MODEL
    if (enable_tp) {
      // TODO it's not good to check type here, mmaybe move to specific model files
      if (name.find(".attn.q_proj.weight") != std::string::npos ||
          name.find(".attn.k_proj.weight") != std::string::npos ||
          name.find(".attn.v_proj.weight") != std::string::npos ||
          name.find(".mlp.fc_in.weight") != std::string::npos ||
          // for llama model
          name.find(".attention.wq.weight") != std::string::npos ||
          name.find(".attention.wk.weight") != std::string::npos ||
          name.find(".attention.wv.weight") != std::string::npos ||
          name.find(".feed_forward.w1.weight") != std::string::npos ||
          name.find(".feed_forward.w3.weight") != std::string::npos) {
        split_type = TP_1D_ROW;
      }
      if (name.find(".mlp.fc_in.bias") != std::string::npos || name.find(".mlp.fc_out.weight") != std::string::npos ||
          name.find(".attn.out_proj.weight") != std::string::npos ||
          // TODO check if this part should be column
          name.find(".attention.wo.weight") != std::string::npos ||
          name.find(".feed_forward.w2.weight") != std::string::npos) {
        split_type = TP_1D_COLUMN;
      }
      if (name.find(".mlp.fc_out.bias") != std::string::npos) {
        split_type = TP_1D_ONLY_MASTER;
      }
    }
#endif
  }

  void calc_ne() {
    const auto& first_shard = shards.at(0);
    for (const auto& shard : shards) {
      if (shard.ne != first_shard.ne) {
        throw format("inconsistent tensor shard shape in '%s': first was %s, other was %s", name.c_str(),
                     model_format_tensor_shape(first_shard.ne).c_str(), model_format_tensor_shape(shard.ne).c_str());
      }
    }
    ne = first_shard.ne;
    MODEL_ASSERT(shards.size() <= UINT32_MAX);
    uint32_t n_shards = (uint32_t)shards.size();
    switch (split_type) {
      case SPLIT_NONE:
        ne = first_shard.ne;
        break;
      case SPLIT_BY_COLUMNS:
        ne = {checked_mul<uint32_t>(first_shard.ne[0], n_shards), first_shard.ne[1]};
        break;
      case SPLIT_BY_ROWS:
        ne = {first_shard.ne[0], checked_mul<uint32_t>(first_shard.ne[1], n_shards)};
        break;
#ifdef NE_TP_MODEL
      case TP_1D_ROW:
        MODEL_ASSERT(first_shard.ne.size() > 1);
        MODEL_ASSERT(first_shard.ne[1] % world_size == 0);
        ne = {first_shard.ne[0], first_shard.ne[1] / world_size};
        break;
      case TP_1D_COLUMN:
        MODEL_ASSERT(first_shard.ne[0] % world_size == 0);
        if (first_shard.ne.size() == 1) {
          ne = {first_shard.ne[0] / world_size};
        } else {
          ne = {first_shard.ne[0] / world_size, first_shard.ne[1]};
        }
        break;
      case TP_1D_ONLY_MASTER:
        ne = first_shard.ne;
        break;
#endif
    }
  }

  void calc_size() { size = model_calc_tensor_size(ne, type); }
};

struct model_load_tensors_map {
  // tensors is kept in a separate vector to preserve file order
  std::vector<model_load_tensor> tensors;
  std::unordered_map<std::string, size_t> name_to_idx;
};

struct model_file_loader {
  model_file file;
  model_file_version file_version;
  model_hparams hparams;
  model_vocab vocab;

  model_file_loader(const char* fname, size_t file_idx, model_load_tensors_map& tensors_map) : file(fname, "rb") {
    fprintf(stderr, "model.cpp: loading model from %s\n", fname);
    read_magic();
    read_hparams();
    read_vocab();
    read_tensor_metadata(file_idx, tensors_map);
  }
  void read_magic() {
    uint32_t magic = file.read_u32();

    if (magic == MODEL_FILE_MAGIC_NE) {
      file_version = MODEL_FILE_VERSION_NE;
      return;
    }

    uint32_t version = file.read_u32();

    switch (magic) {
      case MODEL_FILE_MAGIC_GGMF:
        switch (version) {
          case 1:
            file_version = MODEL_FILE_VERSION_GGMF_V1;
            return;
        }
        break;
      case MODEL_FILE_MAGIC_GGJT:
        switch (version) {
          case 1:
            file_version = MODEL_FILE_VERSION_GGJT_V1;
            return;
          case 2:
            file_version = MODEL_FILE_VERSION_GGJT_V2;
            return;
          case 3:
            file_version = MODEL_FILE_VERSION_GGJT_V3;
            return;
        }
    }

    throw format("unknown (magic, version) combination: %08x, %08x; is this really a NE file?", magic, version);
  }
  void read_hparams() {
    hparams.n_vocab = file.read_u32();
    hparams.n_embd = file.read_u32();
    hparams.n_mult = file.read_u32();
    hparams.n_head = file.read_u32();
    hparams.n_head_kv = file.read_u32();
    hparams.n_layer = file.read_u32();
    hparams.n_rot = file.read_u32();
    hparams.ftype = (enum ne_ftype)file.read_u32();
    hparams.max_seq_len = file.read_u32();
    file.read_raw(&hparams.alibi_bias_max, sizeof(float));
    file.read_raw(&hparams.clip_qkv, sizeof(float));
    hparams.par_res = file.read_u32();

    hparams.word_embed_proj_dim = file.read_u32();
    hparams.do_layer_norm_before = bool(file.read_u32());

    // For ChatGLM-2
    hparams.multi_query_group_num = file.read_u32();
    hparams.ffn_hidden_size = file.read_u32();

    // For ChatGLM-2
    hparams.inner_hidden_size = file.read_u32();

    file.read_raw(&hparams.rms_norm_eps, sizeof(float));
    file.read_raw(&hparams.freq_base, sizeof(float));
    file.read_raw(&hparams.freq_scale, sizeof(float));
    hparams.rope_scaling_factor = file.read_u32();
    hparams.original_max_position_embeddings = file.read_u32();
    hparams.use_yarn = file.read_u32();
    
  }

  void read_vocab() {
    vocab.id_to_token.resize(hparams.n_vocab);
    file.read_raw(&vocab.bos_token_id, sizeof(model_vocab::id));
    file.read_raw(&vocab.eos_token_id, sizeof(model_vocab::id));
    file.read_raw(&vocab.pad_token_id, sizeof(model_vocab::id));
    file.read_raw(&vocab.sep_token_id, sizeof(model_vocab::id));

    for (uint32_t i = 0; i < hparams.n_vocab; i++) {
      uint32_t len = file.read_u32();
      std::string word = file.read_string(len);

      float score = 0.0f;
      if (file_version >= MODEL_FILE_VERSION_GGMF_V1) {
        file.read_raw(&score, sizeof(score));
      }

      vocab.token_to_id[word] = i;

      auto& tok_score = vocab.id_to_token[i];
      tok_score.tok = std::move(word);
      tok_score.score = score;
    }
  }
  void read_tensor_metadata(size_t file_idx, model_load_tensors_map& tensors_map) {
    while (file.tell() < file.size) {
      model_load_tensor_shard shard;
      uint32_t n_dims = file.read_u32();
      uint32_t name_len = file.read_u32();
      shard.type = (enum ne_type)file.read_u32();
      shard.ne.resize(n_dims);
      file.read_raw(shard.ne.data(), sizeof(shard.ne[0]) * n_dims);
      std::string name = file.read_string(name_len);
      if (n_dims < 1 || n_dims > 2) {
        throw format("model.cpp: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims);
      }
      switch (shard.type) {
        case NE_TYPE_F32:
        case NE_TYPE_F16:
        case NE_TYPE_Q4_0:
        case NE_TYPE_Q4_1:
        case NE_TYPE_Q5_0:
        case NE_TYPE_Q5_1:
        case NE_TYPE_Q8_0:
        case NE_TYPE_JBLAS:
          break;
        default: {
          throw format("unrecognized tensor type %u\n", shard.type);
        }
      }

      if (file_version >= MODEL_FILE_VERSION_GGJT_V1) {
        // skip to the next multiple of 32 bytes
        file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
      }
      shard.file_idx = file_idx;
      shard.file_off = file.tell();
      if (shard.type == NE_TYPE_JBLAS) {
        size_t size = 0;
        file.read_raw(&size, sizeof(size_t));
        shard.size = size;
        file.seek(shard.size - sizeof(size_t), SEEK_CUR);
      } else {
        shard.calc_size();
        file.seek(shard.size, SEEK_CUR);
      }

      auto it = tensors_map.name_to_idx.find(name);
      size_t idx;
      if (it != tensors_map.name_to_idx.end()) {
        idx = it->second;
      } else {
        tensors_map.tensors.emplace_back(name);
        idx = tensors_map.tensors.size() - 1;
        tensors_map.name_to_idx.emplace(name, idx);
      }
      tensors_map.tensors.at(idx).shards.push_back(shard);
    }
  }
};

struct model_file_saver {
  model_file file;
  model_file_loader* any_file_loader;
  model_file_saver(const char* fname, model_file_loader* any_file_loader, enum ne_ftype new_ftype)
      : file(fname, "wb"), any_file_loader(any_file_loader) {
    fprintf(stderr, "model.cpp: saving model to %s\n", fname);
    write_magic();
    write_hparams(new_ftype);
    write_vocab();
  }
  void write_magic() {
    file.write_u32(MODEL_FILE_MAGIC);    // magic
    file.write_u32(MODEL_FILE_VERSION);  // version
  }
  void write_hparams(enum ne_ftype new_ftype) {
    const model_hparams& hparams = any_file_loader->hparams;
    file.write_u32(hparams.n_vocab);
    file.write_u32(hparams.n_embd);
    file.write_u32(hparams.n_mult);
    file.write_u32(hparams.n_head);
    file.write_u32(hparams.n_head_kv);
    file.write_u32(hparams.n_layer);
    file.write_u32(hparams.n_rot);
    file.write_u32(hparams.ftype);
    file.write_u32(hparams.max_seq_len);
    file.write_raw(&hparams.alibi_bias_max, sizeof(float));
    file.write_raw(&hparams.clip_qkv, sizeof(float));
    file.write_u32(hparams.par_res);
    file.write_u32(hparams.word_embed_proj_dim);
    file.write_u32(static_cast<int>(hparams.do_layer_norm_before));

    file.write_u32(hparams.multi_query_group_num);
    file.write_u32(hparams.ffn_hidden_size);
    file.write_u32(hparams.inner_hidden_size);

    file.write_raw(&hparams.rms_norm_eps, sizeof(float));
    file.write_raw(&hparams.freq_base, sizeof(float));
    file.write_raw(&hparams.freq_scale, sizeof(float));
  }
  void write_vocab() {
    if (any_file_loader->file_version == MODEL_FILE_VERSION_NE) {
      fprintf(stderr, "model.cpp: WARNING: input is an old file that doesn't have scores; will add dummy scores\n");
    }
    uint32_t n_vocab = any_file_loader->hparams.n_vocab;
    file.write_raw(&(any_file_loader->vocab.bos_token_id), sizeof(model_vocab::id));
    file.write_raw(&(any_file_loader->vocab.eos_token_id), sizeof(model_vocab::id));
    file.write_raw(&(any_file_loader->vocab.pad_token_id), sizeof(model_vocab::id));
    file.write_raw(&(any_file_loader->vocab.sep_token_id), sizeof(model_vocab::id));
    for (uint32_t i = 0; i < n_vocab; i++) {
      const auto& token_score = any_file_loader->vocab.id_to_token.at(i);
      file.write_u32((uint32_t)token_score.tok.size());
      file.write_raw(token_score.tok.data(), token_score.tok.size());
      file.write_raw(&token_score.score, sizeof(token_score.score));
    }
  }
  void write_tensor(model_load_tensor& tensor, enum ne_type new_type, const void* new_data, size_t new_size) {
    switch (new_type) {
      case NE_TYPE_F32:
      case NE_TYPE_F16:
      case NE_TYPE_Q4_0:
      case NE_TYPE_Q4_1:
      case NE_TYPE_Q5_0:
      case NE_TYPE_Q5_1:
      case NE_TYPE_Q8_0:
      case NE_TYPE_JBLAS:
        break;
      default:
        MODEL_ASSERT(false);
    }
    file.write_u32((uint32_t)tensor.ne.size());
    file.write_u32((uint32_t)tensor.name.size());
    file.write_u32(new_type);
    file.write_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * tensor.ne.size());
    file.write_raw(tensor.name.data(), tensor.name.size());
    file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
    if (new_type != NE_TYPE_JBLAS) MODEL_ASSERT(new_size == model_calc_tensor_size(tensor.ne, new_type));
    file.write_raw(new_data, new_size);
  }
};

struct model_model_loader {
  std::vector<std::unique_ptr<model_file_loader>> file_loaders;
  model_load_tensors_map tensors_map;
  bool use_mmap;
  size_t num_ne_tensors_created = 0;
  struct ne_context* ne_ctx = NULL;
  std::unique_ptr<model_mmap> mapping;

  model_model_loader(const std::string& fname_base, bool use_mmap, bool vocab_only) {
    auto* first_file = new model_file_loader(fname_base.c_str(), 0, tensors_map);
    file_loaders.emplace_back(first_file);
    uint32_t n_parts = vocab_only ? 1 : guess_n_parts();
    for (uint32_t i = 1; i < n_parts; i++) {
      std::string fname = fname_base + "." + std::to_string(i);
      auto* ith_file = new model_file_loader(fname.c_str(), i, tensors_map);
      file_loaders.emplace_back(ith_file);
      if (ith_file->hparams != first_file->hparams) {
        throw format("model.cpp: hparams inconsistent between files");
      }
    }
    if (!model_mmap::SUPPORTED) {
      use_mmap = false;
    }
    if (use_mmap && alignment_prevents_mmap()) {
      fprintf(stderr,
              "model.cpp: can't use mmap because tensors are not aligned; convert to new format to avoid this\n");
      use_mmap = false;
    }
    this->use_mmap = use_mmap;
    for (model_load_tensor& lt : tensors_map.tensors) {
      lt.calc_all();
    }
  }

  bool alignment_prevents_mmap() {
    for (const model_load_tensor& lt : tensors_map.tensors) {
      for (const model_load_tensor_shard& shard : lt.shards) {
        if (shard.file_off & 3) {
          return true;
        }
      }
    }
    return false;
  }

  uint32_t guess_n_parts() const {
    auto it = tensors_map.name_to_idx.find("tok_embeddings.weight");
    if (it == tensors_map.name_to_idx.end()) {
      it = tensors_map.name_to_idx.find("transformer.wte.weight");
      if (it == tensors_map.name_to_idx.end()) {
        it = tensors_map.name_to_idx.find("gpt_neox.embed_in.weight");
        if (it == tensors_map.name_to_idx.end()) {
          it = tensors_map.name_to_idx.find("model/wte");
          if (it == tensors_map.name_to_idx.end()) {
            it = tensors_map.name_to_idx.find("model.embed_tokens.weight");  // baichuan13B
            if (it == tensors_map.name_to_idx.end()) {
              it = tensors_map.name_to_idx.find("transformer.word_embeddings.weight");  // ChatGLM-1
              if (it == tensors_map.name_to_idx.end()) {
                it = tensors_map.name_to_idx.find("transformer.embedding.word_embeddings.weight");  // ChatGLM-2
                if (it == tensors_map.name_to_idx.end()) {
                  it = tensors_map.name_to_idx.find("model.decoder.embed_tokens.weight");
                  if (it != tensors_map.name_to_idx.end()) return 1;  // hacky solution for OPT loading
                  if (it == tensors_map.name_to_idx.end()) {
                    throw std::string("missing tok_embeddings.weight");
                  }
                }
              }
            }
          }
        }
      }
    }
    const model_load_tensor& lt = tensors_map.tensors.at(it->second);
    return file_loaders.at(0)->hparams.n_embd / lt.shards.at(0).ne.at(0);
  }

  void calc_sizes(size_t* ctx_size_p, size_t* mmapped_size_p) const {
    *ctx_size_p = *mmapped_size_p = 0;
    size_t size_needed = 0;
    for (const model_load_tensor& lt : tensors_map.tensors) {
      *ctx_size_p += sizeof(struct ne_tensor) + NE_OBJECT_SIZE;
      if (lt.type == NE_TYPE_JBLAS) {
        size_needed = lt.size;
      } else {
        size_needed = (lt.size + NE_MEM_ALIGN - 1) / NE_MEM_ALIGN * NE_MEM_ALIGN;
      }
      *(use_mmap ? mmapped_size_p : ctx_size_p) += size_needed;
    }
  }

  struct ne_tensor* get_tensor(const std::string& name, const std::vector<uint32_t>& ne, ne_backend backend) {
    auto it = tensors_map.name_to_idx.find(name);
    if (it == tensors_map.name_to_idx.end()) {
      throw format("model.cpp: tensor '%s' is missing from model", name.c_str());
    }
    model_load_tensor& lt = tensors_map.tensors.at(it->second);
#ifdef NE_TP_MODEL
    if (lt.enable_tp && (lt.split_type == TP_1D_ROW || lt.split_type == TP_1D_COLUMN)) {
      // check the split dim
      size_t split_dim_size =
          lt.ne.size() == 1 ? lt.ne.at(0) : (lt.split_type == TP_1D_ROW ? lt.ne.at(1) : lt.ne.at(0));
      size_t origin_dim_size = ne.size() == 1 ? ne.at(0) : (lt.split_type == TP_1D_ROW ? ne.at(1) : ne.at(0));
      MODEL_ASSERT(split_dim_size == origin_dim_size / lt.world_size);
      return get_tensor_for(lt, backend);
    }
#endif
    if (lt.ne != ne) {
      throw format("model.cpp: tensor '%s' has wrong shape; expected %s, got %s", name.c_str(),
                   model_format_tensor_shape(ne).c_str(), model_format_tensor_shape(lt.ne).c_str());
    }

    return get_tensor_for(lt, backend);
  }

  struct ne_tensor* get_tensor_for(model_load_tensor& lt, ne_backend backend) {
    struct ne_tensor* tensor;
    if (lt.ne.size() == 2) {
      if (lt.type == NE_TYPE_JBLAS) {
        tensor = ne_new_tensor_2d(ne_ctx, lt.type, lt.ne.at(0), lt.ne.at(1), lt.size);
      } else {
        tensor = ne_new_tensor_2d(ne_ctx, lt.type, lt.ne.at(0), lt.ne.at(1), NE_SIZE_CALC);
      }
    } else {
      MODEL_ASSERT(lt.ne.size() == 1);
      tensor = ne_new_tensor_1d(ne_ctx, lt.type, lt.ne.at(0), NE_SIZE_CALC);
    }
    ne_set_name(tensor, lt.name.c_str());
    MODEL_ASSERT(lt.ne_tensor == NULL);  // if this fails, we called get_tensor twice on the same tensor
    tensor->backend = backend;
    lt.ne_tensor = tensor;
    num_ne_tensors_created++;
    return tensor;
  }

  void done_getting_tensors() const {
    if (num_ne_tensors_created != tensors_map.tensors.size()) {
      throw std::string("model.cpp: file contained more tensors than expected");
    }
  }

  void load_all_data(model_progress_callback progress_callback, void* progress_callback_user_data,
                     model_mlock* lmlock) {
    size_t data_size = 0;
    size_t prefetch_size = 0;
    for (const model_load_tensor& lt : tensors_map.tensors) {
      data_size += lt.size;
      if (lt.ne_tensor->backend == NE_BACKEND_CPU) {
        prefetch_size += lt.size;
      }
    }

    if (use_mmap) {
      mapping.reset(new model_mmap(&file_loaders.at(0)->file, prefetch_size));
      if (!lmlock) {
        // Don't call the callback since the actual loading will be lazy
        // and we can't measure it.
        progress_callback = NULL;
      }
      if (lmlock) {
        lmlock->init(mapping->addr);
      }
    }

    size_t done_size = 0;
    for (model_load_tensor& lt : tensors_map.tensors) {
      if (lt.ne_tensor->backend != NE_BACKEND_CPU) {
        continue;
      }
      if (progress_callback) {
        progress_callback((float)done_size / data_size, progress_callback_user_data);
      }
      MODEL_ASSERT(lt.ne_tensor);  // unused tensors should have been caught by load_data already
      lt.data = (uint8_t*)lt.ne_tensor->data;
      load_data_for(lt);
      lt.ne_tensor->data = lt.data;
      done_size += lt.size;
      if (use_mmap && lmlock) {
        lmlock->grow_to(done_size);
      }
    }
  }

  void jblas_split_weight(void** src, void** dst, size_t src_n, size_t src_k, size_t dst_n, size_t dst_k, size_t n_rank,
                          size_t k_rank) {
    auto src_fp32 = (float*)malloc(src_n * src_k * sizeof(float));
    if (src_fp32 == nullptr) {
      assert(0);
    }
    jblas_unpackweight_fp32(*src, src_n, src_k, src_fp32, src_n);
    // layout will be K * N in the buffer
    auto dst_fp32 = src_fp32 + k_rank * dst_k * src_n + n_rank * dst_n;
    jblas_packweight_copyattr(dst_fp32, *dst, dst_n, dst_k, src_n, *src);
    free(src_fp32);
  }
  void load_data_for(model_load_tensor& lt) {
    if (use_mmap) {
      MODEL_ASSERT(lt.shards.size() == 1);
      lt.data = (uint8_t*)mapping->addr + lt.shards.at(0).file_off;
    } else if (lt.split_type == SPLIT_NONE) {
      model_file& file = file_loaders.at(lt.shards.at(0).file_idx)->file;
      file.seek(lt.shards.at(0).file_off, SEEK_SET);
      file.read_raw(lt.data, lt.size);
    } else if (lt.split_type == SPLIT_BY_ROWS) {
      size_t offset = 0;
      for (model_load_tensor_shard& shard : lt.shards) {
        model_file& file = file_loaders.at(shard.file_idx)->file;
        file.seek(shard.file_off, SEEK_SET);
        file.read_raw(lt.data + offset, shard.size);
        offset += shard.size;
      }
      MODEL_ASSERT(offset == lt.size);
    } else if (lt.split_type == SPLIT_BY_COLUMNS) {
      // Let's load the data into temporary buffers to ensure the OS performs large loads.
      std::vector<model_buffer> tmp_bufs(lt.shards.size());
      for (size_t i = 0; i < lt.shards.size(); i++) {
        model_load_tensor_shard& shard = lt.shards.at(i);
        model_file& file = file_loaders.at(shard.file_idx)->file;
        file.seek(shard.file_off, SEEK_SET);
        tmp_bufs.at(i).resize(shard.size);
        file.read_raw(tmp_bufs.at(i).addr, shard.size);
      }
      // Then reshape.
      size_t num_rows = lt.ne.at(1);
      size_t per_shard_row_size = lt.shards.at(0).size / num_rows;
      size_t out_offset = 0;
      for (size_t row = 0; row < num_rows; row++) {
        for (model_buffer& tmp_buf : tmp_bufs) {
          memcpy(lt.data + out_offset, tmp_buf.addr + row * per_shard_row_size, per_shard_row_size);
          out_offset += per_shard_row_size;
        }
      }
      MODEL_ASSERT(out_offset == lt.size);
    }
#ifdef NE_TP_MODEL
    else if (lt.split_type == TP_1D_ROW) {
      model_load_tensor_shard& shard = lt.shards.at(0);
      model_buffer tmp_buf;
      model_file& file = file_loaders.at(shard.file_idx)->file;
      file.seek(shard.file_off, SEEK_SET);
      size_t num_rows = lt.ne.size() == 1 ? 1 : lt.ne.at(1);
      if (lt.type == NE_TYPE_JBLAS) {
        tmp_buf.resize(shard.size);
        file.read_raw(tmp_buf.addr, shard.size);
        void* dst_data = (void*)lt.data;
        void* src_data = (void*)(tmp_buf.addr);
        jblas_split_weight(&src_data, &dst_data, lt.world_size * num_rows, lt.ne.at(0), num_rows, lt.ne.at(0), lt.rank,
                           0);
      } else {
        // only copy part of weight form the tmp_buf of origin file
        tmp_buf.resize(lt.size * lt.world_size);
        file.read_raw(tmp_buf.addr, lt.size * lt.world_size);
        memcpy(lt.data, tmp_buf.addr + lt.rank * lt.size, lt.size);
      }
    } else if (lt.split_type == TP_1D_COLUMN) {
      if (lt.size == 0) {
        return;
      }
      model_load_tensor_shard& shard = lt.shards.at(0);
      model_buffer tmp_buf;
      model_file& file = file_loaders.at(shard.file_idx)->file;
      file.seek(shard.file_off, SEEK_SET);
      size_t num_rows = lt.ne.size() == 1 ? 1 : lt.ne.at(1);
      if (lt.type == NE_TYPE_JBLAS) {
        tmp_buf.resize(shard.size);
        file.read_raw(tmp_buf.addr, shard.size);
        void* dst_data = (void*)lt.data;
        void* src_data = (void*)(tmp_buf.addr);
        jblas_split_weight(&src_data, &dst_data, num_rows, lt.world_size * lt.ne.at(0), num_rows, lt.ne.at(0), 0,
                           lt.rank);
      } else {
        tmp_buf.resize(lt.size * lt.world_size);
        file.read_raw(tmp_buf.addr, lt.size * lt.world_size);
        size_t offset = 0;
        // different data type may have differnet per_row_size
        size_t per_row_size = lt.size / num_rows;
        for (size_t i = 0; i < num_rows; ++i) {
          memcpy(lt.data + offset, tmp_buf.addr + lt.rank * per_row_size + i * lt.world_size * per_row_size,
                 per_row_size);
          offset += per_row_size;
        }
        MODEL_ASSERT(offset == lt.size);
      }
    } else if (lt.split_type == TP_1D_ONLY_MASTER) {
      // only master node load the tensor, other node set to zero
      model_file& file = file_loaders.at(lt.shards.at(0).file_idx)->file;
      file.seek(lt.shards.at(0).file_off, SEEK_SET);
      if (lt.rank == 0) {
        file.read_raw(lt.data, lt.size);
      } else {
        memset(lt.data, 0, lt.size);
      }
    }
#endif
    if (0) {
      print_checksum(lt);
    }
  }

  static void print_checksum(model_load_tensor& lt) {
    uint32_t sum = 0;
    for (size_t i = 0; i < lt.size; i++) {
      uint8_t byte = lt.data[i];
      sum = byte + (sum << 6) + (sum << 16) - sum;  // sdbm hash
    }
    fprintf(stderr, "%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
            model_format_tensor_shape(lt.ne).c_str(), lt.size);
  }
};

#endif  // MODEL_FILES_H
