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
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#include "models/util.h"
#include "llama_model.h"

#include "core/ne_layers.h"
#include "jblas/jblas/jit_blas_weight_compression.h"

#include <array>
#include <ctime>
#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cstring>
#include <climits>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <numeric>

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

enum model_split_type { SPLIT_NONE, SPLIT_BY_COLUMNS, SPLIT_BY_ROWS };

struct model_load_tensor {
  std::vector<model_load_tensor_shard> shards;

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
    if (type == NE_TYPE_Q4_JBLAS) {
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
    hparams.n_layer = file.read_u32();
    hparams.n_rot = file.read_u32();
    hparams.ftype = (enum ne_ftype)file.read_u32();
  }
  void read_vocab() {
    vocab.id_to_token.resize(hparams.n_vocab);

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
        case NE_TYPE_Q4_JBLAS:
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
      if (shard.type == NE_TYPE_Q4_JBLAS) {
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
    file.write_u32(hparams.n_layer);
    file.write_u32(hparams.n_rot);
    file.write_u32(new_ftype);
  }
  void write_vocab() {
    if (any_file_loader->file_version == MODEL_FILE_VERSION_NE) {
      fprintf(stderr, "model.cpp: WARNING: input is an old file that doesn't have scores; will add dummy scores\n");
    }
    uint32_t n_vocab = any_file_loader->hparams.n_vocab;
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
      case NE_TYPE_Q4_JBLAS:
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
    if (new_type != NE_TYPE_Q4_JBLAS) MODEL_ASSERT(new_size == model_calc_tensor_size(tensor.ne, new_type));
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
      throw std::string("missing tok_embeddings.weight");
    }
    const model_load_tensor& lt = tensors_map.tensors.at(it->second);
    return file_loaders.at(0)->hparams.n_embd / lt.shards.at(0).ne.at(0);
  }

  void calc_sizes(size_t* ctx_size_p, size_t* mmapped_size_p) const {
    *ctx_size_p = *mmapped_size_p = 0;
    for (const model_load_tensor& lt : tensors_map.tensors) {
      *ctx_size_p += sizeof(struct ne_tensor) + NE_OBJECT_SIZE;
      *(use_mmap ? mmapped_size_p : ctx_size_p) += lt.size;
    }
  }

  struct ne_tensor* get_tensor(const std::string& name, const std::vector<uint32_t>& ne, ne_backend backend) {
    auto it = tensors_map.name_to_idx.find(name);
    if (it == tensors_map.name_to_idx.end()) {
      throw format("model.cpp: tensor '%s' is missing from model", name.c_str());
    }
    model_load_tensor& lt = tensors_map.tensors.at(it->second);
    if (lt.ne != ne) {
      throw format("model.cpp: tensor '%s' has wrong shape; expected %s, got %s", name.c_str(),
                   model_format_tensor_shape(ne).c_str(), model_format_tensor_shape(lt.ne).c_str());
    }

    return get_tensor_for(lt, backend);
  }

  struct ne_tensor* get_tensor_for(model_load_tensor& lt, ne_backend backend) {
    struct ne_tensor* tensor;
    if (lt.ne.size() == 2) {
      if (lt.type == NE_TYPE_Q4_JBLAS) {
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

//
// kv cache
//

static bool kv_cache_init(const struct model_hparams& hparams, struct model_kv_cache& cache, ne_type wtype, int n_ctx) {
  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;

  const int64_t n_mem = n_layer * n_ctx;
  const int64_t n_elements = n_embd * n_mem;

  cache.buf.resize(2u * n_elements * ne_type_size(wtype) + 2u * MB);

  struct ne_init_params params;
  params.mem_size = cache.buf.size;
  params.mem_buffer = cache.buf.addr;
  params.no_alloc = false;

  cache.ctx = ne_init(params);

  if (!cache.ctx) {
    fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
    return false;
  }

  cache.k = ne_new_tensor_1d(cache.ctx, wtype, n_elements, NE_SIZE_CALC);
  cache.v = ne_new_tensor_1d(cache.ctx, wtype, n_elements, NE_SIZE_CALC);
  ne_set_name(cache.k, "cache_k");
  ne_set_name(cache.v, "cache_v");

  return true;
}

struct model_context_params model_context_default_params() {
  struct model_context_params result = {
      /*.n_ctx                       =*/512,
      /*.gpu_layers                  =*/0,
      /*.seed                        =*/-1,
      /*.f16_kv                      =*/true,
      /*.logits_all                  =*/false,
      /*.vocab_only                  =*/false,
      /*.use_mmap                    =*/true,
      /*.use_mlock                   =*/false,
      /*.embedding                   =*/false,
      /*.progress_callback           =*/nullptr,
      /*.progress_callback_user_data =*/nullptr,
  };

  return result;
}

bool model_mmap_supported() { return model_mmap::SUPPORTED; }

bool model_mlock_supported() { return model_mlock::SUPPORTED; }

void model_init_backend() {
  ne_time_init();

  // needed to initialize f16 tables
  {
    struct ne_init_params params = {0, NULL, false};
    struct ne_context* ctx = ne_init(params);
    ne_free(ctx);
  }
}

int64_t model_time_us() { return ne_time_us(); }

//
// model loading
//

static const char* model_file_version_name(model_file_version version) {
  switch (version) {
    case MODEL_FILE_VERSION_NE:
      return "'ne' (old version with low tokenizer quality and no mmap support)";
    case MODEL_FILE_VERSION_GGMF_V1:
      return "ggmf v1 (old version with no mmap support)";
    case MODEL_FILE_VERSION_GGJT_V1:
      return "ggjt v1 (pre #1405)";
    case MODEL_FILE_VERSION_GGJT_V2:
      return "ggjt v2 (pre #1508)";
    case MODEL_FILE_VERSION_GGJT_V3:
      return "ggjt v3 (latest)";
  }

  return "unknown";
}

static const char* ne_ftype_name(enum ne_ftype ftype) {
  switch (ftype) {
    case NE_FTYPE_ALL_F32:
      return "all F32";
    case NE_FTYPE_MOSTLY_F16:
      return "mostly F16";
    case NE_FTYPE_MOSTLY_Q4_0:
      return "mostly Q4_0";
    case NE_FTYPE_MOSTLY_Q4_1:
      return "mostly Q4_1";
    case NE_FTYPE_MOSTLY_Q4_1_SOME_F16:
      return "mostly Q4_1, some F16";
    case NE_FTYPE_MOSTLY_Q5_0:
      return "mostly Q5_0";
    case NE_FTYPE_MOSTLY_Q5_1:
      return "mostly Q5_1";
    case NE_FTYPE_MOSTLY_Q8_0:
      return "mostly Q8_0";
    default:
      return "unknown, may not work";
  }
}

static const char* model_model_type_name(e_model type) {
  switch (type) {
    case MODEL_7B:
      return "7B";
    case MODEL_13B:
      return "13B";
    case MODEL_30B:
      return "30B";
    case MODEL_65B:
      return "65B";
    default:
      MODEL_ASSERT(false);
  }
}

static void model_model_load_internal(const std::string& fname, model_context& lctx, int n_ctx, int n_gpu_layers,
                                      ne_type memory_type, bool use_mmap, bool use_mlock, bool vocab_only,
                                      model_progress_callback progress_callback, void* progress_callback_user_data) {
  lctx.t_start_us = ne_time_us();

  std::unique_ptr<model_model_loader> ml(new model_model_loader(fname, use_mmap, vocab_only));

  lctx.vocab = std::move(ml->file_loaders.at(0)->vocab);
  auto& model = lctx.model;
  model.hparams = ml->file_loaders.at(0)->hparams;
  model_file_version file_version = ml->file_loaders.at(0)->file_version;
  auto& hparams = model.hparams;
  uint32_t n_ff = ((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult) * hparams.n_mult;

  {
    switch (hparams.n_layer) {
      case 32:
        model.type = e_model::MODEL_7B;
        break;
      case 40:
        model.type = e_model::MODEL_13B;
        break;
      case 60:
        model.type = e_model::MODEL_30B;
        break;
      case 80:
        model.type = e_model::MODEL_65B;
        break;
    }

    hparams.n_ctx = n_ctx;
  }

  {
    fprintf(stderr, "%s: format     = %s\n", __func__, model_file_version_name(file_version));
    fprintf(stderr, "%s: n_vocab    = %u\n", __func__, hparams.n_vocab);
    fprintf(stderr, "%s: n_ctx      = %u\n", __func__, hparams.n_ctx);
    fprintf(stderr, "%s: n_embd     = %u\n", __func__, hparams.n_embd);
    fprintf(stderr, "%s: n_mult     = %u\n", __func__, hparams.n_mult);
    fprintf(stderr, "%s: n_head     = %u\n", __func__, hparams.n_head);
    fprintf(stderr, "%s: n_layer    = %u\n", __func__, hparams.n_layer);
    fprintf(stderr, "%s: n_rot      = %u\n", __func__, hparams.n_rot);
    fprintf(stderr, "%s: ftype      = %u (%s)\n", __func__, hparams.ftype, ne_ftype_name(hparams.ftype));
    fprintf(stderr, "%s: n_ff       = %u\n", __func__, n_ff);
    fprintf(stderr, "%s: n_parts    = %zu\n", __func__, ml->file_loaders.size());
    fprintf(stderr, "%s: model size = %s\n", __func__, model_model_type_name(model.type));
  }

  if (file_version < MODEL_FILE_VERSION_GGJT_V2) {
    if (hparams.ftype != NE_FTYPE_ALL_F32 && hparams.ftype != NE_FTYPE_MOSTLY_F16 &&
        hparams.ftype != NE_FTYPE_MOSTLY_Q8_0) {
      throw format("this format is no longer supported (see https://github.com/ggerganov/model.cpp/pull/1405)");
    }
  }

  if (file_version < MODEL_FILE_VERSION_GGJT_V3) {
    if (hparams.ftype == NE_FTYPE_MOSTLY_Q4_0 || hparams.ftype == NE_FTYPE_MOSTLY_Q4_1 ||
        hparams.ftype == NE_FTYPE_MOSTLY_Q8_0) {
      throw format("this format is no longer supported (see https://github.com/ggerganov/model.cpp/pull/1508)");
    }
  }

  if (vocab_only) {
    return;
  }

  auto& ctx = model.ctx;

  size_t ctx_size;
  size_t mmapped_size;
  ml->calc_sizes(&ctx_size, &mmapped_size);
  fprintf(stderr, "%s: ne ctx size = %7.2f MB\n", __func__, ctx_size / 1024.0 / 1024.0);

  // create the ne context
  {
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
  }

#ifdef NE_USE_CUBLAS
#define MODEL_BACKEND_OFFLOAD NE_BACKEND_CUDA
#else
#define MODEL_BACKEND_OFFLOAD NE_BACKEND_CPU
#endif

  // prepare memory for the weights
  size_t vram_total = 0;
  {
    const uint32_t n_embd = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;

    ml->ne_ctx = ctx;

    model.tok_embeddings = ml->get_tensor("tok_embeddings.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);
    model.norm = ml->get_tensor("norm.weight", {n_embd}, NE_BACKEND_CPU);

    // "output" tensor
    {
      ne_backend backend_output;
      if (n_gpu_layers > int(n_layer)) {  // NOLINT
        backend_output = MODEL_BACKEND_OFFLOAD;
      } else {
        backend_output = NE_BACKEND_CPU;
      }

      model.output = ml->get_tensor("output.weight", {n_embd, n_vocab}, backend_output);
    }

    const int i_gpu_start = n_layer - n_gpu_layers;

    model.layers.resize(n_layer);
    for (uint32_t i = 0; i < n_layer; ++i) {
      const ne_backend backend = int(i) < i_gpu_start ? NE_BACKEND_CPU : MODEL_BACKEND_OFFLOAD;

      auto& layer = model.layers[i];

      std::string layers_i = "layers." + std::to_string(i);

      layer.attention_norm = ml->get_tensor(layers_i + ".attention_norm.weight", {n_embd}, backend);

      layer.wq = ml->get_tensor(layers_i + ".attention.wq.weight", {n_embd, n_embd}, backend);
      layer.wk = ml->get_tensor(layers_i + ".attention.wk.weight", {n_embd, n_embd}, backend);
      layer.wv = ml->get_tensor(layers_i + ".attention.wv.weight", {n_embd, n_embd}, backend);
      layer.wo = ml->get_tensor(layers_i + ".attention.wo.weight", {n_embd, n_embd}, backend);

      layer.ffn_norm = ml->get_tensor(layers_i + ".ffn_norm.weight", {n_embd}, backend);

      layer.w1 = ml->get_tensor(layers_i + ".feed_forward.w1.weight", {n_embd, n_ff}, backend);
      layer.w2 = ml->get_tensor(layers_i + ".feed_forward.w2.weight", {n_ff, n_embd}, backend);
      layer.w3 = ml->get_tensor(layers_i + ".feed_forward.w3.weight", {n_embd, n_ff}, backend);

      if (backend == NE_BACKEND_CUDA) {
        vram_total += ne_nbytes(layer.attention_norm) + ne_nbytes(layer.wq) + ne_nbytes(layer.wk) +
                      ne_nbytes(layer.wv) + ne_nbytes(layer.wo) + ne_nbytes(layer.attention_norm) +
                      ne_nbytes(layer.w1) + ne_nbytes(layer.w2) + ne_nbytes(layer.w3);
      }
    }
  }

  ml->done_getting_tensors();

  // print memory requirements
  {
    const size_t scale = memory_type == NE_TYPE_F32 ? 2 : 1;

    // this is the total memory required to run the inference
    const size_t mem_required = ctx_size + mmapped_size - vram_total +  // weights in VRAM not in memory
                                MEM_REQ_SCRATCH0().at(model.type) + MEM_REQ_SCRATCH1().at(model.type) +
                                MEM_REQ_EVAL().at(model.type);

    // this is the memory required by one model_state
    const size_t mem_required_state = scale * MEM_REQ_KV_SELF().at(model.type);

    fprintf(stderr, "%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__, mem_required / 1024.0 / 1024.0,
            mem_required_state / 1024.0 / 1024.0);

    (void)n_gpu_layers;
  }

  // populate `tensors_by_name`
  for (model_load_tensor& lt : ml->tensors_map.tensors) {
    model.tensors_by_name.emplace_back(lt.name, lt.ne_tensor);
  }

  ml->load_all_data(progress_callback, progress_callback_user_data, use_mlock ? &lctx.model.mlock_mmap : NULL);

  if (progress_callback) {
    progress_callback(1.0f, progress_callback_user_data);
  }

  model.mapping = std::move(ml->mapping);

  // loading time will be recalculate after the first eval, so
  // we take page faults deferred by mmap() into consideration
  lctx.t_load_us = ne_time_us() - lctx.t_start_us;
}

static bool model_model_load(const std::string& fname, model_context& lctx, int n_ctx, int n_gpu_layers,
                             ne_type memory_type, bool use_mmap, bool use_mlock, bool vocab_only,
                             model_progress_callback progress_callback, void* progress_callback_user_data) {
  try {
    model_model_load_internal(fname, lctx, n_ctx, n_gpu_layers, memory_type, use_mmap, use_mlock, vocab_only,
                              progress_callback, progress_callback_user_data);
    return true;
  } catch (const std::string& err) {
    fprintf(stderr, "error loading model: %s\n", err.c_str());
    return false;
  }
}

//
// tokenizer
//

static size_t utf8_len(char src) {
  const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t highbits = static_cast<uint8_t>(src) >> 4;
  return lookup[highbits];
}

struct model_sp_symbol {
  using index = int;
  index prev;
  index next;
  const char* text;
  size_t n;
};

static_assert(std::is_trivially_copyable<model_sp_symbol>::value, "model_sp_symbol is not trivially copyable");

struct model_sp_bigram {
  struct comparator {
    bool operator()(model_sp_bigram& l, model_sp_bigram& r) {
      return (l.score < r.score) || (l.score == r.score && l.left > r.left);
    }
  };
  using queue_storage = std::vector<model_sp_bigram>;
  using queue = std::priority_queue<model_sp_bigram, queue_storage, comparator>;
  model_sp_symbol::index left;
  model_sp_symbol::index right;
  float score;
  size_t size;
};

// original implementation:
// https://github.com/ggerganov/model.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct model_tokenizer {
  model_tokenizer(const model_vocab& vocab) : vocab_(vocab) {}

  void tokenize(const std::string& text, std::vector<model_vocab::id>& output) {
    // split string into utf8 chars
    int index = 0;
    size_t offs = 0;
    while (offs < text.size()) {
      model_sp_symbol sym;
      size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
      sym.text = text.c_str() + offs;
      sym.n = char_len;
      offs += char_len;
      sym.prev = index - 1;
      sym.next = offs == text.size() ? -1 : index + 1;
      index++;
      symbols_.emplace_back(sym);
    }

    // seed the work queue with all possible 2-character tokens.
    for (size_t i = 1; i < symbols_.size(); ++i) {
      try_add_bigram(i - 1, i);
    }

    // keep substituting the highest frequency pairs for as long as we can.
    while (!work_queue_.empty()) {
      auto bigram = work_queue_.top();
      work_queue_.pop();

      auto& left_sym = symbols_[bigram.left];
      auto& right_sym = symbols_[bigram.right];

      // if one of the symbols already got merged, skip it.
      if (left_sym.n == 0 || right_sym.n == 0 || left_sym.n + right_sym.n != bigram.size) {
        continue;
      }

      // merge the right sym into the left one
      left_sym.n += right_sym.n;
      right_sym.n = 0;

      // printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

      // remove the right sym from the chain
      left_sym.next = right_sym.next;
      if (right_sym.next >= 0) {
        symbols_[right_sym.next].prev = bigram.left;
      }

      // find more substitutions
      try_add_bigram(left_sym.prev, bigram.left);
      try_add_bigram(bigram.left, left_sym.next);
    }

    for (int i = 0; i != -1; i = symbols_[i].next) {
      auto& symbol = symbols_[i];
      auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

      if (token == vocab_.token_to_id.end()) {
        // output any symbols that did not form tokens as bytes.
        for (int j = 0; j < (int)symbol.n; ++j) {
          model_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
          output.push_back(token_id);
        }
      } else {
        output.push_back((*token).second);
      }
    }
  }

 private:
  void try_add_bigram(int left, int right) {
    if (left == -1 || right == -1) {
      return;
    }

    const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
    auto token = vocab_.token_to_id.find(text);

    if (token == vocab_.token_to_id.end()) {
      return;
    }

    if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
      return;
    }

    const auto& tok_score = vocab_.id_to_token[(*token).second];

    model_sp_bigram bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.score = tok_score.score;
    bigram.size = text.size();
    work_queue_.push(bigram);
  }

  const model_vocab& vocab_;
  std::vector<model_sp_symbol> symbols_;
  model_sp_bigram::queue work_queue_;
};

static std::vector<model_vocab::id> model_tokenize(const model_vocab& vocab, const std::string& text, bool bos) {
  model_tokenizer tokenizer(vocab);
  std::vector<model_vocab::id> output;

  if (text.empty()) {
    return output;
  }

  if (bos) {
    output.push_back(model_token_bos());
  }

  tokenizer.tokenize(text, output);
  return output;
}

//
// sampling
//

void model_sample_softmax(struct model_context* ctx, model_token_data_array* candidates) {
  assert(candidates->size > 0);

  const int64_t t_start_sample_us = ne_time_us();

  // Sort the logits in descending order
  if (!candidates->sorted) {
    std::sort(candidates->data, candidates->data + candidates->size,
              [](const model_token_data& a, const model_token_data& b) { return a.logit > b.logit; });
    candidates->sorted = true;
  }

  float max_l = candidates->data[0].logit;
  float cum_sum = 0.0f;
  for (size_t i = 0; i < candidates->size; ++i) {
    float p = expf(candidates->data[i].logit - max_l);
    candidates->data[i].p = p;
    cum_sum += p;
  }
  for (size_t i = 0; i < candidates->size; ++i) {
    candidates->data[i].p /= cum_sum;
  }

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_top_k(struct model_context* ctx, model_token_data_array* candidates, int k, size_t min_keep) {
  const int64_t t_start_sample_us = ne_time_us();

  k = std::max(k, (int)min_keep);
  k = std::min(k, (int)candidates->size);

  // Sort scores in descending order
  if (!candidates->sorted) {
    auto comp = [](const model_token_data& a, const model_token_data& b) { return a.logit > b.logit; };
    if (k == (int)candidates->size) {
      std::sort(candidates->data, candidates->data + candidates->size, comp);
    } else {
      std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
    }
    candidates->sorted = true;
  }
  candidates->size = k;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_top_p(struct model_context* ctx, model_token_data_array* candidates, float p, size_t min_keep) {
  if (p >= 1.0f) {
    return;
  }

  const int64_t t_start_sample_us = ne_time_us();

  model_sample_softmax(ctx, candidates);

  // Compute the cumulative probabilities
  float cum_sum = 0.0f;
  size_t last_idx = candidates->size;

  for (size_t i = 0; i < candidates->size; ++i) {
    cum_sum += candidates->data[i].p;

    // Check if the running sum is greater than p or if we have kept at least min_keep tokens
    if (cum_sum > p && i >= min_keep) {
      last_idx = i;
      break;
    }
  }

  // Resize the output vector to keep only the top-p tokens
  candidates->size = last_idx;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_tail_free(struct model_context* ctx, model_token_data_array* candidates, float z, size_t min_keep) {
  if (z >= 1.0f || candidates->size <= 2) {
    return;
  }

  const int64_t t_start_sample_us = ne_time_us();

  model_sample_softmax(nullptr, candidates);

  // Compute the first and second derivatives
  std::vector<float> first_derivatives(candidates->size - 1);
  std::vector<float> second_derivatives(candidates->size - 2);

  for (size_t i = 0; i < first_derivatives.size(); ++i) {
    first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
  }
  for (size_t i = 0; i < second_derivatives.size(); ++i) {
    second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
  }

  // Calculate absolute value of second derivatives
  for (size_t i = 0; i < second_derivatives.size(); ++i) {
    second_derivatives[i] = abs(second_derivatives[i]);
  }

  // Normalize the second derivatives
  float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);
  for (float& value : second_derivatives) {
    value /= second_derivatives_sum;
  }

  float cum_sum = 0.0f;
  size_t last_idx = candidates->size;
  for (size_t i = 0; i < second_derivatives.size(); ++i) {
    cum_sum += second_derivatives[i];

    // Check if the running sum is greater than z or if we have kept at least min_keep tokens
    if (cum_sum > z && i >= min_keep) {
      last_idx = i;
      break;
    }
  }

  // Resize the output vector to keep only the tokens above the tail location
  candidates->size = last_idx;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_typical(struct model_context* ctx, model_token_data_array* candidates, float p, size_t min_keep) {
  // Reference implementation:
  // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
  if (p >= 1.0f) {
    return;
  }

  const int64_t t_start_sample_us = ne_time_us();

  // Compute the softmax of logits and calculate entropy
  model_sample_softmax(nullptr, candidates);

  float entropy = 0.0f;
  for (size_t i = 0; i < candidates->size; ++i) {
    entropy += -candidates->data[i].p * logf(candidates->data[i].p);
  }

  // Compute the absolute difference between negative log probability and entropy for each candidate
  std::vector<float> shifted_scores;
  for (size_t i = 0; i < candidates->size; ++i) {
    float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
    shifted_scores.push_back(shifted_score);
  }

  // Sort tokens based on the shifted_scores and their corresponding indices
  std::vector<size_t> indices(candidates->size);
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) { return shifted_scores[a] < shifted_scores[b]; });

  // Compute the cumulative probabilities
  float cum_sum = 0.0f;
  size_t last_idx = indices.size();

  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];
    cum_sum += candidates->data[idx].p;

    // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
    if (cum_sum > p && i >= min_keep - 1) {
      last_idx = i + 1;
      break;
    }
  }

  // Resize the output vector to keep only the locally typical tokens
  std::vector<model_token_data> new_candidates;
  for (size_t i = 0; i < last_idx; ++i) {
    size_t idx = indices[i];
    new_candidates.push_back(candidates->data[idx]);
  }

  // Replace the data in candidates with the new_candidates data
  std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
  candidates->size = new_candidates.size();

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_temperature(struct model_context* ctx, model_token_data_array* candidates_p, float temp) {
  const int64_t t_start_sample_us = ne_time_us();

  for (size_t i = 0; i < candidates_p->size; ++i) {
    candidates_p->data[i].logit /= temp;
  }

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_repetition_penalty(struct model_context* ctx, model_token_data_array* candidates,
                                     const model_token* last_tokens, size_t last_tokens_size, float penalty) {
  if (last_tokens_size == 0 || penalty == 1.0f) {
    return;
  }

  const int64_t t_start_sample_us = ne_time_us();

  for (size_t i = 0; i < candidates->size; ++i) {
    const auto* token_iter = std::find(last_tokens, last_tokens + last_tokens_size, candidates->data[i].id);
    if (token_iter == last_tokens + last_tokens_size) {
      continue;
    }

    // The academic publication that described this technique actually just only divided, but that would cause tokens
    // with negative logits to become more likely, which is obviously wrong. This is common fix for this problem, which
    // is to multiply by the penalty instead of dividing.
    if (candidates->data[i].logit <= 0) {
      candidates->data[i].logit *= penalty;
    } else {
      candidates->data[i].logit /= penalty;
    }
  }

  candidates->sorted = false;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

void model_sample_frequency_and_presence_penalties(struct model_context* ctx, model_token_data_array* candidates,
                                                   const model_token* last_tokens_p, size_t last_tokens_size,
                                                   float alpha_frequency, float alpha_presence) {
  if (last_tokens_size == 0 || (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
    return;
  }

  const int64_t t_start_sample_us = ne_time_us();

  // Create a frequency map to count occurrences of each token in last_tokens
  std::unordered_map<model_token, int> token_count;
  for (size_t i = 0; i < last_tokens_size; ++i) {
    token_count[last_tokens_p[i]]++;
  }

  // Apply frequency and presence penalties to the candidates
  for (size_t i = 0; i < candidates->size; ++i) {
    auto token_iter = token_count.find(candidates->data[i].id);
    if (token_iter == token_count.end()) {
      continue;
    }

    int count = token_iter->second;
    candidates->data[i].logit -= float(count) * alpha_frequency + float(count > 0) * alpha_presence;
  }

  candidates->sorted = false;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
}

model_token model_sample_token_mirostat(struct model_context* ctx, model_token_data_array* candidates, float tau,
                                        float eta, int m, float* mu) {
  assert(ctx);
  auto N = float(model_n_vocab(ctx));
  int64_t t_start_sample_us;
  t_start_sample_us = ne_time_us();

  model_sample_softmax(nullptr, candidates);

  // Estimate s_hat using the most probable m tokens
  float s_hat = 0.0;
  float sum_ti_bi = 0.0;
  float sum_ti_sq = 0.0;
  for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
    float t_i = logf(float(i + 2) / float(i + 1));
    float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
    sum_ti_bi += t_i * b_i;
    sum_ti_sq += t_i * t_i;
  }
  s_hat = sum_ti_bi / sum_ti_sq;

  // Compute k from the estimated s_hat and target surprise value
  float epsilon_hat = s_hat - 1;
  float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

  // Sample the next word X using top-k sampling
  model_sample_top_k(nullptr, candidates, int(k), 1);
  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
  model_token X = model_sample_token(ctx, candidates);
  t_start_sample_us = ne_time_us();

  // Compute error as the difference between observed surprise and target surprise value
  size_t X_idx = std::distance(candidates->data,
                               std::find_if(candidates->data, candidates->data + candidates->size,
                                            [&](const model_token_data& candidate) { return candidate.id == X; }));
  float observed_surprise = -log2f(candidates->data[X_idx].p);
  float e = observed_surprise - tau;

  // Update mu using the learning rate and error
  *mu = *mu - eta * e;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
    ctx->n_sample++;
  }
  return X;
}

model_token model_sample_token_mirostat_v2(struct model_context* ctx, model_token_data_array* candidates, float tau,
                                           float eta, float* mu) {
  assert(ctx);
  int64_t t_start_sample_us;
  t_start_sample_us = ne_time_us();

  model_sample_softmax(ctx, candidates);

  // Truncate the words with surprise values greater than mu
  candidates->size = std::distance(
      candidates->data, std::find_if(candidates->data, candidates->data + candidates->size,
                                     [&](const model_token_data& candidate) { return -log2f(candidate.p) > *mu; }));

  // Normalize the probabilities of the remaining words
  model_sample_softmax(ctx, candidates);

  // Sample the next word X from the remaining words
  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
  model_token X = model_sample_token(ctx, candidates);
  t_start_sample_us = ne_time_us();

  // Compute error as the difference between observed surprise and target surprise value
  size_t X_idx = std::distance(candidates->data,
                               std::find_if(candidates->data, candidates->data + candidates->size,
                                            [&](const model_token_data& candidate) { return candidate.id == X; }));
  float observed_surprise = -log2f(candidates->data[X_idx].p);
  float e = observed_surprise - tau;

  // Update mu using the learning rate and error
  *mu = *mu - eta * e;

  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  }
  return X;
}

model_token model_sample_token_greedy(struct model_context* ctx, model_token_data_array* candidates) {
  const int64_t t_start_sample_us = ne_time_us();

  // Find max element
  auto* max_iter =
      std::max_element(candidates->data, candidates->data + candidates->size,
                       [](const model_token_data& a, const model_token_data& b) { return a.logit < b.logit; });

  model_token result = max_iter->id;
  if (ctx) {
    ctx->t_sample_us += ne_time_us() - t_start_sample_us;
    ctx->n_sample++;
  }
  return result;
}

model_token model_sample_token(struct model_context* ctx, model_token_data_array* candidates) {
  assert(ctx);
  const int64_t t_start_sample_us = ne_time_us();
  model_sample_softmax(nullptr, candidates);

  std::vector<float> probs;
  probs.reserve(candidates->size);
  for (size_t i = 0; i < candidates->size; ++i) {
    probs.push_back(candidates->data[i].p);
  }

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  auto& rng = ctx->rng;
  int idx = dist(rng);

  model_token result = candidates->data[idx].id;

  ctx->t_sample_us += ne_time_us() - t_start_sample_us;
  ctx->n_sample++;
  return result;
}

//
// quantization
//

static void model_model_quantize_internal(const std::string& fname_inp, const std::string& fname_out,
                                          enum ne_ftype ftype, int nthread) {
  ne_type quantized_type;
  switch (ftype) {
    case NE_FTYPE_MOSTLY_Q4_0:
      quantized_type = NE_TYPE_Q4_0;
      break;
    case NE_FTYPE_MOSTLY_Q4_1:
      quantized_type = NE_TYPE_Q4_1;
      break;
    case NE_FTYPE_MOSTLY_Q5_0:
      quantized_type = NE_TYPE_Q5_0;
      break;
    case NE_FTYPE_MOSTLY_Q5_1:
      quantized_type = NE_TYPE_Q5_1;
      break;
    case NE_FTYPE_MOSTLY_Q8_0:
      quantized_type = NE_TYPE_Q8_0;
      break;
    case NE_FTYPE_MOSTLY_Q4_JBLAS_B32:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_B128:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_B1024:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_BF16_B32:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B32:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_BF16_B32:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B128:
      quantized_type = NE_TYPE_Q4_JBLAS;
      break;
    default:
      throw format("invalid output file type %d\n", ftype);
  };

  if (nthread <= 0) {
    nthread = std::thread::hardware_concurrency();
  }

  std::unique_ptr<model_model_loader> model_loader(new model_model_loader(fname_inp, /*use_mmap*/ false,
                                                                          /*vocab_only*/ false));
  model_file_saver file_saver(fname_out.c_str(), model_loader->file_loaders.at(0).get(), ftype);

  size_t total_size_org = 0;
  size_t total_size_new = 0;
  std::vector<int64_t> hist_all(1 << 4, 0);

  std::vector<std::thread> workers;
  std::mutex mutex;

  size_t idx = 0;
  for (model_load_tensor& tensor : model_loader->tensors_map.tensors) {
    model_buffer read_data;
    read_data.resize(tensor.size);
    tensor.data = read_data.addr;
    model_loader->load_data_for(tensor);

    printf("[%4zu/%4zu] %36s - %16s, type = %6s, ", ++idx, model_loader->tensors_map.tensors.size(),
           tensor.name.c_str(), model_format_tensor_shape(tensor.ne).c_str(), ne_type_name(tensor.type));

    // This used to be a regex, but <regex> has an extreme cost to compile times.
    bool quantize = tensor.name.rfind("weight") == tensor.name.size() - 6;  // ends with 'weight'?
    bool embedd = false;
    // skip embedding for quantization or use q4_0 instead.
    if (tensor.name.find("embedding") != std::string::npos) {
      embedd = true;
    }
    // quantize only 2D tensors
    quantize &= (tensor.ne.size() == 2);

    // uncomment this to keep the output layer in FP16
    // if (tensor.name == "output.weight") {
    //    quantize = false;
    //}

    enum ne_type new_type;
    void* new_data;
    size_t new_size;
    model_buffer work;

    if (!quantize) {
      new_type = tensor.type;
      new_data = tensor.data;
      new_size = tensor.size;
      printf("size = %8.3f MB\n", tensor.size / 1024.0 / 1024.0);
    } else {
      new_type = quantized_type;
      float* f32_data;
      size_t nelements = tensor.ne.at(0) * tensor.ne.at(1);
      model_buffer f32_conv_buf;
      if (tensor.type == NE_TYPE_F32) {
        f32_data = (float*)tensor.data;
      } else if (tensor.type == NE_TYPE_F16) {
        f32_conv_buf.resize(nelements * sizeof(float));
        f32_data = (float*)f32_conv_buf.addr;
        const auto* f16_data = (const ne_fp16_t*)tensor.data;
        for (size_t i = 0; i < nelements; i++) {
          f32_data[i] = ne_fp16_to_fp32(f16_data[i]);
        }
      } else {
        throw format("type %s unsupported for integer quantization", ne_type_name(tensor.type));
      }

      printf("quantizing .. ");
      fflush(stdout);
      if (quantized_type == NE_TYPE_Q4_JBLAS) {
        if (!embedd) {  // emedding of Q4 is not supported now
          using CompType = jblas::prologue::weight_comp::gemm::WeightCompType;
          using GemmKernel = jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4KBlock;
          using GemmVnniKernel = jblas::wrapper::gemm_default::weight_comp::avx512_vnni::GemmKernelDynamicQuantS4KBlock;
          GemmKernel kernel;
          GemmVnniKernel vnnikernel;
          int k_ = tensor.ne.at(0);
          int n_ = tensor.ne.at(1);
          jblas::prologue::PackedWeight* packedw = NULL;
          int blocksize = 32;
          auto type = CompType::S4_F32;
          if (ftype == NE_FTYPE_MOSTLY_Q4_JBLAS_B32) {
            blocksize = 32;
            type = CompType::S4_F32;
          } else if (ftype == NE_FTYPE_MOSTLY_Q4_JBLAS_B128) {
            blocksize = 128;
            type = CompType::S4_F32;
          } else if (ftype == NE_FTYPE_MOSTLY_Q4_JBLAS_B1024) {
            blocksize = 1024;
            type = CompType::S4_F32;
          } else if (ftype == NE_FTYPE_MOSTLY_Q4_JBLAS_BF16_B32) {
            blocksize = 32;
            type = CompType::S4_Bf16;
          } else if (ftype == NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B32) {
            blocksize = 32;
            type = CompType::S4_F32;
          } else if (ftype == NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B128) {
            blocksize = 128;
            type = CompType::S4_F32;
          } else if (ftype == NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_BF16_B32) {
            blocksize = 32;
            type = CompType::S4_Bf16;
          }
          auto cd = jblas::utils::parallel::CpuDevice::getInstance();
          if (ftype == NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B32 || ftype == NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B128 ||
              ftype == NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_BF16_B32) {
            if (cd->AVX512F()) {
              packedw = vnnikernel.getWeightPtr()->compressWeightTranspose<JblasAVX512F>(n_, k_, (float*)tensor.data,
                                                                                         k_, blocksize, type);
            } else {
              packedw = vnnikernel.getWeightPtr()->compressWeightTranspose<JblasNoSIMD>(n_, k_, (float*)tensor.data, k_,
                                                                                        blocksize, type);
            }
          } else {
            if (cd->AVX512F()) {
              packedw = kernel.getWeightPtr()->compressWeightTranspose<JblasAVX512F>(n_, k_, (float*)tensor.data, k_,
                                                                                     blocksize, type);
            } else {
              packedw = kernel.getWeightPtr()->compressWeightTranspose<JblasNoSIMD>(n_, k_, (float*)tensor.data, k_,
                                                                                    blocksize, type);
            }
          }

          auto tsize = packedw->getSerializedSize();
          work.resize(tsize);  // upper bound on size
          packedw->serializeToBuffer((int8_t*)work.addr);
          delete packedw;
          new_type = quantized_type;
          new_data = work.addr;
          new_size = tsize;
          printf("JBLAS size = %8.2f MB -> %8.2f MB\n", tensor.size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
          goto __WRITE_BACK;
        } else {
          new_type = NE_TYPE_Q4_0;
        }
      }

      work.resize(nelements * 4);  // upper bound on size
      new_data = work.addr;
      std::vector<int64_t> hist_cur(1 << 4, 0);

      int chunk_size = 32 * 512;
      const int nchunk = (nelements + chunk_size - 1) / chunk_size;
      const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
      if (nthread_use < 2) {
        new_size = ne_quantize_chunk(new_type, f32_data, new_data, 0, nelements, hist_cur.data());
      } else {
        size_t counter = 0;
        new_size = 0;
        auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, nelements, chunk_size]() {
          std::vector<int64_t> local_hist;
          size_t local_size = 0;
          while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            size_t first = counter;
            counter += chunk_size;
            if (first >= nelements) {
              if (!local_hist.empty()) {
                for (int j = 0; j < int(local_hist.size()); ++j) {
                  hist_cur[j] += local_hist[j];
                }
                new_size += local_size;
              }
              break;
            }
            lock.unlock();
            size_t last = std::min(nelements, first + chunk_size);
            if (local_hist.empty()) {
              local_hist.resize(hist_cur.size(), 0);
            }
            local_size += ne_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
          }
        };
        if ((int)workers.size() < nthread_use - 1) {
          workers.resize(nthread_use - 1);
        }
        for (int it = 0; it < nthread_use - 1; ++it) {
          workers[it] = std::thread(compute);
        }
        compute();
        for (int it = 0; it < nthread_use - 1; ++it) {
          workers[it].join();
        }
      }

      printf("size = %8.2f MB -> %8.2f MB | hist: ", tensor.size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
      for (size_t i = 0; i < hist_cur.size(); i++) {
        hist_all[i] += hist_cur[i];
      }

      for (size_t i = 0; i < hist_cur.size(); i++) {
        printf("%5.3f ", hist_cur[i] / float(nelements));
      }
      printf("\n");
    }
  __WRITE_BACK:
    total_size_org += tensor.size;
    total_size_new += new_size;
    file_saver.write_tensor(tensor, new_type, new_data, new_size);
  }

  printf("%s: model size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
  printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);

  {
    int64_t sum_all = 0;
    for (size_t i = 0; i < hist_all.size(); i++) {
      sum_all += hist_all[i];
    }

    printf("%s: hist: ", __func__);
    for (size_t i = 0; i < hist_all.size(); i++) {
      printf("%5.3f ", hist_all[i] / float(sum_all));
    }
    printf("\n");
  }
}

//
// interface implementation
//

struct model_context* model_init_from_file(const char* path_model, struct model_context_params params) {
  ne_time_init();

  model_context* ctx = new model_context;

  if (params.seed < 0) {
    params.seed = time(NULL);
  }

  unsigned cur_percentage = 0;
  if (params.progress_callback == NULL) {
    params.progress_callback_user_data = &cur_percentage;
    params.progress_callback = [](float progress, void* ctx) {
      unsigned* cur_percentage_p = (unsigned*)ctx;
      unsigned percentage = (unsigned)(100 * progress);
      while (percentage > *cur_percentage_p) {
        *cur_percentage_p = percentage;
        fprintf(stderr, ".");
        fflush(stderr);
        if (percentage >= 100) {
          fprintf(stderr, "\n");
        }
      }
    };
  }

  ctx->rng = std::mt19937(params.seed);
  ctx->logits_all = params.logits_all;

  ne_type memory_type = params.f16_kv ? NE_TYPE_F16 : NE_TYPE_F32;

  if (!model_model_load(path_model, *ctx, params.n_ctx, params.n_gpu_layers, memory_type, params.use_mmap,
                        params.use_mlock, params.vocab_only, params.progress_callback,
                        params.progress_callback_user_data)) {
    fprintf(stderr, "%s: failed to load model\n", __func__);
    model_free(ctx);
    return nullptr;
  }

  // reserve memory for context buffers
  if (!params.vocab_only) {
    if (!kv_cache_init(ctx->model.hparams, ctx->model.kv_self, memory_type, ctx->model.hparams.n_ctx)) {
      fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
      model_free(ctx);
      return nullptr;
    }

    {
      const size_t memory_size = ne_nbytes(ctx->model.kv_self.k) + ne_nbytes(ctx->model.kv_self.v);
      fprintf(stderr, "%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
    }

    const auto& hparams = ctx->model.hparams;

    // resized during inference
    if (params.logits_all) {
      ctx->logits.reserve(hparams.n_ctx * hparams.n_vocab);
    } else {
      ctx->logits.reserve(hparams.n_vocab);
    }

    if (params.embedding) {
      ctx->embedding.resize(hparams.n_embd);
    }

    ctx->buf_compute.resize(MEM_REQ_EVAL().at(ctx->model.type));

    ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0().at(ctx->model.type));
    ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1().at(ctx->model.type));
  }

  return ctx;
}

void model_free(struct model_context* ctx) { delete ctx; }

int model_model_quantize(const char* fname_inp, const char* fname_out, enum ne_ftype ftype, int nthread) {
  try {
    model_model_quantize_internal(fname_inp, fname_out, ftype, nthread);
    return 0;
  } catch (const std::string& err) {
    fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.c_str());
    return 1;
  }
}

int model_apply_lora_from_file_internal(struct model_context* ctx, const char* path_lora, const char* path_base_model,
                                        int n_threads) {
  fprintf(stderr, "%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

  auto& model = ctx->model;

  const int64_t t_start_lora_us = ne_time_us();

  auto fin = std::ifstream(path_lora, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_lora);
    return 1;
  }

  // verify magic and version
  {
    uint32_t magic;
    fin.read((char*)&magic, sizeof(magic));
    if (magic != MODEL_FILE_MAGIC_GGLA) {
      fprintf(stderr, "%s: bad file magic\n", __func__);
      return 1;
    }
    uint32_t format_version;
    fin.read((char*)&format_version, sizeof(format_version));

    if (format_version != 1) {
      fprintf(stderr, "%s: unsupported file version\n", __func__);
      return 1;
    }
  }

  int32_t lora_r;
  int32_t lora_alpha;
  fin.read((char*)&lora_r, sizeof(lora_r));
  fin.read((char*)&lora_alpha, sizeof(lora_alpha));
  float scaling = (float)lora_alpha / (float)lora_r;

  fprintf(stderr, "%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);

  // create a temporary ne context to store the lora tensors
  // todo: calculate size from biggest possible tensor
  std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
  struct ne_init_params params;
  params.mem_size = lora_buf.size();
  params.mem_buffer = lora_buf.data();
  params.no_alloc = false;

  ne_context* lora_ctx = ne_init(params);
  std::unordered_map<std::string, struct ne_tensor*> lora_tensors;

  // create a name -> tensor map of the model to accelerate lookups
  std::unordered_map<std::string, struct ne_tensor*> model_tensors;
  for (auto& kv : model.tensors_by_name) {
    model_tensors.insert(kv);
  }

  // load base model
  std::unique_ptr<model_model_loader> model_loader;
  ne_context* base_ctx = NULL;
  model_buffer base_buf;
  if (path_base_model) {
    fprintf(stderr, "%s: loading base model from '%s'\n", __func__, path_base_model);
    model_loader.reset(new model_model_loader(path_base_model, /*use_mmap*/ true, /*vocab_only*/ false));

    size_t ctx_size;
    size_t mmapped_size;
    model_loader->calc_sizes(&ctx_size, &mmapped_size);
    base_buf.resize(ctx_size);

    ne_init_params base_params;
    base_params.mem_size = base_buf.size;
    base_params.mem_buffer = base_buf.addr;
    base_params.no_alloc = model_loader->use_mmap;

    base_ctx = ne_init(base_params);

    model_loader->ne_ctx = base_ctx;

    // maybe this should in model_model_loader
    if (model_loader->use_mmap) {
      model_loader->mapping.reset(new model_mmap(&model_loader->file_loaders.at(0)->file, /* prefetch */ 0));
    }
  }

  // read tensors and apply
  bool warned = false;
  int n_tensors = 0;
  while (true) {
    int32_t n_dims;
    int32_t length;
    int32_t ftype;

    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char*>(&length), sizeof(length));
    fin.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));
    if (fin.eof()) {
      break;
    }

    int32_t ne[2] = {1, 1};
    for (int i = 0; i < n_dims; ++i) {
      fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
    }

    std::string name;
    {
      char buf[1024];
      fin.read(buf, length);
      name = std::string(buf, length);
    }

    // check for lora suffix and get the type of tensor
    const std::string lora_suffix = ".lora";
    size_t pos = name.rfind(lora_suffix);
    if (pos == std::string::npos) {
      fprintf(stderr, "%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
      return 1;
    }

    std::string lora_type = name.substr(pos + lora_suffix.length());
    std::string base_name = name;
    base_name.erase(pos);
    // fprintf(stderr, "%s: %s => %s (lora type %s) ", __func__, name.c_str(),base_name.c_str(), lora_type.c_str());

    if (model_tensors.find(base_name) == model_tensors.end()) {
      fprintf(stderr, "%s: unknown tensor '%s' in lora adapter\n", __func__, name.data());
      return 1;
    }

    // create ne tensor
    ne_type wtype;
    switch (ftype) {
      case 0:
        wtype = NE_TYPE_F32;
        break;
      case 1:
        wtype = NE_TYPE_F16;
        break;
      default: {
        fprintf(stderr, "%s: invalid tensor data type '%d'\n", __func__, ftype);
        return false;
      }
    }
    ne_tensor* lora_tensor;
    if (n_dims == 2) {
      lora_tensor = ne_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1], NE_SIZE_CALC);
    } else {
      fprintf(stderr, "%s: unsupported tensor dimension %d\n", __func__, n_dims);
      return 1;
    }

    // load tensor data
    size_t offset = fin.tellg();
    size_t tensor_data_size = ne_nbytes(lora_tensor);
    offset = (offset + 31) & -32;
    fin.seekg(offset);
    fin.read((char*)lora_tensor->data, tensor_data_size);

    lora_tensors[name] = lora_tensor;

    // check if we have both A and B tensors and apply
    if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
        lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {
      ne_tensor* dest_t = model_tensors[base_name];
      ne_tensor* base_t;
      if (model_loader) {
        // load from base model
        if (model_loader->tensors_map.name_to_idx.find(base_name) == model_loader->tensors_map.name_to_idx.end()) {
          fprintf(stderr, "%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
          return 1;
        }
        size_t idx = model_loader->tensors_map.name_to_idx[base_name];
        model_load_tensor& lt = model_loader->tensors_map.tensors[idx];
        base_t =
            model_loader->get_tensor(base_name, {(uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1]}, NE_BACKEND_CPU);
        lt.data = (uint8_t*)lt.ne_tensor->data;
        model_loader->load_data_for(lt);
        lt.ne_tensor->data = lt.data;
      } else {
        base_t = dest_t;
      }

      if (ne_is_quantized(base_t->type)) {
        if (!warned) {
          fprintf(stderr,
                  "%s: warning: using a lora adapter with a quantized model may result in poor quality, "
                  "use a f16 or f32 base model with --lora-base\n",
                  __func__);
          warned = true;
        }
      }

      ne_tensor* loraA = lora_tensors[base_name + ".loraA"];
      ne_tensor* loraB = lora_tensors[base_name + ".loraB"];

      if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
        fprintf(stderr,
                "%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64
                ");"
                " are you sure that this adapter is for this model?\n",
                __func__, base_t->ne[0], loraA->ne[1]);
        return 1;
      }

      // w = w + BA*s
      ne_tensor* BA = ne_mul_mat(lora_ctx, loraA, loraB);

      if (scaling != 1.0f) {
        ne_tensor* scale_tensor = ne_new_f32(lora_ctx, scaling);
        BA = ne_scale_inplace(lora_ctx, BA, scale_tensor);
      }

      ne_tensor* r;
      if (base_t == dest_t) {
        r = ne_add_inplace(lora_ctx, dest_t, BA);
      } else {
        r = ne_add(lora_ctx, base_t, BA);
        r = ne_cpy(lora_ctx, r, dest_t);
      }

      struct ne_cgraph gf = ne_build_forward(r);
      gf.n_threads = n_threads;
      ne_graph_compute(lora_ctx, &gf);

      // we won't need these tensors again, reset the context to save memory
      ne_free(lora_ctx);
      lora_ctx = ne_init(params);
      lora_tensors.clear();

      n_tensors++;
      if (n_tensors % 4 == 0) {
        fprintf(stderr, ".");
      }
    }
  }

  // TODO: this should be in a destructor, it will leak on failure
  ne_free(lora_ctx);
  if (base_ctx) {
    ne_free(base_ctx);
  }

  const int64_t t_lora_us = ne_time_us() - t_start_lora_us;
  fprintf(stderr, " done (%.2f ms)\n", t_lora_us / 1000.0);

  return 0;
}

int model_apply_lora_from_file(struct model_context* ctx, const char* path_lora, const char* path_base_model,
                               int n_threads) {
  try {
    return model_apply_lora_from_file_internal(ctx, path_lora, path_base_model, n_threads);
  } catch (const std::string& err) {
    fprintf(stderr, "%s: failed to apply lora adapter: %s\n", __func__, err.c_str());
    return 1;
  }
}

int model_get_kv_cache_token_count(const struct model_context* ctx) { return ctx->model.kv_self.n; }

#define MODEL_MAX_RNG_STATE (64 * 1024)

void model_set_rng_seed(struct model_context* ctx, int seed) {
  if (seed < 0) {
    seed = time(NULL);
  }
  ctx->rng.seed(seed);
}

// Returns the *maximum* size of the state
size_t model_get_state_size(const struct model_context* ctx) {
  // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized
  // state. for reference, std::mt19937(1337) serializes to 6701 bytes.
  const size_t s_rng_size = sizeof(size_t);
  const size_t s_rng = MODEL_MAX_RNG_STATE;
  const size_t s_logits_capacity = sizeof(size_t);
  const size_t s_logits_size = sizeof(size_t);
  const size_t s_logits = ctx->logits.capacity() * sizeof(float);
  const size_t s_embedding_size = sizeof(size_t);
  const size_t s_embedding = ctx->embedding.size() * sizeof(float);
  const size_t s_kv_size = sizeof(size_t);
  const size_t s_kv_ntok = sizeof(int);
  const size_t s_kv = ctx->model.kv_self.buf.size;

  const size_t s_total = (+s_rng_size + s_rng + s_logits_capacity + s_logits_size + s_logits + s_embedding_size +
                          s_embedding + s_kv_size + s_kv_ntok + s_kv);

  return s_total;
}

// Copies the state to the specified destination address
size_t model_copy_state_data(struct model_context* ctx, uint8_t* dst) {
  uint8_t* out = dst;

  // copy rng
  {
    std::stringstream rng_ss;
    rng_ss << ctx->rng;

    const size_t rng_size = rng_ss.str().size();
    char rng_buf[MODEL_MAX_RNG_STATE];

    memset(&rng_buf[0], 0, MODEL_MAX_RNG_STATE);
    memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

    memcpy(out, &rng_size, sizeof(rng_size));
    out += sizeof(rng_size);
    memcpy(out, &rng_buf[0], MODEL_MAX_RNG_STATE);
    out += MODEL_MAX_RNG_STATE;
  }

  // copy logits
  {
    const size_t logits_cap = ctx->logits.capacity();
    const size_t logits_size = ctx->logits.size();

    memcpy(out, &logits_cap, sizeof(logits_cap));
    out += sizeof(logits_cap);
    memcpy(out, &logits_size, sizeof(logits_size));
    out += sizeof(logits_size);

    if (logits_size) {
      memcpy(out, ctx->logits.data(), logits_size * sizeof(float));
    }

    out += logits_cap * sizeof(float);
  }

  // copy embeddings
  {
    const size_t embedding_size = ctx->embedding.size();

    memcpy(out, &embedding_size, sizeof(embedding_size));
    out += sizeof(embedding_size);

    if (embedding_size) {
      memcpy(out, ctx->embedding.data(), embedding_size * sizeof(float));
      out += embedding_size * sizeof(float);
    }
  }

  // copy kv cache
  {
    const auto& kv_self = ctx->model.kv_self;
    const auto& hparams = ctx->model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = hparams.n_ctx;

    const size_t kv_size = kv_self.buf.size;
    const int kv_ntok = model_get_kv_cache_token_count(ctx);

    memcpy(out, &kv_size, sizeof(kv_size));
    out += sizeof(kv_size);
    memcpy(out, &kv_ntok, sizeof(kv_ntok));
    out += sizeof(kv_ntok);

    if (kv_size) {
      const size_t elt_size = ne_element_size(kv_self.k);

      char buffer[4096];

      ne_context* cpy_ctx = ne_init({sizeof(buffer), buffer, /* no_alloc */ true});
      ne_cgraph gf{};
      gf.n_threads = 1;

      ne_tensor* kout3d = ne_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer, NE_SIZE_CALC);
      kout3d->data = out;
      out += ne_nbytes(kout3d);

      ne_tensor* vout3d = ne_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer, NE_SIZE_CALC);
      vout3d->data = out;
      out += ne_nbytes(vout3d);

      ne_tensor* k3d =
          ne_view_3d(cpy_ctx, kv_self.k, n_embd, kv_ntok, n_layer, elt_size * n_embd, elt_size * n_embd * n_ctx, 0);

      ne_tensor* v3d =
          ne_view_3d(cpy_ctx, kv_self.v, kv_ntok, n_embd, n_layer, elt_size * n_ctx, elt_size * n_ctx * n_embd, 0);

      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, k3d, kout3d));
      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, v3d, vout3d));
      ne_graph_compute(cpy_ctx, &gf);

      ne_free(cpy_ctx);
    }
  }

  const size_t written = out - dst;
  const size_t max_size = model_get_state_size(ctx);

  MODEL_ASSERT(written <= max_size);

  return written;
}

// Sets the state reading from the specified source address
size_t model_set_state_data(struct model_context* ctx, uint8_t* src) {
  uint8_t* inp = src;

  // set rng
  {
    size_t rng_size;
    char rng_buf[MODEL_MAX_RNG_STATE];

    memcpy(&rng_size, inp, sizeof(rng_size));
    inp += sizeof(rng_size);
    memcpy(&rng_buf[0], inp, MODEL_MAX_RNG_STATE);
    inp += MODEL_MAX_RNG_STATE;

    std::stringstream rng_ss;
    rng_ss.str(std::string(&rng_buf[0], rng_size));
    rng_ss >> ctx->rng;

    MODEL_ASSERT(rng_ss.fail() == false);
  }

  // set logits
  {
    size_t logits_cap;
    size_t logits_size;

    memcpy(&logits_cap, inp, sizeof(logits_cap));
    inp += sizeof(logits_cap);
    memcpy(&logits_size, inp, sizeof(logits_size));
    inp += sizeof(logits_size);

    MODEL_ASSERT(ctx->logits.capacity() == logits_cap);

    if (logits_size) {
      ctx->logits.resize(logits_size);
      memcpy(ctx->logits.data(), inp, logits_size * sizeof(float));
    }

    inp += logits_cap * sizeof(float);
  }

  // set embeddings
  {
    size_t embedding_size;

    memcpy(&embedding_size, inp, sizeof(embedding_size));
    inp += sizeof(embedding_size);

    MODEL_ASSERT(ctx->embedding.capacity() == embedding_size);

    if (embedding_size) {
      memcpy(ctx->embedding.data(), inp, embedding_size * sizeof(float));
      inp += embedding_size * sizeof(float);
    }
  }

  // set kv cache
  {
    const auto& kv_self = ctx->model.kv_self;
    const auto& hparams = ctx->model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = hparams.n_ctx;

    size_t kv_size;
    int kv_ntok;

    memcpy(&kv_size, inp, sizeof(kv_size));
    inp += sizeof(kv_size);
    memcpy(&kv_ntok, inp, sizeof(kv_ntok));
    inp += sizeof(kv_ntok);

    if (kv_size) {
      MODEL_ASSERT(kv_self.buf.size == kv_size);

      const size_t elt_size = ne_element_size(kv_self.k);

      char buffer[4096];

      ne_context* cpy_ctx = ne_init({sizeof(buffer), buffer, /* no_alloc */ true});
      ne_cgraph gf{};
      gf.n_threads = 1;

      ne_tensor* kin3d = ne_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer, NE_SIZE_CALC);
      kin3d->data = (void*)inp;
      inp += ne_nbytes(kin3d);

      ne_tensor* vin3d = ne_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer, NE_SIZE_CALC);
      vin3d->data = (void*)inp;
      inp += ne_nbytes(vin3d);

      ne_tensor* k3d =
          ne_view_3d(cpy_ctx, kv_self.k, n_embd, kv_ntok, n_layer, elt_size * n_embd, elt_size * n_embd * n_ctx, 0);

      ne_tensor* v3d =
          ne_view_3d(cpy_ctx, kv_self.v, kv_ntok, n_embd, n_layer, elt_size * n_ctx, elt_size * n_ctx * n_embd, 0);

      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, kin3d, k3d));
      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, vin3d, v3d));
      ne_graph_compute(cpy_ctx, &gf);

      ne_free(cpy_ctx);
    }

    ctx->model.kv_self.n = kv_ntok;
  }

  const size_t nread = inp - src;
  const size_t max_size = model_get_state_size(ctx);

  MODEL_ASSERT(nread <= max_size);

  return nread;
}

bool model_load_session_file(struct model_context* ctx, const char* path_session, model_token* tokens_out,
                             size_t n_token_capacity, size_t* n_token_count_out) {
  model_file file(path_session, "rb");

  // sanity checks
  {
    const uint32_t magic = file.read_u32();
    const uint32_t version = file.read_u32();

    if (magic != MODEL_SESSION_MAGIC || version != MODEL_SESSION_VERSION) {
      fprintf(stderr, "%s : unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
      return false;
    }

    model_hparams session_hparams;
    file.read_raw(&session_hparams, sizeof(model_hparams));

    if (session_hparams != ctx->model.hparams) {
      fprintf(stderr, "%s : model hparams didn't match from session file!\n", __func__);
      return false;
    }
  }

  // load the prompt
  {
    const uint32_t n_token_count = file.read_u32();

    if (n_token_count > n_token_capacity) {
      fprintf(stderr, "%s : token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count,
              n_token_capacity);
      return false;
    }

    file.read_raw(tokens_out, sizeof(model_token) * n_token_count);
    *n_token_count_out = n_token_count;
  }

  // restore the context state
  {
    const size_t n_state_size_cur = file.size - file.tell();
    const size_t n_state_size_max = model_get_state_size(ctx);

    if (n_state_size_cur > n_state_size_max) {
      fprintf(stderr, "%s : the state size in session file is too big! max %zu, got %zu\n", __func__, n_state_size_max,
              n_state_size_cur);
      return false;
    }

    std::vector<uint8_t> state_data(n_state_size_max);
    file.read_raw(state_data.data(), n_state_size_cur);

    model_set_state_data(ctx, state_data.data());
  }

  return true;
}

bool model_save_session_file(struct model_context* ctx, const char* path_session, const model_token* tokens,
                             size_t n_token_count) {
  model_file file(path_session, "wb");

  file.write_u32(MODEL_SESSION_MAGIC);
  file.write_u32(MODEL_SESSION_VERSION);

  file.write_raw(&ctx->model.hparams, sizeof(model_hparams));

  // save the prompt
  file.write_u32((uint32_t)n_token_count);
  file.write_raw(tokens, sizeof(model_token) * n_token_count);

  // save the context state
  {
    const size_t n_state_size_max = model_get_state_size(ctx);

    std::vector<uint8_t> state_data(n_state_size_max);
    const size_t n_state_size_cur = model_copy_state_data(ctx, state_data.data());

    file.write_raw(state_data.data(), n_state_size_cur);
  }

  return true;
}

int model_tokenize(struct model_context* ctx, const char* text, model_token* tokens, int n_max_tokens, bool add_bos) {
  auto res = model_tokenize(ctx->vocab, text, add_bos);

  if (n_max_tokens < (int)res.size()) {
    fprintf(stderr, "%s: too many tokens\n", __func__);
    return -((int)res.size());
  }

  for (size_t i = 0; i < res.size(); i++) {
    tokens[i] = res[i];
  }

  return res.size();
}

int model_n_vocab(const struct model_context* ctx) { return ctx->vocab.id_to_token.size(); }

int model_n_ctx(const struct model_context* ctx) { return ctx->model.hparams.n_ctx; }

int model_n_embd(const struct model_context* ctx) { return ctx->model.hparams.n_embd; }

float* model_get_logits(struct model_context* ctx) { return ctx->logits.data(); }

float* model_get_embeddings(struct model_context* ctx) { return ctx->embedding.data(); }

const char* model_token_to_str(const struct model_context* ctx, model_token token) {
  if (token >= model_n_vocab(ctx)) {
    return nullptr;
  }

  return ctx->vocab.id_to_token[token].tok.c_str();
}

model_token model_token_bos() { return 1; }

model_token model_token_eos() { return 2; }

model_token model_token_nl() { return 13; }

void model_print_timings(struct model_context* ctx) {
  const int64_t t_end_us = ne_time_us();

  const int32_t n_sample = std::max(1, ctx->n_sample);
  const int32_t n_eval = std::max(1, ctx->n_eval);
  const int32_t n_p_eval = std::max(1, ctx->n_p_eval);

  fprintf(stderr, "\n");
  fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0);
  fprintf(stderr, "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per token)\n", __func__,
          1e-3 * ctx->t_sample_us, n_sample, 1e-3 * ctx->t_sample_us / n_sample);
  fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__,
          1e-3 * ctx->t_p_eval_us, n_p_eval, 1e-3 * ctx->t_p_eval_us / n_p_eval);
  fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per token)\n", __func__,
          1e-3 * ctx->t_eval_us, n_eval, 1e-3 * ctx->t_eval_us / n_eval);
  fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us) / 1000.0);
  printf("========== eval time log of each prediction ==========\n");
  for (int i = 0; i < ctx->eval_times.size(); ++i) {
    printf("prediction %3d, time: %.2fms\n", i, ctx->eval_times[i] / 1000.0f);
  }
}

void model_reset_timings(struct model_context* ctx) {
  ctx->t_start_us = ne_time_us();
  ctx->t_sample_us = ctx->n_sample = 0;
  ctx->t_eval_us = ctx->n_eval = 0;
  ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char* model_print_system_info(void) {
  static std::string s;

  s = "";
  s += "AVX = " + std::to_string(ne_cpu_has_avx()) + " | ";
  s += "AVX2 = " + std::to_string(ne_cpu_has_avx2()) + " | ";
  s += "AVX512 = " + std::to_string(ne_cpu_has_avx512()) + " | ";
  s += "AVX512_VBMI = " + std::to_string(ne_cpu_has_avx512_vbmi()) + " | ";
  s += "AVX512_VNNI = " + std::to_string(ne_cpu_has_avx512_vnni()) + " | ";
  s += "FMA = " + std::to_string(ne_cpu_has_fma()) + " | ";
  s += "F16C = " + std::to_string(ne_cpu_has_f16c()) + " | ";
  s += "BLAS = " + std::to_string(ne_cpu_has_blas()) + " | ";
  s += "SSE3 = " + std::to_string(ne_cpu_has_sse3()) + " | ";
  s += "VSX = " + std::to_string(ne_cpu_has_vsx()) + " | ";

  return s.c_str();
}

// For internal test use
std::vector<std::pair<std::string, struct ne_tensor*>>& model_internal_get_tensor_map(struct model_context* ctx) {
  return ctx->model.tensors_by_name;
}
