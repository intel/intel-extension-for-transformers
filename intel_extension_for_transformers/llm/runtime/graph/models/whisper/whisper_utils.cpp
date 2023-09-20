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
#include "models/whisper/whisper.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"
#include "models/models.h"

#include "whisper.h"

// no defaults for now
struct mpt_hparams {
  int32_t d_model = 0;
  int32_t max_seq_len = 0;
  int32_t n_heads = 0;
  int32_t n_layers = 0;
  int32_t n_vocab = 0;
  float alibi_bias_max = 0;
  float clip_qkv = 0;
  int32_t ftype = 0;
  int32_t n_ctx = 0;
};

struct mpt_layer {
  // pre normalization
  struct ne_tensor* norm_1_weight;

  // attention
  struct ne_tensor* c_attn_wqkv_weight;
  struct ne_tensor* c_attn_out_proj_weight;

  // post normalization
  struct ne_tensor* norm_2_weight;

  // ff
  struct ne_tensor* ffn_up_proj;
  struct ne_tensor* ffn_down_proj;
};

struct mpt_model {
  mpt_hparams hparams;

  struct ne_tensor* wte_weight;     // position embedding
  struct ne_tensor* norm_f_weight;  // language model head

  std::vector<mpt_layer> layers;

  // key + value memory
  struct ne_tensor* memory_k;
  struct ne_tensor* memory_v;

  struct ne_context* ctx;
  std::map<std::string, struct ne_tensor*> tensors;
};

struct mpt_params {
  int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

  int32_t seed = -1;        // RNG seed
  int32_t n_predict = 200;  // new tokens to predict
  int32_t n_batch = 8;      // batch size for prompt processing
  int32_t n_ctx = 512;

  std::string model = "";  // model path
  std::string prompt = "";
  std::string token_test = "";

  bool perplexity = false;

  // sampling parameters
  int32_t top_k = 0;
  float top_p = 1.0f;
  float temp = 0.8f;
  int32_t repeat_last_n = 64;
  float repeat_penalty = 1.02f;
};
// load the model's weights from a file
bool mpt_model_load(const std::string& fname, mpt_model& model, gpt_vocab& vocab) {
  printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

  auto fin = std::ifstream(fname, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    return false;
  }

  // verify magic
  {
    uint32_t magic;
    fin.read((char*)&magic, sizeof(magic));
    if (magic != NE_FILE_MAGIC) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
      return false;
    }
  }

  // load hparams
  {
    auto& hparams = model.hparams;

    fin.read((char*)&hparams.d_model, sizeof(hparams.d_model));
    fin.read((char*)&hparams.max_seq_len, sizeof(hparams.max_seq_len));
    fin.read((char*)&hparams.n_heads, sizeof(hparams.n_heads));
    fin.read((char*)&hparams.n_layers, sizeof(hparams.n_layers));
    fin.read((char*)&hparams.n_vocab, sizeof(hparams.n_vocab));
    fin.read((char*)&hparams.alibi_bias_max, sizeof(hparams.alibi_bias_max));
    fin.read((char*)&hparams.clip_qkv, sizeof(hparams.clip_qkv));
    fin.read((char*)&hparams.ftype, sizeof(hparams.ftype));

    hparams.n_ctx = std::min(hparams.max_seq_len, hparams.n_ctx);

    const int32_t qntvr = hparams.ftype / NE_QNT_VERSION_FACTOR;

    printf("%s: d_model        = %d\n", __func__, hparams.d_model);
    printf("%s: max_seq_len    = %d\n", __func__, hparams.max_seq_len);
    printf("%s: n_ctx          = %d\n", __func__, hparams.n_ctx);
    printf("%s: n_heads        = %d\n", __func__, hparams.n_heads);
    printf("%s: n_layers       = %d\n", __func__, hparams.n_layers);
    printf("%s: n_vocab        = %d\n", __func__, hparams.n_vocab);
    printf("%s: alibi_bias_max = %f\n", __func__, hparams.alibi_bias_max);
    printf("%s: clip_qkv       = %f\n", __func__, hparams.clip_qkv);
    printf("%s: ftype          = %d\n", __func__, hparams.ftype);
    printf("%s: qntvr          = %d\n", __func__, qntvr);

    hparams.ftype %= NE_QNT_VERSION_FACTOR;
  }

  // load vocab
  {
    const int32_t n_vocab = model.hparams.n_vocab;

    std::string word;
    std::vector<char> buf(128);

    for (int i = 0; i < n_vocab; i++) {
      uint32_t len;
      fin.read((char*)&len, sizeof(len));

      buf.resize(len);
      fin.read((char*)buf.data(), len);
      word.assign(buf.data(), len);

      // Convert token from utf-8
      std::wstring word_multibytes = convert_to_wstring(word);
      word.resize(word_multibytes.size());
      for (size_t w = 0; w < word_multibytes.size(); w++) {
        word[w] = uint8_t(word_multibytes[w]);
      }

      vocab.token_to_id[word] = i;
      vocab.id_to_token[i] = word;
    }
  }
  fprintf(stderr, "ddddddccccc1\n");
  // for the big tensors, we have the option to store the data in 16-bit
  // floats or quantized in order to save memory and also to speed up the
  // computation
  ne_type wtype = ne_ftype_to_ne_type((ne_ftype)(model.hparams.ftype));
  if (wtype == NE_TYPE_COUNT) {
    fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n", __func__, fname.c_str(), model.hparams.ftype);
    return false;
  }

  auto& ctx = model.ctx;

  size_t ctx_size = 0;

  const auto& hparams = model.hparams;
  const size_t n_ctx = hparams.n_ctx;
  fprintf(stderr, "ddddddccccc\n");
  {
    const size_t n_embd = hparams.d_model;
    const size_t n_layer = hparams.n_layers;
    const size_t n_vocab = hparams.n_vocab;

    ctx_size += n_embd * n_vocab * ne_type_sizef(wtype);  // wte_weight
    ctx_size += n_embd * ne_type_sizef(NE_TYPE_F32);      // norm_f_weight

    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));         // ln_1_weight
    ctx_size += n_layer * (3 * n_embd * n_embd * ne_type_sizef(wtype));  // attn_Wqkv_weight
    ctx_size += n_layer * (n_embd * n_embd * ne_type_sizef(wtype));      // attn_out_proj_weight
    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));         // ln_2_weight
    ctx_size += n_layer * (4 * n_embd * n_embd * ne_type_sizef(wtype));  // mlp_mlp_up_weight
    ctx_size += n_layer * (n_embd * n_embd * 4 * ne_type_sizef(wtype));  // mlp_mlp_down_weight

    ctx_size += n_ctx * n_layer * n_embd * ne_type_sizef(NE_TYPE_F16);  // memory_k
    ctx_size += n_ctx * n_layer * n_embd * ne_type_sizef(NE_TYPE_F16);  // memory_v

    ctx_size += (1 + 6 * n_layer) * 512;  // object overhead

    printf("%s: ne ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));
  }

  // create the ne context
  {
    struct ne_init_params params = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };

    model.ctx = ne_init(params);
    if (!model.ctx) {
      fprintf(stderr, "%s: ne_init() failed\n", __func__);
      return false;
    }
  }

  // prepare memory for the weights
  {
    const auto& hparams = model.hparams;

    const size_t n_embd = hparams.d_model;
    const size_t n_layer = hparams.n_layers;
    const size_t n_vocab = hparams.n_vocab;

    model.layers.resize(n_layer);

    model.wte_weight = ne_new_tensor_2d(ctx, wtype, n_embd, n_vocab, NE_SIZE_CALC);
    model.norm_f_weight = d_ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd);

    // map by name
    model.tensors["transformer.wte.weight"] = model.wte_weight;
    model.tensors["transformer.norm_f.weight"] = model.norm_f_weight;

    for (int i = 0; i < (int)n_layer; ++i) {
      auto& layer = model.layers[i];

      layer.norm_1_weight = d_ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd);
      layer.c_attn_wqkv_weight = ne_new_tensor_2d(ctx, wtype, n_embd, 3 * n_embd, NE_SIZE_CALC);
      layer.c_attn_out_proj_weight = ne_new_tensor_2d(ctx, wtype, n_embd, n_embd, NE_SIZE_CALC);
      layer.norm_2_weight = d_ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd);
      layer.ffn_up_proj = ne_new_tensor_2d(ctx, wtype, n_embd, 4 * n_embd, NE_SIZE_CALC);
      layer.ffn_down_proj = ne_new_tensor_2d(ctx, wtype, 4 * n_embd, n_embd, NE_SIZE_CALC);

      // map by name
      model.tensors["transformer.blocks." + std::to_string(i) + ".norm_1.weight"] = layer.norm_1_weight;
      model.tensors["transformer.blocks." + std::to_string(i) + ".attn.Wqkv.weight"] = layer.c_attn_wqkv_weight;
      model.tensors["transformer.blocks." + std::to_string(i) + ".attn.out_proj.weight"] = layer.c_attn_out_proj_weight;
      model.tensors["transformer.blocks." + std::to_string(i) + ".norm_2.weight"] = layer.norm_2_weight;
      model.tensors["transformer.blocks." + std::to_string(i) + ".ffn.up_proj.weight"] = layer.ffn_up_proj;
      model.tensors["transformer.blocks." + std::to_string(i) + ".ffn.down_proj.weight"] = layer.ffn_down_proj;
    }
  }

  // key + value memory
  {
    const auto& hparams = model.hparams;

    const size_t n_embd = hparams.d_model;
    const size_t n_layer = hparams.n_layers;

    const int64_t n_mem = n_layer * n_ctx;
    const int64_t n_elements = n_embd * n_mem;

    model.memory_k = d_ne_new_tensor_1d(ctx, NE_TYPE_F16, n_elements);
    model.memory_v = d_ne_new_tensor_1d(ctx, NE_TYPE_F16, n_elements);

    const size_t memory_size = ne_nbytes(model.memory_k) + ne_nbytes(model.memory_v);

    printf("%s: memory_size = %8.2f MB, n_mem = %" PRId64 "\n", __func__, memory_size / 1024.0 / 1024.0, n_mem);
  }

  // load weights
  {
    int n_tensors = 0;
    size_t total_size = 0;

    printf("%s: ", __func__);

    while (true) {
      int32_t n_dims;
      int32_t length;
      int32_t ttype;

      fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
      fin.read(reinterpret_cast<char*>(&length), sizeof(length));
      fin.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

      if (fin.eof()) {
        break;
      }

      int32_t nelements = 1;
      int32_t ne[2] = {1, 1};
      for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
      }

      std::string name(length, 0);
      fin.read(&name[0], length);

      if (model.tensors.find(name) == model.tensors.end()) {
        fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
        return false;
      }

      auto tensor = model.tensors[name];
      if (ne_nelements(tensor) != nelements) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
        return false;
      }

      if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong shape in model file: got [%5d, "
                "%5d], expected [%5d, %5d]\n",
                __func__, name.c_str(), (int)tensor->ne[0], (int)tensor->ne[1], ne[0], ne[1]);
        return false;
      }

      // for debugging
      if (0) {
        printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), ne[0], ne[1],
               ne_type_name(ne_type(ttype)), ne_nbytes(tensor) / 1024.0 / 1024.0, ne_nbytes(tensor));
      }

      const size_t bpe = ne_type_size(ne_type(ttype));

      if ((nelements * bpe) / ne_blck_size(tensor->type) != ne_nbytes(tensor)) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong size in model file: got %zu, "
                "expected %zu\n",
                __func__, name.c_str(), ne_nbytes(tensor), nelements * bpe);
        return false;
      }

      fin.read(reinterpret_cast<char*>(tensor->data), ne_nbytes(tensor));

      total_size += ne_nbytes(tensor);
      if (++n_tensors % 8 == 0) {
        printf(".");
        fflush(stdout);
      }
    }

    printf(" done\n");

    printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
  }

  fin.close();

  return true;
}

void model_load_internal(const std::string& fname, model_archs arch, model_context& lctx, int n_ctx, int n_gpu_layers,
                         bool use_mmap, bool use_mlock, bool vocab_only, model_progress_callback progress_callback,
                         void* progress_callback_user_data) {
  lctx.t_start_us = ne_time_us();

  std::unique_ptr<IModel> ms(new WHISPER());
  ms->init(fname.c_str(), lctx, n_ctx, n_gpu_layers, use_mmap, use_mlock, vocab_only);
  ms->load(lctx, progress_callback, progress_callback_user_data);

  lctx.t_load_us = ne_time_us() - lctx.t_start_us;
}

void WHISPER::init(const char* path_model, model_context& lctx, int n_ctx_, int n_gpu_layer_, bool use_mmap_,
                   bool use_mlock_, bool vocab_only_) {
  n_ctx = n_ctx_;
  n_gpu_layer = n_gpu_layer_;
  use_mmap = use_mmap_;
  use_mlock = use_mlock_;
  vocab_only = vocab_only_;
  auto& model = lctx.model;
  gpt_vocab vocab;
  mpt_model mpt_model;
  mpt_params params;

  mpt_model.hparams.n_ctx = params.n_ctx;
  fprintf(stderr, "1333333333 \n");
  mpt_model_load(path_model, mpt_model, vocab);
  fprintf(stderr, "2333333333 \n");
  // ml.reset(new model_model_loader(path_model, use_mmap, vocab_only));
  // lctx.vocab = std::move(ml->file_loaders.at(0)->vocab);
  // model.hparams = ml->file_loaders.at(0)->hparams;
  // model_file_version file_version = ml->file_loaders.at(0)->file_version;
  // auto& hparams = model.hparams;
  // n_ff = 4 * hparams.n_embd;
  // hparams.n_ctx = n_ctx;
  // fprintf(stderr, "%s: n_vocab    = %u\n", __func__, hparams.n_vocab);
  // fprintf(stderr, "%s: n_ctx      = %u\n", __func__, hparams.n_ctx);
  // fprintf(stderr, "%s: n_embd     = %u\n", __func__, hparams.n_embd);
  // fprintf(stderr, "%s: n_mult     = %u\n", __func__, hparams.n_mult);
  // fprintf(stderr, "%s: n_head     = %u\n", __func__, hparams.n_head);
  // fprintf(stderr, "%s: n_layer    = %u\n", __func__, hparams.n_layer);
  // fprintf(stderr, "%s: n_rot      = %u\n", __func__, hparams.n_rot);
  // fprintf(stderr, "%s: n_ff       = %u\n", __func__, n_ff);
  // fprintf(stderr, "%s: n_parts    = %zu\n", __func__, ml->file_loaders.size());
  // n_embd = hparams.n_embd;
  // n_vocab = hparams.n_vocab;
  // n_layer = hparams.n_layer;
  // scratch = whisper_mem_req(n_layer);
  // model.scratchs = scratch;
}

#define MODEL_BACKEND_OFFLOAD NE_BACKEND_CPU
void WHISPER::load(model_context& lctx, model_progress_callback progress_callback, void* progress_callback_user_data) {
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

  model.others[0] =
      ml->get_tensor("encoder.positional_embedding", {n_embd, (unsigned int)n_ctx}, NE_BACKEND_CPU);   // todo wwq
  model.others[1] = ml->get_tensor("encoder.conv1.weight", {n_embd, n_embd, n_embd}, NE_BACKEND_CPU);  // todo nmels???
  model.others[2] = ml->get_tensor("encoder.conv1.bias", {n_embd}, NE_BACKEND_CPU);

  model.others[3] = ml->get_tensor("encoder.conv2.weight", {n_embd, n_embd, n_embd}, NE_BACKEND_CPU);  // todo
  model.others[4] = ml->get_tensor("encoder.conv2.bias", {1, n_embd}, NE_BACKEND_CPU);
  model.others[5] = ml->get_tensor("encoder.ln_post.weight", {n_embd}, NE_BACKEND_CPU);
  model.others[6] = ml->get_tensor("encoder.ln_post.bias", {n_embd}, NE_BACKEND_CPU);
  model.others[7] = ml->get_tensor("decoder.positional_embedding", {n_embd, (unsigned int)n_ctx}, NE_BACKEND_CPU);
  model.others[8] = ml->get_tensor("decoder.token_embedding.weight", {n_embd, n_vocab}, NE_BACKEND_CPU);
  model.others[9] = ml->get_tensor("decoder.ln.weight", {n_embd}, NE_BACKEND_CPU);
  model.others[10] = ml->get_tensor("decoder.ln.bias", {n_embd}, NE_BACKEND_CPU);
  const int i_gpu_start = n_layer - n_gpu_layer;

  model.layers.resize(n_layer);
  size_t vram_total = 0;
  for (uint32_t i = 0; i < n_layer; ++i) {
    const ne_backend backend = int(i) < i_gpu_start ? NE_BACKEND_CPU : MODEL_BACKEND_OFFLOAD;
    auto& layer = model.layers[i];
    std::string layers_i = "gpt_neox.layers." + std::to_string(i);

    // norm: cur = ln_1_g*cur + ln_1_b

    layer.norm[0] = ml->get_tensor("encoder.blocks." + layers_i + ".attn_ln.weight", {n_embd}, backend);
    layer.norm[1] = ml->get_tensor("encoder.blocks." + layers_i + ".attn_ln.bias", {n_embd}, backend);

    // qkv GEMM

    layer.attn[0] = ml->get_tensor("encoder.blocks." + layers_i + ".attn.query.weight", {n_embd, n_embd}, backend);
    layer.attn[1] = ml->get_tensor("encoder.blocks." + layers_i + ".attn.query.bias", {n_embd}, backend);
    layer.attn[2] = ml->get_tensor("encoder.blocks." + layers_i + ".attn.key.weight", {n_embd}, backend);
    layer.attn[3] = ml->get_tensor("encoder.blocks." + layers_i + ".attn.value.weight", {n_embd, n_embd}, backend);
    layer.attn[4] = ml->get_tensor("encoder.blocks." + layers_i + ".attn.value.bias", {n_embd}, backend);
    layer.attn[5] = ml->get_tensor("encoder.blocks." + layers_i + ".attn.out.weight", {n_embd}, backend);
    layer.attn[6] = ml->get_tensor("encoder.blocks." + layers_i + ".attn.out.bias", {n_embd}, backend);

    // ffn GEMM
    layer.ffn[0] = ml->get_tensor("encoder.blocks." + layers_i + ".cross_attn_ln.weight", {n_embd}, backend);
    layer.ffn[1] = ml->get_tensor("encoder.blocks." + layers_i + ".cross_attn_ln.bias", {n_embd}, backend);
    layer.ffn[2] = ml->get_tensor("encoder.blocks." + layers_i + ".cross_attn.query.weight", {n_embd, n_embd}, backend);
    layer.ffn[3] = ml->get_tensor("encoder.blocks." + layers_i + ".cross_attn.query.bias", {n_embd}, backend);
    layer.ffn[0] = ml->get_tensor("encoder.blocks." + layers_i + ".attn.out.bias", {n_embd}, backend);

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

class whisper_quant_layer : public quant_layer_base {
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
REGISTER_QUANT_LAYER_CLASS(whisper);
