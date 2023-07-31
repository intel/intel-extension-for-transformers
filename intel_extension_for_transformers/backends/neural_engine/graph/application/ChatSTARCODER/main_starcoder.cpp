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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "core/ne_layers.h"
#include "common.h"
#include "data_types.h"
#include "ne.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data //TODO
#endif

// default hparams (santacoder GPT-2 117M)
// https://huggingface.co/bigcode/gpt_bigcode-santacoder/blob/main/config.json
// another hparams (StarCoder 15.5B)
// https://huggingface.co/bigcode/starcoder/blob/main/config.json
struct starcoder_hparams {
  int32_t n_vocab = 49280;
  int32_t n_ctx = 2048;
  int32_t n_embd = 2048;
  int32_t n_head = 16;
  int32_t n_layer = 24;
  int32_t ftype = 1;
};

struct starcoder_layer {
  // normalization
  struct ne_tensor* ln_1_g;
  struct ne_tensor* ln_1_b;

  struct ne_tensor* ln_2_g;
  struct ne_tensor* ln_2_b;

  // attention
  struct ne_tensor* c_attn_attn_w;
  struct ne_tensor* c_attn_attn_b;

  struct ne_tensor* c_attn_proj_w;
  struct ne_tensor* c_attn_proj_b;

  // mlp
  struct ne_tensor* c_mlp_fc_w;
  struct ne_tensor* c_mlp_fc_b;

  struct ne_tensor* c_mlp_proj_w;
  struct ne_tensor* c_mlp_proj_b;
};

struct starcoder_model {
  starcoder_hparams hparams;

  // normalization
  struct ne_tensor* ln_f_g;
  struct ne_tensor* ln_f_b;

  struct ne_tensor* wte;      // position embedding
  struct ne_tensor* wpe;      // token embedding
  struct ne_tensor* lm_head;  // language model head

  std::vector<starcoder_layer> layers;

  // key + value memory
  struct ne_tensor* memory_k;
  struct ne_tensor* memory_v;

  struct ne_context* ctx;
  std::map<std::string, struct ne_tensor*> tensors;
};

// load the model's weights from a file
bool starcoder_model_load(const std::string& fname, starcoder_model& model, gpt_vocab& vocab) {
  printf("%s: loading model from '%s'\n", __func__, fname.c_str());

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

    fin.read((char*)&hparams.n_vocab, sizeof(hparams.n_vocab));
    fin.read((char*)&hparams.n_ctx, sizeof(hparams.n_ctx));
    fin.read((char*)&hparams.n_embd, sizeof(hparams.n_embd));
    fin.read((char*)&hparams.n_head, sizeof(hparams.n_head));
    fin.read((char*)&hparams.n_layer, sizeof(hparams.n_layer));
    fin.read((char*)&hparams.ftype, sizeof(hparams.ftype));

    const int32_t qntvr = hparams.ftype / NE_QNT_VERSION_FACTOR;

    printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
    printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
    printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
    printf("%s: n_head  = %d\n", __func__, hparams.n_head);
    printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
    printf("%s: ftype   = %d\n", __func__, hparams.ftype);
    printf("%s: qntvr   = %d\n", __func__, qntvr);

    hparams.ftype %= NE_QNT_VERSION_FACTOR;
  }

  // load vocab
  {
    int32_t n_vocab = 0;
    fin.read((char*)&n_vocab, sizeof(n_vocab));

    if (n_vocab != model.hparams.n_vocab) {
      fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n", __func__, fname.c_str(), n_vocab,
              model.hparams.n_vocab);
      return false;
    }

    std::string word;
    std::vector<char> buf(128);

    for (int i = 0; i < n_vocab; i++) {
      uint32_t len;
      fin.read((char*)&len, sizeof(len));

      buf.resize(len);
      fin.read((char*)buf.data(), len);
      word.assign(buf.data(), len);

      vocab.token_to_id[word] = i;
      vocab.id_to_token[i] = word;
    }

    // Add StarChat special tokens.
    for (const std::string& token : {"<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<fim-prefix>",
                                     "<fim-middle>", "<fim-suffix>", "<fim-pad>", "<|end_of_turn|>", "<|endoftext|>"}) {
      if (vocab.token_to_id.find(token) != vocab.token_to_id.end()) {
        vocab.add_special_token(token);
      }
    }
  }

  // for the big tensors, we have the option to store the data in 16-bit floats or quantized
  // in order to save memory and also to speed up the computation
  ne_type wtype = ne_ftype_to_ne_type((ne_ftype)(model.hparams.ftype));
  if (wtype == NE_TYPE_COUNT) {
    fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n", __func__, fname.c_str(), model.hparams.ftype);
    return false;
  }

  auto& ctx = model.ctx;

  size_t ctx_size = 0;

  {
    const auto& hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;

    const int head_dim = n_embd / hparams.n_head;
    const int kv_heads = hparams.n_head;  // 1 if MQA else hparams.n_head
    const int kv_dim = kv_heads * head_dim;

    ctx_size += n_embd * ne_type_sizef(NE_TYPE_F32);  // ln_f_g
    ctx_size += n_embd * ne_type_sizef(NE_TYPE_F32);  // ln_f_b

    ctx_size += n_vocab * n_embd * ne_type_sizef(NE_TYPE_F32);  // wte
    ctx_size += n_ctx * n_embd * ne_type_sizef(NE_TYPE_F32);    // wpe
    ctx_size += n_vocab * n_embd * ne_type_sizef(NE_TYPE_F32);  // lm_head

    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));  // ln_1_g
    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));  // ln_1_b

    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));  // ln_2_g
    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));  // ln_2_b

    ctx_size += n_layer * ((n_embd + 2 * kv_dim) * n_embd * ne_type_sizef(wtype));  // c_attn_attn_w
    ctx_size += n_layer * ((n_embd + 2 * kv_dim) * ne_type_sizef(NE_TYPE_F32));     // c_attn_attn_b

    ctx_size += n_layer * (n_embd * n_embd * ne_type_sizef(wtype));  // c_attn_proj_w
    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));     // c_attn_proj_b

    ctx_size += n_layer * (4 * n_embd * n_embd * ne_type_sizef(wtype));  // c_mlp_fc_w
    ctx_size += n_layer * (4 * n_embd * ne_type_sizef(NE_TYPE_F32));     // c_mlp_fc_b

    ctx_size += n_layer * (4 * n_embd * n_embd * ne_type_sizef(wtype));  // c_mlp_proj_w
    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));         // c_mlp_proj_b

    ctx_size += n_ctx * n_layer * n_embd * ne_type_sizef(NE_TYPE_F32);  // memory_k
    ctx_size += n_ctx * n_layer * n_embd * ne_type_sizef(NE_TYPE_F32);  // memory_v

    ctx_size += (6 + 12 * n_layer) * 512;  // object overhead

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

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;

    const int head_dim = n_embd / hparams.n_head;
    const int kv_heads = hparams.n_head;  // 1 if MQA else hparams.n_head
    const int kv_dim = kv_heads * head_dim;

    model.layers.resize(n_layer);

    model.ln_f_g = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
    model.ln_f_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);

    // do not quant vocab related weights
    model.wte = ne_new_tensor_2d(ctx, NE_TYPE_F32, n_embd, n_vocab, NE_SIZE_CALC);
    model.wpe = ne_new_tensor_2d(ctx, NE_TYPE_F32, n_embd, n_ctx, NE_SIZE_CALC);
    model.lm_head = ne_new_tensor_2d(ctx, NE_TYPE_F32, n_embd, n_vocab, NE_SIZE_CALC);

    // map by name
    model.tensors["model/ln_f/g"] = model.ln_f_g;
    model.tensors["model/ln_f/b"] = model.ln_f_b;

    model.tensors["model/wte"] = model.wte;
    model.tensors["model/wpe"] = model.wpe;
    model.tensors["model/lm_head"] = model.lm_head;

    for (int i = 0; i < n_layer; ++i) {
      auto& layer = model.layers[i];

      layer.ln_1_g = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
      layer.ln_1_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);

      layer.ln_2_g = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
      layer.ln_2_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);

      layer.c_attn_attn_w = ne_new_tensor_2d(ctx, wtype, n_embd, n_embd + 2 * kv_dim, NE_SIZE_CALC);
      layer.c_attn_attn_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd + 2 * kv_dim, NE_SIZE_CALC);

      layer.c_attn_proj_w = ne_new_tensor_2d(ctx, wtype, n_embd, n_embd, NE_SIZE_CALC);
      layer.c_attn_proj_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);

      layer.c_mlp_fc_w = ne_new_tensor_2d(ctx, wtype, n_embd, 4 * n_embd, NE_SIZE_CALC);
      layer.c_mlp_fc_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, 4 * n_embd, NE_SIZE_CALC);

      layer.c_mlp_proj_w = ne_new_tensor_2d(ctx, wtype, 4 * n_embd, n_embd, NE_SIZE_CALC);
      layer.c_mlp_proj_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);

      // map by name
      model.tensors["model/h" + std::to_string(i) + "/ln_1/g"] = layer.ln_1_g;
      model.tensors["model/h" + std::to_string(i) + "/ln_1/b"] = layer.ln_1_b;

      model.tensors["model/h" + std::to_string(i) + "/ln_2/g"] = layer.ln_2_g;
      model.tensors["model/h" + std::to_string(i) + "/ln_2/b"] = layer.ln_2_b;

      model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
      model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

      model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
      model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

      model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"] = layer.c_mlp_fc_w;
      model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"] = layer.c_mlp_fc_b;

      model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"] = layer.c_mlp_proj_w;
      model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"] = layer.c_mlp_proj_b;
    }
  }

  // key + value memory
  {
    const auto& hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;

    const int n_mem = n_layer * n_ctx;
    const int n_elements = n_embd * n_mem;  // TODO mem overhead in multi-query

    model.memory_k = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_elements, NE_SIZE_CALC);
    model.memory_v = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_elements, NE_SIZE_CALC);

    const size_t memory_size = ne_nbytes(model.memory_k) + ne_nbytes(model.memory_v);

    printf("%s: kv cache memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size / 1024.0 / 1024.0, n_mem);
  }

  // load weights
  {
    int n_tensors = 0;
    size_t total_size = 0;

    printf("%s: ", __func__);

    bool has_lm_head = false;

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

      if (model.tensors.find(name.data()) == model.tensors.end()) {
        fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
        return false;
      }

      auto tensor = model.tensors[name.data()];
      if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
        fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n", __func__,
                name.data(), (int)tensor->ne[0], (int)tensor->ne[1], ne[0], ne[1]);
        return false;
      }
      if (ne_nelements(tensor) != nelements) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file. got %d, expected %d\n", __func__, name.data(),
                (int)ne_nelements(tensor), nelements);
        return false;
      }

      // for debugging
      if (0) {
        printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1],
               ne_type_name(ne_type(ttype)), ne_nbytes(tensor) / 1024.0 / 1024.0, ne_nbytes(tensor));
      }

      const size_t bpe = ne_type_size(ne_type(ttype));

      if ((nelements * bpe) / ne_blck_size(tensor->type) != ne_nbytes(tensor)) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n", __func__, name.data(),
                ne_nbytes(tensor), nelements * bpe);
        return false;
      }

      fin.read(reinterpret_cast<char*>(tensor->data), ne_nbytes(tensor));

      // GPT-2 models share the WTE tensor as the LM head
      if (name == "model/wte" && has_lm_head == false) {
        memcpy(model.lm_head->data, tensor->data, ne_nbytes(tensor));
      }

      if (name == "model/lm_head") {
        has_lm_head = true;
      }

      total_size += ne_nbytes(tensor);
      if (++n_tensors % 8 == 0) {
        printf(".");
        fflush(stdout);
      }
    }

    printf(" done\n");
    printf("%s: model size  = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
  }

  fin.close();

  return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool starcoder_eval(const starcoder_model& model, const int n_threads, const int n_past,
                    const std::vector<gpt_vocab::id>& embd_inp, std::vector<float>& embd_w, size_t& mem_per_token) {
  const int N = embd_inp.size();

  const auto& hparams = model.hparams;

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = hparams.n_ctx;
  const int n_head = hparams.n_head;
  const int n_vocab = hparams.n_vocab;

  static size_t buf_size = 256u * 1024 * 1024;
  static void* buf = malloc(buf_size);

  // use 2 scratch buffers
  // TODO: very hacky solution - reimplement in a more elegant way
  static size_t scr0_size = 256u * 1024 * 1024;
  static void* scr0 = malloc(scr0_size);

  static size_t scr1_size = 256u * 1024 * 1024;
  static void* scr1 = malloc(scr1_size);

  if (mem_per_token > 0 && mem_per_token * N > buf_size) {
    const size_t buf_size_new = 1.1 * (mem_per_token * N);  // add 10% to account for ne object overhead
    // printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

    // reallocate
    buf_size = buf_size_new;
    buf = realloc(buf, buf_size);
    if (buf == nullptr) {
      fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
      return false;
    }
  }

  struct ne_init_params params = {
      .mem_size = buf_size,
      .mem_buffer = buf,
      .no_alloc = false,
  };

  struct ne_context* ctx0 = ne_init(params);
  struct ne_cgraph gf = {};
  gf.n_threads = n_threads;

  struct ne_tensor* embd = ne_new_tensor_1d(ctx0, NE_TYPE_I32, N, NE_SIZE_CALC);
  memcpy(embd->data, embd_inp.data(), N * ne_element_size(embd));

  struct ne_tensor* position = ne_new_tensor_1d(ctx0, NE_TYPE_I32, N, NE_SIZE_CALC);
  for (int i = 0; i < N; ++i) {
    ((int32_t*)position->data)[i] = n_past + i;
  }

  // wte + wpe
  struct ne_tensor* inpL = ne_add(ctx0, ne_get_rows(ctx0, model.wte, embd), ne_get_rows(ctx0, model.wpe, position));

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;

    ne_set_scratch(ctx0, {
                             0,
                             scr0_size,
                             scr0,
                         });

    // norm
    {
      // [ 768, N]
      cur = ne_norm(ctx0, inpL);

      // cur = ln_1_g*cur + ln_1_b
      // [ 768, N]
      cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].ln_1_g, cur), cur),
                   ne_repeat(ctx0, model.layers[il].ln_1_b, cur));
    }

    // attn
    // [2304, 768] - model.layers[il].c_attn_attn_w
    // [2304,   1] - model.layers[il].c_attn_attn_b
    // [ 768,   N] - cur (in)
    // [2304,   N] - cur (out)
    //
    // cur = attn_w*cur + attn_b
    // [2304, N]
    {
      cur = ne_mul_mat(ctx0, model.layers[il].c_attn_attn_w, cur);

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].c_attn_attn_b, cur), cur);
    }

    // self-attention
    {
      size_t fused_qkv_row_nb = (3 * n_embd) * sizeof(float);
      size_t head_dim = n_embd / n_head;
      struct ne_tensor* Qcur = ne_view_3d(ctx0, cur, head_dim, n_head, N, head_dim * sizeof(float), fused_qkv_row_nb,
                                          0 * sizeof(float) * n_embd);
      // head_dim, n_head, N --> head_dim, N, n_head
      struct ne_tensor* Kcur = ne_permute(ctx0,
                                          ne_view_3d(ctx0, cur, head_dim, n_head, N, head_dim * sizeof(float),
                                                     fused_qkv_row_nb, 1 * sizeof(float) * n_embd),
                                          0, 2, 1, 3);
      // head_dim, n_head, N --> N, head_dim, n_head
      struct ne_tensor* Vcur = ne_permute(ctx0,
                                          ne_view_3d(ctx0, cur, head_dim, n_head, N, head_dim * sizeof(float),
                                                     fused_qkv_row_nb, 2 * sizeof(float) * n_embd),
                                          1, 2, 0, 3);

      // store transposed key and value to memory (k_v cache)
      if (N >= 1) {
        // n_embd / n_head as col
        struct ne_tensor* k = ne_view_3d(ctx0, model.memory_k, n_embd / n_head, N, n_head,
                                         ne_element_size(model.memory_k) * n_embd / n_head,
                                         ne_element_size(model.memory_k) * n_embd / n_head * n_ctx,
                                         il * n_ctx * ne_element_size(model.memory_k) * n_embd +
                                             n_past * ne_element_size(model.memory_k) * n_embd / n_head);
        // N as col, n_embd as row
        struct ne_tensor* v = ne_view_3d(
            ctx0, model.memory_v, N, n_embd / n_head, n_head, n_ctx * ne_element_size(model.memory_v),
            n_ctx * ne_element_size(model.memory_v) * head_dim,
            il * n_ctx * ne_element_size(model.memory_v) * n_embd + n_past * ne_element_size(model.memory_v));
        // concat
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur, k));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur, v));
      }

      // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
      // [64, N, 12]
      struct ne_tensor* Q = ne_permute(ctx0, Qcur, 0, 2, 1, 3);

      // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
      // [64, n_past + N, 12]
      struct ne_tensor* K = ne_view_3d(ctx0, model.memory_k, n_embd / n_head, N + n_past, n_head,
                                       ne_element_size(model.memory_k) * n_embd / n_head,
                                       ne_element_size(model.memory_k) * n_embd / n_head * n_ctx,
                                       il * n_ctx * ne_element_size(model.memory_k) * n_embd);

      // GG: flash attention
      // struct ne_tensor * V =
      //    ne_cpy(ctx0,
      //            ne_permute(ctx0,
      //                ne_reshape_3d(ctx0,
      //                    ne_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd,
      //                    il*n_ctx*ne_element_size(model.memory_v)*n_embd), n_embd/n_head, n_head, n_past + N),
      //                1, 2, 0, 3),
      //            ne_new_tensor_3d(ctx0, NE_TYPE_F32, n_past + N, n_embd/n_head, n_head, NE_SIZE_CALC));

      // struct ne_tensor * KQV = ne_flash_attn(ctx0, Q, K, V, true);

      // K * Q
      // [n_past + N, N, 12]
      struct ne_tensor* KQ = ne_mul_mat(ctx0, K, Q);  // TODO: check if it broadcasts

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      // [n_past + N, N, 12]
      struct ne_tensor* KQ_scaled = ne_scale_inplace(ctx0, KQ, ne_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head)));

      // KQ_masked = mask_past(KQ_scaled)
      // [n_past + N, N, 12]
      struct ne_tensor* KQ_masked = ne_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

      // KQ = soft_max(KQ_masked)
      // [n_past + N, N, 12]
      struct ne_tensor* KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_masked);

      // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
      // [n_past + N, 64, 12]
      struct ne_tensor* V_trans =
          ne_view_3d(ctx0, model.memory_v, N + n_past, n_embd / n_head, n_head, n_ctx * ne_element_size(model.memory_v),
                     n_ctx * ne_element_size(model.memory_v) * n_embd / n_head,
                     il * n_ctx * ne_element_size(model.memory_v) * n_embd);

      // KQV = transpose(V) * KQ_soft_max
      // [64, N, 12]
      struct ne_tensor* KQV = ne_mul_mat(ctx0, V_trans, KQ_soft_max);

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      // [64, 12, N]
      struct ne_tensor* KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);

      // cur = KQV_merged.contiguous().view(n_embd, N)
      // [768, N]
      cur = ne_cpy(ctx0, KQV_merged, ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N, NE_SIZE_CALC));
    }

    // projection
    // [ 768, 768] - model.layers[il].c_attn_proj_w
    // [ 768,   1] - model.layers[il].c_attn_proj_b
    // [ 768,   N] - cur (in)
    // [ 768,   N] - cur (out)
    //
    // cur = proj_w*cur + proj_b
    // [768, N]
    {
      cur = ne_mul_mat(ctx0, model.layers[il].c_attn_proj_w, cur);

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].c_attn_proj_b, cur), cur);
    }

    // add the input
    cur = ne_add(ctx0, cur, inpL);

    struct ne_tensor* inpFF = cur;

    ne_set_scratch(ctx0, {
                             0,
                             scr1_size,
                             scr1,
                         });

    // feed-forward network
    {
      // norm
      {
        cur = ne_norm(ctx0, inpFF);

        // cur = ln_2_g*cur + ln_2_b
        // [ 768, N]
        cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].ln_2_g, cur), cur),
                     ne_repeat(ctx0, model.layers[il].ln_2_b, cur));
      }

      // fully connected
      // [3072, 768] - model.layers[il].c_mlp_fc_w
      // [3072,   1] - model.layers[il].c_mlp_fc_b
      // [ 768,   N] - cur (in)
      // [3072,   N] - cur (out)
      //
      // cur = fc_w*cur + fc_b
      // [3072, N]
      cur = ne_mul_mat(ctx0, model.layers[il].c_mlp_fc_w, cur);

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur), cur);

      // GELU activation
      // [3072, N]
      cur = ne_gelu(ctx0, cur);

      // projection
      // [ 768, 3072] - model.layers[il].c_mlp_proj_w
      // [ 768,    1] - model.layers[il].c_mlp_proj_b
      // [3072,    N] - cur (in)
      // [ 768,    N] - cur (out)
      //
      // cur = proj_w*cur + proj_b
      // [768, N]
      cur = ne_mul_mat(ctx0, model.layers[il].c_mlp_proj_w, cur);

      cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur), cur);
    }

    // input for next layer
    inpL = ne_add(ctx0, cur, inpFF);
  }

  ne_set_scratch(ctx0, {
                           0,
                           scr0_size,
                           scr0,
                       });

  // norm
  {
    // [ 768, N]
    inpL = ne_norm(ctx0, inpL);

    // inpL = ln_f_g*inpL + ln_f_b
    // [ 768, N]
    inpL = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.ln_f_g, inpL), inpL), ne_repeat(ctx0, model.ln_f_b, inpL));
  }

  ne_set_scratch(ctx0, {
                           0,
                           0,
                           nullptr,
                       });

  // inpL = WTE * inpL
  // [ 768, 50257] - model.lm_head
  // [ 768, N]     - inpL
  inpL = ne_mul_mat(ctx0, model.lm_head, inpL);

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
  // if (n_past%100 == 0) {
  //     ne_graph_print   (&gf);
  //     ne_graph_dump_dot(&gf, NULL, "gpt-2.dot");
  // }

  // embd_w.resize(n_vocab*N);
  // memcpy(embd_w.data(), ne_get_data(inpL), sizeof(float)*n_vocab*N);

  // return result just for the last token
  embd_w.resize(n_vocab);
  memcpy(embd_w.data(), (float*)ne_get_data(inpL) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);

  if (mem_per_token == 0) {
    mem_per_token = ne_used_mem(ctx0) / N;
  }
  // printf("used_mem = %zu MB\n", ne_used_mem(ctx0)/(1024*1024));

  ne_free(ctx0);

  return true;
}

int main(int argc, char** argv) {
  ne_time_init();

  const int64_t t_main_start_us = ne_time_us();

  common_params params;
  params.model = "models/ne-model.bin";

  if (common_params_parse(argc, argv, params) == false) {
    return 1;
  }

  if (params.seed < 0) {
    params.seed = time(NULL);
  }

  if (params.n_predict < 0) {
    params.n_predict = 0;
  }

  std::mt19937 rng(params.seed);
  if (params.prompt.empty()) {
    params.prompt = gpt_random_prompt(rng);
  }

  int64_t t_load_us = 0;

  gpt_vocab vocab;
  starcoder_model model;

  // load the model
  {
    const int64_t t_start_us = ne_time_us();

    if (!starcoder_model_load(params.model, model, vocab)) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
      return 1;
    }

    t_load_us = ne_time_us() - t_start_us;

    test_gpt_tokenizer(vocab, params.token_test);
  }

  if (params.repeat_last_n == -1) {
    params.repeat_last_n = model.hparams.n_ctx;
  }

  if (params.top_k == 0) {
    params.top_k = model.hparams.n_vocab;
  }

  printf("\n");
  printf("\n");
  printf("%s: seed           = %d\n", __func__, params.seed);
  printf("%s: n_threads      = %d\n", __func__, params.n_threads);
  printf("%s: n_batch        = %d\n", __func__, params.n_batch);
  printf("%s: n_ctx          = %d\n", __func__, params.n_ctx);
  printf("%s: n_predict      = %d\n", __func__, params.n_predict);
  printf("%s: temp           = %.3f\n", __func__, params.temp);
  printf("%s: top_k          = %d\n", __func__, params.top_k);
  printf("%s: top_p          = %.3f\n", __func__, params.top_p);
  printf("%s: repeat_last_n  = %d\n", __func__, params.repeat_last_n);
  printf("%s: repeat_penalty = %.3f\n\n", __func__, params.repeat_penalty);

  int n_past = 0;

  int64_t t_sample_us = 0;
  int64_t t_predict_us = 0;

  std::vector<float> logits;

  std::vector<int32_t> last_n_tokens(model.hparams.n_ctx);
  std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

  // tokenize the prompt
  std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

  params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int)embd_inp.size());

  printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
  printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
  for (int i = 0; i < embd_inp.size(); i++) {
    printf("%s: token[%d] = %6d, %s\n", __func__, i, embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
  }
  printf("\n");

  // Handle StarChat "<|end|>", "<|endoftext|>" and OpenCoder "<|end_of_turn>" tokens.
  gpt_vocab::id starchat_end_token = -1;
  {
    std::vector<std::string> end_tokens = {"<|end|>", "<|end_of_turn|>", "<|endoftext|>"};
    for (const auto& t : end_tokens) {
      const auto it = vocab.token_to_id.find(t);
      if (it != vocab.token_to_id.end()) {
        starchat_end_token = it->second;
        break;
      }
    }
  }

  // submit the input prompt token-by-token
  // this reduces the memory usage during inference, at the cost of a bit of speed at the beginning
  std::vector<gpt_vocab::id> embd;

  // determine the required inference memory per token:
  size_t mem_per_token = 0;
  std::vector<int64_t> eval_times;
  starcoder_eval(model, params.n_threads, 0, {0, 1, 2, 3}, logits, mem_per_token);

  bool first_token = true;
  for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
    // predict
    if (embd.size() > 0) {
      const int64_t t_start_us = ne_time_us();

      if (!starcoder_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
        printf("Failed to predict\n");
        return 1;
      }
      // make first-token as warmup token
      int64_t time_interval = ne_time_us() - t_start_us;
      if (first_token) {
        first_token = false;
        eval_times.push_back(time_interval);
      } else {
        t_predict_us += time_interval;
        eval_times.push_back(time_interval);
      }
    }

    n_past += embd.size();
    embd.clear();

    if (i >= embd_inp.size()) {
      // sample next token
      const int top_k = params.top_k;
      const float top_p = params.top_p;
      const float temp = params.temp;

      const int n_vocab = model.hparams.n_vocab;

      gpt_vocab::id id = 0;

      {
        const int64_t t_start_sample_us = ne_time_us();

        id = gpt_sample_top_k_top_p_repeat(vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens.data(),
                                           last_n_tokens.size(), top_k, top_p, temp, params.repeat_last_n,
                                           params.repeat_penalty, rng);
        t_sample_us += ne_time_us() - t_start_sample_us;
      }

      // add it to the context
      embd.push_back(id);

      last_n_tokens.erase(last_n_tokens.begin());
      last_n_tokens.push_back(id);
    } else {
      // if here, it means we are still processing the input prompt
      for (int k = i; k < embd_inp.size(); k++) {
        embd.push_back(embd_inp[k]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(embd_inp[k]);

        if (embd.size() >= params.n_batch) {
          break;
        }
      }
      i += embd.size() - 1;
    }

    // display text
    for (auto id : embd) {
      printf("%s", vocab.id_to_token[id].c_str());
    }
    fflush(stdout);

    // check if model is santacoder
    if (model.hparams.n_layer <= 30 && embd.back() == 49152) {
      break;
    }
    // check if model is starcoder
    else if (embd.back() == 0) {  // TODO: this is only for starcoder
      break;
    }
    // Handle StarChat "<|end|>" token.
    else if (embd.back() == starchat_end_token && i >= embd_inp.size()) {
      break;
    }
  }

  // report timing
  {
    const int64_t t_main_end_us = ne_time_us();

    printf("\n\n");
    printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
    printf("%s: load time     = %8.2f ms\n", __func__, t_load_us / 1000.0f);
    printf("%s: sample time   = %8.2f ms\n", __func__, t_sample_us / 1000.0f);
    printf("%s: predict time  = %8.2f ms / %d, %.2f ms per token\n", __func__, t_predict_us / 1000.0f,
           params.n_predict - 1, t_predict_us / 1000.0f / (params.n_predict - 1));
    printf("%s: total time    = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    printf("========== eval time log of each prediction ==========\n");
    for (int i = 0; i < eval_times.size(); ++i) {
      printf("prediction %3d, time: %.2f ms\n", i, eval_times[i] / 1000.0f);
    }
  }

  ne_free(model.ctx);

  return 0;
}
