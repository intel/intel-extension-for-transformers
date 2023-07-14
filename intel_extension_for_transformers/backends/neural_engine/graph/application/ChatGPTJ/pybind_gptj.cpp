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
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "core/ne_layers.h"
#include "data_types.h"
#include "ne.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

#define N_thread 56

// default hparams (GPT-J 6B)
struct gptj_hparams {
  int32_t n_vocab = 50400;
  int32_t n_ctx = 2048;
  int32_t n_embd = 4096;
  int32_t n_head = 16;
  int32_t n_layer = 28;
  int32_t n_rot = 64;
  int32_t ftype = 1;
};

struct gptj_layer {
  // normalization
  struct ne_tensor* ln_1_g;
  struct ne_tensor* ln_1_b;

  // attention
  struct ne_tensor* c_attn_q_proj_w;
  struct ne_tensor* c_attn_k_proj_w;
  struct ne_tensor* c_attn_v_proj_w;

  struct ne_tensor* c_attn_proj_w;

  // ff
  struct ne_tensor* c_mlp_fc_w;
  struct ne_tensor* c_mlp_fc_b;

  struct ne_tensor* c_mlp_proj_w;
  struct ne_tensor* c_mlp_proj_b;
};

struct gptj_model {
  gptj_hparams hparams;

  // normalization
  struct ne_tensor* ln_f_g;
  struct ne_tensor* ln_f_b;

  struct ne_tensor* wte;  // position embedding

  struct ne_tensor* lmh_g;  // language model head
  struct ne_tensor* lmh_b;  // language model bias

  std::vector<gptj_layer> layers;

  // key + value memory
  struct ne_tensor* memory_k;
  struct ne_tensor* memory_v;

  //
  struct ne_context* ctx;
  std::map<std::string, struct ne_tensor*> tensors;

  ~gptj_model() { ne_free(ctx); }
};

struct gptj_all {
  struct gptj_model* model;
  struct gpt_vocab* vocab;
  ~gptj_all() {
    delete model;
    delete vocab;
  }
};

// load the model's weights from a file
bool gptj_model_load(const std::string& fname, gptj_model& model, gpt_vocab& vocab) {
  printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

  auto fin = std::ifstream(fname, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    return false;
  }

  // verify magic
  uint32_t magic;
  fin.read((char*)&magic, sizeof(magic));
  if (magic != 0x67676d6c) {
    fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
    return false;
  }

  // load hparams
  auto& hparams = model.hparams;

  fin.read((char*)&hparams.n_vocab, sizeof(hparams.n_vocab));
  fin.read((char*)&hparams.n_ctx, sizeof(hparams.n_ctx));
  fin.read((char*)&hparams.n_embd, sizeof(hparams.n_embd));
  fin.read((char*)&hparams.n_head, sizeof(hparams.n_head));
  fin.read((char*)&hparams.n_layer, sizeof(hparams.n_layer));
  fin.read((char*)&hparams.n_rot, sizeof(hparams.n_rot));
  fin.read((char*)&hparams.ftype, sizeof(hparams.ftype));

  const int32_t qntvr = hparams.ftype / NE_QNT_VERSION_FACTOR;

  printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
  printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
  printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
  printf("%s: n_head  = %d\n", __func__, hparams.n_head);
  printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
  printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
  printf("%s: ftype   = %d\n", __func__, hparams.ftype);
  printf("%s: qntvr   = %d\n", __func__, qntvr);

  hparams.ftype %= NE_QNT_VERSION_FACTOR;

  // load vocab
  int32_t file_n_vocab = 0;
  fin.read((char*)&file_n_vocab, sizeof(file_n_vocab));

  if (file_n_vocab != model.hparams.n_vocab) {
    fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n", __func__, fname.c_str(), file_n_vocab,
            model.hparams.n_vocab);
    return false;
  }

  std::string word;
  std::vector<char> buf(128);

  for (int i = 0; i < file_n_vocab; i++) {
    uint32_t len;
    fin.read((char*)&len, sizeof(len));

    buf.resize(len);
    fin.read((char*)buf.data(), len);
    word.assign(buf.data(), len);

    vocab.token_to_id[word] = i;
    vocab.id_to_token[i] = word;
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

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = hparams.n_ctx;
  const int n_vocab = hparams.n_vocab;

  ctx_size += n_embd * ne_type_sizef(NE_TYPE_F32);  // ln_f_g
  ctx_size += n_embd * ne_type_sizef(NE_TYPE_F32);  // ln_f_b

  ctx_size += n_embd * n_vocab * ne_type_sizef(wtype);  // wte

  ctx_size += n_embd * n_vocab * ne_type_sizef(wtype);  // lmh_g
  ctx_size += n_vocab * ne_type_sizef(NE_TYPE_F32);     // lmh_b

  ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));  // ln_1_g
  ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));  // ln_1_b

  ctx_size += n_layer * (n_embd * n_embd * ne_type_sizef(wtype));  // c_attn_q_proj_w
  ctx_size += n_layer * (n_embd * n_embd * ne_type_sizef(wtype));  // c_attn_k_proj_w
  ctx_size += n_layer * (n_embd * n_embd * ne_type_sizef(wtype));  // c_attn_v_proj_w

  ctx_size += n_layer * (n_embd * n_embd * ne_type_sizef(wtype));  // c_attn_proj_w

  ctx_size += n_layer * (4 * n_embd * n_embd * ne_type_sizef(wtype));  // c_mlp_fc_w
  ctx_size += n_layer * (4 * n_embd * ne_type_sizef(NE_TYPE_F32));     // c_mlp_fc_b

  ctx_size += n_layer * (4 * n_embd * n_embd * ne_type_sizef(wtype));  // c_mlp_proj_w
  ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));         // c_mlp_proj_b

  ctx_size += n_ctx * n_layer * n_embd * ne_type_sizef(NE_TYPE_F16);  // memory_k
  ctx_size += n_ctx * n_layer * n_embd * ne_type_sizef(NE_TYPE_F16);  // memory_v

  ctx_size += (5 + 10 * n_layer) * 512;  // object overhead

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

  // prepare memory for the weights

  model.layers.resize(n_layer);

  model.wte = d_ne_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

  model.ln_f_g = d_ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd);
  model.ln_f_b = d_ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd);

  model.lmh_g = d_ne_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
  model.lmh_b = d_ne_new_tensor_1d(ctx, NE_TYPE_F32, n_vocab);

  // map by name
  model.tensors["transformer.wte.weight"] = model.wte;

  model.tensors["transformer.ln_f.weight"] = model.ln_f_g;
  model.tensors["transformer.ln_f.bias"] = model.ln_f_b;

  model.tensors["lm_head.weight"] = model.lmh_g;
  model.tensors["lm_head.bias"] = model.lmh_b;

  for (int i = 0; i < n_layer; ++i) {
    auto& layer = model.layers[i];

    layer.ln_1_g = d_ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd);
    layer.ln_1_b = d_ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd);

    layer.c_attn_q_proj_w = d_ne_new_tensor_2d(ctx, wtype, n_embd, n_embd);
    layer.c_attn_k_proj_w = d_ne_new_tensor_2d(ctx, wtype, n_embd, n_embd);
    layer.c_attn_v_proj_w = d_ne_new_tensor_2d(ctx, wtype, n_embd, n_embd);

    layer.c_attn_proj_w = d_ne_new_tensor_2d(ctx, wtype, n_embd, n_embd);

    layer.c_mlp_fc_w = d_ne_new_tensor_2d(ctx, wtype, n_embd, 4 * n_embd);
    layer.c_mlp_fc_b = d_ne_new_tensor_1d(ctx, NE_TYPE_F32, 4 * n_embd);

    layer.c_mlp_proj_w = d_ne_new_tensor_2d(ctx, wtype, 4 * n_embd, n_embd);
    layer.c_mlp_proj_b = d_ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd);

    // map by name
    model.tensors["transformer.h." + std::to_string(i) + ".ln_1.weight"] = layer.ln_1_g;
    model.tensors["transformer.h." + std::to_string(i) + ".ln_1.bias"] = layer.ln_1_b;

    model.tensors["transformer.h." + std::to_string(i) + ".attn.q_proj.weight"] = layer.c_attn_q_proj_w;
    model.tensors["transformer.h." + std::to_string(i) + ".attn.k_proj.weight"] = layer.c_attn_k_proj_w;
    model.tensors["transformer.h." + std::to_string(i) + ".attn.v_proj.weight"] = layer.c_attn_v_proj_w;

    model.tensors["transformer.h." + std::to_string(i) + ".attn.out_proj.weight"] = layer.c_attn_proj_w;

    model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_in.weight"] = layer.c_mlp_fc_w;
    model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_in.bias"] = layer.c_mlp_fc_b;

    model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_out.weight"] = layer.c_mlp_proj_w;
    model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_out.bias"] = layer.c_mlp_proj_b;
  }

  // key + value memory
  const int n_mem = n_layer * n_ctx;
  const int n_elements = n_embd * n_mem;
  model.memory_k = d_ne_new_tensor_1d(ctx, NE_TYPE_F16, n_elements);
  model.memory_v = d_ne_new_tensor_1d(ctx, NE_TYPE_F16, n_elements);

  const size_t memory_size = ne_nbytes(model.memory_k) + ne_nbytes(model.memory_v);

  printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size / 1024.0 / 1024.0, n_mem);

  // load weights
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

    if (model.tensors.find(name.data()) == model.tensors.end()) {
      fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
      return false;
    }

    auto tensor = model.tensors[name.data()];
    if (ne_nelements(tensor) != nelements) {
      fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
      return false;
    }

    if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
      fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n", __func__,
              name.data(), (int)tensor->ne[0], (int)tensor->ne[1], ne[0], ne[1]);
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

    // printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ttype == 0 ? "float" : "f16",
    // ne_nbytes(tensor)/1024.0/1024.0);
    total_size += ne_nbytes(tensor);
    if (++n_tensors % 8 == 0) {
      printf(".");
      fflush(stdout);
    }
  }
  printf(" done\n");
  printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);

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
// The GPT-J model requires about 16MB of memory per input token.
//
bool gptj_eval(const gptj_model& model, const int n_threads, const int n_past,
               const std::vector<gpt_vocab::id>& embd_inp, std::vector<float>& embd_w, size_t& mem_per_token) {
  const int N = embd_inp.size();

  const auto& hparams = model.hparams;

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = hparams.n_ctx;
  const int n_head = hparams.n_head;
  const int n_vocab = hparams.n_vocab;
  const int n_rot = hparams.n_rot;

  static size_t buf_size = 256u * 1024 * 1024;
  static void* buf = malloc(buf_size);

  if (mem_per_token > 0 && mem_per_token * N > buf_size) {
    const size_t buf_size_new = 1.1 * (mem_per_token * N);  // add 10% to account for ggml object overhead

    // reallocate
    buf_size = buf_size_new;
    buf = realloc(buf, buf_size);
    if (buf == nullptr) {
      fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
      return false;
    }
  }

  struct ne_init_params params = {
      /*.mem_size   =*/buf_size,
      /*.mem_buffer =*/buf,
      /*.no_alloc   =*/false,
  };

  struct ne_context* ctx0 = ne_init(params);
  struct ne_cgraph gf = {};
  gf.n_threads = n_threads;

  struct ne_tensor* embd = d_ne_new_tensor_1d(ctx0, NE_TYPE_I32, N);
  memcpy(embd->data, embd_inp.data(), N * ne_element_size(embd));

  // wte
  struct ne_tensor* inpL = ne_get_rows(ctx0, model.wte, embd);

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;

    // norm
    cur = ne_norm(ctx0, inpL);

    // cur = ln_1_g*cur + ln_1_b
    cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].ln_1_g, cur), cur),
                 ne_repeat(ctx0, model.layers[il].ln_1_b, cur));

    struct ne_tensor* inpSA = cur;

    // self-attention
    struct ne_tensor* Qcur = ne_rope_inplace(
        ctx0, ne_reshape_3d(ctx0, ne_mul_mat(ctx0, model.layers[il].c_attn_q_proj_w, cur), n_embd / n_head, n_head, N),
        n_past, n_rot, 0);
    struct ne_tensor* Kcur = ne_rope_inplace(
        ctx0, ne_reshape_3d(ctx0, ne_mul_mat(ctx0, model.layers[il].c_attn_k_proj_w, cur), n_embd / n_head, n_head, N),
        n_past, n_rot, 0);

    // store key and value to memory
    struct ne_tensor* Vcur = ne_transpose(ctx0, ne_mul_mat(ctx0, model.layers[il].c_attn_v_proj_w, cur));

    struct ne_tensor* k = ne_view_1d(ctx0, model.memory_k, N * n_embd,
                                     (ne_element_size(model.memory_k) * n_embd) * (il * n_ctx + n_past));
    struct ne_tensor* v =
        ne_view_2d(ctx0, model.memory_v, N, n_embd, (n_ctx)*ne_element_size(model.memory_v),
                   (il * n_ctx) * ne_element_size(model.memory_v) * n_embd + n_past * ne_element_size(model.memory_v));

    ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur, k));
    ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur, v));

    // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
    struct ne_tensor* Q = ne_permute(ctx0, Qcur, 0, 2, 1, 3);

    // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
    struct ne_tensor* K = ne_permute(ctx0,
                                     ne_reshape_3d(ctx0,
                                                   ne_view_1d(ctx0, model.memory_k, (n_past + N) * n_embd,
                                                              il * n_ctx * ne_element_size(model.memory_k) * n_embd),
                                                   n_embd / n_head, n_head, n_past + N),
                                     0, 2, 1, 3);

    // K * Q
    struct ne_tensor* KQ = ne_mul_mat(ctx0, K, Q);

    // KQ_scaled = KQ / sqrt(n_embd/n_head)
    struct ne_tensor* KQ_scaled = ne_scale_inplace(ctx0, KQ, ne_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head)));

    // KQ_masked = mask_past(KQ_scaled)
    struct ne_tensor* KQ_masked = ne_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

    // KQ = soft_max(KQ_masked)
    struct ne_tensor* KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_masked);

    // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
    struct ne_tensor* V =
        ne_view_3d(ctx0, model.memory_v, n_past + N, n_embd / n_head, n_head, n_ctx * ne_element_size(model.memory_v),
                   n_ctx * ne_element_size(model.memory_v) * n_embd / n_head,
                   il * n_ctx * ne_element_size(model.memory_v) * n_embd);

    // KQV = transpose(V) * KQ_soft_max
    struct ne_tensor* KQV = ne_mul_mat(ctx0, V, KQ_soft_max);

    // KQV_merged = KQV.permute(0, 2, 1, 3)
    struct ne_tensor* KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);

    // cur = KQV_merged.contiguous().view(n_embd, N)
    cur = ne_cpy(ctx0, KQV_merged, d_ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N));

    // projection (no bias)
    cur = ne_mul_mat(ctx0, model.layers[il].c_attn_proj_w, cur);

    struct ne_tensor* inpFF = cur;

    // feed-forward network
    // this is independent of the self-attention result, so it could be done in parallel to the self-attention
    // note here we pass inpSA instead of cur
    cur = ne_mul_mat(ctx0, model.layers[il].c_mlp_fc_w, inpSA);

    cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur), cur);

    // GELU activation
    cur = ne_gelu(ctx0, cur);

    // projection
    // cur = proj_w*cur + proj_b
    cur = ne_mul_mat(ctx0, model.layers[il].c_mlp_proj_w, cur);

    cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur), cur);

    // self-attention + FF
    cur = ne_add(ctx0, cur, inpFF);

    // input for next layer
    inpL = ne_add(ctx0, cur, inpL);
  }
  // norm
  inpL = ne_norm(ctx0, inpL);

  // inpL = ln_f_g*inpL + ln_f_b
  inpL = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.ln_f_g, inpL), inpL), ne_repeat(ctx0, model.ln_f_b, inpL));

  // lm_head
  inpL = ne_mul_mat(ctx0, model.lmh_g, inpL);

  inpL = ne_add(ctx0, ne_repeat(ctx0, model.lmh_b, inpL), inpL);

  // logits -> probs
  // inpL = ne_soft_max_inplace(ctx0, inpL);

  // run the computation
  ne_build_forward_expand(&gf, inpL);
  ne_graph_compute(ctx0, &gf);

  // return result for just the last token
  embd_w.resize(n_vocab);
  memcpy(embd_w.data(), (float*)ne_get_data(inpL) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);

  if (mem_per_token == 0) {
    mem_per_token = ne_used_mem(ctx0) / N;
  }
  // printf("used_mem = %zu\n", ne_used_mem(ctx0));

  ne_free(ctx0);

  return true;
}

extern "C" {
void* init_gptj(int seed, int n_predict, int top_k, float top_p, float temp, float repeat_penalty, bool perplexity,
                int n_ctx, const char* model_file) {
  ne_time_init();

  const int64_t t_main_start_us = ne_time_us();

  common_params params;
  params.model = "models/gpt-j-6B/ggml-model.bin";

  params.seed = seed;
  params.n_predict = n_predict;
  params.top_k = top_k;
  params.top_p = top_p;
  params.temp = temp;
  params.repeat_penalty = repeat_penalty;
  params.perplexity = perplexity;
  params.n_ctx = n_ctx;
  params.model = model_file;

  if (params.seed < 0) {
    params.seed = time(NULL);
  }

  printf("%s: seed = %d\n", __func__, params.seed);

  int64_t t_load_us = 0;

  gptj_all* all = new gptj_all;
  all->vocab = new gpt_vocab;
  all->model = new gptj_model;

  // load the model
  {
    const int64_t t_start_us = ne_time_us();

    if (!gptj_model_load(params.model, *(all->model), *(all->vocab))) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
      return nullptr;
    }

    t_load_us = ne_time_us() - t_start_us;
  }
  return (void*)(all);
}

int32_t* eval_gptj_ids(void* all, int32_t* embd_inp_ptr, int ind_size, int n_predict, int top_k, float top_p,
                                   float temp, int n_batch) {
  gptj_all* all_in_one = (gptj_all*)all;
  gpt_vocab* vocab_ptr = (all_in_one->vocab);
  gptj_model* model_ptr = (all_in_one->model);
  std::mt19937 rng(1234);

  int n_past = 0;

  std::vector<float> logits;
  std::vector<gpt_vocab::id> embd_inp(embd_inp_ptr, embd_inp_ptr + ind_size);
  n_predict = std::min(n_predict, model_ptr->hparams.n_ctx - (int)embd_inp.size());
  std::vector<gpt_vocab::id> res;
  std::vector<gpt_vocab::id> embd;

  // determine the required inference memory per token:
  size_t mem_per_token = 0;
  gptj_eval(*model_ptr, N_thread, 0, {0, 1, 2, 3}, logits, mem_per_token);

  for (int i = embd.size(); i < embd_inp.size() + n_predict; i++) {
    // predict
    if (embd.size() > 0) {
      if (!gptj_eval(*model_ptr, N_thread, n_past, embd, logits, mem_per_token)) {
        printf("Failed to predict\n");
        return {};
      }
    }

    n_past += embd.size();
    embd.clear();

    if (i >= embd_inp.size()) {
      const int n_vocab = model_ptr->hparams.n_vocab;
      gpt_vocab::id id = 0;
      id = gpt_sample_top_k_top_p(*vocab_ptr, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);
      // add it to the context
      embd.push_back(id);
    } else {
      // if here, it means we are still processing the input prompt
      for (int k = i; k < embd_inp.size(); k++) {
        embd.push_back(embd_inp[k]);
        if (embd.size() > n_batch) {
          break;
        }
      }
      i += embd.size() - 1;
    }
    res.insert(res.end(), embd.begin(), embd.end());

    // end of text token
    if (embd.back() == 50256) {
      break;
    }
  }
  int32_t* res_ptr = new int32_t[res.size()];
  std::copy(res.begin(), res.end(), res_ptr);
  return res_ptr;
}

char* eval_gptj_char(void* all, const char* prom, int n_predict, int top_k, float top_p, float temp, int n_batch) {
  gptj_all* all_in_one = (gptj_all*)all;
  gpt_vocab* vocab_ptr = (all_in_one->vocab);
  gptj_model* model_ptr = (all_in_one->model);
  std::string prompt(prom);
  std::mt19937 rng(1234);

  int n_past = 0;

  std::vector<float> logits;

  // tokenize the prompt
  std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(*vocab_ptr, prompt);
  n_predict = std::min(n_predict, model_ptr->hparams.n_ctx - (int)embd_inp.size());
  std::string res;
  std::vector<gpt_vocab::id> embd;

  // determine the required inference memory per token:
  size_t mem_per_token = 0;
  gptj_eval(*model_ptr, N_thread, 0, {0, 1, 2, 3}, logits, mem_per_token);

  for (int i = embd.size(); i < embd_inp.size() + n_predict; i++) {
    // predict
    if (embd.size() > 0) {
      if (!gptj_eval(*model_ptr, N_thread, n_past, embd, logits, mem_per_token)) {
        printf("Failed to predict\n");
        return NULL;
      }
    }

    n_past += embd.size();
    embd.clear();

    if (i >= embd_inp.size()) {
      const int n_vocab = model_ptr->hparams.n_vocab;
      gpt_vocab::id id = 0;
      id = gpt_sample_top_k_top_p(*vocab_ptr, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);
      // add it to the context
      embd.push_back(id);
    } else {
      // if here, it means we are still processing the input prompt
      for (int k = i; k < embd_inp.size(); k++) {
        embd.push_back(embd_inp[k]);
        if (embd.size() > n_batch) {
          break;
        }
      }
      i += embd.size() - 1;
    }
    for (auto id : embd) {
      res += vocab_ptr->id_to_token[id];
    }

    // end of text token
    if (embd.back() == 50256) {
      break;
    }
  }

  char* res_c_str = new char[res.size() + 1];
  std::strcpy(res_c_str, res.c_str());
  // ne_free(model_ptr->ctx);
  return res_c_str;
}

void exit_gptj(void* all) {
  gptj_all* gptj_in_all = (gptj_all*)all;
  delete gptj_in_all;
}
}

int main() {
  auto gptj_in_all = init_gptj(1234, 32, 0, 1.0, 0.8, 1.02, false, 2048, "../ne-q4_0.bin");
  auto res = eval_gptj_char(gptj_in_all, "she opened the door and saw", 32, 0, 1.0, 0.8, 1);
  std::cout << res << std::endl;
  auto res1 = eval_gptj_char(gptj_in_all, "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun", 32, 0, 1.0, 0.8, 1);
  std::cout << res1 << std::endl;
  std::vector<int32_t> embd_inp = {7091,4721,262,3420,290,2497};
  auto res_ids = eval_gptj_ids(gptj_in_all, embd_inp.data(), embd_inp.size(), 32, 0, 1.0, 0.8, 1);
  exit_gptj(gptj_in_all);
  delete[] res;
  delete[] res1;
  delete[] res_ids;
  return 0;
}
