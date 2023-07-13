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
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cinttypes>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <thread>

#include "core/ne_layers.h"
#include "common.h"
#include "data_types.h"
#include "ne.h"

// default hparams (Falcon 7B)
// TODO add n_head_kv to support 40B
struct falcon_hparams {
  int32_t n_vocab = 65024;
  int32_t n_ctx = 2048;
  int32_t n_embd = 4544;
  int32_t n_head = 71;
  int32_t n_layer = 32;
  int32_t ftype = 1;
};

struct falcon_layer {
  // normalization
  struct ne_tensor* attention_norm;
  struct ne_tensor* attention_norm_b;

  // attention
  struct ne_tensor* query_key_value;
  struct ne_tensor* wo;

  // ff
  struct ne_tensor* ffn_up;
  struct ne_tensor* ffn_down;
};

struct falcon_model {
  falcon_hparams hparams;

  struct ne_tensor* tok_embeddings;
  struct ne_tensor* output_norm;
  struct ne_tensor* output_norm_b;
  struct ne_tensor* lm_head;

  std::vector<falcon_layer> layers;

  // key + value memory
  struct ne_tensor* memory_k;
  struct ne_tensor* memory_v;

  struct ne_context* ctx;
  std::map<std::string, struct ne_tensor*> tensors;
};

// load the model's weights from a file
bool falcon_model_load(const std::string& fname, falcon_model& model, gpt_vocab& vocab) {
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
    if (magic != 0x67676d6c) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
      return false;
    }
  }

  // load hparams
  {
    auto& hparams = model.hparams;

    fin.read((char*)&hparams.n_vocab, sizeof(hparams.n_vocab));
    fin.read((char*)&hparams.n_embd, sizeof(hparams.n_embd));
    fin.read((char*)&hparams.n_head, sizeof(hparams.n_head));
    fin.read((char*)&hparams.n_layer, sizeof(hparams.n_layer));
    fin.read((char*)&hparams.ftype, sizeof(hparams.ftype));

    const int32_t qntvr = hparams.ftype / NE_QNT_VERSION_FACTOR;
    // hparams.n_ctx = std::min(hparams.max_seq_len, hparams.n_ctx);

    printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
    printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
    printf("%s: n_head  = %d\n", __func__, hparams.n_head);
    printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
    printf("%s: ftype   = %d\n", __func__, hparams.ftype);
    printf("%s: qntvr   = %d\n", __func__, qntvr);

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

      vocab.token_to_id[word] = i;
      vocab.id_to_token[i] = word;
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
    const int n_head = hparams.n_head;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_ff = 4 * model.hparams.n_embd;
    const int n_vocab = hparams.n_vocab;
    const int head_dim = hparams.n_embd / hparams.n_head;

    ctx_size += n_embd * n_vocab * ne_type_sizef(NE_TYPE_F32);  // tok_embeddings

    ctx_size += n_embd * ne_type_sizef(NE_TYPE_F32);  // output_norm
    ctx_size += n_embd * ne_type_sizef(NE_TYPE_F32);  // output_norm_b

    ctx_size += n_embd * n_vocab * ne_type_sizef(NE_TYPE_F32);  // lm_head

    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));  // attention_norm
    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));  // attention_norm_b

    ctx_size += n_layer * (n_embd * (n_embd + 2 * (n_embd / n_head)) * ne_type_sizef(wtype));  // query_key_value
    ctx_size += n_layer * (n_embd * n_embd * ne_type_sizef(wtype));                            // wo

    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));  // ffn_norm
    ctx_size += n_layer * (n_embd * ne_type_sizef(NE_TYPE_F32));  // ffn_norm_b

    ctx_size += n_layer * (n_ff * n_embd * ne_type_sizef(wtype));  // ffn_up
    ctx_size += n_layer * (n_ff * n_embd * ne_type_sizef(wtype));  // ffn_down

    ctx_size += n_ctx * n_layer * head_dim * ne_type_sizef(NE_TYPE_F32);  // memory_k
    ctx_size += n_ctx * n_layer * head_dim * ne_type_sizef(NE_TYPE_F32);  // memory_v

    ctx_size += (5 + 10 * n_layer) * 256;  // TODO object overhead
    printf("%s: ne ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));
  }

  // create the ne context
  {
    struct ne_init_params params = {
        /* .mem_size =*/ctx_size,
        /* .mem_buffer = */ NULL,
        /*.no_alloc = */ false,
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
    const int n_head = hparams.n_head;
    const int n_layer = hparams.n_layer;
    const int n_ff = 4 * model.hparams.n_embd;
    const int n_vocab = hparams.n_vocab;

    model.layers.resize(n_layer);

    model.tok_embeddings = ne_new_tensor_2d(ctx, NE_TYPE_F32, n_embd, n_vocab, NE_SIZE_CALC);
    model.output_norm = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
    model.output_norm_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
    model.lm_head = ne_new_tensor_2d(ctx, NE_TYPE_F32, n_embd, n_vocab, NE_SIZE_CALC);

    // map by name
    model.tensors["transformer.word_embeddings.weight"] = model.tok_embeddings;
    model.tensors["transformer.ln_f.weight"] = model.output_norm;
    model.tensors["transformer.ln_f.bias"] = model.output_norm_b;
    model.tensors["lm_head.weight"] = model.lm_head;

    for (int i = 0; i < n_layer; ++i) {
      auto& layer = model.layers[i];

      layer.attention_norm = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
      layer.attention_norm_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
      // query_key_value shape for config.multi_query == True:
      layer.query_key_value = ne_new_tensor_2d(ctx, wtype, n_embd, n_embd + 2 * (n_embd / n_head), NE_SIZE_CALC);
      layer.wo = ne_new_tensor_2d(ctx, wtype, n_embd, n_embd, NE_SIZE_CALC);
      layer.ffn_up = ne_new_tensor_2d(ctx, wtype, n_embd, n_ff, NE_SIZE_CALC);
      layer.ffn_down = ne_new_tensor_2d(ctx, wtype, n_ff, n_embd, NE_SIZE_CALC);

      // map by name
      model.tensors["transformer.h." + std::to_string(i) + ".input_layernorm.weight"] = layer.attention_norm;
      model.tensors["transformer.h." + std::to_string(i) + ".input_layernorm.bias"] = layer.attention_norm_b;
      model.tensors["transformer.h." + std::to_string(i) + ".self_attention.query_key_value.weight"] =
          layer.query_key_value;
      model.tensors["transformer.h." + std::to_string(i) + ".self_attention.dense.weight"] = layer.wo;
      model.tensors["transformer.h." + std::to_string(i) + ".mlp.dense_h_to_4h.weight"] = layer.ffn_up;
      model.tensors["transformer.h." + std::to_string(i) + ".mlp.dense_4h_to_h.weight"] = layer.ffn_down;
    }
  }

  // key + value memory
  {
    const auto& hparams = model.hparams;

    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int head_dim = hparams.n_embd / hparams.n_head;

    const int64_t n_mem = n_layer * n_ctx;
    const int64_t n_elements = head_dim * n_mem;

    model.memory_k = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_elements, NE_SIZE_CALC);
    model.memory_v = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_elements, NE_SIZE_CALC);

    const size_t memory_size = ne_nbytes(model.memory_k) + ne_nbytes(model.memory_v);

    printf("%s: kv cache memory_size = %8.2f MB, n_mem = %" PRId64 "\n", __func__, memory_size / 1024.0 / 1024.0,
           n_mem);
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
        fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%5d, %5d], expected [%5d, %5d]\n",
                __func__, name.data(), (int)tensor->ne[0], (int)tensor->ne[1], ne[0], ne[1]);
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

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool falcon_eval(const falcon_model& model, const int n_threads, const int n_past,
                 const std::vector<gpt_vocab::id>& embd_inp, std::vector<float>& embd_w, size_t& mem_per_token) {
  const int N = embd_inp.size();

  const auto& hparams = model.hparams;

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = hparams.n_ctx;
  const int n_head = hparams.n_head;
  const int n_vocab = hparams.n_vocab;
  const size_t head_dim = n_embd / n_head;

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
      /* .mem_size =*/buf_size,
      /* .mem_buffer = */ buf,
      /*.no_alloc = */ false,
  };

  struct ne_context* ctx0 = ne_init(params);
  struct ne_cgraph gf = {};
  gf.n_threads = n_threads;

  struct ne_tensor* embd = ne_new_tensor_1d(ctx0, NE_TYPE_I32, N, NE_SIZE_CALC);
  memcpy(embd->data, embd_inp.data(), N * ne_element_size(embd));

  // wte
  struct ne_tensor* inpL = ne_get_rows(ctx0, model.tok_embeddings, embd);
  struct ne_tensor* repeat_dummy = ne_new_tensor_3d(ctx0, inpL->type, head_dim, N + n_past, n_head, NE_SIZE_CALC);

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* cur;
    struct ne_tensor* layernorm_output;

    ne_set_scratch(ctx0, {
                             0,
                             scr0_size,
                             scr0,
                         });

    // self-attention
    {
      {
        cur = ne_norm(ctx0, inpL);

        cur = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.layers[il].attention_norm, cur), cur),
                     ne_repeat(ctx0, model.layers[il].attention_norm_b, cur));
      }
      layernorm_output = cur;

      // compute QKV
      cur = ne_mul_mat(ctx0, model.layers[il].query_key_value, cur);

      size_t fused_qkv_row_nb = (n_embd + 2 * (n_embd / n_head)) * sizeof(float);

      struct ne_tensor* Qcur =
          ne_view_3d(ctx0, cur, head_dim, n_head, N, head_dim * sizeof(float), fused_qkv_row_nb, 0);

      struct ne_tensor* Kcur =
          ne_view_3d(ctx0, cur, head_dim, 1, N, head_dim * sizeof(float), fused_qkv_row_nb, n_embd * sizeof(float));

      struct ne_tensor* Vcur = ne_view_3d(ctx0, cur, head_dim, 1, N, head_dim * sizeof(float), fused_qkv_row_nb,
                                          (n_embd + head_dim) * sizeof(float));

      // using mode = 2 for neox mode
      Qcur = ne_rope_inplace(ctx0, Qcur, n_past, head_dim, 2);
      Kcur = ne_rope_inplace(ctx0, Kcur, n_past, head_dim, 2);

      // store key and value to memory
      {
        struct ne_tensor* k = ne_view_1d(ctx0, model.memory_k, N * head_dim,
                                         (ne_element_size(model.memory_k) * head_dim) * (il * n_ctx + n_past));
        struct ne_tensor* v = ne_view_1d(ctx0, model.memory_v, N * head_dim,
                                         (ne_element_size(model.memory_v) * head_dim) * (il * n_ctx + n_past));

        ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur, k));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur, v));
      }

      // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
      struct ne_tensor* Q = ne_permute(ctx0, Qcur, 0, 2, 1, 3);

      struct ne_tensor* K =
          ne_permute(ctx0,
                     ne_reshape_3d(ctx0,
                                   ne_view_1d(ctx0, model.memory_k, (n_past + N) * head_dim,
                                              il * n_ctx * ne_element_size(model.memory_k) * head_dim),
                                   head_dim, 1, n_past + N),
                     0, 2, 1, 3);

      // K * Q
      K = ne_cont(ctx0, ne_repeat(ctx0, K, repeat_dummy));
      struct ne_tensor* KQ = ne_mul_mat(ctx0, K, Q);

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      struct ne_tensor* KQ_scaled = ne_scale_inplace(ctx0, KQ, ne_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head)));

      // KQ_masked = mask_past(KQ_scaled)
      struct ne_tensor* KQ_masked = ne_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

      // KQ = soft_max(KQ_masked)
      struct ne_tensor* KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_masked);

      // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
      struct ne_tensor* V =
          ne_permute(ctx0,
                     ne_reshape_3d(ctx0,
                                   ne_view_1d(ctx0, model.memory_v, (n_past + N) * head_dim,
                                              il * n_ctx * ne_element_size(model.memory_v) * head_dim),
                                   head_dim, 1, n_past + N),
                     0, 2, 1, 3);

      V = ne_cont(ctx0, ne_transpose(ctx0, ne_repeat(ctx0, V, repeat_dummy)));

      // KQV = transpose(V) * KQ_soft_max
      struct ne_tensor* KQV = ne_mul_mat(ctx0, V, KQ_soft_max);

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct ne_tensor* KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);

      // cur = KQV_merged.contiguous().view(n_embd, N)
      cur = ne_cpy(ctx0, KQV_merged, ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N, NE_SIZE_CALC));

      // projection
      { cur = ne_mul_mat(ctx0, model.layers[il].wo, cur); }
    }

    ne_set_scratch(ctx0, {
                             0,
                             scr1_size,
                             scr1,
                         });

    struct ne_tensor* inpFF = layernorm_output;
    struct ne_tensor* attn_out = ne_cpy(ctx0, cur, ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N, NE_SIZE_CALC));

    // FFN (pre_layer_norm output)
    {
      cur = ne_mul_mat(ctx0, model.layers[il].ffn_up, inpFF);
      cur = ne_gelu(ctx0, cur);
      cur = ne_mul_mat(ctx0, model.layers[il].ffn_down, cur);
    }

    cur = ne_add(ctx0, cur, attn_out);
    cur = ne_add(ctx0, cur, inpL);
    // input for next layer
    inpL = cur;
  }

  ne_set_scratch(ctx0, {
                           0,
                           scr0_size,
                           scr0,
                       });

  // norm
  {
    inpL = ne_norm(ctx0, inpL);

    // inpL = ln_f_g*inpL + ln_f_b
    inpL = ne_add(ctx0, ne_mul(ctx0, ne_repeat(ctx0, model.output_norm, inpL), inpL),
                  ne_repeat(ctx0, model.output_norm_b, inpL));
  }

  ne_set_scratch(ctx0, {
                           0,
                           0,
                           nullptr,
                       });

  // lm_head
  {
    inpL = ne_mul_mat(ctx0, model.lm_head, inpL);

    // inpL = ne_add(ctx0,
    //         ne_repeat(ctx0, model.lmh_b, inpL),
    //         inpL);
  }

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

// usage:
//  ./main_falcon -m models/ne-model.bin -p "Once upon a time," -n 64
int main(int argc, char** argv) {
  ne_time_init();

  const int64_t t_main_start_us = ne_time_us();

  common_params params;
  params.model = "models/falcon-7b/ne-falcon-f16.bin";

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
  falcon_model model;

  // load the model
  {
    const int64_t t_start_us = ne_time_us();

    if (!falcon_model_load(params.model, model, vocab)) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
      return 1;
    }

    t_load_us = ne_time_us() - t_start_us;

    test_gpt_tokenizer(vocab, params.token_test);
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

  std::vector<int32_t> last_n_tokens(params.n_ctx);
  std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

  std::vector<float> logits;

  // tokenize the prompt
  std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

  params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int)embd_inp.size());

  printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
  for (int i = 0; i < embd_inp.size(); i++) {
    printf("%s: token[%d] = %6d, %s\n", __func__, i, embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
  }
  printf("\n");

  std::vector<gpt_vocab::id> embd;

  // determine the required inference memory per token:
  size_t mem_per_token = 0;
  falcon_eval(model, params.n_threads, 0, {0, 1, 2, 3}, logits, mem_per_token);

  bool first_token = true;
  std::vector<int64_t> eval_times;
  for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
    // predict
    if (embd.size() > 0) {
      const int64_t t_start_us = ne_time_us();

      if (!falcon_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
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
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        t_sample_us += ne_time_us() - t_start_sample_us;
      }

      // add it to the context
      embd.push_back(id);
    } else {
      // if here, it means we are still processing the input prompt
      for (int k = i; k < embd_inp.size(); k++) {
        embd.push_back(embd_inp[k]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(embd_inp[k]);
        if (embd.size() > params.n_batch) {
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

    // end of text token
    if (embd.back() == 0) {
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
      printf("prediction %3d, time: %.2fms\n", i, eval_times[i] / 1000.0f);
    }
  }

  ne_free(model.ctx);

  return 0;
}
