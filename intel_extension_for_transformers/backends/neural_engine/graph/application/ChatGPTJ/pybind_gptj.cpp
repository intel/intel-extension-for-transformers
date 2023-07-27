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
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

#define N_threads 56
static model_context** g_ctx;

bool gptj_model_eval_ids(model_context* ctx, model_token* tokens, size_t n_eval, size_t n_past, size_t n_threads) {
  const int n_ctx = model_n_ctx(ctx);
  if ((int)n_eval > n_ctx - 4) {
    fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)n_eval, n_ctx - 4);
    return 1;
  }

  if (model_eval(ctx, tokens, n_eval, n_past, n_threads)) {
    fprintf(stderr, "%s : failed to eval\n", __func__);
    return 1;
  }
  return true;
}

extern "C" {
void* init_gptj(int seed, int n_predict, int n_batch, int top_k, float top_p, float temp, float repeat_penalty,
                bool perplexity, int n_ctx, const char* model_file, bool beam_search = false, int beam_size = 4,
                int batch_size = 1) {
  gpt_params params;
  params.n_threads = N_threads;
  params.seed = seed;
  params.name = MODEL_GPTJ;
  params.n_ctx = n_ctx;
  params.n_predict = n_predict;
  params.n_batch = n_batch;
  params.model = std::string(model_file);
  params.n_predict = n_predict;
  params.top_k = top_k;
  params.top_p = top_p;
  params.temp = temp;
  params.repeat_penalty = repeat_penalty;
  params.perplexity = perplexity;
  params.batch_size = batch_size;
  params.beam_search = beam_search;
  params.beam_size = beam_size;
  model_init_backend();
  model_context* ctx;
  g_ctx = &ctx;
  ctx = model_init_from_gpt_params(params);
  if (ctx == NULL) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return nullptr;
  }
  return (void*)ctx;
}

int32_t* eval_gptj_ids(void* ctx, int32_t* embd_inp_ptr, int ind_size, int n_predict, int top_k, float top_p,
                       float temp, int n_batch) {
  model_context* lctx = (model_context*)ctx;
  int n_past = 0;

  auto hparams = lctx->model.hparams;

  n_predict = std::min(n_predict, (int)hparams.n_ctx - (int)ind_size);
  std::vector<model_token> res;
  bool do_beam_search = beam_search;

  if (do_beam_search) {
    res = beam_search(lctx->beam_size, n_predict, lctx, embd_inp_ptr, ind_size, N_threads);
  } else {
    std::vector<model_token> embd_inp(embd_inp_ptr, embd_inp_ptr + ind_size);
    std::vector<model_token> embd;
    for (int i = embd.size(); i < embd_inp.size() + n_predict; i++) {
      // predict
      if (embd.size() > 0) {
        if (!gptj_model_eval_ids(lctx, embd.data(), embd.size(), n_past, N_threads)) {
          printf("Failed to predict\n");
          return {};
        }
      }

      auto logits = model_get_logits(lctx);
      n_past += embd.size();
      embd.clear();

      if (i >= embd_inp.size()) {
        const int n_vocab = hparams.n_vocab;
        gpt_vocab::id id = 0;
        id = model_sample_top_k_top_p(lctx, n_vocab, logits, top_k, top_p, temp);
        // add it to the context
        embd.push_back(id);
        res.push_back(id);
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

      // end of text token
      if (embd.back() == 50256) {
        break;
      }
    }
  }
  int32_t* res_ptr = new int32_t[res.size() + 1];
  res_ptr[0] = res.size();
  std::copy(res.begin(), res.end(), &res_ptr[1]);
  return res_ptr;
}

char* eval_gptj_char(void* ctx, const char* prom, int n_predict, int top_k, float top_p, float temp, int n_batch) {
  model_context* lctx = (model_context*)ctx;
  int n_past = 0;

  auto hparams = lctx->model.hparams;
  std::vector<model_token> embd_inp = ::model_tokenize(lctx, std::string(prom), false);
  n_predict = std::min(n_predict, (int)hparams.n_ctx - (int)embd_inp.size());
  std::string res;
  std::vector<model_token> embd;

  bool do_beam_search = lctx->beam_search;
  if (do_beam_search) {
    embd = beam_search(lctx->beam_size, n_predict, lctx, embd_inp.data(), embd_inp.size(), N_threads);
    for (auto id : embd_inp) {
      res += model_token_to_str(lctx, id);
    }
    for (auto id : embd) {
      res += model_token_to_str(lctx, id);
    }
  } else {
    std::vector<float> logits;
    for (int i = embd.size(); i < embd_inp.size() + n_predict; i++) {
      // predict
      if (embd.size() > 0) {
        if (!gptj_model_eval_ids(lctx, embd.data(), embd.size(), n_past, N_threads)) {
          printf("Failed to predict\n");
          return {};
        }
      }

      auto logits = model_get_logits(lctx);
      n_past += embd.size();
      embd.clear();

      if (i >= embd_inp.size()) {
        const int n_vocab = hparams.n_vocab;
        model_token id = 0;
        id = model_sample_top_k_top_p(lctx, n_vocab, logits, top_k, top_p, temp);
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
        res += model_token_to_str(lctx, id);
      }

      // end of text token
      if (embd.back() == 50256) {
        break;
      }
    }
  }

  char* res_c_str = new char[res.size() + 1];
  std::strcpy(res_c_str, res.c_str());
  return res_c_str;
}

void exit_gptj(void* ctx) {
  model_context* lctx = (model_context*)ctx;
  model_free(lctx);
}
}

int main() {
  // auto gptj_in_all_tk = init_gptj(1234, 32, 32, 40, 1.0, 0.8, 1.02, false, 2048, "/home/zhentao/q4_j_new.bin");
  auto gptj_in_all_bs = init_gptj(1234, 32, 32, 40, 1.0, 0.8, 1.02, false, 2048, "/home/zhentao/q4_j_new.bin", true, 4, 1);
  std::vector<void*> ctxs = {gptj_in_all_bs};
  for (auto gptj_in_all : ctxs) {
    auto res = eval_gptj_char(gptj_in_all, "she opened the door and saw", 32, 40, 1.0, 0.8, 32);
    std::cout << res << std::endl;
    auto res1 = eval_gptj_char(gptj_in_all,
                              "Once upon a time, there existed a little girl, who liked to have adventures. She wanted "
                              "to go to places and meet new people, and have fun",
                              32, 40, 1.0, 0.8, 32);
    std::cout << res1 << std::endl;
    std::vector<int32_t> embd_inp = {7091, 4721, 262, 3420, 290, 2497};
    auto res_ids = eval_gptj_ids(gptj_in_all, embd_inp.data(), embd_inp.size(), 32, 40, 1.0, 0.8, 32);
    exit_gptj(gptj_in_all);
    delete[] res;
    delete[] res1;
    delete[] res_ids;
  }
  return 0;
}
