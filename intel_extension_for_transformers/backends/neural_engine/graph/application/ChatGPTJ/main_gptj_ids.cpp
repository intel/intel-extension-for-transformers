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

#include "common.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

static model_context** g_ctx;

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

bool gptj_model_eval_idx(model_context* ctx, model_token* tokens, size_t n_eval, size_t n_past, size_t n_threads) {
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

int main(int argc, char** argv) {
  gpt_params params;
  params.name = MODEL_GPTJ;
  if (gpt_params_parse(argc, argv, params) == false) {
    return 1;
  }

  if (params.perplexity) {
    printf("\n************\n");
    printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
    printf("************\n\n");

    return 0;
  }

  if (params.embedding) {
    printf("\n************\n");
    printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
    printf("************\n\n");

    return 0;
  }

  if (params.n_ctx > 2048) {
    fprintf(stderr,
            "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
            "expect poor results\n",
            __func__, params.n_ctx);
  }

  if (params.seed < 0) {
    params.seed = time(NULL);
  }
  printf("%s: seed = %d\n", __func__, params.seed);

  std::mt19937 rng(params.seed);
  if (params.ids.empty()) {
    params.ids = gpt_random_ids(rng);
  }
  auto embd_inp = params.ids;
  std::vector<model_token> embd;
  int n_past = 0;
  const int64_t t_main_start_us = ne_time_us();

  params.n_predict = std::min(params.n_predict, params.n_ctx - (int)embd_inp.size());
  printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
  printf("\n");

  model_init_backend();
  model_context* ctx;
  g_ctx = &ctx;

  ctx = model_init_from_gpt_params(params);
  if (ctx == NULL) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return 1;
  }

  for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
    // predict
    if (embd.size() > 0) {
      if (!gptj_model_eval_idx(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
        printf("Failed to predict\n");
        return 1;
      }
    }
    auto logits = model_get_logits(ctx);
    auto n_vocab = model_n_vocab(ctx);
    n_past += embd.size();
    embd.clear();

    if (i >= embd_inp.size()) {
      // sample next token
      const int top_k = params.top_k;
      const float top_p = params.top_p;
      const float temp = params.temp;

      gpt_vocab::id id = 0;
      id = model_sample_top_k_top_p(ctx, n_vocab, logits, top_k, top_p, temp);

      // add it to the context
      embd.push_back(id);
    } else {
      // if here, it means we are still processing the input prompt
      for (int k = i; k < embd_inp.size(); k++) {
        embd.push_back(embd_inp[k]);
        if (embd.size() > params.n_batch) {
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

  model_print_timings(ctx);
  model_free(ctx);

  return 0;
}
