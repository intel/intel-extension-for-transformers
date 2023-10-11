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
// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <thread>
#include <unordered_map>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "common.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#endif

std::shared_ptr<quant_layer_base> get_model_quant_layer(const std::string model_name) {
  return ql_registry::create_ql(model_name);
}

class Model {
 public:
  Model() { model_init_backend(); }
  ~Model() {
    if (ctx) model_free(ctx);
  }
  void init_model(const std::string& model_path, int n_predict, int batch_size, int ctx_size, int seed, int threads,
                  float repeat_penalty, int num_beams, bool do_sample, int top_k, float top_p, float temperature);
  void reinit();
  std::vector<int> generate(const std::vector<int>& input_ids);
  std::vector<int> generate_tokens(const std::vector<int>& input_ids);
  bool is_token_end() { return token_eos; }
  static int quant_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                         const std::string& alg, int group_size, const std::string& scale_dtype,
                         const std::string& compute_dtype, bool use_ggml);

 private:
  model_context* ctx = nullptr;
  gpt_params params;
  std::vector<int> curr_input_ids;
  int n_past = 0;
  int n_vocab = 0;
  int n_ctx = 0;
  std::vector<model_token> last_n_tokens;
  bool token_eos = false;

  int post_process(float* logits);
  int post_greedy_search(float* logits);
  int post_beam_search(float* logits);
  int post_sample_top_k_top_p_repeat(float* logits);
};

void Model::init_model(const std::string& model_path, int max_new_tokens, int batch_size, int ctx_size, int seed,
                       int threads, float repeat_penalty, int num_beams, bool do_sample, int top_k, float top_p,
                       float temperature) {
#ifdef MODEL_NAME
  params.model_name = MODEL_NAME;
#endif
  params.model_arch = model_name_to_arch::init().find(params.model_name);
  params.model = model_path;
  params.n_predict = max_new_tokens;
  params.n_batch = batch_size;
  params.n_ctx = ctx_size;
  params.seed = seed;
  params.n_threads = threads;
  params.repeat_penalty = repeat_penalty;
  params.beam_size = num_beams;
  params.do_sample = do_sample;
  params.top_k = top_k;
  params.top_p = top_p;
  params.temp = temperature;

  printf("beam_size: %d, do_sample: %d, top_k: %d, top_p: %f\n", params.beam_size, params.do_sample, params.top_k,
         params.top_p);

  n_past = 0;
  token_eos = false;
  curr_input_ids.clear();
  ctx = model_init_from_gpt_params(params);
  n_vocab = model_n_vocab(ctx);
  n_ctx = model_n_ctx(ctx);
  last_n_tokens.resize(n_ctx, 0);
}

void Model::reinit() {
  n_past = 0;
  last_n_tokens.clear();
  last_n_tokens.resize(n_ctx, 0);
  token_eos = false;
  curr_input_ids.clear();
  ctx->n_sample = 0;
  ctx->t_sample_us = 0;
}

std::vector<int> Model::generate(const std::vector<int>& input_ids) {
  if (curr_input_ids.empty()) {
    curr_input_ids = input_ids;
  }
  for (auto item : curr_input_ids) {
    last_n_tokens.erase(last_n_tokens.begin());
    last_n_tokens.push_back(item);
  }
  // infinite text generation via context swapping
  // if we run out of context:
  // - take the n_keep first tokens from the original prompt (via n_past)
  // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
  if (n_past + curr_input_ids.size() > n_ctx) {
    const int n_left = n_past - params.n_keep;

    // always keep the first token - BOS
    n_past = std::max(1, params.n_keep);

    // insert n_left/2 tokens at the start of embd from last_n_tokens
    curr_input_ids.insert(curr_input_ids.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - curr_input_ids.size(),
                          last_n_tokens.end() - curr_input_ids.size());
  }
  model_eval(ctx, &curr_input_ids[0], curr_input_ids.size(), n_past, params.n_threads);
  n_past += curr_input_ids.size();

  float* logits = model_get_logits(ctx);
  int next_token_id = post_process(logits);
  curr_input_ids = {next_token_id};

  if (next_token_id == ctx->vocab.eos_token_id || n_past - input_ids.size() == params.n_predict) {
    token_eos = true;
  }

  return {next_token_id};
}

std::vector<int> Model::generate_tokens(const std::vector<int>& input_ids) {
  int n_remain = params.n_predict;
  std::vector<int> output_ids;

  if (curr_input_ids.empty()) {
    curr_input_ids = input_ids;
  }

  while (output_ids.size() < n_remain) {
    for (auto item : curr_input_ids) {
      last_n_tokens.erase(last_n_tokens.begin());
      last_n_tokens.push_back(item);
    }
    // infinite text generation via context swapping
    // if we run out of context:
    // - take the n_keep first tokens from the original prompt (via n_past)
    // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
    if (n_past + curr_input_ids.size() > n_ctx) {
      const int n_left = n_past - params.n_keep;

      // always keep the first token - BOS
      n_past = std::max(1, params.n_keep);

      // insert n_left/2 tokens at the start of embd from last_n_tokens
      curr_input_ids.insert(curr_input_ids.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - curr_input_ids.size(),
                            last_n_tokens.end() - curr_input_ids.size());
    }
    model_eval(ctx, &curr_input_ids[0], curr_input_ids.size(), n_past, params.n_threads);
    n_past += curr_input_ids.size();

    float* logits = model_get_logits(ctx);
    int next_token_id = post_process(logits);
    curr_input_ids = {next_token_id};
    output_ids.push_back(next_token_id);
    if (next_token_id == ctx->vocab.eos_token_id || n_past - input_ids.size() == params.n_predict) {
      token_eos = true;
      break;
    }
  }

  return output_ids;
}

int Model::post_greedy_search(float* logits) {
  int id = std::max_element(logits, logits + n_vocab) - logits;
  return id;
}

int Model::post_beam_search(float* logits) {
  // TODO: to implement
  fprintf(stderr, "\nERROR: beam search is not supported!\n");
  return -1;
}

int Model::post_sample_top_k_top_p_repeat(float* logits) {
  int n_logits = n_vocab;
  std::random_device rd;
  std::mt19937 rng{rd()};

  const auto* plogits = logits;
  int repeat_last_n = 64;
  float repeat_penalty = 1.02;
  if (params.temp <= 0) {
    // select the token with the highest logit directly
    float max_logit = plogits[0];
    gpt_vocab::id max_id = 0;

    for (int i = 1; i < n_logits; ++i) {
      if (plogits[i] > max_logit) {
        max_logit = plogits[i];
        max_id = i;
      }
    }
    return max_id;
  }

  std::vector<std::pair<double, gpt_vocab::id>> logits_id;
  logits_id.reserve(n_logits);

  {
    const float scale = 1.0f / params.temp;
    for (int i = 0; i < n_logits; ++i) {
      // repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
      // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
      if (repeat_last_n > 0 &&
          std::find(last_n_tokens.end() - repeat_last_n, last_n_tokens.end(), i) != last_n_tokens.end()) {
        // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
        if (plogits[i] < 0.0f) {
          logits_id.push_back(std::make_pair(plogits[i] * scale * repeat_penalty, i));
        } else {
          logits_id.push_back(std::make_pair(plogits[i] * scale / repeat_penalty, i));
        }
      } else {
        logits_id.push_back(std::make_pair(plogits[i] * scale, i));
      }
    }
  }

  // find the top K tokens
  std::partial_sort(logits_id.begin(), logits_id.begin() + params.top_k, logits_id.end(),
                    [](const std::pair<double, gpt_vocab::id>& a, const std::pair<double, gpt_vocab::id>& b) {
                      return a.first > b.first;
                    });

  logits_id.resize(params.top_k);

  double maxl = -INFINITY;
  for (const auto& kv : logits_id) {
    maxl = std::max(maxl, kv.first);
  }

  // compute probs for the top K tokens
  std::vector<double> probs;
  probs.reserve(logits_id.size());

  double sum = 0.0;
  for (const auto& kv : logits_id) {
    double p = exp(kv.first - maxl);
    probs.push_back(p);
    sum += p;
  }

  // normalize the probs
  for (auto& p : probs) {
    p /= sum;
  }

  if (params.top_p < 1.0f) {
    double cumsum = 0.0f;
    for (int i = 0; i < params.top_k; i++) {
      cumsum += probs[i];
      if (cumsum >= params.top_p) {
        params.top_k = i + 1;
        probs.resize(params.top_k);
        logits_id.resize(params.top_k);
        break;
      }
    }

    cumsum = 1.0 / cumsum;
    for (int i = 0; i < (int)probs.size(); i++) {
      probs[i] *= cumsum;
    }
  }

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  int idx = dist(rng);

  return logits_id[idx].second;
}

int Model::post_process(float* logits) {
  if (params.beam_size == 1) {
    if (params.do_sample == false) {
      return post_greedy_search(logits);
    } else {
      return post_sample_top_k_top_p_repeat(logits);
    }
  } else {
    if (params.do_sample == false) {
      return post_beam_search(logits);
    }
  }
  fprintf(stderr, "\nERROR: post process (beam_size=%d, do_sample=%d) is not supported!\n", params.beam_size,
          params.do_sample);
  return -1;
}

int Model::quant_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                       const std::string& alg, int group_size, const std::string& scale_dtype,
                       const std::string& compute_dtype, bool use_ggml) {
  quant_params q_params;
#ifdef MODEL_NAME
  q_params.model_name = MODEL_NAME;
#endif
  model_archs mt = model_name_to_arch::init().find(q_params.model_name);
  if (mt == MODEL_UNKNOWN) {
    fprintf(stderr, "error, please set model_name \n");
    exit(0);
  }
  q_params.model_arch = mt;
  q_params.model_file = model_path;
  q_params.out_file = out_path;
  q_params.weight_dtype = weight_dtype;
  q_params.alg = alg;
  q_params.group_size = group_size;
  q_params.scale_dtype = scale_dtype;
  q_params.compute_dtype = compute_dtype;
  q_params.use_ggml = use_ggml;

  ne_ftype ftype = quant_params_to_ftype(q_params);
  printf("ne_ftype: %d\n", ftype);
  const int nthread = q_params.nthread;

  auto quant_layer = get_model_quant_layer(q_params.model_name);
  if (model_quantize(q_params, quant_layer)) {
    fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, q_params.model_file.c_str());
    return 1;
  }
  return 0;
}

namespace py = pybind11;

#if MODEL_NAME_ID == 1

PYBIND11_MODULE(gptj_cpp, m)
#elif MODEL_NAME_ID == 2

PYBIND11_MODULE(falcon_cpp, m)

#elif MODEL_NAME_ID == 3

PYBIND11_MODULE(gptneox_cpp, m)

#elif MODEL_NAME_ID == 4

PYBIND11_MODULE(dolly_cpp, m)

#elif MODEL_NAME_ID == 5

PYBIND11_MODULE(llama_cpp, m)

#elif MODEL_NAME_ID == 6

PYBIND11_MODULE(mpt_cpp, m)

#elif MODEL_NAME_ID == 7

PYBIND11_MODULE(starcoder_cpp, m)

#elif MODEL_NAME_ID == 8

PYBIND11_MODULE(opt_cpp, m)

#elif MODEL_NAME_ID == 9

PYBIND11_MODULE(bloom_cpp, m)

#elif MODEL_NAME_ID == 10

PYBIND11_MODULE(chatglm2_cpp, m)

#elif MODEL_NAME_ID == 11

PYBIND11_MODULE(chatglm_cpp, m)

#endif
{
  m.doc() = "cpp model python binding";
  py::class_<Model>(m, "Model", py::module_local())
      .def(py::init())
      .def("init_model", &Model::init_model, "initial model with model path and parameters", py::arg("model_path"),
           py::arg("max_new_tokens") = -1, py::arg("batch_size") = 512, py::arg("ctx_size") = 512, py::arg("seed") = -1,
           py::arg("threads") = 8, py::arg("repeat_penalty") = 1.1f, py::arg("num_beams") = 1,
           py::arg("do_sample") = false, py::arg("top_k") = 40, py::arg("top_p") = 0.95, py::arg("temperature") = 0.8)
      .def("generate", &Model::generate, "Generate token with input ids", py::arg("input_ids"))
      .def("generate_tokens", &Model::generate_tokens, "Generate tokens with input ids", py::arg("input_ids"))
      .def_static("quant_model", &Model::quant_model, "Quantize model", py::arg("model_path"), py::arg("out_path"),
                  py::arg("weight_dtype") = "int4", py::arg("alg") = "sym", py::arg("group_size") = 32,
                  py::arg("scale_dtype") = "fp32", py::arg("compute_dtype") = "ggml", py::arg("use_ggml") = false)
      .def("is_token_end", &Model::is_token_end)
      .def("reinit", &Model::reinit);
}
