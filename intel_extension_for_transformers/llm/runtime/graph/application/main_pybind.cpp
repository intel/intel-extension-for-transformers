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
                  float repetition_penalty, int num_beams, bool do_sample, int top_k, float top_p, float temperature,
                  int min_new_tokens, float length_penalty, bool early_stopping, int n_keep, int n_discard);
  void reinit();
  std::vector<model_token> generate(const std::vector<model_token>& input_ids);
  std::vector<model_token> generate_tokens(const std::vector<model_token>& input_ids);
  bool is_token_end() { return token_eos; }
  static int quant_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                         const std::string& alg, int group_size, const std::string& scale_dtype,
                         const std::string& compute_dtype, bool use_ggml);
  void reset_token_end() { token_eos = false; curr_input_ids.clear();}
 private:
  model_context* ctx = nullptr;
  gpt_params params;
  std::vector<model_token> curr_input_ids;
  int n_past = 0;
  int n_vocab = 0;
  int n_ctx = 0;
  std::vector<model_token> last_n_tokens;
  bool token_eos = false;
  long int generate_count = 0;

  model_token post_process(float* logits);
  model_token post_greedy_search(float* logits);
  std::vector<model_token> post_beam_search(model_context* lctx, const int& n_predict, const model_token* tokens_inp,
                                            const int& n_tokens, const int& n_threads);
  model_token post_sample_top_k_top_p_repeat(float* logits);
};

void Model::init_model(const std::string& model_path, int max_new_tokens, int batch_size, int ctx_size, int seed,
                       int threads, float repetition_penalty, int num_beams, bool do_sample, int top_k, float top_p,
                       float temperature, int min_new_tokens, float length_penalty, bool early_stopping, int n_keep,
                       int n_discard) {
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
  params.repeat_penalty = repetition_penalty;
  params.beam_size = num_beams;
  params.do_sample = do_sample;
  params.beam_search = (num_beams > 1 && !do_sample) ? true : false;
  if (params.beam_search) {
    params.memory_type = KV_MEM_TYPE_F16;  // TODO NO MHA IN BEAM SEARCH
  }
  params.top_k = top_k;
  params.top_p = top_p;
  params.temp = temperature;
  params.n_keep = n_keep;
  params.n_discard = n_discard;

  printf("beam_size: %d, do_sample: %d, top_k: %d, top_p: %f\n", params.beam_size, params.do_sample, params.top_k,
         params.top_p);

  n_past = 0;
  token_eos = false;
  curr_input_ids.clear();
  ctx = model_init_from_gpt_params(params);
  n_vocab = model_n_vocab(ctx);
  n_ctx = model_n_ctx(ctx);
  last_n_tokens.resize(n_ctx, 0);
  ctx->generation_conf.min_new_tokens = min_new_tokens;
  ctx->generation_conf.length_penalty = length_penalty;
  ctx->generation_conf.do_early_stopping = early_stopping;
}

void Model::reinit() {
  n_past = 0;
  last_n_tokens.clear();
  last_n_tokens.resize(n_ctx, 0);
  token_eos = false;
  curr_input_ids.clear();
  ctx->n_sample = 0;
  ctx->t_sample_us = 0;
  generate_count = 0;
}

std::vector<model_token> Model::generate(const std::vector<model_token>& input_ids) {
  if (curr_input_ids.empty()) {
    // if (params.model_arch == MODEL_CHATGLM2) {
    //   std::vector<std::string> prompts;
    //   prompts.push_back(params.prompt);
    //   std::string prompt = build_prompt_glm2(prompts);
    //   embd_inp = ::model_tokenize(ctx, prompt, false);
    //   embd_inp.insert(embd_inp.begin(), {64790, 64792});  // special prefix
    // } 
    curr_input_ids = input_ids;
  }
  for (auto item : curr_input_ids) {
    last_n_tokens.erase(last_n_tokens.begin());
    last_n_tokens.push_back(item);
  }
  // infinite text generation via context swapping
  if (n_past + curr_input_ids.size() > n_ctx) {
    // always keep the first token
    n_past = std::max(1, params.n_keep);

    int n_discard = params.n_discard;
    if (n_discard == -1) n_discard = (n_ctx - curr_input_ids.size() - params.n_keep) / 2;
    // drop n_discard tokens
    curr_input_ids.insert(curr_input_ids.begin(), last_n_tokens.begin() + params.n_keep + n_discard,
                          last_n_tokens.end() - curr_input_ids.size());
  }
  model_eval(ctx, &curr_input_ids[0], curr_input_ids.size(), n_past, params.n_threads);
  n_past += curr_input_ids.size();

  float* logits = model_get_logits(ctx);
  model_token next_token_id = post_process(logits);
  curr_input_ids = {next_token_id};

  generate_count++;
  if (next_token_id == ctx->vocab.eos_token_id || generate_count >= params.n_predict) {
    token_eos = true;
  }

  return {next_token_id};
}

std::vector<model_token> Model::generate_tokens(const std::vector<model_token>& input_ids) {
  int n_remain = params.n_predict;
  std::vector<model_token> output_ids;

  if (curr_input_ids.empty()) {
    curr_input_ids = input_ids;
  }

  while (output_ids.size() < n_remain) {
    for (auto item : curr_input_ids) {
      last_n_tokens.erase(last_n_tokens.begin());
      last_n_tokens.push_back(item);
    }
    // infinite text generation via context swapping
    if (n_past + curr_input_ids.size() > n_ctx) {
      // always keep the first token
      n_past = std::max(1, params.n_keep);

      int n_discard = params.n_discard;
      if (n_discard == -1) n_discard = (n_ctx - curr_input_ids.size() - params.n_keep) / 2;
      // drop n_discard tokens
      curr_input_ids.insert(curr_input_ids.begin(), last_n_tokens.begin() + params.n_keep + n_discard,
                            last_n_tokens.end() - curr_input_ids.size());
    }
    if (ctx->beam_search) {
      output_ids = post_beam_search(ctx, n_remain, curr_input_ids.data(), curr_input_ids.size(), params.n_threads);
      break;
    }
    model_eval(ctx, &curr_input_ids[0], curr_input_ids.size(), n_past, params.n_threads);
    n_past += curr_input_ids.size();

    float* logits = model_get_logits(ctx);
    model_token next_token_id = post_process(logits);
    curr_input_ids = {next_token_id};
    output_ids.push_back(next_token_id);
    generate_count++;
    if (next_token_id == ctx->vocab.eos_token_id || generate_count >= params.n_predict) {
      token_eos = true;
      break;
    }
  }

  return output_ids;
}

model_token Model::post_greedy_search(float* logits) {
  model_token id = std::max_element(logits, logits + n_vocab) - logits;
  return id;
}

std::vector<model_token> Model::post_beam_search(model_context* lctx, const int& n_predict,
                                                 const model_token* tokens_inp, const int& n_tokens,
                                                 const int& n_threads) {
  // TODO: to implement
  static std::set<model_archs> supported_archs = {MODEL_GPTJ, MODEL_GPTNEOX};
  if (supported_archs.count(params.model_arch) != 0) {
    return beam_search(lctx, n_predict, tokens_inp, n_tokens, n_threads);
  } else {
    fprintf(stderr, "\nERROR: this model does not support beam search generation!\n");
    return std::vector<model_token>();
  }
}

model_token Model::post_sample_top_k_top_p_repeat(float* logits) {
  int alpha_frequency = 0;
  int alpha_presence = 0;
  int repeat_last_n = 64;
  int top_k = params.top_k;
  float tfs_z = 1.00f;
  float typical_p = 1.00f;
  float top_p = params.top_p;
  float temp = params.temp;
  std::vector<model_token_data> candidates;
  candidates.reserve(n_vocab);
  for (model_token token_id = 0; token_id < n_vocab; token_id++) {
    candidates.emplace_back(model_token_data{token_id, logits[token_id], 0.0f});
  }
  model_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

  // Apply penalties
  float nl_logit = logits[model_token_nl()];
  auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
  model_sample_repetition_penalty(ctx, &candidates_p, last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                  last_n_repeat, params.repeat_penalty);
  model_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                last_n_repeat, alpha_frequency, alpha_presence);
  // int id = model_sample_token_greedy(ctx, &candidates_p);
  // Temperature sampling
  model_sample_top_k(ctx, &candidates_p, top_k, 1);
  model_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
  model_sample_typical(ctx, &candidates_p, typical_p, 1);
  model_sample_top_p(ctx, &candidates_p, top_p, 1);
  model_sample_temperature(ctx, &candidates_p, temp);
  int id = model_sample_token(ctx, &candidates_p);
  return id;
}

model_token Model::post_process(float* logits) {
  if (params.beam_size == 1) {
    if (params.do_sample == false) {
      return post_greedy_search(logits);
    } else {
      return post_sample_top_k_top_p_repeat(logits);
    }
  }
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

#elif MODEL_NAME_ID == 12

PYBIND11_MODULE(baichuan_cpp, m)

#elif MODEL_NAME_ID == 13

PYBIND11_MODULE(polyglot_cpp, m)

#endif
{
  m.doc() = "cpp model python binding";
  py::class_<Model>(m, "Model", py::module_local())
      .def(py::init())
      .def("init_model", &Model::init_model, "initial model with model path and parameters", py::arg("model_path"),
           py::arg("max_new_tokens") = -1, py::arg("batch_size") = 512, py::arg("ctx_size") = 512, py::arg("seed") = -1,
           py::arg("threads") = 8, py::arg("repetition_penalty") = 1.1f, py::arg("num_beams") = 1,
           py::arg("do_sample") = false, py::arg("top_k") = 40, py::arg("top_p") = 0.95, py::arg("temperature") = 0.8,
           py::arg("min_new_tokens") = 0, py::arg("length_penalty") = 1.0, py::arg("early_stopping") = false,
           py::arg("n_keep") = 0, py::arg("n_discard") = -1)
      .def("generate", &Model::generate, "Generate token with input ids", py::arg("input_ids"))
      .def("generate_tokens", &Model::generate_tokens, "Generate tokens with input ids", py::arg("input_ids"))
      .def_static("quant_model", &Model::quant_model, "Quantize model", py::arg("model_path"), py::arg("out_path"),
                  py::arg("weight_dtype") = "int4", py::arg("alg") = "sym", py::arg("group_size") = 32,
                  py::arg("scale_dtype") = "fp32", py::arg("compute_dtype") = "ggml", py::arg("use_ggml") = false)
      .def("is_token_end", &Model::is_token_end)
      .def("reset_token_end", &Model::reset_token_end)
      .def("reinit", &Model::reinit);
}
