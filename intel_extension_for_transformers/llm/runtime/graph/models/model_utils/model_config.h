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
// Various helper functions and utilities

#pragma once

#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "models/model_utils/model_types.h"

#if !defined(_WIN32)
#include <stdio.h>
#include <termios.h>
#endif

struct gpt_params {
  std::string model_name;
  model_archs model_arch = MODEL_UNKNOWN;
  int n_layers;
  int32_t seed = -1;  // RNG seed
  int32_t n_threads = get_num_physical_cores();
  int32_t n_predict = -1;  // new tokens to predict
  int32_t n_ctx = 512;     // context size
  // start size to keep; n_ctx = n_keep + n_recent; refer the streaming-llm paper for details:
  // https://arxiv.org/abs/2309.17453
  int32_t n_keep = 0;
  int32_t n_batch = 512;     // batch size for prompt processing (must be >=32 to use BLAS)
  int32_t n_discard = -1;    // number of tokens to drop when reaching n_ctx
  int32_t n_gpu_layers = 0;  // number of layers to store in VRAM

  // sampling parameters
  bool do_sample = false;
  std::unordered_map<model_token, float> logit_bias;  // logit bias for specific tokens
  int32_t top_k = 40;                                 // <= 0 to use vocab size
  float top_p = 0.95f;                                // 1.0 = disabled
  float tfs_z = 1.00f;                                // 1.0 = disabled
  float typical_p = 1.00f;                            // 1.0 = disabled
  float temp = 0.80f;                                 // 1.0 = disabled
  float repeat_penalty = 1.10f;                       // 1.0 = disabled
  int32_t repeat_last_n = 64;       // last n tokens to penalize (0 = disable penalty, -1 = context size)
  float frequency_penalty = 0.00f;  // 0.0 = disabled
  float presence_penalty = 0.00f;   // 0.0 = disabled
  int mirostat = 0;                 // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
  float mirostat_tau = 5.00f;       // target entropy
  float mirostat_eta = 0.10f;       // learning rate

  std::string model = "models/7B/ne_core-model.bin";  // model path

  // if input are words
  std::string prompt = "";
  std::string path_prompt_cache = "";   // path to file for saving/loading prompt eval state
  std::string input_prefix = "";        // string to prefix user inputs with
  std::string input_suffix = "";        // string to suffix user inputs with
  std::vector<std::string> antiprompt;  // string upon seeing which more user input is prompted

  // if input are ids
  std::vector<model_token> ids;

  std::string lora_adapter = "";  // lora adapter path
  std::string lora_base = "";     // base model path for the lora adapter

  KV_MEM_TYPE memory_type = KV_MEM_TYPE_AUTO;  // Memory kv data type
  bool shift_roped_k = false;                  // whether to store non-RoPEd K cache
  bool random_prompt = false;                  // do not randomize prompt if none provided
  bool use_color = false;                      // use color to distinguish generations and inputs
  bool interactive = false;                    // interactive mode
  bool prompt_cache_all = false;               // save user input and generations to prompt cache

  bool embedding = false;          // get only sentence embedding
  bool interactive_first = false;  // wait for user input immediately
  bool multiline_input = false;    // reverse the usage of `\`

  bool instruct = false;        // instruction mode (used for Alpaca models)
  bool penalize_nl = true;      // consider newlines as a repeatable token
  bool perplexity = false;      // compute perplexity over the prompt
  bool use_mmap = false;        // use mmap for faster loads
  bool use_mlock = false;       // use mlock to keep model in memory
  bool mem_test = false;        // compute maximum memory usage
  bool verbose_prompt = false;  // print prompt tokens before generation
  int batch_size = 1;           // number batch of prompt
  bool beam_search = false;     // use beam_search or not
  int beam_size = 1;            // only valid if use beam search
};

bool gpt_params_parse(int argc, char** argv, gpt_params& params);

void gpt_print_usage(int argc, char** argv, const gpt_params& params);

//
// Vocab utils
//

std::vector<model_token> model_tokenize(struct model_context* ctx, const std::string& text, bool add_bos);

//
// Model utils
//

struct model_context* model_init_from_gpt_params(const gpt_params& params);

// KV cache elements per layer per batch per beam
void get_batch_kv_elements_from_gpt_params(int heads_kv, int head_size, int n_ctx, ne_type wtype, int32_t* k_size,
                                           int32_t* v_size);
