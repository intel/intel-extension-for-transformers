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

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <map>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <random>
#include <regex>
#include <thread>
#include <functional>

#include "core/data_types.h"
#include "core/ne_layers.h"
#include "models/model_utils/model_types.h"

#if !defined(_WIN32)
#include <stdio.h>
#include <termios.h>
#endif

#define COMMON_SAMPLE_RATE 16000
//
// CLI argument parsing
//

int32_t get_num_physical_cores();

struct common_params {
  int32_t n_threads = get_num_physical_cores();

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

bool common_params_parse(int argc, char** argv, common_params& params);

bool isValidFilename(const std::string& filename);

void gpt_print_usage(int argc, char** argv, const common_params& params);

std::string gpt_random_prompt(std::mt19937& rng);

std::vector<int> gpt_random_ids(std::mt19937& rng);

//
// Vocab utils
//

std::string trim(const std::string& s);

std::string replace(const std::string& s, const std::string& from, const std::string& to);

struct gpt_vocab {
  using id = int32_t;
  using token = std::string;

  std::map<token, id> token_to_id;
  std::map<id, token> id_to_token;
  std::vector<std::string> special_tokens;

  void add_special_token(const std::string& token);
};

// poor-man's JSON parsing
std::map<std::string, int32_t> json_parse(const std::string& fname);

std::string convert_to_utf8(const std::wstring& input);

std::wstring convert_to_wstring(const std::string& input);

// split text into tokens
//
// ref: https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L53
//
// Regex (Python):
// r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
//
// Regex (C++):
// R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"
//
std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab& vocab, const std::string& text);

// load the tokens from encoder.json
bool gpt_vocab_init(const std::string& fname, gpt_vocab* vocab);

// sample next token given probabilities for each embedding
//
//   - consider only the top K tokens
//   - from them, consider only the top tokens with cumulative probability > P
//
// TODO: not sure if this implementation is correct
// TODO: temperature is not implemented
//
gpt_vocab::id gpt_sample_top_k_top_p(const gpt_vocab& vocab, const float* logits, int top_k, double top_p, double temp,
                                     std::mt19937& rng);

gpt_vocab::id gpt_sample_top_k_top_p_repeat(const gpt_vocab& vocab, const float* logits,
                                            const int32_t* last_n_tokens_data, size_t last_n_tokens_data_size,
                                            int top_k, double top_p, double temp, int repeat_last_n,
                                            float repeat_penalty, std::mt19937& rng);

struct quant_params {
  std::string model_file = "";
  std::string out_file = "";
  std::string config = "";
  int nthread = 1;

  // [int4, int8, fp8_e5m2, fp8_e4m3, fp4_e2m1, nf4]
  std::string weight_dtype = "int4";
  // [sym, asym]
  std::string alg = "sym";
  // [-1, 32, 128]
  int32_t group_size = 32;
  // [fp32, bf16, fp8]
  std::string scale_dtype = "fp32";
  // [fp32, fp16, bf16, int8]
  std::string compute_dtype = "fp32";
  std::string model_name = "unknown";
  bool use_ggml = false;
  // set by model_name automatically
  model_archs model_arch = MODEL_UNKNOWN;
};

ne_ftype quant_params_to_ftype(const quant_params& params);

bool quant_params_parse(int argc, char** argv, quant_params& params);

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_BOLD "\x1b[1m"

enum console_color_t { CONSOLE_COLOR_DEFAULT = 0, CONSOLE_COLOR_PROMPT, CONSOLE_COLOR_USER_INPUT };

struct console_state {
  bool multiline_input = false;
  bool use_color = false;
  console_color_t color = CONSOLE_COLOR_DEFAULT;

  FILE* out = stdout;
#if defined(_WIN32)
  void* hConsole;
#else
  FILE* tty = nullptr;
  termios prev_state;
#endif
};

void console_init(console_state& con_st);
void console_cleanup(console_state& con_st);
void console_set_color(console_state& con_st, console_color_t color);
bool console_readline(console_state& con_st, std::string& line);

std::string build_prompt_glm2(const std::vector<std::string>& history);
std::string build_prompt_glm1(const std::vector<std::string>& history);
static std::string regex_replace(const std::string& input, const std::regex& regex,
                                 std::function<std::string(const std::smatch&)> format);
std::string postprocess(const std::string& text);
