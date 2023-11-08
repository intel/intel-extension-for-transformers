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

bool isValidFilename(const std::string& filename);

std::string gpt_random_prompt(std::mt19937& rng);

struct quant_params {
  std::string model_file = "";
  std::string out_file = "";
  std::string config = "";
  int nthread = 1;

  std::string weight_dtype = "int4";
  std::string alg = "sym";
  int32_t group_size = 32;
  std::string scale_dtype = "fp32";
  std::string compute_dtype = "int8";
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
