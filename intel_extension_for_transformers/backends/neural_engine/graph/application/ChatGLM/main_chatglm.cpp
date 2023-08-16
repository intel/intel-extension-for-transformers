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

static console_state con_st;
static model_context** g_ctx;

static bool is_interacting = false;

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
void sigint_handler(int signo) {
  if (signo == SIGINT) {
    if (!is_interacting) {
      is_interacting = true;
    } else {
      console_cleanup(con_st);
      printf("\n");
      model_print_timings(*g_ctx);
      _exit(130);
    }
  }
}
#endif

int main(int argc, char** argv) {
  gpt_params params;
  params.name = MODEL_CHATGLM;
  if (gpt_params_parse(argc, argv, params) == false) {
    return 1;
  }

  // save choice to use color for later
  // (note for later: this is a slightly awkward choice)
  con_st.use_color = params.use_color;
  con_st.multiline_input = params.multiline_input;
  console_init(con_st);
  atexit([]() { console_cleanup(con_st); });

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

  fprintf(stderr, "%s: seed  = %d\n", __func__, params.seed);

  std::mt19937 rng(params.seed);
  if (params.random_prompt) {
    params.prompt = gpt_random_prompt(rng);
  }

  model_init_backend();

  model_context* ctx;
  g_ctx = &ctx;

  // load the model and apply lora adapter, if any
  ctx = model_init_from_gpt_params(params);
  if (ctx == NULL) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return 1;
  }

  // print system information
  {
    fprintf(stderr, "\n");
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n", params.n_threads, std::thread::hardware_concurrency(),
            model_print_system_info());
  }

  // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
  // uncomment the "used_mem" line in gptj.cpp to see the results
  // if (params.mem_test) {
  //   {
  //     const std::vector<model_token> tmp(params.n_batch, model_token_bos());
  //     model_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
  //   }

  //   {
  //     const std::vector<model_token> tmp = {
  //         0,
  //     };
  //     model_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads);
  //   }

  //   model_print_timings(ctx);
  //   model_free(ctx);

  //   return 0;
  // }

  // Add a space in front of the first character to match OG gptj tokenizer behavior
  // params.prompt.insert(0, 1, ' ');

  std::string path_session = params.path_prompt_cache;
  std::vector<model_token> session_tokens;

  if (!path_session.empty()) {
    fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());

    // fopen to check for existing session
    FILE* fp = std::fopen(path_session.c_str(), "rb");
    if (fp != NULL) {
      std::fclose(fp);

      session_tokens.resize(params.n_ctx);
      size_t n_token_count_out = 0;
      if (!model_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(),
                                   &n_token_count_out)) {
        fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
        return 1;
      }
      session_tokens.resize(n_token_count_out);

      fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
    } else {
      fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
    }
  }

  // tokenize the prompt
  // std::vector<int> embd_inp = ::model_tokenize(ctx, params.prompt, false);
  std::vector<int> embd_inp = ctx->model.tokenizer->encode_history({params.prompt}, 512);

  const int n_ctx = model_n_ctx(ctx);

  if ((int)embd_inp.size() > n_ctx - 4) {
    fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)embd_inp.size(), n_ctx - 4);
    return 1;
  }

  // debug message about similarity of saved session, if applicable
  size_t n_matching_session_tokens = 0;
  if (session_tokens.size()) {
    for (model_token id : session_tokens) {
      if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
        break;
      }
      n_matching_session_tokens++;
    }
    if (n_matching_session_tokens >= embd_inp.size()) {
      fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
    } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
      fprintf(stderr,
              "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
              __func__, n_matching_session_tokens, embd_inp.size());
    } else {
      fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n", __func__, n_matching_session_tokens,
              embd_inp.size());
    }
  }

  // number of tokens to keep when resetting context
  if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size() || params.instruct) {
    params.n_keep = (int)embd_inp.size();
  }

  // prefix & suffix for instruct mode
  const auto inp_pfx = ::model_tokenize(ctx, "\n\n### Instruction:\n\n", true);
  const auto inp_sfx = ::model_tokenize(ctx, "\n\n### Response:\n\n", false);

  // in instruct mode, we inject a prefix and a suffix to each input by the user
  if (params.instruct) {
    params.interactive_first = true;
    params.antiprompt.push_back("### Instruction:\n\n");
  }

  // enable interactive mode if interactive start is specified
  if (params.interactive_first) {
    params.interactive = true;
  }

  // determine newline token
  auto model_token_newline = ::model_tokenize(ctx, "\n", false);

  if (params.verbose_prompt) {
    fprintf(stderr, "\n");
    fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    for (int i = 0; i < (int)embd_inp.size(); i++) {
      fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], model_token_to_str(ctx, embd_inp[i]));
    }
    if (params.n_keep > 0) {
      fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
      for (int i = 0; i < params.n_keep; i++) {
        fprintf(stderr, "%s", model_token_to_str(ctx, embd_inp[i]));
      }
      fprintf(stderr, "'\n");
    }
    fprintf(stderr, "\n");
  }

  if (params.interactive) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler =
        +[](DWORD ctrl_type) -> BOOL { return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false; };
    SetConsoleCtrlHandler(static_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    fprintf(stderr, "%s: interactive mode on.\n", __func__);

    if (params.antiprompt.size()) {
      for (auto antiprompt : params.antiprompt) {
        fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
      }
    }

    if (!params.input_prefix.empty()) {
      fprintf(stderr, "Input prefix: '%s'\n", params.input_prefix.c_str());
    }

    if (!params.input_suffix.empty()) {
      fprintf(stderr, "Input suffix: '%s'\n", params.input_suffix.c_str());
    }
  }
  fprintf(stderr,
          "sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty = %f, frequency_penalty = %f, top_k = "
          "%d, tfs_z = %f, top_p = %f, typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, mirostat_ent = %f\n",
          params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k,
          params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta,
          params.mirostat_tau);
  fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch,
          params.n_predict, params.n_keep);
  fprintf(stderr, "\n\n");

  // TODO: replace with ring-buffer
  std::vector<model_token> last_n_tokens(n_ctx);
  std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

  if (params.interactive) {
    const char* control_message;
    if (con_st.multiline_input) {
      control_message =
          " - To return control to LLaMa, end your input with '\\'.\n"
          " - To return control without starting a new line, end your input with '/'.\n";
    } else {
      control_message =
          " - Press Return to return control to LLaMa.\n"
          " - To return control without starting a new line, end your input with '/'.\n"
          " - If you want to submit another line, end your input with '\\'.\n";
    }
    fprintf(stderr,
            "== Running in interactive mode. ==\n"
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
            " - Press Ctrl+C to interject at any time.\n"
#endif
            "%s\n",
            control_message);

    is_interacting = params.interactive_first;
  }

  bool is_antiprompt = false;
  bool input_echo = true;
  bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

  int n_past = 0;
  int n_remain = params.n_predict;
  int n_consumed = 0;
  int n_session_consumed = 0;

  // the first thing we will do is to output the prompt, so set color accordingly
  console_set_color(con_st, CONSOLE_COLOR_PROMPT);

  std::vector<model_token> embd;
  // out of user input, sample next token
  const float temp = params.temp;
  const int32_t top_k = params.top_k <= 0 ? model_n_vocab(ctx) : params.top_k;
  const float top_p = params.top_p;
  const float tfs_z = params.tfs_z;
  const float typical_p = params.typical_p;
  const int32_t repeat_last_n = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
  const float repeat_penalty = params.repeat_penalty;
  const float alpha_presence = params.presence_penalty;
  const float alpha_frequency = params.frequency_penalty;
  const int mirostat = params.mirostat;
  const float mirostat_tau = params.mirostat_tau;
  const float mirostat_eta = params.mirostat_eta;
  const bool penalize_nl = params.penalize_nl;
  model_token id = 0;

  int max_length = 512;
  // int n_past = 0;
  int n_eval = embd_inp.size();
  std::vector<int> curr_input_ids(embd_inp);
  std::vector<int> output_ids;
  output_ids.reserve(max_length);
  int vocab_size = 65024;

  while ((int)output_ids.size() < n_remain) {
    model_eval(ctx, &curr_input_ids[0], curr_input_ids.size(), n_past, params.n_threads);
    n_past += curr_input_ids.size();

    float* logits = model_get_logits(ctx);
    int next_token_id = std::max_element(logits, logits + vocab_size) - logits;
    curr_input_ids = {next_token_id};

    output_ids.emplace_back(next_token_id);
    printf("%s", ctx->model.tokenizer->decode({next_token_id}).c_str());

    fflush(stdout);

    if (next_token_id == ctx->model.tokenizer->eos_token_id) {
      break;
    }
  }
  printf("\n");


  model_print_timings(ctx);
  model_free(ctx);

  return 0;
}
