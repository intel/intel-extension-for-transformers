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
#include <codecvt>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <regex>   // NOLINT
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <iostream>

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

int main(int argc, char** argv) {  // NOLINT
  gpt_params params;
#ifdef MODEL_NAME
  params.model_name = MODEL_NAME;
  std::cout << "Welcome to use the " << params.model_name << " on the ITREX! " << std::endl;
#endif
  if (gpt_params_parse(argc, argv, params) == false) {
    return 1;
  }

  model_archs mt = model_name_to_arch::init().find(params.model_name);
  if (mt == MODEL_UNKNOWN) {
    fprintf(stderr, "error, please set model_name \n");
    exit(0);
  }
  params.model_arch = mt;

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
  // uncomment the "used_mem" line in graph to see the results
  if (params.mem_test) {
    {
      const std::vector<model_token> tmp(params.n_batch, ctx->vocab.bos_token_id);
      std::vector<model_input> inputs = {model_input{
          /*.tokens              =*/tmp.data(),
          /*.n_tokens           =*/(uint32_t)tmp.size(),
          /*.n_prompt_tokens    =*/0,
          /*.n_past             =*/0,
          /*.n_total            =*/0,
          /*.request_idx        =*/0,
          /*.beam_idx           =*/0,
          /*.padding_side       =*/0,
          /*n_padding           =*/0,
      }};
      model_eval(ctx, inputs.data(), inputs.size(), params.n_threads);
    }

    {
      const std::vector<model_token> tmp = {
          0,
      };
      std::vector<model_input> inputs = {model_input{
          /*.tokens              =*/tmp.data(),
          /*.n_tokens           =*/(uint32_t)tmp.size(),
          /*.n_prompt_tokens    =*/0,
          /*.n_past             =*/(uint32_t)(params.n_predict - 1),
          /*.n_total            =*/(uint32_t)(params.n_predict - 1),
          /*.request_idx        =*/0,
          /*.beam_idx           =*/0,
          /*.padding_side       =*/0,
          /*n_padding           =*/0,
      }};
      model_eval(ctx, inputs.data(), inputs.size(), params.n_threads);
    }

    model_print_timings(ctx);
    model_free(ctx);

    return 0;
  }

  // Add a space in front of the first character to match OG llama tokenizer behavior
  if (params.model_arch == MODEL_LLAMA) {
    params.prompt.insert(0, 1, ' ');
  }

  std::string path_session = params.path_prompt_cache;
  std::vector<model_token> session_tokens;

  if (!path_session.empty()) {
    fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());

    // fopen to check for existing session
    FILE* fp = std::fopen(path_session.c_str(), "rb");
    if (fp != NULL) {
      std::fclose(fp);

      session_tokens.resize(params.n_ctx);
      size_t n_token_total_out = 0;
      if (!model_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(),
                                   &n_token_total_out)) {
        fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
        return 1;
      }
      session_tokens.resize(n_token_total_out);

      fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__,
              static_cast<int>(session_tokens.size()));
    } else {
      fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
    }
  }

  // tokenize the prompt
  bool add_bos = false;
  if (params.model_arch == MODEL_LLAMA) {
    add_bos = true;
  }

  std::vector<int> embd_inp;
  if (params.model_arch == MODEL_CHATGLM2) {
    std::vector<std::string> prompts;
    prompts.push_back(params.prompt);
    std::string prompt = build_prompt_glm2(prompts);
    embd_inp = ::model_tokenize(ctx, prompt, false);
    embd_inp.insert(embd_inp.begin(), {64790, 64792});  // special prefix
  } else if (params.model_arch == MODEL_CHATGLM || params.model_arch == MODEL_BAICHUAN) {
    for (auto& i : params.ids) {
      embd_inp.emplace_back(i);
    }
  } else {
    embd_inp = ::model_tokenize(ctx, params.prompt, add_bos);
  }

  const int n_ctx = model_n_ctx(ctx);

  if (static_cast<int>(embd_inp.size()) > n_ctx - 4) {
    fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, static_cast<int>(embd_inp.size()),
            n_ctx - 4);
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
  if (params.n_keep < 0 || params.n_keep > static_cast<int>(embd_inp.size()) || params.instruct) {
    params.n_keep = static_cast<int>(embd_inp.size());
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
    for (size_t i = 0; i < embd_inp.size(); i++) {
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

  // TODO(Bo): replace with ring-buffer
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

  int n_past = 0;   // offset to which the kv-cache will be stored
  int n_total = 0;  // total number of tokens evaluated
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

  while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
    // predict
    if (embd.size() > 0) {
      // infinite text generation via context swapping
      // if we run out of context:
      // - take the n_keep first tokens from the original prompt (via n_past)
      // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
      if (n_past + static_cast<int>(embd.size()) > n_ctx) {
        // always keep the first token
        n_past = std::max(1, params.n_keep);

        int n_discard = params.n_discard;
        if (!params.shift_roped_k) {  // shift_roped_k can use ring-buffer and thus does not need re-computing
          if (n_discard == -1) n_discard = (n_ctx - embd.size() - params.n_keep) / 2;
          // drop n_discard tokens
          embd.insert(embd.begin(), last_n_tokens.begin() + params.n_keep + n_discard,
                      last_n_tokens.end() - embd.size());

          // stop saving session if we run out of context
          path_session.clear();
        } else {
          NE_ASSERT(("n_discard cannot be used with shift_roped_k!", n_discard == -1 || n_discard == 1));
        }
      }

      // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
      if (n_session_consumed < static_cast<int>(session_tokens.size())) {
        size_t i = 0;
        for (; i < embd.size(); i++) {
          if (embd[i] != session_tokens[n_session_consumed]) {
            session_tokens.resize(n_session_consumed);
            break;
          }

          n_total++;
          n_past++;
          n_session_consumed++;

          if (n_session_consumed >= static_cast<int>(session_tokens.size())) {
            ++i;
            break;
          }
        }
        if (i > 0) {
          embd.erase(embd.begin(), embd.begin() + i);
        }
      }

      // evaluate tokens in batches
      // embd is typically prepared beforehand to fit within a batch, but not always
      for (size_t i = 0; i < embd.size(); i += params.n_batch) {
        int n_eval = static_cast<int>(embd.size() - i);
        if (n_eval > params.n_batch) {
          n_eval = params.n_batch;
        }
        std::vector<model_input> inputs = {model_input{
            /*.tokens              =*/&embd[i],
            /*.n_tokens           =*/(uint32_t)n_eval,
            /*.n_prompt_tokens    =*/0,
            /*.n_past             =*/(uint32_t)n_past,
            /*.n_total            =*/(uint32_t)n_total,
            /*.request_idx        =*/0,
            /*.beam_idx           =*/0,
            /*.padding_side       =*/0,
            /*n_padding           =*/0,
        }};
        if (model_eval(ctx, inputs.data(), inputs.size(), params.n_threads)) {
          fprintf(stderr, "%s : failed to eval\n", __func__);
          return 1;
        }
        n_past += n_eval;
        n_total += n_eval;
      }

      {
        auto logits = model_get_logits(ctx);
        auto n_vocab = model_n_vocab(ctx);

        // Apply params.logit_bias map
        for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
          logits[it->first] += it->second;
        }

        std::vector<model_token_data> candidates;
        candidates.reserve(n_vocab);
        for (model_token token_id = 0; token_id < n_vocab; token_id++) {
          candidates.emplace_back(model_token_data{token_id, logits[token_id], 0.0f});
        }
        model_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

#ifdef NE_BUILD_TESTS
        std::ofstream outFile("logits.txt", std::ios::app);
        for (model_token token_id = 0; token_id < n_vocab; token_id++) {
          outFile << logits[token_id] << " ";
        }
        outFile << "\n";
#endif
        // Apply penalties
        float nl_logit = logits[model_token_nl()];
        auto last_n_repeat = std::min(std::min(static_cast<int>(last_n_tokens.size()), repeat_last_n), n_ctx);
        model_sample_repetition_penalty(ctx, &candidates_p, last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                        last_n_repeat, repeat_penalty);
        model_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                      last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                      last_n_repeat, alpha_frequency, alpha_presence);
        if (!penalize_nl) {
          logits[model_token_nl()] = nl_logit;
        }

        if (temp <= 0) {
          // Greedy sampling
          id = model_sample_token_greedy(ctx, &candidates_p);
        } else {
          if (mirostat == 1) {
            static float mirostat_mu = 2.0f * mirostat_tau;
            const int mirostat_m = 100;
            model_sample_temperature(ctx, &candidates_p, temp);
            id = model_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
          } else if (mirostat == 2) {
            static float mirostat_mu = 2.0f * mirostat_tau;
            model_sample_temperature(ctx, &candidates_p, temp);
            id = model_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
          } else {
            // Temperature sampling
            model_sample_top_k(ctx, &candidates_p, top_k, 1);
            model_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
            model_sample_typical(ctx, &candidates_p, typical_p, 1);
            model_sample_top_p(ctx, &candidates_p, top_p, 1);
            model_sample_temperature(ctx, &candidates_p, temp);
            id = model_sample_token(ctx, &candidates_p);
          }
        }
        // printf("`%d`", candidates_p.size);

        if (embd.size() > 0 && !path_session.empty()) {
          session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
          n_session_consumed = session_tokens.size();
        }
      }

      embd.clear();

      if (static_cast<int>(embd_inp.size()) <= n_consumed && !is_interacting) {
        // optionally save the session on first sample (for faster prompt loading next time)
        if (!path_session.empty() && need_to_save_session) {
          need_to_save_session = false;
          model_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
        }

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
      }

      // replace end of text token with newline token when in interactive mode
      if (id == ctx->vocab.eos_token_id && params.interactive && !params.instruct) {
        id = model_token_newline.front();
        if (params.antiprompt.size() != 0) {
          // tokenize and inject first reverse prompt
          const auto first_antiprompt = ::model_tokenize(ctx, params.antiprompt.front(), false);
          embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
        }
      }

      // add it to the context
      embd.push_back(id);

      // echo this to console
      input_echo = true;

      // decrement remaining sampling budget
      --n_remain;
    } else {
      // some user input remains from prompt or interaction, forward it to processing
      while (static_cast<int>(embd_inp.size()) > n_consumed) {
        embd.push_back(embd_inp[n_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(embd_inp[n_consumed]);
        ++n_consumed;
        if (static_cast<int>(embd.size()) >= params.n_batch) {
          break;
        }
      }
    }

    // display text
    if (params.model_arch == MODEL_CHATGLM || params.model_arch == MODEL_CHATGLM2 ||
        params.model_arch == MODEL_BAICHUAN) {
      static bool is_prompt = true;
      if (input_echo) {
        if (is_prompt == true) {
          is_prompt = false;
          continue;
        }
        for (auto id : embd) {
          std::string s(model_token_to_str(ctx, id));
          s = postprocess(s);
          std::cout << s;
        }
        fflush(stdout);
      }
    } else {
      if (input_echo) {
        for (auto id : embd) {
          printf("%s", model_token_to_str(ctx, id));
        }
        fflush(stdout);
      }
    }

    // reset color to default if we there is no pending user input
    if (input_echo && static_cast<int>(embd_inp.size()) == n_consumed) {
      console_set_color(con_st, CONSOLE_COLOR_DEFAULT);
    }

    // if not currently processing queued inputs;
    if (static_cast<int>(embd_inp.size()) <= n_consumed) {
      // check for reverse prompt
      if (params.antiprompt.size()) {
        std::string last_output;
        for (auto id : last_n_tokens) {
          last_output += model_token_to_str(ctx, id);
        }

        is_antiprompt = false;
        // Check if each of the reverse prompts appears at the end of the output.
        // If we're not running interactively, the reverse prompt might be tokenized with some following characters
        // so we'll compensate for that by widening the search window a bit.
        for (std::string& antiprompt : params.antiprompt) {
          size_t extra_padding = params.interactive ? 0 : 2;
          size_t search_start_pos =
              last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                  ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                  : 0;

          if (last_output.find(antiprompt.c_str(), search_start_pos) != std::string::npos) {
            if (params.interactive) {
              is_interacting = true;
              console_set_color(con_st, CONSOLE_COLOR_USER_INPUT);
            }
            is_antiprompt = true;
            fflush(stdout);
            break;
          }
        }
      }

      if (n_total > 0 && is_interacting) {
        if (params.instruct) {
          printf("\n> ");
        }

        std::string buffer;
        if (!params.input_prefix.empty()) {
          buffer += params.input_prefix;
          printf("%s", buffer.c_str());
        }

        std::string line;
        bool another_line = true;
        do {
          another_line = console_readline(con_st, line);
          buffer += line;
        } while (another_line);

        // done taking input, reset color
        console_set_color(con_st, CONSOLE_COLOR_DEFAULT);

        // Add tokens to embd only if the input buffer is non-empty
        // Entering a empty line lets the user pass control back
        if (buffer.length() > 1) {
          // append input suffix if any
          if (!params.input_suffix.empty()) {
            buffer += params.input_suffix;
            printf("%s", params.input_suffix.c_str());
          }

          // instruct mode: insert instruction prefix
          if (params.instruct && !is_antiprompt) {
            n_consumed = embd_inp.size();
            embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
          }

          auto line_inp = ::model_tokenize(ctx, buffer, false);
          embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

          // instruct mode: insert response suffix
          if (params.instruct) {
            embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
          }

          n_remain -= line_inp.size();
        }

        input_echo = false;  // do not echo this again
      }

      if (n_total > 0) {
        is_interacting = false;
      }
    }

    // end of text token
    if (!embd.empty() && embd.back() == ctx->vocab.eos_token_id) {
      if (params.instruct) {
        is_interacting = true;
      } else {
        fprintf(stderr, " [end of text]\n");
        break;
      }
    }

    // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
    if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
      n_remain = params.n_predict;
      is_interacting = true;
    }
  }

  if (!path_session.empty() && params.prompt_cache_all) {
    fprintf(stderr, "\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
    model_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
  }

  model_print_timings(ctx);
  model_free(ctx);

  return 0;
}  // NOLINT
