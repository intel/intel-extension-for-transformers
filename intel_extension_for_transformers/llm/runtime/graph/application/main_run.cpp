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

std::string build_prompt(const std::vector<std::string>& history) {
  std::ostringstream oss_prompt;
  for (size_t i = 0; i < history.size(); i += 2) {
    oss_prompt << "[Round " << i / 2 + 1 << "]\n\n问：" << history[i] << "\n\n答：";
    if (i < history.size() - 1) {
      oss_prompt << history[i + 1] << "\n\n";
    }
  }
  return oss_prompt.str();
}

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

#define WhisperTest 1

#if not WhisperTest
int main(int argc, char** argv) {
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
      const std::vector<model_token> tmp(params.n_batch, model_token_bos());
      model_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
    }

    {
      const std::vector<model_token> tmp = {
          0,
      };
      model_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads);
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
  bool add_bos = false;
  if (params.model_arch == MODEL_LLAMA) {
    add_bos = true;
  }

  std::vector<int> embd_inp;
  if (params.model_arch == MODEL_CHATGLM2 || params.model_arch == MODEL_CHATGLM1) {
    std::vector<std::string> prompts;
    prompts.push_back(params.prompt);
    std::string prompt = build_prompt(prompts);
    embd_inp = ::model_tokenize(ctx, prompt, false);
    embd_inp.insert(embd_inp.begin(), {64790, 64792});  // special prefix
  } else {
    embd_inp = ::model_tokenize(ctx, params.prompt, add_bos);
  }

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

  while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
    // predict
    if (embd.size() > 0) {
      // infinite text generation via context swapping
      // if we run out of context:
      // - take the n_keep first tokens from the original prompt (via n_past)
      // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
      if (n_past + (int)embd.size() > n_ctx) {
        const int n_left = n_past - params.n_keep;

        // always keep the first token - BOS
        n_past = std::max(1, params.n_keep);

        // insert n_left/2 tokens at the start of embd from last_n_tokens
        embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(),
                    last_n_tokens.end() - embd.size());

        // stop saving session if we run out of context
        path_session.clear();

        // printf("\n---\n");
        // printf("resetting: '");
        // for (int i = 0; i < (int) embd.size(); i++) {
        //     printf("%s", model_token_to_str(ctx, embd[i]));
        // }
        // printf("'\n");
        // printf("\n---\n");
      }

      // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
      if (n_session_consumed < (int)session_tokens.size()) {
        size_t i = 0;
        for (; i < embd.size(); i++) {
          if (embd[i] != session_tokens[n_session_consumed]) {
            session_tokens.resize(n_session_consumed);
            break;
          }

          n_past++;
          n_session_consumed++;

          if (n_session_consumed >= (int)session_tokens.size()) {
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
      for (int i = 0; i < (int)embd.size(); i += params.n_batch) {
        int n_eval = (int)embd.size() - i;
        if (n_eval > params.n_batch) {
          n_eval = params.n_batch;
        }
        if (model_eval(ctx, &embd[i], n_eval, n_past, params.n_threads)) {
          fprintf(stderr, "%s : failed to eval\n", __func__);
          return 1;
        }
        n_past += n_eval;
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
        std::ofstream outFile("logits.txt", std::ios::app);
        for (model_token token_id = 0; token_id < n_vocab; token_id++) {
          outFile << logits[token_id] << " ";
          candidates.emplace_back(model_token_data{token_id, logits[token_id], 0.0f});
        }
        outFile << "\n";

        model_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

        // Apply penalties
        float nl_logit = logits[model_token_nl()];
        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
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

      if ((int)embd_inp.size() <= n_consumed && !is_interacting) {
        // optionally save the session on first sample (for faster prompt loading next time)
        if (!path_session.empty() && need_to_save_session) {
          need_to_save_session = false;
          model_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
        }

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
      }

      // replace end of text token with newline token when in interactive mode
      if (id == model_token_eos() && params.interactive && !params.instruct) {
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
      while ((int)embd_inp.size() > n_consumed) {
        embd.push_back(embd_inp[n_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(embd_inp[n_consumed]);
        ++n_consumed;
        if ((int)embd.size() >= params.n_batch) {
          break;
        }
      }
    }

    // display text
    if (input_echo) {
      for (auto id : embd) {
        printf("%s", model_token_to_str(ctx, id));
      }
      fflush(stdout);
    }
    // reset color to default if we there is no pending user input
    if (input_echo && (int)embd_inp.size() == n_consumed) {
      console_set_color(con_st, CONSOLE_COLOR_DEFAULT);
    }

    // if not currently processing queued inputs;
    if ((int)embd_inp.size() <= n_consumed) {
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

      if (n_past > 0 && is_interacting) {
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

      if (n_past > 0) {
        is_interacting = false;
      }
    }

    // end of text token
    if (!embd.empty() && embd.back() == model_token_eos()) {
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
}
#else
#include "models/whisper/whisper.h"
#include "models/whisper/dr_wav.h"
#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

// Terminal color map. 10 colors grouped in ranges [0.0, 0.1, ..., 0.9]
// Lowest is red, middle is yellow, highest is green.
const std::vector<std::string> k_colors = {
    "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
    "\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
};

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma = false) {
  int64_t msec = t * 10;
  int64_t hr = msec / (1000 * 60 * 60);
  msec = msec - hr * (1000 * 60 * 60);
  int64_t min = msec / (1000 * 60);
  msec = msec - min * (1000 * 60);
  int64_t sec = msec / 1000;
  msec = msec - sec * 1000;

  char buf[32];
  snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int)hr, (int)min, (int)sec, comma ? "," : ".", (int)msec);

  return std::string(buf);
}

int timestamp_to_sample(int64_t t, int n_samples) {
  return std::max(0, std::min((int)n_samples - 1, (int)((t * WHISPER_SAMPLE_RATE) / 100)));
}

// helper function to replace substrings
void replace_all(std::string& s, const std::string& search, const std::string& replace) {
  for (size_t pos = 0;; pos += replace.length()) {
    pos = s.find(search, pos);
    if (pos == std::string::npos) break;
    s.erase(pos, search.length());
    s.insert(pos, replace);
  }
}

// command-line parameters
struct whisper_params {
  int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
  int32_t n_processors = 1;
  int32_t offset_t_ms = 0;
  int32_t offset_n = 0;
  int32_t duration_ms = 0;
  int32_t progress_step = 5;
  int32_t max_context = -1;
  int32_t max_len = 0;
  int32_t best_of = 2;
  int32_t beam_size = -1;

  float word_thold = 0.01f;
  float entropy_thold = 2.40f;
  float logprob_thold = -1.00f;

  bool speed_up = false;
  bool debug_mode = false;
  bool translate = false;
  bool detect_language = false;
  bool diarize = false;
  bool tinydiarize = false;
  bool split_on_word = false;
  bool no_fallback = false;
  bool output_txt = false;
  bool output_vtt = false;
  bool output_srt = false;
  bool output_wts = false;
  bool output_csv = false;
  bool output_jsn = false;
  bool output_lrc = false;
  bool print_special = false;
  bool print_colors = false;
  bool print_progress = false;
  bool no_timestamps = false;
  bool log_score = false;

  std::string language = "en";
  std::string prompt;
  std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
  std::string model = "models/ggml-base.en.bin";

  // [TDRZ] speaker turn string
  std::string tdrz_speaker_turn = " [SPEAKER_TURN]";  // TODO: set from command line

  std::string openvino_encode_device = "CPU";

  std::vector<std::string> fname_inp = {};
  std::vector<std::string> fname_out = {};
};

void whisper_print_usage(int argc, char** argv, const whisper_params& params);

bool whisper_params_parse(int argc, char** argv, whisper_params& params) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-") {
      params.fname_inp.push_back(arg);
      continue;
    }

    if (arg[0] != '-') {
      params.fname_inp.push_back(arg);
      continue;
    }

    if (arg == "-h" || arg == "--help") {
      whisper_print_usage(argc, argv, params);
      exit(0);
    } else if (arg == "-t" || arg == "--threads") {
      params.n_threads = std::stoi(argv[++i]);
    } else if (arg == "-p" || arg == "--processors") {
      params.n_processors = std::stoi(argv[++i]);
    } else if (arg == "-ot" || arg == "--offset-t") {
      params.offset_t_ms = std::stoi(argv[++i]);
    } else if (arg == "-on" || arg == "--offset-n") {
      params.offset_n = std::stoi(argv[++i]);
    } else if (arg == "-d" || arg == "--duration") {
      params.duration_ms = std::stoi(argv[++i]);
    } else if (arg == "-mc" || arg == "--max-context") {
      params.max_context = std::stoi(argv[++i]);
    } else if (arg == "-ml" || arg == "--max-len") {
      params.max_len = std::stoi(argv[++i]);
    } else if (arg == "-bo" || arg == "--best-of") {
      params.best_of = std::stoi(argv[++i]);
    } else if (arg == "-bs" || arg == "--beam-size") {
      params.beam_size = std::stoi(argv[++i]);
    } else if (arg == "-wt" || arg == "--word-thold") {
      params.word_thold = std::stof(argv[++i]);
    } else if (arg == "-et" || arg == "--entropy-thold") {
      params.entropy_thold = std::stof(argv[++i]);
    } else if (arg == "-lpt" || arg == "--logprob-thold") {
      params.logprob_thold = std::stof(argv[++i]);
    }
    // else if (arg == "-su"   || arg == "--speed-up")        { params.speed_up        = true; }
    else if (arg == "-debug" || arg == "--debug-mode") {
      params.debug_mode = true;
    } else if (arg == "-tr" || arg == "--translate") {
      params.translate = true;
    } else if (arg == "-di" || arg == "--diarize") {
      params.diarize = true;
    } else if (arg == "-tdrz" || arg == "--tinydiarize") {
      params.tinydiarize = true;
    } else if (arg == "-sow" || arg == "--split-on-word") {
      params.split_on_word = true;
    } else if (arg == "-nf" || arg == "--no-fallback") {
      params.no_fallback = true;
    } else if (arg == "-otxt" || arg == "--output-txt") {
      params.output_txt = true;
    } else if (arg == "-ovtt" || arg == "--output-vtt") {
      params.output_vtt = true;
    } else if (arg == "-osrt" || arg == "--output-srt") {
      params.output_srt = true;
    } else if (arg == "-owts" || arg == "--output-words") {
      params.output_wts = true;
    } else if (arg == "-olrc" || arg == "--output-lrc") {
      params.output_lrc = true;
    } else if (arg == "-fp" || arg == "--font-path") {
      params.font_path = argv[++i];
    } else if (arg == "-ocsv" || arg == "--output-csv") {
      params.output_csv = true;
    } else if (arg == "-oj" || arg == "--output-json") {
      params.output_jsn = true;
    } else if (arg == "-of" || arg == "--output-file") {
      params.fname_out.emplace_back(argv[++i]);
    } else if (arg == "-ps" || arg == "--print-special") {
      params.print_special = true;
    } else if (arg == "-pc" || arg == "--print-colors") {
      params.print_colors = true;
    } else if (arg == "-pp" || arg == "--print-progress") {
      params.print_progress = true;
    } else if (arg == "-nt" || arg == "--no-timestamps") {
      params.no_timestamps = true;
    } else if (arg == "-l" || arg == "--language") {
      params.language = argv[++i];
    } else if (arg == "-dl" || arg == "--detect-language") {
      params.detect_language = true;
    } else if (arg == "--prompt") {
      params.prompt = argv[++i];
    } else if (arg == "-m" || arg == "--model") {
      params.model = argv[++i];
    } else if (arg == "-f" || arg == "--file") {
      params.fname_inp.emplace_back(argv[++i]);
    } else if (arg == "-oved" || arg == "--ov-e-device") {
      params.openvino_encode_device = argv[++i];
    } else if (arg == "-ls" || arg == "--log-score") {
      params.log_score = true;
    } else {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      whisper_print_usage(argc, argv, params);
      exit(0);
    }
  }

  return true;
}

void whisper_print_usage(int /*argc*/, char** argv, const whisper_params& params) {
  fprintf(stderr, "\n");
  fprintf(stderr, "usage: %s [options] file0.wav file1.wav ...\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h,        --help              [default] show this help message and exit\n");
  fprintf(stderr, "  -t N,      --threads N         [%-7d] number of threads to use during computation\n",
          params.n_threads);
  fprintf(stderr, "  -p N,      --processors N      [%-7d] number of processors to use during computation\n",
          params.n_processors);
  fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n", params.offset_t_ms);
  fprintf(stderr, "  -on N,     --offset-n N        [%-7d] segment index offset\n", params.offset_n);
  fprintf(stderr, "  -d  N,     --duration N        [%-7d] duration of audio to process in milliseconds\n",
          params.duration_ms);
  fprintf(stderr, "  -mc N,     --max-context N     [%-7d] maximum number of text context tokens to store\n",
          params.max_context);
  fprintf(stderr, "  -ml N,     --max-len N         [%-7d] maximum segment length in characters\n", params.max_len);
  fprintf(stderr, "  -sow,      --split-on-word     [%-7s] split on word rather than on token\n",
          params.split_on_word ? "true" : "false");
  fprintf(stderr, "  -bo N,     --best-of N         [%-7d] number of best candidates to keep\n", params.best_of);
  fprintf(stderr, "  -bs N,     --beam-size N       [%-7d] beam size for beam search\n", params.beam_size);
  fprintf(stderr, "  -wt N,     --word-thold N      [%-7.2f] word timestamp probability threshold\n",
          params.word_thold);
  fprintf(stderr, "  -et N,     --entropy-thold N   [%-7.2f] entropy threshold for decoder fail\n",
          params.entropy_thold);
  fprintf(stderr, "  -lpt N,    --logprob-thold N   [%-7.2f] log probability threshold for decoder fail\n",
          params.logprob_thold);
  // fprintf(stderr, "  -su,       --speed-up          [%-7s] speed up audio by x2 (reduced accuracy)\n",
  // params.speed_up ? "true" : "false");
  fprintf(stderr, "  -debug,    --debug-mode        [%-7s] enable debug mode (eg. dump log_mel)\n",
          params.debug_mode ? "true" : "false");
  fprintf(stderr, "  -tr,       --translate         [%-7s] translate from source language to english\n",
          params.translate ? "true" : "false");
  fprintf(stderr, "  -di,       --diarize           [%-7s] stereo audio diarization\n",
          params.diarize ? "true" : "false");
  fprintf(stderr, "  -tdrz,     --tinydiarize       [%-7s] enable tinydiarize (requires a tdrz model)\n",
          params.tinydiarize ? "true" : "false");
  fprintf(stderr, "  -nf,       --no-fallback       [%-7s] do not use temperature fallback while decoding\n",
          params.no_fallback ? "true" : "false");
  fprintf(stderr, "  -otxt,     --output-txt        [%-7s] output result in a text file\n",
          params.output_txt ? "true" : "false");
  fprintf(stderr, "  -ovtt,     --output-vtt        [%-7s] output result in a vtt file\n",
          params.output_vtt ? "true" : "false");
  fprintf(stderr, "  -osrt,     --output-srt        [%-7s] output result in a srt file\n",
          params.output_srt ? "true" : "false");
  fprintf(stderr, "  -olrc,     --output-lrc        [%-7s] output result in a lrc file\n",
          params.output_lrc ? "true" : "false");
  fprintf(stderr, "  -owts,     --output-words      [%-7s] output script for generating karaoke video\n",
          params.output_wts ? "true" : "false");
  fprintf(stderr, "  -fp,       --font-path         [%-7s] path to a monospace font for karaoke video\n",
          params.font_path.c_str());
  fprintf(stderr, "  -ocsv,     --output-csv        [%-7s] output result in a CSV file\n",
          params.output_csv ? "true" : "false");
  fprintf(stderr, "  -oj,       --output-json       [%-7s] output result in a JSON file\n",
          params.output_jsn ? "true" : "false");
  fprintf(stderr, "  -of FNAME, --output-file FNAME [%-7s] output file path (without file extension)\n", "");
  fprintf(stderr, "  -ps,       --print-special     [%-7s] print special tokens\n",
          params.print_special ? "true" : "false");
  fprintf(stderr, "  -pc,       --print-colors      [%-7s] print colors\n", params.print_colors ? "true" : "false");
  fprintf(stderr, "  -pp,       --print-progress    [%-7s] print progress\n", params.print_progress ? "true" : "false");
  fprintf(stderr, "  -nt,       --no-timestamps     [%-7s] do not print timestamps\n",
          params.no_timestamps ? "true" : "false");
  fprintf(stderr, "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for auto-detect)\n",
          params.language.c_str());
  fprintf(stderr, "  -dl,       --detect-language   [%-7s] exit after automatically detecting language\n",
          params.detect_language ? "true" : "false");
  fprintf(stderr, "             --prompt PROMPT     [%-7s] initial prompt\n", params.prompt.c_str());
  fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n", params.model.c_str());
  fprintf(stderr, "  -f FNAME,  --file FNAME        [%-7s] input WAV file path\n", "");
  fprintf(stderr, "  -oved D,   --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n",
          params.openvino_encode_device.c_str());
  fprintf(stderr, "  -ls,       --log-score         [%-7s] log best decoder scores of tokens\n",
          params.log_score ? "true" : "false");
  fprintf(stderr, "\n");
}

struct whisper_print_user_data {
  const whisper_params* params;

  const std::vector<std::vector<float> >* pcmf32s;
  int progress_prev;
};

std::string estimate_diarization_speaker(std::vector<std::vector<float> > pcmf32s, int64_t t0, int64_t t1,
                                         bool id_only = false) {
  std::string speaker = "";
  const int64_t n_samples = pcmf32s[0].size();

  const int64_t is0 = timestamp_to_sample(t0, n_samples);
  const int64_t is1 = timestamp_to_sample(t1, n_samples);

  double energy0 = 0.0f;
  double energy1 = 0.0f;

  for (int64_t j = is0; j < is1; j++) {
    energy0 += fabs(pcmf32s[0][j]);
    energy1 += fabs(pcmf32s[1][j]);
  }

  if (energy0 > 1.1 * energy1) {
    speaker = "0";
  } else if (energy1 > 1.1 * energy0) {
    speaker = "1";
  } else {
    speaker = "?";
  }

  // printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, speaker = %s\n", is0, is1, energy0, energy1,
  // speaker.c_str());

  if (!id_only) {
    speaker.insert(0, "(speaker ");
    speaker.append(")");
  }

  return speaker;
}
void whisper_print_progress_callback(struct whisper_context* /*ctx*/, struct whisper_state* /*state*/, int progress,
                                     void* user_data) {
  int progress_step = ((whisper_print_user_data*)user_data)->params->progress_step;
  int* progress_prev = &(((whisper_print_user_data*)user_data)->progress_prev);
  if (progress >= *progress_prev + progress_step) {
    *progress_prev += progress_step;
    fprintf(stderr, "%s: progress = %3d%%\n", __func__, progress);
  }
}

void whisper_print_segment_callback(struct whisper_context* ctx, struct whisper_state* /*state*/, int n_new,
                                    void* user_data) {
  const auto& params = *((whisper_print_user_data*)user_data)->params;
  const auto& pcmf32s = *((whisper_print_user_data*)user_data)->pcmf32s;

  const int n_segments = whisper_full_n_segments(ctx);

  std::string speaker = "";

  int64_t t0 = 0;
  int64_t t1 = 0;

  // print the last n_new segments
  const int s0 = n_segments - n_new;

  if (s0 == 0) {
    printf("\n");
  }

  for (int i = s0; i < n_segments; i++) {
    if (!params.no_timestamps || params.diarize) {
      t0 = whisper_full_get_segment_t0(ctx, i);
      t1 = whisper_full_get_segment_t1(ctx, i);
    }

    if (!params.no_timestamps) {
      printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
    }

    if (params.diarize && pcmf32s.size() == 2) {
      speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
    }

    if (params.print_colors) {
      for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
        if (params.print_special == false) {
          const whisper_token id = whisper_full_get_token_id(ctx, i, j);
          if (id >= whisper_token_eot(ctx)) {
            continue;
          }
        }

        const char* text = whisper_full_get_token_text(ctx, i, j);
        const float p = whisper_full_get_token_p(ctx, i, j);

        const int col = std::max(0, std::min((int)k_colors.size() - 1, (int)(std::pow(p, 3) * float(k_colors.size()))));

        printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
      }
    } else {
      const char* text = whisper_full_get_segment_text(ctx, i);

      printf("%s%s", speaker.c_str(), text);
    }

    if (params.tinydiarize) {
      if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
        printf("%s", params.tdrz_speaker_turn.c_str());
      }
    }

    // with timestamps or speakers: each segment on new line
    if (!params.no_timestamps || params.diarize) {
      printf("\n");
    }

    fflush(stdout);
  }
}

bool output_txt(struct whisper_context* ctx, const char* fname, const whisper_params& params,
                std::vector<std::vector<float> > pcmf32s) {
  std::ofstream fout(fname);
  if (!fout.is_open()) {
    fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname);
    return false;
  }

  fprintf(stderr, "%s: saving output to '%s'\n", __func__, fname);

  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const char* text = whisper_full_get_segment_text(ctx, i);
    std::string speaker = "";

    if (params.diarize && pcmf32s.size() == 2) {
      const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
      const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
      speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
    }

    fout << speaker << text << "\n";
  }

  return true;
}

bool output_vtt(struct whisper_context* ctx, const char* fname, const whisper_params& params,
                std::vector<std::vector<float> > pcmf32s) {
  std::ofstream fout(fname);
  if (!fout.is_open()) {
    fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname);
    return false;
  }

  fprintf(stderr, "%s: saving output to '%s'\n", __func__, fname);

  fout << "WEBVTT\n\n";

  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const char* text = whisper_full_get_segment_text(ctx, i);
    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
    std::string speaker = "";

    if (params.diarize && pcmf32s.size() == 2) {
      speaker = estimate_diarization_speaker(pcmf32s, t0, t1, true);
      speaker.insert(0, "<v Speaker");
      speaker.append(">");
    }

    fout << to_timestamp(t0) << " --> " << to_timestamp(t1) << "\n";
    fout << speaker << text << "\n\n";
  }

  return true;
}

bool output_srt(struct whisper_context* ctx, const char* fname, const whisper_params& params,
                std::vector<std::vector<float> > pcmf32s) {
  std::ofstream fout(fname);
  if (!fout.is_open()) {
    fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname);
    return false;
  }

  fprintf(stderr, "%s: saving output to '%s'\n", __func__, fname);

  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const char* text = whisper_full_get_segment_text(ctx, i);
    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
    std::string speaker = "";

    if (params.diarize && pcmf32s.size() == 2) {
      speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
    }

    fout << i + 1 + params.offset_n << "\n";
    fout << to_timestamp(t0, true) << " --> " << to_timestamp(t1, true) << "\n";
    fout << speaker << text << "\n\n";
  }

  return true;
}

char* escape_double_quotes_and_backslashes(const char* str) {
  if (str == NULL) {
    return NULL;
  }

  size_t escaped_length = strlen(str) + 1;

  for (size_t i = 0; str[i] != '\0'; i++) {
    if (str[i] == '"' || str[i] == '\\') {
      escaped_length++;
    }
  }

  char* escaped = (char*)calloc(escaped_length, 1);  // pre-zeroed
  if (escaped == NULL) {
    return NULL;
  }

  size_t pos = 0;
  for (size_t i = 0; str[i] != '\0'; i++) {
    if (str[i] == '"' || str[i] == '\\') {
      escaped[pos++] = '\\';
    }
    escaped[pos++] = str[i];
  }

  // no need to set zero due to calloc() being used prior

  return escaped;
}

bool output_csv(struct whisper_context* ctx, const char* fname, const whisper_params& params,
                std::vector<std::vector<float> > pcmf32s) {
  std::ofstream fout(fname);
  if (!fout.is_open()) {
    fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname);
    return false;
  }

  fprintf(stderr, "%s: saving output to '%s'\n", __func__, fname);

  const int n_segments = whisper_full_n_segments(ctx);
  fout << "start,end,";
  if (params.diarize && pcmf32s.size() == 2) {
    fout << "speaker,";
  }
  fout << "text\n";

  for (int i = 0; i < n_segments; ++i) {
    const char* text = whisper_full_get_segment_text(ctx, i);
    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
    char* text_escaped = escape_double_quotes_and_backslashes(text);

    // need to multiply times returned from whisper_full_get_segment_t{0,1}() by 10 to get milliseconds.
    fout << 10 * t0 << "," << 10 * t1 << ",";
    if (params.diarize && pcmf32s.size() == 2) {
      fout << estimate_diarization_speaker(pcmf32s, t0, t1, true) << ",";
    }
    fout << "\"" << text_escaped << "\"\n";
  }

  return true;
}

bool output_score(struct whisper_context* ctx, const char* fname, const whisper_params& /*params*/,
                  std::vector<std::vector<float> > /*pcmf32s*/) {
  std::ofstream fout(fname);
  fprintf(stderr, "%s: saving output to '%s'\n", __func__, fname);

  const int n_segments = whisper_full_n_segments(ctx);
  // fprintf(stderr,"segments: %d\n",n_segments);
  for (int i = 0; i < n_segments; ++i) {
    const int n_tokens = whisper_full_n_tokens(ctx, i);
    // fprintf(stderr,"tokens: %d\n",n_tokens);
    for (int j = 0; j < n_tokens; j++) {
      auto token = whisper_full_get_token_text(ctx, i, j);
      auto probability = whisper_full_get_token_p(ctx, i, j);
      fout << token << '\t' << probability << std::endl;
      // fprintf(stderr,"token: %s %f\n",token,probability);
    }
  }
  return true;
}

bool output_json(struct whisper_context* ctx, const char* fname, const whisper_params& params,
                 std::vector<std::vector<float> > pcmf32s) {
  std::ofstream fout(fname);
  int indent = 0;

  auto doindent = [&]() {
    for (int i = 0; i < indent; i++) fout << "\t";
  };

  auto start_arr = [&](const char* name) {
    doindent();
    fout << "\"" << name << "\": [\n";
    indent++;
  };

  auto end_arr = [&](bool end) {
    indent--;
    doindent();
    fout << (end ? "]\n" : "},\n");
  };

  auto start_obj = [&](const char* name) {
    doindent();
    if (name) {
      fout << "\"" << name << "\": {\n";
    } else {
      fout << "{\n";
    }
    indent++;
  };

  auto end_obj = [&](bool end) {
    indent--;
    doindent();
    fout << (end ? "}\n" : "},\n");
  };

  auto start_value = [&](const char* name) {
    doindent();
    fout << "\"" << name << "\": ";
  };

  auto value_s = [&](const char* name, const char* val, bool end) {
    start_value(name);
    char* val_escaped = escape_double_quotes_and_backslashes(val);
    fout << "\"" << val_escaped << (end ? "\"\n" : "\",\n");
    free(val_escaped);
  };

  auto end_value = [&](bool end) { fout << (end ? "\n" : ",\n"); };

  auto value_i = [&](const char* name, const int64_t val, bool end) {
    start_value(name);
    fout << val;
    end_value(end);
  };

  auto value_b = [&](const char* name, const bool val, bool end) {
    start_value(name);
    fout << (val ? "true" : "false");
    end_value(end);
  };

  if (!fout.is_open()) {
    fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname);
    return false;
  }

  fprintf(stderr, "%s: saving output to '%s'\n", __func__, fname);
  start_obj(nullptr);
  value_s("systeminfo", whisper_print_system_info(), false);
  start_obj("model");
  value_s("type", whisper_model_type_readable(ctx), false);
  value_b("multilingual", whisper_is_multilingual(ctx), false);
  value_i("vocab", whisper_model_n_vocab(ctx), false);
  start_obj("audio");
  value_i("ctx", whisper_model_n_audio_ctx(ctx), false);
  value_i("state", whisper_model_n_audio_state(ctx), false);
  value_i("head", whisper_model_n_audio_head(ctx), false);
  value_i("layer", whisper_model_n_audio_layer(ctx), true);
  end_obj(false);
  start_obj("text");
  value_i("ctx", whisper_model_n_text_ctx(ctx), false);
  value_i("state", whisper_model_n_text_state(ctx), false);
  value_i("head", whisper_model_n_text_head(ctx), false);
  value_i("layer", whisper_model_n_text_layer(ctx), true);
  end_obj(false);
  value_i("mels", whisper_model_n_mels(ctx), false);
  value_i("ftype", whisper_model_ftype(ctx), true);
  end_obj(false);
  start_obj("params");
  value_s("model", params.model.c_str(), false);
  value_s("language", params.language.c_str(), false);
  value_b("translate", params.translate, true);
  end_obj(false);
  start_obj("result");
  value_s("language", whisper_lang_str(whisper_full_lang_id(ctx)), true);
  end_obj(false);
  start_arr("transcription");

  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const char* text = whisper_full_get_segment_text(ctx, i);

    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

    start_obj(nullptr);
    start_obj("timestamps");
    value_s("from", to_timestamp(t0, true).c_str(), false);
    value_s("to", to_timestamp(t1, true).c_str(), true);
    end_obj(false);
    start_obj("offsets");
    value_i("from", t0 * 10, false);
    value_i("to", t1 * 10, true);
    end_obj(false);
    value_s("text", text, !params.diarize && !params.tinydiarize);

    if (params.diarize && pcmf32s.size() == 2) {
      value_s("speaker", estimate_diarization_speaker(pcmf32s, t0, t1, true).c_str(), true);
    }

    if (params.tinydiarize) {
      value_b("speaker_turn_next", whisper_full_get_segment_speaker_turn_next(ctx, i), true);
    }
    end_obj(i == (n_segments - 1));
  }

  end_arr(true);
  end_obj(true);
  return true;
}

// karaoke video generation
// outputs a bash script that uses ffmpeg to generate a video with the subtitles
// TODO: font parameter adjustments
bool output_wts(struct whisper_context* ctx, const char* fname, const char* fname_inp, const whisper_params& params,
                float t_sec, std::vector<std::vector<float> > pcmf32s) {
  std::ofstream fout(fname);

  fprintf(stderr, "%s: saving output to '%s'\n", __func__, fname);

  static const char* font = params.font_path.c_str();

  std::ifstream fin(font);
  if (!fin.is_open()) {
    fprintf(stderr, "%s: font not found at '%s', please specify a monospace font with -fp\n", __func__, font);
    return false;
  }

  fout << "#!/bin/bash"
       << "\n";
  fout << "\n";

  fout << "ffmpeg -i " << fname_inp << " -f lavfi -i color=size=1200x120:duration=" << t_sec
       << ":rate=25:color=black -vf \"";

  for (int i = 0; i < whisper_full_n_segments(ctx); i++) {
    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

    const int n = whisper_full_n_tokens(ctx, i);

    std::vector<whisper_token_data> tokens(n);
    for (int j = 0; j < n; ++j) {
      tokens[j] = whisper_full_get_token_data(ctx, i, j);
    }

    if (i > 0) {
      fout << ",";
    }

    // background text
    fout << "drawtext=fontfile='" << font
         << "':fontsize=24:fontcolor=gray:x=(w-text_w)/2:y=h/2:text='':enable='between(t," << t0 / 100.0 << ","
         << t0 / 100.0 << ")'";

    bool is_first = true;
    std::string speaker = "";

    if (params.diarize && pcmf32s.size() == 2) {
      speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
    }

    for (int j = 0; j < n; ++j) {
      const auto& token = tokens[j];

      if (tokens[j].id >= whisper_token_eot(ctx)) {
        continue;
      }

      std::string txt_bg = "";
      std::string txt_fg = "";  // highlight token
      std::string txt_ul = "";  // underline

      if (params.diarize && pcmf32s.size() == 2) {
        txt_bg = speaker;
        txt_fg = speaker;
        txt_ul = "\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ";
      }

      txt_bg.append("> ");
      txt_fg.append("> ");
      txt_ul.append("\\ \\ ");

      {
        for (int k = 0; k < n; ++k) {
          const auto& token2 = tokens[k];

          if (tokens[k].id >= whisper_token_eot(ctx)) {
            continue;
          }

          const std::string txt = whisper_token_to_str(ctx, token2.id);

          txt_bg += txt;

          if (k == j) {
            for (int l = 0; l < (int)txt.size(); ++l) {
              txt_fg += txt[l];
              txt_ul += "_";
            }
            txt_fg += "|";
          } else {
            for (int l = 0; l < (int)txt.size(); ++l) {
              txt_fg += "\\ ";
              txt_ul += "\\ ";
            }
          }
        }

        ::replace_all(txt_bg, "'", "\u2019");
        ::replace_all(txt_bg, "\"", "\\\"");
        ::replace_all(txt_fg, "'", "\u2019");
        ::replace_all(txt_fg, "\"", "\\\"");
      }

      if (is_first) {
        // background text
        fout << ",drawtext=fontfile='" << font << "':fontsize=24:fontcolor=gray:x=(w-text_w)/2:y=h/2:text='" << txt_bg
             << "':enable='between(t," << t0 / 100.0 << "," << t1 / 100.0 << ")'";
        is_first = false;
      }

      // foreground text
      fout << ",drawtext=fontfile='" << font << "':fontsize=24:fontcolor=lightgreen:x=(w-text_w)/2+8:y=h/2:text='"
           << txt_fg << "':enable='between(t," << token.t0 / 100.0 << "," << token.t1 / 100.0 << ")'";

      // underline
      fout << ",drawtext=fontfile='" << font << "':fontsize=24:fontcolor=lightgreen:x=(w-text_w)/2+8:y=h/2+16:text='"
           << txt_ul << "':enable='between(t," << token.t0 / 100.0 << "," << token.t1 / 100.0 << ")'";
    }
  }

  fout << "\" -c:v libx264 -pix_fmt yuv420p -y " << fname_inp << ".mp4"
       << "\n";

  fout << "\n\n";
  fout << "echo \"Your video has been saved to " << fname_inp << ".mp4\""
       << "\n";
  fout << "\n";
  fout << "echo \"  ffplay " << fname_inp << ".mp4\"\n";
  fout << "\n";

  fout.close();

  fprintf(stderr, "%s: run 'source %s' to generate karaoke video\n", __func__, fname);

  return true;
}

bool output_lrc(struct whisper_context* ctx, const char* fname, const whisper_params& params,
                std::vector<std::vector<float> > pcmf32s) {
  std::ofstream fout(fname);
  if (!fout.is_open()) {
    fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname);
    return false;
  }

  fprintf(stderr, "%s: saving output to '%s'\n", __func__, fname);

  fout << "[by:whisper.cpp]\n";

  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const char* text = whisper_full_get_segment_text(ctx, i);
    const int64_t t = whisper_full_get_segment_t0(ctx, i);

    int64_t msec = t * 10;
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[16];
    snprintf(buf, sizeof(buf), "%02d:%02d.%02d", (int)min, (int)sec, (int)(msec / 10));
    std::string timestamp_lrc = std::string(buf);
    std::string speaker = "";

    if (params.diarize && pcmf32s.size() == 2) {
      const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
      const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
      speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
    }

    fout << '[' << timestamp_lrc << ']' << speaker << text << "\n";
  }

  return true;
}
bool read_wav(const std::string& fname, std::vector<float>& pcmf32, std::vector<std::vector<float> >& pcmf32s,
              bool stereo) {
  drwav wav;
  std::vector<uint8_t> wav_data;  // used for pipe input from stdin

  if (fname == "-") {
    {
      uint8_t buf[1024];
      while (true) {
        const size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n == 0) {
          break;
        }
        wav_data.insert(wav_data.end(), buf, buf + n);
      }
    }

    if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false) {
      fprintf(stderr, "error: failed to open WAV file from stdin\n");
      return false;
    }

    fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, wav_data.size());
  } else if (drwav_init_file(&wav, fname.c_str(), nullptr) == false) {
    fprintf(stderr, "error: failed to open '%s' as WAV file\n", fname.c_str());
    return false;
  }

  if (wav.channels != 1 && wav.channels != 2) {
    fprintf(stderr, "%s: WAV file '%s' must be mono or stereo\n", __func__, fname.c_str());
    return false;
  }

  if (stereo && wav.channels != 2) {
    fprintf(stderr, "%s: WAV file '%s' must be stereo for diarization\n", __func__, fname.c_str());
    return false;
  }

  if (wav.sampleRate != COMMON_SAMPLE_RATE) {
    fprintf(stderr, "%s: WAV file '%s' must be %i kHz\n", __func__, fname.c_str(), COMMON_SAMPLE_RATE / 1000);
    return false;
  }

  if (wav.bitsPerSample != 16) {
    fprintf(stderr, "%s: WAV file '%s' must be 16-bit\n", __func__, fname.c_str());
    return false;
  }

  const uint64_t n =
      wav_data.empty() ? wav.totalPCMFrameCount : wav_data.size() / (wav.channels * wav.bitsPerSample / 8);

  std::vector<int16_t> pcm16;
  pcm16.resize(n * wav.channels);
  drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
  drwav_uninit(&wav);

  // convert to mono, float
  pcmf32.resize(n);
  if (wav.channels == 1) {
    for (uint64_t i = 0; i < n; i++) {
      pcmf32[i] = float(pcm16[i]) / 32768.0f;
    }
  } else {
    for (uint64_t i = 0; i < n; i++) {
      pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
    }
  }

  if (stereo) {
    // convert to stereo, float
    pcmf32s.resize(2);

    pcmf32s[0].resize(n);
    pcmf32s[1].resize(n);
    for (uint64_t i = 0; i < n; i++) {
      pcmf32s[0][i] = float(pcm16[2 * i]) / 32768.0f;
      pcmf32s[1][i] = float(pcm16[2 * i + 1]) / 32768.0f;
    }
  }

  return true;
}

int main(int argc, char** argv) {
  whisper_params params;

  if (whisper_params_parse(argc, argv, params) == false) {
    whisper_print_usage(argc, argv, params);
    return 1;
  }

  if (params.fname_inp.empty()) {
    fprintf(stderr, "error: no input files specified\n");
    whisper_print_usage(argc, argv, params);
    return 2;
  }

  if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
    fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
    whisper_print_usage(argc, argv, params);
    exit(0);
  }

  if (params.diarize && params.tinydiarize) {
    fprintf(stderr, "error: cannot use both --diarize and --tinydiarize\n");
    whisper_print_usage(argc, argv, params);
    exit(0);
  }

  // whisper init

  struct whisper_context* ctx = whisper_init_from_file(params.model.c_str());

  if (ctx == nullptr) {
    fprintf(stderr, "error: failed to initialize whisper context\n");
    return 3;
  }

  // initialize openvino encoder. this has no effect on whisper.cpp builds that don't have OpenVINO configured
  whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);

  for (int f = 0; f < (int)params.fname_inp.size(); ++f) {
    const auto fname_inp = params.fname_inp[f];
    const auto fname_out =
        f < (int)params.fname_out.size() && !params.fname_out[f].empty() ? params.fname_out[f] : params.fname_inp[f];

    std::vector<float> pcmf32;                 // mono-channel F32 PCM
    std::vector<std::vector<float> > pcmf32s;  // stereo-channel F32 PCM

    if (!read_wav(fname_inp, pcmf32, pcmf32s, params.diarize)) {
      fprintf(stderr, "error: failed to read WAV file '%s'\n", fname_inp.c_str());
      continue;
    }

    // print system information
    {
      fprintf(stderr, "\n");
      fprintf(stderr, "system_info: n_threads = %d / %d | %s\n", params.n_threads * params.n_processors,
              std::thread::hardware_concurrency(), whisper_print_system_info());
    }

    // print some info about the processing
    {
      fprintf(stderr, "\n");
      if (!whisper_is_multilingual(ctx)) {
        if (params.language != "en" || params.translate) {
          params.language = "en";
          params.translate = false;
          fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n",
                  __func__);
        }
      }
      if (params.detect_language) {
        params.language = "auto";
      }
      fprintf(stderr,
              "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, lang = %s, task = %s, "
              "%stimestamps = %d ...\n",
              __func__, fname_inp.c_str(), int(pcmf32.size()), float(pcmf32.size()) / WHISPER_SAMPLE_RATE,
              params.n_threads, params.n_processors, params.language.c_str(),
              params.translate ? "translate" : "transcribe", params.tinydiarize ? "tdrz = 1, " : "",
              params.no_timestamps ? 0 : 1);

      fprintf(stderr, "\n");
    }

    // run the inference
    {
      whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

      wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

      wparams.print_realtime = false;
      wparams.print_progress = params.print_progress;
      wparams.print_timestamps = !params.no_timestamps;
      wparams.print_special = params.print_special;
      wparams.translate = params.translate;
      wparams.language = params.language.c_str();
      wparams.detect_language = params.detect_language;
      wparams.n_threads = params.n_threads;
      wparams.n_max_text_ctx = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
      wparams.offset_ms = params.offset_t_ms;
      wparams.duration_ms = params.duration_ms;

      wparams.token_timestamps = params.output_wts || params.max_len > 0;
      wparams.thold_pt = params.word_thold;
      wparams.max_len = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
      wparams.split_on_word = params.split_on_word;

      wparams.speed_up = params.speed_up;
      wparams.debug_mode = params.debug_mode;

      wparams.tdrz_enable = params.tinydiarize;  // [TDRZ]

      wparams.initial_prompt = params.prompt.c_str();

      wparams.greedy.best_of = params.best_of;
      wparams.beam_search.beam_size = params.beam_size;

      wparams.temperature_inc = params.no_fallback ? 0.0f : wparams.temperature_inc;
      wparams.entropy_thold = params.entropy_thold;
      wparams.logprob_thold = params.logprob_thold;

      whisper_print_user_data user_data = {&params, &pcmf32s, 0};

      // this callback is called on each new segment
      if (!wparams.print_realtime) {
        wparams.new_segment_callback = whisper_print_segment_callback;
        wparams.new_segment_callback_user_data = &user_data;
      }

      if (wparams.print_progress) {
        wparams.progress_callback = whisper_print_progress_callback;
        wparams.progress_callback_user_data = &user_data;
      }

      // example for abort mechanism
      // in this example, we do not abort the processing, but we could if the flag is set to true
      // the callback is called before every encoder run - if it returns false, the processing is aborted
      {
        static bool is_aborted = false;  // NOTE: this should be atomic to avoid data race

        wparams.encoder_begin_callback = [](struct whisper_context* /*ctx*/, struct whisper_state* /*state*/,
                                            void* user_data) {
          bool is_aborted = *(bool*)user_data;
          return !is_aborted;
        };
        wparams.encoder_begin_callback_user_data = &is_aborted;
      }

      if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
        fprintf(stderr, "%s: failed to process audio\n", argv[0]);
        return 10;
      }
    }

    // output stuff
    {
      printf("\n");

      // output to text file
      if (params.output_txt) {
        const auto fname_txt = fname_out + ".txt";
        output_txt(ctx, fname_txt.c_str(), params, pcmf32s);
      }

      // output to VTT file
      if (params.output_vtt) {
        const auto fname_vtt = fname_out + ".vtt";
        output_vtt(ctx, fname_vtt.c_str(), params, pcmf32s);
      }

      // output to SRT file
      if (params.output_srt) {
        const auto fname_srt = fname_out + ".srt";
        output_srt(ctx, fname_srt.c_str(), params, pcmf32s);
      }

      // output to WTS file
      if (params.output_wts) {
        const auto fname_wts = fname_out + ".wts";
        output_wts(ctx, fname_wts.c_str(), fname_inp.c_str(), params, float(pcmf32.size() + 1000) / WHISPER_SAMPLE_RATE,
                   pcmf32s);
      }

      // output to CSV file
      if (params.output_csv) {
        const auto fname_csv = fname_out + ".csv";
        output_csv(ctx, fname_csv.c_str(), params, pcmf32s);
      }

      // output to JSON file
      if (params.output_jsn) {
        const auto fname_jsn = fname_out + ".json";
        output_json(ctx, fname_jsn.c_str(), params, pcmf32s);
      }

      // output to LRC file
      if (params.output_lrc) {
        const auto fname_lrc = fname_out + ".lrc";
        output_lrc(ctx, fname_lrc.c_str(), params, pcmf32s);
      }

      // output to score file
      if (params.log_score) {
        const auto fname_score = fname_out + ".score.txt";
        output_score(ctx, fname_score.c_str(), params, pcmf32s);
      }
    }
  }

  whisper_print_timings(ctx);
  whisper_free(ctx);

  return 0;
}

#endif