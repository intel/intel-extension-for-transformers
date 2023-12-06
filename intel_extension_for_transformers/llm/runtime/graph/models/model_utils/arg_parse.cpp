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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"

void process_escapes(std::string& input) {  // NOLINT
  std::size_t input_len = input.length();
  std::size_t output_idx = 0;

  for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
    if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
      switch (input[++input_idx]) {
        case 'n':
          input[output_idx++] = '\n';
          break;
        case 'r':
          input[output_idx++] = '\r';
          break;
        case 't':
          input[output_idx++] = '\t';
          break;
        case '\'':
          input[output_idx++] = '\'';
          break;
        case '\"':
          input[output_idx++] = '\"';
          break;
        case '\\':
          input[output_idx++] = '\\';
          break;
        default:
          input[output_idx++] = '\\';
          input[output_idx++] = input[input_idx];
          break;
      }
    } else {
      input[output_idx++] = input[input_idx];
    }
  }

  input.resize(output_idx);
}

bool gpt_params_parse(int argc, char** argv, gpt_params& params) {  // NOLINT
  bool invalid_param = false;
  bool escape_prompt = false;
  std::string arg;
  gpt_params default_params;
  const std::string arg_prefix = "--";

  for (int i = 1; i < argc; i++) {
    arg = argv[i];
    if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
      std::replace(arg.begin(), arg.end(), '_', '-');
    }

    if (arg == "-s" || arg == "--seed") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.seed = std::stoi(argv[i]);
    } else if (arg == "-t" || arg == "--threads") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.n_threads = std::stoi(argv[i]);
    } else if (arg == "-p" || arg == "--prompt") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.prompt = argv[i];
    } else if (arg == "--ids") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      std::vector<model_token> ids;
      std::stringstream ss(argv[i]);
      int i;
      while (ss >> i) {
        ids.push_back(i);
        if (ss.peek() == ',' || ss.peek() == ' ') ss.ignore();
      }
      params.ids = ids;
    } else if (arg == "-e") {
      escape_prompt = true;
    } else if (arg == "--prompt-cache") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.path_prompt_cache = argv[i];
    } else if (arg == "--prompt-cache-all") {
      params.prompt_cache_all = true;
    } else if (arg == "-f" || arg == "--file") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      std::ifstream file(argv[i]);
      if (!file) {
        fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
        invalid_param = true;
        break;
      }
      std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
      if (params.prompt.back() == '\n') {
        params.prompt.pop_back();
      }
    } else if (arg == "-n" || arg == "--n-predict") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.n_predict = std::stoi(argv[i]);
    } else if (arg == "--top-k") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.top_k = std::stoi(argv[i]);
    } else if (arg == "-c" || arg == "--ctx-size") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.n_ctx = std::stoi(argv[i]);
    } else if (arg == "--memory-f32") {
      params.memory_type = KV_MEM_TYPE_F32;
    } else if (arg == "--memory-f16") {
      params.memory_type = KV_MEM_TYPE_F16;
    } else if (arg == "--memory-auto") {
      params.memory_type = KV_MEM_TYPE_AUTO;
    } else if (arg == "--top-p") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.top_p = std::stof(argv[i]);
    } else if (arg == "--temp") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.temp = std::stof(argv[i]);
    } else if (arg == "--tfs") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.tfs_z = std::stof(argv[i]);
    } else if (arg == "--typical") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.typical_p = std::stof(argv[i]);
    } else if (arg == "--repeat-last-n") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.repeat_last_n = std::stoi(argv[i]);
    } else if (arg == "--repeat-penalty") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.repeat_penalty = std::stof(argv[i]);
    } else if (arg == "--frequency-penalty") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.frequency_penalty = std::stof(argv[i]);
    } else if (arg == "--presence-penalty") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.presence_penalty = std::stof(argv[i]);
    } else if (arg == "--mirostat") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.mirostat = std::stoi(argv[i]);
    } else if (arg == "--mirostat-lr") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.mirostat_eta = std::stof(argv[i]);
    } else if (arg == "--mirostat-ent") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.mirostat_tau = std::stof(argv[i]);
    } else if (arg == "-b" || arg == "--batch-size-truncate") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.n_batch = std::stoi(argv[i]);
    } else if (arg == "--keep") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.n_keep = std::stoi(argv[i]);
    } else if (arg == "--n_discard") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.n_discard = std::stoi(argv[i]);
    } else if (arg == "--shift-roped-k") {
      params.shift_roped_k = true;
    } else if (arg == "-m" || arg == "--model") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.model = argv[i];
    } else if (arg == "--lora") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.lora_adapter = argv[i];
      params.use_mmap = false;
    } else if (arg == "--lora-base") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.lora_base = argv[i];
    } else if (arg == "-i" || arg == "--interactive") {
      params.interactive = true;
    } else if (arg == "--embedding") {
      params.embedding = true;
    } else if (arg == "--interactive-first") {
      params.interactive_first = true;
    } else if (arg == "-ins" || arg == "--instruct") {
      params.instruct = true;
    } else if (arg == "--multiline-input") {
      params.multiline_input = true;
    } else if (arg == "--color") {
      params.use_color = true;
    } else if (arg == "--mlock") {
      params.use_mlock = true;
    } else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.n_gpu_layers = std::stoi(argv[i]);
    } else if (arg == "--use-mmap") {
      params.use_mmap = true;
    } else if (arg == "--mtest") {
      params.mem_test = true;
    } else if (arg == "--verbose-prompt") {
      params.verbose_prompt = true;
    } else if (arg == "-r" || arg == "--reverse-prompt") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.antiprompt.push_back(argv[i]);
    } else if (arg == "--perplexity") {
      params.perplexity = true;
    } else if (arg == "--ignore-eos") {
      // params.logit_bias[ctx->vocab.eos_token_id] = -INFINITY;
    } else if (arg == "--no-penalize-nl") {
      params.penalize_nl = false;
    } else if (arg == "-l" || arg == "--logit-bias") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      std::stringstream ss(argv[i]);
      model_token key;
      char sign;
      std::string value_str;
      try {
        if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
          params.logit_bias[key] = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
        } else {
          throw std::exception();
        }
      } catch (const std::exception& e) {
        invalid_param = true;
        break;
      }
    } else if (arg == "-h" || arg == "--help") {
      gpt_print_usage(argc, argv, default_params);
      exit(0);
    } else if (arg == "--random-prompt") {
      params.random_prompt = true;
    } else if (arg == "--in-prefix") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.input_prefix = argv[i];
    } else if (arg == "--in-suffix") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.input_suffix = argv[i];
    } else if (arg == "--batch-size") {  // ambiguous with batch-size-truncate (n_batch)
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.batch_size = std::stoi(argv[i]);
    } else if (arg == "--beam-search") {
      params.beam_search = true;
    } else if (arg == "--beam-size") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.beam_size = std::stoi(argv[i]);
    } else if (arg == "--model-name") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.model_name = argv[i];
      model_archs mt = model_name_to_arch::init().find(params.model_name);
      if (mt == MODEL_UNKNOWN) {
        invalid_param = true;
        break;
      } else {
        params.model_arch = mt;
      }
    } else {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      gpt_print_usage(argc, argv, default_params);
      exit(1);
    }
  }
  if (invalid_param) {
    fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
    gpt_print_usage(argc, argv, default_params);
    exit(1);
  }
  if (params.prompt_cache_all &&
      (params.interactive || params.interactive_first || params.instruct || params.antiprompt.size())) {
    fprintf(stderr, "error: --prompt-cache-all not supported in interactive mode yet\n");
    gpt_print_usage(argc, argv, default_params);
    exit(1);
  }
  if (escape_prompt) {
    process_escapes(params.prompt);
  }

  return true;
}

void gpt_print_usage(int /*argc*/, char** argv, const gpt_params& params) {
  fprintf(stderr, "usage: %s [options]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h, --help            show this help message and exit\n");
  fprintf(stderr, "  -i, --interactive     run in interactive mode\n");
  fprintf(stderr, "  --interactive-first   run in interactive mode and wait for input right away\n");
  fprintf(stderr, "  -ins, --instruct      run in instruction mode (use with Alpaca models)\n");
  fprintf(stderr, "  --multiline-input     allows you to write or paste multiple lines without ending each in '\\'\n");
  fprintf(stderr, "  -r PROMPT, --reverse-prompt PROMPT\n");
  fprintf(stderr, "                        run in interactive mode and poll user input upon seeing PROMPT (can be\n");
  fprintf(stderr, "                        specified more than once for multiple prompts).\n");
  fprintf(stderr, "  --color               colorise output to distinguish prompt and user input from generations\n");
  fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)\n");
  fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n",
          params.n_threads);
  fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
  fprintf(stderr, "                        prompt to start generation with (default: empty)\n");
  fprintf(stderr, "  -e                    process prompt escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\)\n");
  fprintf(stderr, "  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)\n");
  fprintf(stderr, "  --prompt-cache-all    if specified, saves user input and generations to cache as well.\n");
  fprintf(stderr, "                        not supported with --interactive or other interactive options\n");
  fprintf(stderr, "  --random-prompt       start with a randomized prompt.\n");
  fprintf(stderr, "  --in-prefix STRING    string to prefix user inputs with (default: empty)\n");
  fprintf(stderr, "  --in-suffix STRING    string to suffix after user inputs with (default: empty)\n");
  fprintf(stderr, "  --ids VECTOR<INT>     input ids after tokenizer (default: \"1,2,3,4\")\n");
  fprintf(stderr, "  -f FNAME, --file FNAME\n");
  fprintf(stderr, "                        prompt file to start generation.\n");
  fprintf(stderr, "  -n N, --n-predict N   number of tokens to predict (default: %d, -1 = infinity)\n",
          params.n_predict);
  fprintf(stderr, "  --top-k N             top-k sampling (default: %d, 0 = disabled)\n", params.top_k);
  fprintf(stderr, "  --top-p N             top-p sampling (default: %.1f, 1.0 = disabled)\n",
          static_cast<double>(params.top_p));
  fprintf(stderr, "  --tfs N               tail free sampling, parameter z (default: %.1f, 1.0 = disabled)\n",
          static_cast<double>(params.tfs_z));
  fprintf(stderr, "  --typical N           locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)\n",
          static_cast<double>(params.typical_p));
  fprintf(stderr,
          "  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)\n",
          params.repeat_last_n);
  fprintf(stderr, "  --repeat-penalty N    penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)\n",
          static_cast<double>(params.repeat_penalty));
  fprintf(stderr, "  --presence-penalty N  repeat alpha presence penalty (default: %.1f, 0.0 = disabled)\n",
          static_cast<double>(params.presence_penalty));
  fprintf(stderr, "  --frequency-penalty N repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)\n",
          static_cast<double>(params.frequency_penalty));
  fprintf(stderr, "  --mirostat N          use Mirostat sampling.\n");
  fprintf(stderr,
          "                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n");
  fprintf(stderr, "                        (default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)\n",
          params.mirostat);
  fprintf(stderr, "  --mirostat-lr N       Mirostat learning rate, parameter eta (default: %.1f)\n",
          static_cast<double>(params.mirostat_eta));
  fprintf(stderr, "  --mirostat-ent N      Mirostat target entropy, parameter tau (default: %.1f)\n",
          static_cast<double>(params.mirostat_tau));
  fprintf(stderr, "  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS\n");
  fprintf(stderr, "                        modifies the likelihood of token appearing in the completion,\n");
  fprintf(stderr, "                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n");
  fprintf(stderr, "                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'\n");
  fprintf(stderr, "  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
  fprintf(stderr,
          "  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)\n");
  fprintf(stderr, "  --no-penalize-nl      do not penalize newline token\n");
  fprintf(stderr, "  --memory-f32          use f32 for memory key+value\n");
  fprintf(stderr, "  --memory-f16          use f16 for memory key+value\n");
  fprintf(stderr, "  --memory-auto         use internal format for memory key+value\n");
  fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", static_cast<double>(params.temp));
  fprintf(stderr, "  -b N, --batch-size-truncate N  batch size for prompt processing (default: %d)\n", params.n_batch);
  fprintf(stderr, "  --perplexity          compute perplexity over the prompt\n");
  fprintf(stderr, "  --keep                number of tokens to keep from the initial prompt (default: %d, -1 = all)\n",
          params.n_keep);
  fprintf(stderr,
          "  --n_discard           number of tokens will be discarded (default: %d, -1 = half of tokens will be "
          "discarded)\n",
          params.n_discard);
  fprintf(stderr,
          "  --shift-roped-k       use ring-buffer and thus do not need re-computing after reaching context size "
          "(default: disabled)\n");
  if (model_mlock_supported()) {
    fprintf(stderr, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
  }
  if (model_mmap_supported()) {
    fprintf(stderr, "  --use-mmap             use memory-map model (faster load but may have NUMA issue)\n");
  }
  fprintf(stderr, "  -ngl N, --n-gpu-layers N\n");
  fprintf(stderr, "                        number of layers to store in VRAM\n");
  fprintf(stderr, "  --mtest               compute maximum memory usage\n");
  fprintf(stderr, "  --verbose-prompt      print prompt before generation\n");
  fprintf(stderr, "  --lora FNAME          apply LoRA adapter (implies --no-mmap)\n");
  fprintf(stderr,
          "  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter\n");
  fprintf(stderr, "  -m FNAME, --model FNAME\n");
  fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
  fprintf(stderr, "  --batch-size 2        number batch of prompt\n");
  fprintf(stderr, "  --beam-search         use beam search for text generation\n");
  fprintf(stderr, "  --beam-size 4         number of beams for beam_search, only valid after --beam-search\n");
  fprintf(stderr, "  --model-name          input model name, options are: ");
  model_name_to_arch::init().valid_options();
  fprintf(stderr, "\n");
}
