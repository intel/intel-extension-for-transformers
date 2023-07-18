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
#include <cinttypes>
#include <fstream>
#include <random>
#include <unordered_map>
#include <cassert>
#include <cstring>
#include <memory>
#include <algorithm>
#include <exception>
#include <iterator>
#include <string>
#include <vector>

#include "core/ne_layers.h"
#include "llama_model.h"
#include "llama_config.h"
#include "data_types.h"
#include "ne.h"
#include "util.h"

void process_escapes(std::string& input) {
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

bool gpt_params_parse(int argc, char** argv, gpt_params& params) {
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
#if defined(GGML_USE_CUBLAS)
      fprintf(stderr, "WARNING: when using cuBLAS generation results are NOT guaranteed to be reproducible.\n");
#endif
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
      params.memory_f16 = false;
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
    } else if (arg == "-b" || arg == "--batch-size") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.n_batch = std::stoi(argv[i]);
      params.n_batch = std::min(512, params.n_batch);
    } else if (arg == "--keep") {
      if (++i >= argc) {
        invalid_param = true;
        break;
      }
      params.n_keep = std::stoi(argv[i]);
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
    } else if (arg == "--no-mmap") {
      params.use_mmap = false;
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
      params.logit_bias[model_token_eos()] = -INFINITY;
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
  fprintf(stderr, "  -f FNAME, --file FNAME\n");
  fprintf(stderr, "                        prompt file to start generation.\n");
  fprintf(stderr, "  -n N, --n-predict N   number of tokens to predict (default: %d, -1 = infinity)\n",
          params.n_predict);
  fprintf(stderr, "  --top-k N             top-k sampling (default: %d, 0 = disabled)\n", params.top_k);
  fprintf(stderr, "  --top-p N             top-p sampling (default: %.1f, 1.0 = disabled)\n", (double)params.top_p);
  fprintf(stderr, "  --tfs N               tail free sampling, parameter z (default: %.1f, 1.0 = disabled)\n",
          (double)params.tfs_z);
  fprintf(stderr, "  --typical N           locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)\n",
          (double)params.typical_p);
  fprintf(stderr,
          "  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)\n",
          params.repeat_last_n);
  fprintf(stderr, "  --repeat-penalty N    penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)\n",
          (double)params.repeat_penalty);
  fprintf(stderr, "  --presence-penalty N  repeat alpha presence penalty (default: %.1f, 0.0 = disabled)\n",
          (double)params.presence_penalty);
  fprintf(stderr, "  --frequency-penalty N repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)\n",
          (double)params.frequency_penalty);
  fprintf(stderr, "  --mirostat N          use Mirostat sampling.\n");
  fprintf(stderr,
          "                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n");
  fprintf(stderr, "                        (default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)\n",
          params.mirostat);
  fprintf(stderr, "  --mirostat-lr N       Mirostat learning rate, parameter eta (default: %.1f)\n",
          (double)params.mirostat_eta);
  fprintf(stderr, "  --mirostat-ent N      Mirostat target entropy, parameter tau (default: %.1f)\n",
          (double)params.mirostat_tau);
  fprintf(stderr, "  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS\n");
  fprintf(stderr, "                        modifies the likelihood of token appearing in the completion,\n");
  fprintf(stderr, "                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n");
  fprintf(stderr, "                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'\n");
  fprintf(stderr, "  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
  fprintf(stderr,
          "  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)\n");
  fprintf(stderr, "  --no-penalize-nl      do not penalize newline token\n");
  fprintf(stderr, "  --memory-f32          use f32 instead of f16 for memory key+value\n");
  fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", (double)params.temp);
  fprintf(stderr, "  -b N, --batch-size N  batch size for prompt processing (default: %d)\n", params.n_batch);
  fprintf(stderr, "  --perplexity          compute perplexity over the prompt\n");
  fprintf(stderr, "  --keep                number of tokens to keep from the initial prompt (default: %d, -1 = all)\n",
          params.n_keep);
  if (model_mlock_supported()) {
    fprintf(stderr, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
  }
  if (model_mmap_supported()) {
    fprintf(
        stderr,
        "  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
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
  fprintf(stderr, "\n");
}

std::string gpt_random_prompt(std::mt19937& rng) {
  const int r = rng() % 10;
  switch (r) {
    case 0:
      return "So";
    case 1:
      return "Once upon a time";
    case 2:
      return "When";
    case 3:
      return "The";
    case 4:
      return "After";
    case 5:
      return "If";
    case 6:
      return "import";
    case 7:
      return "He";
    case 8:
      return "She";
    case 9:
      return "They";
    default:
      return "To";
  }

  return "The";
}

// evaluate the transformer
//
//   - lctx:      model context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool llama_model_eval_internal(model_context& lctx, const model_token* tokens, const int n_tokens,
                                      const int n_past, const int n_threads) {
  // enforce that the first token is BOS
  if (n_past == 0 && tokens[0] != model_token_bos()) {
    fprintf(stderr, "%s: first token must be BOS\n", __func__);
    return false;
  }

  const int64_t t_start_us = ne_time_us();

  const int N = n_tokens;

  const auto& model = lctx.model;
  const auto& hparams = model.hparams;

  const auto& kv_self = model.kv_self;

  MODEL_ASSERT(!!kv_self.ctx);

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = hparams.n_ctx;
  const int n_head = hparams.n_head;
  const int n_vocab = hparams.n_vocab;
  const int n_rot = hparams.n_embd / hparams.n_head;

  auto& mem_per_token = lctx.mem_per_token;
  auto& buf_compute = lctx.buf_compute;

  struct ne_init_params params = {
      /*.mem_size   =*/buf_compute.size,
      /*.mem_buffer =*/buf_compute.addr,
      /*.no_alloc   =*/false,
  };

  struct ne_context* ctx0 = ne_init(params);

  // for big prompts, if BLAS is enabled, it is better to use only one thread
  // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
  ne_cgraph gf = {};
  gf.n_threads = N >= 32 && ne_cpu_has_blas() ? 1 : n_threads;

  struct ne_tensor* embd = ne_new_tensor_1d(ctx0, NE_TYPE_I32, N, NE_SIZE_CALC);
  ne_set_name(embd, "embd");
  memcpy(embd->data, tokens, N * ne_element_size(embd));

  struct ne_tensor* inpL = ne_get_rows(ctx0, model.tok_embeddings, embd);

  for (int il = 0; il < n_layer; ++il) {
    struct ne_tensor* inpSA = inpL;

    struct ne_tensor* cur;

    lctx.use_buf(ctx0, 0);

    // norm
    {
      cur = ne_rms_norm(ctx0, inpL);

      // cur = cur*attention_norm(broadcasted)
      cur = ne_mul(ctx0, cur, model.layers[il].attention_norm);
    }
    ne_tensor *Qcur, *Kcur, *Vcur;
    if (model.layers[il].wq->type == NE_TYPE_Q4_JBLAS) {  // fused execution of QKV
      struct ne_tensor* QKVcur = ne_mul_qkv(ctx0, model.layers[il].wq, model.layers[il].wk, model.layers[il].wv, cur);
      Qcur = ne_rope_inplace(
          ctx0,
          ne_reshape_3d(ctx0, ne_view_1d(ctx0, QKVcur, N * n_embd, 0 * N * n_embd * ne_element_size(QKVcur)),
                        n_embd / n_head, n_head, N),
          n_past, n_rot, 0);
      Kcur = ne_rope_inplace(
          ctx0,
          ne_reshape_3d(ctx0, ne_view_1d(ctx0, QKVcur, N * n_embd, 1 * N * n_embd * ne_element_size(QKVcur)),
                        n_embd / n_head, n_head, N),
          n_past, n_rot, 0);
      Vcur = ne_transpose(
          ctx0, ne_reshape_2d(ctx0, ne_view_1d(ctx0, QKVcur, N * n_embd, 2 * N * n_embd * ne_element_size(QKVcur)),
                              n_embd, N));

    } else {
      Qcur = ne_rope_inplace(
          ctx0, ne_reshape_3d(ctx0, ne_mul_mat(ctx0, model.layers[il].wq, cur), n_embd / n_head, n_head, N), n_past,
          n_rot, 0);
      Kcur = ne_rope_inplace(
          ctx0, ne_reshape_3d(ctx0, ne_mul_mat(ctx0, model.layers[il].wk, cur), n_embd / n_head, n_head, N), n_past,
          n_rot, 0);
      Vcur = ne_transpose(ctx0, ne_reshape_2d(ctx0, ne_mul_mat(ctx0, model.layers[il].wv, cur), n_embd, N));
    }
    ne_set_name(Qcur, "Qcur");
    ne_set_name(Kcur, "Kcur");
    ne_set_name(Vcur, "Vcur");
    // self-attention
    {
      // store key and value to memory
      {
        struct ne_tensor* k =
            ne_view_1d(ctx0, kv_self.k, N * n_embd, (ne_element_size(kv_self.k) * n_embd) * (il * n_ctx + n_past));
        struct ne_tensor* v =
            ne_view_2d(ctx0, kv_self.v, N, n_embd, (n_ctx)*ne_element_size(kv_self.v),
                       (il * n_ctx) * ne_element_size(kv_self.v) * n_embd + n_past * ne_element_size(kv_self.v));

        // important: storing RoPE-ed version of K in the KV cache!
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur, k));
        ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur, v));
      }

      struct ne_tensor* Q = ne_permute(ctx0, Qcur, 0, 2, 1, 3);
      ne_set_name(Q, "Q");

      struct ne_tensor* K = ne_permute(ctx0,
                                       ne_reshape_3d(ctx0,
                                                     ne_view_1d(ctx0, kv_self.k, (n_past + N) * n_embd,
                                                                il * n_ctx * ne_element_size(kv_self.k) * n_embd),
                                                     n_embd / n_head, n_head, n_past + N),
                                       0, 2, 1, 3);
      ne_set_name(K, "K");

      // K * Q
      struct ne_tensor* KQ = ne_mul_mat(ctx0, K, Q);
      ne_set_name(KQ, "KQ");

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      struct ne_tensor* KQ_scale = ne_new_f32(ctx0, 1.0f / sqrtf(float(n_embd) / n_head));
      ne_set_name(KQ_scale, "1/sqrt(n_embd/n_head)");

      // KQ_scaled shape [n_past + N, N, n_head, 1]
      struct ne_tensor* KQ_scaled = ne_scale_inplace(ctx0, KQ, KQ_scale);
      ne_set_name(KQ_scaled, "KQ_scaled");

      // KQ_masked = mask_past(KQ_scaled)
      struct ne_tensor* KQ_masked = ne_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
      ne_set_name(KQ_masked, "KQ_masked");

      // KQ = soft_max(KQ_masked)
      struct ne_tensor* KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_masked);
      ne_set_name(KQ_soft_max, "KQ_soft_max");

      // split cached V into n_head heads
      struct ne_tensor* V = ne_view_3d(
          ctx0, kv_self.v, n_past + N, n_embd / n_head, n_head, n_ctx * ne_element_size(kv_self.v),
          n_ctx * ne_element_size(kv_self.v) * n_embd / n_head, il * n_ctx * ne_element_size(kv_self.v) * n_embd);
      ne_set_name(V, "V");

#if 1
      struct ne_tensor* KQV = ne_mul_mat(ctx0, V, KQ_soft_max);
      ne_set_name(KQV, "KQV");
#else
      // make V contiguous in memory to speed up the matmul, however we waste time on the copy
      // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
      // is there a better way?
      struct ne_tensor* V_cont =
          ne_cpy(ctx0, V, ne_new_tensor_3d(ctx0, kv_self.v->type, n_past + N, n_embd / n_head, n_head));
      struct ne_tensor* KQV = ne_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct ne_tensor* KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);
      ne_set_name(KQV_merged, "KQV_merged");

      // cur = KQV_merged.contiguous().view(n_embd, N)
      cur = ne_cpy(ctx0, KQV_merged, ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N, NE_SIZE_CALC));
      ne_set_name(cur, "KQV_merged_contiguous");

      // projection (no bias)
      cur = ne_mul_mat(ctx0, model.layers[il].wo, cur);
    }

    lctx.use_buf(ctx0, 1);

    struct ne_tensor* inpFF = ne_add(ctx0, cur, inpSA);

    // feed-forward network
    {
      // norm
      {
        cur = ne_rms_norm(ctx0, inpFF);

        // cur = cur*ffn_norm(broadcasted)
        cur = ne_mul(ctx0, cur, model.layers[il].ffn_norm);
      }

      if (model.layers[il].w1->type == NE_TYPE_Q4_JBLAS) {
        cur = ne_ffn_silu(ctx0, model.layers[il].w1, model.layers[il].w2, model.layers[il].w3, cur);
      } else {
        struct ne_tensor* tmp = ne_mul_mat(ctx0, model.layers[il].w3, cur);

        cur = ne_mul_mat(ctx0, model.layers[il].w1, cur);

        // SILU activation
        cur = ne_silu(ctx0, cur);

        cur = ne_mul(ctx0, cur, tmp);

        cur = ne_mul_mat(ctx0, model.layers[il].w2, cur);
      }
    }

    cur = ne_add(ctx0, cur, inpFF);

    // input for next layer
    inpL = cur;
  }

  lctx.use_buf(ctx0, 0);

  // used at the end to optionally extract the embeddings
  struct ne_tensor* embeddings = NULL;

  // norm
  {
    inpL = ne_rms_norm(ctx0, inpL);

    // inpL = inpL*norm(broadcasted)
    inpL = ne_mul(ctx0, inpL, model.norm);

    embeddings = inpL;
  }

  // lm_head
  inpL = ne_mul_mat(ctx0, model.output, inpL);

  lctx.use_buf(ctx0, -1);

  // logits -> probs
  // inpL = ne_soft_max_inplace(ctx0, inpL);

  // run the computation
  ne_build_forward_expand(&gf, inpL);
  ne_graph_compute(ctx0, &gf);

#ifdef NE_PERF
  bool engine_profiling_ = (getenv("ENGINE_PROFILING") != NULL);
  if (engine_profiling_) {
    ne_graph_profiling(&gf);
  }
#endif
  // plot the computation graph in dot format (for debugging purposes)
  // if (n_past%100 == 0) {
  //    ne_graph_dump_dot(&gf, NULL, "model.dot");
  //}

  // embd_w.resize(n_vocab*N);
  // memcpy(embd_w.data(), ne_get_data(inpL), sizeof(float)*n_vocab*N);

  // update kv token count
  lctx.model.kv_self.n = n_past + N;

  // extract logits
  {
    auto& logits_out = lctx.logits;

    if (lctx.logits_all) {
      logits_out.resize(n_vocab * N);
      memcpy(logits_out.data(), (float*)ne_get_data(inpL), sizeof(float) * n_vocab * N);
    } else {
      // return result for just the last token
      logits_out.resize(n_vocab);
      memcpy(logits_out.data(), (float*)ne_get_data(inpL) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);
    }
  }

  // extract embeddings
  if (!lctx.embedding.empty()) {
    auto& embedding_out = lctx.embedding;

    embedding_out.resize(n_embd);
    memcpy(embedding_out.data(), (float*)ne_get_data(embeddings) + (n_embd * (N - 1)), sizeof(float) * n_embd);
  }

  if (mem_per_token == 0) {
    mem_per_token = ne_used_mem(ctx0) / N;
  }

#if 0
    printf("\n%s: used_mem = %.3f MB, scratch -- %.3f MB %.3f MB\n", __func__,
            ne_used_mem(ctx0)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0);
#endif

  ne_free(ctx0);

  // measure the performance only for the single-token evals
  int64_t time_interval = ne_time_us() - t_start_us;
  if (N == 1) {
    lctx.t_eval_us += time_interval;
    lctx.n_eval++;
  } else if (N > 1) {
    lctx.t_p_eval_us += time_interval;
    lctx.n_p_eval += N;
  }
  lctx.eval_times.push_back(time_interval);

  return true;
}

int model_eval(struct model_context* ctx, const model_token* tokens, int n_tokens, int n_past, int n_threads) {
  if (!llama_model_eval_internal(*ctx, tokens, n_tokens, n_past, n_threads)) {
    fprintf(stderr, "%s: failed to eval\n", __func__);
    return 1;
  }

  // get a more accurate load time, upon first eval
  // TODO: fix this
  if (!ctx->has_evaluated_once) {
    ctx->t_load_us = ne_time_us() - ctx->t_start_us;
    ctx->has_evaluated_once = true;
  }

  return 0;
}

// TODO: not great allocating this every time
std::vector<model_token> model_tokenize(struct model_context* ctx, const std::string& text, bool add_bos) {
  // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
  std::vector<model_token> res(text.size() + (int)add_bos);
  const int n = model_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
  assert(n >= 0);
  res.resize(n);

  return res;
}

struct model_context* model_init_from_gpt_params(const gpt_params& params) {
  auto lparams = model_context_default_params();

  lparams.n_ctx = params.n_ctx;
  lparams.n_gpu_layers = params.n_gpu_layers;
  lparams.seed = params.seed;
  lparams.f16_kv = params.memory_f16;
  lparams.use_mmap = params.use_mmap;
  lparams.use_mlock = params.use_mlock;
  lparams.logits_all = params.perplexity;
  lparams.embedding = params.embedding;

  model_context* lctx = model_init_from_file(params.model.c_str(), lparams);

  if (lctx == NULL) {
    fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
    return NULL;
  }

  if (!params.lora_adapter.empty()) {
    int err = model_apply_lora_from_file(lctx, params.lora_adapter.c_str(),
                                         params.lora_base.empty() ? NULL : params.lora_base.c_str(), params.n_threads);
    if (err != 0) {
      fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
      return NULL;
    }
  }

  return lctx;
}
