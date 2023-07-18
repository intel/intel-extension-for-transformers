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
#include "common.h"
#ifndef _WIN32
#include <ext/alloc_traits.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <unordered_set>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>

// third-party utilities
// use your favorite implementations
#define DR_WAV_IMPLEMENTATION
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <locale>
#include <codecvt>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int32_t get_num_physical_cores() {
#ifdef __linux__
  // enumerate the set of thread siblings, num entries is num cores
  std::unordered_set<std::string> siblings;
  for (uint32_t cpu = 0; cpu < UINT32_MAX; ++cpu) {
    std::ifstream thread_siblings("/sys/devices/system/cpu" + std::to_string(cpu) + "/topology/thread_siblings");
    if (!thread_siblings.is_open()) {
      break;  // no more cpus
    }
    std::string line;
    if (std::getline(thread_siblings, line)) {
      siblings.insert(line);
    }
  }
  if (siblings.size() > 0) {
    return static_cast<int32_t>(siblings.size());
  }
#elif defined(__APPLE__) && defined(__MACH__)
  int32_t num_physical_cores;
  size_t len = sizeof(num_physical_cores);
  int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
  if (result == 0) {
    return num_physical_cores;
  }
  result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
  if (result == 0) {
    return num_physical_cores;
  }
#elif defined(_WIN32)
  // TODO: Implement
#endif
  unsigned int n_threads = std::thread::hardware_concurrency();
  return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

bool isValidFilename(const std::string& filename) {
  std::ifstream infile(filename.c_str());
  return infile.good();
}

void gpt_print_usage(int /*argc*/, char** argv, const common_params& params) {
  fprintf(stderr, "usage: %s [options]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h, --help            show this help message and exit\n");
  fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
  fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n",
          params.n_threads);
  fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
  fprintf(stderr, "                        prompt to start generation with (default: random)\n");
  fprintf(stderr, "  -f FNAME, --file FNAME\n");
  fprintf(stderr, "                        load prompt from a file\n");
  fprintf(stderr, "  -tt TOKEN_TEST, --token_test TOKEN_TEST\n");
  fprintf(stderr, "                        test tokenization\n");
  fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d)\n", params.n_predict);
  fprintf(stderr, "  --top_k N             top-k sampling (default: %d, 0 = n_vocab)\n", params.top_k);
  fprintf(stderr, "  --top_p N             top-p sampling (default: %.2f)\n", params.top_p);
  fprintf(stderr, "  --temp N              temperature (default: %.2f)\n", params.temp);
  fprintf(stderr,
          "  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)\n",
          params.repeat_last_n);
  fprintf(stderr, "  --repeat-penalty N    penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)\n",
          (double)params.repeat_penalty);
  fprintf(stderr, "  --perplexity          compute perplexity over the prompt\n");
  fprintf(stderr, "  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
  fprintf(stderr, "  -b N, --batch_size N  batch size for prompt processing (default: %d)\n", params.n_batch);
  fprintf(stderr, "  -m FNAME, --model FNAME\n");
  fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
  fprintf(stderr, "\n");
}

bool common_params_parse(int argc, char** argv, common_params& params) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-s" || arg == "--seed") {
      params.seed = std::stoi(argv[++i]);
    } else if (arg == "-t" || arg == "--threads") {
      params.n_threads = std::stoi(argv[++i]);
    } else if (arg == "-p" || arg == "--prompt") {
      params.prompt = argv[++i];
    } else if (arg == "-n" || arg == "--n_predict") {
      params.n_predict = std::stoi(argv[++i]);
    } else if (arg == "--top_k") {
      params.top_k = std::max(1, std::stoi(argv[++i]));
    } else if (arg == "--top_p") {
      params.top_p = std::stof(argv[++i]);
    } else if (arg == "--temp") {
      params.temp = std::stof(argv[++i]);
    } else if (arg == "--repeat-last-n") {
      params.repeat_last_n = std::stof(argv[++i]);
    } else if (arg == "--repeat-penalty") {
      params.repeat_penalty = std::stof(argv[++i]);
    } else if (arg == "--perplexity") {
      params.perplexity = true;
    } else if (arg == "-c" || arg == "--ctx-size") {
      params.n_ctx = std::stoi(argv[++i]);
    } else if (arg == "-b" || arg == "--batch_size") {
      params.n_batch = std::stoi(argv[++i]);
    } else if (arg == "-m" || arg == "--model") {
      if (!isValidFilename(argv[i + 1])) return false;
      params.model = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      gpt_print_usage(argc, argv, params);
      exit(0);
    } else if (arg == "-f" || arg == "--file") {
      if (++i > argc) {
        fprintf(stderr, "Invalid file param");
        break;
      }
      std::ifstream file(argv[i]);
      if (!file) {
        fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
        break;
      }
      params.prompt.clear();
      std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
      if (params.prompt.back() == '\n') {
        params.prompt.pop_back();
      }
    } else if (arg == "-tt" || arg == "--token_test") {
      params.token_test = argv[++i];
    } else {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      gpt_print_usage(argc, argv, params);
      exit(0);
    }
  }

  return true;
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

std::string trim(const std::string& s) {
  std::regex e("^\\s+|\\s+$");
  return std::regex_replace(s, e, "");
}

std::string replace(const std::string& s, const std::string& from, const std::string& to) {
  std::string result = s;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.length(), to);
    pos += to.length();
  }
  return result;
}

void gpt_vocab::add_special_token(const std::string& token) { special_tokens.push_back(token); }

std::map<std::string, int32_t> json_parse(const std::string& fname) {
  std::map<std::string, int32_t> result;

  // read file into string
  std::string json;
  {
    std::ifstream ifs(fname);
    if (!ifs) {
      fprintf(stderr, "Failed to open %s\n", fname.c_str());
      exit(1);
    }

    json = std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
  }

  if (json[0] != '{') {
    return result;
  }

  // parse json
  {
    bool has_key = false;
    bool in_token = false;

    std::string str_key = "";
    std::string str_val = "";

    int n = json.size();
    for (int i = 1; i < n; ++i) {
      if (!in_token) {
        if (json[i] == ' ') continue;
        if (json[i] == '"') {
          in_token = true;
          continue;
        }
      } else {
        if (json[i] == '\\' && i + 1 < n) {
          if (has_key == false) {
            str_key += json[i];
          } else {
            str_val += json[i];
          }
          ++i;
        } else if (json[i] == '"') {
          if (has_key == false) {
            has_key = true;
            ++i;
            while (json[i] == ' ') ++i;
            ++i;  // :
            while (json[i] == ' ') ++i;
            if (json[i] != '\"') {
              while (json[i] != ',' && json[i] != '}') {
                str_val += json[i++];
              }
              has_key = false;
            } else {
              in_token = true;
              continue;
            }
          } else {
            has_key = false;
          }

          str_key = ::replace(str_key, "\\u0120", " ");   // \u0120 -> space
          str_key = ::replace(str_key, "\\u010a", "\n");  // \u010a -> new line
          str_key = ::replace(str_key, "\\\"", "\"");     // \\\"   -> "

          try {
            result[str_key] = std::stoi(str_val);
          } catch (...) {
            // fprintf(stderr, "%s: ignoring key '%s' with value '%s'\n", fname.c_str(), str_key.c_str(),
            // str_val.c_str());
          }
          str_key = "";
          str_val = "";
          in_token = false;
          continue;
        }
        if (has_key == false) {
          str_key += json[i];
        } else {
          str_val += json[i];
        }
      }
    }
  }

  return result;
}

std::string convert_to_utf8(const std::wstring& input) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(input);
}

std::wstring convert_to_wstring(const std::string& input) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.from_bytes(input);
}

std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab& vocab, const std::string& text) {
  std::vector<std::string> words;

  // first split the text into words
  {
    std::string str = text;
    std::string pat =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

    // Generate the subpattern from the special_tokens vector if it's not empty
    if (!vocab.special_tokens.empty()) {
      std::string special_tokens_subpattern;
      for (const auto& token : vocab.special_tokens) {
        if (!special_tokens_subpattern.empty()) {
          special_tokens_subpattern += "|";
        }
        special_tokens_subpattern += token;
      }

      // Modify the regex pattern with the generated special tokens subpattern
      pat = special_tokens_subpattern + "|" + pat;
    }

    std::regex re(pat);
    std::smatch m;

    while (std::regex_search(str, m, re)) {
      for (auto x : m) {
        words.push_back(x);
      }
      str = m.suffix();
    }
  }

  // find the longest token that forms each word in words:
  std::vector<gpt_vocab::id> tokens;
  for (const auto& word : words) {
    for (int i = 0; i < word.size();) {
      for (int j = word.size() - 1; j >= i; j--) {
        auto cand = word.substr(i, j - i + 1);
        auto it = vocab.token_to_id.find(cand);
        if (it != vocab.token_to_id.end()) {  // word.substr(i, j-i+1) in vocab
          tokens.push_back(it->second);
          i = j + 1;
          break;
        } else if (j == i) {  // word.substr(i, 1) has no matching
          fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
          i++;
        }
      }
    }
  }

  return tokens;
}

std::vector<gpt_vocab::id> parse_tokens_from_string(const std::string& input, char delimiter) {
  std::vector<gpt_vocab::id> output;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    output.push_back(std::stoi(token));
  }

  return output;
}

std::map<std::string, std::vector<gpt_vocab::id>> extract_tests_from_file(const std::string& fpath_test) {
  if (fpath_test.empty()) {
    fprintf(stderr, "%s : No test file found.\n", __func__);
    return std::map<std::string, std::vector<gpt_vocab::id>>();
  }

  std::map<std::string, std::vector<gpt_vocab::id>> tests;

  auto fin = std::ifstream(fpath_test, std::ios_base::in);
  const char* delimeter = " => ";
  const char del_tok = ',';
  std::string line;
  while (std::getline(fin, line)) {
    size_t delimiterPos = line.find(delimeter);
    if (delimiterPos != std::string::npos) {
      std::string text = line.substr(0, delimiterPos);
      std::string s_tokens = line.substr(delimiterPos + std::strlen(delimeter));
      tests[text] = parse_tokens_from_string(s_tokens, del_tok);
    }
  }
  return tests;
}

void test_gpt_tokenizer(gpt_vocab& vocab, const std::string& fpath_test) {
  std::map<std::string, std::vector<gpt_vocab::id>> tests = extract_tests_from_file(fpath_test);

  size_t n_fails = 0;

  for (const auto& test : tests) {
    std::vector<gpt_vocab::id> tokens = gpt_tokenize(vocab, test.first);

    if (tokens != test.second) {
      n_fails++;

      // print out failure cases
      fprintf(stderr, "%s : failed test: '%s'\n", __func__, test.first.c_str());
      fprintf(stderr, "%s : tokens in hf:   ", __func__);
      for (const auto& t : test.second) {
        fprintf(stderr, "%s(%d), ", vocab.id_to_token[t].c_str(), t);
      }
      fprintf(stderr, "\n");
      fprintf(stderr, "%s : tokens in ggml: ", __func__);
      for (const auto& t : tokens) {
        fprintf(stderr, "%s(%d), ", vocab.id_to_token[t].c_str(), t);
      }
      fprintf(stderr, "\n");
    }
  }

  fprintf(stderr, "%s : %lu tests failed out of %lu tests.\n", __func__, n_fails, tests.size());
}

bool gpt_vocab_init(const std::string& fname, gpt_vocab& vocab) {
  printf("%s: loading vocab from '%s'\n", __func__, fname.c_str());

  vocab.token_to_id = ::json_parse(fname);

  for (const auto& kv : vocab.token_to_id) {
    vocab.id_to_token[kv.second] = kv.first;
  }

  printf("%s: vocab size = %d\n", __func__, (int)vocab.token_to_id.size());

  // print the vocabulary
  // for (auto kv : vocab.token_to_id) {
  //    printf("'%s' -> %d\n", kv.first.data(), kv.second);
  //}

  return true;
}

gpt_vocab::id gpt_sample_top_k_top_p(const gpt_vocab& vocab, const float* logits, int top_k, double top_p, double temp,
                                     std::mt19937& rng) {
  int n_logits = vocab.id_to_token.size();

  std::vector<std::pair<double, gpt_vocab::id>> logits_id;
  logits_id.reserve(n_logits);

  {
    const double scale = 1.0 / temp;
    for (int i = 0; i < n_logits; ++i) {
      logits_id.push_back(std::make_pair(logits[i] * scale, i));
    }
  }

  // find the top K tokens
  std::partial_sort(logits_id.begin(), logits_id.begin() + top_k, logits_id.end(),
                    [](const std::pair<double, gpt_vocab::id>& a, const std::pair<double, gpt_vocab::id>& b) {
                      return a.first > b.first;
                    });

  logits_id.resize(top_k);

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

  if (top_p < 1.0f) {
    double cumsum = 0.0f;
    for (int i = 0; i < top_k; i++) {
      cumsum += probs[i];
      if (cumsum >= top_p) {
        top_k = i + 1;
        probs.resize(top_k);
        logits_id.resize(top_k);
        break;
      }
    }

    cumsum = 1.0 / cumsum;
    for (int i = 0; i < (int)probs.size(); i++) {
      probs[i] *= cumsum;
    }
  }

  // printf("\n");
  // for (int i = 0; i < (int) probs.size(); i++) {
  //     printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
  // }
  // exit(0);

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  int idx = dist(rng);

  return logits_id[idx].second;
}

gpt_vocab::id gpt_sample_top_k_top_p_repeat(const gpt_vocab& vocab, const float* logits,
                                            const int32_t* last_n_tokens_data, size_t last_n_tokens_data_size,
                                            int top_k, double top_p, double temp, int repeat_last_n,
                                            float repeat_penalty, std::mt19937& rng) {
  int n_logits = vocab.id_to_token.size();

  const auto* plogits = logits;

  const auto last_n_tokens = std::vector<int32_t>(last_n_tokens_data, last_n_tokens_data + last_n_tokens_data_size);

  if (temp <= 0) {
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
    const float scale = 1.0f / temp;
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
  std::partial_sort(logits_id.begin(), logits_id.begin() + top_k, logits_id.end(),
                    [](const std::pair<double, gpt_vocab::id>& a, const std::pair<double, gpt_vocab::id>& b) {
                      return a.first > b.first;
                    });

  logits_id.resize(top_k);

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

  if (top_p < 1.0f) {
    double cumsum = 0.0f;
    for (int i = 0; i < top_k; i++) {
      cumsum += probs[i];
      if (cumsum >= top_p) {
        top_k = i + 1;
        probs.resize(top_k);
        logits_id.resize(top_k);
        break;
      }
    }

    cumsum = 1.0 / cumsum;
    for (int i = 0; i < (int)probs.size(); i++) {
      probs[i] *= cumsum;
    }
  }

  //    printf("\n");
  //    for (int i = 0; i < (int) probs.size(); i++) {
  //    for (int i = 0; i < 10; i++) {
  //        printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
  //    }

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  int idx = dist(rng);

  return logits_id[idx].second;
}

void quant_print_usage(int argc, char** argv, const quant_params& params) {
  fprintf(stderr, "usage: %s [options]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h, --help            show this help message and exit\n");
  fprintf(stderr, "  --model_file          path to fp32 model\n");
  fprintf(stderr, "  --out_file            path to save quantized model\n");
  fprintf(stderr, "  --nthread N           thread number (default: 1)\n");
  fprintf(stderr, "  --bits N              quant bits (default: 4)\n");
  fprintf(stderr, "  --alg                 quantize algorithm: sym/asym   (default: sym)\n");
  fprintf(stderr, "  --block_size N        block size (default: 32)\n");
  fprintf(stderr, "  --scale_dtype dtype   fp32/bf16 type for scales (default: fp32)\n");
  fprintf(stderr, "  --gemm_isa            vnni/ams/none (default: none)\n");
  fprintf(stderr, "\n");
}

bool quant_params_parse(int argc, char** argv, quant_params& params) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--model_file") {
      params.model_file = argv[++i];
    } else if (arg == "--out_file") {
      params.out_file = argv[++i];
    } else if (arg == "--bits") {
      params.bits = std::stoi(argv[++i]);
    } else if (arg == "--alg") {
      params.alg = argv[++i];
    } else if (arg == "--block_size") {
      params.block_size = std::stoi(argv[++i]);
    } else if (arg == "--scale_dtype") {
      params.scale_dtype = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      quant_print_usage(argc, argv, params);
      exit(0);
    } else {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      quant_print_usage(argc, argv, params);
      exit(0);
    }
  }
  return true;
}

bool ne_common_quantize_0(std::ifstream& finp, std::ofstream& fout, const ne_ftype ftype,
                          const std::vector<std::string>& to_quant, const std::vector<std::string>& to_skip) {
  ne_type qtype = NE_TYPE_F32;

  switch (ftype) {
    case NE_FTYPE_MOSTLY_Q4_0:
      qtype = NE_TYPE_Q4_0;
      break;
    case NE_FTYPE_MOSTLY_Q4_1:
      qtype = NE_TYPE_Q4_1;
      break;
    case NE_FTYPE_MOSTLY_Q5_0:
      qtype = NE_TYPE_Q5_0;
      break;
    case NE_FTYPE_MOSTLY_Q5_1:
      qtype = NE_TYPE_Q5_1;
      break;
    case NE_FTYPE_MOSTLY_Q8_0:
      qtype = NE_TYPE_Q8_0;
      break;
    case NE_FTYPE_MOSTLY_Q4_JBLAS_B32:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_B128:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_B1024:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_BF16_B32:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B32:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_BF16_B32:
    case NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B128:
      qtype = NE_TYPE_Q4_JBLAS;
      break;
    case NE_FTYPE_UNKNOWN:
    case NE_FTYPE_ALL_F32:
    case NE_FTYPE_MOSTLY_F16:
    case NE_FTYPE_MOSTLY_Q4_1_SOME_F16: {
      fprintf(stderr, "%s: invalid model type %d\n", __func__, ftype);
      return false;
    }
  };

  if (!ne_is_quantized(qtype)) {
    fprintf(stderr, "%s: invalid quantization type %d (%s)\n", __func__, qtype, ne_type_name(qtype));
    return false;
  }

  size_t total_size_org = 0;
  size_t total_size_new = 0;

  std::vector<float> work;

  std::vector<uint8_t> data_u8;
  std::vector<ne_fp16_t> data_f16;
  std::vector<float> data_f32;

  std::vector<int64_t> hist_all(1 << 4, 0);

  while (true) {
    int32_t n_dims;
    int32_t length;
    int32_t ttype;

    finp.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    finp.read(reinterpret_cast<char*>(&length), sizeof(length));
    finp.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

    if (finp.eof()) {
      break;
    }

    int32_t nelements = 1;
    int32_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
      finp.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
      nelements *= ne[i];
    }

    std::string name(length, 0);
    finp.read(&name[0], length);

    printf("%64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ne_type_name((ne_type)ttype));

    bool quantize = false;

    // check if we should quantize this tensor
    for (const auto& s : to_quant) {
      if (std::regex_match(name, std::regex(s))) {
        quantize = true;
        break;
      }
    }

    // check if we should skip this tensor
    for (const auto& s : to_skip) {
      if (std::regex_match(name, std::regex(s))) {
        quantize = false;
        break;
      }
    }

    // quantize only 2D tensors
    quantize &= (n_dims == 2);

    if (quantize) {
      if (ttype != NE_TYPE_F32 && ttype != NE_TYPE_F16) {
        fprintf(stderr, "%s: unsupported ttype %d (%s) for integer quantization\n", __func__, ttype,
                ne_type_name((ne_type)ttype));
        return false;
      }

      if (ttype == NE_TYPE_F16) {
        data_f16.resize(nelements);
        finp.read(reinterpret_cast<char*>(data_f16.data()), nelements * sizeof(ne_fp16_t));
        data_f32.resize(nelements);
        for (int i = 0; i < nelements; ++i) {
          data_f32[i] = ne_fp16_to_fp32(data_f16[i]);
        }
      } else {
        data_f32.resize(nelements);
        finp.read(reinterpret_cast<char*>(data_f32.data()), nelements * sizeof(float));
      }

      ttype = qtype;
    } else {
      const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);

      data_u8.resize(nelements * bpe);
      finp.read(reinterpret_cast<char*>(data_u8.data()), nelements * bpe);
    }

    fout.write(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    fout.write(reinterpret_cast<char*>(&length), sizeof(length));
    fout.write(reinterpret_cast<char*>(&ttype), sizeof(ttype));
    for (int i = 0; i < n_dims; ++i) {
      fout.write(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
    }
    fout.write(&name[0], length);

    if (quantize) {
      work.resize(nelements);  // for quantization

      size_t cur_size = 0;
      std::vector<int64_t> hist_cur(1 << 4, 0);

      switch ((ne_type)ttype) {
        case NE_TYPE_Q4_0: {
          cur_size = ne_quantize_q4_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
        } break;
        case NE_TYPE_Q4_1: {
          cur_size = ne_quantize_q4_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
        } break;
        case NE_TYPE_Q5_0: {
          cur_size = ne_quantize_q5_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
        } break;
        case NE_TYPE_Q5_1: {
          cur_size = ne_quantize_q5_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
        } break;
        case NE_TYPE_Q8_0: {
          cur_size = ne_quantize_q8_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
        } break;
        case NE_TYPE_F32:
        case NE_TYPE_F16:
        case NE_TYPE_I8:
        case NE_TYPE_I16:
        case NE_TYPE_I32:
        case NE_TYPE_Q8_1:
        case NE_TYPE_COUNT: {
          fprintf(stderr, "%s: unsupported quantization type %d (%s)\n", __func__, ttype, ne_type_name((ne_type)ttype));
          return false;
        }
      }

      fout.write(reinterpret_cast<char*>(work.data()), cur_size);
      total_size_new += cur_size;

      printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float) / 1024.0 / 1024.0,
             cur_size / 1024.0 / 1024.0);
      for (int i = 0; i < (int)hist_cur.size(); ++i) {
        hist_all[i] += hist_cur[i];
      }

      for (int i = 0; i < (int)hist_cur.size(); ++i) {
        printf("%5.3f ", hist_cur[i] / (float)nelements);
      }
      printf("\n");
    } else {
      printf("size = %8.3f MB\n", data_u8.size() / 1024.0 / 1024.0);
      fout.write(reinterpret_cast<char*>(data_u8.data()), data_u8.size());
      total_size_new += data_u8.size();
    }

    total_size_org += nelements * sizeof(float);
  }

  printf("%s: model size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
  printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__, total_size_new / 1024.0 / 1024.0, ftype,
         ne_type_name(qtype));

  {
    int64_t sum_all = 0;
    for (int i = 0; i < (int)hist_all.size(); ++i) {
      sum_all += hist_all[i];
    }

    printf("%s: hist: ", __func__);
    for (int i = 0; i < (int)hist_all.size(); ++i) {
      printf("%5.3f ", hist_all[i] / (float)sum_all);
    }
    printf("\n");
  }

  return true;
}
