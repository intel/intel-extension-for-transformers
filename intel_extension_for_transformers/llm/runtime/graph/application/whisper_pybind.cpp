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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "models/whisper/whisper.h"
#include "models/model_utils/quant_utils.h"
#include "application/common.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <signal.h>
#include <windows.h>
#endif

namespace py = pybind11;

#define STATIC_INPUT_HEAD_IDX 0
class Model {
 public:
  Model() {
    struct ne_init_params params = {0, NULL, false};
    struct ne_context* ctx = ne_init(params);
    ne_free(ctx);
  }
  ~Model() {
    if (ctx) delete (ctx);
  }
  void init_model(const std::string& model_path);
  static void quant_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                          const std::string& alg, int group_size, const std::string& scale_dtype,
                          const std::string& compute_dtype, bool use_ggml, int threads);
  void inference(const std::string& fname_inp);

 private:
  whisper_context* ctx = nullptr;
  whisper_params params;
};

const std::vector<std::string> k_colors = {
    "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
    "\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
};

void Model::init_model(const std::string& model_path) {
  params.model = model_path;
  ctx = whisper_init_from_file(params.model.c_str());

  if (ctx == nullptr) {
    fprintf(stderr, "error: failed to initialize whisper context\n");
    return;
  }
}

void Model::quant_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                        const std::string& alg, int group_size, const std::string& scale_dtype,
                        const std::string& compute_dtype, bool use_ggml, int threads) {
  quant_params q_params;
  q_params.model_file = model_path;
  q_params.out_file = out_path;
  q_params.use_ggml = use_ggml;
  q_params.nthread = threads;
  // needed to initialize f16 tables
  {
    struct ne_init_params params = {0, NULL, false};
    struct ne_context* ctx = ne_init(params);
    ne_free(ctx);
  }
  const std::string fname_inp = q_params.model_file;
  const std::string fname_out = q_params.out_file;
  // printf("input_model_file:%s \n",fname_inp.c_str());

  const ne_ftype ftype = quant_params_to_ftype(q_params);
  if (ftype != NE_FTYPE_MOSTLY_Q4_0) {
    fprintf(stderr, "%s: ITREX now only support quantize model to q4_0 \n", __func__);
    return;
  }

  const int64_t t_main_start_us = common_time_us();

  int64_t t_quantize_us = 0;

  // load the model
  {
    const int64_t t_start_us = common_time_us();

    if (!whisper_model_quantize(fname_inp, fname_out, ne_ftype(ftype))) {
      fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
      return;
    }

    t_quantize_us = common_time_us() - t_start_us;
  }

  // report timing
  {
    const int64_t t_main_end_us = common_time_us();

    printf("\n");
    printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0f);
    printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
  }

  return;
}

bool read_wav(const std::string& fname, std::vector<float>* pcmf32, std::vector<std::vector<float>>* pcmf32s,
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

  if (wav.sampleRate != WHISPER_SAMPLE_RATE) {
    fprintf(stderr, "%s: WAV file '%s' must be %i kHz\n", __func__, fname.c_str(), WHISPER_SAMPLE_RATE / 1000);
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
  (*pcmf32).resize(n);
  if (wav.channels == 1) {
    for (uint64_t i = 0; i < n; i++) {
      (*pcmf32)[i] = static_cast<float>(pcm16[i]) / 32768.0f;
    }
  } else {
    for (uint64_t i = 0; i < n; i++) {
      (*pcmf32)[i] = static_cast<float>(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
    }
  }

  if (stereo) {
    // convert to stereo, float
    (*pcmf32s).resize(2);

    (*pcmf32s)[0].resize(n);
    (*pcmf32s)[1].resize(n);
    for (uint64_t i = 0; i < n; i++) {
      (*pcmf32s)[0][i] = static_cast<float>(pcm16[2 * i]) / 32768.0f;
      (*pcmf32s)[1][i] = static_cast<float>(pcm16[2 * i + 1]) / 32768.0f;
    }
  }

  return true;
}

std::string estimate_diarization_speaker(std::vector<std::vector<float>> pcmf32s, int64_t t0, int64_t t1,
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

void whisper_print_segment_callback(struct whisper_context* ctx, struct whisper_state* /*state*/, int n_new,
                                    void* user_data) {
  const auto& params = *(reinterpret_cast<whisper_print_user_data*>(user_data))->params;
  const auto& pcmf32s = *(reinterpret_cast<whisper_print_user_data*>(user_data))->pcmf32s;

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

        const int col = std::max(0, std::min(static_cast<int>(k_colors.size()) - 1,
                                             static_cast<int>(std::pow(p, 3) * static_cast<float>(k_colors.size()))));

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
void Model::inference(const std::string& fname_inp) {
  params.fname_inp.emplace_back(fname_inp);
  if (params.fname_inp.empty()) {
    fprintf(stderr, "error: no input files specified\n");
    return;
  }

  if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
    fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
    exit(0);
  }

  if (params.diarize && params.tinydiarize) {
    fprintf(stderr, "error: cannot use both --diarize and --tinydiarize\n");
    exit(0);
  }
  for (size_t f = 0; f < params.fname_inp.size(); ++f) {
    const auto fname_inp = params.fname_inp[f];
    const auto fname_out =
        f < params.fname_out.size() && !params.fname_out[f].empty() ? params.fname_out[f] : params.fname_inp[f];

    std::vector<float> pcmf32;                // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s;  // stereo-channel F32 PCM

    if (!read_wav(fname_inp, &pcmf32, &pcmf32s, params.diarize)) {
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
              __func__, fname_inp.c_str(), static_cast<int>(pcmf32.size()),
              static_cast<float>(pcmf32.size()) / WHISPER_SAMPLE_RATE, params.n_threads, params.n_processors,
              params.language.c_str(), params.translate ? "translate" : "transcribe",
              params.tinydiarize ? "tdrz = 1, " : "", params.no_timestamps ? 0 : 1);

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

      wparams.tdrz_enable = params.tinydiarize;  // [TDRZ]

      wparams.initial_prompt = params.prompt.c_str();

      wparams.greedy.best_of = params.best_of;
      wparams.beam_search.beam_size = params.beam_size;

      wparams.temperature_inc = params.no_fallback ? 0.0f : wparams.temperature_inc;
      wparams.entropy_thold = params.entropy_thold;
      wparams.logprob_thold = params.logprob_thold;

      whisper_print_user_data user_data = {&params, &pcmf32s};

      // this callback is called on each new segment
      if (!wparams.print_realtime) {
        wparams.new_segment_callback = whisper_print_segment_callback;
        wparams.new_segment_callback_user_data = &user_data;
      }

      // example for abort mechanism
      // in this example, we do not abort the processing, but we could if the flag is set to true
      // the callback is called before every encoder run - if it returns false, the processing is aborted
      {
        static bool is_aborted = false;  // NOTE: this should be atomic to avoid data race

        wparams.encoder_begin_callback = [](struct whisper_context* /*ctx*/, struct whisper_state* /*state*/,
                                            void* user_data) {
          bool is_aborted = *(reinterpret_cast<bool*>(user_data));
          return !is_aborted;
        };
        wparams.encoder_begin_callback_user_data = &is_aborted;
      }

      if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
        fprintf(stderr, "%s: failed to process audio\n", fname_inp);
        return;
      }
    }
  }
  whisper_print_timings(ctx);
  return;
}

#if MODEL_NAME_ID == 16

PYBIND11_MODULE(whisper_cpp, m)
#endif
{
  m.doc() = "cpp model python binding";
  py::class_<Model>(m, "Model", py::module_local())
      .def(py::init())
      .def("init_model", &Model::init_model, "initial model with model path and parameters",
                  py::arg("model_path"))
      .def_static("quant_model", &Model::quant_model, "Quantize model", py::arg("model_path"), py::arg("out_path"),
           py::arg("weight_dtype") = "int4", py::arg("alg") = "sym", py::arg("group_size") = 32,
           py::arg("scale_dtype") = "fp32", py::arg("compute_dtype") = "int8", py::arg("use_ggml") = false,
           py::arg("threads") = 8)
      .def("inference", &Model::inference, "Translate audio to text");
}
