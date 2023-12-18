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
  void inference(const std::string& fname_inp);
  bool read_wav(const std::string& fname, std::vector<float>* pcmf32, std::vector<std::vector<float>>* pcmf32s,
                bool stereo);

 private:
  whisper_context* ctx = nullptr;
  whisper_params params;
};

void Model::init_model(const std::string& model_path) {
  params.model = model_path;
  ctx = whisper_init_from_file(params.model.c_str());

  if (ctx == nullptr) {
    fprintf(stderr, "error: failed to initialize whisper context\n");
    return;
  }
}

bool Model::read_wav(const std::string& fname, std::vector<float>* pcmf32, std::vector<std::vector<float>>* pcmf32s,
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
      // if (!wparams.print_realtime) {
      //   wparams.new_segment_callback = whisper_print_segment_callback;
      //   wparams.new_segment_callback_user_data = &user_data;
      // }

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
      .def("init_model", &Model::init_model, "initial model with model path and parameters", py::arg("model_path"))
      // .dif("read_wav", &Model::read_wav, "Read audio", py::arg("fname"), py::arg())
      .def("inference", &Model::inference, "Translate audio to text");
}