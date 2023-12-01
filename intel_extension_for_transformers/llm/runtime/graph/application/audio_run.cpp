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
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <iostream>

#include "common.h"
#include "models/whisper/whisper.h"

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
static whisper_context** g_ctx;

static bool is_interacting = false;

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
void sigint_handler(int signo) {
  if (signo == SIGINT) {
    if (!is_interacting) {
      is_interacting = true;
    } else {
      console_cleanup(con_st);
      printf("\n");
      whisper_print_timings(*g_ctx);
      _exit(130);
    }
  }
}
#endif

// Terminal color map. 10 colors grouped in ranges [0.0, 0.1, ..., 0.9]
// Lowest is red, middle is yellow, highest is green.
const std::vector<std::string> k_colors = {
    "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
    "\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
};

bool read_wav(const std::string& fname, std::vector<float> pcmf32, std::vector<std::vector<float>> pcmf32s,
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
  pcmf32.resize(n);
  if (wav.channels == 1) {
    for (uint64_t i = 0; i < n; i++) {
      pcmf32[i] = static_cast<float>(pcm16[i]) / 32768.0f;
    }
  } else {
    for (uint64_t i = 0; i < n; i++) {
      pcmf32[i] = static_cast<float>(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
    }
  }

  if (stereo) {
    // convert to stereo, float
    pcmf32s.resize(2);

    pcmf32s[0].resize(n);
    pcmf32s[1].resize(n);
    for (uint64_t i = 0; i < n; i++) {
      pcmf32s[0][i] = static_cast<float>(pcm16[2 * i]) / 32768.0f;
      pcmf32s[1][i] = static_cast<float>(pcm16[2 * i + 1]) / 32768.0f;
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

  char* escaped = reinterpret_cast<char*>(calloc(escaped_length, 1));  // pre-zeroed
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

// helper function to replace substrings
void replace_all(std::string s, const std::string& search, const std::string& replace) {
  for (size_t pos = 0;; pos += replace.length()) {
    pos = s.find(search, pos);
    if (pos == std::string::npos) break;
    s.erase(pos, search.length());
    s.insert(pos, replace);
  }
}

bool output_txt(struct whisper_context* ctx, const char* fname, const whisper_params& params,
                std::vector<std::vector<float>> pcmf32s) {
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
                std::vector<std::vector<float>> pcmf32s) {
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
                std::vector<std::vector<float>> pcmf32s) {
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

bool output_csv(struct whisper_context* ctx, const char* fname, const whisper_params& params,
                std::vector<std::vector<float>> pcmf32s) {
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

bool output_json(struct whisper_context* ctx, const char* fname, const whisper_params& params,
                 std::vector<std::vector<float>> pcmf32s) {
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
// TODO(Bo): font parameter adjustments
bool output_wts(struct whisper_context* ctx, const char* fname, const char* fname_inp, const whisper_params& params,
                float t_sec, std::vector<std::vector<float>> pcmf32s) {
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
            for (size_t l = 0; l < txt.size(); ++l) {
              txt_fg += txt[l];
              txt_ul += "_";
            }
            txt_fg += "|";
          } else {
            for (size_t l = 0; l < txt.size(); ++l) {
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
                std::vector<std::vector<float>> pcmf32s) {
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
    snprintf(buf, sizeof(buf), "%02d:%02d.%02d", static_cast<int>(min), static_cast<int>(sec),
             static_cast<int>(msec / 100));
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

int main(int argc, char** argv) {
  whisper_params params;
#ifdef MODEL_NAME
  params.model_name = MODEL_NAME;
#endif

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

  for (size_t f = 0; f < params.fname_inp.size(); ++f) {
    const auto fname_inp = params.fname_inp[f];
    const auto fname_out =
        f < params.fname_out.size() && !params.fname_out[f].empty() ? params.fname_out[f] : params.fname_inp[f];

    std::vector<float> pcmf32;                // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s;  // stereo-channel F32 PCM

    if (!::read_wav(fname_inp, pcmf32, pcmf32s, params.diarize)) {
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
        output_wts(ctx, fname_wts.c_str(), fname_inp.c_str(), params,
                   static_cast<float>(pcmf32.size() + 1000) / WHISPER_SAMPLE_RATE, pcmf32s);
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
    }
  }

  whisper_print_timings(ctx);
  whisper_free(ctx);

  return 0;
}
