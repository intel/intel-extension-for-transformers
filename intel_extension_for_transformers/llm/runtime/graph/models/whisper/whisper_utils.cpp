//  Copyright (c) 2023 Intel Corporation
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>  //NOLINT
#include <vector>

#include "whisper.h"
#include "models/model_utils/quant_utils.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

std::string to_timestamp(int64_t t, bool comma) {
  int64_t msec = t * 10;
  int64_t hr = msec / (1000 * 60 * 60);
  msec = msec - hr * (1000 * 60 * 60);
  int64_t min = msec / (1000 * 60);
  msec = msec - min * (1000 * 60);
  int64_t sec = msec / 1000;
  msec = msec - sec * 1000;

  char buf[32];
  snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", static_cast<int>(hr), static_cast<int>(min), static_cast<int>(sec),
           comma ? "," : ".", static_cast<int>(msec));

  return std::string(buf);
}

int timestamp_to_sample(int64_t t, int n_samples) {
  return std::max(0, std::min(static_cast<int>(n_samples) - 1, static_cast<int>(((t * WHISPER_SAMPLE_RATE) / 100))));
}

bool whisper_params_parse(int argc, char** argv, whisper_params* params) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-") {
      params->fname_inp.push_back(arg);
      continue;
    }

    if (arg[0] != '-') {
      params->fname_inp.push_back(arg);
      continue;
    }

    if (arg == "-h" || arg == "--help") {
      whisper_print_usage(argc, argv, *params);
      exit(0);
    } else if (arg == "-t" || arg == "--threads") {
      params->n_threads = std::stoi(argv[++i]);
    } else if (arg == "-p" || arg == "--processors") {
      params->n_processors = std::stoi(argv[++i]);
    } else if (arg == "-ot" || arg == "--offset-t") {
      params->offset_t_ms = std::stoi(argv[++i]);
    } else if (arg == "-on" || arg == "--offset-n") {
      params->offset_n = std::stoi(argv[++i]);
    } else if (arg == "-d" || arg == "--duration") {
      params->duration_ms = std::stoi(argv[++i]);
    } else if (arg == "-mc" || arg == "--max-context") {
      params->max_context = std::stoi(argv[++i]);
    } else if (arg == "-ml" || arg == "--max-len") {
      params->max_len = std::stoi(argv[++i]);
    } else if (arg == "-bo" || arg == "--best-of") {
      params->best_of = std::stoi(argv[++i]);
    } else if (arg == "-bs" || arg == "--beam-size") {
      params->beam_size = std::stoi(argv[++i]);
    } else if (arg == "-wt" || arg == "--word-thold") {
      params->word_thold = std::stof(argv[++i]);
    } else if (arg == "-et" || arg == "--entropy-thold") {
      params->entropy_thold = std::stof(argv[++i]);
    } else if (arg == "-lpt" || arg == "--logprob-thold") {
      params->logprob_thold = std::stof(argv[++i]);
    } else if (arg == "-su" || arg == "--speed-up") {
      params->speed_up = true;
    } else if (arg == "-tr" || arg == "--translate") {
      params->translate = true;
    } else if (arg == "-di" || arg == "--diarize") {
      params->diarize = true;
    } else if (arg == "-tdrz" || arg == "--tinydiarize") {
      params->tinydiarize = true;
    } else if (arg == "-sow" || arg == "--split-on-word") {
      params->split_on_word = true;
    } else if (arg == "-nf" || arg == "--no-fallback") {
      params->no_fallback = true;
    } else if (arg == "-otxt" || arg == "--output-txt") {
      params->output_txt = true;
    } else if (arg == "-ovtt" || arg == "--output-vtt") {
      params->output_vtt = true;
    } else if (arg == "-osrt" || arg == "--output-srt") {
      params->output_srt = true;
    } else if (arg == "-owts" || arg == "--output-words") {
      params->output_wts = true;
    } else if (arg == "-olrc" || arg == "--output-lrc") {
      params->output_lrc = true;
    } else if (arg == "-fp" || arg == "--font-path") {
      params->font_path = argv[++i];
    } else if (arg == "-ocsv" || arg == "--output-csv") {
      params->output_csv = true;
    } else if (arg == "-oj" || arg == "--output-json") {
      params->output_jsn = true;
    } else if (arg == "-of" || arg == "--output-file") {
      params->fname_out.emplace_back(argv[++i]);
    } else if (arg == "-ps" || arg == "--print-special") {
      params->print_special = true;
    } else if (arg == "-pc" || arg == "--print-colors") {
      params->print_colors = true;
    } else if (arg == "-pp" || arg == "--print-progress") {
      params->print_progress = true;
    } else if (arg == "-nt" || arg == "--no-timestamps") {
      params->no_timestamps = true;
    } else if (arg == "-l" || arg == "--language") {
      params->language = argv[++i];
    } else if (arg == "-dl" || arg == "--detect-language") {
      params->detect_language = true;
    } else if (arg == "--prompt") {
      params->prompt = argv[++i];
    } else if (arg == "-m" || arg == "--model") {
      params->model = argv[++i];
    } else if (arg == "-f" || arg == "--file") {
      params->fname_inp.emplace_back(argv[++i]);
    } else {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      whisper_print_usage(argc, argv, *params);
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
  fprintf(stderr,
          "  -h,        --help              [default] show this help message "
          "and exit\n");
  fprintf(stderr,
          "  -t N,      --threads N         [%-7d] number of threads to use "
          "during computation\n",
          params.n_threads);
  fprintf(stderr,
          "  -p N,      --processors N      [%-7d] number of processors to use "
          "during computation\n",
          params.n_processors);
  fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n", params.offset_t_ms);
  fprintf(stderr, "  -on N,     --offset-n N        [%-7d] segment index offset\n", params.offset_n);
  fprintf(stderr,
          "  -d  N,     --duration N        [%-7d] duration of audio to "
          "process in milliseconds\n",
          params.duration_ms);
  fprintf(stderr,
          "  -mc N,     --max-context N     [%-7d] maximum number of text "
          "context tokens to store\n",
          params.max_context);
  fprintf(stderr,
          "  -ml N,     --max-len N         [%-7d] maximum segment length in "
          "characters\n",
          params.max_len);
  fprintf(stderr,
          "  -sow,      --split-on-word     [%-7s] split on word rather than "
          "on token\n",
          params.split_on_word ? "true" : "false");
  fprintf(stderr,
          "  -bo N,     --best-of N         [%-7d] number of best candidates "
          "to keep\n",
          params.best_of);
  fprintf(stderr, "  -bs N,     --beam-size N       [%-7d] beam size for beam search\n", params.beam_size);
  fprintf(stderr,
          "  -wt N,     --word-thold N      [%-7.2f] word timestamp "
          "probability threshold\n",
          params.word_thold);
  fprintf(stderr,
          "  -et N,     --entropy-thold N   [%-7.2f] entropy threshold for "
          "decoder fail\n",
          params.entropy_thold);
  fprintf(stderr,
          "  -lpt N,    --logprob-thold N   [%-7.2f] log probability threshold "
          "for decoder fail\n",
          params.logprob_thold);
  fprintf(stderr,
          "  -su,       --speed-up          [%-7s] speed up audio by x2 "
          "(reduced accuracy)\n",
          params.speed_up ? "true" : "false");
  fprintf(stderr,
          "  -tr,       --translate         [%-7s] translate from source "
          "language to english\n",
          params.translate ? "true" : "false");
  fprintf(stderr, "  -di,       --diarize           [%-7s] stereo audio diarization\n",
          params.diarize ? "true" : "false");
  fprintf(stderr,
          "  -tdrz,     --tinydiarize       [%-7s] enable tinydiarize "
          "(requires a tdrz model)\n",
          params.tinydiarize ? "true" : "false");
  fprintf(stderr,
          "  -nf,       --no-fallback       [%-7s] do not use temperature "
          "fallback while decoding\n",
          params.no_fallback ? "true" : "false");
  fprintf(stderr, "  -otxt,     --output-txt        [%-7s] output result in a text file\n",
          params.output_txt ? "true" : "false");
  fprintf(stderr, "  -ovtt,     --output-vtt        [%-7s] output result in a vtt file\n",
          params.output_vtt ? "true" : "false");
  fprintf(stderr, "  -osrt,     --output-srt        [%-7s] output result in a srt file\n",
          params.output_srt ? "true" : "false");
  fprintf(stderr, "  -olrc,     --output-lrc        [%-7s] output result in a lrc file\n",
          params.output_lrc ? "true" : "false");
  fprintf(stderr,
          "  -owts,     --output-words      [%-7s] output script for "
          "generating karaoke video\n",
          params.output_wts ? "true" : "false");
  fprintf(stderr,
          "  -fp,       --font-path         [%-7s] path to a monospace font "
          "for karaoke video\n",
          params.font_path.c_str());
  fprintf(stderr, "  -ocsv,     --output-csv        [%-7s] output result in a CSV file\n",
          params.output_csv ? "true" : "false");
  fprintf(stderr, "  -oj,       --output-json       [%-7s] output result in a JSON file\n",
          params.output_jsn ? "true" : "false");
  fprintf(stderr,
          "  -of FNAME, --output-file FNAME [%-7s] output file path (without "
          "file extension)\n",
          "");
  fprintf(stderr, "  -ps,       --print-special     [%-7s] print special tokens\n",
          params.print_special ? "true" : "false");
  fprintf(stderr, "  -pc,       --print-colors      [%-7s] print colors\n", params.print_colors ? "true" : "false");
  fprintf(stderr, "  -pp,       --print-progress    [%-7s] print progress\n", params.print_progress ? "true" : "false");
  fprintf(stderr, "  -nt,       --no-timestamps     [%-7s] do not print timestamps\n",
          params.no_timestamps ? "true" : "false");
  fprintf(stderr,
          "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for "
          "auto-detect)\n",
          params.language.c_str());
  fprintf(stderr,
          "  -dl,       --detect-language   [%-7s] exit after automatically "
          "detecting language\n",
          params.detect_language ? "true" : "false");
  fprintf(stderr, "             --prompt PROMPT     [%-7s] initial prompt\n", params.prompt.c_str());
  fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n", params.model.c_str());
  fprintf(stderr, "  -f FNAME,  --file FNAME        [%-7s] input WAV file path\n", "");
  fprintf(stderr, "\n");
}
