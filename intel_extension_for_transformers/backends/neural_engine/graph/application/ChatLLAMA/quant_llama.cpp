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

#include <stdint.h>
#include <cstdio>
#include <map>
#include <string>
#include <exception>
#include <utility>
#include <unordered_map>
#include <tuple>

#include "models/llama/llama_model.h"

struct MyHash {
  std::size_t operator()(const std::tuple<int, std::string, int, std::string, std::string>& k) const {
    return std::hash<int>()(std::get<0>(k))
           ^ (std::hash<std::string>()(std::get<1>(k)))
           ^ std::hash<int>()(std::get<2>(k))
           ^ (std::hash<std::string>()(std::get<3>(k)))
           ^ (std::hash<std::string>()(std::get<4>(k)));
  }
};


static std::unordered_map<std::tuple<int, std::string, int, std::string, std::string>, enum ne_ftype, MyHash>
NE_FTYPE_MAP = {
    // bits, alg, block size, scale dtype, gemm_isa -> ne_ftype
    {{4,  "sym",   QK4_0,  "fp32",  "none"}, NE_FTYPE_MOSTLY_Q4_0},
    {{4, "asym",   QK4_1,  "fp32",  "none"}, NE_FTYPE_MOSTLY_Q4_1},
    {{5,  "sym",   QK5_0,  "fp32",  "none"}, NE_FTYPE_MOSTLY_Q5_0},
    {{5, "asym",   QK5_1,  "fp32",  "none"}, NE_FTYPE_MOSTLY_Q5_1},
    {{8,  "sym",   QK8_0,  "fp32",  "none"}, NE_FTYPE_MOSTLY_Q8_0},
    {{4,  "sym",      32,  "fp32",  "amx"}, NE_FTYPE_MOSTLY_Q4_JBLAS_B32},
    {{4,  "sym",      32,  "bf16",  "amx"}, NE_FTYPE_MOSTLY_Q4_JBLAS_BF16_B32},
    {{4,  "sym",     128,  "fp32",  "amx"}, NE_FTYPE_MOSTLY_Q4_JBLAS_B128},
    {{4,  "sym",   -1024,  "fp32",  "amx"}, NE_FTYPE_MOSTLY_Q4_JBLAS_B128},
    {{4,  "sym",      32,  "fp32",  "vnni"}, NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B32},
    {{4,  "sym",     128,  "fp32",  "vnni"}, NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B128},
    {{4,  "sym",      32,  "bf16",  "vnni"}, NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_BF16_B32},
};

struct quant_params {
    std::string model_file = "";
    std::string out_file = "";
    std::string config = "";
    int nthread = 1;

    int32_t bits = 4;
    std::string alg = "sym";
    int32_t block_size = 32;
    std::string scale_dtype = "fp32";
    std::string gemm_isa = "none";
};

void quant_print_usage(int argc, char** argv, const quant_params& params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  --model_file          path to the fp32 model\n");
    fprintf(stderr, "  --out_file            path to the quantized model\n");
    fprintf(stderr, "  --config              path to the configuration file (default: "")\n");
    fprintf(stderr, "  --nthread N           number of threads to use (default: 1)\n");
    fprintf(stderr, "  --bits N              number of bits to use for quantization (default: 4)\n");
    fprintf(stderr, "  --alg                 qquantization algorithm to use: sym/asym (default: sym)\n");
    fprintf(stderr, "  --block_size N        block size (default: 32)\n");
    fprintf(stderr, "  --scale_dtype dtype   fp32/bf16 type for scales (default: fp32)\n");
    fprintf(stderr, "  --gemm_isa            instruction set architecture to use for GEMM computation: vnni/ams/none (default: none)\n");
    fprintf(stderr, "\n");
}

bool quant_params_parse(int argc, char** argv, quant_params& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model_file") {
            params.model_file = argv[++i];
        } else if (arg == "--out_file") {
            params.out_file = argv[++i];
        } else if (arg == "--config") {
            params.config = argv[++i];
        } else if (arg == "--nthread") {
            params.nthread = std::stoi(argv[++i]);
        } else if (arg == "--bits") {
            params.bits = std::stoi(argv[++i]);
        } else if (arg == "--alg") {
            params.alg = argv[++i];
        } else if (arg == "--block_size") {
            params.block_size = std::stoi(argv[++i]);
        } else if (arg == "--scale_dtype") {
            params.scale_dtype = argv[++i];
        } else if (arg == "--gemm_isa") {
            params.gemm_isa = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            quant_print_usage(argc, argv, params);
            exit(0);
        } else {
            exit(0);
        }
    }

    return true;
}

int main(int argc, char** argv) {
  model_init_backend();
  quant_params q_params;
  if (quant_params_parse(argc, argv, q_params) == false) {
      return 1;
  }
  const std::string fname_inp = q_params.model_file;
  const std::string fname_out = q_params.out_file;
  ne_ftype ftype = NE_FTYPE_MAP[
      std::make_tuple(q_params.bits, q_params.alg, q_params.block_size, q_params.scale_dtype, q_params.gemm_isa)];
  printf("ne_ftype: %d\n", ftype);
  const int nthread = q_params.nthread;

  const int64_t t_main_start_us = model_time_us();

  int64_t t_quantize_us = 0;

  // load the model
  {
    const int64_t t_start_us = model_time_us();

    if (model_model_quantize(fname_inp.c_str(), fname_out.c_str(), ftype, nthread)) {
      fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
      return 1;
    }

    t_quantize_us = model_time_us() - t_start_us;
  }

  // report timing
  {
    const int64_t t_main_end_us = model_time_us();

    printf("\n");
    printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0);
    printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0);
  }

  return 0;
}
