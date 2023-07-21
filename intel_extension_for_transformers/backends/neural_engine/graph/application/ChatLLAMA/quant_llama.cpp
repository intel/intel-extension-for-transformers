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

#include "common.h"
#include "models/llama/llama_model.h"

int main(int argc, char** argv) {
  model_init_backend();
  quant_params q_params;
  if (quant_params_parse(argc, argv, q_params) == false) {
    return 1;
  }
  const std::string fname_inp = q_params.model_file;
  const std::string fname_out = q_params.out_file;
  ne_ftype ftype = quant_params_to_ftype(q_params);
  printf("ne_ftype: %d\n", ftype);
  const int nthread = q_params.nthread;

  const int64_t t_main_start_us = model_time_us();

  int64_t t_quantize_us = 0;

  // load the model
  {
    const int64_t t_start_us = model_time_us();

    if (model_model_quantize(fname_inp.c_str(), fname_out.c_str(), q_params, ftype, nthread)) {
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
