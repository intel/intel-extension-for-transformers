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
#include "models/model_utils/model_utils.h"
#include "common.h"
#include "models/model_utils/quant_utils.h"

std::shared_ptr<quant_layer_base> get_model_quant_layer(const std::string model_name) {
  return ql_registry::create_ql(model_name);
}

int main(int argc, char** argv) {
  model_init_backend();
  quant_params q_params;
#ifdef MODEL_NAME
  q_params.model_name = MODEL_NAME;
#endif

  if (quant_params_parse(argc, argv, q_params) == false) {
    return 1;
  }
  model_archs mt = model_name_to_arch::init().find(q_params.model_name);
  if (mt == MODEL_UNKNOWN) {
    fprintf(stderr, "error, please set model_name \n");
    exit(0);
  }
  q_params.model_arch = mt;

  const std::string fname_inp = q_params.model_file;
  const std::string fname_out = q_params.out_file;
  ne_ftype ftype = quant_params_to_ftype(q_params);
  printf("ne_ftype: %d\n", ftype);
  const int nthread = q_params.nthread;

  const int64_t t_main_start_us = common_time_us();

  int64_t t_quantize_us = 0;
  auto quant_layer = get_model_quant_layer(q_params.model_name);
  // load the model
  {
    const int64_t t_start_us = common_time_us();

    if (model_quantize(q_params, quant_layer)) {
      fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
      return 1;
    }

    t_quantize_us = common_time_us() - t_start_us;
  }
  // report timing
  {
    const int64_t t_main_end_us = common_time_us();

    printf("\n");
    printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0);
    printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0);
  }

  return 0;
}
