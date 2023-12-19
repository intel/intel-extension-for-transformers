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
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex> //NOLINT
#include "models/model_utils/quant_utils.h"
#include "common.h"

int main(int argc, char** argv) {
  quant_params q_params;
  if (quant_params_parse(argc, argv, q_params) == false) {
    return 1;
  }

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
    return 1;
  }

  const int64_t t_main_start_us = common_time_us();

  int64_t t_quantize_us = 0;

  // load the model
  {
    const int64_t t_start_us = common_time_us();

    if (!whisper_model_quantize(fname_inp, fname_out, ne_ftype(ftype))) {
      fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
      return 1;
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

  return 0;
}
