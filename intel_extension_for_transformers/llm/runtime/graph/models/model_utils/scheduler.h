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

#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "models/model_utils/pool.h"
#include "models/model_utils/model_utils.h"

class worker {
 public:
  void step(model_input* inputs, const int& n_input);
  void steps(model_input* inputs, const int& n_input, const int& n_step);
  void model_forward();
  void post_process();
  void greedy_search();

 protected:
  model_context* ctx = nullptr;
  int threads;
  beam_search_flow bsf;
};

// iteration-level scheduler
class il_scheduler {
 public:
  void step();
  void steps();
  void print_progress();

 protected:
  serve_policy policy;
  serve_pool waiting_pool;
  serve_pool running_pool;
  worker wr;
  model_context* ctx;
  std::vector<sequence> results;
};

// single-prompt-batched-generation scheduler
class spbg_scheduler : public il_scheduler {
 public:
  void steps_decoding_for_next_prefill();

 protected:
  int max_batch_size = 32;
  int cur_decoding_size = -1;
};

#endif  // SCHEDULER_H
