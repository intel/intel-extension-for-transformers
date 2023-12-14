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

// iteration-level worker
class il_worker {
 public:
  explicit il_worker(const gpt_params& params);
  il_worker(const gpt_params& params, const int& n_threads);
  virtual ~il_worker();

  virtual const int step(sequence* seqs, const int& n_input) = 0;
  virtual const int steps(sequence* seqs, const int& n_input, const int& n_step) = 0;
  // virtual const int greedy_search_step(sequence* seqs, const int& n_input) = 0;
  virtual const int beam_search_step(sequence* seqs, const int& n_input) = 0;
  virtual const int prepare_inputs(sequence* seqs, const int& n_input, model_input* inputs) = 0;
  void set_threads(const int& n_threads);

 protected:
  model_context* m_ctx = NULL;
  int threads;
  beam_search_flow* bsf = nullptr;
  std::vector<int> request_done_ids;
};

// iteration-level scheduler
class il_scheduler {
 public:
  void add_request(const sequence& seq);
  const int step();
  const int steps();
  std::vector<sequence> pop_completed_requests();
  void print_progress();
  virtual void prepare_inputs();

 protected:
  serve_policy policy;
  serve_pool waiting_pool;
  serve_pool running_pool;
  serve_pool holding_pool;
  serve_pool finished_pool;
  gpt_params params;
};

// single-prompt-batched-generation worker
class spbg_worker : public il_worker {
 public:
  explicit spbg_worker(const gpt_params& params);
  spbg_worker(const gpt_params& params, const int& n_threads);
  ~spbg_worker();

  const int step(sequence* seqs, const int& n_input) override;
  const int steps(sequence* seqs, const int& n_input, const int& n_step) override;
  // const int greedy_search_step(sequence* seqs, const int& n_input) override;
  const int beam_search_step(sequence* seqs, const int& n_input) override;
  const int prepare_inputs(sequence* seqs, const int& n_input, model_input* inputs) override;
};

// single-prompt-batched-generation scheduler
class spbg_scheduler : public il_scheduler {
 public:
  void steps_decoding_for_next_prefill();

 protected:
  int max_requests = 32;
  int cur_decoding_size = -1;
  spbg_worker wr;
};

#endif  // SCHEDULER_H
