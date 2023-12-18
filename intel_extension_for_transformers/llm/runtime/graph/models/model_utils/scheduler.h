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
  il_worker(const gpt_params& params, const int& n_threads);  // TODO rm n_threads
  virtual ~il_worker();

  virtual const bool step(sequence* seqs, const int& n_input) = 0;
  virtual const bool steps(sequence* seqs, const int& n_input, const int& n_step) = 0;
  // virtual const int greedy_search_step(sequence* seqs, const int& n_input) = 0;
  virtual const bool beam_search_step(sequence* seqs, const int& n_input) = 0;
  virtual const bool prepare_inputs(sequence* seqs, const int& n_input, model_input* inputs) = 0;
  void set_threads(const int& n_threads);
  const std::vector<int>& get_request_done_ids() const;

 protected:
  model_context* m_ctx = NULL;
  int threads;
  beam_search_flow* bsf = nullptr;
  std::vector<int> request_done_ids;
};

// iteration-level scheduler
class il_scheduler {
 public:
  explicit il_scheduler(const gpt_params& params);
  il_scheduler(const gpt_params& params, const serve_policy& policy);
  virtual ~il_scheduler();

  virtual const bool add_request(sequence* seq);
  virtual const bool step() = 0;
  virtual const bool steps() = 0;
  const bool has_finished_seq();
  std::vector<sequence*> pop_completed_requests();
  // void print_progress();
  // virtual void prepare_inputs();

 protected:
  virtual void update_seqs() = 0;

  const serve_policy policy;
  const gpt_params params;
  serve_pool waiting_pool;
  serve_pool running_pool;
  serve_pool holding_pool;
  serve_pool finished_pool;
};

// single-prompt-batched-generation worker
class spbg_worker : public il_worker {
 public:
  explicit spbg_worker(const gpt_params& params);
  spbg_worker(const gpt_params& params, const int& n_threads);
  ~spbg_worker();

  const bool step(sequence* seqs, const int& n_input) override;
  const bool steps(sequence* seqs, const int& n_input, const int& n_step) override;
  // const int greedy_search_step(sequence* seqs, const int& n_input) override;
  const bool beam_search_step(sequence* seqs, const int& n_input) override;
  const bool prepare_inputs(sequence* seqs, const int& n_input, model_input* inputs) override;
};

// single-prompt-batched-generation scheduler
class spbg_scheduler : public il_scheduler {
 public:
  explicit spbg_scheduler(const gpt_params& params);
  spbg_scheduler(const gpt_params& params, const serve_policy& policy);
  ~spbg_scheduler();

  const bool step() override;
  const bool steps() override;
  void steps_decoding_for_next_prefill();

 protected:
  void update_seqs() override;

  const int max_requests;
  int cur_decoding_size = -1;
  spbg_worker wr;
};

#endif  // SCHEDULER_H
