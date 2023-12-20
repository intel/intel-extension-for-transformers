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
  virtual const bool step(std::vector<sequence*>* seqs, const int& n_input) = 0;
  // virtual const int greedy_search_step(sequence* seqs, const int& n_input) = 0;
  virtual const bool beam_search_step(std::vector<sequence*>* seqs, const int& n_input) = 0;

  void set_threads(const int& n_threads);
  const std::vector<int>& get_request_done_ids() const;
  void empty_request_done_ids();

 protected:
  virtual const bool prepare_inputs(std::vector<sequence*>* seqs, const int& n_input, model_input* inputs) = 0;
  virtual const bool update_seqs(std::vector<sequence*>* seqs, const int& n_input) = 0;

  model_context* m_ctx = NULL;
  int threads;
  beam_search_flow* bsf = nullptr;
  std::vector<int> request_done_ids;
  std::unordered_map<int, int> reqidx_to_vecid;
};

// single-prompt-batched-generation worker
class spbg_worker : public il_worker {
 public:
  explicit spbg_worker(const gpt_params& params);
  spbg_worker(const gpt_params& params, const int& n_threads);
  ~spbg_worker();

  const bool step(std::vector<sequence*>* seqs, const int& n_input) override;
  // const int greedy_search_step(sequence* seqs, const int& n_input) override;
  const bool beam_search_step(std::vector<sequence*>*, const int& n_input) override;

 protected:
  const bool prepare_inputs(std::vector<sequence*>*, const int& n_input, model_input* inputs) override;
  const bool update_seqs(std::vector<sequence*>* seqs, const int& n_input) override;
};

// iteration-level scheduler
class il_scheduler {
 public:
  explicit il_scheduler(const gpt_params& params);
  il_scheduler(const gpt_params& params, const serve_policy& policy);
  virtual ~il_scheduler();

  virtual const bool add_request(sequence* seq) = 0;
  virtual const bool step() = 0;
  virtual const bool done() = 0;
  const bool has_finished_seq();
  std::vector<sequence*> pop_completed_requests();
  // void print_progress();

 protected:
  virtual const bool prepare_seqs() = 0;
  virtual const bool update_pools() = 0;

  const serve_policy policy;
  const gpt_params params;
  serve_pool waiting_pool;
  serve_pool running_pool;
  serve_pool finished_pool;
};

// single-prompt-batched-generation scheduler
class spbg_scheduler : public il_scheduler {
 public:
  explicit spbg_scheduler(const gpt_params& params);
  spbg_scheduler(const gpt_params& params, const serve_policy& policy);
  ~spbg_scheduler();

  const bool add_request(sequence* seq) override;
  const bool step() override;
  const bool done() override;

 protected:
  const bool prepare_seqs() override;
  const bool update_pools() override;
  const int query_free_req_idx();

  const int max_requests;
  spbg_worker wr;
  std::vector<sequence*> executed_seqs;
  std::vector<bool> free_req_idx;
  // only execute one prompt in spbg_scheduler;
  const int pre_prefill_num = 1;
  int cur_decoding_num = -1;
  // reserve at least one position for next prompt hidden states prefilling
  // when running_pool is full (size == max_requests)
  bool steps_decoding_for_next_prefill = false;
};

#endif  // SCHEDULER_H
