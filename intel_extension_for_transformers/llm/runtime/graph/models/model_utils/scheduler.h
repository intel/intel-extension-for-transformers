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
  virtual ~il_worker();
  virtual bool step(std::vector<sequence*>* seqs, const int& n_input) = 0;
  // virtual bool greedy_search_step(sequence* seqs, const int& n_input) = 0;
  virtual bool beam_search_step(std::vector<sequence*>* seqs, const int& n_input) = 0;

  void set_threads(const int& n_threads);
  std::vector<int> get_request_done_ids() const;
  void empty_request_done_ids();

 protected:
  virtual bool prepare_inputs(std::vector<sequence*>* seqs, const int& n_input, model_input* inputs) = 0;
  virtual bool update_seqs(std::vector<sequence*>* seqs, const int& n_input) = 0;

  model_context* m_ctx = NULL;
  int threads;
  beam_search_flow* bsf = nullptr;
  std::vector<int> request_done_ids;
  std::unordered_map<int, int> reqidx_to_vecid;
};

// continuous batching generation worker
class cbg_worker : public il_worker {
 public:
  explicit cbg_worker(const gpt_params& params);
  cbg_worker(const gpt_params& params, const int& n_threads);
  ~cbg_worker();

  bool step(std::vector<sequence*>* seqs, const int& n_input) override;
  // bool greedy_search_step(sequence* seqs, const int& n_input) override;
  bool beam_search_step(std::vector<sequence*>*, const int& n_input) override;

 protected:
  bool prepare_inputs(std::vector<sequence*>*, const int& n_input, model_input* inputs) override;
  bool update_seqs(std::vector<sequence*>* seqs, const int& n_input) override;
};

// iteration-level scheduler
class il_scheduler {
 public:
  explicit il_scheduler(const gpt_params& params);
  il_scheduler(const gpt_params& params, const serve_policy& policy);
  virtual ~il_scheduler();

  // TODO (YZT) kv cache ptr as input params
  virtual bool add_request(sequence* seq) = 0;
  virtual bool step() = 0;
  virtual bool done() = 0;
  bool has_finished_seq();
  std::vector<sequence*> pop_completed_requests();
  // void print_progress();

 protected:
  virtual bool prepare_seqs() = 0;
  virtual bool update_pools() = 0;

  const serve_policy policy;
  const gpt_params params;
  serve_pool waiting_pool;
  serve_pool running_pool;
  serve_pool finished_pool;
};

// continuous batching generation scheduler
class cbg_scheduler : public il_scheduler {
 public:
  explicit cbg_scheduler(const gpt_params& params);
  cbg_scheduler(const gpt_params& params, const serve_policy& policy);
  ~cbg_scheduler();

  bool add_request(sequence* seq) override;
  bool step() override;
  bool done() override;

 protected:
  bool prepare_seqs() override;
  bool update_pools() override;
  int query_free_req_idx();

  const int max_requests;
  cbg_worker wr;
  std::vector<sequence*> executed_seqs;
  std::vector<bool> free_req_idx;
  // TODO (YZT) too long will hurt performance?
  int64_t max_input_length;
  int cur_running_num = -1;
  // reserve at least one position for next prompt hidden states prefilling
  // when running_pool is full (size == max_requests)
  bool steps_decoding_for_next_prefill = false;
};

#endif  // SCHEDULER_H
