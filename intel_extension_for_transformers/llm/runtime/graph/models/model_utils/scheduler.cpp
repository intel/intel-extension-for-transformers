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

#include "models/model_utils/scheduler.h"

il_worker::il_worker(const gpt_params& params) {
  m_ctx = model_init_from_gpt_params(params);
  if (m_ctx == NULL) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    exit(0);
  }
  if (m_ctx->beam_search && bsf == nullptr) {
    bsf = new beam_search_flow(m_ctx, m_ctx->max_request_num);
  }
  threads = params.n_threads;
}

il_worker::il_worker(const gpt_params& params, const int& n_threads) : il_worker(params) { threads = n_threads; }

il_worker::~il_worker() {
  // TODO (YZT) consider thread safe?
  if (m_ctx != NULL) {
    model_free(m_ctx);
  }
  if (bsf != nullptr) {
    delete bsf;
  }
}

void il_worker::set_threads(const int& n_threads) { threads = n_threads; }
const std::vector<int>& il_worker::get_request_done_ids() const { return request_done_ids; }

spbg_worker::spbg_worker(const gpt_params& params) : il_worker(params) {}
spbg_worker::spbg_worker(const gpt_params& params, const int& n_threads) : il_worker(params, n_threads) {}
spbg_worker::~spbg_worker() {}

const bool spbg_worker::prepare_inputs(sequence* seqs, const int& n_input, model_input* inputs) {
  for (int ni = 0; ni < n_input; ++ni) {
    if ((seqs + ni)->status == seq_status::WAITING || (seqs + ni)->status == seq_status::UNKNOWN) {
      fprintf(stderr, "%s: error: request status is unright.\n", __func__);
      return false;
    } else if ((seqs + ni)->status == seq_status::PREFILL) {
      (inputs + ni)->tokens = (seqs + ni)->prompt_ids.data();
      (inputs + ni)->n_tokens = (seqs + ni)->n_prompt_tokens;
      (inputs + ni)->n_past = 0;
      (inputs + ni)->n_total = 0;
      (inputs + ni)->request_idx = (seqs + ni)->request_idx;
      // do not support padding for now
      (inputs + ni)->n_padding = 0;
    } else if ((seqs + ni)->status == seq_status::DECODING) {
      (inputs + ni)->tokens = &(seqs + ni)->generated_ids.back();
      (inputs + ni)->n_tokens = 1;
      (inputs + ni)->n_past = (seqs + ni)->n_past;
      (inputs + ni)->n_total = (seqs + ni)->n_total;
      (inputs + ni)->request_idx = (seqs + ni)->request_idx;
      // do not support padding for now
      (inputs + ni)->n_padding = 0;
    } else {
      continue;
      ;
    }
  }
  return true;
}

const bool spbg_worker::beam_search_step(sequence* seqs, const int& n_input) {
  // add new request
  // step prefill
  if (n_input == 1) {
    if (seqs->status != seq_status::PREFILL) {
      fprintf(stderr, "%s: error: request status must be PERFILL when n_input = 1.\n", __func__);
      return false;
    }
    model_input pr_input;
    if (!prepare_inputs(seqs, n_input, &pr_input)) {
      return false;
    }
    if (!bsf->step_prefill(pr_input)) {
      return false;
    }
    seqs->status = seq_status::DECODING;
    seqs->n_past = seqs->n_prompt_tokens;
    return true;
  }
  // no new request
  // step beam search decoding
  for (int ni = 0; ni < n_input; ++ni) {
    if ((seqs + ni)->status != seq_status::DECODING) {
      fprintf(stderr, "%s: error: all requests status must be DECODING when n_input > 1.\n", __func__);
      return false;
    }
  }
  if (!bsf->step_decoding()) {
    return false;
  }
  request_done_ids.clear();
  request_done_ids = bsf->request_done_ids();
  return true;
}

const bool spbg_worker::step(sequence* seqs, const int& n_input) {
  if (m_ctx->beam_search && bsf != nullptr) {
    return beam_search_step(seqs, n_input);
  }
  return false;  // TODO (YZT) greedy search and top_p-top_k sampling
}

il_scheduler::il_scheduler(const gpt_params& params, const serve_policy& policy)
    : params(params),
      policy(policy),
      waiting_pool(pool_property::WAITING),
      running_pool(pool_property::RUNNING),
      holding_pool(pool_property::HOLDING),
      finished_pool(pool_property::FINISHED) {}

il_scheduler::il_scheduler(const gpt_params& params) : il_scheduler(params, serve_policy::FCFS) {}

il_scheduler::~il_scheduler() {}

const bool il_scheduler::add_request(sequence* seq) {
  if (seq->status != seq_status::UNKNOWN) {
    fprintf(stderr, "%s: error: seq status is not UNKNOWN, can not decide to add into which pool.\n", __func__);
    return false;
  }
  // add into waiting_pool by default
  return waiting_pool.add(seq);
}

const bool il_scheduler::has_finished_seq() { return (finished_pool.size() > 0); }

std::vector<sequence*> il_scheduler::pop_completed_requests() {
  std::vector<sequence*> ret_seqs;
  const int length = finished_pool.size();
  if (length > 0) {
    return ret_seqs;
  }
  ret_seqs.resize(length);
  for (int l = 0; l < length; ++l) {
    if (!finished_pool.pop(ret_seqs[l])) {
      fprintf(stderr, "%s: error: pop finished_pool %dth elements failed.\n", __func__, l);
      return std::vector<sequence*>();
    }
  }
  return ret_seqs;
}

spbg_scheduler::spbg_scheduler(const gpt_params& params)
    : il_scheduler(params), max_requests(params.max_request_num), wr(params) {}

spbg_scheduler::spbg_scheduler(const gpt_params& params, const serve_policy& policy)
    : il_scheduler(params, policy), max_requests(params.max_request_num), wr(params) {}

spbg_scheduler::~spbg_scheduler() {}