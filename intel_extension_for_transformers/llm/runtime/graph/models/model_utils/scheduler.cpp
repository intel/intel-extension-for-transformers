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

// il_worker
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

std::vector<int> il_worker::get_request_done_ids() const { return request_done_ids; }

void il_worker::empty_request_done_ids() { request_done_ids.clear(); }

// cbg_worker
cbg_worker::cbg_worker(const gpt_params& params) : il_worker(params) { m_ctx->cont_batching = true; }
cbg_worker::~cbg_worker() {}

bool cbg_worker::prepare_inputs(std::vector<sequence*>* seqs, const int& n_input, model_input* inputs) {
  for (int ni = 0; ni < n_input; ++ni) {
    if ((seqs->at(ni))->status != seq_status::PREFILL || (seqs->at(ni))->status != seq_status::DECODING) {
      fprintf(stderr, "%s: error: request status is unright.\n", __func__);
      return false;
    } else if ((seqs->at(ni))->status == seq_status::PREFILL) {
      (inputs + ni)->tokens = (seqs->at(ni))->prompt_ids.data();
      (inputs + ni)->n_tokens = (seqs->at(ni))->n_prompt_tokens;
      (inputs + ni)->n_past = 0;
      (inputs + ni)->n_total = 0;
      (inputs + ni)->request_idx = (seqs->at(ni))->request_idx;
      // do not support padding for now
      (inputs + ni)->n_padding = 0;
    } else if ((seqs->at(ni))->status == seq_status::DECODING) {
      (inputs + ni)->tokens = &(seqs->at(ni))->generated_ids.back();
      (inputs + ni)->n_tokens = 1;
      (inputs + ni)->n_past = (seqs->at(ni))->n_past;
      (inputs + ni)->n_total = (seqs->at(ni))->n_total;
      (inputs + ni)->request_idx = (seqs->at(ni))->request_idx;
      // do not support padding for now
      (inputs + ni)->n_padding = 0;
    } else {
      continue;
    }
  }
  return true;
}

bool cbg_worker::beam_search_step(std::vector<sequence*>* seqs, const int& n_input) {
  std::vector<model_input> step_inputs(n_input);
  if (!prepare_inputs(seqs, n_input, step_inputs.data())) {
    return false;
  }
  // step beam search decoding
  if (!bsf->step(step_inputs)) {
    return false;
  }
  return true;
}

bool cbg_worker::step(std::vector<sequence*>* seqs, const int& n_input) {
  reqidx_to_vecid.clear();
  for (int ni = 0; ni < n_input; ++ni) {
    reqidx_to_vecid.emplace(std::make_pair(seqs->at(ni)->request_idx, ni));
  }
  if (m_ctx->beam_search && bsf != nullptr) {
    if (!beam_search_step(seqs, n_input)) {
      return false;
    }
  }
  // TODO (YZT) greedy search and top_p-top_k sampling
  return update_seqs(seqs, n_input);
}

bool cbg_worker::update_seqs(std::vector<sequence*>* seqs, const int& n_input) {
  empty_request_done_ids();
  for (int ni = 0; ni < n_input; ++ni) {
    if (seqs->at(ni)->status == seq_status::PREFILL) {
      seqs->at(ni)->status = seq_status::DECODING;
      seqs->at(ni)->n_past = seqs->at(ni)->n_prompt_tokens;
    }
  }
  if (m_ctx->beam_search && bsf != nullptr) {
    request_done_ids = bsf->request_done_ids();
    std::vector<std::vector<model_token>> req_done_res = bsf->request_done_reponse();
    if (request_done_ids.size() != req_done_res.size()) {
      fprintf(stderr,
              "%s: error: beam search give mis-matched size between finished request ids and generated "
              "tokens.\n",
              __func__);
      return false;
    }
    for (int r = 0; r < request_done_ids.size(); ++r) {
      const int idx = request_done_ids[r];
      if (reqidx_to_vecid.count(idx) == 0) {
        fprintf(stderr, "%s: error: done request idx: %d not in executed_seqs.\n", __func__, idx);
        return false;
      }
      seqs->at(reqidx_to_vecid[idx])->generated_ids = std::move(req_done_res[r]);
      seqs->at(reqidx_to_vecid[idx])->status = seq_status::FINISHED;
    }
    return true;
  }
  return false;  // TODO (YZT) greedy search and top_p-top_k sampling
}

// il_scheduler
il_scheduler::il_scheduler(const gpt_params& params, const serve_policy& policy)
    : params(params),
      policy(policy),
      waiting_pool(pool_property::WAITING),
      running_pool(pool_property::RUNNING),
      finished_pool(pool_property::FINISHED) {}

il_scheduler::il_scheduler(const gpt_params& params) : il_scheduler(params, serve_policy::FCFS) {}

il_scheduler::~il_scheduler() {}

bool il_scheduler::has_finished_seq() { return (finished_pool.size() > 0); }

std::vector<sequence*> il_scheduler::pop_completed_requests() {
  std::vector<sequence*> ret_seqs;
  const int length = finished_pool.size();
  if (length > 0) {
    return ret_seqs;
  }
  ret_seqs.resize(length);
  for (int l = 0; l < length; ++l) {
    if (!finished_pool.pop(ret_seqs[l])) {
      fprintf(stderr, "%s: error: pop finished_pool %dth seq failed.\n", __func__, l);
      return std::vector<sequence*>();
    }
  }
  return ret_seqs;
}

// cbg_scheduler
cbg_scheduler::cbg_scheduler(const gpt_params& params)
    : il_scheduler(params),
      max_requests(params.max_request_num),
      wr(params),
      free_req_idx(max_requests, true),
      max_input_length(max_requests * params.n_ctx) {}

cbg_scheduler::cbg_scheduler(const gpt_params& params, const serve_policy& policy)
    : il_scheduler(params, policy),
      max_requests(params.max_request_num),
      wr(params),
      free_req_idx(max_requests, true),
      max_input_length(max_requests * params.n_ctx) {}

cbg_scheduler::~cbg_scheduler() {}

int cbg_scheduler::query_free_req_idx() {
  auto iter = std::find_if(free_req_idx.begin(), free_req_idx.end(), [](const bool flag) { return flag; });
  if (iter == free_req_idx.end()) {
    return -1;
  } else {
    return std::distance(free_req_idx.begin(), iter) - 1;
  }
}

bool cbg_scheduler::add_request(sequence* seq) {
  if (seq->status != seq_status::UNKNOWN) {
    fprintf(stderr, "%s: error: seq status is not UNKNOWN, can not decide to add into which pool.\n", __func__);
    return false;
  }
  // add into waiting_pool by default
  seq->status = seq_status::WAITING;
  seq->request_idx = query_free_req_idx();
  return waiting_pool.add(seq);
}

bool cbg_scheduler::prepare_seqs() {
  executed_seqs.clear();
  cur_running_num = running_pool.size();
  if (cur_running_num > max_requests) {
    fprintf(stderr, "%s: error: cur_running_num is larger than max_request_num.\n", __func__);
    return false;
  }
  const int n_perfill_seqs = std::min(max_requests - cur_running_num, waiting_pool.size());
  executed_seqs.resize(n_perfill_seqs + cur_running_num);
  if (waiting_pool.size() > 0) {
    // pop prompts
    if (cur_running_num < max_requests) {
      for (int np = 0; np < n_perfill_seqs; ++np) {
        if (waiting_pool.pop(executed_seqs[cur_running_num + np])) {
          executed_seqs[cur_running_num + np]->status == seq_status::PREFILL;
          if (executed_seqs[cur_running_num + np]->request_idx == -1) {
            const int fidx = query_free_req_idx();
            if (fidx == -1) {
              fprintf(stderr, "%s: error: no free position to put the request.\n", __func__);
              return false;
            }
            executed_seqs[cur_running_num + np]->request_idx = fidx;
          }
        } else {
          fprintf(stderr, "%s: error: pop waiting seq failed.\n", __func__);
          return false;
        }
      }
    } else {
      // steps generation
      steps_decoding_for_next_prefill = true;
    }
  }
  // step generation
  for (int dn = 0; dn < cur_running_num; ++dn) {
    if (!running_pool.pop(executed_seqs[dn]) || executed_seqs[dn]->status != seq_status::DECODING) {
      fprintf(stderr, "%s: error: pop running_pool %dth seq failed.\n", __func__, dn);
      return false;
    }
  }
  cur_running_num = executed_seqs.size();
  return true;
}

bool cbg_scheduler::step() {
  if (done()) {
    fprintf(stderr,
            "%s: warning: scheduler has no more requests, please add extra requests or just stop "
            "calling.\n",
            __func__);
    return true;
  }
  if (!prepare_seqs()) {
    return false;
  }
  // one step
  if (!steps_decoding_for_next_prefill) {
    if (!wr.step(&executed_seqs, executed_seqs.size())) {
      return false;
    }
  } else {
    // steps for next prompt prefilling
    fprintf(stdout, "%s: info: running_pool size = max request num, will execute several steps.\n", __func__);
    wr.empty_request_done_ids();
    while (wr.get_request_done_ids().empty()) {
      if (!wr.step(&executed_seqs, executed_seqs.size())) {
        return false;
      }
    }
    steps_decoding_for_next_prefill = false;
  }
  return update_pools();
}

bool cbg_scheduler::update_pools() {
  for (int ns = 0; ns < executed_seqs.size(); ++ns) {
    if (executed_seqs[ns]->status == seq_status::DECODING) {
      running_pool.add(executed_seqs[ns]);
    } else if (executed_seqs[ns]->status == seq_status::FINISHED) {
      finished_pool.add(executed_seqs[ns]);
      free_req_idx[executed_seqs[ns]->request_idx] = true;
    } else {
      fprintf(stderr, "%s: error: wrong seq status, seq_idx: %d should be in DECODING OR FINISHED.\n", __func__);
      return false;
    }
  }
  return true;
}

bool cbg_scheduler::done() {
  if (waiting_pool.empty() && running_pool.empty()) {
    return true;
  } else {
    return false;
  }
}
