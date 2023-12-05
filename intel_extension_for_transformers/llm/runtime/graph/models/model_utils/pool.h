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

#ifndef POOL_H
#define POOL_H

#include <queue>

#include "models/model_utils/model_types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum seq_status {
  SEQ_UNKNOWN,
  SEQ_WAITING,
  SEQ_PREFILL,
  SEQ_DECODING,
  SEQ_FINISHED,
};

enum pool_property {
  POOL_WAITING,
  POOL_RUNNING,
};

enum serve_policy {
  SPY_FCFS,  // first come first serve
};

struct sequence {
  int request_id = -1;  // -1 means unknown
  int64_t receive_time;
  int64_t end_time;
  model_token* prompt_ids = NULL;
  uint32_t n_prompt_tokens;
  model_token* generated_ids = NULL;
  uint32_t n_generated_tokens;
  generation_config* gen_conf = NULL;
  seq_status status = SEQ_UNKNOWN;
};

#ifdef __cplusplus
}
#endif

// abstract class
class pool {
 public:
  explicit pool(const pool_property& property) : property(property) {}
  virtual ~pool() {}
  virtual const bool add(const sequence& seq) = 0;
  virtual const bool pop(sequence* seq) = 0;
  virtual void clear() = 0;
  virtual const bool empty() = 0;
  virtual const int size() = 0;

 protected:
  const pool_property property;
};

class fcfs_pool : public pool {
 public:
  explicit fcfs_pool(const pool_property& property) : pool(property) {}
  ~fcfs_pool() {}
  const bool add(const sequence& seq) override;
  const bool pop(sequence* seq) override;
  void clear() override;
  const bool empty() override;
  const int size() override;

 protected:
  std::queue<sequence> context;
};

class serve_pool {
 public:
  explicit serve_pool(const pool_property& property);
  serve_pool(const serve_policy& policy, const pool_property& property);
  ~serve_pool();
  const bool add(const sequence& seq);
  const bool pop(sequence* seq);
  void clear();
  const bool empty();
  const int size();

 protected:
  pool* internel_pool = nullptr;
  std::mutex mtx;
};

#endif  // POOL_H
