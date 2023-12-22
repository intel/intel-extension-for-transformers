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

enum class seq_status : int {
  UNKNOWN = 0,
  WAITING,
  PREFILL,
  DECODING,
  FINISHED,
};

enum class pool_property : int {
  WAITING = 0,
  RUNNING,
  FINISHED,
};

enum class serve_policy : int {
  FCFS = 0,  // first come first serve
};

struct sequence {
  int request_idx = -1;  // -1 means unknown
  int64_t receive_time;
  int64_t end_time;
  std::vector<model_token> prompt_ids;
  std::vector<model_token> generated_ids;
  uint32_t n_prompt_tokens;
  uint32_t n_past;
  uint32_t n_total;
  uint32_t n_tokens;
  generation_config gen_conf;
  seq_status status = seq_status::UNKNOWN;
};

// abstract class
class pool {
 public:
  explicit pool(const pool_property& property) : property(property) {}
  virtual ~pool() {}
  virtual bool add(sequence* seq) = 0;
  virtual bool pop(sequence* seq) = 0;
  virtual void clear() = 0;
  virtual bool empty() = 0;
  virtual int size() = 0;

 protected:
  const pool_property property;
};

class fcfs_pool : public pool {
 public:
  explicit fcfs_pool(const pool_property& property) : pool(property) {}
  ~fcfs_pool() {}
  bool add(sequence* seq) override;
  bool pop(sequence* seq) override;
  void clear() override;
  bool empty() override;
  int size() override;

 protected:
  std::queue<sequence*> context;
};

class serve_pool {
 public:
  explicit serve_pool(const pool_property& property);
  serve_pool(const serve_policy& policy, const pool_property& property);
  ~serve_pool();
  bool add(sequence* seq);
  bool pop(sequence* seq);
  void clear();
  bool empty();
  int size();

 protected:
  pool* internel_pool = nullptr;
  std::mutex mtx;
};

#endif  // POOL_H
