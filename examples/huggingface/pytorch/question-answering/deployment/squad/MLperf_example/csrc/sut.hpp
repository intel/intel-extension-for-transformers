//  Copyright (c) 2021 Intel Corporation
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
#pragma once

#include <query_sample.h>
#include <system_under_test.h>
#include <query_sample_library.h>
#include <cassert>
#include <condition_variable>  // NOLINT
#include <deque>
#include <iostream>
#include <map>
#include <mutex>  // NOLINT
#include <thread>  // NOLINT
#include <vector>
#include <atomic>
#include <list>
#include <string>
#include "bert_qsl.hpp"
#include "model.hpp"

class BertSUT : public mlperf::SystemUnderTest {
  using Queue_t = std::list<mlperf::QuerySample>;
  // using Queue_t = std::forward_list<mlperf::QuerySample>
  // using Queue_t = std::deque<mlperf::QuerySample>;
 public:
  // configure inter parallel and intra paralel
  // 4x10 core required for expected performance
  BertSUT(
    const std::string& conf,
    const std::string& weight,
    const std::string& dataset,
    const std::string& scenario,
    int instance,
    int cores_per_instance,
    int batch, bool ht = true, bool minilm = false);

  ~BertSUT();

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;

  void FlushQueries() override {}
  // void ReportLatencyResults(
  //   const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {
  // }

  const std::string& Name() override {
    static const std::string name("BERT");
    return name;
  }

  static void QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    executor::Tensor* results);

  static void QuerySamplesComplete(
    const mlperf::QuerySample& sample,
    executor::Tensor* result);

  mlperf::QuerySampleLibrary* GetQSL() {
    return &qsl_;
  }
void WarmUp(
  executor::Model bertmodel,
  int bs,
  int seq_len);

  
 private:
  qsl::SquadQuerySampleLibrary qsl_;
  std::string weight_;
  std::string conf_;
  std::string scenario_;

  std::condition_variable ctrl_;
  std::mutex mtx_;

  Queue_t mQueue_;
  bool mStop_{ false };

  // Control over max samples a instance will peek
  size_t mThreshold_;
  size_t slength_{ 384 };

  std::vector<std::thread> mInstances_;
  int nProcsPerInstance_;
  int nInstances_;
  bool mHt_;

 private:
  int rootProc(int index);
  void thInstance(int index);
};


