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

#include <query_sample_library.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <list>
#include "tensor.hpp"

namespace qsl {
using Queue_t = std::list<mlperf::QuerySample>;

class SquadQuerySampleLibrary : public mlperf::QuerySampleLibrary {
 public:
  explicit SquadQuerySampleLibrary(const std::string& root);

  virtual ~SquadQuerySampleLibrary() = default;

  const std::string& Name() override {
    static const std::string name("BERT Squad QSL");
    return name;
  }

  size_t TotalSampleCount() override {
    return input_ids_set_.size();
  }

  size_t PerformanceSampleCount() override {
    return TotalSampleCount();
  }

  //
  // SQuAD is small enough to be in Memory
  //
  void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {}
  void UnloadSamplesFromRam(
    const std::vector<mlperf::QuerySampleIndex>& samples) override {
  }

  // static SquadQuerySampleLibrary Create(const std::string& root);

  void GetSample(mlperf::QuerySampleIndex index, std::vector<executor::Tensor>* tensors) const;

  void AssembleSamples(const std::vector<mlperf::QuerySampleIndex>& indices,
    std::vector<executor::Tensor>* tensors);

  // List of tensor of 1d
  size_t GetFeatureLength(size_t index) const {
    return input_ids_set_[index].shape()[0];
  }

  // Sort SQuAD data for batch behavior

  Queue_t Sort(
    const std::vector<mlperf::QuerySample>& samples, bool reverse = true,
    size_t minLength = 40, size_t maxLength = 384) const;
  bool minilm = false;
 private:
  std::vector<executor::Tensor> input_ids_set_;
  std::vector<executor::Tensor> input_mask_set_;
  std::vector<executor::Tensor> segment_ids_set_;
};

}  // namespace qsl
