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
#include "bert_qsl.hpp"
#include <omp.h>
#include <iostream>
#include <mutex>  // NOLINT
#include <cassert>
#include <utility>
#include "common.hpp"

namespace qsl {
  std::vector<executor::Tensor> loadTensorListFromFile(std::string file_name) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
      std::cout << "Open file failure" << std::endl;
      return std::vector<executor::Tensor>{};
    }
    int32_t lines;
    file >> lines;
    std::vector<executor::Tensor> result;
    for (int i = 0; i < lines; i++) {
      int size;
      file >> size;
      int32_t* data = static_cast<int32_t*>(calloc(size, sizeof(int32_t)));
      int32_t* data_begin = data;
      for (int j = 0; j < size; j++, data++) file >> *data;
      result.emplace_back(data_begin, vector<int64_t>{size}, "int32");
    }
    return result;
  }

  SquadQuerySampleLibrary::SquadQuerySampleLibrary(const std::string& root) {
    input_ids_set_ = loadTensorListFromFile(root + "input_ids.txt");
    input_mask_set_ = loadTensorListFromFile(root + "input_mask.txt");
    segment_ids_set_ = loadTensorListFromFile(root + "segment_ids.txt");
  }

  // Parallel bucket sort (unstable) would be the most efficient choice
  // For length 40 ~ 384, each with a bucket of std::list
  //
  Queue_t SquadQuerySampleLibrary::Sort(const std::vector<mlperf::QuerySample>& samples,
    bool reverse, size_t minLength, size_t maxLength) const {
    const auto lengthOffset = minLength;
    const auto nBucket = maxLength - lengthOffset + 1;

    std::vector<Queue_t> Buckets(nBucket);
    std::vector<std::mutex> lks(nBucket);

    // (Parallel) sort
    // (TODO) support other parallel library
#pragma omp parallel for
    for (int i = 0; i < samples.size(); i++) {
      const auto& sample = samples[i];
      auto length = GetFeatureLength(sample.index);

      auto idx = reverse ? maxLength - length : length - lengthOffset;
      auto& bucket = Buckets[idx];
      auto& l = lks[idx];

      {
        std::unique_lock<std::mutex> guard(l);
        bucket.emplace_back(sample);
      }
    }

    // Splice them togather
    Queue_t result;
    for (auto& q : Buckets)
      result.splice(result.end(), std::move(q));

    return result;
  }

  //
  // Assemble samples into larger batch
  //
  void SquadQuerySampleLibrary::AssembleSamples(
    const std::vector<mlperf::QuerySampleIndex>& indices, std::vector<executor::Tensor>* tensors) {
    tensors->clear();
    // as we have already sort the samples,
    // from the large seq length to small so we use the first seq
    int64_t maxLength = input_ids_set_[indices[0]].shape()[0];
    if (!minilm) {
      if (maxLength > 192) {
        maxLength = 384;
      } else if (maxLength <= 192 && maxLength > 160){
        maxLength = 192;
      } else if (maxLength <= 160 && maxLength > 128){
        maxLength = 160;
      } else {
        maxLength = 128;
      }
    } else {
      int64_t align = 32;
      int64_t offset = maxLength % align == 0 ? 0 : 1;
      maxLength = (maxLength / align + offset) * align;
    }
    vector<int64_t> newShape{ static_cast<int64_t>(indices.size()), maxLength };
    size_t stride = executor::Product(newShape);
    int32_t* input_ids_data = static_cast<int32_t*>(calloc(stride, sizeof(int32_t)));
    int32_t* input_mask_data = static_cast<int32_t*>(calloc(stride, sizeof(int32_t)));
    int32_t* segment_ids_data = static_cast<int32_t*>(calloc(stride, sizeof(int32_t)));
 #pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i) {
      auto index = indices[i];
      auto input_ids = input_ids_set_[index];
      auto input_mask = input_mask_set_[index];
      auto segment_ids = segment_ids_set_[index];
      // (TODO) overhead need to remove about memory copy
      memcpy(input_ids_data + i * maxLength, input_ids.data(),
        input_ids.size() * sizeof(int32_t));
      memcpy(input_mask_data + i * maxLength, input_mask.data(),
        input_mask.size() * sizeof(int32_t));
      memcpy(segment_ids_data + i * maxLength, segment_ids.data(),
        segment_ids.size() * sizeof(int32_t));
    }
    tensors->emplace_back(input_ids_data, newShape, "int32");
    tensors->emplace_back(segment_ids_data, newShape, "int32");
    tensors->emplace_back(input_mask_data, newShape, "int32");
  }

  void SquadQuerySampleLibrary::GetSample(mlperf::QuerySampleIndex index,
    std::vector<executor::Tensor>* tensors) const {
    tensors->clear();
    executor::Tensor input_id = input_ids_set_.at(index);
    vector<int64_t> newShape(input_id.shape());
    newShape.insert(newShape.begin(), 1);
    input_id.set_shape(newShape);

    executor::Tensor input_mask = input_mask_set_.at(index);
    input_mask.set_shape(newShape);

    executor::Tensor segment_id = segment_ids_set_.at(index);
    segment_id.set_shape(newShape);

    tensors->push_back(input_id);
    tensors->push_back(segment_id);
    tensors->push_back(input_mask);
  }

}  // namespace qsl
