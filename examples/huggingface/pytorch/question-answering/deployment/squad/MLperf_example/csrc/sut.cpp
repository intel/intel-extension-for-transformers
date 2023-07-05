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
#include "sut.hpp"
#include <loadgen.h>
#include <query_sample.h>
#include <vector>
#include <chrono>  // NOLINT
#include <algorithm>
#include <condition_variable>  // NOLINT
#include <type_traits>
#include <string>
#include "kmp_launcher.hpp"
#include "executor.hpp"
#include <future>

BertSUT::BertSUT(
  const std::string& conf_file,
  const std::string& weight_root,
  const std::string& dataset_root,
  const std::string& scenario,
  int instance,
  int cores_per_instance,
  int batch, bool ht, bool minilm
) : mThreshold_(batch), mHt_(ht), qsl_(dataset_root), weight_(weight_root), conf_(conf_file),
nProcsPerInstance_(cores_per_instance), nInstances_(instance), scenario_(scenario) {
  auto nMaxProc = kmp::KMPLauncher::getMaxProc();
  if (nProcsPerInstance_ * nInstances_ > nMaxProc)
    nInstances_ = nMaxProc / nProcsPerInstance_;
  qsl_.minilm = minilm;
  // Construct instances
  for (int i = 0; i < nInstances_; ++i)
    mInstances_.emplace_back(&BertSUT::thInstance, this, i);
}

int BertSUT::rootProc(int index) {
  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  // XXX : Assumed 2-sockets, HT on !!!
  int part[] = {nMaxProc, nMaxProc*(2 + (int)mHt_)/4};

  auto select = index & 1;
  auto root = part[select] - nProcsPerInstance_ * ((index>>1) + 1);

  // Assert root > 0
  return root;
}

void BertSUT::thInstance(int index) {
  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nProcsPerInstance_);
  auto root = rootProc(index);
  for (int i = 0; i < nProcsPerInstance_; ++ i) {
    places[i] = (root + i);
  }

  thCtrl.setAffinityPlaces(places).pinThreads();
  executor::Model bertmodel(conf_, weight_);
  // warm up 
{
  vector<executor::Tensor> wm_tensors;
  vector<string> input_dtype;
  vector<vector<float>> input_range;
  vector<vector<int64_t>> input_shape;
  auto input_configs = bertmodel.input_configs();
  int bs = 4;
  int seq_len = 384;

  for (int i = 0; i < bertmodel.num_inputs(); ++i) {
    wm_tensors.push_back(executor::Tensor(*(input_configs[i])));
    input_dtype.push_back(wm_tensors[i].dtype());
    input_range.push_back(vector<float>({0, 2}));
    input_shape.push_back(wm_tensors[i].shape());
    if (input_shape[i][0] == -1 && input_shape[i][1] == -1) {
      input_shape[i][0] = bs;
      input_shape[i][1] = 128;
    } else if (input_shape[i][0] == -1) {
      input_shape[i][0] = seq_len;
    }
  }
  executor::DataLoader* dataloader;
  dataloader = new executor::DummyDataLoader(input_shape, input_dtype, input_range);
  auto raw_data = dataloader->prepare_batch(0);
  for (int j = 0; j < wm_tensors.size(); ++j) {
    wm_tensors[j].set_data(raw_data[j]);
    wm_tensors[j].set_shape(input_shape[j]);
  }
  auto warm_output = bertmodel.Forward(wm_tensors);

  wm_tensors.clear();
  input_shape.clear();
  input_dtype.clear();

  for (int i = 0; i < bertmodel.num_inputs(); ++i) {
    wm_tensors.push_back(executor::Tensor(*(input_configs[i])));
    input_dtype.push_back(wm_tensors[i].dtype());
    input_range.push_back(vector<float>({0, 2}));
    input_shape.push_back(wm_tensors[i].shape());
    if (input_shape[i][0] == -1 && input_shape[i][1] == -1) {
      input_shape[i][0] = bs;
      input_shape[i][1] = 192;
    } else {
      input_shape[i][0] = bs;
      input_shape[i][1] = 192;
    }
  }
  dataloader = new executor::DummyDataLoader(input_shape, input_dtype, input_range);
  raw_data = dataloader->prepare_batch(0);
  for (int j = 0; j < wm_tensors.size(); ++j) {
    wm_tensors[j].set_data(raw_data[j]);
    wm_tensors[j].set_shape(input_shape[j]);
  }
  warm_output = bertmodel.Forward(wm_tensors);
  input_shape.clear();
  wm_tensors.clear();
  input_dtype.clear();

  for (int i = 0; i < bertmodel.num_inputs(); ++i) {
    wm_tensors.push_back(executor::Tensor(*(input_configs[i])));
    input_dtype.push_back(wm_tensors[i].dtype());
    input_range.push_back(vector<float>({0, 2}));
    input_shape.push_back(wm_tensors[i].shape());
    if (input_shape[i][0] == -1 && input_shape[i][1] == -1) {
      input_shape[i][0] = bs;
      input_shape[i][1] = 160;
    } else {
      input_shape[i][0] = bs;
      input_shape[i][1] = 160;
    }
  }
  dataloader = new executor::DummyDataLoader(input_shape, input_dtype, input_range);
  raw_data = dataloader->prepare_batch(0);
  for (int j = 0; j < wm_tensors.size(); ++j) {
    wm_tensors[j].set_data(raw_data[j]);
    wm_tensors[j].set_shape(input_shape[j]);
  }
  warm_output = bertmodel.Forward(wm_tensors);
  input_shape.clear();
  wm_tensors.clear();
  input_dtype.clear();

  for (int i = 0; i < bertmodel.num_inputs(); ++i) {
    wm_tensors.push_back(executor::Tensor(*(input_configs[i])));
    input_dtype.push_back(wm_tensors[i].dtype());
    input_range.push_back(vector<float>({0, 2}));
    input_shape.push_back(wm_tensors[i].shape());
    if (input_shape[i][0] == -1 && input_shape[i][1] == -1) {
      input_shape[i][0] = bs;
      input_shape[i][1] = 384;
    } else {
      input_shape[i][0] = bs;
      input_shape[i][1] = 384;
    }
  }
  dataloader = new executor::DummyDataLoader(input_shape, input_dtype, input_range);
  raw_data = dataloader->prepare_batch(0);
  for (int j = 0; j < wm_tensors.size(); ++j) {
    wm_tensors[j].set_data(raw_data[j]);
    wm_tensors[j].set_shape(input_shape[j]);
  }
  warm_output = bertmodel.Forward(wm_tensors);
}


  Queue_t snippet;

  // Wait for work
  while (true) {
    {
      std::unique_lock<std::mutex> l(mtx_);
      ctrl_.wait(l, [this] {return mStop_ || !mQueue_.empty();});

      if (mStop_)
        break;

      auto nPeek = std::min(mQueue_.size(), mThreshold_);
      auto it = mQueue_.begin();
      // XXX: pointer chaser, choose better solution
      std::advance(it, nPeek);
      snippet.clear();
      snippet.splice(snippet.begin(), mQueue_, mQueue_.begin(), it);
      ctrl_.notify_one();
    }

    std::vector<mlperf::QuerySample> samples(snippet.begin(), snippet.end());

    // Model Inference, switch single/multiple batch model
    // According to length
    std::vector<executor::Tensor> input_tensors;
    if (scenario_ == "Offline") {
      std::vector<mlperf::QuerySampleIndex> indices(samples.size());
      std::transform(samples.cbegin(), samples.cend(), indices.begin(),
        [](mlperf::QuerySample sample) {return sample.index;});
      qsl_.AssembleSamples(std::move(indices), &input_tensors);
      auto output = bertmodel.Forward(input_tensors);
      QuerySamplesComplete(samples, &output[0]);
    } else {
      for (auto sample : samples) {
        qsl_.GetSample(sample.index, &input_tensors);
        auto output = bertmodel.Forward(input_tensors);
        QuerySamplesComplete(sample, &output[0]);
      }
    }
  }
}

void BertSUT::QuerySamplesComplete(
  const std::vector<mlperf::QuerySample>& samples,
  executor::Tensor* results) {
  auto shape = results->shape();
  shape.erase(shape.begin());
  int64_t stride = executor::Product(shape);
  // assumption output dtype is float
  float* output_data = static_cast<float*>(results->mutable_data());
  string dtype = results->dtype();

  std::vector<mlperf::QuerySampleResponse> responses(samples.size());

#pragma omp parallel for
  for (size_t i = 0; i < samples.size(); ++i) {
    responses[i].id = samples[i].id;
    responses[i].data = reinterpret_cast<uintptr_t>(output_data + i * stride);
    responses[i].size = stride * sizeof(float);
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
}



void BertSUT::QuerySamplesComplete(
  const mlperf::QuerySample& sample,
  executor::Tensor* result) {
  mlperf::QuerySampleResponse response;

  response.id = sample.id;
  response.data = reinterpret_cast<uintptr_t>(result->mutable_data());
  response.size = result->size() * executor::type2bytes[result->dtype()];

  mlperf::QuerySamplesComplete(&response, 1);
}

void BertSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  std::unique_lock<std::mutex> l(mtx_);
  if (scenario_ == "Offline") {
    mQueue_ = qsl_.Sort(samples);
  } else {
    for (auto sample : samples) mQueue_.emplace_back(sample);
  }

  ctrl_.notify_one();
}

BertSUT::~BertSUT() {
  {
    std::unique_lock<std::mutex> l(mtx_);
    mStop_ = true;
    ctrl_.notify_all();
  }

  for (auto& Instance : mInstances_)
    Instance.join();
}
