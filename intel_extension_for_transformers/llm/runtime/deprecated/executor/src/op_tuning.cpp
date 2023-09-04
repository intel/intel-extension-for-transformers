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

#include "op_tuning.hpp"
#include "model.hpp"

namespace executor {

// call related tune function by type and kernel_name
// prepare tuning config, like extra tensor life
void OpTuning::Start(const string& kernel_name, std::shared_ptr<Operator> kernel,
                     const vector<Tensor*>& input, const vector<Tensor*>& output,
                     const bool& reshape_model) {
  set_tune_func_name(kernel_name);
  stage_ = "start";
  (this->*(tune_func_map_[tune_func_name_]))(kernel, input, output, reshape_model);
  stage_.clear();
}

// tune kernel and record time and dispatch kernel config
void OpTuning::Run(const string& kernel_name, std::shared_ptr<Operator> kernel,
                   const vector<Tensor*>& input, const vector<Tensor*>& output,
                   const bool& reshape_model) {
  kernel_config_.clear();
  kernel_config_.push_back(kernel_name);
  set_tune_func_name(kernel_name);
  (this->*(tune_func_map_[tune_func_name_]))(kernel, input, output, reshape_model);
}

void OpTuning::BaseTune(std::shared_ptr<Operator> kernel, const vector<Tensor*>& input,
                        const vector<Tensor*>& output, const bool& reshape_model) {
  if (stage_ == "start") {
    extra_tensor_life_ += 1;
    return;
  }
  int64_t start_time = 0;
  int64_t end_time = 0;
  // consider kernel forward class creation time in Reshape
  // it may be time-consuming under dynamic shape
  float reshape_time = 0;
  if (reshape_model) {
    start_time = Time();
    kernel->Reshape(input, output);
    end_time = Time();
    reshape_time = Duration(start_time, end_time);
  }
  start_time = Time();
  kernel->Forward(input, output);
  end_time = Time();
  float forward_time = Duration(start_time, end_time);
  best_execute_time_ = forward_time + reshape_time;
  DLOG(INFO) << "BaseTune forward time is " << best_execute_time_ << "ms";
}

// MxK x KxN to
// Nx1xWxK(c_i) x 1x1xK(c_i)xN(c_o)
// find the best N-W combination
void OpTuning::IpToConvTune(std::shared_ptr<Operator> kernel, const vector<Tensor*>& input,
                            const vector<Tensor*>& output, const bool& reshape_model) {
  // only for tuning fp32 and bf16 dtype
  if (input[1]->dtype() != "fp32"  && input[1]->dtype() != "bf16") {
    LOG(WARNING) << "Only support fp32 or bf16 dtype when tuning kernel between InnerProduct and Convolution!";
    best_execute_time_ = std::numeric_limits<float>::max();
    kernel_config_.clear();
    return;
  }
  std::map<float, string, std::less<float>> input_shape_timer;
  vector<string> nw_comb;
  bool is_src0_transposed = input[0]->is_transposed();
  vector<int64_t> src0_shape = input[0]->shape();
  // prepare period does not change src0 shape
  int64_t m_dim;
  int64_t k_dim;
  if (!is_src0_transposed) {
    m_dim = src0_shape[0];
    k_dim = src0_shape[1];
  } else {
    m_dim = src0_shape[1];
    k_dim = src0_shape[0];
  }
  for (int64_t i = 1; i <= m_dim; ++i) {
    if (m_dim % i == 0) {
    if (!is_src0_transposed) {
      // [N, H, W, C], C=K, N*H*W=M
      nw_comb.push_back(std::to_string(i) + ",1," + std::to_string(m_dim / i) + ","
                        + std::to_string(k_dim));
    } else {
      // [C, N, H, W], C=K, N*H*W=M
      nw_comb.push_back(std::to_string(k_dim) + "," + std::to_string(i) + ",1,"
                        + std::to_string(m_dim / i));
    }
    }
  }
  // add tensor life
  if (stage_ == "start") {
    extra_tensor_life_ += nw_comb.size();
    return;
  }
  vector<string> kernel_config_cpy = {kernel_config_[0], ""};
  for (const auto& comb : nw_comb) {
    kernel_config_cpy[1] = comb;
    kernel->set_dispatch_config(kernel_config_cpy);
    int64_t start_time = 0;
    int64_t end_time = 0;
    float reshape_time = 0;
    start_time = Time();
    kernel->Reshape(input, output);
    end_time = Time();
    reshape_time = Duration(start_time, end_time);
    start_time = Time();
    kernel->Forward(input, output);
    end_time = Time();
    float execute_time = Duration(start_time, end_time);
    if (reshape_model) execute_time += reshape_time;
    input_shape_timer[execute_time] = comb;
    DLOG(INFO) << "IpToConvTune forward time is " << execute_time << "ms with src0 shape " << comb;
  }
  if (input_shape_timer.size() > 0) {
    best_execute_time_ = input_shape_timer.begin()->first;
    kernel_config_.push_back(input_shape_timer.begin()->second);
  } else {
    LOG(FATAL) << "InnerProduct tuning fails with kernel convolution...";
  }
}

// split the dimension from 2D to 3D when use sparselib gemm
void OpTuning::IpToSparseLibTune(std::shared_ptr<Operator> kernel, const vector<Tensor*>& input,
                                 const vector<Tensor*>& output, const bool& reshape_model) {
  // only for tuning int8 dtype
  if (input[1]->dtype() != "u8") {
    LOG(WARNING) << "Only support int8 dtype when tuning InnerProduct kernel with SparseLib!";
    best_execute_time_ = std::numeric_limits<float>::max();
    kernel_config_.clear();
    return;
  }
  // sparselib search space
  vector<int64_t> bs_space = {64, 128, 192, 256, 384, 512};
  // micro_oc is positive integer fulfilling micro_oc <= OC && micro_oc % 4 == 0 determined the
  // size along the output dimension is processed in each OMP iteration. If it is set to 0, the
  // kernel will choose a value so that number of OMP iterations is equal to omp_get_max_threads().
  // That is : micro_oc = ceil(OC / (num_threads / num_microbs)) && (micro_oc % 4) == 0
  vector<string> micro_oc_space = {"0", "64", "128", "256"};
  // higher sub_func value means more operations are done in sub-function (i.e. less unrolling).
  vector<string> sub_func_space = {"0", "1", "2"};
  // sparselib dispatch kernel config is {"input_shape", "micro_oc", "sub_func"}
  std::map<float, vector<string>, std::less<float>> bs_attr_timer;
  // M x k -> mic_bs x K x bs
  vector<string> micbs_bs_comb;
  // sparselib graph ir should switch position of src and weight
  vector<int64_t> src1_shape = input[1]->shape();
  int64_t m_dim = src1_shape[1];
  int64_t k_dim = src1_shape[0];
  int64_t model_input_bs = model_->input_shape()[0];
  bool oneKM_shape_filling = false;
  for (const auto& bs : bs_space) {
    if (bs == 0) continue;
    if (m_dim % bs > 0 && !oneKM_shape_filling) {
    micbs_bs_comb.push_back("1," + std::to_string(k_dim) + "," + std::to_string(m_dim));
    oneKM_shape_filling = true;
    }
    if (m_dim < bs) break;
    if (m_dim % bs == 0) {
    if ((m_dim / bs) == 1 && oneKM_shape_filling) continue;
    if ((m_dim / bs) > model_input_bs || model_input_bs % (m_dim / bs) > 0) continue;
    micbs_bs_comb.push_back(std::to_string(m_dim / bs) + "," + std::to_string(k_dim) + "," + std::to_string(bs));
    if ((m_dim / bs) == 1) oneKM_shape_filling = true;
    }
  }
  if (micbs_bs_comb.empty() || !oneKM_shape_filling) {
    micbs_bs_comb.push_back("1," + std::to_string(k_dim) + "," + std::to_string(m_dim));
  }
  // select the micro_os which is <= oc
  vector<string> micro_oc_space_filtered;
  for (const auto& mc : micro_oc_space) {
    int64_t mc_int = std::atoi(mc.c_str());
    if (mc_int > m_dim) break;
    micro_oc_space_filtered.push_back(mc);
  }
  vector<vector<string>> bs_attr_comb(micbs_bs_comb.size() * micro_oc_space_filtered.size() * sub_func_space.size());
#pragma omp parallel for
  for (int i = 0; i < micbs_bs_comb.size(); ++i) {
    for (int j = 0; j < micro_oc_space_filtered.size(); ++j) {
#pragma omp simd
      for (int k = 0; k < sub_func_space.size(); ++k) {
        bs_attr_comb[i * micro_oc_space_filtered.size() * sub_func_space.size() + j * sub_func_space.size() + k ] = \
        {micbs_bs_comb[i], micro_oc_space_filtered[j], sub_func_space[k]};
      }
    }
  }
  // add tensor life
  if (stage_ == "start") {
    extra_tensor_life_ += bs_attr_comb.size();
    return;
  }
  vector<string> kernel_config_cpy = {kernel_config_[0], "", "", ""};
  for (const auto& comb : bs_attr_comb) {
    for (int i = 0; i < comb.size(); ++i) kernel_config_cpy[i + 1] = comb[i];
    kernel->set_dispatch_config(kernel_config_cpy);
    int64_t start_time = 0;
    int64_t end_time = 0;
    float reshape_time = 0;
    start_time = Time();
    kernel->Reshape(input, output);
    end_time = Time();
    reshape_time = Duration(start_time, end_time);
    kernel->AdaptTensors(input, output, "in");
    start_time = Time();
    kernel->Forward(input, output);
    end_time = Time();
    kernel->AdaptTensors(input, output, "out");
    float execute_time = Duration(start_time, end_time);
    if (reshape_model) execute_time += reshape_time;
    bs_attr_timer[execute_time] = kernel_config_cpy;
    DLOG(INFO) << "IpToSparseLibTune forward time is " << execute_time << "ms, activation shape: " << comb[0]
            << ", micro_oc: " << comb[1] << ", sub_func: " << comb[2];
  }
  if (bs_attr_timer.size() > 0) {
    best_execute_time_ = bs_attr_timer.begin()->first;
    kernel_config_ = bs_attr_timer.begin()->second;
  } else {
    LOG(FATAL) << "InnerProduct tuning fails with kernel SparseLib...";
  }
}

void OpTuning::set_tune_func_name(const string& kernel_name) {
  tune_func_name_ = type_ + "_to_" + kernel_name;
  if (tune_func_map_.count(tune_func_name_) == 0) {
    if (kernel_name != type_) LOG(WARNING) << "No matching tuning function for " << tune_func_name_
                                           << ", gonna use BaseTune function instead...";
    tune_func_name_ = "Base";
  }
}

std::unordered_map<string, OpTuning::TuneFunc> OpTuning::tune_func_map_ = {
    {"Base", &OpTuning::BaseTune},
    {"InnerProduct_to_Convolution", &OpTuning::IpToConvTune},
    {"InnerProduct_to_SparseLib", &OpTuning::IpToSparseLibTune}
};

}  // namespace executor
