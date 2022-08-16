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

#ifndef ENGINE_EXECUTOR_INCLUDE_OP_TUNING_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OP_TUNING_HPP_

#include <map>
#include <functional>
#include <algorithm>
#include <limits>

#include "dispatch_table.hpp"
#include "common.hpp"

namespace executor {

/**
* @brief  get each kernel's best execute time and related dispatch config for dispatcher
*
*/

class OpTuning {
 public:
  typedef void(OpTuning::*TuneFunc)(std::shared_ptr<Operator>, const vector<Tensor*>&,
                                    const vector<Tensor*>&, const bool&);
  explicit OpTuning(const string& type) : type_(type) {}
  ~OpTuning() {}

  // call realted tune function by type and kernel_name
  // prepare tuning config, like extra tensor life
  void Start(const string& kernel_name, std::shared_ptr<Operator> kernel,
             const vector<Tensor*>& input, const vector<Tensor*>& output,
             const bool& reshape_model) {
    set_tune_func_name(kernel_name);
    stage_ = "start";
    (this->*(tune_func_map_[tune_func_name_]))(kernel, input, output, reshape_model);
    stage_.clear();
  }

  // tune kernel and record time and dispatch kernel config
  void Run(const string& kernel_name, std::shared_ptr<Operator> kernel,
           const vector<Tensor*>& input, const vector<Tensor*>& output,
           const bool& reshape_model) {
    kernel_config_.clear();
    kernel_config_.push_back(kernel_name);
    set_tune_func_name(kernel_name);
    (this->*(tune_func_map_[tune_func_name_]))(kernel, input, output, reshape_model);
  }

  void BaseTune(std::shared_ptr<Operator> kernel, const vector<Tensor*>& input,
                const vector<Tensor*>& output, const bool& reshape_model) {
    if (stage_ == "start") {
      extra_tensor_life_ += 1;
      return;
    }
    float start_time = 0;
    // consider kernel forward class creation time in Reshape
    // it may be time-consuming under dynamic shape
    float reshape_time = 0;
    if (reshape_model) {
      start_time = Time("start");
      kernel->Reshape(input, output);
      reshape_time = Time("end") - start_time;
    }
    start_time = Time("start");
    kernel->Forward(input, output);
    best_execute_time_ = Time("end") - start_time + reshape_time;
    LOG(INFO) << "BaseTune forward time is " << best_execute_time_ << "ms";
  }

  // MxK x KxN to
  // Nx1xWxK(c_i) x 1x1xK(c_i)xN(c_o)
  // find the best N-W combination
  void IpToConvTune(std::shared_ptr<Operator> kernel, const vector<Tensor*>& input,
                    const vector<Tensor*>& output, const bool& reshape_model) {
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
      float start_time = 0;
      float reshape_time = 0;
      start_time = Time("start");
      kernel->Reshape(input, output);
      reshape_time = Time("end") - start_time;
      start_time = Time("start");
      kernel->Forward(input, output);
      float execute_time = Time("end") - start_time;
      if (reshape_model) execute_time += reshape_time;
      input_shape_timer[execute_time] = comb;
      LOG(INFO) << "IpToConvTune forward time is " << execute_time << "ms with src0 shape " << comb;
    }
    if (input_shape_timer.size() > 0) {
      best_execute_time_ = input_shape_timer.begin()->first;
      kernel_config_.push_back(input_shape_timer.begin()->second);
    } else {
      LOG(FATAL) << "InnerProduct tuning fails with kernel convolution...";
    }
  }

  inline const float& best_execute_time() const { return best_execute_time_;}
  inline const vector<string>& kernel_config() const { return kernel_config_; }
  inline const int& extra_tensor_life() const { return extra_tensor_life_; }
  inline void reset_extra_tensor_life() { extra_tensor_life_ = -1; }

 private:
  void set_tune_func_name(const string& kernel_name) {
    tune_func_name_ = type_ + "_to_" + kernel_name;
    if (tune_func_map_.count(tune_func_name_) == 0) {
      if (kernel_name != type_) LOG(WARNING) << "No matching tuning function for " << tune_func_name_
                                              << ", gonna use BaseTune function instead...";
      tune_func_name_ = "Base";
    }
  }

  string type_;
  float best_execute_time_ = std::numeric_limits<float>::max();
  vector<string> kernel_config_;
  int extra_tensor_life_ = -1;
  string tune_func_name_;
  string stage_;
  static std::unordered_map<string, TuneFunc> tune_func_map_;
};

std::unordered_map<string, OpTuning::TuneFunc> OpTuning::tune_func_map_ = {
    {"Base", &OpTuning::BaseTune},
    {"InnerProduct_to_Convolution", &OpTuning::IpToConvTune}
};
}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_OP_TUNING_HPP_
