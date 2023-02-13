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
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <limits>

#include "dispatch_table.hpp"
#include "common.hpp"

namespace executor {

/**
* @brief  get each kernel's best execute time and related dispatch config for dispatcher
*
*/

class Model;  // forward declaration

class OpTuning {
 public:
  typedef void(OpTuning::*TuneFunc)(std::shared_ptr<Operator>, const vector<Tensor*>&,
                                    const vector<Tensor*>&, const bool&);
  explicit OpTuning(const string& type, const Model* m_ptr) : type_(type), model_(m_ptr) {}
  ~OpTuning() {}

  // call realted tune function by type and kernel_name
  // prepare tuning config, like extra tensor life
  void Start(const string& kernel_name, std::shared_ptr<Operator> kernel,
             const vector<Tensor*>& input, const vector<Tensor*>& output,
             const bool& reshape_model);

  // tune kernel and record time and dispatch kernel config
  void Run(const string& kernel_name, std::shared_ptr<Operator> kernel,
           const vector<Tensor*>& input, const vector<Tensor*>& output,
           const bool& reshape_model);

  void BaseTune(std::shared_ptr<Operator> kernel, const vector<Tensor*>& input,
                const vector<Tensor*>& output, const bool& reshape_model);

  // MxK x KxN to
  // Nx1xWxK(c_i) x 1x1xK(c_i)xN(c_o)
  // find the best N-W combination
  void IpToConvTune(std::shared_ptr<Operator> kernel, const vector<Tensor*>& input,
                    const vector<Tensor*>& output, const bool& reshape_model);

  // split the dimension from 2D to 3D when use sparselib gemm
  void IpToSparseLibTune(std::shared_ptr<Operator> kernel, const vector<Tensor*>& input,
                    const vector<Tensor*>& output, const bool& reshape_model);

  inline const float& best_execute_time() const { return best_execute_time_;}
  inline const vector<string>& kernel_config() const { return kernel_config_; }
  inline const int& extra_tensor_life() const { return extra_tensor_life_; }
  inline void reset_extra_tensor_life() { extra_tensor_life_ = -1; }

 private:
  void set_tune_func_name(const string& kernel_name);

  string type_;
  float best_execute_time_ = std::numeric_limits<float>::max();
  vector<string> kernel_config_;
  int extra_tensor_life_ = -1;
  string tune_func_name_;
  string stage_;
  static std::unordered_map<string, TuneFunc> tune_func_map_;
  const Model* model_ = nullptr;
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_OP_TUNING_HPP_
