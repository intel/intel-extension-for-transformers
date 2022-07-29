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

#ifndef ENGINE_EXECUTOR_INCLUDE_DISPATCHER_HPP_
#define ENGINE_EXECUTOR_INCLUDE_DISPATCHER_HPP_

#include "op_tuning.hpp"
#include "isa.hpp"

namespace executor {

/**
* @brief Use dispatcher to choose a specific kernel at runtime.
*        Every operator in model should have its own default 
*        execute kernel, however it may be not the best one in
*        some scenarios.
*        Dispatcher has the ability to find the best kernel before 
*        deployment if user turn on tuning mechanism.
*        for example:
*        InnerProduct Dispatcher --  InnerProduct Kernel
*                                |-- Convolution Kernel
*        1. If do tuning, dispatcher will get the info about what
*           is the better kernel under the feed input's shape, dtype, etc.
*           And write these info into dispatcher table file, like
*           InnerProduct xxxxxx Convolution 4,1,256,256
*        2. If do not tuning, dispatcher will try to load the dispatcher 
*           table file. If dispatcher find the best kernl by hash key, it
*           will execut that kernel (maybe 1x1 conv). Otherwise, dispatcher 
*           will execute its default kernel (innerproduct there).
*
*/

class Dispatcher {
 public:
 // kernel implementation table
  typedef std::unordered_map<string, shared_ptr<Operator>> KernelHandler;

  explicit Dispatcher(const OperatorConfig& conf): operator_conf_(conf) {
    name_ = operator_conf_.name();
    type_ = operator_conf_.type();
    cpu_isa_ = get_max_isa();
    OperatorRegistry::CreatorRegistry& registry = OperatorRegistry::Registry();
    CHECK_EQ(registry.count(type_), 1) << "Unknown operator type: " << type_
        << " (known type: " << OperatorRegistry::OperatorTypeListString() << ").";
    for (const auto& k_pair: registry[type_]){
      auto kernel_name = k_pair.first;
      auto kernel = k_pair.second;
      CHECK_EQ(registry[type_].count(kernel_name), 1) << "Unknown dispatch kernel: "
        << kernel_name;
      kernel_handler_[kernel_name] = registry[type_][kernel_name](conf);
    }
    execute_kernel_ = type_;
    do_tuning_ = (getenv("ENGINE_DISPATCHER_TUNING_ON") != NULL);
  }

  ~Dispatcher() {}
  
  // prepare all kernel when model init
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
    // (TODO) handle the case that different kernel with different output data type
    // Prepare will change some status on kernel, but should not on output
    for (const auto& k_pair: kernel_handler_) {
      auto kernel = k_pair.second;
      kernel->set_dispatch_from_type(type_);
      kernel->Prepare(input, output);
      if (kernel->monopolize_dispatcher()) {
        disable_dispatch_ = true;
        break;
      }
    }
  }
  
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
    if (kernel_handler_[type_]->do_shape_infer()) {
      // reset
      kernel_handler_[type_]->set_do_shape_infer(false);
      kernel_handler_[execute_kernel_]->Reshape(input, output);
    }
  }
  
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
    kernel_handler_[execute_kernel_]->Forward(input, output);
  }
  
  void GetExecuteKernel(const vector<Tensor*>& input, const vector<Tensor*>& output, 
                        const bool& reshape_model, const string& dispatch_table_file_root,
                        const bool& has_dispatch_table_file) {
    // reset
    execute_kernel_ = type_;
    if (!do_tuning_) { 
      // get input tensor info if is under dynamic model inputs
      if (reshape_model) {
        if (kernel_handler_.size() > 1) kernel_handler_[type_]->set_do_shape_infer(true);
        kernel_handler_[type_]->Reshape(input, output);
      }
      if (!disable_dispatch_ && has_dispatch_table_file) {
        // generate hash key and find the best kernel if has dispatch table
        // only load once
        if (DispatchTable::Size() == 0) {
          LOG(INFO) << "Loading diapatch table file...";
          DispatchTable::Load(dispatch_table_file_root);
        }
        vector<string> kernel_config = DispatchTable::Find(type_, GetHash(input));
        if (!kernel_config.empty()) {
          string kernel_name = kernel_config[0];
          if (kernel_handler_.count(kernel_name) > 0) {
            execute_kernel_ = kernel_name;
            kernel_handler_[kernel_name]->set_dispatch_config(kernel_config);
          }
        }
      } 
      LOG(INFO) << "Operator " << name_ << " with type " << type_
                << " gonna dispatch by kernel " << execute_kernel_;
    } else {
      LOG(INFO) << "Dispatcher tuning mode is ON, operator " << name_ << " gonna tune kernel...";
      // skip Input and Output op
      if (type_ == "Input" || type_ == "Output") return;
      // skip same input_hash
      size_t input_hash = GetHash(input);
      if (!disable_dispatch_ && DispatchTable::Find(type_, input_hash).empty()) {
        // keep kernel with the least time as first pair
        std::map<float, vector<string>, std::less<float>> timer;
        OpTuning op_tuning(type_);
        // increase input tensors' life when tune
        // default kernel does not count towards the extra life
        for (const auto& k_pair: kernel_handler_) {
          auto kernel_name = k_pair.first;
          auto kernel = k_pair.second;
          op_tuning.Start(kernel_name, kernel, input, output, reshape_model);
        }
        for (auto& tensor : input) tensor->disposable_extra_life(op_tuning.extra_tensor_life());
        op_tuning.reset_extra_tensor_life();
        // tune kernel
        for (const auto& k_pair: kernel_handler_) {
          auto kernel_name = k_pair.first;
          auto kernel = k_pair.second;
          try {
            op_tuning.Run(kernel_name, kernel, input, output, reshape_model);
            timer[op_tuning.best_execute_time()] = op_tuning.kernel_config();
          // some kernels don't support specific dtype, fusion, etc.
          } catch (const std::exception& e) {
            LOG(WARNING) << kernel_name << " kernel tuning failure: " << e.what();
          }
        }
        if (timer.size() > 0) {
          execute_kernel_ = timer.begin()->second[0];
          LOG(INFO) << "best kernel is " << execute_kernel_ << " with time " << timer.begin()->first << "ms";
          if (execute_kernel_ != type_) DispatchTable::Insert(type_, input_hash, timer.begin()->second);
        }
      } else {
        LOG(INFO) << "Skip tuning function due to existing input hash...";
        if (reshape_model) kernel_handler_[type_]->Reshape(input, output);
        kernel_handler_[type_]->Forward(input, output);
      }
    } 
  }
  
  // turn on or turn off tuning mechanism in some specific operators if need
  inline void set_tuning_mode(const bool& mode) { do_tuning_ = mode; }
  inline const bool& do_tuning() const { return do_tuning_; }
  inline const string& name() const { return name_; }
  inline const string& type() const { return type_; }
  inline const OperatorConfig& operator_conf() const { return operator_conf_; }
  inline const string& execute_kernel() const { return execute_kernel_; }
  inline const bool& disable_dispatch() const { return disable_dispatch_; }

 protected:
  // get input_hash
  size_t GetHash(const vector<Tensor*>& input) {
    vector<size_t> combine_hash{static_cast<size_t>(cpu_isa_)};
    size_t input_hash = 0;
    for (const auto& tensor : input) combine_hash.push_back(tensor->get_hash());
    input_hash = get_array_hash(input_hash, combine_hash, combine_hash.size());
    return input_hash;
  }

  string name_;
  string type_;
  OperatorConfig operator_conf_;
  isa cpu_isa_;
  KernelHandler kernel_handler_;
  string execute_kernel_;
  bool do_tuning_ = false;
  bool disable_dispatch_ = false;
};
} //namespace executor

#endif //ENGINE_EXECUTOR_INCLUDE_DISPATCHER_HPP_