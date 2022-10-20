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

#include <functional>
#include <map>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
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
    for (const auto& k_pair : registry[type_]) {
      auto kernel_name = k_pair.first;
      auto kernel = k_pair.second;
      CHECK_EQ(registry[type_].count(kernel_name), 1) << "Unknown dispatch kernel: "
        << kernel_name;
      kernel_handler_[kernel_name] = registry[type_][kernel_name](conf);
    }
    execute_kernel_ = type_;
    do_tuning_ = (getenv("ENGINE_DISPATCHER_TUNING_ON") != NULL);
  }

  explicit Dispatcher(const shared_ptr<Operator>& op) : operator_conf_(op->operator_conf()) {
    name_ = operator_conf_.name();
    type_ = operator_conf_.type();
    cpu_isa_ = get_max_isa();
    kernel_handler_[name_] = op;
    execute_kernel_ = type_;
    do_tuning_ = (getenv("ENGINE_DISPATCHER_TUNING_ON") != NULL);
  }

  ~Dispatcher() {}

  // prepare all kernel when model init
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
    // (TODO) handle the case that different kernel with different output data type
    // Prepare will change some status on kernel, but should not on output
    for (int i = 0; i < kernel_handler_.size(); ++i) sparselib_available_.push_back(false);
    int idx = 0;
    // let default kernel prepare first
    kernel_handler_[type_]->Prepare(input, output);
    for (const auto& k_pair : kernel_handler_) {
      auto kernel_name = k_pair.first;
      auto kernel = k_pair.second;
      kernel->set_dispatch_from_type(type_);
      if (kernel_name != type_) kernel->Prepare(input, output);
      sparselib_available_[idx++] = kernel->kernel_type() == SparseLib ? true : false;
      if (tune_dense_in_sparse_ && do_tuning_ && kernel->kernel_type() == SparseLib) {
        kernel->set_kernel_type(Dense);
        kernel->Prepare(input, output);
        kernel->set_kernel_type(SparseLib);
      }
      if ((kernel_handler_.size() < 2 || kernel->monopolize_dispatcher())
          && !sparselib_available_[0]) no_tuning_space_ = true;
      if (kernel->monopolize_dispatcher()) {
        monopoly_kernel_ = kernel_name;
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
    AdaptTensors(input, output, "in");
    kernel_handler_[execute_kernel_]->Forward(input, output);
    AdaptTensors(input, output, "out");
  }

  // modify tensors when dispatched kernel needs different format, etc.
  // should call it before (in) or after (out) Forward
  void AdaptTensors(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage) {
    if (!sparselib_available_[0] || kernel_handler_[execute_kernel_]->dispatch_config().empty()) return;
    kernel_handler_[execute_kernel_]->AdaptTensors(input, output, stage);
  }

  void GetExecuteKernel(const vector<Tensor*>& input, const vector<Tensor*>& output,
                        const bool& reshape_model, const string& dispatch_table_file_root,
                        const bool& has_dispatch_table_file) {
    LOG(INFO) << "Operator " << name_ << " with type " << type_ << " is ready to get execute kernel...";
    // reset
    execute_kernel_ = type_;
    if (!do_tuning_) {
      // get input tensor info if is under dynamic model inputs
      if (reshape_model) {
        if (kernel_handler_.size() > 1 || (!sparselib_available_.empty() && sparselib_available_[0])) {
          kernel_handler_[type_]->set_do_shape_infer(true);
          kernel_handler_[type_]->ShapeInfer(input, output);
        }
        kernel_handler_[type_]->Reshape(input, output);
      }
      vector<string> kernel_config;
      if (!no_tuning_space_ && has_dispatch_table_file) {
        // generate hash key and find the best kernel if has dispatch table
        // only load once
        if (DispatchTable::Size() == 0) {
          LOG(INFO) << "Loading diapatch table file...";
          DispatchTable::Load(dispatch_table_file_root);
        }
        kernel_config = DispatchTable::Find(type_, GetHash(input));
        if (!kernel_config.empty()) {
          string kernel_name = kernel_config[0];
          // sparselib
          if (kernel_name == "SparseLib") {
            execute_kernel_ = type_;
            kernel_handler_[type_]->set_dispatch_config(kernel_config);
          } else {
            // dense
            if (kernel_handler_.count(kernel_name) > 0) {
              execute_kernel_ = kernel_name;
              kernel_handler_[kernel_name]->set_dispatch_config(kernel_config);
            }
          }
        }
      }
      LOG(INFO) << "Operator " << name_ << " with type " << type_ << " gonna dispatch by kernel "
                << (kernel_config.empty() ? execute_kernel_ : kernel_config[0]);
    } else {
      LOG(INFO) << "Dispatcher tuning mode is ON, operator " << name_ << " gonna tune kernel...";
      // skip Input and Output op
      if (type_ == "Input" || type_ == "Output") return;
      // skip same input_hash
      size_t input_hash = GetHash(input);
      iter_cnt_ += 1;
      // consider warmup when tuning
      if (!no_tuning_space_ && (iter_cnt_<= warmup_iter_ + 1 || DispatchTable::Find(type_, input_hash).empty())) {
        // keep kernel with the least time as first pair
        std::map<float, vector<string>, std::less<float>> timer;
        OpTuning op_tuning(type_);
        // increase input tensors' life when tune
        // default kernel does not count towards the extra life
        int idx = 0;
        string suffix;
        for (const auto& k_pair : kernel_handler_) {
          auto kernel_name = k_pair.first;
          auto kernel = k_pair.second;
          suffix = sparselib_available_[idx++] ? "SparseLib" : kernel_name;
          if (tune_dense_in_sparse_ && suffix == "SparseLib") {
            kernel->set_kernel_type(Dense);
            op_tuning.Start(kernel_name, kernel, input, output, reshape_model);
            kernel->set_kernel_type(SparseLib);
          }
          op_tuning.Start(suffix, kernel, input, output, reshape_model);
          if (monopoly_kernel_ == kernel_name) break;
        }
        for (auto& tensor : input) tensor->disposable_extra_life(op_tuning.extra_tensor_life());
        op_tuning.reset_extra_tensor_life();
        // tune kernel
        idx = 0;
        for (const auto& k_pair : kernel_handler_) {
          auto kernel_name = k_pair.first;
          auto kernel = k_pair.second;
          suffix = sparselib_available_[idx++] == true ? "SparseLib" : kernel_name;
          try {
            if (tune_dense_in_sparse_ && suffix == "SparseLib") {
              kernel->set_kernel_type(Dense);
              op_tuning.Run(kernel_name, kernel, input, output, reshape_model);
              kernel->set_kernel_type(SparseLib);
            }
            op_tuning.Run(suffix, kernel, input, output, reshape_model);
            timer[op_tuning.best_execute_time()] = op_tuning.kernel_config();
          // some kernels don't support specific dtype, fusion, etc.
          } catch (const std::exception& e) {
            LOG(WARNING) << kernel_name << " kernel tuning failure: " << e.what();
          }
          if (monopoly_kernel_ == kernel_name) break;
        }
        if (timer.size() > 0) {
          execute_kernel_ = timer.begin()->second[0];
          LOG(INFO) << "best kernel is " << execute_kernel_ << " with time " << timer.begin()->first << "ms";
          if (execute_kernel_ != type_) DispatchTable::Insert(type_, input_hash, timer.begin()->second);
        }
      } else {
        LOG(INFO) << "Skip tuning function due to existing input hash or no tuning space...";
        vector<string> kernel_config = DispatchTable::Find(type_, input_hash);
        string kernel_name = (!kernel_config.empty() && kernel_config[0] != "SparseLib") ? kernel_config[0] : type_;
        kernel_handler_[kernel_name]->set_dispatch_config(kernel_config);
        if (reshape_model || !kernel_config.empty()) kernel_handler_[kernel_name]->Reshape(input, output);
        execute_kernel_ = kernel_name;
        Forward(input, output);
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
  inline const bool& no_tuning_space() const { return no_tuning_space_; }
  inline const void set_warmup_iter(const int& warmup_iter) { warmup_iter_ = warmup_iter; }
  // for profiling
  inline void set_post_op(const string& post_op) { kernel_handler_[execute_kernel_]->set_post_op(post_op); }
  inline const string& post_op() { return kernel_handler_[execute_kernel_]->post_op(); }
  inline void set_latency(const float latency) { kernel_handler_[execute_kernel_]->set_latency(latency); }
  inline const vector<float>& latency() { return kernel_handler_[execute_kernel_]->latency(); }
  inline void set_enable_sparse(const bool enable_sparse) {
    kernel_handler_[execute_kernel_]->set_enable_sparse(enable_sparse);
  }
  inline const float& enable_sparse() { return kernel_handler_[execute_kernel_]->enable_sparse(); }
  inline const KERNEL_TYPE& kernel_type() { return kernel_handler_[execute_kernel_]->kernel_type(); }
  inline const float& weight_zero_ratio() { return kernel_handler_[execute_kernel_]->weight_zero_ratio(); }
  inline void set_weight_shape(const vector<int64_t>& weight_shape) {
    kernel_handler_[execute_kernel_]->set_weight_shape(weight_shape);
  }
  inline const vector<int64_t>& weight_shape() { return kernel_handler_[execute_kernel_]->weight_shape(); }
  inline void set_table_id(string table_id) {
    kernel_handler_[execute_kernel_]->set_table_id(table_id);
  }
  inline const string& table_id() { return kernel_handler_[execute_kernel_]->table_id(); }
  inline void set_perf_ratio_id(string perf_ratio_id) {
    kernel_handler_[execute_kernel_]->set_perf_ratio_id(perf_ratio_id);
  }
  inline const string& perf_ratio_id() { return kernel_handler_[execute_kernel_]->perf_ratio_id(); }
  inline void set_it_shape(const vector<int64_t>input_shape) {
                           kernel_handler_[execute_kernel_]->set_it_shape(input_shape); }
  inline void set_ot_shape(const vector<int64_t>output_shape) {
                           kernel_handler_[execute_kernel_]->set_ot_shape(output_shape); }
  inline const vector<vector<int64_t>>& get_it_shape() {
                           return kernel_handler_[execute_kernel_]->get_it_shape(); }
  inline const vector<vector<int64_t>>& get_ot_shape() {
                           return kernel_handler_[execute_kernel_]->get_ot_shape(); }
  inline void set_reshape_time(const float reshape_time_) {
                           kernel_handler_[execute_kernel_]->set_reshape_time(reshape_time_); }
  inline const vector<float>& get_reshape_time() {
                           return kernel_handler_[execute_kernel_]->get_reshape_time(); }
  inline void set_attrs(const std::map<string, string>input_attrs) {
                           kernel_handler_[execute_kernel_]->set_attrs(input_attrs);}
  inline const std::map<string, string>& get_attrs() { return kernel_handler_[execute_kernel_]->get_attrs();}

 protected:
  // get input_hash
  size_t GetHash(const vector<Tensor*>& input) {
    vector<size_t> combine_hash{static_cast<size_t>(cpu_isa_)};
    size_t input_hash = 0;
    for (const auto& tensor : input) combine_hash.push_back(tensor->get_hash());
    input_hash = get_array_hash(input_hash, combine_hash, combine_hash.size());
    input_hash = get_array_hash(input_hash, sparselib_available_, sparselib_available_.size());
    return input_hash;
  }

  string name_;
  string type_;
  OperatorConfig operator_conf_;
  isa cpu_isa_;
  KernelHandler kernel_handler_;
  string execute_kernel_;
  bool do_tuning_ = false;
  bool no_tuning_space_ = false;
  int64_t warmup_iter_ = 1;
  int64_t iter_cnt_ = 0;
  vector<bool> sparselib_available_;
  bool tune_dense_in_sparse_ = false;
  string monopoly_kernel_;
};
}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_DISPATCHER_HPP_
