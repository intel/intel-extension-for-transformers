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

#include "dispatcher.hpp"
#include "op_tuning.hpp"
#include "model.hpp"

namespace executor {

Dispatcher::Dispatcher(const shared_ptr<OperatorConfig>& conf, const ExecutionOptions* e_ptr, const Model* m_ptr)
    : operator_conf_(conf),
      execution_options_ptr_(e_ptr),
      model_(m_ptr) {
  name_ = operator_conf_->name();
  type_ = operator_conf_->type();
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
    kernel_handler_[kernel_name]->execution_options_ptr_ = execution_options_ptr_;
    kernel_handler_[kernel_name]->model_ = model_;
  }
  execute_kernel_ = type_;
  do_tuning_ = (execution_options_ptr_->execution_mode == ExecutionMode::TUNING);
  adapt_action_ = (model_->has_dispatch_table_file() && execution_options_ptr_->execution_mode != ExecutionMode::DEBUG)
                      ? true
                      : false;
}

Dispatcher::Dispatcher(const shared_ptr<Operator>& op, const ExecutionOptions* e_ptr, const Model* m_ptr)
    : operator_conf_(op->operator_conf()),
      execution_options_ptr_(e_ptr),
      model_(m_ptr) {
  name_ = operator_conf_->name();
  type_ = operator_conf_->type();
  cpu_isa_ = get_max_isa();
  kernel_handler_[name_] = op;
  execute_kernel_ = type_;
  do_tuning_ = (execution_options_ptr_->execution_mode == ExecutionMode::TUNING);
  adapt_action_ = (model_->has_dispatch_table_file() && execution_options_ptr_->execution_mode != ExecutionMode::DEBUG)
                      ? true
                      : false;
}

// prepare all kernel when model init
void Dispatcher::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // (TODO) handle the case that different kernel with different output data type
  // Prepare will change some status on kernel, but should not on output
  for (int i = 0; i < kernel_handler_.size(); ++i) sparselib_available_.push_back(false);
  int idx = 0;
  // let default kernel prepare first
  kernel_handler_[type_]->Prepare(input, output);
  if (execution_options_ptr_->execution_mode == ExecutionMode::INFERENCE && model_ != nullptr &&
      !model_->has_dispatch_table_file()) return;
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

void Dispatcher::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (execution_options_ptr_->execution_mode == ExecutionMode::INFERENCE && dispatch_table_file_exists_) {
    if (kernel_handler_[type_]->do_shape_infer()) {
      // reset
      kernel_handler_[type_]->set_do_shape_infer(false);
      if (adapt_action_) AdaptAttrs(input, output, "in");
      kernel_handler_[execute_kernel_]->Reshape(input, output);
      if (adapt_action_) AdaptAttrs(input, output, "out");
    }
  } else {
    if (adapt_action_) AdaptAttrs(input, output, "in");
    kernel_handler_[execute_kernel_]->Reshape(input, output);
    if (adapt_action_) AdaptAttrs(input, output, "out");
  }
}

void Dispatcher::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (adapt_action_) AdaptTensors(input, output, "in");
  kernel_handler_[execute_kernel_]->Forward(input, output);
  if (adapt_action_) AdaptTensors(input, output, "out");
}

// modify op attrs if need.
// e.g. SparseLib 3D gemm - BatchMatmul, prem related attrs in BatchMatMul need to adjust.
void Dispatcher::AdaptAttrs(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage) {
  kernel_handler_[execute_kernel_]->AdaptAttrs(input, output, stage);
}

// modify tensors when dispatched kernel needs different format, etc.
// should call it before (in) or after (out) Forward
void Dispatcher::AdaptTensors(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage) {
  kernel_handler_[execute_kernel_]->AdaptTensors(input, output, stage);
  if (!output.empty() && stage == "out") {
    DLOG(INFO) << "Operator " << name_ << " output tensor's format is " << int(output[0]->tensor_format())
              << " (please see tensor.hpp for format details)...";
  }
}

// reset op like tensor format after finishing inference iteration
void Dispatcher::ResetOpStatus(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (execution_options_ptr_->execution_mode == ExecutionMode::INFERENCE && dispatch_table_file_exists_) {
    kernel_handler_[execute_kernel_]->ResetOpStatus(input, output);
  }
}

void Dispatcher::GetExecuteKernel(const vector<Tensor*>& input, const vector<Tensor*>& output,
                                  const bool& reshape_model, const bool& has_dispatch_table_file) {
  DLOG(INFO) << "Operator " << name_ << " with type " << type_ << " is ready to get execute kernel...";
  // reset
  execute_kernel_ = type_;
  if (!do_tuning_) {
    vector<string> kernel_config;
    if (has_dispatch_table_file && execution_options_ptr_->execution_mode != ExecutionMode::DEBUG) {
      dispatch_table_file_exists_ = true;
      // dispatch table only load once
      if (DispatchTable::Size() == 0) {
        DLOG(INFO) << "Loading diapatch table file...";
        DispatchTable::Load(execution_options_ptr_->dispatch_table_file_root);
      }
      // get input tensor info if is under dynamic model inputs
      // generate hash key and find the best kernel if has dispatch table
      if (reshape_model) {
        if (kernel_handler_.size() > 1 || (!sparselib_available_.empty() && sparselib_available_[0])) {
          kernel_handler_[type_]->set_do_shape_infer(true);
          kernel_handler_[type_]->ShapeInfer(input, output);
        }
        AdaptAttrs(input, output, "in");
        kernel_handler_[type_]->Reshape(input, output);
        AdaptAttrs(input, output, "out");
      }
      if (!no_tuning_space_) {
        kernel_config = DispatchTable::Find(type_, GetHash(input));
        if (!kernel_config.empty()) {
          string kernel_name = kernel_config[0];
          // sparselib
          if (kernel_name == "SparseLib") {
            execute_kernel_ = type_;
            kernel_handler_[type_]->set_dispatch_config(kernel_config);
            // pass SparseLib 3D shape
            kernel_handler_[type_]->ShapeInfer(input, output);
          } else {
            // dense
            if (kernel_handler_.count(kernel_name) > 0) {
              execute_kernel_ = kernel_name;
              kernel_handler_[kernel_name]->set_dispatch_config(kernel_config);
            }
          }
        } else {
          // reset kernel_config to empty if next shape not in table under dynamic shapes
          kernel_handler_[type_]->set_dispatch_config({});
        }
      }
    } else {
      adapt_action_ = false;
    }
    DLOG(INFO) << "Operator " << name_ << " with type " << type_ << " gonna dispatch by kernel "
              << (kernel_config.empty() ? execute_kernel_ : kernel_config[0]);
  } else {
    DLOG(INFO) << "Dispatcher tuning mode is ON, operator " << name_ << " gonna tune kernel...";
    // skip Input and Output op
    if (type_ == "Input" || type_ == "Output") return;
    // skip same input_hash
    size_t input_hash = GetHash(input);
    iter_cnt_ += 1;
    // consider warmup when tuning
    DLOG(INFO) << "tuning warm up iterations is " << (execution_options_ptr_->warmup_iter);
    if (!no_tuning_space_ && (iter_cnt_<= (execution_options_ptr_->warmup_iter + 1) ||
        DispatchTable::Find(type_, input_hash).empty())) {
      // keep kernel with the least time as first pair
      std::map<float, vector<string>, std::less<float>> timer;
      OpTuning op_tuning(type_, model_);
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
        DLOG(INFO) << "best kernel is " << execute_kernel_ << " with time " << timer.begin()->first << "ms";
        if (execute_kernel_ != type_) DispatchTable::Insert(type_, input_hash, timer.begin()->second);
      }
    } else {
      DLOG(INFO) << "Skip tuning function due to existing input hash or no tuning space...";
      vector<string> kernel_config = DispatchTable::Find(type_, input_hash);
      string kernel_name = (!kernel_config.empty() && kernel_config[0] != "SparseLib") ? kernel_config[0] : type_;
      kernel_handler_[kernel_name]->set_dispatch_config(kernel_config);
      execute_kernel_ = kernel_name;
      if (reshape_model || !kernel_config.empty()) Reshape(input, output);
      Forward(input, output);
    }
  }
}

// get input_hash
size_t Dispatcher::GetHash(const vector<Tensor*>& input) {
  vector<size_t> combine_hash{static_cast<size_t>(cpu_isa_)};
  size_t input_hash = 0;
  for (const auto& tensor : input) combine_hash.push_back(tensor->get_hash());
  input_hash = get_array_hash(input_hash, combine_hash, combine_hash.size());
  input_hash = get_array_hash(input_hash, sparselib_available_, sparselib_available_.size());
  return input_hash;
}

}  // namespace executor
