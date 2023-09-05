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

#include "model.hpp"
#include "activation_dag_handler.hpp"

namespace executor {

// ActivationDAGHandler class construction
ActivationDAGHandler::ActivationDAGHandler(const Model* model) : model_(model) {}

ActivationDAGHandler::ActivationDAGHandler(const Model* model, const string& dag_dir) : model_(model) {
  LoadDAG(dag_dir);
}

void ActivationDAGHandler::InplaceAnalysis(const vector<shared_ptr<Dispatcher>>& ops,
                                           const vector<vector<Tensor*>>& input_vecs,
                                           const vector<vector<Tensor*>>& output_vecs) {
  DLOG(INFO) << "Start to implement model inplace tensors anaysis...";
  inplace_alias_info_.clear();
  tensor2alias_.clear();
  for (int i = 0; i < ops.size(); ++i) {
    vector<vector<string>> inplace_pairs = ops[i]->InplacePairs(input_vecs[i], output_vecs[i]);
    // assume pairs will not corrleate with each other
    for (const auto& inplace_pair : inplace_pairs) {
      CHECK_EQ(inplace_pair.size(), 2) << "Inplace tensor name pair length should be 2!";
      if (tensor2alias_.count(inplace_pair[0]) != 0) {
        string alias = tensor2alias_[inplace_pair[0]];
        LOG_IF(FATAL, inplace_alias_info_.count(alias) == 0) << "Inplace alias " << alias << " not exists!";
        inplace_alias_info_[alias].push_back(inplace_pair[1]);
        tensor2alias_[inplace_pair[1]] = alias;
      } else {
        string alias = inplace_pair[0] + "_inplace";
        LOG_IF(FATAL, inplace_alias_info_.count(alias) != 0) << "Inplace alias " << alias << " has already existed!";
        inplace_alias_info_[alias].assign(inplace_pair.begin(), inplace_pair.end());
        tensor2alias_[inplace_pair[0]] = alias;
        tensor2alias_[inplace_pair[1]] = alias;
      }
    }
    // for inplace info collection
    ops[i]->decrease_consumers(input_vecs[i]);
  }
  DLOG(INFO) << "Finish model inplace tensors anaysis...";
}

shared_ptr<ActivationTensor> ActivationDAGHandler::BuildTensor(const Tensor* tensor) {
  DLOG(INFO) << "Building activation tensor " << tensor->name() << " in DAG...";
  if (building_status_.count(tensor->name()) != 0) {
    return name2tensor_[tensor->name()];
  } else {
    string memory_name = find_alias(tensor->name()) == "" ? tensor->name() : find_alias(tensor->name());
    shared_ptr<ActivationTensor> t_ptr = std::make_shared<ActivationTensor>(
        memory_name, tensor->alloc_bytes(), tensor->dtype(), tensor->shape(), tensor->name());
    name2tensor_[tensor->name()] = t_ptr;
    building_status_[tensor->name()] = true;
    return t_ptr;
  }
}

// only change shape and alloc_bytes
void ActivationDAGHandler::UpdateTensor(shared_ptr<ActivationTensor> dag_tensor, const Tensor* model_tensor) {
  LOG_IF(FATAL, dag_tensor->semantic_alias() != model_tensor->name())
      << "Activation tensor " << dag_tensor->semantic_alias() << " does not match model tensor "
      << model_tensor->name();
  LOG_IF(FATAL, dag_tensor->dtype() != "" && dag_tensor->dtype() != model_tensor->dtype())
      << "Activation tensor " << dag_tensor->semantic_alias() << " has dtype " << dag_tensor->dtype()
      << ", but model tensor has dtype " << model_tensor->dtype();
  DLOG(INFO) << "Updating activation tensor " << model_tensor->name() << " in DAG...";
  if (building_status_.count(model_tensor->name()) != 0 && building_status_[model_tensor->name()]) {
    dag_tensor->Update(model_tensor->alloc_bytes(), model_tensor->shape());
    building_status_[model_tensor->name()] = false;
  }
}

shared_ptr<ActivationOperator> ActivationDAGHandler::BuildOperator(const string& name, const int64_t order,
                                                                   const vector<Tensor*> input,
                                                                   const vector<Tensor*> output) {
  DLOG(INFO) << "Building activation operator " << name << " in DAG...";
  vector<shared_ptr<ActivationTensor>> t_in;
  vector<shared_ptr<ActivationTensor>> t_out;
  for (const auto t : input) {
    if (is_activation(t, model_->model_input_tensors_)) {
      t_in.push_back(BuildTensor(t));
    }
  }
  // op should only produce activation in neural engine
  for (const auto t : output) {
    t_out.push_back(BuildTensor(t));
  }
  return std::make_shared<ActivationOperator>(name, order, t_in, t_out);
}

void ActivationDAGHandler::UpdateOperator(shared_ptr<ActivationOperator> op, const vector<Tensor*> input,
                                          const vector<Tensor*> output) {
  DLOG(INFO) << "Updating activation operator " << op->name() << " in DAG...";
  if (!op->input().empty() && !input.empty()) {
    int j = 0;
    for (int i = 0; i < input.size(); ++i) {
      if (!is_activation(input[i], model_->model_input_tensors_)) continue;
      UpdateTensor(op->input()[j++], input[i]);
    }
  }
  if (!op->output().empty() && !output.empty()) {
    LOG_IF(FATAL, op->output().size() != output.size())
        << "Operator " << op->name() << " should only produce activation in neural engine";
    for (int i = 0; i < output.size(); ++i) {
      UpdateTensor(op->output()[i], output[i]);
    }
  }
}

// first init
void ActivationDAGHandler::BuildDAG(const vector<shared_ptr<Dispatcher>>& ops,
                                    const vector<vector<Tensor*>>& input_vecs,
                                    const vector<vector<Tensor*>>& output_vecs) {
  DLOG(INFO) << "Building the activation DAG...";
  name2tensor_.clear();
  building_status_.clear();
  vector<shared_ptr<ActivationOperator>> operators;
  int64_t topological_order = 0;
  for (int i = 0; i < ops.size(); ++i) {
    if (ops[i]->type() == "Input") continue;
    operators.push_back(BuildOperator(ops[i]->name(), topological_order++, input_vecs[i], output_vecs[i]));
  }
  if (inplace_alias_info_.empty()) {
    dag_ = ActivationDAG(operators);
  } else {
    dag_ = ActivationDAG(operators, inplace_alias_info_);
  }
}

// other input_shapes
// only chang the tensor's shape and alloc_bytes but keeping the rest of patrs of DAG
void ActivationDAGHandler::UpdateDAG(const vector<shared_ptr<Dispatcher>>& ops,
                                     const vector<vector<Tensor*>>& input_vecs,
                                     const vector<vector<Tensor*>>& output_vecs) {
  DLOG(INFO) << "Updating the activation DAG...";
  LOG_IF(FATAL, dag_.operators().empty()) << "The activation DAG is empty, please call BuildDAG first";
  int64_t topological_order = 0;
  for (int i = 0; i < ops.size(); ++i) {
    if (ops[i]->type() == "Input") continue;
    UpdateOperator(dag_.operators()[topological_order++], input_vecs[i], output_vecs[i]);
  }
}

// public APIs
// check if the DAG is legitimate or not
Status ActivationDAGHandler::CheckDAG() {
  DLOG(INFO) << "Start to check the activation DAG...";
  if (dag_.operators().empty()) {
    DLOG(INFO) << "Skip check since the activation DAG is empty...";
    return Status::Unknown;
  }
  for (int i = 0; i < dag_.operators().size(); ++i) {
    // check operator topological_order
    if (dag_.operators()[i]->topological_order() == -1) {
      LOG(WARNING) << "Activation operator " << dag_.operators()[i]->name() << " topological_order must be assigned!";
      return Status::InvalidGraph;
    }
    // check operator input tensors memory
    for (auto t : dag_.operators()[i]->input()) {
      if (memory_status(t) != Status::Success) return Status::InvalidGraph;
    }
    // check operator output tensors memory
    for (auto t : dag_.operators()[i]->output()) {
      if (memory_status(t) != Status::Success) return Status::InvalidGraph;
    }
    DLOG(INFO) << "Finish checking the activation DAG which is valid.";
    return Status::Success;
  }
}

// create or update the DAG
const ActivationDAG& ActivationDAGHandler::GetDAG(const vector<shared_ptr<Dispatcher>>& ops,
                                                  const vector<vector<Tensor*>>& input_vecs,
                                                  const vector<vector<Tensor*>>& output_vecs) {
  LOG_IF(FATAL, ops.size() != input_vecs.size() || ops.size() != output_vecs.size())
      << "The model operators size is not matched with its related input_vecs or output vecs...";
  DLOG(INFO) << "Start to get activation DAG...";
  if (!update_) {
    InplaceAnalysis(ops, input_vecs, output_vecs);
    BuildDAG(ops, input_vecs, output_vecs);
  } else {
    UpdateDAG(ops, input_vecs, output_vecs);
  }
  update_ = true;
  DLOG(INFO) << "Got the activation DAG...";
  Status dag_status = CheckDAG();
  LOG_IF(FATAL, (dag_status == Status::Unknown || dag_status == Status::InvalidGraph))
      << "The activation DAG is invalid, please call DumpDAG() function to check the graph!";
  return dag_;
}

void ActivationDAGHandler::DumpDAG(const string& output_dir) { dag_.Dump(output_dir); }

void ActivationDAGHandler::LoadDAG(const string& input_dir) { dag_ = ActivationDAG(input_dir); }

// utils
string ActivationDAGHandler::find_alias(const string& tensor_name) {
  string alias = "";
  if (!tensor2alias_.empty()) {
    auto iter = tensor2alias_.find(tensor_name);
    if (iter != tensor2alias_.end()) alias = iter->second;
  }
  return alias;
}

bool ActivationDAGHandler::is_activation(const Tensor* tensor, const vector<Tensor*> model_input_tensors) {
  if (!name2tensor_.empty() && name2tensor_.count(tensor->name()) != 0) {
    return true;
  }
  bool is_weight = tensor->location().empty() ? false : true;
  bool is_model_input = false;
  for (auto t : model_input_tensors) {
    if (tensor->name() == t->name()) {
      is_model_input = true;
      break;
    }
  }
  return (!is_weight && !is_model_input);
}

Status ActivationDAGHandler::memory_status(shared_ptr<ActivationTensor> tensor) {
  if (tensor->alloc_bytes() == 0) {
    LOG(WARNING) << "Activation tensor " << tensor->name() << " alloc_bytes must be assigned!";
    return Status::OutOfMemory;
  }
  if (tensor->shape().empty() || tensor->dtype().empty()) {
    LOG(WARNING) << "Can not check activation tensor " << tensor->name()
                 << " memory alloc_bytes status due to empty shape or dtype.";
    return Status::Unknown;
  }
  size_t bytes = std::accumulate(tensor->shape().begin(), tensor->shape().end(), size_t(1), std::multiplies<size_t>()) *
                 type2bytes[tensor->dtype()];
  if (tensor->alloc_bytes() != bytes) {
    LOG(WARNING) << "Activation tensor " << tensor->name() << " has mismatched shape, dtype, alloc_bytes!";
    return Status::OutOfMemory;
  } else {
    return Status::Success;
  }
}

}  // namespace executor
