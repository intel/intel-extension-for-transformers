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

#ifndef ENGINE_EXECUTOR_INCLUDE_ACTIVATION_DAG_HANDLER_HPP_
#define ENGINE_EXECUTOR_INCLUDE_ACTIVATION_DAG_HANDLER_HPP_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include "tensor.hpp"
#include "activation_dag.hpp"

namespace executor {

/**
 * @brief The ActivationDAGHandler class is used to create or update activation DAG for activation
 *        memory buffer pre allocation and analysis.
 *        Public APIs are:
 *        1. GetDAG(input_shapes) - return the corresponding activation DAG class.
 *        2. CheckDAG() - check the activation DAG is valid or not.
 *        3. DumpDAG(output_dir) - save the activation DAG into the disk.
 *        4. LoadDAG(input_dir) - load the activation DAG from the disk.
 *
 */

class Model;

class ActivationDAGHandler {
 public:
  explicit ActivationDAGHandler(const Model* model);
  explicit ActivationDAGHandler(const Model* model, const string& dag_dir);
  ActivationDAGHandler() = default;
  ~ActivationDAGHandler() {}

  const ActivationDAG& GetDAG(const vector<shared_ptr<Dispatcher>>& ops, const vector<vector<Tensor*>>& input_vecs,
                              const vector<vector<Tensor*>>& output_vecs);
  Status CheckDAG();
  void DumpDAG(const string& output_dir);
  void LoadDAG(const string& input_dir);

  inline const bool& update() const { return update_; }

 protected:
  void InplaceAnalysis(const vector<shared_ptr<Dispatcher>>& ops, const vector<vector<Tensor*>>& input_vecs,
                       const vector<vector<Tensor*>>& output_vecs);

  shared_ptr<ActivationTensor> BuildTensor(const Tensor* tensor);
  void UpdateTensor(shared_ptr<ActivationTensor> dag_tensor, const Tensor* model_tensor);
  shared_ptr<ActivationOperator> BuildOperator(const string& name, const int64_t order, const vector<Tensor*> input,
                                               const vector<Tensor*> output);
  void UpdateOperator(shared_ptr<ActivationOperator> op, const vector<Tensor*> input, const vector<Tensor*> output);
  void BuildDAG(const vector<shared_ptr<Dispatcher>>& ops, const vector<vector<Tensor*>>& input_vecs,
                const vector<vector<Tensor*>>& output_vecs);
  void UpdateDAG(const vector<shared_ptr<Dispatcher>>& ops, const vector<vector<Tensor*>>& input_vecs,
                 const vector<vector<Tensor*>>& output_vecs);

  string find_alias(const string& tensor_name);
  bool is_activation(const Tensor* tensor, const vector<Tensor*> model_input_tensors);
  Status memory_status(shared_ptr<ActivationTensor> tensor);

  ActivationDAG dag_;
  const Model* model_ = nullptr;
  bool update_ = false;
  unordered_map<string, shared_ptr<ActivationTensor>> name2tensor_;
  // tensor_name: true -> finish building
  // tensor_name: false -> finish updating
  unordered_map<string, bool> building_status_;
  // tensor memory name -> [tensor semantic names]
  unordered_map<string, vector<string>> inplace_alias_info_;
  // tensor semantic name -> tensor memory name
  unordered_map<string, string> tensor2alias_;
};
}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_ACTIVATION_DAG_HANDLER_HPP_
