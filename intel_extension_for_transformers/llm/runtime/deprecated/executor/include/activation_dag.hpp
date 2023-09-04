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

#ifndef ENGINE_EXECUTOR_INCLUDE_ACTIVATION_DAG_HPP_
#define ENGINE_EXECUTOR_INCLUDE_ACTIVATION_DAG_HPP_

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <unordered_map>

#include "glog/logging.h"
#include "yaml-cpp/yaml.h"

namespace executor {

/**
 * @brief The ActivationDAG is used for model activation tensors memory buffer pre allocating
 *        and runtime assignment.
 *        ActivationDAG (model, inplace_tensors_info)
 *                  /\
 *                  ||
 *        ActivationOperator (op_name, topological_order, input_tensors, output_tensors)
 *                  /\
 *                  ||
 *        ActivationTensor (tensor_memory_name, memory_bytes, dtype, shape, tensor_semantic_name)
 *        The tensors which have same memory name meaning their memory would be inplaced
 *        serialization exmaple (yaml):
 *        ActivationDAG:
 *          ActivationOperator:
 *            op_name:
 *              topological_order: 0
 *              input:
 *                tensor_memory_name:
 *                  alloc_bytes: 1024
 *                  dtype: fp32
 *                  shape: [16, 16]
 *                  semantic_alias: ""
 *                tensor_memory_name:
 *                  alloc_bytes: 1024
 *                  dtype: fp32
 *                  shape: [16, 16]
 *                  semantic_alias: ""
 *              output:
 *                tensor_memory_name:
 *                  alloc_bytes: 1024
 *                  dtype: fp32
 *                  shape: [16, 16]
 *                  semantic_alias: ""
 *            ......
 *          InplaceAliasHolder:
 *            tensor_memory_name:
 *              tensor_semantic_name
 *              tensor_semantic_name
 *
 */

using std::shared_ptr;
using std::string;
using std::unordered_map;
using std::vector;

class ActivationTensor {
 public:
  explicit ActivationTensor(const string& name, const size_t& bytes, const string& dtype = "fp32",
                            const vector<int64_t>& shape = {}, const string& alias = "");
  explicit ActivationTensor(const string& name, const YAML::Node& node);
  ActivationTensor() = default;
  ~ActivationTensor() {}

  void Update(const size_t& bytes, const vector<int64_t>& shape = {}, const string& name = "", const string& dtype = "",
              const string& alias = "");
  void LoadConfig(const YAML::Node& node);
  YAML::Node DumpConfig();

  inline const string& name() const { return name_; }
  inline const size_t& alloc_bytes() const { return alloc_bytes_; }
  inline const string& dtype() const { return dtype_; }
  inline const vector<int64_t>& shape() const { return shape_; }
  inline const string& semantic_alias() const { return semantic_alias_; }

 protected:
  // memory buffer name which is different tensor semantic name
  string name_;
  size_t alloc_bytes_;
  string dtype_;
  vector<int64_t> shape_;
  // tensor semantic name (different semantic names may use same memory buffer)
  string semantic_alias_;
};

class ActivationOperator {
 public:
  explicit ActivationOperator(const string& name, const int64_t& order,
                              const vector<shared_ptr<ActivationTensor>>& input,
                              const vector<shared_ptr<ActivationTensor>>& output);
  explicit ActivationOperator(const string& name, const YAML::Node& node);
  ActivationOperator() = default;
  ~ActivationOperator() {}

  void LoadConfig(const YAML::Node& node);
  YAML::Node DumpConfig();

  inline const string& name() const { return name_; }
  inline const int64_t& topological_order() const { return topological_order_; }
  inline const vector<shared_ptr<ActivationTensor>>& input() const { return input_; }
  inline const vector<shared_ptr<ActivationTensor>>& output() const { return output_; }

 protected:
  string name_;
  int64_t topological_order_;
  vector<shared_ptr<ActivationTensor>> input_;
  vector<shared_ptr<ActivationTensor>> output_;
};

class ActivationDAG {
 public:
  explicit ActivationDAG(vector<shared_ptr<ActivationOperator>> operators);
  explicit ActivationDAG(vector<shared_ptr<ActivationOperator>> operators,
                         unordered_map<string, vector<string>> inplace_alias_holder);
  explicit ActivationDAG(const string& file_path);
  ActivationDAG() = default;
  ~ActivationDAG() {}

  void LoadConfig(const YAML::Node& node);
  YAML::Node DumpConfig();
  void Load(const string& input_dir);
  void Dump(const string& output_dir);

  inline const vector<shared_ptr<ActivationOperator>>& operators() const { return operators_; }
  inline const unordered_map<string, vector<string>>& inplace_alias_holder() const { return inplace_alias_holder_; }

 protected:
  vector<shared_ptr<ActivationOperator>> operators_;
  // keep inplace tensors in order
  unordered_map<string, vector<string>> inplace_alias_holder_;
};
}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_ACTIVATION_DAG_HPP_
