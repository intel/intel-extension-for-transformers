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

#include "activation_dag.hpp"

namespace executor {

// ActivationTensor class construction
ActivationTensor::ActivationTensor(const string& name, const size_t& bytes, const string& dtype,
                                   const vector<int64_t>& shape, const string& alias)
    : name_(name), alloc_bytes_(bytes), dtype_(dtype), shape_(shape), semantic_alias_(alias) {}

ActivationTensor::ActivationTensor(const string& name, const YAML::Node& node)
    : name_(name), alloc_bytes_(0), dtype_("fp32"), shape_({}), semantic_alias_("") {
  LoadConfig(node);
}

// set ActivationTensor class members
void ActivationTensor::Update(const size_t& bytes, const vector<int64_t>& shape, const string& name,
                              const string& dtype, const string& alias) {
  if (alloc_bytes_ < bytes) {
    alloc_bytes_ = bytes;
    shape_ = shape;
  }
  if (!name.empty()) name_ = name;
  if (!dtype.empty()) dtype_ = dtype;
  if (!alias.empty()) semantic_alias_ = alias;
}

// ActivationTensor class serialization and deserialization
void ActivationTensor::LoadConfig(const YAML::Node& node) {
  shape_.clear();
  for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
    YAML::Node key = it->first;
    YAML::Node value = it->second;
    if (key.as<string>() == "alloc_bytes") {
      alloc_bytes_ = value.as<size_t>();
    }
    if (key.as<string>() == "dtype") {
      dtype_ = value.as<string>();
    }
    if (key.as<std::string>() == "shape") {
      shape_ = value.as<vector<int64_t>>();
    }
    if (key.as<string>() == "semantic_alias") {
      semantic_alias_ = value.as<string>();
    }
  }
}

YAML::Node ActivationTensor::DumpConfig() {
  YAML::Node tensor_node;
  tensor_node["alloc_bytes"] = alloc_bytes_;
  if (!dtype_.empty()) tensor_node["dtype"] = dtype_;
  if (!shape_.empty()) tensor_node["shape"] = shape_;
  if (!semantic_alias_.empty()) tensor_node["semantic_alias"] = semantic_alias_;
  return tensor_node;
}

// ActivationOperator class construction
ActivationOperator::ActivationOperator(const string& name, const int64_t& order,
                                       const vector<shared_ptr<ActivationTensor>>& input,
                                       const vector<shared_ptr<ActivationTensor>>& output)
    : name_(name), topological_order_(order), input_(input), output_(output) {}

ActivationOperator::ActivationOperator(const string& name, const YAML::Node& node)
    : name_(name), topological_order_(-1), input_({}), output_({}) {
  LoadConfig(node);
}

// ActivationOperator class serialization and deserialization
void ActivationOperator::LoadConfig(const YAML::Node& node) {
  input_.clear();
  output_.clear();
  for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
    YAML::Node key = it->first;
    YAML::Node value = it->second;
    if (key.as<string>() == "topological_order") {
      topological_order_ = value.as<int64_t>();
    }
    if (key.as<string>() == "input") {
      for (YAML::const_iterator vit = value.begin(); vit != value.end(); ++vit) {
        input_.push_back(std::make_shared<ActivationTensor>(vit->first.as<string>(), vit->second));
      }
    }
    if (key.as<std::string>() == "output") {
      for (YAML::const_iterator vit = value.begin(); vit != value.end(); ++vit) {
        output_.push_back(std::make_shared<ActivationTensor>(vit->first.as<string>(), vit->second));
      }
    }
  }
}

YAML::Node ActivationOperator::DumpConfig() {
  YAML::Node operator_node;
  operator_node["topological_order"] = topological_order_;
  // ignore the first operator which receives model input and emits first activation tensor
  if (!input_.empty()) {
    YAML::Node input_node;
    for (auto t : input_) {
      input_node[t->name()] = t->DumpConfig();
    }
    operator_node["input"] = input_node;
  }
  // ignore the model Output node which has no output tensors
  if (!output_.empty()) {
    YAML::Node output_node;
    for (auto t : output_) {
      output_node[t->name()] = t->DumpConfig();
    }
    operator_node["output"] = output_node;
  }
  return operator_node;
}

// ActivationDAG class construction
ActivationDAG::ActivationDAG(const vector<shared_ptr<ActivationOperator>> operators) : operators_(operators) {}

ActivationDAG::ActivationDAG(vector<shared_ptr<ActivationOperator>> operators,
                             unordered_map<string, vector<string>> inplace_alias_holder)
    : operators_(operators), inplace_alias_holder_(inplace_alias_holder) {}

ActivationDAG::ActivationDAG(const string& file_path) { Load(file_path); }

// ActivationDAG serialization and deserialization
void ActivationDAG::LoadConfig(const YAML::Node& node) {
  operators_.clear();
  inplace_alias_holder_.clear();
  for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
    YAML::Node key = it->first;
    YAML::Node value = it->second;
    if (key.as<string>() == "ActivationOperator") {
      for (YAML::const_iterator vit = value.begin(); vit != value.end(); ++vit) {
        operators_.push_back(std::make_shared<ActivationOperator>(vit->first.as<string>(), vit->second));
      }
    }
    if (key.as<string>() == "InplaceAliasHolder") {
      for (YAML::const_iterator vit = value.begin(); vit != value.end(); ++vit) {
        auto tensor_names_config = vit->second;
        vector<string> tensor_names;
        for (YAML::const_iterator tn_it = tensor_names_config.begin(); tn_it != tensor_names_config.end(); ++tn_it) {
          tensor_names.push_back(tn_it->first.as<string>());
        }
        inplace_alias_holder_[vit->first.as<string>()] = tensor_names;
      }
    }
  }
}

YAML::Node ActivationDAG::DumpConfig() {
  YAML::Node dag_node;
  LOG_IF(WARNING, operators_.size() == 0) << "activation DAG has no operators!";
  YAML::Node operators_node;
  for (auto op : operators_) {
    operators_node[op->name()] = op->DumpConfig();
  }
  dag_node["ActivationOperator"] = operators_node;
  // inplaced tensors info
  if (!inplace_alias_holder_.empty()) {
    YAML::Node inplace_alias_holder_node;
    for (const auto& in_pair : inplace_alias_holder_) {
      YAML::Node in_t_node;
      for (const auto& t : in_pair.second) {
        in_t_node[t] = YAML::Null;
      }
      inplace_alias_holder_node[in_pair.first] = in_t_node;
    }
    dag_node["InplaceAliasHolder"] = inplace_alias_holder_node;
  }
  return dag_node;
}

void ActivationDAG::Load(const string& input_dir) {
  YAML::Node load_node = YAML::LoadFile(input_dir);
  LoadConfig(load_node["ActivationDAG"]);
}

void ActivationDAG::Dump(const string& output_dir) {
  YAML::Node dump_node;
  dump_node["ActivationDAG"] = DumpConfig();
  std::ofstream fout(output_dir);
  fout << dump_node << std::endl;
  fout.close();
}
}  // namespace executor
