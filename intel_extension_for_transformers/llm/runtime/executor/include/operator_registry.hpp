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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATOR_REGISTRY_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATOR_REGISTRY_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "conf.hpp"
#include "operator.hpp"

namespace executor {

class Operator;

class OperatorRegistry {
 public:
  typedef shared_ptr<Operator> (*Creator)(const shared_ptr<OperatorConfig>&);
  // op_type-kernel_name-kernel
  // register kernel class
  typedef std::map<string, std::map<string, Creator>> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const string& type, const string& kernel_name, Creator creator) {
    CreatorRegistry& registry = Registry();
    DLOG(INFO) << "Gonna register " << type << ", dispatch kernel " << kernel_name
              <<  "....";
    if (registry.count(type) > 0) {
      CHECK_EQ(registry[type].count(kernel_name), 0)
              << "Operator type " << type << ", dispatch kernel " << kernel_name
              << " has already beed registered.";
    }
    registry[type][kernel_name] = creator;
  }

  // Get a operator using a OperatorConfig.
  static shared_ptr<Operator> CreateOperator(const shared_ptr<OperatorConfig>& conf, const string& kernel_name) {
    const string& type = conf->type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown operator type: " << type
                                      << " (known types: " << OperatorTypeListString() << ").";
    CHECK_EQ(registry[type].count(kernel_name), 1) << "Unknown dispatch kernel: " << kernel_name
                                      << " in operator type: " << type;
    return registry[type][kernel_name](conf);
  }

  static vector<string> OperatorTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> operator_types;
    for (typename CreatorRegistry::iterator iter = registry.begin(); iter != registry.end(); ++iter) {
      operator_types.push_back(iter->first);
    }
    return operator_types;
  }

  static string OperatorTypeListString() {
    vector<string> operator_types = OperatorTypeList();
    string operator_types_str;
    for (vector<string>::iterator iter = operator_types.begin(); iter != operator_types.end(); ++iter) {
      if (iter != operator_types.begin()) {
        operator_types_str += ", ";
      }
      operator_types_str += *iter;
    }
    return operator_types_str;
  }

 private:
  // Operator registry should never be instantiated - everything is done with its
  // static variables.
  OperatorRegistry() {}
};

class OperatorRegisterer {
 public:
  OperatorRegisterer(const string& type, const string& kernel_name,
                    shared_ptr<Operator> (*creator)(const shared_ptr<OperatorConfig>&)) {
    OperatorRegistry::AddCreator(type, kernel_name, creator);
  }
};

#define REGISTER_OPERATOR_CREATOR(type, kernel_name, creator)                      \
  static OperatorRegisterer g_creator_##type##kernel_name(#type, #kernel_name, creator);

#define REGISTER_KERNEL_CLASS(type, kernel_name)                                   \
  shared_ptr<Operator> Creator_##type##kernel_name##Operator(const shared_ptr<OperatorConfig>& conf) \
  {                                                                                \
    return shared_ptr<Operator>(new kernel_name##Operator(conf));                  \
  }                                                                                \
  REGISTER_OPERATOR_CREATOR(type, kernel_name, Creator_##type##kernel_name##Operator)

#define REGISTER_OPERATOR_CLASS(type)                                              \
  shared_ptr<Operator> Creator_##type##type##Operator(const shared_ptr<OperatorConfig>& conf)  \
  {                                                                                \
    return shared_ptr<Operator> (new type##Operator(conf));                        \
  }                                                                                \
  REGISTER_OPERATOR_CREATOR(type, type, Creator_##type##type##Operator)

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATOR_REGISTRY_HPP_
