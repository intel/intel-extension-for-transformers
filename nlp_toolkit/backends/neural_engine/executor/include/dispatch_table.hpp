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

#ifndef ENGINE_EXECUTOR_INCLUDE_DISPATCH_TABLE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_DISPATCH_TABLE_HPP_

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unistd.h>
#include <sys/stat.h>
#include <glog/logging.h>

#include "conf.hpp"
#include "operator.hpp"

namespace executor {

class DispatchTable {
 public:
  // The first element is op_type (such as InnerProduct) and the internal map store the
  // hash key (combine shape, dtype, etc...) and kernel config
  typedef std::unordered_map<string, std::unordered_map<size_t, vector<string>>> DispatchMap;
 
  // load dispatch table
  static void Load(const string& root="") {
    if (root != "") {
      DispatchMap& d_table = DispatchTable::GetTable();
      std::ifstream f_table;
      f_table.open(root, std::ios::in);
      if (!f_table.is_open()) LOG(FATAL) << "Load dispatch table file failed, please check the file...";
      // The dispatch file contains lines number (first row) and the rest of lines contains 
      // op_type (string) - hash key (size_t) - kernel_config(kernel_name (string) - input_shape
      // - ... <if has specific dispatch kernel config except kernel name>)
      // example:
      // 1
      // InnerProduct 3025159985633461085 Convolution 4,1,40,1024
      size_t num_lines;
      f_table >> num_lines;
      for (size_t i = 0; i < num_lines; ++i) {
        string op_type;
        size_t hash_key;
        string kernel_name;
        vector<string> kernel_config;
        f_table >> op_type >> hash_key >> kernel_name;
        kernel_config.push_back(kernel_name);
        if (dispatch_kernel_config.count(op_type + "_to_" + kernel_name) > 0) {
          vector<string> extra_config_list = dispatch_kernel_config[op_type + "_to_" + kernel_name];
          string tmp;
          for (int i = 0; i < extra_config_list.size(); ++i){
            f_table >> tmp;
            kernel_config.push_back(tmp);
          }
        }
        d_table[op_type][hash_key] = kernel_config;
      }
      f_table.close();
    } else {
      LOG(FATAL) << "Please supply right dispatch table root rather than empty path...";
    }
  }

  // save dispatch table
  static void Save(const string& root="") {
    if (root == "") LOG(FATAL) << "Please supply dispatch table saving path...";
    // only for Linux system now
    int index = root.find_last_of("/");
    string folder_root = root.substr(0, index);
    if (access(folder_root.c_str(), F_OK) == -1) mkdir(folder_root.c_str(), ACCESSPERMS);
    DispatchMap& d_table = DispatchTable::GetTable();
    std::ofstream f_table;
    // rewrite the dispatch table file if it exists
    f_table.open(root, std::ios::out | std::ios::trunc);
    size_t num_lines = 0;
    for (const auto& d_pair : d_table) num_lines += d_pair.second.size();
    f_table << num_lines << "\n";
    for (const auto& d_pair : d_table) {
      string op_type = d_pair.first;
      auto kernel_map = d_pair.second;
      for (const auto& k_pair : kernel_map) {
        f_table << op_type << " " << k_pair.first << " ";
        vector<string> kernel_config = k_pair.second;
        string type_to_kernel = op_type + "_to_" + kernel_config[0];
        if (dispatch_kernel_config.count(type_to_kernel) >0 && dispatch_kernel_config[type_to_kernel].size() \
            != (kernel_config.size() -1)) LOG(FATAL) << op_type << " has wrong dispatch kernel config...";
        for (const auto& tmp : kernel_config) f_table << tmp << " ";
        f_table << "\n"; 
      }
    }
    f_table.close();
  }

  // insert pair into dispatch table
  static void Insert(const string& op_type, const size_t& hash_key, const vector<string>& kernel_config) {
    DispatchMap& d_table = DispatchTable::GetTable();
    d_table[op_type][hash_key] = kernel_config;
  }

  // find kernel_config from dispatch table
  static vector<string> Find(const string& op_type, const size_t& hash_key) {
    DispatchMap& d_table = DispatchTable::GetTable();
    vector<string> kernel_config;
    if (d_table.count(op_type) > 0 && d_table[op_type].count(hash_key) > 0) {
      kernel_config = d_table[op_type][hash_key];
    }
    return kernel_config;
  }
  
  static void Clear() {
    DispatchMap& d_table =  DispatchTable::GetTable();
    d_table.clear();
  }

  static int Size() {
    DispatchMap& d_table =  DispatchTable::GetTable();
    int size = 0;
    for (const auto& d_pair : d_table) size += d_pair.second.size();
    return size;
  }
  
 private:
  // DispatchTable should never be instantiated - everything is done with its
  // static variables.
  DispatchTable() {}
  ~DispatchTable() {}
  
  static DispatchMap& GetTable() {
    static thread_local DispatchMap dispatch_map_;
    return dispatch_map_;
  }
};
}  //namespace executor

#endif  //ENGINE_EXECUTOR_INCLUDE_DISPATCH_TABLE_HPP_