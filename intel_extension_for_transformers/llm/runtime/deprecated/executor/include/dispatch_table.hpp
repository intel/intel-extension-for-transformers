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

#include <sys/stat.h>
#include <glog/logging.h>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <functional>
#include <utility>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif

#include "conf.hpp"
#include "operator.hpp"
#include "memory_allocator.hpp"

namespace executor {
// shared with multi-process
namespace SharedTable {
  using Segment = ipc::managed_shared_memory;
  using Manager = Segment::segment_manager;
  template <typename T>
  using Alloc = ipc::allocator<T, Manager>;
  template <typename K, typename V, typename KH = boost::hash<K>, typename KEq = std::equal_to<K>>
  using HashMap = boost::unordered_map<K, V, KH, KEq, Alloc<std::pair<const K, V>>>;
  template <typename T>
  using Vector = ipc::vector<T, Alloc<T>>;
  using String = ipc::basic_string<char, std::char_traits<char>, Alloc<char>>;
}  // namespace SharedTable

class DispatchTable {
 public:
  // The first element is op_type (such as InnerProduct) and the internal map store the
  // hash key (combine shape, dtype, etc...) and kernel config
  typedef SharedTable::HashMap<size_t, SharedTable::Vector<SharedTable::String>> MappedType;
  typedef SharedTable::HashMap<SharedTable::String, MappedType> DispatchMap;

  // load dispatch table
  static void Load(const std::string& root = "") {
    SharedTable::Segment utils_shm(ipc::open_or_create, "UtilsShm", 1024);
    ipc::interprocess_mutex* mtx_read = utils_shm.find_or_construct<ipc::interprocess_mutex>("mtx_read")();
    mtx_read->lock();
    int* num_read = utils_shm.find_or_construct<int>("num_read")(0);
    (*num_read)++;
    if (*num_read == 1) {
      if (root != "") {
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
          std::string op_type;
          size_t hash_key;
          std::string kernel_name;
          std::vector<std::string> kernel_config;
          f_table >> op_type >> hash_key >> kernel_name;
          kernel_config.push_back(kernel_name);
          if (dispatch_kernel_config.count(op_type + "_to_" + kernel_name) > 0) {
            std::vector<std::string> extra_config_list = dispatch_kernel_config[op_type + "_to_" + kernel_name];
            std::string tmp;
            for (int i = 0; i < extra_config_list.size(); ++i) {
              f_table >> tmp;
              kernel_config.push_back(tmp);
            }
          }
          Insert(op_type, hash_key, kernel_config);
        }
        f_table.close();
      } else {
        LOG(FATAL) << "Please supply right dispatch table root rather than empty path...";
      }
    }
    mtx_read->unlock();
  }

  // save dispatch table
  static void Save(const std::string& root = "") {
    SharedTable::Segment utils_shm(ipc::open_or_create, "UtilsShm", 1024);
    ipc::interprocess_mutex* mtx_save = utils_shm.find_or_construct<ipc::interprocess_mutex>("mtx_save")();
    mtx_save->lock();
    if (root == "") LOG(FATAL) << "Please supply dispatch table saving path...";
    int index = root.find_last_of("/");
    std::string folder_root = root.substr(0, index);
    #ifdef _WIN32
    _mkdir(folder_root.c_str());
    #else
    if (access(folder_root.c_str(), F_OK) == -1) mkdir(folder_root.c_str(), ACCESSPERMS);
    #endif
    auto shm_handle = GetTableHandle();
    auto& TableShm = OpenShm();
    auto& d_table = *(static_cast<DispatchMap*>(TableShm.get_address_from_handle(shm_handle)));
    std::ofstream f_table;
    // rewrite the dispatch table file if it exists
    f_table.open(root, std::ios::out | std::ios::trunc);
    size_t num_lines = 0;
    for (const auto& d_pair : d_table) num_lines += d_pair.second.size();
    f_table << num_lines << "\n";
    for (const auto& d_pair : d_table) {
      auto op_type = d_pair.first;
      auto kernel_map = d_pair.second;
      for (const auto& k_pair : kernel_map) {
        f_table << op_type << " " << k_pair.first << " ";
        auto kernel_config = k_pair.second;
        std::string type_to_kernel = to_string(op_type + "_to_" + kernel_config[0]);
        if (dispatch_kernel_config.count(type_to_kernel) >0 && dispatch_kernel_config[type_to_kernel].size() \
            != (kernel_config.size() -1)) LOG(FATAL) << op_type << " has wrong dispatch kernel config...";
        for (const auto& tmp : kernel_config) f_table << tmp << " ";
        f_table << "\n";
      }
    }
    f_table.close();
    mtx_save->unlock();
  }

  // insert pair into dispatch table
  static void Insert(const std::string& op_type, const size_t& hash_key,
                     const std::vector<std::string>& kernel_config) {
    SharedTable::Segment utils_shm(ipc::open_or_create, "UtilsShm", 1024);
    ipc::interprocess_mutex* mtx_insert = utils_shm.find_or_construct<ipc::interprocess_mutex>("mtx_insert")();
    mtx_insert->lock();
    auto shm_handle = GetTableHandle();
    auto& TableShm = OpenShm();
    auto& d_table = *(static_cast<DispatchMap*>(TableShm.get_address_from_handle(shm_handle)));
    auto shm_op_type = to_shm_string(op_type);
    if (d_table.count(shm_op_type) == 0) {
      auto shm_val_map = MappedType(TableShm.get_segment_manager());
      shm_val_map.emplace(hash_key, to_shm_str_vector(kernel_config));
      d_table.emplace(shm_op_type, shm_val_map);
    } else {
      auto& shm_val_map = d_table.find(shm_op_type)->second;
      if (shm_val_map.count(hash_key) == 0) {
        shm_val_map.emplace(hash_key, to_shm_str_vector(kernel_config));
      } else {
        shm_val_map.erase(hash_key);
        shm_val_map.emplace(hash_key, to_shm_str_vector(kernel_config));
      }
    }
    mtx_insert->unlock();
  }

  // find kernel_config from dispatch table
  static std::vector<std::string> Find(const std::string& op_type, const size_t& hash_key) {
    auto shm_handle = GetTableHandle();
    auto& TableShm = OpenShm();
    auto& d_table = *(static_cast<DispatchMap*>(TableShm.get_address_from_handle(shm_handle)));
    std::vector<std::string> kernel_config;
    auto shm_op_type = to_shm_string(op_type);
    if (d_table.count(shm_op_type) > 0 && d_table.find(shm_op_type)->second.count(hash_key) > 0) {
      kernel_config = to_str_vector(d_table.find(shm_op_type)->second.find(hash_key)->second);
    }
    return kernel_config;
  }

  static void Clear() {
    SharedTable::Segment utils_shm(ipc::open_or_create, "UtilsShm", 1024);
    ipc::interprocess_mutex* mtx_clear = utils_shm.find_or_construct<ipc::interprocess_mutex>("mtx_clear")();
    mtx_clear->lock();
    auto shm_handle = GetTableHandle();
    auto& TableShm = OpenShm();
    auto& d_table = *(static_cast<DispatchMap*>(TableShm.get_address_from_handle(shm_handle)));
    d_table.clear();
    mtx_clear->unlock();
  }

  static int Size() {
    auto shm_handle = GetTableHandle();
    auto& TableShm = OpenShm();
    auto& d_table = *(static_cast<DispatchMap*>(TableShm.get_address_from_handle(shm_handle)));
    int size = 0;
    for (const auto& d_pair : d_table) size += d_pair.second.size();
    return size;
  }

  static void CleanShm(const string& state = "begin") {
    SharedTable::Segment utils_shm(ipc::open_or_create, "UtilsShm", 1024);
    ipc::interprocess_mutex* mtx_clean = utils_shm.find_or_construct<ipc::interprocess_mutex>("mtx_clean")();
    mtx_clean->lock();
    if (state == "begin") {
      int* ins_begin = utils_shm.find_or_construct<int>("ins_begin")(0);
      (*ins_begin)++;
      DLOG(INFO) << "Constructing dispatch table singleton in instance " << *ins_begin;
      if (*ins_begin == 1) ipc::shared_memory_object::remove("DispatchTableSegment");
      if (*ins_begin == MemoryAllocator::InstNum()) utils_shm.destroy_ptr(ins_begin);
    } else if (state == "end") {
      int* ins_end = utils_shm.find_or_construct<int>("ins_end")(0);
      (*ins_end)++;
      if (*ins_end == MemoryAllocator::InstNum()) {
        DLOG(INFO) << "Deconstructing shared dispatch table memory object in instance " << *ins_end \
                  << ", the total instance num is " << MemoryAllocator::InstNum();
        ipc::shared_memory_object::remove("DispatchTableSegment");
        ipc::shared_memory_object::remove("UtilsShm");
      }
    } else {
      LOG(FATAL) << "Only support begin and end state value...";
    }
    mtx_clean->unlock();
  }

 private:
  // DispatchTable should never be instantiated - everything is done with its
  // static variables.
  DispatchTable() {
    CleanShm("begin");
  }

  ~DispatchTable() {
    CleanShm("end");
  }

  static ipc::managed_shared_memory::handle_t GetTableHandle() {
    static DispatchTable instance;
    static auto TableShm = SharedTable::Segment(ipc::open_or_create, "DispatchTableSegment", 1024 * 1024 * 100);
    void* dispatch_map_ = TableShm.find_or_construct<DispatchMap>("dispatch_map")(TableShm.get_segment_manager());
    const auto& shm_handle = TableShm.get_handle_from_address(dispatch_map_);
    return shm_handle;
  }

  static SharedTable::Segment& OpenShm(char* shm_name = "DispatchTableSegment") {
    static SharedTable::Segment shm_obj(ipc::open_only, shm_name);
    return shm_obj;
  }

  static SharedTable::String to_shm_string(const std::string& val) {
    auto& TableShm = OpenShm();
    std::ostringstream ostr;
    ostr << val;
    return SharedTable::String(ostr.str().c_str(), TableShm.get_segment_manager());
  }

  static SharedTable::Vector<SharedTable::String> to_shm_str_vector(const std::vector<std::string>& vec) {
    auto& TableShm = OpenShm();
    auto shm_str_vector = SharedTable::Vector<SharedTable::String>(TableShm.get_segment_manager());
    for (const auto& v : vec) shm_str_vector.push_back(to_shm_string(v));
    return shm_str_vector;
  }

  static std::string to_string(const SharedTable::String& val) { return std::string(val.begin(), val.end()); }

  static std::vector<std::string> to_str_vector(const SharedTable::Vector<SharedTable::String>& vec) {
    std::vector<std::string> str_vector;
    for (const auto& v : vec) str_vector.push_back(to_string(v));
    return str_vector;
  }
};
}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_DISPATCH_TABLE_HPP_
