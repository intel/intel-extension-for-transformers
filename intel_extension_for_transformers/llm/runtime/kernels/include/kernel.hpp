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
#ifndef ENGINE_SPARSELIB_INCLUDE_KERNEL_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNEL_HPP_
#include <memory>
#include <vector>

#include "kernel_desc.hpp"
#include "memory_storage.hpp"

namespace jd {
/**
 * @brief kernel/primitive implementation real class.
 */
enum mem_type_t { host_mem, ocl_mem, sycl_mem };
class stream_t;
struct exec_context_t {
  explicit exec_context_t(const stream_t* stream) : stream_(stream) {}
  const stream_t* get_stream() const { return stream_; }
  void add_input(memory_storage_t* input) { inputs_.push_back(input); }
  void set_input(size_t index, memory_storage_t* input) {
    if (index < inputs_.size()) {
      inputs_[index] = input;
    }
  }
  void set_inputs(const std::vector<memory_storage_t*>& inputs) { inputs_ = inputs; }
  void add_output(memory_storage_t* output) { outputs_.push_back(output); }
  void set_output(size_t index, memory_storage_t* output) {
    if (index < outputs_.size()) {
      outputs_[index] = output;
    }
  }
  void set_outputs(const std::vector<memory_storage_t*>& outputs) { outputs_ = outputs; }

  std::vector<memory_storage_t*> inputs() const { return inputs_; }
  memory_storage_t* input(size_t index) const {
    if (index < inputs_.size()) {
      return inputs_[index];
    }
    return nullptr;
  }
  std::vector<memory_storage_t*> outputs() const { return outputs_; }
  memory_storage_t* output(size_t index) const {
    if (index < outputs_.size()) {
      return outputs_[index];
    }
    return nullptr;
  }

  void set_workspace(memory_storage_t* workspace) { workspace_ = workspace; }
  memory_storage_t* workspace() const { return workspace_; }

  mem_type_t mem_type() const { return mem_type_; }

  void set_dynamic_shape(const std::vector<dim_t> dynamic_shape) { dynamic_shape_ = dynamic_shape; }
  const std::vector<dim_t> get_dynamic_shape() const { return dynamic_shape_; }

 private:
  mem_type_t mem_type_;
  const stream_t* stream_;
  std::vector<memory_storage_t*> inputs_;
  std::vector<memory_storage_t*> outputs_;
  memory_storage_t* workspace_;
  std::vector<dim_t> dynamic_shape_;
};

class SPARSE_TEST_API_ kernel_t {
 public:
  explicit kernel_t(const std::shared_ptr<const kernel_desc_t>& kd);
  virtual ~kernel_t() {}
  // Delete move constructor and move operator
  kernel_t(kernel_t&& other) = delete;
  kernel_t& operator=(kernel_t&& other) = delete;
  // // Delete copy constructor and copy operator
  kernel_t(const kernel_t& other) = delete;
  kernel_t& operator=(const kernel_t& other) = delete;

 public:
  // Self-created API, provided for external users to call.
  template <typename derived_k_t, typename derived_kd_t>
  static bool create(std::shared_ptr<const kernel_t>& k_ref,  // NOLINT
                     const std::shared_ptr<const kernel_desc_t>& kd) {
    const auto& derived_kd_temp = std::dynamic_pointer_cast<const derived_kd_t>(kd);
    std::shared_ptr<derived_k_t> prim = std::make_shared<derived_k_t>(derived_kd_temp);
    if (prim == nullptr) {
      return false;
    }
    auto status = prim->init();
    if (!status) {
      prim.reset();  // prim failed and destroy.
      return false;
    }
    k_ref = prim;
    return true;
  }
  // init kernel_t
  virtual bool init() = 0;
  virtual bool init(const exec_context_t&) { return true; }

  virtual bool execute(const std::vector<const void*>&) const { return true; }
  virtual bool execute(const exec_context_t&) const { return true; }
  virtual bool execute() const { return false; }
  virtual size_t get_workspace_size() const { return 0; }

 public:
  const std::shared_ptr<const kernel_desc_t>& kd() const { return kd_; }

 protected:
  // kernel_desc_t has no cache management. So use shared_ptr to cache and
  // destruct automatically.
  std::shared_ptr<const kernel_desc_t> kd_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNEL_HPP_
