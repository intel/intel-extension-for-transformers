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

#ifndef ENGINE_SPARSELIB_INCLUDE_ENGINE_HPP_
#define ENGINE_SPARSELIB_INCLUDE_ENGINE_HPP_
#include <vector>
#include <memory>

#include "impl_list_item.hpp"
#include "param_types.hpp"

namespace jd {
class memory_storage_t;
class stream_t;
class engine_t {
 public:
  engine_t(const engine_kind& engine_kind, const runtime_kind& runtime_kind)
      : engine_kind_(engine_kind), runtime_kind_(runtime_kind) {}
  virtual ~engine_t() {}

 public:
  const engine_kind& get_engine_kind() const { return engine_kind_; }
  const runtime_kind& get_runtime_kind() const { return runtime_kind_; }
  virtual const std::vector<impl_list_item_t>* get_implementation_list(const operator_desc& op_desc) const = 0;
  virtual bool create_kernel(const operator_desc&, std::shared_ptr<kernel_t>&, const stream_t*) const = 0;  // NOLINT
  virtual bool create_stream(stream_t**) const = 0;
  virtual bool create_memory_storage(memory_storage_t**) const = 0;

 protected:
  engine_kind engine_kind_;
  runtime_kind runtime_kind_;
};
}  // namespace jd
#endif
