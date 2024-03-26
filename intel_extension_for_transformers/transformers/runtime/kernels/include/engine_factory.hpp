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

#ifndef ENGINE_SPARSELIB_SRC_ENGINE_FACTORY_HPP_
#define ENGINE_SPARSELIB_SRC_ENGINE_FACTORY_HPP_
#include <memory>
#include <unordered_map>

#include "common.h"
#include "param_types.hpp"

namespace jd {
class engine_t;
class SPARSE_API_ engine_factory {
 public:
  engine_factory();
  const engine_t* create(const engine_kind& engine_kind, const runtime_kind& runtime_kind);

 private:
  void register_class();
  static const engine_t* create_cpu_engine(const runtime_kind& runtime_kind);
  static const engine_t* create_gpu_engine(const runtime_kind& runtime_kind);

 private:
  using create_fptr = const engine_t* (*)(const runtime_kind&);
  std::unordered_map<engine_kind, create_fptr> mp_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_ENGINE_FACTORY_HPP_
