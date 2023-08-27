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
#include "engine_factory.hpp"

#include "src/cpu/engine/cpu_engine.hpp"
#include "src/utils.hpp"
#ifdef SPARSELIB_GPU
#include "src/gpu/engine/gpu_ocl_engine.hpp"
#endif

namespace jd {
const engine_t* engine_factory::create(const engine_kind& engine_kind, const runtime_kind& runtime_kind) {
  const auto& it = mp_.find(engine_kind);
  if (it != mp_.end()) {
    return (*(it->second))(runtime_kind);
  } else {
    return nullptr;
  }
}

void engine_factory::register_class() {
  if (!mp_.count(engine_kind::cpu)) {
    mp_[engine_kind::cpu] = &engine_factory::create_cpu_engine;
  }
  if (!mp_.count(engine_kind::gpu)) {
    mp_[engine_kind::gpu] = &engine_factory::create_gpu_engine;
  }
}

const engine_t* engine_factory::create_cpu_engine(const runtime_kind&) {
  static std::shared_ptr<cpu_engine_t> obj = std::make_shared<cpu_engine_t>();
  return reinterpret_cast<engine_t*>(obj.get());
}

const engine_t* engine_factory::create_gpu_engine(const runtime_kind& runtime_kind) {
  switch (runtime_kind) {
    case runtime_kind::opencl:
#ifdef SPARSELIB_GPU
      static std::shared_ptr<const gpu_ocl_engine_t> obj = std::make_shared<const gpu_ocl_engine_t>();
      return reinterpret_cast<const engine_t*>(obj.get());
#endif
    case runtime_kind::sycl:

    default:
      SPARSE_LOG(FATAL) << "Create engine error";
      return nullptr;
  }
}
engine_factory::engine_factory() { register_class(); }
}  // namespace jd
