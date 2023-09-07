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

#include "interface.hpp"
#include "singleton.hpp"
#include "engine_factory.hpp"
#include "src/utils.hpp"

namespace jd {
kernel_desc_proxy::kernel_desc_proxy(const operator_desc& op_desc) {
  std::shared_ptr<const kernel_desc_t> result = nullptr;
  auto status = create_proxy_object(result, op_desc);
  if (!status) SPARSE_LOG(ERROR) << "Found no kernel_desc supported" << std::endl;
  reset_sp(result);
}

bool kernel_desc_proxy::create_proxy_object(std::shared_ptr<const kernel_desc_t>& result_ref,
                                            const operator_desc& op_desc) {
  // Step 1: Get the pd (or kernel_desc_t) if it's in the cache.
  kernel_cache* global_primitive_cache = Singleton<kernel_cache>::GetInstance();
  std::shared_ptr<const kernel_desc_t> candidate_kd = global_primitive_cache->get_kd(op_desc);
  if (candidate_kd != nullptr) {
    result_ref = candidate_kd;
    return true;
  }

  // Step 2.1: get impl_list_
  const auto& eng_kind = op_desc.engine_kind();
  const auto& runtime_kind = op_desc.runtime_kind();
  const engine_t* eng = Singleton<engine_factory>::GetInstance()->create(eng_kind, runtime_kind);
  if (eng == nullptr) {
    SPARSE_LOG(ERROR) << "Found no engine_t supported" << std::endl;
    return false;
  }
  impl_list_ = eng->get_implementation_list(op_desc);
  if (impl_list_ == nullptr) {
    return false;
  }
  // Step 2.2: Get the first && success object in impl_list_.
  auto& impl_list = (*impl_list_);
  for (auto& impl : impl_list) {
    candidate_kd = nullptr;
    auto status = impl(candidate_kd, op_desc);  // kd->create() + kd->init()
    if (status) {
      result_ref = candidate_kd;
      return true;
    }
  }
  return false;
}

kernel_proxy::kernel_proxy(const kernel_desc_proxy& kdp) {
  std::shared_ptr<const kernel_t> result = nullptr;
  auto status = create_proxy_object(result, kdp.get_sp());
  if (!status) SPARSE_LOG(ERROR) << "Found no kernel supported" << std::endl;
  reset_sp(result);
}

bool kernel_proxy::create_proxy_object(std::shared_ptr<const kernel_t>& result_ref,
                                       const std::shared_ptr<const kernel_desc_t>& kd) {
  kernel_cache* global_primitive_cache = Singleton<kernel_cache>::GetInstance();
  const auto& callback = std::bind(&kernel_desc_t::create_primitive, kd, std::placeholders::_1,
                                   kd);  // k_t->create() + k_t->init()
  std::shared_ptr<const kernel_t> value = global_primitive_cache->find_or_construct(kd->get_operator_desc(), callback);
  if (value == nullptr) {
    return false;
  }
  result_ref = value;
  return true;
}

size_t kernel_proxy::get_workspace_size() const { return get_sp()->get_workspace_size(); }

namespace {
// Helper function to implement execute with rt_data & ctx at the same time
template <typename T>
inline void execute_(const std::shared_ptr<const jd::kernel_t> sp, const T& data) {
  bool status = false;
#ifdef SPARSE_LIB_USE_VTUNE
  auto vtune_wrapper = vtune_wrapper_t();
  if (get_vtune()) {
    vtune_wrapper.profiling_begin(get_sp()->kd()->info());
  }
#endif
  if (get_verbose()) {
    double start_ms = get_msec();
    status = sp->execute(data);
    double duration_ms = get_msec() - start_ms;
    std::string stamp;
    if (get_verbose_timestamp()) stamp = "," + std::to_string(start_ms);

    printf("sparselib_verbose%s,exec,%s,%g\n", stamp.c_str(), sp->kd()->info(), duration_ms);
    fflush(stdout);
  } else {
    status = sp->execute(data);
  }
#ifdef SPARSE_LIB_USE_VTUNE
  if (get_vtune()) {
    vtune_wrapper.profiling_end();
  }
#endif
  if (!status) SPARSE_LOG(ERROR) << "Execution failed" << std::endl;
  return;
}
}  // namespace

void kernel_proxy::execute(const std::vector<const void*>& rt_data) const { execute_(get_sp(), rt_data); }
void kernel_proxy::execute(const exec_context_t& ctx) const { execute_(get_sp(), ctx); }
}  // namespace jd
