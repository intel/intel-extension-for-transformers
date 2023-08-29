//  Copyright (c) 2023 Intel Corporation
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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_CTX_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_CTX_HPP_
#include <memory>
#include <vector>

#include "include/engine.hpp"
#include "include/engine_factory.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "kernels/exposed_enum.hpp"
#include "operator_desc.hpp"

namespace jd {
// Convert rt_data to exec_ctx for old implementation.
inline std::unique_ptr<exec_context_t, void (*)(exec_context_t*)> get_mha_dense_ctx(
    const std::vector<const void*>& rt_data) {
  using io = exposed_enum::mha_dense::io;
  using io_src = exposed_enum::mha_dense_src::src;
  using io_dst = exposed_enum::mha_dense_dst::dst;
  using io_shape = exposed_enum::mha_dense_shape::shape;

  static engine_factory factory;
  static const engine_t* cpu_engine = factory.create(engine_kind::cpu, runtime_kind::undef);
  static const stream_t* stream = []() {
    stream_t* s = nullptr;
    cpu_engine->create_stream(&s);
    return s;
  }();
  const auto ctx = new exec_context_t(stream);

  const auto ptr2mem = [](const void* ptr) {
    memory_storage_t* mem;
    cpu_engine->create_memory_storage(&mem);
    mem->set_handle(const_cast<void*>(ptr));
    return mem;
  };
  const auto ptr2shape = [](const void* ptr) { return ptr ? reinterpret_cast<const int32_t*>(ptr)[0] : 0; };

  std::vector<memory_storage_t*> mem_src(io_src::SIZE), mem_dst(io_dst::SIZE);
  std::vector<dim_t> dynamic_shapes(io_shape::SIZE);
  mem_src[io_src::SRC_Q] = ptr2mem(rt_data[io::SRC_Q]);
  mem_src[io_src::SRC_K] = ptr2mem(rt_data[io::SRC_K]);
  mem_src[io_src::SRC_V] = ptr2mem(rt_data[io::SRC_V]);
  mem_src[io_src::MASK] = ptr2mem(rt_data[io::MASK]);
  mem_src[io_src::BINARY_ADD] = ptr2mem(rt_data[io::BINARY_ADD]);
  mem_src[io_src::ATT_SCALE] = ptr2mem(rt_data[io::ATT_SCALE]);
  mem_src[io_src::Q_SCALE] = ptr2mem(rt_data[io::Q_SCALE]);
  mem_src[io_src::Q_ZP] = ptr2mem(rt_data[io::Q_ZP]);
  mem_src[io_src::K_SCALE] = ptr2mem(rt_data[io::K_SCALE]);
  mem_src[io_src::K_ZP] = ptr2mem(rt_data[io::K_ZP]);
  mem_src[io_src::V_SCALE] = ptr2mem(rt_data[io::V_SCALE]);
  mem_src[io_src::V_ZP] = ptr2mem(rt_data[io::V_ZP]);
  mem_src[io_src::SRC_DST_SCALE] = ptr2mem(rt_data[io::SRC_DST_SCALE]);
  mem_src[io_src::SRC_DST_ZP] = ptr2mem(rt_data[io::SRC_DST_ZP]);
  mem_dst[io_dst::DST] = ptr2mem(rt_data[io::DST]);
  mem_dst[io_dst::DST_SCALE] = ptr2mem(rt_data[io::DST_SCALE]);
  mem_dst[io_dst::DST_ZP] = ptr2mem(rt_data[io::DST_ZP]);
  dynamic_shapes[io_shape::BATCH_SIZE] = rt_data.size() > io::BATCH_SIZE ? ptr2shape(rt_data[io::BATCH_SIZE]) : 0;
  dynamic_shapes[io_shape::HEAD_NUM] = rt_data.size() > io::HEAD_NUM ? ptr2shape(rt_data[io::HEAD_NUM]) : 0;
  dynamic_shapes[io_shape::HEAD_SIZE] = rt_data.size() > io::HEAD_SIZE ? ptr2shape(rt_data[io::HEAD_SIZE]) : 0;
  dynamic_shapes[io_shape::M] = rt_data.size() > io::M ? ptr2shape(rt_data[io::M]) : 0;
  dynamic_shapes[io_shape::N] = rt_data.size() > io::N ? ptr2shape(rt_data[io::N]) : 0;

  ctx->set_outputs(mem_dst);
  ctx->set_inputs(mem_src);
  ctx->set_dynamic_shape(dynamic_shapes);
  ctx->set_workspace(ptr2mem(rt_data[io::WORKSPACE]));
  return {
      ctx,
      [](exec_context_t* p) {
        for (auto mem : p->inputs()) delete mem;
        for (auto mem : p->outputs()) delete mem;
        delete p->workspace();
        delete p;
      },
  };
}
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_CTX_HPP_
