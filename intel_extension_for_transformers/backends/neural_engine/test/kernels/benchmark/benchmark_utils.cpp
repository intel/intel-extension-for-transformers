//  Copyright (c) 2022 Intel Corporation
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

#include "benchmark_utils.hpp"

#include <functional>

#include "common_utils.hpp"
#include "data_type/data_types.hpp"
#include "engine.hpp"
#include "engine_factory.hpp"

// Internal Control Variables
int benchmark_iter = 100;
bool benchmark_refresh = true;
namespace {
void read_benchmark_env() {
  const char* input_benchmark_iter = std::getenv("BENCHMARK_ITER");
  if (input_benchmark_iter != nullptr) {
    benchmark_iter = atoi(input_benchmark_iter);
    if (benchmark_iter == 0) {
      LOG(WARNING) << "BENCHMARK_ITER is 0! Please ensure you set this variable to an integer\n";
    }
  }
  const char* input_benchmark_refresh = std::getenv("BENCHMARK_NO_REFRESH");
  if (input_benchmark_refresh != nullptr) {
    if (strcmp(input_benchmark_refresh, "1") == 0) {
      benchmark_refresh = false;
    }
  }
}

using bench::aligned_allocator_t;
using bench::init_vector;
static jd::engine_factory factory;
static const jd::engine_t* cpu_engine = factory.create(jd::engine_kind::cpu, jd::runtime_kind::undef);

auto create_cpu_memory_storage(void* ptr) {
  jd::memory_storage_t* mem;
  cpu_engine->create_memory_storage(&mem);
  if (ptr) mem->set_handle(ptr);
  return mem;
}

template <typename T>
double exec_time(std::shared_ptr<jd::kernel_proxy> kp, const T& rt_data) {
  auto begin = std::chrono::high_resolution_clock::now();
  kp->execute(rt_data);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

bool alloc_new_mem(const std::vector<jd::tensor_desc>& ts_descs, std::vector<const void*>* rt_data_pointer,
                   std::vector<void*>* new_data_pointer, const std::vector<int>& idx,
                   const std::vector<int>& desc_idx) {
  SPARSE_DLOG_IF(FATAL, idx.size() != desc_idx.size()) << "desc_idx should have the same length as idx.";
  std::vector<const void*>& rt_data = *rt_data_pointer;
  std::vector<void*>& new_data = *new_data_pointer;
  for (size_t i = 0; i < idx.size(); ++i) {
    const auto& desc = ts_descs[desc_idx[i]];
    int elem_num = std::accumulate(desc.shape().begin(), desc.shape().end(), size_t{1}, std::multiplies<size_t>());
    int byte_size = elem_num * jd::type_size[desc.dtype()];
    void* new_mem = aligned_allocator_t<uint8_t, 64>::allocate(pad_to(byte_size, 64));
    SPARSE_LOG_IF(ERROR, !new_mem) << "malloc failed.";
    rt_data[idx[i]] = new_mem;
    new_data.emplace_back(new_mem);
  }
  return true;
}

void free_new_mem(std::vector<void*> new_data) {
  for (auto ptr : new_data) aligned_allocator_t<uint8_t, 64>::deallocate(ptr);
}

void refresh_data(const std::vector<jd::tensor_desc>& ts_descs, const float lo, const float hi,
                  std::vector<void*>* new_data_pointer, const std::vector<int>& desc_idx) {
  auto new_data = *new_data_pointer;
  for (size_t i = 0; i < desc_idx.size(); ++i) {
    int elem_num = std::accumulate(ts_descs[desc_idx[i]].shape().begin(), ts_descs[desc_idx[i]].shape().end(),
                                   size_t{1}, std::multiplies<size_t>());
    switch (ts_descs[desc_idx[i]].dtype()) {
      case jd::data_type::fp32:
        init_vector(static_cast<float*>(new_data[i]), elem_num, lo, hi, rand());
        break;
      case jd::data_type::s32:
        init_vector(static_cast<int32_t*>(new_data[i]), elem_num, lo, hi, rand());
        break;
      case jd::data_type::u8:
        init_vector(static_cast<uint8_t*>(new_data[i]), elem_num, lo, hi, rand());
        break;
      case jd::data_type::s8:
        init_vector(static_cast<int8_t*>(new_data[i]), elem_num, lo, hi, rand());
        break;
      case jd::data_type::bf16:
        init_vector(static_cast<jd::bfloat16_t*>(new_data[i]), elem_num, lo, hi, rand());
        break;
      default:
        break;
    }
  }
}
}  // namespace
namespace bench {
bench_res_t bench_op::run_bench(bench_mode mode) {
  bench_res_t res;
  kb->gen_case();
  try {
    kb->set_kernel_proxy();
    res = benchmarkOrExecute(mode);
  } catch (const std::exception& e) {
    res.stat = bench_status::fail;
    return res;
  }
  if (mode == bench_mode::acc) res.correct = kb->check_result();
  return res;
}

bench_res_t bench_op::benchmarkOrExecute(bench_mode mode) {
  const auto using_bench_data = kb->bench_data.op_desc.engine_kind() != jd::engine_kind::undef;
  auto& p = kb->args.first;
  auto& q = kb->args.second;
  bench_res_t res;
  // prepare workspace
  const auto workspace_idx = kb->get_workspace_idx();
  const auto workspace_size = kb->kp->get_workspace_size();
  std::shared_ptr<void> with_workspace;
  if (workspace_idx >= 0 && workspace_size > 0) {
    const auto workspace_p = aligned_allocator_t<char>::allocate(workspace_size);
    const auto workspace_q = aligned_allocator_t<char>::allocate(workspace_size);
    if (using_bench_data) {
      kb->bench_data.ctx_kern.set_workspace(create_cpu_memory_storage(workspace_p));
      kb->bench_data.ctx_ref.set_workspace(create_cpu_memory_storage(workspace_q));
    } else {
      p.rt_data[workspace_idx] = workspace_p;
      q.rt_data[workspace_idx] = workspace_q;
    }
    with_workspace = {
        nullptr,
        [using_bench_data, workspace_idx, workspace_p, workspace_q, &p, &q, this](...) {
          // free workspace memory
          aligned_allocator_t<char>::deallocate(workspace_p);
          aligned_allocator_t<char>::deallocate(workspace_q);
          if (using_bench_data) {
            delete kb->bench_data.ctx_kern.workspace();
            delete kb->bench_data.ctx_ref.workspace();
          } else {
            p.rt_data[workspace_idx] = nullptr;
            q.rt_data[workspace_idx] = nullptr;
          }
        },
    };
  }
  (using_bench_data) ? kb->kp->execute(kb->bench_data.ctx_kern) : kb->kp->execute(p.rt_data);
  if (mode == bench_mode::acc) {
    res.stat = bench_status::success;
    return res;
  }
  read_benchmark_env();
  // Use op_desc to get kernel kind and tensor shape
  const auto& op_desc = using_bench_data ? kb->bench_data.op_desc : kb->kp->get_sp()->kd()->get_operator_desc();
  const auto& ts_descs = op_desc.tensor_descs();
  double ns = 0.;
  if (using_bench_data) {
    std::vector<int> src_idx = kb->get_refresh_src_data_idx(), dst_idx = kb->get_refresh_dst_data_idx();
    std::vector<int> src_desc_idx = kb->get_refresh_src_desc_idx(), dst_desc_idx = kb->get_refresh_dst_desc_idx();
    const auto ctx = kb->bench_data.ctx_kern;
    std::vector<const void*> tmp_src_data(ctx.inputs().size(), nullptr), tmp_dst_data(ctx.outputs().size(), nullptr);
    std::vector<void*> new_src_data, new_dst_data;  // vector of new data pointers
    if (!alloc_new_mem(ts_descs, &tmp_src_data, &new_src_data, src_idx, src_desc_idx) ||
        !alloc_new_mem(ts_descs, &tmp_dst_data, &new_dst_data, dst_idx, dst_desc_idx)) {
      res.stat = bench_status::fail;
      return res;
    }
    std::vector<void*> old_src_data(ctx.inputs().size(), nullptr), old_dst_data(ctx.outputs().size(), nullptr);
    for (auto& idx : src_idx) {
      ctx.inputs()[idx]->get_handle(&old_src_data[idx]);
      ctx.inputs()[idx]->set_handle(const_cast<void*>(tmp_src_data[idx]));
    }
    for (auto& idx : dst_idx) {
      ctx.outputs()[idx]->get_handle(&old_dst_data[idx]);
      ctx.outputs()[idx]->set_handle(const_cast<void*>(tmp_dst_data[idx]));
    }
    for (int i = 0; i < benchmark_iter; ++i) {
      if (benchmark_refresh) {
        refresh_data(ts_descs, kb->ranges[0], kb->ranges[1], &new_src_data, src_desc_idx);
        refresh_data(ts_descs, kb->ranges[0], kb->ranges[1], &new_dst_data, dst_desc_idx);
      }
      ns += exec_time(kb->kp, ctx);
    }

    // free new memory
    free_new_mem(new_src_data);
    free_new_mem(new_dst_data);
    for (auto& idx : src_idx) ctx.inputs()[idx]->set_handle(const_cast<void*>(old_src_data[idx]));
    for (auto& idx : dst_idx) ctx.outputs()[idx]->set_handle(const_cast<void*>(old_dst_data[idx]));
  } else {
    // We may need to refresh some parts of runtime data, allocate new memory for them first
    std::vector<const void*> tmp_data(p.rt_data);
    std::vector<void*> new_data;
    std::vector<int> idx, raw_idx = kb->get_refresh_data_idx();
    for (auto i : raw_idx)
      if (i < static_cast<int>(ts_descs.size()) && ts_descs[i].size() != 0) idx.push_back(i);
    SPARSE_LOG_IF(FATAL, std::any_of(idx.begin(), idx.end(), [workspace_idx](int i) { return i == workspace_idx; }))
        << "workspace should not be refreshed!";
    if (!alloc_new_mem(ts_descs, &tmp_data, &new_data, idx, idx)) {
      res.stat = bench_status::fail;
      return res;
    }
    refresh_data(ts_descs, kb->ranges[0], kb->ranges[1], &new_data, idx);
    for (int i = 0; i < benchmark_iter; ++i) {
      if (benchmark_refresh) refresh_data(ts_descs, kb->ranges[0], kb->ranges[1], &new_data, idx);  // refresh data
      ns += exec_time(kb->kp, tmp_data);
    }
    // free new memory
    free_new_mem(new_data);
  }

  // get execution time and calculate GFLOPS
  ns = ns / benchmark_iter;
  res.ms = ns / 1e6;
  res.gflops = kb->calc_flop() / ns;
  res.stat = bench_status::success;
  return res;
}
}  // namespace bench
