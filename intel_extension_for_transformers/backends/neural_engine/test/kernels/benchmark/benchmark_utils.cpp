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
#include "utils.hpp"

// Internal Control Variables
int benchmark_iter = 100;
bool benchmark_refresh = true;

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

namespace jd {
using dt = jd::data_type;
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
  auto& p = kb->args.first;
  auto& q = kb->args.second;
  bench_res_t res;

  // prepare workspace
  const auto workspace_idx = kb->get_workspace_idx();
  const auto workspace_size = kb->kp->get_workspace_size();
  std::shared_ptr<void> with_workspace;
  if (workspace_idx >= 0 && workspace_size > 0) {
    p.rt_data[workspace_idx] = aligned_allocator_t<char>::allocate(workspace_size);
    q.rt_data[workspace_idx] = aligned_allocator_t<char>::allocate(workspace_size);
    with_workspace = {
        nullptr,
        [workspace_idx, &p, &q](...) {
          // free workspace memory
          aligned_allocator_t<char>::deallocate(const_cast<void*>(p.rt_data[workspace_idx]));
          aligned_allocator_t<char>::deallocate(const_cast<void*>(q.rt_data[workspace_idx]));
          p.rt_data[workspace_idx] = nullptr;
          q.rt_data[workspace_idx] = nullptr;
        },
    };
  }

  kb->kp->execute(p.rt_data);

  if (mode == bench_mode::acc) {
    res.stat = bench_status::success;
    return res;
  }

  read_benchmark_env();

  // Use op_desc to get kernel kind and tensor shape
  const auto& op_desc = kb->kp->get_sp()->kd()->get_operator_desc();
  const auto& ts_descs = op_desc.tensor_descs();

  // We may need to refresh some parts of runtime data, allocate new memory for them first
  std::vector<const void*> tmp_data(p.rt_data);
  std::vector<void*> new_data;
  std::vector<int> idx = kb->get_refresh_data_idx();
  SPARSE_LOG_IF(FATAL, std::any_of(idx.begin(), idx.end(), [workspace_idx](int i) { return i == workspace_idx; }))
      << "workspace must not be refreshed!";
  if (!alloc_new_mem(ts_descs, &tmp_data, &new_data, idx)) {
    res.stat = bench_status::fail;
    return res;
  }
  double ns = 0.0;
  for (int i = 0; i < benchmark_iter; ++i) {
    // refresh data
    if (benchmark_refresh) {
      refresh_data(&new_data, idx);
    }
    ns += exec_time(kb->kp, tmp_data);
  }

  // get execution time and calculate GFLOPS
  ns = ns / benchmark_iter;
  res.ms = ns / 1e6;
  res.gflops = kb->calc_flop() / ns;

  // free new memory
  free_new_mem(&new_data);
  res.stat = bench_status::success;
  return res;
}
void bench_op::refresh_data(std::vector<void*>* new_data_pointer, const std::vector<int>& idx) {
  auto new_data = *new_data_pointer;
  for (size_t i = 0; i < idx.size(); ++i) {
    int elem_num = std::accumulate(kb->ts_descs[idx[i]].shape().begin(), kb->ts_descs[idx[i]].shape().end(), size_t{1},
                                   std::multiplies<size_t>());
    switch (kb->ts_descs[idx[i]].dtype()) {
      case dt::fp32:
        init_vector(static_cast<float*>(new_data[i]), elem_num, kb->ranges[0], kb->ranges[1], rand());
        break;
      case dt::s32:
        init_vector(static_cast<int32_t*>(new_data[i]), elem_num, kb->ranges[0], kb->ranges[1], rand());
        break;
      case dt::u8:
        init_vector(static_cast<uint8_t*>(new_data[i]), elem_num, kb->ranges[0], kb->ranges[1], rand());
        break;
      case dt::s8:
        init_vector(static_cast<int8_t*>(new_data[i]), elem_num, kb->ranges[0], kb->ranges[1], rand());
        break;
      case dt::bf16:
        init_vector(static_cast<bfloat16_t*>(new_data[i]), elem_num, kb->ranges[0], kb->ranges[1], rand());
        break;
      default:
        break;
    }
  }
}
double bench_op::exec_time(std::shared_ptr<kernel_proxy> kp, const std::vector<const void*>& rt_data) {
  auto begin = std::chrono::high_resolution_clock::now();
  kp->execute(rt_data);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

bool bench_op::alloc_new_mem(const std::vector<tensor_desc>& ts_descs, std::vector<const void*>* rt_data_pointer,
                             std::vector<void*>* new_data_pointer, const std::vector<int>& idx) {
  std::vector<const void*>& rt_data = *rt_data_pointer;
  std::vector<void*>& new_data = *new_data_pointer;
  for (size_t i = 0; i < idx.size(); ++i) {
    int elem_num = std::accumulate(ts_descs[idx[i]].shape().begin(), ts_descs[idx[i]].shape().end(), size_t{1},
                                   std::multiplies<size_t>());
    int byte_size = elem_num * type_size[ts_descs[idx[i]].dtype()];
    void* new_mem = aligned_allocator_t<uint8_t, 64>::allocate(pad_to(byte_size, 64));
    SPARSE_LOG_IF(ERROR, !new_mem) << "malloc failed.";
    rt_data[idx[i]] = new_mem;
    new_data.emplace_back(new_mem);
  }
  return true;
}

void bench_op::free_new_mem(std::vector<void*>* new_data_pointer) {
  std::vector<void*>& new_data = *new_data_pointer;
  for (size_t i = 0; i < new_data.size(); ++i) {
    aligned_allocator_t<uint8_t, 64>::deallocate(new_data[i]);
  }
}

}  // namespace jd
