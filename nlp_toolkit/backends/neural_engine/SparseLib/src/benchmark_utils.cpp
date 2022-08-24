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

#include "utils.hpp"
#include "benchmark_utils.hpp"

namespace jd {
using dt = jd::data_type;

void benchmarkOrExecute(kernel_proxy* kp, const std::vector<const void*>& rt_data) {
  kp->execute(rt_data);

  const char* env_val = std::getenv("SPARSE_LIB_USE_BENCHMARK");
  if (env_val == nullptr || strcmp(env_val, "1") != 0) {
    return;
  }

  // Use op_desc to get kernel kind and tensor shape
  const auto& op_desc = kp->get_sp()->kd()->operator_desc();
  const auto ker_kind = op_desc.kernel_kind();
  const auto& ts_descs = op_desc.tensor_descs();

  // We may need to refresh some parts of runtime data, allocate new memory for them first
  std::vector<const void*> tmp_data(rt_data);
  std::vector<void*> new_data;
  std::vector<int> idx = get_refresh_data_idx(ker_kind);
  if (!alloc_new_mem(ts_descs, tmp_data, new_data, idx)) {
    return;
  }

  int benchmark_iter = 100;  // by default
  const char* input_benchmark_iter = std::getenv("BENCHMARK_ITER");
  if (input_benchmark_iter != nullptr) {
    benchmark_iter = atoi(input_benchmark_iter);
    if (benchmark_iter == 0) {
      printf("BENCHMARK_ITER is 0! Please ensure you set an integer to this variable\n");
    }
  }

  const char* bool_benchmark_refresh = std::getenv("BENCHMARK_NO_REFRESH");
  bool if_refresh = true;
  if (bool_benchmark_refresh != nullptr) {
    if (strcmp(bool_benchmark_refresh, "1") == 0) {
      if_refresh = false;
    }
  }

  double ns = 0.0;
  for (int i = 0; i < benchmark_iter; ++i) {
    // refresh data
    if (if_refresh) {
      refresh_data(ts_descs, new_data, idx);
    }

    ns += exec_time(kp, tmp_data);
  }

  // get execution time and calculate GFLOPS
  ns = ns / benchmark_iter;
  double gflops = calc_flop(ker_kind, ts_descs) / ns;
  printf("kernel execution time: %lfms,  GFLOPS:%lf\n", ns / 1e6, gflops);

  // free new memory
  free_new_mem(new_data);
}

double exec_time(kernel_proxy* kp, const std::vector<const void*>& rt_data) {
  auto begin = std::chrono::high_resolution_clock::now();
  kp->execute(rt_data);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

double calc_flop(const kernel_kind ker_kind, const std::vector<tensor_desc>& ts_descs) {
#define CASE(kind)        \
  case kernel_kind::kind: \
    return calc_flop_##kind(ts_descs);

  switch (ker_kind) {
    CASE(sparse_matmul);
    CASE(postop);
    default:
      printf("calc_flop_<kernel_kind %d> not implemented.\n", ker_kind);
      return 0.0;
  }

#undef CASE
}

std::vector<int> get_refresh_data_idx(const kernel_kind ker_kind) {
#define CASE(kind)        \
  case kernel_kind::kind: \
    return get_refresh_data_idx_##kind();

  switch (ker_kind) {
    CASE(sparse_matmul);
    CASE(postop);
    default:
      printf("get_refresh_data_idx_<kernel_kind %d> not implemented.\n", ker_kind);
      return std::vector<int>(0);
  }

#undef CASE
}

bool alloc_new_mem(const std::vector<tensor_desc>& ts_descs, std::vector<const void*>& rt_data,  // NOLINT
                   std::vector<void*>& new_data, const std::vector<int>& idx) {                  // NOLINT
  for (int i = 0; i < idx.size(); ++i) {
    int elem_num =
        std::accumulate(ts_descs[idx[i]].shape().begin(), ts_descs[idx[i]].shape().end(), 1, std::multiplies<size_t>());
    int byte_size = elem_num * type_size[ts_descs[idx[i]].dtype()];
    void* new_mem = malloc(byte_size);
    if (!new_mem) {
      printf("malloc failed.\n");
      return false;
    }
    rt_data[idx[i]] = new_mem;
    new_data.emplace_back(new_mem);
  }
  return true;
}

void free_new_mem(std::vector<void*>& new_data) {  // NOLINT
  for (int i = 0; i < new_data.size(); ++i) {
    free(new_data[i]);
  }
}

void refresh_data(const std::vector<tensor_desc>& ts_descs, std::vector<void*>& new_data, // NOLINT
                  const std::vector<int>& idx, const std::vector<float>& ranges) {
  for (int i = 0; i < idx.size(); ++i) {
    int elem_num =
        std::accumulate(ts_descs[idx[i]].shape().begin(), ts_descs[idx[i]].shape().end(), 1, std::multiplies<size_t>());
    switch (ts_descs[idx[i]].dtype()) {
      case dt::fp32:
        init_vector(static_cast<float*>(new_data[i]), elem_num, ranges[0], ranges[1], rand());
        break;
      case dt::s32:
        init_vector(static_cast<int32_t*>(new_data[i]), elem_num, ranges[0], ranges[1], rand());
        break;
      case dt::u8:
        init_vector(static_cast<uint8_t*>(new_data[i]), elem_num, ranges[0], ranges[1], rand());
        break;
      case dt::s8:
        init_vector(static_cast<int8_t*>(new_data[i]), elem_num, ranges[0], ranges[1], rand());
        break;
      case dt::bf16:
        init_vector(static_cast<bfloat16_t*>(new_data[i]), elem_num, ranges[0], ranges[1], rand());
        break;
      default:
        break;
    }
  }
}

double calc_flop_sparse_matmul(const std::vector<tensor_desc>& ts_descs) {
  const auto& src0_desc = ts_descs[ssd::WEI];
  const auto& src1_desc = ts_descs[ssd::SRC];
  const int oc = src0_desc.shape()[0];
  const int ic = src0_desc.shape()[1];
  if (std::find(src1_desc.shape().begin(), src1_desc.shape().end(), ic) == src1_desc.shape().end()) {
    printf("ic is not found in SRC shape!\n");
    return 0.0;
  }
  const int other_dim =
      std::accumulate(src1_desc.shape().begin(), src1_desc.shape().end(), 1, std::multiplies<size_t>()) / ic;
  return static_cast<double>(oc) * other_dim * ic * 2;
}

double calc_flop_postop(const std::vector<tensor_desc>& ts_descs) {
  return std::accumulate(ts_descs[0].shape().begin(), ts_descs[0].shape().end(), 1, std::multiplies<size_t>());
}

std::vector<int> get_refresh_data_idx_sparse_matmul() { return std::vector<int>{ssd::SRC, ssd::DST}; }

std::vector<int> get_refresh_data_idx_postop() { return std::vector<int>{0, 1}; }

}  // namespace jd
