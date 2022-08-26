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

#include "utils.hpp"
#include "benchmark_utils.hpp"

// Internal Control Variables
int benchmark_iter = 100;
bool benchmark_refresh = true;

void read_benchmark_env() {
  const char* input_benchmark_iter = std::getenv("BENCHMARK_ITER");
  if (input_benchmark_iter != nullptr) {
    benchmark_iter = atoi(input_benchmark_iter);
    if (benchmark_iter == 0) {
      printf("BENCHMARK_ITER is 0! Please ensure you set this variable to an integer\n");
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

bench_res_t benchmarkOrExecute(kernel_proxy* kp, const std::vector<const void*>& rt_data, bench_mode mode) {
  bench_res_t res;

  kp->execute(rt_data);

  if (mode == bench_mode::acc) {
    res.stat = bench_status::success;
    return res;
  }

  read_benchmark_env();

  // Use op_desc to get kernel kind and tensor shape
  const auto& op_desc = kp->get_sp()->kd()->operator_desc();
  const auto ker_kind = op_desc.kernel_kind();
  const auto& ts_descs = op_desc.tensor_descs();

  // We may need to refresh some parts of runtime data, allocate new memory for them first
  std::vector<const void*> tmp_data(rt_data);
  std::vector<void*> new_data;
  std::vector<int> idx = get_refresh_data_idx(ker_kind);
  if (!alloc_new_mem(ts_descs, tmp_data, new_data, idx)) {
    res.stat = bench_status::fail;
    return res;
  }

  double ns = 0.0;
  for (int i = 0; i < benchmark_iter; ++i) {
    // refresh data
    if (benchmark_refresh) {
      refresh_data(ts_descs, new_data, idx);
    }

    ns += exec_time(kp, tmp_data);
  }

  // get execution time and calculate GFLOPS
  ns = ns / benchmark_iter;
  res.ms = ns / 1e6;
  res.gflops = calc_flop(ker_kind, ts_descs) / ns;

  // free new memory
  free_new_mem(new_data);

  res.stat = bench_status::success;
  return res;
}

double exec_time(kernel_proxy* kp, const std::vector<const void*>& rt_data) {
  auto begin = std::chrono::high_resolution_clock::now();
  kp->execute(rt_data);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

double calc_flop(const kernel_kind ker_kind, const std::vector<tensor_desc>& ts_descs) {
  switch (ker_kind) {
    case (kernel_kind::sparse_matmul): {
      const auto& src0_desc = ts_descs[ssd::WEI];
      const auto& src1_desc = ts_descs[ssd::SRC];
      int oc = src0_desc.shape()[0];
      int ic = src0_desc.shape()[1];

      // Since avx512f kernel performs activation x weight, the shape of weight tensor is {ic, oc}
      if (src0_desc.dtype() == dt::fp32 && src1_desc.dtype() == dt::fp32) {
        std::swap(oc, ic);
      }

      if (std::find(src1_desc.shape().begin(), src1_desc.shape().end(), ic) == src1_desc.shape().end()) {
        printf("ic is not found in SRC shape!\n");
        return 0.0;
      }
      const int other_dim =
          std::accumulate(src1_desc.shape().begin(), src1_desc.shape().end(), 1, std::multiplies<size_t>()) / ic;
      return static_cast<double>(oc) * other_dim * ic * 2;
    }
    case (kernel_kind::eltwiseop): {
      return std::accumulate(ts_descs[0].shape().begin(), ts_descs[0].shape().end(), 1, std::multiplies<size_t>());
    }
    case (kernel_kind::layernorm_ba): {
      int elem_num =
          std::accumulate(ts_descs[0].shape().begin(), ts_descs[0].shape().end(), 1, std::multiplies<size_t>());
      // flop taken into consideration
      // compute mean: elem_num
      // compute variance: 3 * elem_num
      // compute final result: 2 * elem_num
      return 6 * elem_num;
    }
    default:
      std::cerr << "calc_flop for this kernel is not implemented." << std::endl;
      return 0.0;
  }
}

std::vector<int> get_refresh_data_idx(const kernel_kind ker_kind) {
  switch (ker_kind) {
    case (kernel_kind::sparse_matmul):
      return std::vector<int>{ssd::SRC, ssd::DST};
    case (kernel_kind::eltwiseop):
      return std::vector<int>{0, 1};
    case (kernel_kind::layernorm_ba):
      return std::vector<int>{0};
    default:
      std::cerr << "get_refresh_data_idx for this kernel is not implemented." << std::endl;
      return std::vector<int>(0);
  }
}

bool alloc_new_mem(const std::vector<tensor_desc>& ts_descs, std::vector<const void*>& rt_data,  // NOLINT
                   std::vector<void*>& new_data, const std::vector<int>& idx) {                  // NOLINT
  for (size_t i = 0; i < idx.size(); ++i) {
    int elem_num =
        std::accumulate(ts_descs[idx[i]].shape().begin(), ts_descs[idx[i]].shape().end(), 1, std::multiplies<size_t>());
    int byte_size = elem_num * type_size[ts_descs[idx[i]].dtype()];
    void* new_mem = malloc(byte_size);
    if (!new_mem) {
      std::cerr << "malloc failed." << std::endl;
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

void refresh_data(const std::vector<tensor_desc>& ts_descs, std::vector<void*>& new_data,  // NOLINT
                  const std::vector<int>& idx, const std::vector<float>& ranges) {
  for (size_t i = 0; i < idx.size(); ++i) {
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

}  // namespace jd
