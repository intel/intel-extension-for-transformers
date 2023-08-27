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
#include "spmm_amx_bf16_x16.hpp"
#include "common_utils.hpp"

namespace bench {
bench_res_t spmm_amx_bf16_x16_bench::set_config(int argc, char** argv) {
  LOG(INFO) << "spmm_amx_bf16_x16\n";
  if (argc < SPMM_AMX_BF16_X16_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  M = str_to_num<int64_t>(argv[0]);
  K = str_to_num<int64_t>(argv[1]);
  N = str_to_num<int64_t>(argv[2]);
  sparse_ratio = str_to_num<float>(argv[3]);
  micro_bs = str_to_num<int64_t>(argv[4]);
  micro_oc = str_to_num<int64_t>(argv[5]);
  bf16_out = !strcmp(argv[6], "1");
  return {bench_status::success};
}
void spmm_amx_bf16_x16_bench::get_true_data() {
  auto& op_desc = args.second.op_desc;
  auto& rt_data = args.second.rt_data;
  // shape configure alias
  const auto& ts_descs = op_desc.tensor_descs();
  const auto& wei_desc = ts_descs[0];
  const auto& src_desc = ts_descs[1];
  const auto& dst_desc = ts_descs[3];
  int N = wei_desc.shape()[0];
  int K = wei_desc.shape()[1];
  int NUM_M = src_desc.shape()[0];
  int M_MICRO = src_desc.shape()[2];
  const auto& dst_dt = dst_desc.dtype();
  auto attrs_map = op_desc.attrs();
  // runtime data alias
  const auto wei_data = static_cast<const jd::bfloat16_t*>(rt_data[0]);
  const auto src_data = static_cast<const jd::bfloat16_t*>(rt_data[1]);
  const auto bia_data = static_cast<const float*>(rt_data[2]);
  void* dst_data = const_cast<void*>(rt_data[3]);
  std::vector<float> float_dst_data(N * NUM_M * M_MICRO, 0);
  jd::bfloat16_t* bf_dst_data = static_cast<jd::bfloat16_t*>(dst_data);
  float* fp_dst_data = static_cast<float*>(dst_data);
  // Computing the kernel
  for (int num_m = 0; num_m < NUM_M; ++num_m) {
#pragma omp parallel for
    for (int n = 0; n < N; ++n) {
#pragma omp parallel for
      for (int m = 0; m < M_MICRO; ++m) {
#pragma omp parallel for
        for (int k = 0; k < K; ++k) {
          float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] +=
              static_cast<float>(wei_data[n * K + k]) *
              static_cast<float>(src_data[num_m * K * M_MICRO + k * M_MICRO + m]);
        }
        float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] += bia_data[n];
        if (dst_dt == jd::data_type::bf16) {
          bf_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] = (float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m]);
        } else {
          fp_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] = float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m];
        }
      }
    }
  }
}
bool spmm_amx_bf16_x16_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;
  get_true_data();
  auto buf1 = p.rt_data[3];
  auto size1 = p.op_desc.tensor_descs()[3].size();
  auto buf2 = q.rt_data[3];
  auto size2 = q.op_desc.tensor_descs()[3].size();
  const auto& dst_type = p.op_desc.tensor_descs()[3].dtype();
  if (dst_type == jd::data_type::bf16) {
    return compare_data<jd::bfloat16_t>(buf1, size1, buf2, size2, 3e-2);
  }
  return compare_data<float>(buf1, size1, buf2, size2, 3e-2);
}
template <typename T>
void prepare_sparse_data_spmm_amx_bf16_x16(T* weight, dim_t N, dim_t K, dim_t n_blksize, dim_t k_blksize, float ratio) {
  uint32_t seed = 9527;
  std::srand(seed);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      weight[n * K + k] = static_cast<float>(std::rand() % 10 - 5) / 10;
    }
  }
  // sparsify a_mat
  for (int nb = 0; nb < N / n_blksize; ++nb) {
    for (int kb = 0; kb < K / k_blksize; ++kb) {
      bool fill_zero = std::rand() % 100 <= (dim_t)(ratio * 100);
      if (fill_zero) {
        for (int n = 0; n < n_blksize; ++n) {
          for (int k = 0; k < k_blksize; ++k) {
            weight[(nb * n_blksize + n) * K + kb * k_blksize + k] = 0.f;
          }
        }
      }
    }
  }
}
template void prepare_sparse_data_spmm_amx_bf16_x16<jd::bfloat16_t>(jd::bfloat16_t*, dim_t, dim_t, dim_t, dim_t, float);
std::pair<const void*, const void*> make_data_obj_spmm_amx_bf16_x16(const jd::data_type& tensor_dt, dim_t rows,
                                                                    dim_t cols, dim_t index, float ratio,
                                                                    const std::vector<float>& ranges) {
  dim_t elem_num = rows * cols;
  dim_t bytes_size = elem_num * jd::type_size[tensor_dt];
  void* data_ptr = nullptr;
  switch (index) {
    case 0: {  // prepare wei
      data_ptr = aligned_allocator_t<jd::bfloat16_t>::allocate(elem_num);
      jd::bfloat16_t* bf16_ptr = static_cast<jd::bfloat16_t*>(data_ptr);
      prepare_sparse_data_spmm_amx_bf16_x16<jd::bfloat16_t>(bf16_ptr, rows, cols, 16, 1, ratio);
      break;
    }
    case 1: {  // prepare src
      data_ptr = aligned_allocator_t<jd::bfloat16_t>::allocate(elem_num);
      jd::bfloat16_t* bf16_ptr = static_cast<jd::bfloat16_t*>(data_ptr);
      init_vector(bf16_ptr, elem_num, ranges[0], ranges[1]);
      break;
    }
    case 2: {  // prepare bias
      data_ptr = aligned_allocator_t<float>::allocate(elem_num);
      float* fp32_ptr = static_cast<float*>(data_ptr);
      init_vector(fp32_ptr, elem_num, ranges[0], ranges[1]);
      break;
    }
    case 3: {  // prepare dst
      if (tensor_dt == jd::data_type::bf16) {
        data_ptr = aligned_allocator_t<jd::bfloat16_t>::allocate(elem_num, true);
      } else {
        data_ptr = aligned_allocator_t<float>::allocate(elem_num, true);
      }
      break;
    }
  }
  void* data_ptr_copy = aligned_allocator_t<char>::allocate(bytes_size);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}
void spmm_amx_bf16_x16_bench::gen_case() {
  std::unordered_map<std::string, std::string> op_attrs;
  // Step 1: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  jd::tensor_desc wei_desc = {{N, K}, jd::data_type::bf16, jd::format_type::bsr};
  jd::tensor_desc src_desc = {{M / micro_bs, K, micro_bs}, jd::data_type::bf16, jd::format_type::abc};
  jd::tensor_desc bia_desc = {{N, 1}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc dst_desc = {{M / micro_bs, N, micro_bs}, jd::data_type::fp32, jd::format_type::abc};
  if (bf16_out) {
    dst_desc = {{M / micro_bs, N, micro_bs}, jd::data_type::bf16, jd::format_type::abc};
  }
  ts_descs = {wei_desc, src_desc, bia_desc, dst_desc};
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    dim_t rows = ts_descs[index].shape()[0];
    dim_t cols = ts_descs[index].shape()[1];
    if (index == 1 || index == 3) {
      rows = ts_descs[index].shape()[1];
      cols = ts_descs[index].shape()[0] * ts_descs[index].shape()[2];
    }
    auto data_pair = make_data_obj_spmm_amx_bf16_x16(ts_descs[index].dtype(), rows, cols, index, sparse_ratio);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  // Step 2: sparse data encoding
  if (micro_oc == -1) {
    micro_oc = N;
  }
  volatile auto sparse_ptr = jd::spns::reorder_to_bsr_amx<jd::bfloat16_t, 32>(N, K, micro_oc, rt_data1[0]);
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  op_attrs["micro_oc"] = std::to_string(micro_oc);
  jd::operator_desc an_op_desc(jd::kernel_kind::sparse_matmul, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                               ts_descs, op_attrs);
  // Step 3: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1};
  op_args_t op_args_copy = {an_op_desc, rt_data2};
  args = {op_args, op_args_copy};
}
}  // namespace bench
