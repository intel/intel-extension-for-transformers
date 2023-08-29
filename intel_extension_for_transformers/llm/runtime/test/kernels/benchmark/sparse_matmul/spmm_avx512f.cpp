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

#include "spmm_avx512f.hpp"

#include <functional>
#include <utility>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
namespace bench {
bench_res_t spmm_avx512f_bench::set_config(int argc, char** argv) {
  LOG(INFO) << "spmm_avx512f\n";
  if (argc < SPMM_AVX512F_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  M = str_to_num<int64_t>(argv[0]);
  K = str_to_num<int64_t>(argv[1]);
  N = str_to_num<int64_t>(argv[2]);
  sparse_ratio = str_to_num<float>(argv[3]);
  postop_algs.clear();
  for (int i = SPMM_AVX512F_ARG_NUM; i < argc; ++i) {
    LOG(INFO) << argv[i];
    if (!strcmp(argv[i], "gelu")) {
      postop_algs.push_back(jd::postop_alg::gelu);
    } else if (!strcmp(argv[i], "exp")) {
      postop_algs.push_back(jd::postop_alg::exp);
    } else if (!strcmp(argv[i], "relu")) {
      postop_algs.push_back(jd::postop_alg::relu);
    } else if (!strcmp(argv[i], "tanh")) {
      postop_algs.push_back(jd::postop_alg::tanh);
    } else {
      LOG(ERROR) << "post-op " << argv[i] << " is not supported.";
    }
  }
  return {bench_status::success};
}
void spmm_avx512f_bench::get_true_data() {
  auto& op_desc = args.second.op_desc;
  auto& rt_data = args.second.rt_data;
  // shape configure alias
  const auto& ts_descs = op_desc.tensor_descs();
  const auto& wei_desc = ts_descs[jd::ssd::WEI];
  const auto& src_desc = ts_descs[jd::ssd::SRC];
  const auto& bias_desc = ts_descs[jd::ssd::BIAS];
  int M = src_desc.shape()[0];
  int K = wei_desc.shape()[0];
  int N = wei_desc.shape()[1];
  bool has_bias = !bias_desc.shape().empty();
  auto attrs_map = op_desc.attrs();
  std::vector<int64_t> left_stride = {K, 1};
  std::vector<int64_t> right_stride = {N, 1};
  std::vector<int64_t> dst_stride = {N, 1};
  // runtime data alias
  const auto left_fp32 = static_cast<const float*>(rt_data[jd::ssd::SRC]);
  const auto right_fp32 = static_cast<const float*>(rt_data[jd::ssd::WEI]);
  const auto bias_fp32 = static_cast<const float*>(rt_data[jd::ssd::BIAS]);
  auto dst_fp32 = static_cast<float*>(const_cast<void*>(rt_data[jd::ssd::DST]));
  // Computing the kernel
  assert(wei_desc.shape().size() == 2);
  for (int i = 0; i < M; ++i) {
#pragma omp parallel for
    for (int j = 0; j < N; ++j) {
      float value = 0;  // Consistent with the actual precision (float or double) of cpu instructions.
#pragma omp simd
      for (int k = 0; k < K; ++k) {
        auto left_k = left_fp32[i * left_stride[0] + k * left_stride[1]];
        auto right_k = right_fp32[k * right_stride[0] + j * right_stride[1]];
        value += left_k * right_k;
      }
      // Accumulate bias or post sum
      if (has_bias) {
        value += bias_fp32[j];
      }
      value = apply_postop_list(value, op_desc.apply_postops_list());
      dst_fp32[i * dst_stride[0] + j * dst_stride[1]] = static_cast<float>(value);
    }
  }
}
bool spmm_avx512f_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;
  get_true_data();
  auto buf1 = p.rt_data[jd::ssd::DST];
  auto size1 = p.op_desc.tensor_descs()[jd::ssd::DST].size();
  auto buf2 = q.rt_data[jd::ssd::DST];
  auto size2 = q.op_desc.tensor_descs()[jd::ssd::DST].size();
  const auto& dst_type = p.op_desc.tensor_descs()[jd::ssd::DST].dtype();
  if (dst_type == jd::data_type::fp32) {
    return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == jd::data_type::s32) {
    return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == jd::data_type::u8) {
    return compare_data<uint8_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == jd::data_type::s8) {
    return compare_data<int8_t>(buf1, size1, buf2, size2, 5e-3);
  }
  return false;
}
template <typename T>
void prepare_blocked_sparse_data_spmm_avx512f(T* data, const std::vector<dim_t>& a_shape,
                                              const std::vector<dim_t>& block_shape, float sparsity,
                                              unsigned int* seed) {
  dim_t K = a_shape[0], N = a_shape[1], BK = block_shape[0], BN = block_shape[1];
  LOG_IF(FATAL, (K % BK | N % BN) != 0) << "Matrix dim must be a multiple of block dim.";
  LOG_IF(FATAL, sparsity < 0 && sparsity > 1) << "Sparsity should be a value between 0 and 1.";
  dim_t nb_k = K / BK;
  dim_t nb_n = N / BN;
  std::srand(*seed);
  for (int ibk = 0; ibk < nb_k; ++ibk) {
    for (int ibn = 0; ibn < nb_n; ++ibn) {
      bool fill_zero = std::rand() % 100 <= (sparsity * 100);
      if (fill_zero) {
        dim_t i_start = ibk * BK;
        dim_t j_start = ibn * BN;
        for (dim_t i = i_start; i < i_start + BK; ++i) {
          for (dim_t j = j_start; j < j_start + BN; ++j) {
            data[i * N + j] = 0.f;
          }
        }
      }
    }
  }
}

std::pair<const void*, const void*> make_data_obj_spmm_avx512f(const std::vector<dim_t>& a_shape,
                                                               const jd::data_type& a_dt, bool is_clear, float sparsity,
                                                               const jd::format_type& a_ft,
                                                               const std::vector<float>& ranges) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int bytes_size = elem_num * jd::type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = aligned_allocator_t<char>::allocate(bytes_size, true);
  } else {
    switch (a_dt) {
      case jd::data_type::fp32:
        data_ptr = aligned_allocator_t<float>::allocate(elem_num);
        init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
        break;
      default:
        SPARSE_LOG(ERROR) << "Unsupprted data type!";
        break;
    }
    if (sparsity != 0.f) {
      switch (a_ft) {
        case jd::format_type::bsc: {
          std::vector<dim_t> block_shape = {1, 16};
          unsigned int seed = 123;
          prepare_blocked_sparse_data_spmm_avx512f(static_cast<float*>(data_ptr), a_shape, block_shape, sparsity,
                                                   &seed);
          break;
        }
        default:
          break;
      }
    }
  }
  void* data_ptr_copy = aligned_allocator_t<char>::allocate(bytes_size);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}
void spmm_avx512f_bench::gen_case() {
  // Step 1: Construct operator config
  std::unordered_map<std::string, std::string> op_attrs = {};
  // Step 2: Construct runtime data
  jd::tensor_desc src_desc = {{M, K}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc wei_desc = {{K, N}, jd::data_type::fp32, jd::format_type::bsc};
  jd::tensor_desc bia_desc = {{N, 1}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc dst_desc = {{M, N}, jd::data_type::fp32, jd::format_type::abc};
  ts_descs = {wei_desc, src_desc, bia_desc, dst_desc};
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  for (size_t i = 0; i < ts_descs.size(); ++i) {
    bool is_clear = i == jd::ssd::DST || i == jd::ssd::BIAS;
    std::vector<float> ranges = (i == jd::ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    float data_sparsity = (i == jd::ssd::WEI) ? sparse_ratio : 0;
    auto data_pair = make_data_obj_spmm_avx512f(ts_descs[i].shape(), ts_descs[i].dtype(), is_clear, data_sparsity,
                                                ts_descs[i].ftype(), ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  std::vector<const float*> rt_data_ins;
  for (auto p : rt_data1) rt_data_ins.push_back(static_cast<const float*>(p));
  // Step 3: sparse data encoding
  jd::bsc_data_t<float> bsc_obj =
      jd::spns::tobsc<float>(K, N, 1, 16, static_cast<const float*>(rt_data1[jd::ssd::WEI]));
  auto sparse_ptr = new jd::bsc_data_t<float>(bsc_obj);  // Will be deleted in `~sparse_matmul_bench()`
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  if (postop_algs.size()) {
    auto accu_op = [](std::string str_lists, jd::postop_alg alg) { return str_lists + '_' + jd::postop_alg_name[alg]; };
    op_attrs["postop_list"] = std::accumulate(postop_algs.begin() + 1, postop_algs.end(),
                                              std::string(jd::postop_alg_name[postop_algs[0]]), accu_op);
  }
  std::vector<jd::postop_attr> apply_postops_list;
  std::for_each(postop_algs.begin(), postop_algs.end(), [&apply_postops_list](jd::postop_alg alg) {
    return apply_postops_list.push_back({jd::data_type::fp32, jd::postop_type::eltwise, alg});
  });
  jd::operator_desc an_op_desc(jd::kernel_kind::sparse_matmul, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                               ts_descs, op_attrs, apply_postops_list);
  // Step 4: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1};
  op_args_t op_args_copy = {an_op_desc, rt_data2};
  args = {op_args, op_args_copy};
}
}  // namespace bench
