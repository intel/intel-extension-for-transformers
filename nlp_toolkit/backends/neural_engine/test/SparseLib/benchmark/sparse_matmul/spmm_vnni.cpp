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
#include "common_utils.hpp"
#include "sparse_matmul/spmm_vnni.hpp"

namespace jd {

using dt = jd::data_type;
using ft = jd::format_type;

void get_true_data_spmm_vnni(const operator_desc& op_desc, const std::vector<const void*>& rt_data) {
  // shape configure alias
  const auto& ts_descs = op_desc.tensor_descs();
  const auto& src0_desc = ts_descs[ssd::WEI];
  const auto& src1_desc = ts_descs[ssd::SRC];
  const auto& bias_desc = ts_descs[ssd::BIAS];
  const auto& dst_desc = ts_descs[ssd::DST];
  int dims = src0_desc.shape().size();
  int M = src0_desc.shape()[0];
  int K = src0_desc.shape()[1];
  int N = src1_desc.shape()[1];
  const auto& left_dt = src0_desc.dtype();
  const auto& right_dt = src1_desc.dtype();
  const auto& dst_dt = dst_desc.dtype();
  bool has_bias = !bias_desc.shape().empty();
  auto attrs_map = op_desc.attrs();
  bool append_sum = (attrs_map["post_op"] == "append_sum");
  std::vector<int64_t> left_stride = {K, 1};
  std::vector<int64_t> right_stride = {N, 1};
  std::vector<int64_t> dst_stride = {N, 1};

  // runtime data alias
  const auto left_data = rt_data[ssd::WEI];
  const auto right_data = rt_data[ssd::SRC];
  const auto bias_data = static_cast<const int32_t*>(rt_data[ssd::BIAS]);
  auto dst_data = const_cast<void*>(rt_data[ssd::DST]);
  const auto scales_data = static_cast<const float*>(rt_data[ssd::SCALES]);

  // buffer data
  auto left_fp32 = static_cast<const float*>(left_data);  // ptr alias
  auto left_u8 = static_cast<const uint8_t*>(left_data);
  auto left_s8 = static_cast<const int8_t*>(left_data);

  auto right_fp32 = static_cast<const float*>(right_data);  // ptr alias
  auto right_u8 = static_cast<const uint8_t*>(right_data);
  auto right_s8 = static_cast<const int8_t*>(right_data);

  auto dst_fp32 = static_cast<float*>(dst_data);  // ptr alias
  auto dst_s32 = static_cast<int32_t*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);
  auto dst_s8 = static_cast<int8_t*>(dst_data);

  // Computing the kernel
  if (dims == 2) {
    for (int i = 0; i < M; ++i) {
#pragma omp parallel for
      for (int j = 0; j < N; ++j) {
        float value = 0;  // Consistent with the actual precision (float or double) of cpu instructions.
#pragma omp simd
        for (int k = 0; k < K; ++k) {
          int idx0 = i * left_stride[0] + k * left_stride[1];
          int idx1 = k * right_stride[0] + j * right_stride[1];
          auto left_k = (left_dt == dt::fp32)
                            ? left_fp32[idx0]
                            : ((left_dt == dt::u8) ? left_u8[idx0] : ((left_dt == dt::s8) ? left_s8[idx0] : 0));
          auto right_k = (right_dt == dt::fp32)
                             ? right_fp32[idx1]
                             : ((right_dt == dt::u8) ? right_u8[idx1] : ((right_dt == dt::s8) ? right_s8[idx1] : 0));
          value += left_k * right_k;
        }

        // Accumulate bias or post sum
        if (has_bias) {
          value += bias_data[i];
        }
        int dst_idx = i * dst_stride[0] + j * dst_stride[1];
        int scale_idx = i;
        if (dst_dt == dt::fp32) {
          value = value * scales_data[scale_idx];
        }
        if (append_sum) {
          value += dst_fp32[dst_idx];
        }

        // Quantize dst data
        if (dst_dt == dt::fp32) {
          dst_fp32[dst_idx] = static_cast<float>(value);
        } else if (dst_dt == dt::s32) {
          dst_s32[dst_idx] = static_cast<int32_t>(value);
        } else if (dst_dt == dt::s8) {
          dst_s8[dst_idx] = fp32_2_s8(value, scales_data[scale_idx]);
        }
      }
    }
  }
}

bool check_result_spmm_vnni(const std::pair<op_args_t, op_args_t>& args) {
  const auto& p = args.first;
  const auto& q = args.second;

  get_true_data_spmm_vnni(q.op_desc, q.rt_data);
  auto buf1 = p.rt_data[ssd::DST];
  auto size1 = p.op_desc.tensor_descs()[ssd::DST].size();
  auto buf2 = q.rt_data[ssd::DST];
  auto size2 = q.op_desc.tensor_descs()[ssd::DST].size();
  // Should compare buffer with different addresses
  if (buf1 == buf2) {
    printf("comparing the same buffer\n");
    return false;
  }

  const auto& dst_type = p.op_desc.tensor_descs()[ssd::DST].dtype();
  if (dst_type == dt::fp32) {
    return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::s32) {
    return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::u8) {
    return compare_data<uint8_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::s8) {
    return compare_data<int8_t>(buf1, size1, buf2, size2, 5e-3);
  }
  return false;
}

template <typename T>
void prepare_sparse_data_spmm_vnni(T* vector_data, std::vector<int64_t> a_shape, float sparse_ratio) {
  int64_t M = a_shape[0];
  int64_t K = a_shape[1];
  // Blocks zeros in the M dimension.
  int64_t BLOCK = 4;
  uint32_t seed = 123;
  for (int mb = 0; mb < M / BLOCK; ++mb) {
    for (int kb = 0; kb < K; ++kb) {
      bool fill_zero = rand_r(&seed) % 100 <= (sparse_ratio * 100);
      if (fill_zero) {
        for (int m = 0; m < BLOCK; ++m) {
          for (int k = 0; k < 1; ++k) {
            vector_data[(mb * BLOCK + m) * K + kb + k] = 0;
          }
        }
      }
    }
  }
}
template void prepare_sparse_data_spmm_vnni<int8_t>(int8_t*, std::vector<int64_t>, float);

std::pair<const void*, const void*> make_data_obj_spmm_vnni(const std::vector<int64_t>& a_shape, const dt& a_dt,
                                                            bool is_clear, float sparse_ratio,
                                                            const std::vector<float>& ranges) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int bytes_size = elem_num * type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else {
    if (a_dt == dt::fp32) {
      data_ptr = new float[elem_num];
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s32) {
      data_ptr = new int32_t[elem_num];
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::u8) {
      data_ptr = new uint8_t[elem_num];
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s8) {
      data_ptr = new int8_t[elem_num];
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
      if (sparse_ratio != 0.f) {
        int8_t* s8_ptr = static_cast<int8_t*>(data_ptr);
        prepare_sparse_data_spmm_vnni<int8_t>(s8_ptr, a_shape, sparse_ratio);
      }
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::pair<op_args_t, op_args_t> gen_case_spmm_vnni(dim_t M, dim_t K, dim_t N, float sparsity, jd::data_type dt_dst,
                                                   const std::string& mkn_blocks, const std::string& tile_shape,
                                                   std::string post_op) {
  // Step 1: Construct operator config
  std::unordered_map<std::string, std::string> op_attrs = {
      {"mkn_blocks", mkn_blocks}, {"tile_shape", tile_shape}, {"post_op", post_op}};
  bool append_sum = (op_attrs["post_op"] == "append_sum");

  // Step 2: Construct runtime data
  tensor_desc wei_desc = {{M, K}, dt::s8, ft::bsr};
  tensor_desc src_desc = {{K, N}, dt::u8, ft::ab};
  tensor_desc bia_desc = {{M, 1}, dt::s32, ft::ab};
  tensor_desc dst_desc = {{M, N}, dt_dst, ft::ab};
  tensor_desc scales_desc = {{M, 1}, dt::fp32, ft::ab};
  std::vector<tensor_desc> ts_descs = {wei_desc, src_desc, bia_desc, dst_desc, scales_desc};

  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    bool is_clear = (index == ssd::DST && !append_sum);
    float data_sparsity = (index == ssd::WEI) ? sparsity : 0;
    auto ranges = (index == ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    auto data_pair =
        make_data_obj_spmm_vnni(ts_descs[index].shape(), ts_descs[index].dtype(), is_clear, data_sparsity, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  // Step 3: sparse data encoding
  auto sparse_ptr = new bsr_data_t<int8_t>(
      spns::reorder_to_bsr_group<int8_t, 4>(M, K, 4, 1, static_cast<const int8_t*>(rt_data1[ssd::WEI])));
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  operator_desc an_op_desc(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                           op_attrs);

  // Step 4: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1};
  op_args_t op_args_copy = {an_op_desc, rt_data2};

  return {op_args, op_args_copy};
}

bench_res_t run_bench_spmm_vnni(bench_mode mode, int argc, char** argv) {
  bench_res_t res;
  int64_t M = str_to_num<int64_t>(argv[0]);
  int64_t K = str_to_num<int64_t>(argv[1]);
  int64_t N = str_to_num<int64_t>(argv[2]);
  float sparse_ratio = str_to_num<float>(argv[3]);
  dt dt_dst = strcmp(argv[4], "1") ? dt::s8 : dt::fp32;
  std::string mkn_blocks(argv[5]);
  std::string tile_shape(argv[6]);
  std::string post_op = strcmp(argv[7], "1") ? "" : "append_sum";

  std::pair<op_args_t, op_args_t> args =
      gen_case_spmm_vnni(M, K, N, sparse_ratio, dt_dst, mkn_blocks, tile_shape, post_op);
  try {
    const auto& p = args.first;
    const auto& op_desc = p.op_desc;
    sparse_matmul_desc spmm_desc(op_desc);
    sparse_matmul spmm_kern(spmm_desc);
    res = benchmarkOrExecute(&spmm_kern, p.rt_data, mode);
  } catch (const std::exception& e) {
    std::cerr << "kernel exception occurred" << std::endl;
    res.stat = bench_status::fail;
    return res;
  }

  if (mode == bench_mode::acc) {
    res.correct = check_result_spmm_vnni(args);
  }

  return res;
}

}  // namespace jd
