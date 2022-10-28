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

#include "sparse_matmul/spmm_vnni.hpp"

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "utils.hpp"

namespace jd {

using dt = jd::data_type;
using ft = jd::format_type;

void spmm_vnni_bench::get_true_data() {
  auto& op_desc = args.second.op_desc;
  auto& rt_data = args.second.rt_data;
  // shape configure alias
  const auto& ts_descs = op_desc.tensor_descs();
  const auto& wei_shape = ts_descs[ssd::WEI].shape();
  const auto& wei_type = ts_descs[ssd::WEI].dtype();
  const auto& src_type = ts_descs[ssd::SRC].dtype();
  const auto& src_shape = ts_descs[ssd::SRC].shape();
  const auto& dst_type = ts_descs[ssd::DST].dtype();
  const auto& dst_shape = ts_descs[ssd::DST].shape();
  assert((src_shape.size() == 2 || src_shape.size() == 3) && src_shape.size() == dst_shape.size());

  int oc = wei_shape[0];
  int ic = wei_shape[1];
  int micro_bs = src_shape.back();
  int num_mbs = src_shape.size() == 2 ? 1 : src_shape[0];

  const auto& left_dt = wei_type;
  const auto& right_dt = src_type;
  const auto& dst_dt = dst_type;
  bool has_bias = !ts_descs[ssd::BIAS].shape().empty();
  auto attrs_map = op_desc.attrs();
  bool append_sum = (attrs_map["post_op"] == "append_sum");
  std::vector<dim_t> left_stride = {ic, 1};
  std::vector<dim_t> right_stride = {micro_bs * ic, micro_bs, 1};
  std::vector<dim_t> dst_stride = {micro_bs * oc, micro_bs, 1};

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
  auto dst_s8 = static_cast<int8_t*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);

  // TODO(zhe1wang): add per channel support for post-op;
  auto postop_list = op_desc.apply_postops_list();

// Computing the kernel
#pragma omp parallel for collapse(3)
  for (dim_t idx_mbs = 0; idx_mbs < num_mbs; ++idx_mbs) {
    for (dim_t i = 0; i < oc; ++i) {
      for (dim_t j = 0; j < micro_bs; ++j) {
        float value = 0;  // Consistent with the actual precision (float or
                          // double) of cpu instructions.
#pragma omp simd
        for (dim_t k = 0; k < ic; ++k) {
          dim_t l_idx = i * left_stride[0] + k * left_stride[1];
          dim_t r_idx = idx_mbs * right_stride[0] + k * right_stride[1] + j * right_stride[2];
          auto left_k = (left_dt == dt::fp32)
                            ? left_fp32[l_idx]
                            : ((left_dt == dt::u8) ? left_u8[l_idx] : ((left_dt == dt::s8) ? left_s8[l_idx] : 0));
          auto right_k = (right_dt == dt::fp32)
                             ? right_fp32[r_idx]
                             : ((right_dt == dt::u8) ? right_u8[r_idx] : ((right_dt == dt::s8) ? right_s8[r_idx] : 0));
          value += left_k * right_k;
        }

        // Accumulate bias or post sum
        if (has_bias) {
          value += bias_data[i];
        }
        dim_t dst_idx = idx_mbs * dst_stride[0] + i * dst_stride[1] + j * dst_stride[2];
        dim_t scale_idx = i;
        if (dst_dt != dt::s32) {
          value = value * scales_data[scale_idx];
        }
        if (append_sum) {
          value += dst_fp32[dst_idx];
        }
        value = apply_postop_list(value, postop_list);
        // Quantize dst data
        if (dst_dt == dt::fp32) {
          dst_fp32[dst_idx] = static_cast<float>(value);
        } else if (dst_dt == dt::s32) {
          dst_s32[dst_idx] = static_cast<int32_t>(value);
        } else if (dst_dt == dt::s8) {
          value = value < -128 ? -128 : value;
          value = value > 127 ? 127 : value;
          dst_s8[dst_idx] = static_cast<int8_t>(value);
        } else if (dst_dt == dt::u8) {
          dst_u8[dst_idx] = static_cast<uint8_t>(value);
        }
      }
    }
  }
}

bool spmm_vnni_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;

  get_true_data();
  auto buf1 = p.rt_data[ssd::DST];
  auto size1 = p.op_desc.tensor_descs()[ssd::DST].size();
  auto buf2 = q.rt_data[ssd::DST];
  auto size2 = q.op_desc.tensor_descs()[ssd::DST].size();
  const auto& dst_type = p.op_desc.tensor_descs()[ssd::DST].dtype();
  if (dst_type == dt::fp32) {
    return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::s32) {
    return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::u8) {
    return compare_data<uint8_t>(buf1, size1, buf2, size2, 8e-3);
  } else if (dst_type == dt::s8) {
    return compare_data<int8_t>(buf1, size1, buf2, size2, 8e-3);
  }
  return false;
}

template <typename T>
void prepare_sparse_data_spmm_vnni(T* vector_data, dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, float sparsity,
                                   uint32_t* seed = nullptr) {
  uint32_t default_seed = 123;
  if (seed == nullptr) seed = &default_seed;
  for (int i = 0; i < rows; i += blk_row) {
    for (int j = 0; j < cols; j += blk_col) {
      bool fill_zero = rand_r(seed) % 100 <= (sparsity * 100);
      if (fill_zero) {
        for (int bi = i; bi < i + blk_row; ++bi) {
          for (int bj = j; bj < j + blk_col; ++bj) {
            vector_data[bi * cols + bj] = 0;
          }
        }
      }
    }
  }
}

std::pair<void*, void*> make_data_obj_spmm_vnni(const std::vector<int64_t>& a_shape, const dt& a_dt, bool is_clear,
                                                float sparse_ratio, const std::vector<float>& ranges) {
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
        prepare_sparse_data_spmm_vnni<int8_t>(s8_ptr, a_shape[0], a_shape[1], 4, 1, sparse_ratio);
      }
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<void*, void*>{data_ptr, data_ptr_copy};
}

std::vector<float> make_output_scale(dim_t size, const std::vector<float>& ranges = {-10, 10}) {
  std::vector<float> output_scale(size, 0);
  init_vector(output_scale.data(), size * sizeof(float), ranges[0], ranges[1]);
  return output_scale;
}

void spmm_vnni_bench::gen_case() {
  // Step 1: Construct operator config
  bool append_sum = (op_attrs["append_sum"] == "true");
  micro_bs = micro_bs == -1 ? N : micro_bs;
  LOG_IF(FATAL, N % micro_bs != 0) << "micro_bs must be a multiple of N";
  dim_t num_mbs = N / micro_bs;

  // Step 2: Construct runtime data
  tensor_desc wei_desc = {{M, K}, dt::s8, ft::bsr};
  tensor_desc src_desc = {{num_mbs, K, micro_bs}, dt::u8, ft::ab};
  tensor_desc bia_desc = {{M, 1}, dt::s32, ft::ab};
  tensor_desc dst_desc = {{num_mbs, M, micro_bs}, dt_dst, ft::ab};
  tensor_desc scales_desc = {{M, 1}, dt::fp32, ft::ab};
  ts_descs = {wei_desc, src_desc, bia_desc, dst_desc, scales_desc};

  std::vector<void*> rt_data1;
  std::vector<void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    bool is_clear = (index == ssd::DST && !append_sum);
    float data_sparsity = (index == ssd::WEI) ? sparse_ratio : 0;
    auto ranges = (index == ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    auto data_pair = make_data_obj_spmm_vnni(tsd.shape(), tsd.dtype(), is_clear, data_sparsity, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  float scale = make_output_scale(1)[0];

  // Step 3: sparse data encoding
  auto sparse_ptr = new bsr_data_t<int8_t>(
      spns::reorder_to_bsr_group<int8_t, 4>(M, K, 4, 1, static_cast<const int8_t*>(rt_data1[ssd::WEI])));
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  std::vector<postop_attr> apply_postops_list;
  if (postop_algs.size()) {
    auto accu_op = [](std::string str_lists, postop_alg alg) { return str_lists + '_' + postop_alg_name[alg]; };
    op_attrs["postop_list"] = std::accumulate(postop_algs.begin() + 1, postop_algs.end(),
                                              std::string(postop_alg_name[postop_algs[0]]), accu_op);
    for (auto& alg : postop_algs) {
      postop_attr attr(dt::fp32, postop_type::eltwise, alg, 0.0, 0.0, scale);
      apply_postops_list.push_back(attr);
    }
  }
  if (dt_dst == dt::s8 || dt_dst == dt::u8) {
    postop_attr attr(dt_dst, postop_type::eltwise, postop_alg::quantize, 0.0, 0.0, scale);
    apply_postops_list.push_back(attr);
  }
  operator_desc an_op_desc(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                           op_attrs, apply_postops_list);

  // Step 4: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1};
  op_args_t op_args_copy = {an_op_desc, rt_data2};

  args = {op_args, op_args_copy};
}

bench_res_t spmm_vnni_bench::set_config(int argc, char** argv) {
  LOG(INFO) << "spmm_vnni\n";
  if (argc < SPMM_VNNI_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  M = str_to_num<int64_t>(argv[0]);                        // M
  K = str_to_num<int64_t>(argv[1]);                        // K
  N = str_to_num<int64_t>(argv[2]);                        // N
  sparse_ratio = str_to_num<float>(argv[3]);               // sparse_ratio
  micro_bs = str_to_num<int64_t>(argv[4]);                 // micro_bs
  dt_dst = strcmp(argv[5], "1") == 0 ? dt::fp32 : dt::s8;  // is_fp32_out
  bool append_sum = strcmp(argv[6], "1") == 0;             // has_append_sum
  if (append_sum && dt_dst != dt::fp32) {
    LOG(ERROR) << "append_sum requires fp32";
    bench_res_t res;
    res.stat = bench_status::fail;
    return res;
  }
  int64_t micro_oc = str_to_num<int64_t>(argv[7]);
  int sub_func_level = strlen(argv[8]) == 0 ? -1 : str_to_num<int>(argv[8]);  // -1 for invalid/defalut config
  if (argc > 9 && strcmp(argv[9], "gelu")) postop_algs.push_back(postop_alg::gelu);
  if (argc > 9 && strcmp(argv[9], "exp")) postop_algs.push_back(postop_alg::exp);

  if (append_sum) op_attrs["post_op"] = "append_sum";
  if (micro_oc >= 0) op_attrs["micro_oc"] = std::to_string(micro_oc);
  if (sub_func_level >= 0) {
    if (sub_func_level <= static_cast<uint8_t>(ssd::subfunc_level::subfunc_level_MAX)) {
      op_attrs["sub_func"] = std::to_string(sub_func_level);
    }
  }
  return {bench_status::success};
}

}  // namespace jd
