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
#include "spmm_vnni.hpp"
#include <memory>
#include <functional>

#include "src/cpu/kernels/spmm_ref.hpp"
#include "benchmark_utils.hpp"
#include "common_utils.hpp"

#define WORKSPACE

namespace bench {

void spmm_vnni_bench::get_true_data() {}
bool spmm_vnni_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;
  std::shared_ptr<const jd::kernel_desc_t> spmm_ref_desc;
  jd::kernel_desc_t::create<jd::spmm_ref_kd_t>(spmm_ref_desc, q.op_desc);
  std::shared_ptr<const jd::kernel_t> spmm_ref_kernel;
  jd::kernel_t::create<jd::spmm_ref_k_t, jd::spmm_ref_kd_t>(spmm_ref_kernel, spmm_ref_desc);
  spmm_ref_kernel->execute(q.rt_data);
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
    return compare_data<uint8_t>(buf1, size1, buf2, size2, 8e-3);
  } else if (dst_type == jd::data_type::s8) {
    return compare_data<int8_t>(buf1, size1, buf2, size2, 8e-3);
  }
  return false;
}
template <typename T>
void prepare_sparse_data_spmm_vnni(T* vector_data, dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, float sparsity,
                                   uint32_t* seed = nullptr) {
  uint32_t default_seed = 123;
  std::srand(default_seed);
  if (seed == nullptr) seed = &default_seed;
  for (int i = 0; i < rows; i += blk_row) {
    for (int j = 0; j < cols; j += blk_col) {
      bool fill_zero = std::rand() % 100 <= (sparsity * 100);
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
std::pair<const void*, const void*> make_data_obj_spmm_vnni(const std::vector<int64_t>& a_shape,
                                                            const jd::data_type& a_dt, bool is_clear,
                                                            float sparse_ratio, const std::vector<float>& ranges) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int bytes_size = elem_num * jd::type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = aligned_allocator_t<uint8_t, 64>::allocate(bytes_size);
    memset(data_ptr, 0, bytes_size);
  } else {
    if (a_dt == jd::data_type::fp32) {
      data_ptr = aligned_allocator_t<float, 64>::allocate(elem_num);
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::s32) {
      data_ptr = aligned_allocator_t<int32_t, 64>::allocate(elem_num);
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::u8) {
      data_ptr = aligned_allocator_t<uint8_t, 64>::allocate(elem_num);
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::s8) {
      data_ptr = aligned_allocator_t<int8_t, 64>::allocate(elem_num);
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
      if (sparse_ratio != 0.f) {
        int8_t* s8_ptr = static_cast<int8_t*>(data_ptr);
        prepare_sparse_data_spmm_vnni<int8_t>(s8_ptr, a_shape[0], a_shape[1], 4, 1, sparse_ratio);
      }
    }
  }
  void* data_ptr_copy = aligned_allocator_t<uint8_t, 64>::allocate(bytes_size);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}
std::vector<float> make_output_scale(dim_t size, const std::vector<float>& ranges = {-10, 10}) {
  std::vector<float> output_scale(size, 0);
  init_vector(output_scale.data(), size, ranges[0], ranges[1]);
  return output_scale;
}
void spmm_vnni_bench::gen_case() {
  // Step 1: Construct operator config
  bool append_sum = (op_attrs["append_sum"] == "true");
  bool welford = (op_attrs["welford"] == "true");
  micro_bs = micro_bs == -1 ? N : micro_bs;
  LOG_IF(FATAL, N % micro_bs != 0) << "micro_bs must be a multiple of N";
  dim_t num_mbs = N / micro_bs;
  // Step 2: Construct runtime data
  jd::tensor_desc wei_desc = {{M, K}, jd::data_type::s8, jd::format_type::bsr};
  jd::tensor_desc src_desc = {{num_mbs, K, micro_bs}, jd::data_type::u8, jd::format_type::ab};
  jd::tensor_desc bia_desc = {{M, 1}, jd::data_type::s32, jd::format_type::ab};
  jd::tensor_desc dst_desc = {{num_mbs, M, micro_bs}, dt_dst, jd::format_type::ab};
  jd::tensor_desc scales_desc = {{M, 1}, jd::data_type::fp32, jd::format_type::ab};
  ts_descs = {wei_desc, src_desc, bia_desc, dst_desc, scales_desc};
  if (welford) {
    jd::tensor_desc mean_desc = {{num_mbs, micro_bs}, jd::data_type::fp32, jd::format_type::a};
    jd::tensor_desc var_desc = {{num_mbs, micro_bs}, jd::data_type::fp32, jd::format_type::a};
    ts_descs.push_back(mean_desc);
    ts_descs.push_back(var_desc);
  }
#ifdef WORKSPACE
  jd::tensor_desc workspace_desc = {{M * 2, N}, jd::data_type::fp32, jd::format_type::ab};
  ts_descs.push_back(workspace_desc);
#endif
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    bool is_clear = (index == jd::ssd::DST && !append_sum);
    float data_sparsity = (index == jd::ssd::WEI) ? sparse_ratio : 0;
    auto ranges = (index == jd::ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    auto data_pair = make_data_obj_spmm_vnni(tsd.shape(), tsd.dtype(), is_clear, data_sparsity, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  float scale = make_output_scale(1)[0];
  // Step 3: sparse data encoding
  auto sparse_ptr = new jd::bsr_data_t<int8_t>(
      jd::spns::reorder_to_bsr_group<int8_t, 4>(M, K, 4, 1, static_cast<const int8_t*>(rt_data1[jd::ssd::WEI])));
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  std::vector<jd::postop_attr> apply_postops_list;
  if (postop_algs.size()) {
    auto accu_op = [](std::string str_lists, jd::postop_alg alg) { return str_lists + '_' + jd::postop_alg_name[alg]; };
    op_attrs["postop_list"] = std::accumulate(postop_algs.begin() + 1, postop_algs.end(),
                                              std::string(jd::postop_alg_name[postop_algs[0]]), accu_op);
    for (auto& alg : postop_algs) {
      jd::postop_attr attr(jd::data_type::fp32, jd::postop_type::eltwise, alg, 0.0, 0.0, scale);
      apply_postops_list.push_back(attr);
    }
  }
  if (dt_dst == jd::data_type::s8 || dt_dst == jd::data_type::u8) {
    jd::postop_attr attr(dt_dst, jd::postop_type::eltwise, jd::postop_alg::quantize, 0.0, 0.0, scale);
    apply_postops_list.push_back(attr);
  }
  jd::operator_desc an_op_desc(jd::kernel_kind::sparse_matmul, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                               ts_descs, op_attrs, apply_postops_list);
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
  M = str_to_num<int64_t>(argv[0]);                                              // M
  K = str_to_num<int64_t>(argv[1]);                                              // K
  N = str_to_num<int64_t>(argv[2]);                                              // N
  sparse_ratio = str_to_num<float>(argv[3]);                                     // sparse_ratio
  micro_bs = str_to_num<int64_t>(argv[4]);                                       // micro_bs
  dt_dst = strcmp(argv[5], "1") == 0 ? jd::data_type::fp32 : jd::data_type::s8;  // is_fp32_out
  calc_mean_var = strcmp(argv[6], "2") == 0;                                     // calc mean and var
  if (calc_mean_var && dt_dst != jd::data_type::fp32) {
    LOG(ERROR) << "calc_mean_var requires fp32";
    bench_res_t res;
    res.stat = bench_status::fail;
    return res;
  }
  bool append_sum = strcmp(argv[6], "1") == 0 || calc_mean_var;  // has_append_sum
  if (append_sum && dt_dst != jd::data_type::fp32) {
    LOG(ERROR) << "append_sum requires fp32";
    bench_res_t res;
    res.stat = bench_status::fail;
    return res;
  }
  int64_t micro_oc = str_to_num<int64_t>(argv[7]);
  int sub_func_level = strlen(argv[8]) == 0 ? -1 : str_to_num<int>(argv[8]);  // -1 for invalid/defalut config
  if (argc > 9 && strcmp(argv[9], "gelu")) postop_algs.push_back(jd::postop_alg::gelu);
  if (argc > 9 && strcmp(argv[9], "exp")) postop_algs.push_back(jd::postop_alg::exp);
  if (append_sum) op_attrs["post_op"] = "append_sum";
  if (calc_mean_var) op_attrs["welford"] = "true";
  if (micro_oc >= 0) op_attrs["micro_oc"] = std::to_string(micro_oc);
  if (sub_func_level >= 0) {
    if (sub_func_level <= static_cast<uint8_t>(jd::ssd::subfunc_level::subfunc_level_MAX)) {
      op_attrs["sub_func"] = std::to_string(sub_func_level);
    }
  }
  return {bench_status::success};
}
}  // namespace bench
