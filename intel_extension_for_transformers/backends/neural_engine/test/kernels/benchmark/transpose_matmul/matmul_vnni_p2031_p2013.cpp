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

#include "matmul_vnni_p2031_p2013.hpp"
#include <functional>
#include <utility>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"

namespace bench {

bool matmul_vnni_p2031_p2013_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;
  get_true_data();
  auto buf1 = p.rt_data[io::DST0];
  auto size1 = p.op_desc.tensor_descs()[io::DST0].size();
  auto buf2 = q.rt_data[io::DST0];
  auto size2 = q.op_desc.tensor_descs()[io::DST0].size();
  const auto& dst_type = p.op_desc.tensor_descs()[io::DST0].dtype();
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
std::pair<const void*, const void*> make_data_obj_matmul_vnni_p2031_p2013(  //
    const std::vector<int64_t>& a_shape, const jd::data_type& a_dt, bool is_clear = false,
    const std::vector<float>& ranges = {-10, 10}) {
  if (a_shape.size() == 0) return {nullptr, nullptr};
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<dim_t>());
  int bytes_size = elem_num * jd::type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = aligned_allocator_t<char>::allocate(bytes_size, true);
  } else {
    if (a_dt == jd::data_type::fp32) {
      data_ptr = aligned_allocator_t<float>::allocate(elem_num);
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::s32) {
      data_ptr = aligned_allocator_t<int32_t>::allocate(elem_num);
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::u8) {
      data_ptr = aligned_allocator_t<uint8_t>::allocate(elem_num);
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::s8) {
      data_ptr = aligned_allocator_t<int8_t>::allocate(elem_num);
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    }
  }
  void* data_ptr_copy = aligned_allocator_t<char>::allocate(bytes_size);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}
void matmul_vnni_p2031_p2013_bench::gen_case() {
  // Step 1: Construct operator config
  const jd::data_type dst_type = post_ops.size() != 0 ? post_ops.back().dt : jd::data_type::fp32;
  const jd::tensor_desc src0_desc{{bs1, K, bs0, M}, jd::data_type::s8, jd::format_type::ab};
  const jd::tensor_desc src1_desc{{bs1, K, bs0, N}, jd::data_type::s8, jd::format_type::ab};
  const jd::tensor_desc dst_desc{{bs0, bs1, M, N}, dst_type, jd::format_type::ab};
  const jd::tensor_desc src2_desc =
      has_binary_add ? jd::tensor_desc{{bs0, bs1, M, N}, jd::data_type::fp32, jd::format_type::ab} : jd::tensor_desc();
  const std::vector<jd::tensor_desc> ts_descs{src0_desc, src1_desc, dst_desc, src2_desc};
  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    const bool is_clear = (index == io::DST0);
    const auto ranges = std::vector<float>{-10, 10};
    const auto data_pair = make_data_obj_matmul_vnni_p2031_p2013(tsd.shape(), tsd.dtype(), is_clear, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  jd::operator_desc op_desc(jd::kernel_kind::transpose_matmul, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs, post_ops);
  // Step 3: op_args_t testcase pair
  op_args_t op_args = {op_desc, rt_data1};
  op_args_t op_args_copy = {op_desc, rt_data2};
  args = {op_args, op_args_copy};
}
bench_res_t matmul_vnni_p2031_p2013_bench::set_config(int argc, char** argv) {
  LOG(INFO) << "matmul_vnni_p2031_p2013\n";
  if (argc < MATMUL_VNNI_P2031_P2013_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  M = str_to_num<int64_t>(argv[0]);                        // M
  K = str_to_num<int64_t>(argv[1]);                        // K
  N = str_to_num<int64_t>(argv[2]);                        // N
  bs0 = str_to_num<int64_t>(argv[3]);                      // bs0
  bs1 = str_to_num<int64_t>(argv[4]);                      // bs1
  has_binary_add = argc > 5 && strcmp(argv[5], "1") == 0;  // has_binary_add
  op_attrs["src0_scale"] = argc > 6 ? argv[6] : "1";       // src0_scale
  op_attrs["src1_scale"] = argc > 7 ? argv[7] : "1";       // src1_scale
  op_attrs["out_scale"] = argc > 8 ? argv[8] : "1";        // out_scale
  if (argc > 9 && strlen(argv[9]) != 0) {                  // post_quant
    if (strcmp(argv[9], "s8quant") == 0) {
      post_ops.emplace_back(jd::data_type::s8, jd::postop_type::eltwise, jd::postop_alg::quantize, 0, 0, 0.0001);
    } else if (strcmp(argv[9], "u8quant") == 0) {
      post_ops.emplace_back(jd::data_type::u8, jd::postop_type::eltwise, jd::postop_alg::quantize, 121, 0, 0.0001);
    } else {
      SPARSE_LOG(ERROR) << "Unexpected postop!";
    }
  }
  return {bench_status::success};
}
}  // namespace bench
