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
#include "matmul_avx512f_8bit.hpp"

#include <functional>
#include <utility>

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "kernels/data_pack.hpp"

namespace bench {
bool matmul_avx512f_8bit_bench::check_result() {
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
    return compare_data<uint8_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == jd::data_type::s8) {
    return compare_data<int8_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == jd::data_type::bf16) {
    return compare_data<jd::bfloat16_t>(buf1, size1, buf2, size2, 0.12);
  }
  return false;
}
std::pair<const void*, const void*> make_data_obj_matmul_avx512f_8bit(  //
    const std::vector<int64_t>& a_shape, const jd::data_type& a_dt, bool is_clear = false,
    const std::vector<float>& ranges = {-10, 10}) {
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
    } else if (a_dt == jd::data_type::bf16) {
      data_ptr = aligned_allocator_t<jd::bfloat16_t>::allocate(elem_num);
      init_vector(static_cast<jd::bfloat16_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::f8_e4m3) {
      data_ptr = aligned_allocator_t<jd::float8_e4m3_t>::allocate(elem_num);
      init_vector(static_cast<jd::float8_e4m3_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == jd::data_type::f8_e5m2) {
      data_ptr = aligned_allocator_t<jd::float8_e5m2_t>::allocate(elem_num);
      init_vector(static_cast<jd::float8_e5m2_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    }
  }
  void* data_ptr_copy = aligned_allocator_t<char>::allocate(bytes_size);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}
void matmul_avx512f_8bit_bench::gen_case() {
  // Step 1: Construct operator config
  jd::tensor_desc src0_desc = {{M, K}, jd::data_type::bf16, jd::format_type::ab};
  jd::tensor_desc src1_desc = {{N, K}, src1_dtype, jd::format_type::ab};
  jd::tensor_desc dst_desc = {{M, N}, jd::data_type::bf16, jd::format_type::ab};
  jd::tensor_desc bias_desc = {{N}, jd::data_type::bf16, jd::format_type::ab};
  jd::tensor_desc scale_desc = {{N}, jd::data_type::fp32, jd::format_type::a};
  ts_descs = {src0_desc, src1_desc, dst_desc, bias_desc, scale_desc};
  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    bool is_clear = (index == io::DST0);
    auto ranges = std::vector<float>{-2, 10};
    auto data_pair = make_data_obj_matmul_avx512f_8bit(tsd.shape(), tsd.dtype(), is_clear, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  std::unordered_map<std::string, std::string> attrs1 = op_attrs;
  std::unordered_map<std::string, std::string> attrs2 = op_attrs;
  if (src1_dtype == jd::data_type::bf16) {
    attrs1["weight_bf16"] = std::to_string(reinterpret_cast<intptr_t>(rt_data1[io::SRC1]));
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(aligned_allocator_t<int8_t>::allocate(N * K)));
    attrs2["weight_bf16"] = std::to_string(reinterpret_cast<intptr_t>(rt_data2[io::SRC1]));
    attrs2["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(aligned_allocator_t<int8_t>::allocate(N * K)));
  } else if (src1_dtype == jd::data_type::s8) {
    std::function<int8_t(int8_t)> cast_func_s8 = [](int8_t x) { return x; };
    int8_t* weight_8bit = aligned_allocator_t<int8_t>::allocate(N * K);
    int8_t* src1_s8 = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data1[io::SRC1]));
    jd::pack<int8_t, int8_t>(weight_8bit, src1_s8, N, K, cast_func_s8);
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_8bit));
  } else if (src1_dtype == jd::data_type::f8_e4m3) {
    std::function<jd::float8_e4m3_t(jd::float8_e4m3_t)> cast_func_fp8 = [](jd::float8_e4m3_t x) { return x; };
    jd::float8_e4m3_t* src1_fp8 = reinterpret_cast<jd::float8_e4m3_t*>(const_cast<void*>(rt_data1[io::SRC1]));
    jd::float8_e4m3_t* weight_8bit = aligned_allocator_t<jd::float8_e4m3_t>::allocate(N * K);
    jd::pack<jd::float8_e4m3_t, jd::float8_e4m3_t>(weight_8bit, src1_fp8, N, K, cast_func_fp8);
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_8bit));
  } else if (src1_dtype == jd::data_type::f8_e5m2) {
    std::function<jd::float8_e5m2_t(jd::float8_e5m2_t)> cast_func_fp8 = [](jd::float8_e5m2_t x) { return x; };
    jd::float8_e5m2_t* src1_fp8 = reinterpret_cast<jd::float8_e5m2_t*>(const_cast<void*>(rt_data1[io::SRC1]));
    jd::float8_e5m2_t* weight_8bit = aligned_allocator_t<jd::float8_e5m2_t>::allocate(N * K);
    jd::pack<jd::float8_e5m2_t, jd::float8_e5m2_t>(weight_8bit, src1_fp8, N, K, cast_func_fp8);
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_8bit));
  }
  jd::operator_desc op_desc1(jd::kernel_kind::transpose_matmul, jd::kernel_prop::forward_inference,
                             jd::engine_kind::cpu, ts_descs, attrs1);
  jd::operator_desc op_desc2(jd::kernel_kind::transpose_matmul, jd::kernel_prop::forward_inference,
                             jd::engine_kind::cpu, ts_descs, attrs2);
  op_args_t op_args = {op_desc1, rt_data1};
  op_args_t op_args_copy = {op_desc2, rt_data2};
  args = {op_args, op_args_copy};
}
bench_res_t matmul_avx512f_8bit_bench::set_config(int argc, char** argv) {
  LOG(INFO) << "matmul_avx512f_8bit\n";
  if (argc < matmul_avx512f_8bit_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  M = str_to_num<int64_t>(argv[0]);  // M
  K = str_to_num<int64_t>(argv[1]);  // K
  N = str_to_num<int64_t>(argv[2]);  // N
  switch (str_to_num<int64_t>(argv[3])) {
    case 0:
      src1_dtype = jd::data_type::bf16;
      break;
    case 1:
      src1_dtype = jd::data_type::s8;
      break;
    case 2:
      src1_dtype = jd::data_type::f8_e5m2;
      break;
    case 3:
      src1_dtype = jd::data_type::f8_e4m3;
      break;
    default:
      src1_dtype = jd::data_type::bf16;
      break;
  }
  op_attrs["alpha"] = argc > 4 ? argv[4] : "1";  // alpha
  op_attrs["beta"] = argc > 5 ? argv[5] : "1";   // beta
  return {bench_status::success};
}
}  // namespace bench
