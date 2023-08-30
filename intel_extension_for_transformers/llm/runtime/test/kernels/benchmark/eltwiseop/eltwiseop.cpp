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

#include "eltwiseop.hpp"
#include "common_utils.hpp"

namespace bench {
bench_res_t eltwiseop_bench::set_config(int argc, char** argv) {
  if (argc < ELTWISEOP_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed\n";
    return {bench_status::wrong_input};
  }
  LOG(INFO) << argv[0];
  M = str_to_num<int64_t>(argv[0]);
  N = str_to_num<int64_t>(argv[1]);
  std::string ranges_str(argv[3]);  // in the form of "lower_bound,upper_bound"
  size_t sep_idx = ranges_str.find(",");
  ranges[0] = std::stof(ranges_str.substr(0, sep_idx));
  ranges[1] = std::stof(ranges_str.substr(sep_idx + 1));
  op_attrs = {{"postop_list", argv[0]}};
  postop_attrs = get_postop_attr(argv[2], &dt);
  jd::tensor_desc ts_desc = {{M, N}, dt, jd::format_type::undef};
  ts_descs = {ts_desc, ts_desc};
  if (postop_attrs.size() == 0) {
    LOG(ERROR) << "No valid postop found";
    return {bench_status::wrong_input};
  }
  return {bench_status::success};
}
void eltwiseop_bench::gen_case() {
  jd::operator_desc eltwiseop_desc(jd::kernel_kind::eltwiseop, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                                   ts_descs, op_attrs, postop_attrs);
  int num = get_element_num(eltwiseop_desc);
  auto in_dt = ts_descs[0].dtype();
  auto out_dt = ts_descs[1].dtype();
  void* src = aligned_allocator_t<char>::allocate(jd::type_size.at(in_dt) * num);
  void* dst = aligned_allocator_t<char>::allocate(jd::type_size.at(in_dt) * num, true);
  void* src_ref = aligned_allocator_t<char>::allocate(jd::type_size.at(out_dt) * num);
  void* dst_ref = aligned_allocator_t<char>::allocate(jd::type_size.at(out_dt) * num, true);
  const unsigned int seed = 667095;
  std::srand(seed);
  for (int i = 0; i < num; i++) {
    float rand_val = std::rand() % 256 - 128;
    assign_val(src, in_dt, rand_val, i);
    assign_val(src_ref, in_dt, rand_val, i);
  }
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  rt_data1.emplace_back(reinterpret_cast<void*>(src));
  rt_data1.emplace_back(reinterpret_cast<void*>(dst));
  rt_data2.emplace_back(reinterpret_cast<void*>(src_ref));
  rt_data2.emplace_back(reinterpret_cast<void*>(dst_ref));
  op_args_t p = {eltwiseop_desc, rt_data1};
  op_args_t q = {eltwiseop_desc, rt_data2};
  args = {p, q};
}
void eltwiseop_bench::get_true_data() {
  auto src_tensor = args.second.op_desc.tensor_descs()[0];
  auto dst_tensor = args.second.op_desc.tensor_descs()[1];
  int size = src_tensor.size();
  auto src_dt = src_tensor.dtype();
  auto dst_dt = dst_tensor.dtype();
  const void* src = args.second.rt_data[0];
  void* dst = const_cast<void*>(args.second.rt_data[1]);
  float* src_fp32 = new float[size];
  if (src_dt == jd::data_type::s8) {
    cast_to_float_array<int8_t>(src, src_fp32, size);
  } else if (src_dt == jd::data_type::u8) {
    cast_to_float_array<uint8_t>(src, src_fp32, size);
  } else if (src_dt == jd::data_type::bf16) {
    cast_to_float_array<jd::bfloat16_t>(src, src_fp32, size);
  } else if (src_dt == jd::data_type::s32) {
    cast_to_float_array<int>(src, src_fp32, size);
  } else if (src_dt == jd::data_type::fp32) {
    cast_to_float_array<float>(src, src_fp32, size);
  }
  auto attr = args.second.op_desc.apply_postops_list();
  for (int i = 0; i < size; i++) {
    src_fp32[i] = apply_postop_list(src_fp32[i], attr);
  }
  if (dst_dt == jd::data_type::s8) {
    cast_from_float_array<int8_t>(src_fp32, dst, size);
  } else if (dst_dt == jd::data_type::u8) {
    cast_from_float_array<uint8_t>(src_fp32, dst, size);
  } else if (dst_dt == jd::data_type::bf16) {
    cast_from_float_array<jd::bfloat16_t>(src_fp32, dst, size);
  } else if (dst_dt == jd::data_type::s32) {
    cast_from_float_array<int>(src_fp32, dst, size);
  } else if (dst_dt == jd::data_type::fp32) {
    cast_from_float_array<float>(src_fp32, dst, size);
  }
  delete[] src_fp32;
}
bool eltwiseop_bench::check_result() {
  get_true_data();
  auto buf1 = args.first.rt_data[1];
  auto size1 = args.first.op_desc.tensor_descs()[1].size();
  auto buf2 = args.second.rt_data[1];
  auto size2 = args.second.op_desc.tensor_descs()[1].size();
  auto dst_type = args.second.op_desc.tensor_descs()[1].dtype();
  bool ans = false;
  if (dst_type == jd::data_type::fp32) {
    ans = compare_data<float>(buf1, size1, buf2, size2, 1e-1);
  } else if (dst_type == jd::data_type::u8) {
    ans = compare_data<uint8_t>(buf1, size1, buf2, size2, 1);
  } else if (dst_type == jd::data_type::s8) {
    ans = compare_data<int8_t>(buf1, size1, buf2, size2, 1);
  } else if (dst_type == jd::data_type::bf16) {
    ans = compare_data<jd::bfloat16_t>(buf1, size1, buf2, size2, 1);
  } else if (dst_type == jd::data_type::s32) {
    ans = compare_data<int>(buf1, size1, buf2, size2, 1);
  }
  return ans;
}
}  // namespace bench
