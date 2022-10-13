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

#include "eltwiseop/eltwiseop.hpp"

namespace jd {
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
  tensor_desc ts_desc = {{M, N}, dt, format_type::undef};
  ts_descs = {ts_desc, ts_desc};

  if (postop_attrs.size() == 0) {
    LOG(ERROR) << "No valid postop found";
    return {bench_status::wrong_input};
  }
  return {bench_status::success};
}
void eltwiseop_bench::gen_case() {
  operator_desc eltwiseop_desc(kernel_kind::eltwiseop, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                               op_attrs, postop_attrs);

  int num = get_element_num(eltwiseop_desc);
  void* src = nullptr;
  void* dst = nullptr;
  memo_mode MALLOC = memo_mode::MALLOC;
  memo_mode MEMSET = memo_mode::MEMSET;

  auto in_dt = ts_descs[0].dtype();
  auto out_dt = ts_descs[1].dtype();

  src = memo_op(src, num, in_dt, MALLOC);
  dst = memo_op(dst, num, out_dt, MALLOC);
  memo_op(dst, num, out_dt, MEMSET);

  float* src_ref = reinterpret_cast<float*>(malloc(num * sizeof(float)));
  float* dst_ref = reinterpret_cast<float*>(malloc(num * sizeof(float)));

  const unsigned int seed = 667095;
  memset(dst_ref, 0, num * sizeof(float));

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(ranges[0], ranges[1]);
  for (int i = 0; i < num; i++) {
    float rand_val = dis(gen);
    assign_val(src, in_dt, rand_val, i);
    if (in_dt == data_type::u8)
      src_ref[i] = *(reinterpret_cast<uint8_t*>(src) + i);
    else
      src_ref[i] = rand_val;
  }

  std::vector<const void*> rf_data1;
  std::vector<const void*> rf_data2;

  rf_data1.emplace_back(reinterpret_cast<void*>(src));
  rf_data1.emplace_back(reinterpret_cast<void*>(dst));
  rf_data2.emplace_back(reinterpret_cast<void*>(src_ref));
  rf_data2.emplace_back(reinterpret_cast<void*>(dst_ref));

  op_args_t p = {eltwiseop_desc, rf_data1};
  op_args_t q = {eltwiseop_desc, rf_data2};
  args = {p, q};
}
void eltwiseop_bench::get_true_data() {
  float* src = reinterpret_cast<float*>(const_cast<void*>(args.second.rt_data[0]));
  float* dst = reinterpret_cast<float*>(const_cast<void*>(args.second.rt_data[1]));

  int num = get_element_num(args.second.op_desc);
  auto attr = args.second.op_desc.apply_postops_list();
  for (int i = 0; i < num; i++) {
    dst[i] = src[i];
    dst[i] = apply_postop_list(dst[i], attr);
  }
}
bool eltwiseop_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;

  get_true_data();
  int num = get_element_num(q.op_desc);
  void* buf1;
  auto buf2 = q.rt_data[1];
  auto dtype = q.op_desc.tensor_descs()[1].dtype();
  float err_rate;
  if (dtype == jd::data_type::fp32) {
    buf1 = const_cast<void*>(p.rt_data[1]);
    err_rate = 1e-1;
  } else if (dtype == jd::data_type::bf16) {
    buf1 = reinterpret_cast<float*>(malloc(num * sizeof(float)));
    auto bf16_buf1 = const_cast<void*>(p.rt_data[1]);
    for (int i = 0; i < num; i++) {
      *(reinterpret_cast<float*>(buf1) + i) = make_fp32(*(reinterpret_cast<bfloat16_t*>(bf16_buf1) + i));
    }
    err_rate = 5;
  } else if (dtype == jd::data_type::s8 || dtype == jd::data_type::u8) {
    err_rate = 1e-1;
    buf1 = reinterpret_cast<float*>(malloc(num * sizeof(float)));
    auto int8_buf1 = const_cast<void*>(p.rt_data[1]);
    if (dtype == jd::data_type::u8) {
      for (int i = 0; i < num; i++)
        *(reinterpret_cast<float*>(buf1) + i) = uint8_2_int32(*(reinterpret_cast<uint8_t*>(int8_buf1) + i));
    } else {
      for (int i = 0; i < num; i++) *(reinterpret_cast<float*>(buf1) + i) = *(reinterpret_cast<int8_t*>(int8_buf1) + i);
    }
  }
  auto ans = compare_data<float>(buf1, num, buf2, num, err_rate);
  return ans;
}

}  // namespace jd
