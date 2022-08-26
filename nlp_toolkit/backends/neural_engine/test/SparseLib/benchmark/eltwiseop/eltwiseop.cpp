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
#include "eltwiseop/eltwiseop.hpp"

namespace jd {

void get_true_data_eltwiseop(const operator_desc& op_desc, const std::vector<const void*>& rf_data) {
  float* src = reinterpret_cast<float*>(const_cast<void*>(rf_data[0]));
  float* dst = reinterpret_cast<float*>(const_cast<void*>(rf_data[1]));

  int num = get_element_num(op_desc);
  auto attr = op_desc.apply_postops_list();
  for (int i = 0; i < num; i++) {
    dst[i] = src[i];
    dst[i] = apply_postop_list(dst[i], attr);
  }
}

bool check_result_eltwiseop(const std::pair<op_args_t, op_args_t>& args) {
  const auto& p = args.first;
  const auto& q = args.second;

  get_true_data_eltwiseop(q.op_desc, q.rt_data);
  int num = get_element_num(q.op_desc);
  void* buf1;
  auto buf2 = q.rt_data[1];
  auto dtype = p.op_desc.apply_postops_list().back().dt;
  float err_rate;
  if (p.op_desc.apply_postops_list().back().op_alg != postop_alg::quantize && dtype == jd::data_type::fp32) {
    buf1 = const_cast<void*>(p.rt_data[1]);
    err_rate = 1e-1;
  } else if (dtype == jd::data_type::bf16) {
    buf1 = reinterpret_cast<float*>(malloc(num * sizeof(float)));
    auto bf16_buf1 = const_cast<void*>(p.rt_data[1]);
    for (int i = 0; i < num; i++) {
      *(reinterpret_cast<float*>(buf1) + i) = make_fp32(*(reinterpret_cast<uint16_t*>(bf16_buf1) + i));
    }
    err_rate = 5;
  } else if (p.op_desc.apply_postops_list().back().op_alg == postop_alg::quantize) {
    err_rate = 1e-1;
    buf1 = reinterpret_cast<float*>(malloc(num * sizeof(float)));
    auto int8_buf1 = const_cast<void*>(p.rt_data[1]);
    for (int i = 0; i < num; i++)
      *(reinterpret_cast<float*>(buf1) + i) = uint8_2_int32(*(reinterpret_cast<uint8_t*>(int8_buf1) + i));
  }
  // Should compare buffer with different addresses
  if (buf1 == buf2) {
    printf("comparing the same buffer\n");
    return false;
  }
  auto ans = compare_data<float>(buf1, num, buf2, num, err_rate);
  free(const_cast<void*>(p.rt_data[0]));
  free(const_cast<void*>(p.rt_data[1]));
  free(const_cast<void*>(q.rt_data[0]));
  free(const_cast<void*>(q.rt_data[1]));
  return ans;
}

std::pair<op_args_t, op_args_t> gen_case_eltwiseop(const std::vector<tensor_desc>& ts_descs,
                                                   const std::vector<float>& ranges,
                                                   const std::unordered_map<std::string, std::string> op_attrs,
                                                   const std::vector<postop_attr>& postop_attr) {
  operator_desc eltwiseop_desc(kernel_kind::eltwiseop, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                               op_attrs, postop_attr);

  int num = get_element_num(eltwiseop_desc);
  void* src = nullptr;
  void* dst = nullptr;
  memo_mode MALLOC = memo_mode::MALLOC;
  memo_mode MEMSET = memo_mode::MEMSET;

  auto in_dt = eltwiseop_desc.apply_postops_list().front().dt;
  auto out_dt = eltwiseop_desc.apply_postops_list().back().dt;

  src = memo_op(src, num, in_dt, MALLOC);

  if (eltwiseop_desc.apply_postops_list().back().op_alg == postop_alg::quantize) {
    dst = memo_op(dst, num, jd::data_type::u8, MALLOC);
    memo_op(dst, num, jd::data_type::u8, MEMSET);
  } else if (eltwiseop_desc.apply_postops_list().front().op_alg == postop_alg::dequantize) {
    dst = memo_op(dst, num, jd::data_type::fp32, MALLOC);
    memo_op(dst, num, jd::data_type::fp32, MEMSET);
  } else {
    dst = memo_op(dst, num, out_dt, MALLOC);
    memo_op(dst, num, out_dt, MEMSET);
  }

  float* src_ref = reinterpret_cast<float*>(malloc(num * sizeof(float)));
  float* dst_ref = reinterpret_cast<float*>(malloc(num * sizeof(float)));

  const unsigned int seed = 667095;
  memset(dst_ref, 0, num * sizeof(float));

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(ranges[0], ranges[1]);
  for (int i = 0; i < num; i++) {
    float rand_val = dis(gen);
    assign_val(src, in_dt, rand_val, i);
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
  return {p, q};
}

bench_res_t run_bench_eltwiseop(bench_mode mode, int argc, char** argv) {
  bench_res_t res;
  int64_t M = str_to_num<int64_t>(argv[0]);
  int64_t N = str_to_num<int64_t>(argv[1]);
  std::vector<float> ranges(2);
  std::string ranges_str(argv[3]);  // in the form of "lower_bound,upper_bound"
  size_t sep_idx = ranges_str.find(",");
  ranges[0] = std::stof(ranges_str.substr(0, sep_idx));
  ranges[1] = std::stof(ranges_str.substr(sep_idx + 1));
  data_type dt = data_type::fp32;
  std::vector<postop_attr> postop_attrs = get_postop_attr(argv[2], &dt);

  if (postop_attrs.size() == 0) {
    std::cerr << "No valid postop found" << std::endl;
    res.stat = bench_status::unimplemented;
    return res;
  }

  tensor_desc data_desc = {{M, N}, dt, format_type::undef};

  try {
    std::pair<op_args_t, op_args_t> args =
        gen_case_eltwiseop({data_desc, data_desc}, ranges, {{"postop_list", argv[0]}}, postop_attrs);
    const auto& p = args.first;
    const auto& op_desc = p.op_desc;
    eltwiseop_desc eltwiseop_desc(op_desc);
    eltwiseop eltwiseop_kern(eltwiseop_desc);
    res = benchmarkOrExecute(&eltwiseop_kern, p.rt_data, mode);
    if (mode == bench_mode::acc) {
      res.correct = check_result_eltwiseop(args);
    }
  } catch (const std::exception& e) {
    std::cerr << "kernel exception occurred" << std::endl;
    res.stat = bench_status::fail;
    return res;
  }

  return res;
}

}  // namespace jd
