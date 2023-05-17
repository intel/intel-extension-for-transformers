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

#include "dynamic_quant_matmul.hpp"
#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "kernels/spmm_types.hpp"
#include "src/cpu/kernels/dynamic_quant_matmul_ref.hpp"

namespace bench {
bench_res_t dynamic_quant_matmul_bench::set_config(int argc, char** argv) {
  if (argc < DYNAMIC_QUANT_MATMUL_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  LOG(INFO) << "dynamic_quant_matmul\n";
  b = str_to_num<dim_t>(argv[0]);
  m = str_to_num<dim_t>(argv[1]);
  n = str_to_num<dim_t>(argv[2]);
  k = str_to_num<dim_t>(argv[3]);
  large_wei_threshold = argv[4];
  add_bias = argv[5];
  return {bench_status::success};
}

bool dynamic_quant_matmul_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;
  std::shared_ptr<const jd::kernel_desc_t> dynamic_quant_matmul_ref_desc;
  jd::kernel_desc_t::create<jd::dynamic_quant_matmul_ref_kd_t>(dynamic_quant_matmul_ref_desc, q.op_desc);
  std::shared_ptr<const jd::kernel_t> dynamic_quant_matmul_ref_ker;
  jd::kernel_t::create<jd::dynamic_quant_matmul_ref_k_t, jd::dynamic_quant_matmul_ref_kd_t>(
      dynamic_quant_matmul_ref_ker, dynamic_quant_matmul_ref_desc);
  dynamic_quant_matmul_ref_ker->execute(q.rt_data);
  auto buf1 = p.rt_data[io::DST];
  auto size = p.op_desc.tensor_descs()[io::DST].size();
  auto buf2 = q.rt_data[io::DST];
  auto ans1 = compare_data<int8_t>(buf1, size, buf2, size, 1e-2);
  auto buf3 = p.rt_data[io::SCALE_DST];
  auto size2 = p.op_desc.tensor_descs()[io::SCALE_DST].size();
  auto buf4 = q.rt_data[io::SCALE_DST];
  auto ans2 = compare_data<float>(buf3, size2, buf4, size2, 5e-3);
  return ans1 && ans2;
}

void dynamic_quant_matmul_bench::gen_case() {
  op_attrs = {};
  op_attrs["large_wei_threshold"] = large_wei_threshold;
  int pad_n = (n + 15) / 16 * 16;
  void* activation = aligned_alloc(64, b * m * k);
  void* wei = aligned_alloc(64, k * pad_n);
  void* dst = aligned_alloc(64, b * m * n);
  void* correct_dst = aligned_alloc(64, b * m * n);
  void* scale_a = aligned_alloc(64, b * m * sizeof(float));
  void* scale_w = aligned_alloc(64, n * sizeof(float));
  void* bias = aligned_alloc(64, n * sizeof(float));
  void* scale_dst = aligned_alloc(64, b * m * sizeof(float));
  void* correct_scale_dst = aligned_alloc(64, b * m * sizeof(float));
  init_vector(static_cast<int8_t*>(activation), b * m * k, ranges[0], ranges[1], rand());
  init_vector(static_cast<int8_t*>(wei), k * pad_n, ranges[0], ranges[1], rand());
  init_vector(static_cast<float*>(scale_a), b * m, ranges[0], ranges[1], rand());
  init_vector(static_cast<float*>(scale_w), n, ranges[0], ranges[1], rand());
  init_vector(static_cast<float*>(bias), n, ranges[0], ranges[1], rand());
  std::vector<const void*> rt_data_p(8, nullptr);
  std::vector<const void*> rt_data_q(8, nullptr);
  jd::tensor_desc activation_desc = {{b, m, k}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc weight_desc = {{k, pad_n}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc dst_desc = {{b, m, n}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc sclae_a_desc = {{b, m}, jd::data_type::fp32, jd::format_type::undef};
  jd::tensor_desc scale_w_desc = {{n}, jd::data_type::fp32, jd::format_type::undef};
  jd::tensor_desc scale_dst_desc = {{b, m}, jd::data_type::fp32, jd::format_type::undef};
  jd::tensor_desc workspace_desc = {{}, jd::data_type::fp32, jd::format_type::undef};
  jd::tensor_desc bias_desc = {{n}, jd::data_type::fp32, jd::format_type::undef};
  ts_descs = {activation_desc, weight_desc, dst_desc, sclae_a_desc, scale_w_desc, scale_dst_desc, workspace_desc};
  if (add_bias == "true") ts_descs.push_back(bias_desc);
  rt_data_p[io::ACTIVATION] = activation;
  rt_data_p[io::WEIGHT] = wei;
  rt_data_p[io::DST] = dst;
  rt_data_p[io::SCALE_A] = scale_a;
  rt_data_p[io::SCALE_W] = scale_w;
  rt_data_p[io::SCALE_DST] = scale_dst;
  rt_data_p[io::BIAS] = bias;
  rt_data_q[io::ACTIVATION] = activation;
  rt_data_q[io::WEIGHT] = wei;
  rt_data_q[io::DST] = correct_dst;
  rt_data_q[io::SCALE_A] = scale_a;
  rt_data_q[io::SCALE_W] = scale_w;
  rt_data_q[io::SCALE_DST] = correct_scale_dst;
  rt_data_q[io::BIAS] = bias;
  jd::operator_desc op_desc(jd::kernel_kind::dynamic_quant_matmul, jd::kernel_prop::forward_inference,
                            jd::engine_kind::cpu, ts_descs, op_attrs);
  // Step 3: op_args_t testcase pair
  op_args_t op_args_p = {op_desc, rt_data_p};
  op_args_t op_args_q = {op_desc, rt_data_q};
  args = {op_args_p, op_args_q};
}
}  // namespace bench
