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

#include "dynamic_quant.hpp"
#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "kernels/spmm_types.hpp"
#include "src/cpu/kernels/dynamic_quant_ref.hpp"

namespace bench {
bench_res_t dynamic_quant_bench::set_config(int argc, char** argv) {
  if (argc < DYNAMIC_QUANT_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  LOG(INFO) << "dynamic_quant\n";
  channel_num = str_to_num<dim_t>(argv[0]);
  quantize_dim_elt_num = str_to_num<dim_t>(argv[1]);
  input_dt = argv[2];
  return {bench_status::success};
}
bool dynamic_quant_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;
  std::shared_ptr<const jd::kernel_desc_t> dynamic_quant_ref_desc;
  jd::kernel_desc_t::create<jd::dynamic_quant_ref_kd_t>(dynamic_quant_ref_desc, q.op_desc);
  std::shared_ptr<const jd::kernel_t> dynamic_quant_ref_ker;
  jd::kernel_t::create<jd::dynamic_quant_ref_k_t, jd::dynamic_quant_ref_kd_t>(dynamic_quant_ref_ker,
                                                                              dynamic_quant_ref_desc);
  dynamic_quant_ref_ker->execute(q.rt_data);
  auto buf1 = p.rt_data[io::MAT_DST];
  auto size = p.op_desc.tensor_descs()[io::MAT_DST].size();
  auto buf2 = q.rt_data[io::MAT_DST];
  auto ans1 = compare_data<int8_t>(buf1, size, buf2, size, 1e-2);
  auto buf3 = p.rt_data[io::SCALE_DST];
  auto size2 = p.op_desc.tensor_descs()[io::SCALE_DST].size();
  auto buf4 = q.rt_data[io::SCALE_DST];
  auto ans2 = compare_data<float>(buf3, size2, buf4, size2, 5e-3);
  return ans1 && ans2;
}
void dynamic_quant_bench::gen_case() {
  op_attrs = {};
  op_attrs["input_dt"] = input_dt;
  SPARSE_LOG_IF(FATAL, input_dt != "fp32" && input_dt != "bf16") << "input_dt must be fp32/bf16";
  jd::data_type src_dt = jd::data_type::bf16;
  if (input_dt == "fp32") src_dt = jd::data_type::fp32;
  std::vector<float> fp32_vec(channel_num * quantize_dim_elt_num);
  init_vector(fp32_vec.data(), fp32_vec.size(), 100.f, 200.f, rand());
  void* src = malloc(fp32_vec.size() * jd::type_size.at(src_dt));
  if (src_dt == jd::data_type::fp32) {
    memcpy(src, fp32_vec.data(), fp32_vec.size() * sizeof(float));
  } else {
    cast_from_float_array<jd::bfloat16_t>(fp32_vec.data(), src, fp32_vec.size());
  }
  void* dst = malloc(fp32_vec.size());
  void* correct_dst = malloc(fp32_vec.size());
  void* scale_dst = malloc(channel_num * sizeof(float));
  void* correct_scale_dst = malloc(channel_num * sizeof(float));
  std::vector<const void*> rt_data_p(3, nullptr);
  std::vector<const void*> rt_data_q(3, nullptr);
  jd::tensor_desc src_desc = {{channel_num, quantize_dim_elt_num}, src_dt, jd::format_type::undef};
  jd::tensor_desc dst_desc = {{channel_num, quantize_dim_elt_num}, jd::data_type::s8, jd::format_type::undef};
  jd::tensor_desc scale_dst_desc = {{channel_num}, jd::data_type::fp32, jd::format_type::undef};
  ts_descs = {src_desc, dst_desc, scale_dst_desc};
  rt_data_p[io::SRC] = src;
  rt_data_p[io::MAT_DST] = dst;
  rt_data_p[io::SCALE_DST] = scale_dst;
  rt_data_q[io::SRC] = src;
  rt_data_q[io::MAT_DST] = correct_dst;
  rt_data_q[io::SCALE_DST] = correct_scale_dst;
  jd::operator_desc op_desc(jd::kernel_kind::dynamic_quant, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs);
  // Step 3: op_args_t testcase pair
  op_args_t op_args_p = {op_desc, rt_data_p};
  op_args_t op_args_q = {op_desc, rt_data_q};
  args = {op_args_p, op_args_q};
}
}  // namespace bench
