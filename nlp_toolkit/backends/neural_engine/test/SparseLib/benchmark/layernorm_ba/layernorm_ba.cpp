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
#include "layernorm_ba/layernorm_ba.hpp"

namespace jd {

void get_true_data_layernorm_ba(const operator_desc& op_desc, const std::vector<const void*>& rf_data) {
  auto tensor_desc = op_desc.tensor_descs();
  int row = tensor_desc[0].shape()[0];
  int col = tensor_desc[0].shape()[1];
  float* dst = nullptr;
  float* alpha = nullptr;
  float* beta = nullptr;

  dst = reinterpret_cast<float*>(const_cast<void*>(rf_data[0]));
  alpha = reinterpret_cast<float*>(const_cast<void*>(rf_data[1]));
  beta = reinterpret_cast<float*>(const_cast<void*>(rf_data[2]));

  for (int i = 0; i < col; i++) {
    // calculate mean.
    float mean = 0;
    for (int j = 0; j < row; j++) mean += dst[j * col + i];
    mean /= row;
    // calculate var
    float var = 0;
    for (int j = 0; j < row; j++) var += (dst[j * col + i] - mean) * (dst[j * col + i] - mean);
    var /= row;
    var += 1e-5;
    var = sqrt(var);
    var = 1 / var;
    // calculate layernorm.
    for (int j = 0; j < row; j++) dst[j * col + i] = (dst[j * col + i] - mean) * var;

    // affine.
    for (int j = 0; j < row; j++) dst[j * col + i] = dst[j * col + i] * alpha[j] + beta[j];
  }

  // apply postop.
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++) dst[i * col + j] = apply_postop_list(dst[i * col + j], op_desc.apply_postops_list());
}

bool check_result_layernorm_ba(const std::pair<op_args_t, op_args_t>& args) {
  const auto& p = args.first;
  const auto& q = args.second;

  get_true_data_layernorm_ba(q.op_desc, q.rt_data);
  int num = get_element_num(q.op_desc);
  float err_rate = 1e-3;
  auto buf2 = q.rt_data[0];
  auto buf1 = p.rt_data[1];
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
  free(const_cast<void*>(q.rt_data[2]));
  return ans;
}

std::pair<op_args_t, op_args_t> gen_case_layernorm_ba(const std::vector<tensor_desc>& ts_descs,
                                                      std::unordered_map<std::string, std::string> op_attrs,
                                                      bool affine, const std::vector<postop_attr>& postop_attr) {
  // malloc memory
  int row = ts_descs[0].shape()[0];
  int col = ts_descs[0].shape()[1];
  int num = row * col;
  float* src = new float[num];
  float* dst = new float[num];
  float* src_ref = new float[num];
  float* alpha = new float[row];
  float* beta = new float[row];

  // init alpha&beta
  for (int i = 0; i < row; i++) alpha[i] = 1 + rand_float_postfix();
  for (int i = 0; i < row; i++) beta[i] = 1 + rand_float_postfix();

  // init matrix.
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      float tmp = 5 + rand_float_postfix();
      src[i * col + j] = tmp;
      src_ref[i * col + j] = tmp;
    }
  }

  std::vector<const void*> rf_data1;
  std::vector<const void*> rf_data2;

  rf_data1.emplace_back(reinterpret_cast<void*>(src));
  rf_data1.emplace_back(reinterpret_cast<void*>(dst));
  rf_data1.emplace_back(alpha);
  rf_data1.emplace_back(beta);
  rf_data2.emplace_back(reinterpret_cast<void*>(src_ref));
  rf_data2.push_back(alpha);
  rf_data2.push_back(beta);

  operator_desc layernorm_ba_desc(kernel_kind::layernorm_ba, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                                  op_attrs, postop_attr);

  op_args_t p = {layernorm_ba_desc, rf_data1};
  op_args_t q = {layernorm_ba_desc, rf_data2};
  return {p, q};
}

bench_res_t run_bench_layernorm_ba(bench_mode mode, int argc, char** argv) {
  bench_res_t res;
  data_type dt;
  int64_t M = str_to_num<int64_t>(argv[0]);
  int64_t N = str_to_num<int64_t>(argv[1]);
  std::string shape_str = std::string(argv[0]) + std::string("x") + std::string(argv[1]);
  bool is_affine = true;
  std::vector<postop_attr> postop_attrs(0);

  tensor_desc data_desc = {{M, N}, data_type::fp32, jd::format_type::ba};

  std::pair<op_args_t, op_args_t> args =
      gen_case_layernorm_ba({data_desc}, {{"matrix_shape", shape_str}}, is_affine, postop_attrs);
  try {
    const auto& p = args.first;
    const auto& op_desc = p.op_desc;
    layernorm_ba_desc layernorm_ba_desc(op_desc);
    layernorm_ba layernorm_ba_kern(layernorm_ba_desc);
    res = benchmarkOrExecute(&layernorm_ba_kern, p.rt_data, mode);
  } catch (const std::exception& e) {
    std::cerr << "kernel exception occurred" << std::endl;
    res.stat = bench_status::fail;
    return res;
  }

  if (mode == bench_mode::acc) {
    res.correct = check_result_layernorm_ba(args);
  }

  return res;
}

}  // namespace jd
