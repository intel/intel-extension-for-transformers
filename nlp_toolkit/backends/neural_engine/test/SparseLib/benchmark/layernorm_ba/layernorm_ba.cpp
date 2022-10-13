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
#include "layernorm_ba/layernorm_ba.hpp"

namespace jd {

int gert_reduce_row_num(const std::vector<tensor_desc>& ts_descs) {
  int row = 1;
  for (int i = 0; i < ts_descs[0].shape().size() - 1; i++) row *= ts_descs[0].shape()[i];
  return row;
}

bench_res_t layernorm_ba_bench::set_config(int argc, char** argv) {
  if (argc < LAYERNORM_BA_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";

    return {bench_status::wrong_input};
  }
  LOG(INFO) << "layernorm_ba\n";
  M = str_to_num<int64_t>(argv[0]);
  N = str_to_num<int64_t>(argv[1]);
  std::string in_dt_str = argv[2];
  std::string out_dt_str = argv[3];
  std::string shape_str = std::string(argv[0]) + std::string("x") + std::string(argv[1]);
  in_dt = str_2_dt(in_dt_str);
  out_dt = str_2_dt(out_dt_str);
  tensor_desc input_data_desc = {{M, N}, in_dt, jd::format_type::ba};
  tensor_desc output_data_desc = {{M, N}, out_dt, jd::format_type::ba};
  if (out_dt == data_type::s8 || out_dt == data_type::u8) {
    postop_attrs.push_back(
        {out_dt, postop_type::eltwise, postop_alg::quantize, rand_float_postfix(), 0, rand_float_postfix()});
  }
  op_attrs = {{"matrix_shape", shape_str}};
  ts_descs = {
      {{M, N}, in_dt, jd::format_type::ba}, {{M, N}, out_dt, jd::format_type::ba}, {{}, in_dt, jd::format_type::ba}};
  return {bench_status::success};
}

void layernorm_ba_bench::get_true_data() {
  auto& op_desc = args.second.op_desc;
  auto& rf_data = args.second.rt_data;
  auto tensor_desc = op_desc.tensor_descs();
  int row = gert_reduce_row_num(tensor_desc);
  int col = tensor_desc[0].shape().back();
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

bool layernorm_ba_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;

  get_true_data();
  int num = get_element_num(q.op_desc);
  float err_rate = 1e-3;
  auto buf2 = q.rt_data[0];
  void* buf1;
  // Should compare buffer with different addresses
  if (buf1 == buf2) {
    printf("comparing the same buffer\n");
    return false;
  }
  auto dtype = p.op_desc.tensor_descs()[1].dtype();
  if (dtype == jd::data_type::fp32) buf1 = const_cast<void*>(p.rt_data[1]);
  if (dtype == jd::data_type::u8 || dtype == jd::data_type::s8) {
    buf1 = reinterpret_cast<float*>(malloc(num * sizeof(float)));
    auto int8_buf = const_cast<void*>(p.rt_data[1]);
    for (int i = 0; i < num; i++) {
      if (dtype == jd::data_type::u8)
        *(reinterpret_cast<float*>(buf1) + i) = uint8_2_int32(*(reinterpret_cast<uint8_t*>(int8_buf) + i));
      if (dtype == jd::data_type::s8)
        *(reinterpret_cast<float*>(buf1) + i) = *(reinterpret_cast<int8_t*>(int8_buf) + i);
    }
  }
  auto ans = compare_data<float>(buf1, num, buf2, num, err_rate);
  if (dtype == jd::data_type::u8 || dtype == jd::data_type::s8) free(buf1);
  return ans;
}

void layernorm_ba_bench::gen_case() {
  // malloc memory
  int row = gert_reduce_row_num(ts_descs);
  int col = ts_descs[0].shape().back();
  int num = row * col;
  void* src = nullptr;
  void* dst = nullptr;
  memo_mode MALLOC = memo_mode::MALLOC;
  memo_mode MEMSET = memo_mode::MEMSET;

  auto in_dt = ts_descs[0].dtype();
  auto out_dt = ts_descs[1].dtype();

  src = memo_op(src, num, in_dt, MALLOC);
  dst = memo_op(dst, num, out_dt, MALLOC);
  float* src_ref = reinterpret_cast<float*>(aligned_alloc(64, num * sizeof(float)));
  float* alpha = reinterpret_cast<float*>(aligned_alloc(64, row * sizeof(float)));
  float* beta = reinterpret_cast<float*>(aligned_alloc(64, row * sizeof(float)));

  // init alpha&beta
  for (int i = 0; i < row; i++) alpha[i] = 1 + rand_float_postfix();
  for (int i = 0; i < row; i++) beta[i] = 1 + rand_float_postfix();

  // init matrix.
  const unsigned int seed = 667095;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      unsigned int seed_tmp = seed + i;
      float rand_val = rand_r(&seed_tmp) % 256 - 128 + rand_float_postfix();
      assign_val(src, in_dt, rand_val, i * col + j);
      src_ref[i * col + j] = rand_val;
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
                                  op_attrs, postop_attrs);

  op_args_t p = {layernorm_ba_desc, rf_data1};
  op_args_t q = {layernorm_ba_desc, rf_data2};
  args = {p, q};
}

}  // namespace jd
