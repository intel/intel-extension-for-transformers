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

#include "layernorm_ba.hpp"

#include <memory>
#include <functional>

namespace bench {
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
  jd::tensor_desc input_data_desc = {{M, N}, in_dt, jd::format_type::ba};
  jd::tensor_desc output_data_desc = {{M, N}, out_dt, jd::format_type::ba};
  if (out_dt == jd::data_type::s8 || out_dt == jd::data_type::u8) {
    postop_attrs.push_back(
        {out_dt, jd::postop_type::eltwise, jd::postop_alg::quantize, rand_float_postfix(), 0, rand_float_postfix()});
  }
  op_attrs = {{"matrix_shape", shape_str}};
  ts_descs = {
      {{M, N}, in_dt, jd::format_type::ba}, {{M, N}, out_dt, jd::format_type::ba}, {{}, in_dt, jd::format_type::ba}};
  return {bench_status::success};
}
void layernorm_ba_bench::get_true_data() {
  auto& op_desc = args.second.op_desc;
  auto& rt_data = args.second.rt_data;
  auto tensor_desc = op_desc.tensor_descs();
  auto op_attr = op_desc.attrs();
  int batch = tensor_desc[0].shape().size() == 2 ? 1 : tensor_desc[0].shape()[0];
  int row = tensor_desc[0].shape().size() == 2 ? tensor_desc[0].shape()[0] : tensor_desc[0].shape()[1];
  int col = tensor_desc[0].shape().back();
  auto src_dt = tensor_desc[0].dtype();
  auto dst_dt = tensor_desc[1].dtype();
  float* src = reinterpret_cast<float*>(const_cast<void*>(rt_data[0]));
  float* alpha = reinterpret_cast<float*>(const_cast<void*>(rt_data[2]));
  float* beta = reinterpret_cast<float*>(const_cast<void*>(rt_data[3]));
  void* dst_data = const_cast<void*>(rt_data[1]);
  auto dst_fp32 = static_cast<float*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);
  auto dst_s8 = static_cast<int8_t*>(dst_data);
  auto store_data = [&](int dst_idx, float value) {
    if (dst_dt == jd::data_type::fp32) {
      dst_fp32[dst_idx] = static_cast<float>(value);
    } else if (dst_dt == jd::data_type::s8) {
      dst_s8[dst_idx] = static_cast<int8_t>(value);
    } else if (dst_dt == jd::data_type::u8) {
      dst_u8[dst_idx] = static_cast<uint8_t>(value);
    }
  };
  auto normal_translnorm = [&]() {
    LOG_IF(FATAL, src_dt != jd::data_type::fp32);
    for (int k = 0; k < batch; k++) {
      for (int i = 0; i < col; i++) {
        // calculate mean.
        float mean = 0;
        for (int j = 0; j < row; j++) mean += src[k * col * row + j * col + i];
        mean /= row;
        // calculate var
        float var = 0;
        for (int j = 0; j < row; j++)
          var += (src[k * col * row + j * col + i] - mean) * (src[k * col * row + j * col + i] - mean);
        var /= row;
        var += 1e-5;
        var = sqrt(var);
        var = 1 / var;
        // calculate layernorm.
        auto binary_op_list = op_desc.get_binaryop_list();
        for (int j = 0; j < row; j++) {
          int dst_idx = k * row * col + j * col + i;
          float value = (src[dst_idx] - mean) * var;
          value = alpha[j] * value + beta[j];
          value = apply_postop_list(value, op_desc.apply_postops_list());
          if (!binary_op_list.empty()) {
            value =
                get_quantize(value, binary_op_list[0].zp[j], 1 / binary_op_list[0].scale[j], binary_op_list[0].op_dt);
          }
          store_data(dst_idx, value);
        }
      }
    }
  };
  auto direct_translnorm = [&]() {
    LOG_IF(FATAL, src_dt != jd::data_type::fp32 && src_dt != jd::data_type::s32);
    float* mean_data = reinterpret_cast<float*>(const_cast<void*>(rt_data[4]));
    float* var_data = reinterpret_cast<float*>(const_cast<void*>(rt_data[5]));
    for (int i = 0; i < batch; i++) {
      for (int j = 0; j < row; j++) {
        for (int k = 0; k < col; k++) {
          int dst_idx = i * row * col + j * col + k;
          float value = src[dst_idx];
          float var = var_data[i * col + k];
          var += 1e-5;
          var = sqrt(var);
          float scale = alpha[j] / var;
          value = (value - mean_data[i * col + k]) * scale + beta[j];
          value = apply_postop_list(value, op_desc.apply_postops_list());
          store_data(dst_idx, value);
        }
      }
    }
  };
  if (op_attr.count("spec_type") == 0) {
    op_attr["spec_type"] = "normal";
    SPARSE_LOG(INFO) << "layernorm_ba spec_type set to normal by default.";
    normal_translnorm();
  } else if (op_attr["spec_type"] == "normal") {
    normal_translnorm();
  } else if (op_attr["spec_type"] == "direct") {
    direct_translnorm();
  } else {
    LOG(FATAL) << "unsupported translnorm spec type.";
  }
}
bool layernorm_ba_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;
  get_true_data();
  auto buf1 = p.rt_data[1];
  auto size1 = p.op_desc.tensor_descs()[1].size();
  auto buf2 = q.rt_data[1];
  auto size2 = p.op_desc.tensor_descs()[1].size();
  auto dst_type = p.op_desc.tensor_descs()[1].dtype();
  bool ans = false;
  if (dst_type == jd::data_type::fp32) {
    ans = compare_data<float>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == jd::data_type::u8) {
    ans = compare_data<uint8_t>(buf1, size1, buf2, size2, 1);
  } else if (dst_type == jd::data_type::s8) {
    ans = compare_data<int8_t>(buf1, size1, buf2, size2, 1);
  }
  return ans;
}
void layernorm_ba_bench::gen_case() {
  // malloc memory
  int row = ts_descs[0].reduce_rows();
  int col = ts_descs[0].shape().back();
  int num = row * col;
  void* src = nullptr;
  void* dst = nullptr;
  void* src_ref = nullptr;
  void* dst_ref = nullptr;
  auto in_dt = ts_descs[0].dtype();
  auto out_dt = ts_descs[1].dtype();
  src = aligned_allocator_t<char>::allocate(jd::type_size.at(in_dt) * num);
  dst = aligned_allocator_t<char>::allocate(jd::type_size.at(out_dt) * num, true);
  src_ref = aligned_allocator_t<char>::allocate(jd::type_size.at(in_dt) * num);
  dst_ref = aligned_allocator_t<char>::allocate(jd::type_size.at(out_dt) * num, true);
  float* alpha = aligned_allocator_t<float>::allocate(row);
  float* beta = aligned_allocator_t<float>::allocate(row);
  // init alpha&beta
  for (int i = 0; i < row; i++) alpha[i] = 1 + rand_float_postfix();
  for (int i = 0; i < row; i++) beta[i] = 1 + rand_float_postfix();
  // init matrix.
  const unsigned int seed = 667095;
  std::srand(seed);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      float rand_val = std::rand() % 256 - 128 + rand_float_postfix();
      assign_val(src, in_dt, rand_val, i * col + j);
      assign_val(src_ref, in_dt, rand_val, i * col + j);
    }
  }
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  rt_data1.emplace_back(src);
  rt_data1.emplace_back(dst);
  rt_data1.emplace_back(alpha);
  rt_data1.emplace_back(beta);
  rt_data2.emplace_back(src_ref);
  rt_data2.emplace_back(dst_ref);
  rt_data2.push_back(alpha);
  rt_data2.push_back(beta);
  jd::operator_desc layernorm_ba_desc(jd::kernel_kind::layernorm_ba, jd::kernel_prop::forward_inference,
                                      jd::engine_kind::cpu, ts_descs, op_attrs, postop_attrs);
  op_args_t p = {layernorm_ba_desc, rt_data1};
  op_args_t q = {layernorm_ba_desc, rt_data2};
  args = {p, q};
}
}  // namespace bench
