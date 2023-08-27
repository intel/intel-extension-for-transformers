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

namespace jd {

bool layernorm_ba_kd_t::init() {
  if (!isa_available(avx512_core)) return false;
  auto op_attr = op_desc_.attrs();
  if (op_attr.count("spec_type") == 0) {
    op_attr["spec_type"] = "normal";
    SPARSE_LOG(INFO) << "layernorm_ba spec_type set to normal by default.";
    normal_translnorm_init();
  } else if (op_attr["spec_type"] == "normal") {
    normal_translnorm_init();
  } else if (op_attr["spec_type"] == "direct") {
    direct_translnorm_init();
  } else {
    SPARSE_LOG(FATAL) << "unsupported translnorm spec_type type.";
  }
  return true;
}

void layernorm_ba_kd_t::handle_3D() {
  auto input_tensor_shape = op_desc_.tensor_descs().front().shape();
  SPARSE_LOG_IF(FATAL, !(input_tensor_shape.size() >= 2 && input_tensor_shape.size() <= 3))
      << "only support 2D/3D input.";
  if (input_tensor_shape.size() == 2) input_tensor_shape.insert(input_tensor_shape.begin(), 1);
  batch_num = input_tensor_shape[0];
  row_num = input_tensor_shape[1];
  col_num = input_tensor_shape[2];
}

void layernorm_ba_kd_t::direct_translnorm_init() {
  auto tensor_desc = op_desc_.tensor_descs();
  auto op_attrs = op_desc_.attrs();
  SPARSE_LOG_IF(FATAL, tensor_desc.size() != 2) << "need at 2 tensor descs.";
  auto input_dt = tensor_desc[0].dtype();
  auto output_dt = tensor_desc[1].dtype();
  SPARSE_LOG_IF(FATAL, input_dt != data_type::fp32 && input_dt != data_type::s32)
      << "only support fp32 or int32 input.";
  handle_3D();

  // init param
  int max_thr = omp_get_max_threads();
  ssd::layernorm_ba_param_t param;
  param.spec_type = ssd::spec_translnorm_type::direct;
  param.input_dt = input_dt;
  param.output_dt = output_dt;
  param.row_num = row_num;
  param.col_num = col_num;
  param.batch_num = batch_num;
  if (op_attrs["split_output"] == "true") {
    SPARSE_LOG_IF(FATAL, op_desc_.apply_postops_list().size() == 0 ||
                             op_desc_.apply_postops_list().back().op_alg != postop_alg::quantize ||
                             tensor_desc[1].dtype() != data_type::fp32)
        << "split output feature need quant op and fp32 data type in dst tensor descriptor";
    param.split_output = true;
    param.output2_dt = op_desc_.apply_postops_list().back().dt;
  }
  param.direct_process_row = row_num / max_thr;
  param.postop_attrs = op_desc_.apply_postops_list();
  params_.push_back(param);
}

void layernorm_ba_kd_t::normal_translnorm_init() {
  auto tensor_desc = op_desc_.tensor_descs();
  auto op_attr = op_desc_.attrs();
  SPARSE_LOG_IF(FATAL, op_attr["split_output"] == "true") << "split output feature only for directlnorm";
  auto input_dt = tensor_desc[0].dtype();
  auto output_dt = tensor_desc[1].dtype();
  SPARSE_LOG_IF(FATAL, input_dt != data_type::fp32) << "only support fp32 input.";
  handle_3D();

  // init params
  // TODO(zhe1wang): support col nums can't divded by 16.
  SPARSE_LOG_IF(FATAL, col_num % 16 != 0) << "col nums should divded by 16 now";
  auto ker_num = col_num / 16;
  // TODO(zhe1wang): set most appreciate thread num when fuse with quantize.
  params_.resize(ker_num);
  int col_per_thr = col_num / ker_num;
  for (int i = 0; i < ker_num; i++) {
    int thread_elt_offset = col_per_thr * i;
    ssd::layernorm_ba_param_t param;
    param.spec_type = ssd::spec_translnorm_type::normal;
    param.input_dt = input_dt;
    param.output_dt = output_dt;
    param.row_num = row_num;
    param.col_num = col_num;
    param.process_col = col_per_thr;
    param.process_batch_per_ker = omp_get_max_threads() >= batch_num ? 1 : batch_num;
    param.ker_per_batch = ker_num;
    param.thread_elt_offset = thread_elt_offset;
    param.postop_attrs = op_desc_.apply_postops_list();
    param.binaryop_attrs = op_desc_.get_binaryop_list();
    params_[i] = param;
  }
}

bool layernorm_ba_k_t::init() {
  auto spec_type = derived_kd()->params().front().spec_type;
  switch (spec_type) {
    case ssd::spec_translnorm_type::normal:
      return normal_init();
    case ssd::spec_translnorm_type::direct:
      return direct_init();
    default:
      return false;
  }
}

bool layernorm_ba_k_t::normal_init() {
  auto op_desc = kd()->get_operator_desc();
  auto tensor_shape = op_desc.tensor_descs()[0].shape();
  src_dt = op_desc.tensor_descs()[0].dtype();
  dst_dt = op_desc.tensor_descs()[1].dtype();
  auto params = derived_kd()->params();
  per_ker_process_batch = params[0].process_batch_per_ker;
  ker_num = params[0].ker_per_batch;
  if (tensor_shape.size() == 2) tensor_shape.insert(tensor_shape.begin(), 1);
  batch_loop = tensor_shape[0] / per_ker_process_batch;
  row_num = tensor_shape[1];
  col_num = tensor_shape[2];
  // TODO(zhe1wang): set most appreciate thread num when fuse with quantize.
  jit_kers_.resize(ker_num);
  for (int i = 0; i < ker_num; i++) {
    jit_layernorm_ba_t* ker = new jit_layernorm_ba_t(derived_kd()->params()[i]);
    if (ker == nullptr) return false;
    if (!(ker->create_kernel())) return false;
    jit_kers_[i] = ker;
  }

  return true;
}

bool layernorm_ba_k_t::direct_init() {
  auto op_desc = kd()->get_operator_desc();
  auto tensor_shape = op_desc.tensor_descs()[0].shape();
  src_dt = op_desc.tensor_descs()[0].dtype();
  dst_dt = op_desc.tensor_descs()[1].dtype();
  auto param = derived_kd()->params().front();
  batch_loop = param.batch_num;
  row_num = param.row_num;
  col_num = param.col_num;
  split_output = param.split_output;
  dst2_dt = param.output2_dt;
  ker_num = omp_get_max_threads();
  int offset = 0, process_row;
  for (int i = 0; i < ker_num; i++) {
    if (i < row_num % ker_num)
      process_row = param.direct_process_row + 1;
    else
      process_row = param.direct_process_row;
    direct_row_helper.push_back({process_row, offset});
    offset += process_row;
  }
  jit_layernorm_ba_t* ker = new jit_layernorm_ba_t(param);
  if (ker == nullptr) return false;
  if (!(ker->create_kernel())) return false;
  jit_kers_.push_back(ker);
  return true;
}

bool layernorm_ba_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto op_attrs = derived_kd().get()->get_operator_desc().attrs();
  if (op_attrs.count("spec_type") == 0 || op_attrs["spec_type"] == "normal") {
    normal_execute(rt_data);
  } else if (op_attrs["spec_type"] == "direct") {
    direct_execute(rt_data);
  } else {
    SPARSE_LOG(FATAL) << "unsupported translnorm spec_type type.";
  }

  return true;
}

void layernorm_ba_k_t::direct_execute(const std::vector<const void*>& rt_data) const {
  auto param = derived_kd()->params().front();
  const jit_layernorm_ba_t* jit_impl = jit_kers_[0];
  for (int i = 0; i < batch_loop; i++) {
#pragma omp parallel for
    for (int j = 0; j < ker_num; j++) {
      ssd::layernorm_ba_data_t data_param;
      auto process_row = direct_row_helper[j].first;
      auto row_offset = direct_row_helper[j].second;
      data_param.src = const_cast<char*>(reinterpret_cast<const char*>(rt_data[0]) +
                                         (i * col_num * row_num + row_offset * col_num) * get_data_size(src_dt));
      data_param.dst = const_cast<char*>(reinterpret_cast<const char*>(rt_data[1]) +
                                         (i * col_num * row_num + row_offset * col_num) * get_data_size(dst_dt));
      data_param.alpha = reinterpret_cast<float*>(const_cast<void*>(rt_data[2])) + row_offset;
      data_param.beta = reinterpret_cast<float*>(const_cast<void*>(rt_data[3])) + row_offset;
      data_param.mean = reinterpret_cast<float*>(const_cast<void*>(rt_data[4])) + i * col_num;
      data_param.var = reinterpret_cast<float*>(const_cast<void*>(rt_data[5])) + i * col_num;
      if (split_output)
        data_param.dst2 = const_cast<char*>(reinterpret_cast<const char*>(rt_data[6]) +
                                            (i * col_num * row_num + row_offset * col_num) * get_data_size(dst2_dt));
      data_param.process_row = process_row;
      (*jit_impl)(&data_param);
    }
  }
}

void layernorm_ba_k_t::normal_execute(const std::vector<const void*>& rt_data) const {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch_loop; i++) {
    for (int j = 0; j < ker_num; j++) {
      const jit_layernorm_ba_t* jit_impl = jit_kers_[j];
      ssd::layernorm_ba_data_t data_param;
      data_param.src =
          const_cast<char*>(reinterpret_cast<const char*>(rt_data[0]) + i * row_num * col_num * get_data_size(src_dt));
      data_param.dst =
          const_cast<char*>(reinterpret_cast<const char*>(rt_data[1]) + i * row_num * col_num * get_data_size(dst_dt));
      data_param.alpha = reinterpret_cast<float*>(const_cast<void*>(rt_data[2]));
      data_param.beta = reinterpret_cast<float*>(const_cast<void*>(rt_data[3]));
      data_param.n = row_num;
      (*jit_impl)(&data_param);
    }
  }
}
}  // namespace jd
