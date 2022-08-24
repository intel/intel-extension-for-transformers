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

#include "kernels/layernorm_ba.hpp"

namespace jd {

bool layernorm_ba_kd_t::init() {
  auto tensor_desc = op_desc_.tensor_descs();
  assert(tensor_desc.size() == 1);
  // TODO(zhe1wang): support more data_type.
  auto dt = tensor_desc[0].dtype();
  assert(dt == data_type::fp32);
  assert(tensor_desc[0].ftype() == format_type::ba);
  auto tensor_shape = tensor_desc[0].shape();
  // TODO(zhe1wang): support reduce dim.
  assert(tensor_shape.size() == 2);

  int row_num = tensor_shape[0];
  int col_num = tensor_shape[1];

  // init ptr
  one_ = new float(1.0);
  one_div_n_ = new float[col_num];
  eps_ = new float[col_num];
  for (int i = 0; i < col_num; i++) {
    one_div_n_[i] = 1.0 / row_num;
    eps_[i] = 1e-5;
  }

  // init params
  int nthr = op_desc_.impl_nthr();
  params_.resize(nthr);
  int col_per_thr = col_num / nthr;
  for (int i = 0; i < nthr; i++) {
    int thread_offset;
    int cur_thr_col;
    thread_offset = col_per_thr * i * get_data_size(dt);
    if (i != nthr - 1)
      cur_thr_col = col_per_thr;
    else
      cur_thr_col = col_num - i * col_per_thr;
    ssd::layernorm_ba_param_t param;
    param.dt = dt;
    // TODO(zhe1wang): support more dt in the future.
    assert(dt == data_type::fp32);
    param.row_num = row_num;
    param.col_num = col_num;
    param.process_col = cur_thr_col;
    param.thread_offset = thread_offset;
    param.postop_attrs = op_desc_.apply_postops_list();
    auto op_attr = op_desc_.attrs();
    if (op_attr.count("affine") != 0) {
      param.affine = true;
      const auto& alpha_ptr = str_to_num<uint64_t>(op_attr["alpha"]);
      const auto beta_ptr = str_to_num<uint64_t>(op_attr["beta"]);
      param.alpha = reinterpret_cast<float*>(alpha_ptr);
      param.beta = reinterpret_cast<float*>(beta_ptr);
    }
    params_[i] = param;
  }
  return true;
}

bool layernorm_ba_k_t::init() {
  int nthr = kd()->operator_desc().impl_nthr();
  jit_kers_.resize(nthr);
  for (int i = 0; i < nthr; i++) {
    td.push_back(new ssd::layernorm_ba_data_t());
    jit_layernorm_ba_t* ker = new jit_layernorm_ba_t(derived_kd()->params()[i]);
    if (ker == nullptr) return false;
    if (!(ker->create_kernel())) return false;
    jit_kers_[i] = ker;
  }

  return true;
}

bool layernorm_ba_k_t::execute(const std::vector<const void*>& rt_data) const {
  int nthr = kd()->operator_desc().impl_nthr();
  int col_num = derived_kd()->params()[0].col_num;
  int row_num = derived_kd()->params()[0].row_num;
#pragma omp parallel for
  for (int i = 0; i < nthr; i++) {
    const jit_layernorm_ba_t* jit_impl = jit_kers_[i];
    auto data_param = td[i];
    data_param->src = const_cast<void*>(rt_data[0]);
    data_param->dst = const_cast<void*>(rt_data[1]);
    data_param->one_div_n = derived_kd()->one_div_n_ptr();
    data_param->eps = derived_kd()->eps_ptr();
    data_param->one = derived_kd()->one_ptr();
    (*jit_impl)(td[i]);
  }

  return true;
}

}  // namespace jd

